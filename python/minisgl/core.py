from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Literal

import torch
from minisgl.reasoning import (
    ReasoningState,
    advance_reasoning_state,
    initial_reasoning_state,
)

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


class RequestLifecycleState(Enum):
    ACTIVE = auto()
    TERMINAL_PENDING_RETIRE = auto()
    RETIRED = auto()


@dataclass
class RequestLifecycle:
    """Host commit state and issued-forward ownership for one request generation."""

    generation_id: int = -1
    state: RequestLifecycleState = RequestLifecycleState.ACTIVE
    next_issue_epoch: int = 0
    outstanding_epochs: set[int] = field(default_factory=set)
    terminal_finish_reason: str | None = None
    resources_released: bool = False

    def issue(self) -> int:
        if self.state is not RequestLifecycleState.ACTIVE:
            raise RuntimeError(
                f"cannot issue forward for request in lifecycle state {self.state.name}"
            )
        epoch = self.next_issue_epoch
        self.next_issue_epoch += 1
        if epoch in self.outstanding_epochs:
            raise RuntimeError(f"request issue epoch {epoch} is already outstanding")
        self.outstanding_epochs.add(epoch)
        return epoch

    def complete(self, *, generation_id: int, issue_epoch: int) -> None:
        if generation_id != self.generation_id:
            raise RuntimeError(
                "request generation mismatch while completing issued forward: "
                f"expected={self.generation_id}, got={generation_id}"
            )
        if issue_epoch not in self.outstanding_epochs:
            raise RuntimeError(f"request issue epoch {issue_epoch} is not outstanding")
        self.outstanding_epochs.remove(issue_epoch)

    def commit_terminal(self, finish_reason: str) -> None:
        if self.state is not RequestLifecycleState.ACTIVE:
            raise RuntimeError(
                f"terminal status may be committed exactly once; current state is {self.state.name}"
            )
        self.state = RequestLifecycleState.TERMINAL_PENDING_RETIRE
        self.terminal_finish_reason = finish_reason

    @property
    def accepts_completion(self) -> bool:
        return self.state is RequestLifecycleState.ACTIVE

    @property
    def ready_to_release_resources(self) -> bool:
        return (
            self.state is RequestLifecycleState.TERMINAL_PENDING_RETIRE
            and not self.outstanding_epochs
            and not self.resources_released
        )

    def mark_resources_released(self) -> None:
        if not self.ready_to_release_resources:
            raise RuntimeError(
                "request resources may retire only once after all issued forwards complete"
            )
        self.resources_released = True
        self.state = RequestLifecycleState.RETIRED


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    reasoning_effort: str | None = None
    swa_evicted_seqlen: int = 0
    lifecycle: RequestLifecycle = field(default_factory=RequestLifecycle)
    reasoning_state: ReasoningState = field(init=False)
    _host_token_buffer: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        self.swa_evicted_seqlen = max(0, int(self.swa_evicted_seqlen))
        self.reasoning_state = initial_reasoning_state(self.reasoning_effort)

    def observe_generated_token(self, token_id: int, *, think_end_token_id: int) -> None:
        self.reasoning_state = advance_reasoning_state(
            self.reasoning_state,
            token_id,
            think_end_token_id=think_end_token_id,
        )

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        if next_token.device.type != "cpu" or next_token.dtype != self.input_ids.dtype:
            raise ValueError("next_token must match the request's CPU token dtype")
        if next_token.numel() != 1:
            raise ValueError("append_host expects exactly one token")

        current_len = len(self.input_ids)
        if current_len >= self.max_device_len:
            raise RuntimeError("request has no remaining host token capacity")
        if self._host_token_buffer is None:
            # Allocate on first decode; chunked-prefill requests must not copy growing prompts.
            self._host_token_buffer = torch.empty(
                self.max_device_len,
                dtype=self.input_ids.dtype,
                device=self.input_ids.device,
                pin_memory=self.input_ids.is_pinned(),
            )
            self._host_token_buffer[:current_len].copy_(self.input_ids)

        self._host_token_buffer[current_len].copy_(next_token.reshape(()))
        self.input_ids = self._host_token_buffer[: current_len + 1]

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    @property
    def can_commit_token(self) -> bool:
        """Whether another sampled token can be committed to the host result."""
        return len(self.input_ids) < self.max_device_len

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # Graph-visible scalar containing the number of semantic (non-padding)
    # token rows.  CUDA graph replay binds this to a stable capture-buffer
    # address; eager execution materializes an exact-sized scalar.
    num_token_non_padded: torch.Tensor = field(init=False)
    # One graph-visible generation state per real request row.  Graph padding
    # owns separate capture-buffer rows and never aliases request-local state.
    reasoning_states: torch.Tensor = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX

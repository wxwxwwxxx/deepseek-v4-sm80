from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Sequence, Tuple

import torch
from minisgl.core import Batch, Req, RequestLifecycle
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


def allocate_fair_prefill_chunks(
    *,
    token_budget: int,
    alignment_quantum: int,
    scheduled_tokens: Sequence[int],
    remaining_tokens: Sequence[int],
) -> list[int]:
    """Allocate one forward's total token budget with bounded max-min fairness.

    Every selected request receives one positive quantum (or its final tail)
    before any request receives a second share. Further quanta go to the
    request with the least scheduler-granted prefill progress, with FIFO index
    as the deterministic tie-breaker. A sub-quantum grant is used only for a
    final request tail, except for the progress-preserving fallback when the
    entire configured budget is smaller than one quantum.
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if alignment_quantum <= 0:
        raise ValueError("alignment_quantum must be positive")
    if len(scheduled_tokens) != len(remaining_tokens):
        raise ValueError("scheduled_tokens and remaining_tokens must have equal length")
    if not remaining_tokens:
        return []
    if any(value < 0 for value in scheduled_tokens):
        raise ValueError("scheduled_tokens must be non-negative")
    if any(value <= 0 for value in remaining_tokens):
        raise ValueError("remaining_tokens must be positive")

    allocations = [0] * len(remaining_tokens)
    budget = int(token_budget)

    # First share: no selected request can receive a second share before all
    # selected requests have made positive progress.
    for idx, remaining in enumerate(remaining_tokens):
        if budget <= 0:
            break
        if budget >= alignment_quantum:
            grant = min(int(remaining), alignment_quantum)
        else:
            # Release budgets are quantum-aligned. Keep a smaller diagnostic
            # budget live instead of producing a zero-token batch.
            grant = min(int(remaining), budget)
        allocations[idx] += grant
        budget -= grant

    while budget > 0:
        active = [
            idx
            for idx, remaining in enumerate(remaining_tokens)
            if allocations[idx] < remaining
        ]
        if not active:
            break
        active.sort(
            key=lambda idx: (
                int(scheduled_tokens[idx]) + allocations[idx],
                idx,
            )
        )
        made_progress = False
        for idx in active:
            remaining = int(remaining_tokens[idx]) - allocations[idx]
            if remaining < alignment_quantum:
                if remaining > budget:
                    continue
                grant = remaining
            else:
                if budget < alignment_quantum:
                    continue
                grant = alignment_quantum
            allocations[idx] += grant
            budget -= grant
            made_progress = True
            if budget <= 0:
                break
        if not made_progress:
            break

    if sum(allocations) > token_budget:
        raise RuntimeError("fair prefill allocation exceeded the total token budget")
    if any(value <= 0 for value in allocations):
        raise RuntimeError("selected prefill request received no token allocation")
    return allocations


@dataclass(frozen=True)
class PreparedPrefillReq:
    pending_req: PendingReq
    cache_handle: BaseCacheHandle
    table_idx: int
    cached_len: int

    @property
    def remaining_tokens(self) -> int:
        return self.pending_req.input_len - self.cached_len


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    alignment_quantum: int = 1

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        acquire_sequence_slot = getattr(
            getattr(self.cache_manager, "kv_cache", None),
            "acquire_sequence_slot",
            None,
        )
        if callable(acquire_sequence_slot):
            acquire_sequence_slot(table_idx, req.lifecycle.generation_id)
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        chunk_size: int | None = None,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        if chunk_size is None:
            chunk_size = min(self.token_budget, remain_len)
        if not 0 < chunk_size <= remain_len:
            raise ValueError(
                f"invalid prefill chunk_size={chunk_size} for remain_len={remain_len}"
            )
        if chunk_size > self.token_budget:
            raise ValueError(
                f"prefill chunk_size={chunk_size} exceeds remaining budget={self.token_budget}"
            )
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        previous_chunk = pending_req.chunked_req
        lifecycle = (
            previous_chunk.lifecycle if previous_chunk is not None else pending_req.lifecycle
        )
        pending_req.lifecycle = lifecycle
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
            reasoning_effort=pending_req.reasoning_effort,
            lifecycle=lifecycle,
            swa_evicted_seqlen=(
                0
                if previous_chunk is None
                else int(getattr(previous_chunk, "swa_evicted_seqlen", 0))
            ),
        )

    def try_prepare_one(self, pending_req: PendingReq) -> PreparedPrefillReq | None:
        if pending_req.chunked_req is not None:
            chunked_req = pending_req.chunked_req
            return PreparedPrefillReq(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            cached_len = cache_handle.cached_len
            self.reserved_size += (
                pending_req.input_len - cached_len + pending_req.output_len
            )
            return PreparedPrefillReq(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cached_len,
            )
        return None

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            self.reserved_size += (
                pending_req.input_len - chunked_req.cached_len
                + pending_req.output_len
            )
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            self.reserved_size += (
                pending_req.input_len - cache_handle.cached_len
                + pending_req.output_len
            )
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg, *, generation_id: int = -1) -> None:
        self.pending_list.append(
            PendingReq(
                req.uid,
                req.input_ids,
                req.sampling_params,
                reasoning_effort=req.reasoning_effort,
                lifecycle=RequestLifecycle(generation_id=generation_id),
            )
        )

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        alignment_quantum = max(
            1,
            int(getattr(self.cache_manager, "page_size", 1)),
        )
        # Existing chunked requests were admitted against their full remaining
        # prompt plus output. Re-establish that reservation before considering
        # any new request, independent of queue rotation.
        resident_reservation = sum(
            pending_req.input_len
            - pending_req.chunked_req.cached_len
            + pending_req.output_len
            for pending_req in self.pending_list
            if pending_req.chunked_req is not None
        )
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens + resident_reservation,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            alignment_quantum=alignment_quantum,
        )
        # Reserve the constructive first share before admitting each candidate.
        # A short final tail may cost less than one quantum, so do not impose a
        # fixed budget // quantum request-count cap: later short requests can
        # consume otherwise stranded tail capacity.
        first_share_budget = prefill_budget
        prepared: list[PreparedPrefillReq] = []
        for pending_req in self.pending_list:
            known_remaining = (
                pending_req.input_len - pending_req.chunked_req.cached_len
                if pending_req.chunked_req is not None
                else pending_req.input_len
            )
            conservative_first_share = min(alignment_quantum, known_remaining)
            if conservative_first_share > first_share_budget:
                continue
            candidate = adder.try_prepare_one(pending_req)
            if candidate is not None:
                actual_first_share = min(
                    alignment_quantum,
                    candidate.remaining_tokens,
                )
                if actual_first_share > first_share_budget:
                    raise RuntimeError(
                        "prepared prefill request exceeded reserved first-share budget"
                    )
                prepared.append(candidate)
                first_share_budget -= actual_first_share

        if not prepared and prefill_budget > 0:
            # Non-release diagnostic budgets smaller than one page still make
            # finite progress. Production 8192/2048 budgets never take this path.
            for pending_req in self.pending_list:
                candidate = adder.try_prepare_one(pending_req)
                if candidate is not None:
                    prepared.append(candidate)
                    break

        if not prepared:
            return None

        chunks = allocate_fair_prefill_chunks(
            token_budget=prefill_budget,
            alignment_quantum=alignment_quantum,
            scheduled_tokens=[
                candidate.pending_req.prefill_scheduled_tokens
                for candidate in prepared
            ],
            remaining_tokens=[
                candidate.remaining_tokens for candidate in prepared
            ],
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        selected_ids = {id(candidate.pending_req) for candidate in prepared}
        for candidate, chunk_size in zip(prepared, chunks, strict=True):
            pending_req = candidate.pending_req
            req = adder._add_one_req(
                pending_req=pending_req,
                cache_handle=candidate.cache_handle,
                table_idx=candidate.table_idx,
                cached_len=candidate.cached_len,
                chunk_size=chunk_size,
            )
            pending_req.prefill_scheduled_tokens += chunk_size
            pending_req.chunked_req = None
            if isinstance(req, ChunkedReq):
                pending_req.chunked_req = req
                chunked_list.append(pending_req)
            reqs.append(req)

        # Requests that did not receive this forward's first share lead the
        # next pass; unfinished selected requests rotate behind them. This is a
        # deterministic FIFO round-robin and prevents a long resident request
        # from permanently monopolizing the queue head.
        unscheduled = [
            pending_req
            for pending_req in self.pending_list
            if id(pending_req) not in selected_ids
        ]
        self.pending_list = unscheduled + chunked_list
        if sum(req.extend_len for req in reqs) > prefill_budget:
            raise RuntimeError("prefill batch exceeded configured total token budget")
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.reasoning import ReasoningState, ReasoningTokenIds
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def mask_reasoning_logits_(
    logits: torch.Tensor,
    reasoning_states: torch.Tensor,
    token_ids: ReasoningTokenIds,
    *,
    current_input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the DSV4 structural grammar before every sampling transform.

    Each row always forbids ``<think>``.  Its other forbidden token is EOS in
    THINKING and ``</think>`` in CHAT/ANSWER.  During decode, a current input
    of ``</think>`` promotes a host-stale THINKING row for the immediately
    following sample, which is the overlap-scheduling contract.
    """

    if logits.ndim != 2:
        raise ValueError(f"reasoning logits must be rank 2, got shape={tuple(logits.shape)}")
    rows = logits.shape[0]
    if reasoning_states.ndim != 1 or reasoning_states.numel() != rows:
        raise ValueError(
            "reasoning state rows must match logits rows: "
            f"states={tuple(reasoning_states.shape)}, logits={tuple(logits.shape)}"
        )
    thinking = reasoning_states == int(ReasoningState.THINKING)
    if current_input_ids is not None:
        if current_input_ids.ndim != 1 or current_input_ids.numel() != rows:
            raise ValueError(
                "decode input rows must match logits rows: "
                f"input_ids={tuple(current_input_ids.shape)}, logits={tuple(logits.shape)}"
            )
        thinking = thinking & (current_input_ids != token_ids.think_end)

    # Two structural writes per row preserve the conditional distribution over
    # every allowed token and occur before argmax, temperature, top-k, or top-p.
    logits[:, token_ids.think_start] = -torch.inf
    second_forbidden = torch.where(
        thinking,
        torch.full_like(reasoning_states, token_ids.eos),
        torch.full_like(reasoning_states, token_ids.think_end),
    )
    logits.scatter_(1, second_forbidden.unsqueeze(1).to(torch.int64), -torch.inf)
    return logits


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(
        self,
        logits: torch.Tensor,
        args: BatchSamplingArgs,
        batch: Batch | None = None,
    ) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)


@dataclass
class ReasoningSampler(Sampler):
    """Production sampler with the DeepSeek V4 grammar contract enabled."""

    reasoning_token_ids: ReasoningTokenIds

    def __post_init__(self) -> None:
        for label, token_id in self.reasoning_token_ids.__dict__.items():
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"reasoning {label} token ID {token_id} is outside vocab_size "
                    f"{self.vocab_size}"
                )

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        batch.reasoning_states = make_device_tensor(
            [int(r.reasoning_state) for r in batch.reqs],
            torch.int32,
            self.device,
        )
        return super().prepare(batch)

    @nvtx_annotate("Sampler")
    def sample(
        self,
        logits: torch.Tensor,
        args: BatchSamplingArgs,
        batch: Batch | None = None,
    ) -> torch.Tensor:
        if batch is None:
            raise ValueError("reasoning sampler requires batch protocol metadata")
        with torch.cuda.nvtx.range("Sampler"):
            mask_reasoning_logits_(
                logits,
                batch.reasoning_states,
                self.reasoning_token_ids,
                current_input_ids=(
                    batch.input_ids[: batch.size] if batch.is_decode else None
                ),
            )
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)

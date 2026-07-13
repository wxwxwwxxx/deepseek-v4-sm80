from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class ReasoningState(IntEnum):
    """Generation-time DeepSeek V4 wire-protocol state."""

    CHAT = 0
    THINKING = 1
    ANSWER = 2


@dataclass(frozen=True)
class ReasoningTokenIds:
    bos: int
    eos: int
    think_start: int
    think_end: int


def initial_reasoning_state(reasoning_effort: str | None) -> ReasoningState:
    if reasoning_effort is None:
        return ReasoningState.CHAT
    if reasoning_effort in ("high", "max"):
        return ReasoningState.THINKING
    raise ValueError(
        "reasoning_effort must be None, 'high', or 'max', " f"got {reasoning_effort!r}"
    )


def advance_reasoning_state(
    state: ReasoningState,
    generated_token_id: int,
    *,
    think_end_token_id: int,
) -> ReasoningState:
    """Advance exactly once when THINKING generates the legal delimiter."""

    if state == ReasoningState.THINKING and generated_token_id == think_end_token_id:
        return ReasoningState.ANSWER
    return state


def effective_reasoning_state(
    state: ReasoningState,
    *,
    current_input_token_id: int | None,
    think_end_token_id: int,
) -> ReasoningState:
    """Account for the one-token CPU-state lag in overlap decode.

    During overlapping scheduling the next decode can launch before the CPU has
    consumed the preceding sample.  That sample is already the current GPU
    input token, so a closing delimiter makes the immediately following sample
    an ANSWER sample even while the request object's host state is THINKING.
    """

    if state == ReasoningState.THINKING and current_input_token_id == think_end_token_id:
        return ReasoningState.ANSWER
    return state


def _resolve_single_token_id(
    tokenizer: PreTrainedTokenizerBase,
    marker: str,
    *,
    label: str,
) -> int:
    encoded = tokenizer.encode(marker, add_special_tokens=False)
    converted = tokenizer.convert_tokens_to_ids(marker)
    if not isinstance(encoded, list) or len(encoded) != 1:
        raise RuntimeError(
            "DeepSeek V4 reasoning sampler contract requires "
            f"{label} {marker!r} to encode as exactly one token; got {encoded!r}."
        )
    token_id = int(encoded[0])
    if converted is None or int(converted) != token_id:
        raise RuntimeError(
            "DeepSeek V4 tokenizer token lookup disagrees with encoding for "
            f"{label} {marker!r}: encode={encoded!r}, convert={converted!r}."
        )
    return token_id


def resolve_reasoning_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> ReasoningTokenIds:
    """Resolve and validate all formatter-owned structural token IDs."""

    if tokenizer.bos_token is None or tokenizer.bos_token_id is None:
        raise RuntimeError("DeepSeek V4 tokenizer must define a BOS token and token ID.")
    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        raise RuntimeError("DeepSeek V4 tokenizer must define an EOS token and token ID.")

    bos = _resolve_single_token_id(tokenizer, tokenizer.bos_token, label="BOS")
    eos = _resolve_single_token_id(tokenizer, tokenizer.eos_token, label="EOS")
    if bos != int(tokenizer.bos_token_id):
        raise RuntimeError(
            "DeepSeek V4 BOS encoding disagrees with bos_token_id: "
            f"encode={bos}, bos_token_id={tokenizer.bos_token_id}."
        )
    if eos != int(tokenizer.eos_token_id):
        raise RuntimeError(
            "DeepSeek V4 EOS encoding disagrees with eos_token_id: "
            f"encode={eos}, eos_token_id={tokenizer.eos_token_id}."
        )

    token_ids = ReasoningTokenIds(
        bos=bos,
        eos=eos,
        think_start=_resolve_single_token_id(tokenizer, "<think>", label="opening marker"),
        think_end=_resolve_single_token_id(tokenizer, "</think>", label="closing marker"),
    )
    values = tuple(token_ids.__dict__.values())
    if len(set(values)) != len(values):
        raise RuntimeError(
            "DeepSeek V4 reasoning structural token IDs must be distinct; " f"resolved {token_ids}."
        )
    return token_ids

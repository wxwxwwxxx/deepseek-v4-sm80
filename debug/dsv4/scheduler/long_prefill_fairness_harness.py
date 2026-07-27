#!/usr/bin/env python3
"""No-weight TARGET 16.31 resident-batch scheduler harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from minisgl.core import SamplingParams
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.phase_policy import MixedPhaseFairPolicy
from minisgl.scheduler.prefill import (
    PrefillManager,
    allocate_fair_prefill_chunks,
)
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq

PAGE_SIZE = 256
RELEASE_BUDGET = 8192
MIXED_BUDGET = 2048


class _Handle:
    def __init__(self, cached_len: int):
        self.cached_len = cached_len

    def get_matched_indices(self) -> torch.Tensor:
        return torch.arange(self.cached_len, dtype=torch.int32)


class _SequenceSlots:
    def __init__(self) -> None:
        self.acquired: list[tuple[int, int]] = []
        self.released: list[tuple[int, int]] = []

    def acquire_sequence_slot(self, table_idx: int, generation_id: int) -> None:
        self.acquired.append((table_idx, generation_id))

    def release_sequence_slot(self, table_idx: int, generation_id: int) -> None:
        self.released.append((table_idx, generation_id))


class _CacheManager:
    page_size = PAGE_SIZE

    def __init__(
        self,
        *,
        capacity_tokens: int,
        prefix_lens: dict[int, int] | None = None,
    ) -> None:
        self.available_size = capacity_tokens
        self.prefix_lens = prefix_lens or {}
        self.kv_cache = _SequenceSlots()
        self.locked: list[_Handle] = []

    def match_req(self, req: PendingReq):
        return SimpleNamespace(
            cuda_handle=_Handle(self.prefix_lens.get(req.uid, 0))
        )

    def lock(self, handle: _Handle) -> None:
        self.locked.append(handle)

    def unlock(self, handle: _Handle) -> None:
        self.locked.remove(handle)


def _pending(
    uid: int,
    input_len: int,
    *,
    generation_id: int | None = None,
) -> PendingReq:
    req = PendingReq(
        uid=uid,
        input_ids=torch.arange(input_len, dtype=torch.int32) + uid * 100_000,
        sampling_params=SamplingParams(max_tokens=8, ignore_eos=True),
    )
    req.lifecycle.generation_id = uid if generation_id is None else generation_id
    return req


def _manager(
    input_lens: list[int],
    *,
    slots: int | None = None,
    capacity_tokens: int = 10_000_000,
    prefix_lens: dict[int, int] | None = None,
) -> tuple[PrefillManager, _CacheManager, TableManager]:
    slot_count = len(input_lens) if slots is None else slots
    max_len = max(input_lens) + 8
    table = TableManager(
        slot_count,
        torch.empty((slot_count, max_len), dtype=torch.int32),
    )
    cache = _CacheManager(
        capacity_tokens=capacity_tokens,
        prefix_lens=prefix_lens,
    )
    manager = PrefillManager(cache, table, DecodeManager(PAGE_SIZE))
    manager.pending_list = [
        _pending(uid, input_len) for uid, input_len in enumerate(input_lens)
    ]
    return manager, cache, table


def _batch_row(batch) -> dict[str, Any]:
    return {
        "request_ids": [req.uid for req in batch.reqs],
        "chunk_sizes": [req.extend_len for req in batch.reqs],
        "total_tokens": sum(req.extend_len for req in batch.reqs),
        "table_slots": [req.table_idx for req in batch.reqs],
        "cached_lens": [req.cached_len for req in batch.reqs],
    }


def _first_batch_case(
    request_count: int,
    budget: int,
    *,
    input_len: int = 65_536,
) -> dict[str, Any]:
    manager, cache, _ = _manager([input_len] * request_count)
    batch = manager.schedule_next_batch(budget)
    assert batch is not None
    row = _batch_row(batch)
    return {
        "request_count": request_count,
        "budget": budget,
        **row,
        "budget_respected": row["total_tokens"] <= budget,
        "all_requests_positive": all(value > 0 for value in row["chunk_sizes"]),
        "max_progress_delta": max(row["chunk_sizes"]) - min(row["chunk_sizes"]),
        "alignment_quantum": PAGE_SIZE,
        "all_intermediate_chunks_aligned": all(
            value % PAGE_SIZE == 0 for value in row["chunk_sizes"]
        ),
        "stable_slots_acquired": [list(value) for value in cache.kv_cache.acquired],
        "pass": (
            row["total_tokens"] <= budget
            and len(row["request_ids"]) == request_count
            and all(value > 0 for value in row["chunk_sizes"])
            and max(row["chunk_sizes"]) - min(row["chunk_sizes"]) <= PAGE_SIZE
        ),
    }


def _legacy_serialized_owner() -> dict[str, Any]:
    """Faithful host-only model of the pre-change PrefillManager loop."""
    prompt_len = 512 * 1024
    budget = 256
    pending = [prompt_len] * 4
    decode_remaining: list[int] = []
    policy = MixedPhaseFairPolicy(isolated_prefill_budget=budget)
    first_rows = []
    prefill_batches = 0
    decode_batches = 0
    prefill_request_batch_max = 0
    decode_m_max = 0
    resident_owner_max = 0
    step = 0
    while pending or decode_remaining:
        prefill_runnable = bool(pending)
        decode_runnable = bool(decode_remaining)
        decision = policy.choose(
            prefill_runnable=prefill_runnable,
            decode_runnable=decode_runnable,
        )
        phase = decision.phase
        if phase == "prefill":
            chunk = min(decision.prefill_budget, pending[0])
            pending[0] -= chunk
            if pending[0] == 0:
                pending.pop(0)
                # The terminal prefill forward produces the first output token.
                decode_remaining.append(7)
            prefill_batches += 1
            prefill_request_batch_max = max(prefill_request_batch_max, 1)
            row = {
                "step": step,
                "phase": phase,
                "request_batch": 1,
                "chunk_tokens": chunk,
                "pending_head_remaining": pending[0] if pending else 0,
            }
        else:
            decode_m = len(decode_remaining)
            decode_remaining = [value - 1 for value in decode_remaining]
            decode_remaining = [value for value in decode_remaining if value > 0]
            decode_batches += 1
            decode_m_max = max(decode_m_max, decode_m)
            row = {
                "step": step,
                "phase": phase,
                "request_batch": decode_m,
                "chunk_tokens": 0,
                "pending_head_remaining": pending[0] if pending else 0,
            }
        if step < 3 or step in (2047, 2048, 2055, 2056):
            first_rows.append(row)
        policy.record_scheduled(
            phase,
            prefill_runnable=prefill_runnable,
            decode_runnable=decode_runnable,
        )
        resident_owner_max = max(
            resident_owner_max,
            int(bool(pending)) + int(bool(decode_remaining)),
        )
        step += 1

    return {
        "kind": "pre-change production-loop model",
        "requests": 4,
        "prompt_tokens_per_request": prompt_len,
        "configured_total_budget": budget,
        "first_rows": first_rows,
        "prefill_batches": prefill_batches,
        "decode_batches": decode_batches,
        "prefill_request_batch_max": prefill_request_batch_max,
        "decode_m_max": decode_m_max,
        "resident_owner_max": resident_owner_max,
        "unfinished_request_returns_to_head": True,
        "exact_pre_change_live_run_observed_steps": 8220,
        "exact_pre_change_live_run_observed_prefill_batch_max": 1,
        "exact_pre_change_live_run_observed_decode_m_max": 1,
        "pass": (
            prefill_batches == 8192
            and decode_batches == 28
            and prefill_request_batch_max == 1
            and decode_m_max == 1
        ),
    }


def _resident_decode_m4() -> dict[str, Any]:
    manager, cache, _ = _manager([65_536] * 4)
    rows = []
    while manager.runnable:
        batch = manager.schedule_next_batch(RELEASE_BUDGET)
        assert batch is not None
        rows.append(_batch_row(batch))
        for req in batch.reqs:
            req.complete_one()
        manager.decode_manager.filter_reqs(batch.reqs)
    decode_ids = sorted(req.uid for req in manager.decode_manager.running_reqs)
    slots = sorted(req.table_idx for req in manager.decode_manager.running_reqs)
    return {
        "prefill_forwards": len(rows),
        "prefill_request_batch_max": max(len(row["request_ids"]) for row in rows),
        "prefill_total_token_max": max(row["total_tokens"] for row in rows),
        "decode_m_after_prefill": len(decode_ids),
        "decode_request_ids": decode_ids,
        "overlapping_live_sequence_slots": slots,
        "sequence_slot_acquires": [list(value) for value in cache.kv_cache.acquired],
        "all_forward_totals_within_budget": all(
            row["total_tokens"] <= RELEASE_BUDGET for row in rows
        ),
        "equal_progress_each_forward": all(
            max(row["chunk_sizes"]) - min(row["chunk_sizes"]) <= PAGE_SIZE
            for row in rows
        ),
        "pass": (
            len(decode_ids) == 4
            and len(slots) == 4
            and all(row["total_tokens"] <= RELEASE_BUDGET for row in rows)
        ),
    }


def _prefix_case() -> dict[str, Any]:
    manager, _, _ = _manager(
        [8193, 8193, 8193],
        prefix_lens={0: 8192, 1: 4096, 2: 0},
    )
    batch = manager.schedule_next_batch(RELEASE_BUDGET)
    assert batch is not None
    return {
        **_batch_row(batch),
        "prefix_kinds": ["full-safe-prefix", "partial", "miss"],
        "uncached_offsets_preserved": [req.cached_len for req in batch.reqs]
        == [8192, 4096, 0],
        "pass": all(req.extend_len > 0 for req in batch.reqs),
    }


def _abort_and_reuse_case() -> dict[str, Any]:
    manager, cache, table = _manager([65_536, 65_536], slots=2)
    first = manager.schedule_next_batch(RELEASE_BUDGET)
    assert first is not None
    for req in first.reqs:
        req.complete_one()
    aborted = manager.abort_req(0)
    assert aborted is not None
    cache.kv_cache.release_sequence_slot(
        aborted.table_idx, aborted.lifecycle.generation_id
    )
    table.free(aborted.table_idx)
    manager.pending_list.append(_pending(2, 65_536, generation_id=2002))
    second = manager.schedule_next_batch(RELEASE_BUDGET)
    assert second is not None
    replacement = next(req for req in second.reqs if req.uid == 2)
    return {
        "aborted_uid": aborted.uid,
        "aborted_slot": aborted.table_idx,
        "aborted_generation": aborted.lifecycle.generation_id,
        "replacement_uid": replacement.uid,
        "replacement_slot": replacement.table_idx,
        "replacement_generation": replacement.lifecycle.generation_id,
        "release_events": [list(value) for value in cache.kv_cache.released],
        "acquire_events": [list(value) for value in cache.kv_cache.acquired],
        "pass": (
            replacement.table_idx == aborted.table_idx
            and replacement.lifecycle.generation_id == 2002
            and len(cache.kv_cache.released) == 1
        ),
    }


def _resource_pressure_cases() -> dict[str, Any]:
    capacity_manager, _, _ = _manager(
        [65_536, 65_536, 2048],
        capacity_tokens=68_000,
    )
    capacity_batch = capacity_manager.schedule_next_batch(RELEASE_BUDGET)
    assert capacity_batch is not None
    table_manager, table_cache, _ = _manager([65_536] * 4, slots=2)
    first = table_manager.schedule_next_batch(RELEASE_BUDGET)
    assert first is not None
    first_slots = {req.uid: req.table_idx for req in first.reqs}
    for req in first.reqs:
        req.complete_one()
    second = table_manager.schedule_next_batch(RELEASE_BUDGET)
    assert second is not None
    return {
        "capacity_subset": {
            **_batch_row(capacity_batch),
            "pending_after": [req.uid for req in capacity_manager.pending_list],
            "short_bypassed_unreservable_long": [
                req.uid for req in capacity_batch.reqs
            ]
            == [0, 2],
        },
        "table_subset": {
            "first": _batch_row(first),
            "second": _batch_row(second),
            "stable_slots": {
                req.uid: req.table_idx for req in second.reqs
            }
            == first_slots,
            "acquire_events": [list(value) for value in table_cache.kv_cache.acquired],
        },
        "pass": (
            [req.uid for req in capacity_batch.reqs] == [0, 2]
            and {req.uid: req.table_idx for req in second.reqs} == first_slots
        ),
    }


def _mixed_phase_case() -> dict[str, Any]:
    policy = MixedPhaseFairPolicy(isolated_prefill_budget=RELEASE_BUDGET)
    phases = []
    budgets = []
    for _ in range(5):
        decision = policy.choose(prefill_runnable=True, decode_runnable=True)
        phases.append(decision.phase)
        budgets.append(decision.prefill_budget)
        policy.record_scheduled(
            decision.phase,
            prefill_runnable=True,
            decode_runnable=True,
        )
    manager, _, _ = _manager([65_536] * 4)
    batch = manager.schedule_next_batch(MIXED_BUDGET)
    assert batch is not None
    return {
        "phase_sequence": phases,
        "prefill_budgets": budgets,
        "mixed_prefill_batch": _batch_row(batch),
        "pass": (
            phases == ["decode", "decode", "decode", "decode", "prefill"]
            and [req.extend_len for req in batch.reqs] == [512] * 4
        ),
    }


def build_report() -> dict[str, Any]:
    matrix = {
        "one_request_8192": _first_batch_case(1, RELEASE_BUDGET),
        "two_requests_8192": _first_batch_case(2, RELEASE_BUDGET),
        "four_requests_8192": _first_batch_case(4, RELEASE_BUDGET),
        "eight_requests_8192": _first_batch_case(8, RELEASE_BUDGET),
        "four_requests_mixed_2048": _first_batch_case(4, MIXED_BUDGET),
    }
    unequal_chunks = allocate_fair_prefill_chunks(
        token_budget=RELEASE_BUDGET,
        alignment_quantum=PAGE_SIZE,
        scheduled_tokens=[0, 0, 0, 0],
        remaining_tokens=[100, 1000, 9000, 9000],
    )
    report = {
        "target": "16.31",
        "kind": "no-weight production scheduler harness",
        "page_size": PAGE_SIZE,
        "alignment_quantum": PAGE_SIZE,
        "baseline_serialized_owner": _legacy_serialized_owner(),
        "matrix": matrix,
        "unequal_tails": {
            "remaining_tokens": [100, 1000, 9000, 9000],
            "chunk_sizes": unequal_chunks,
            "total_tokens": sum(unequal_chunks),
            "unused_budget": RELEASE_BUDGET - sum(unequal_chunks),
            "pass": RELEASE_BUDGET - sum(unequal_chunks) < PAGE_SIZE,
        },
        "prefix_hit": _prefix_case(),
        "abort_and_slot_reuse": _abort_and_reuse_case(),
        "resource_pressure": _resource_pressure_cases(),
        "mixed_decode": _mixed_phase_case(),
        "resident_decode_m4": _resident_decode_m4(),
    }
    leaves = [
        report["baseline_serialized_owner"]["pass"],
        *(case["pass"] for case in matrix.values()),
        report["unequal_tails"]["pass"],
        report["prefix_hit"]["pass"],
        report["abort_and_slot_reuse"]["pass"],
        report["resource_pressure"]["pass"],
        report["mixed_decode"]["pass"],
        report["resident_decode_m4"]["pass"],
    ]
    report["all_pass"] = all(leaves)
    report["phase_policy"] = {
        "max_consecutive_decode": 4,
        "mixed_prefill_budget": MIXED_BUDGET,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.phase_policy import MixedPhaseFairPolicy
from minisgl.scheduler.prefill import (
    ChunkedReq,
    PrefillManager,
    allocate_fair_prefill_chunks,
)
from minisgl.scheduler.table import TableManager
from minisgl.scheduler.utils import PendingReq


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
    page_size = 256

    def __init__(
        self,
        *,
        capacity_tokens: int = 10_000_000,
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
    output_len: int = 8,
    generation_id: int | None = None,
) -> PendingReq:
    req = PendingReq(
        uid=uid,
        input_ids=torch.arange(input_len, dtype=torch.int32) + uid * 100_000,
        sampling_params=SamplingParams(max_tokens=output_len, ignore_eos=True),
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
    manager = PrefillManager(cache, table, DecodeManager(page_size=256))
    manager.pending_list = [
        _pending(uid, input_len) for uid, input_len in enumerate(input_lens)
    ]
    return manager, cache, table


@pytest.mark.parametrize(
    ("request_count", "expected_chunk"),
    [(1, 8192), (2, 4096), (4, 2048), (8, 1024)],
)
def test_release_budget_is_total_and_fair_across_long_requests(
    request_count: int, expected_chunk: int
) -> None:
    manager, cache, _ = _manager([65_536] * request_count)

    batch = manager.schedule_next_batch(8192)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == list(range(request_count))
    assert [req.extend_len for req in batch.reqs] == [
        expected_chunk
    ] * request_count
    assert sum(req.extend_len for req in batch.reqs) == 8192
    assert all(isinstance(req, ChunkedReq) for req in batch.reqs)
    assert len({req.table_idx for req in batch.reqs}) == request_count
    assert len(cache.kv_cache.acquired) == request_count


def test_mixed_budget_advances_four_prefills_without_changing_phase_bound() -> None:
    manager, _, _ = _manager([65_536] * 4)
    policy = MixedPhaseFairPolicy(isolated_prefill_budget=8192)

    phases = []
    for _ in range(5):
        decision = policy.choose(prefill_runnable=True, decode_runnable=True)
        phases.append(decision.phase)
        policy.record_scheduled(
            decision.phase,
            prefill_runnable=True,
            decode_runnable=True,
        )

    assert phases == ["decode", "decode", "decode", "decode", "prefill"]
    batch = manager.schedule_next_batch(policy.mixed_prefill_budget)
    assert batch is not None
    assert [req.extend_len for req in batch.reqs] == [512] * 4
    assert sum(req.extend_len for req in batch.reqs) == 2048


def test_unequal_tails_redistribute_budget_and_preserve_alignment() -> None:
    chunks = allocate_fair_prefill_chunks(
        token_budget=8192,
        alignment_quantum=256,
        scheduled_tokens=[0, 0, 0, 0],
        remaining_tokens=[100, 1000, 9000, 9000],
    )

    assert chunks == [100, 1000, 3584, 3328]
    # Arbitrary final tails can leave less than one 256-token quantum. Do not
    # create an unaligned intermediate boundary merely to consume 180 tokens.
    assert sum(chunks) == 8012
    assert 8192 - sum(chunks) < 256
    assert chunks[2] % 256 == chunks[3] % 256 == 0
    assert abs(chunks[2] - chunks[3]) <= 256


def test_remainder_bias_does_not_accumulate_across_forwards() -> None:
    scheduled = [0, 0, 0]
    for _ in range(8):
        chunks = allocate_fair_prefill_chunks(
            token_budget=8192,
            alignment_quantum=256,
            scheduled_tokens=scheduled,
            remaining_tokens=[65_536 - value for value in scheduled],
        )
        scheduled = [
            progress + chunk
            for progress, chunk in zip(scheduled, chunks, strict=True)
        ]
        assert max(scheduled) - min(scheduled) <= 256


def test_prefix_hits_keep_uncached_offset_and_receive_positive_shares() -> None:
    manager, _, _ = _manager(
        [8193, 8193, 8193],
        prefix_lens={0: 8192, 1: 4096, 2: 0},
    )

    batch = manager.schedule_next_batch(8192)

    assert batch is not None
    assert [req.cached_len for req in batch.reqs] == [8192, 4096, 0]
    assert [req.extend_len for req in batch.reqs] == [1, 4097, 3840]
    assert 8192 - sum(req.extend_len for req in batch.reqs) < 256
    assert all(req.extend_len > 0 for req in batch.reqs)


def test_short_request_can_bypass_unreservable_long_request() -> None:
    manager, _, _ = _manager(
        [65_536, 65_536, 2048],
        capacity_tokens=68_000,
    )

    batch = manager.schedule_next_batch(8192)

    assert batch is not None
    assert [req.uid for req in batch.reqs] == [0, 2]
    assert [req.extend_len for req in batch.reqs] == [6144, 2048]
    assert [req.uid for req in manager.pending_list] == [1, 0]


def test_short_request_behind_budget_saturated_longs_runs_next_round() -> None:
    manager, _, _ = _manager([4096, 4096, 1])
    first = manager.schedule_next_batch(512)
    assert first is not None
    for req in first.reqs:
        req.complete_one()

    second = manager.schedule_next_batch(512)

    assert second is not None
    assert [req.uid for req in first.reqs] == [0, 1]
    assert second.reqs[0].uid == 2
    assert second.reqs[0].extend_len == 1
    assert sum(req.extend_len for req in second.reqs) <= 512


def test_table_pressure_schedules_safe_subset_and_keeps_stable_slots() -> None:
    manager, cache, _ = _manager([65_536] * 4, slots=2)

    first = manager.schedule_next_batch(8192)
    assert first is not None
    first_slots = {req.uid: req.table_idx for req in first.reqs}
    for req in first.reqs:
        req.complete_one()

    second = manager.schedule_next_batch(8192)

    assert second is not None
    assert [req.uid for req in first.reqs] == [0, 1]
    assert [req.uid for req in second.reqs] == [0, 1]
    assert {req.uid: req.table_idx for req in second.reqs} == first_slots
    assert cache.kv_cache.acquired == [
        (first_slots[0], 0),
        (first_slots[1], 1),
    ]


def test_abort_owner_can_release_once_and_slot_is_reused_by_new_generation() -> None:
    manager, cache, table = _manager([65_536, 65_536], slots=2)
    first = manager.schedule_next_batch(8192)
    assert first is not None
    for req in first.reqs:
        req.complete_one()

    aborted = manager.abort_req(0)
    assert aborted is not None
    cache.kv_cache.release_sequence_slot(
        aborted.table_idx, aborted.lifecycle.generation_id
    )
    table.free(aborted.table_idx)
    manager.pending_list.append(
        _pending(2, 65_536, generation_id=2002)
    )

    second = manager.schedule_next_batch(8192)

    assert second is not None
    replacement = next(req for req in second.reqs if req.uid == 2)
    assert replacement.table_idx == aborted.table_idx
    assert replacement.lifecycle.generation_id == 2002
    assert cache.kv_cache.released == [
        (aborted.table_idx, aborted.lifecycle.generation_id)
    ]
    assert cache.kv_cache.acquired.count((replacement.table_idx, 2002)) == 1


def test_allocation_and_order_are_deterministic() -> None:
    traces = []
    for _ in range(3):
        manager, _, _ = _manager([65_536] * 4)
        trace = []
        for _ in range(4):
            batch = manager.schedule_next_batch(8192)
            assert batch is not None
            trace.append(
                (
                    [req.uid for req in batch.reqs],
                    [req.extend_len for req in batch.reqs],
                    [req.table_idx for req in batch.reqs],
                )
            )
            for req in batch.reqs:
                req.complete_one()
        traces.append(trace)

    assert traces[0] == traces[1] == traces[2]

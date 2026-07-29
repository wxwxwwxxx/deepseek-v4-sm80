from __future__ import annotations

from types import SimpleNamespace

import minisgl.core as core
import torch
from minisgl.core import SamplingParams
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder
from minisgl.scheduler.utils import PendingReq


class _FakeDsv4KVCache:
    def __init__(self, *, component_pages: int, swa_pages: int):
        self._component_pages = component_pages
        self._swa_pages = swa_pages

    def available_component_pages(self) -> int:
        return self._component_pages

    def available_swa_pages(self) -> int:
        return self._swa_pages


class _SequenceStateKVCache:
    def __init__(self) -> None:
        self.acquired: list[tuple[int, int]] = []

    def acquire_sequence_slot(self, table_idx: int, generation_id: int) -> None:
        self.acquired.append((table_idx, generation_id))


class _MissHandle:
    cached_len = 0


def test_chunked_prefill_acquires_sequence_state_once_and_keeps_stable_table_slot():
    kv_cache = _SequenceStateKVCache()
    handle = _MissHandle()
    cache_manager = SimpleNamespace(
        kv_cache=kv_cache,
        available_size=128,
        match_req=lambda req: SimpleNamespace(cuda_handle=handle),
        lock=lambda matched: None,
    )
    token_pool = torch.empty((1, 32), dtype=torch.int32)
    table_manager = SimpleNamespace(
        available_size=1,
        allocate=lambda: 0,
        token_pool=token_pool,
        page_table=torch.empty((1, 32), dtype=torch.int32),
    )
    pending = PendingReq(
        uid=7,
        input_ids=torch.arange(9, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=2),
    )
    pending.lifecycle.generation_id = 1601

    first = PrefillAdder(3, 0, cache_manager, table_manager).try_add_one(pending)
    assert isinstance(first, ChunkedReq)
    pending.chunked_req = first
    second = PrefillAdder(3, 0, cache_manager, table_manager).try_add_one(pending)

    assert isinstance(second, ChunkedReq)
    assert first.table_idx == second.table_idx == 0
    assert first.lifecycle is second.lifecycle
    assert kv_cache.acquired == [(0, 1601)]


def test_dsv4_swa_tail_pages_do_not_cap_full_request_admission_capacity():
    page_size = 4
    ctx = core.Context(page_size=page_size)
    old_ctx = core._GLOBAL_CTX
    core.set_global_ctx(ctx)
    try:
        manager = CacheManager(
            num_pages=100,
            page_size=page_size,
            page_table=torch.empty((1,), dtype=torch.int32),
            kv_cache=_FakeDsv4KVCache(component_pages=80, swa_pages=2),
        )

        assert manager.available_size == 80 * page_size
    finally:
        core._GLOBAL_CTX = old_ctx


def test_chunked_prefill_carries_swa_eviction_frontier_to_next_chunk():
    token_pool = torch.empty((1, 32), dtype=torch.int32)
    table_manager = SimpleNamespace(token_pool=token_pool)
    adder = PrefillAdder(
        token_budget=8,
        reserved_size=0,
        cache_manager=SimpleNamespace(),
        table_manager=table_manager,
    )
    sampling_params = SamplingParams(max_tokens=4)
    handle = object()
    pending = PendingReq(
        uid=7,
        input_ids=torch.arange(24, dtype=torch.int32),
        sampling_params=sampling_params,
        chunked_req=ChunkedReq(
            input_ids=torch.arange(9, dtype=torch.int32),
            table_idx=0,
            cached_len=8,
            output_len=4,
            uid=7,
            cache_handle=handle,
            sampling_params=sampling_params,
            swa_evicted_seqlen=4096,
        ),
    )

    req = adder._add_one_req(
        pending_req=pending,
        cache_handle=handle,
        table_idx=0,
        cached_len=8,
    )

    assert req.swa_evicted_seqlen == 4096
    assert req.cached_len == 8
    assert req.extend_len == 8
    assert token_pool[0, 8:16].tolist() == list(range(8, 16))

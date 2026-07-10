from __future__ import annotations

from types import SimpleNamespace

import torch

import minisgl.core as core
from minisgl.core import SamplingParams
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.prefill import ChunkedReq, PrefillAdder
from minisgl.scheduler.utils import PendingReq


class _FakeDsv4KVCache:
    component_loc_ownership_enabled = True
    swa_independent_lifecycle_enabled = True

    def __init__(self, *, component_pages: int, swa_pages: int):
        self._component_pages = component_pages
        self._swa_pages = swa_pages

    def available_component_pages(self) -> int:
        return self._component_pages

    def available_swa_pages(self) -> int:
        return self._swa_pages


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
            type="radix",
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

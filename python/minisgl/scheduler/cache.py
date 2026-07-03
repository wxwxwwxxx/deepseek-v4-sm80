from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool, MatchResult, create_prefix_cache
from minisgl.utils import align_down, div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


@dataclass
class PrefixCacheMetrics:
    match_requests: int = 0
    hit_requests: int = 0
    miss_requests: int = 0
    full_hit_requests: int = 0
    partial_hit_requests: int = 0
    total_hit_tokens: int = 0
    max_hit_tokens: int = 0
    saved_prefill_tokens: int = 0
    suffix_prefill_tokens_after_hit: int = 0
    inserted_tokens: int = 0
    evictions: int = 0
    evicted_tokens: int = 0

    def snapshot(self, page_size: int) -> dict[str, Any]:
        hit_rate = (
            0.0 if self.match_requests == 0 else self.hit_requests / self.match_requests
        )
        avg_hit = 0.0 if self.hit_requests == 0 else self.total_hit_tokens / self.hit_requests
        return {
            "match_requests": self.match_requests,
            "hit_requests": self.hit_requests,
            "miss_requests": self.miss_requests,
            "full_hit_requests": self.full_hit_requests,
            "partial_hit_requests": self.partial_hit_requests,
            "hit_rate": hit_rate,
            "total_hit_tokens": self.total_hit_tokens,
            "avg_hit_tokens": avg_hit,
            "max_hit_tokens": self.max_hit_tokens,
            "saved_prefill_tokens": self.saved_prefill_tokens,
            "suffix_prefill_tokens_after_hit": self.suffix_prefill_tokens_after_hit,
            "inserted_tokens": self.inserted_tokens,
            "evictions": self.evictions,
            "evicted_tokens": self.evicted_tokens,
            "evicted_pages": self.evicted_tokens // page_size,
        }


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        kv_cache: BaseKVCachePool | None = None,
    ):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
        self.kv_cache = kv_cache
        self.metrics = PrefixCacheMetrics()

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        result = self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
        self._record_prefix_match(result.cuda_handle.cached_len, input_len)
        return result

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        self.metrics.inserted_tokens += max(0, new_handle.cached_len - cached_len)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        live_pages = self.num_pages - len(self.free_slots)
        if self.kv_cache is not None:
            self.kv_cache.check_allocation_integrity(live_pages, self.page_size)
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            if len(indices) > 0 and self.kv_cache is not None:
                self.kv_cache.on_token_indices_freed(indices, self.page_size)
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self._record_prefix_eviction(evicted)
            if self.kv_cache is not None:
                self.kv_cache.on_token_indices_freed(evicted, self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        if self.kv_cache is not None:
            self.kv_cache.on_pages_allocated(allocated, self.page_size)
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            if self.kv_cache is not None:
                self.kv_cache.on_token_indices_freed(indices, self.page_size)
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()

    def prefix_metrics_snapshot(self) -> dict[str, Any]:
        size_info = self.prefix_cache.size_info
        retained_tokens = size_info.total_size
        snapshot = self.metrics.snapshot(self.page_size)
        snapshot.update(
            {
                "retained_prefix_tokens": retained_tokens,
                "retained_prefix_pages": retained_tokens // self.page_size,
                "evictable_prefix_tokens": size_info.evictable_size,
                "protected_prefix_tokens": size_info.protected_size,
                "evictable_prefix_pages": size_info.evictable_size // self.page_size,
                "protected_prefix_pages": size_info.protected_size // self.page_size,
            }
        )
        retention_estimator = getattr(self.kv_cache, "estimate_prefix_retention", None)
        if callable(retention_estimator):
            snapshot["dsv4_retention"] = retention_estimator(retained_tokens, self.page_size)
        return snapshot

    def _record_prefix_match(self, cached_len: int, input_len: int) -> None:
        self.metrics.match_requests += 1
        safe_matchable_len = align_down(max(input_len - 1, 0), self.page_size)
        if cached_len <= 0:
            self.metrics.miss_requests += 1
            return

        self.metrics.hit_requests += 1
        self.metrics.total_hit_tokens += cached_len
        self.metrics.max_hit_tokens = max(self.metrics.max_hit_tokens, cached_len)
        self.metrics.saved_prefill_tokens += cached_len
        self.metrics.suffix_prefill_tokens_after_hit += max(input_len - cached_len, 0)
        if safe_matchable_len > 0 and cached_len >= safe_matchable_len:
            self.metrics.full_hit_requests += 1
        else:
            self.metrics.partial_hit_requests += 1

    def _record_prefix_eviction(self, evicted: torch.Tensor) -> None:
        evicted_tokens = int(evicted.numel())
        if evicted_tokens == 0:
            return
        self.metrics.evictions += 1
        self.metrics.evicted_tokens += evicted_tokens


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    pin_memory = page_table.is_cuda
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin_memory)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=pin_memory)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool, MatchResult, create_prefix_cache
from minisgl.utils import align_down, div_ceil, dsv4_owner_timing

if TYPE_CHECKING:
    from .utils import PendingReq

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
DSV4_CACHE_SYNC_DEBUG_ENV = "MINISGL_DSV4_CACHE_SYNC_DEBUG"
DSV4_CASE_BOUNDARY_DEBUG_ENV = "MINISGL_DSV4_CASE_BOUNDARY_DEBUG"


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
        self._lazy_free_depth = 0
        self.dsv4_component_ownership = bool(
            getattr(kv_cache, "component_loc_ownership_enabled", False)
        )
        if self.dsv4_component_ownership:
            self.dsv4_swa_independent_lifecycle = bool(
                getattr(kv_cache, "swa_independent_lifecycle_enabled", False)
            )
            setattr(self.prefix_cache, "dsv4_component_ownership_enabled", True)
            setattr(
                self.prefix_cache,
                "dsv4_component_evict_callback",
                self._release_dsv4_component_pages,
            )
            if self.dsv4_swa_independent_lifecycle:
                setattr(self.prefix_cache, "dsv4_swa_independent_lifecycle_enabled", True)
                setattr(
                    self.prefix_cache,
                    "dsv4_swa_evict_callback",
                    self._release_dsv4_swa_pages,
                )
        else:
            self.dsv4_swa_independent_lifecycle = False

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        with dsv4_owner_timing.maybe_host_range(
            "dsv4.scheduler.prefix.match_req",
            {"input_len": int(input_len), "page_size": int(self.page_size)},
        ):
            result = self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
        self._record_prefix_match(result.cuda_handle.cached_len, input_len)
        return result

    @property
    def available_size(self) -> int:
        if self.dsv4_component_ownership:
            live_full_tokens = int(
                getattr(
                    self.prefix_cache,
                    "dsv4_evictable_live_full_tokens",
                    self.prefix_cache.size_info.evictable_size,
                )
            )
            component_tokens = int(
                getattr(
                    self.prefix_cache,
                    "dsv4_evictable_component_tokens",
                    self.prefix_cache.size_info.evictable_size,
                )
            )
            component_available_pages = int(self.kv_cache.available_component_pages())
            full_pages = len(self.free_slots) + live_full_tokens // self.page_size
            component_pages = component_available_pages + component_tokens // self.page_size
            if self.dsv4_swa_independent_lifecycle:
                swa_tokens = int(getattr(self.prefix_cache, "dsv4_evictable_swa_tokens", 0))
                swa_pages = int(self.kv_cache.available_swa_pages()) + swa_tokens // self.page_size
                return min(full_pages, component_pages, swa_pages) * self.page_size
            return min(full_pages, component_pages) * self.page_size
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        with dsv4_owner_timing.maybe_host_range(
            "dsv4.scheduler.page.allocate_paged",
            {"reqs": int(len(reqs)), "page_size": int(self.page_size)},
        ):
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
        with dsv4_owner_timing.maybe_host_range(
            "dsv4.scheduler.prefix.cache_req",
            {
                "finished": bool(finished),
                "cached_len": int(req.cached_len),
                "page_size": int(self.page_size),
                "component_loc_ownership": bool(self.dsv4_component_ownership),
                "swa_independent_lifecycle": bool(self.dsv4_swa_independent_lifecycle),
            },
        ):
            insert_ids = req.input_ids[: req.cached_len]
            page_indices = self.page_table[req.table_idx, : req.cached_len]
            old_handle = req.cache_handle
            if self.dsv4_component_ownership:
                def make_new_component_pages(
                    start: int,
                    end: int,
                ):
                    return self.kv_cache.make_component_page_handles(
                        page_indices[start:end],
                        self.page_size,
                    )

                def make_new_swa_pages(
                    start: int,
                    end: int,
                ):
                    if not self.dsv4_swa_independent_lifecycle:
                        return None
                    handles = self.kv_cache.make_swa_page_handles(
                        page_indices[start:end],
                        self.page_size,
                    )
                    evicted_until = align_down(
                        max(int(getattr(req, "swa_evicted_seqlen", 0)), 0),
                        self.page_size,
                    )
                    tombstone_end = align_down(
                        max(min(int(end), evicted_until) - int(start), 0),
                        self.page_size,
                    )
                    if handles is not None and tombstone_end > 0:
                        handles, released = handles.tombstone_tokens(0, tombstone_end)
                        if released.live_pages > 0:
                            raise RuntimeError(
                                "DSV4 SWA cache boundary found live pages below "
                                "request eviction frontier: "
                                f"uid={getattr(req, 'uid', None)}, start={start}, "
                                f"end={end}, frontier={evicted_until}, "
                                f"live_pages={released.live_pages}"
                            )
                    return handles

                cached_len, new_handle = self.prefix_cache.insert_prefix(
                    insert_ids,
                    page_indices,
                    dsv4_component_pages_builder=make_new_component_pages,
                    dsv4_swa_pages_builder=make_new_swa_pages,
                )
            else:
                cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
            self.metrics.inserted_tokens += max(0, new_handle.cached_len - cached_len)
            already_cached_indices = page_indices[old_handle.cached_len : cached_len].clone()
            if self.dsv4_component_ownership and cached_len > old_handle.cached_len:
                matched_indices = new_handle.get_matched_indices()
                page_indices[old_handle.cached_len : cached_len].copy_(
                    matched_indices[old_handle.cached_len : cached_len]
                )
            # unlock until all operations on handle is done
            self.unlock(old_handle)
            # this part is already in the prefix cache, free it
            self._free(already_cached_indices)
            if finished:  # this tail part should be freed
                self._free(page_indices[new_handle.cached_len :])
            else:  # keep the tail part, update the handle
                req.cache_handle = new_handle
                self.lock(new_handle)
            self._release_dsv4_swa_out_of_window(new_handle)
            self._release_dsv4_component_owned_full_head(new_handle)
            if self.dsv4_component_ownership and not finished and new_handle.cached_len > 0:
                page_indices[: new_handle.cached_len].copy_(new_handle.get_matched_indices())
            self._debug_sync_integrity(
                "cache_req",
                req=req,
                finished=finished,
                cached_len=int(req.cached_len),
                new_cached_len=int(new_handle.cached_len),
            )

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        live_pages = self.num_pages - len(self.free_slots)
        if self.kv_cache is not None:
            if self.dsv4_component_ownership:
                live_full_slots = self.kv_cache.allocation_counts.full_slots
                if live_full_slots % self.page_size != 0:
                    raise RuntimeError(
                        "DSV4 component ownership full-slot refcount is not page-aligned: "
                        f"full_slots={live_full_slots}, page_size={self.page_size}"
                    )
                live_pages = live_full_slots // self.page_size
            self.kv_cache.check_allocation_integrity(live_pages, self.page_size)
        if self.dsv4_component_ownership:
            expected_pages = self.num_pages
            actual_pages = len(self.free_slots) + live_pages
        else:
            expected_pages = self.num_pages
            actual_pages = len(self.free_slots) + cache_pages
        if actual_pages != expected_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" live_pages({live_pages}) + cache_pages({cache_pages}) "
                f"!= num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            valid = self._valid_full_indices(indices)
            if len(valid) > 0 and self.kv_cache is not None:
                self.kv_cache.on_token_indices_freed(valid, self.page_size)
            if len(valid) > 0:
                lazy_free_list.append(valid[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._lazy_free_depth += 1
            self._free = lazy_free
            yield
        finally:
            self._lazy_free_depth -= 1
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)
            self._debug_sync_integrity(
                "lazy_free_region_exit",
                lazy_free_pages=sum(int(chunk.numel()) for chunk in lazy_free_list),
            )

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        with dsv4_owner_timing.maybe_host_range(
            "dsv4.scheduler.page.allocate",
            {
                "needed_pages": int(needed_pages),
                "free_pages_before": int(len(self.free_slots)),
                "component_loc_ownership": bool(self.dsv4_component_ownership),
                "swa_independent_lifecycle": bool(self.dsv4_swa_independent_lifecycle),
            },
        ):
            if self.dsv4_component_ownership:
                self._evict_for_dsv4_component_capacity(needed_pages)
            elif needed_pages > (free_pages := len(self.free_slots)):
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
        with dsv4_owner_timing.maybe_host_range(
            "dsv4.scheduler.page.free",
            {"tokens": int(indices.numel()), "page_size": int(self.page_size)},
        ):
            indices = self._valid_full_indices(indices)
            if len(indices) > 0:
                if self.kv_cache is not None:
                    self.kv_cache.on_token_indices_freed(indices, self.page_size)
                self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _valid_full_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if len(indices) == 0:
            return indices
        if not self.dsv4_component_ownership:
            return indices
        if indices.numel() % self.page_size != 0:
            page_starts = indices[:: self.page_size]
            page_starts = page_starts[page_starts >= 0]
            if page_starts.numel() == 0:
                return indices.new_empty((0,))
            return self._page_to_token(page_starts)
        pages = indices.view(-1, self.page_size)
        valid = pages[:, 0] >= 0
        if not bool(torch.any(valid)):
            return indices.new_empty((0,))
        return pages[valid].reshape(-1)

    def _release_dsv4_component_pages(self, handles) -> None:
        if self.kv_cache is None:
            return
        release = getattr(self.kv_cache, "release_component_page_handles", None)
        if callable(release):
            release(handles)

    def _release_dsv4_swa_pages(self, handles, tombstone: bool = False) -> None:
        if self.kv_cache is None:
            return
        release = getattr(self.kv_cache, "release_swa_page_handles", None)
        if callable(release):
            release(handles, tombstone=tombstone)

    def _release_dsv4_swa_out_of_window(self, handle: BaseCacheHandle) -> None:
        if not self.dsv4_swa_independent_lifecycle:
            return
        releaser = getattr(self.prefix_cache, "release_dsv4_swa_out_of_window", None)
        if not callable(releaser):
            return
        window = int(getattr(self.kv_cache, "_window_size", self.page_size))
        tail_tokens = div_ceil(max(window, 1), self.page_size) * self.page_size
        tail_tokens += self.page_size
        with dsv4_owner_timing.maybe_host_range(
            "dsv4.scheduler.swa.release_prefix_out_of_window",
            {
                "cached_len": int(handle.cached_len),
                "tail_tokens": int(tail_tokens),
                "page_size": int(self.page_size),
            },
        ):
            releaser(handle, tail_tokens=tail_tokens)

    def release_active_dsv4_swa_out_of_window(self, req: Req) -> None:
        if not self.dsv4_swa_independent_lifecycle:
            return
        releaser = getattr(self.kv_cache, "release_swa_for_full_indices", None)
        if not callable(releaser):
            return
        window = int(getattr(self.kv_cache, "_window_size", self.page_size))
        active_window_frontier = align_down(
            max(int(req.device_len) - window - self.page_size, 0),
            self.page_size,
        )
        protected_len = self._dsv4_swa_cache_protected_len(req)
        evicted_until = align_down(
            max(int(getattr(req, "swa_evicted_seqlen", 0)), 0),
            self.page_size,
        )
        release_start = max(evicted_until, protected_len)
        release_end = max(release_start, active_window_frontier)
        if release_end <= release_start:
            return
        active_range = self.page_table[req.table_idx, release_start:release_end].clone()
        active_range = self._valid_full_indices(active_range)
        with dsv4_owner_timing.maybe_host_range(
            "dsv4.scheduler.swa.release_active_out_of_window",
            {
                "release_pages": int(active_range.numel() // self.page_size),
                "release_start": int(release_start),
                "release_end": int(release_end),
                "protected_len": int(protected_len),
                "swa_evicted_seqlen": int(evicted_until),
                "cached_len": int(req.cached_len),
                "device_len": int(req.device_len),
            },
        ):
            if active_range.numel() > 0:
                releaser(active_range, self.page_size, tombstone=True)
            req.swa_evicted_seqlen = release_end

    def _dsv4_swa_cache_protected_len(self, req: Req) -> int:
        handle = getattr(req, "cache_handle", None)
        cached_len = int(getattr(handle, "cached_len", 0) or 0)
        return align_down(max(cached_len, 0), self.page_size)

    def _release_dsv4_component_owned_full_head(self, handle: BaseCacheHandle) -> None:
        if not self.dsv4_component_ownership:
            return
        releaser = getattr(self.prefix_cache, "release_dsv4_full_head", None)
        if not callable(releaser):
            return
        released = releaser(
            handle,
            tail_tokens=self.page_size,
        )
        released = self._valid_full_indices(released)
        if released.numel() == 0:
            return
        if self.kv_cache is not None:
            self.kv_cache.on_token_indices_freed(
                released,
                self.page_size,
                free_components=False,
                free_swa=not self.dsv4_swa_independent_lifecycle,
            )
        self.free_slots = torch.cat([self.free_slots, released[:: self.page_size]])

    def _evict_for_dsv4_component_capacity(self, needed_pages: int) -> None:
        assert self.kv_cache is not None
        while (
            len(self.free_slots) < needed_pages
            or self.kv_cache.available_component_pages() < needed_pages
            or (
                self.dsv4_swa_independent_lifecycle
                and self.kv_cache.available_swa_pages() < needed_pages
            )
        ):
            full_deficit = max(0, needed_pages - len(self.free_slots))
            component_deficit = max(0, needed_pages - self.kv_cache.available_component_pages())
            swa_deficit = (
                max(0, needed_pages - self.kv_cache.available_swa_pages())
                if self.dsv4_swa_independent_lifecycle
                else 0
            )
            if (
                self.dsv4_swa_independent_lifecycle
                and swa_deficit > 0
                and full_deficit == 0
                and component_deficit == 0
            ):
                release_swa = getattr(self.prefix_cache, "release_dsv4_evictable_swa_pages", None)
                released_swa = release_swa(swa_deficit) if callable(release_swa) else 0
                if released_swa > 0:
                    continue
            evict_pages = max(full_deficit, component_deficit, swa_deficit, 1)
            evicted = self.prefix_cache.evict(evict_pages * self.page_size)
            self._record_prefix_eviction(evicted)
            valid = self._valid_full_indices(evicted)
            if valid.numel() > 0:
                self.kv_cache.on_token_indices_freed(
                    valid,
                    self.page_size,
                    free_components=False,
                    free_swa=not self.dsv4_swa_independent_lifecycle,
                )
                self.free_slots = torch.cat([self.free_slots, valid[:: self.page_size]])
            if (
                valid.numel() == 0
                and (
                    self.kv_cache.available_component_pages() < needed_pages
                    or (
                        self.dsv4_swa_independent_lifecycle
                        and self.kv_cache.available_swa_pages() < needed_pages
                    )
                )
                and self.prefix_cache.size_info.evictable_size == 0
            ):
                break
        assert len(self.free_slots) >= needed_pages, "Eviction did not free enough full pages."
        assert (
            self.kv_cache.available_component_pages() >= needed_pages
        ), "Eviction did not free enough DSV4 component pages."
        if self.dsv4_swa_independent_lifecycle:
            assert (
                self.kv_cache.available_swa_pages() >= needed_pages
            ), "Eviction did not free enough DSV4 SWA pages."

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
        if self.dsv4_component_ownership and self.kv_cache is not None:
            counts = self.kv_cache.allocation_counts
            swa_counters = (
                self.kv_cache.runtime_swa_counters()
                if hasattr(self.kv_cache, "runtime_swa_counters")
                else {}
            )
            retained_swa_tokens = int(
                getattr(self.prefix_cache, "dsv4_evictable_swa_tokens", 0)
            ) + int(getattr(self.prefix_cache, "dsv4_protected_swa_tokens", 0))
            retained_swa_pages = retained_swa_tokens // self.page_size
            snapshot["dsv4_component_ownership"] = {
                "enabled": True,
                "live_full_pages": counts.full_slots // self.page_size,
                "live_full_slots": counts.full_slots,
                "live_swa_pages": counts.swa_pages,
                "live_swa_slots": counts.swa_slots,
                "live_c4_slots": counts.c4_slots,
                "live_c128_slots": counts.c128_slots,
                "live_c4_indexer_slots": counts.c4_indexer_slots,
                "live_c4_state_slots": counts.c4_state_slots,
                "live_c128_state_slots": counts.c128_state_slots,
                "live_c4_indexer_state_slots": counts.c4_indexer_state_slots,
                "available_component_pages": self.kv_cache.available_component_pages(),
                "evictable_live_full_tokens": int(
                    getattr(self.prefix_cache, "dsv4_evictable_live_full_tokens", 0)
                ),
                "evictable_component_tokens": int(
                    getattr(self.prefix_cache, "dsv4_evictable_component_tokens", 0)
                ),
            }
            if self.dsv4_swa_independent_lifecycle:
                snapshot["dsv4_swa_lifecycle"] = {
                    **swa_counters,
                    "retained_prefix_swa_pages": retained_swa_pages,
                    "retained_prefix_swa_tokens": retained_swa_tokens,
                    "evictable_prefix_swa_pages": int(
                        getattr(self.prefix_cache, "dsv4_evictable_swa_tokens", 0)
                    )
                    // self.page_size,
                    "protected_prefix_swa_pages": int(
                        getattr(self.prefix_cache, "dsv4_protected_swa_tokens", 0)
                    )
                    // self.page_size,
                    "active_decode_swa_pages_estimate": max(
                        0,
                        int(swa_counters.get("current_swa_tail_pages", 0))
                        - retained_swa_pages,
                    ),
                }
        return snapshot

    def debug_case_boundary_snapshot(
        self,
        stage: str,
        *,
        graph_runner: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if os.environ.get(DSV4_CASE_BOUNDARY_DEBUG_ENV, "").strip().lower() not in _TRUE_ENV_VALUES:
            return {}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.free_slots.numel() > 0:
            if torch.any(self.free_slots < 0) or torch.any(self.free_slots >= self.num_pages * self.page_size):
                raise RuntimeError(f"DSV4 case-boundary debug found out-of-range free slots at {stage}")
            if self.page_size > 1 and torch.any(self.free_slots % self.page_size != 0):
                raise RuntimeError(f"DSV4 case-boundary debug found unaligned free slots at {stage}")
            if torch.unique(self.free_slots).numel() != self.free_slots.numel():
                raise RuntimeError(f"DSV4 case-boundary debug found duplicate free slots at {stage}")
        self.check_integrity()
        kv_snapshot: dict[str, Any] = {}
        validator = getattr(self.kv_cache, "debug_validate_swa_lifecycle", None)
        if callable(validator):
            kv_snapshot = dict(validator(stage=stage))
        size_info = self.prefix_cache.size_info
        return {
            "stage": stage,
            "free_full_pages": int(self.free_slots.numel()),
            "retained_prefix_pages": int(size_info.total_size // self.page_size),
            "evictable_prefix_pages": int(size_info.evictable_size // self.page_size),
            "protected_prefix_pages": int(size_info.protected_size // self.page_size),
            "retained_prefix_swa_pages": int(
                (
                    int(getattr(self.prefix_cache, "dsv4_evictable_swa_tokens", 0))
                    + int(getattr(self.prefix_cache, "dsv4_protected_swa_tokens", 0))
                )
                // self.page_size
            ),
            "kv_cache": kv_snapshot,
            "graph_runner": dict(graph_runner or {}),
        }

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

    def _debug_sync_integrity(self, context: str, **extra: Any) -> None:
        if os.environ.get(DSV4_CACHE_SYNC_DEBUG_ENV, "").strip().lower() not in _TRUE_ENV_VALUES:
            return
        if self._lazy_free_depth > 0:
            return
        try:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.check_integrity()
        except Exception as exc:
            details = ", ".join(
                f"{key}={getattr(value, 'uid', value)}" for key, value in extra.items()
            )
            raise RuntimeError(
                f"DSV4 cache sync/integrity debug failed at {context}: {details}"
            ) from exc


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

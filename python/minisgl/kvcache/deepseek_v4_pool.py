from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Literal

import torch
from minisgl.dsv4_runtime import get_dsv4_runtime_config
from minisgl.utils import div_ceil

from .base import BaseKVCachePool

DSV4CacheLayout = Literal["bf16_flat", "flashmla_fp8_packed"]


def _indexer_fp8_cache_enabled() -> bool:
    return get_dsv4_runtime_config().optimized


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _align_up(value: int, alignment: int) -> int:
    return div_ceil(value, alignment) * alignment


def _clear_allocated_kv_modes() -> set[str]:
    scope = get_dsv4_runtime_config().clear_allocated_page_scope
    return {scope} if scope is not None else set()


@dataclass(frozen=True)
class DSV4CacheLayoutPolicy:
    """Storage policy for the first DSV4 KV-cache implementation.

    The v1 runtime stores everything as plain BF16 tensors.  The layout name is
    kept explicit so TARGET 05 can swap selected buffers to FlashMLA packed FP8
    without changing the scheduler-facing pool API.
    """

    storage_dtype: torch.dtype = torch.bfloat16
    compress_state_dtype: torch.dtype = torch.bfloat16
    layout: DSV4CacheLayout = "bf16_flat"
    indexer_layout: DSV4CacheLayout = "bf16_flat"


@dataclass(frozen=True)
class DSV4LayerCacheMapping:
    layer_id: int
    compress_ratio: Literal[0, 4, 128]
    normal_layer_id: int | None
    c4_layer_id: int | None
    c128_layer_id: int | None
    indexer_layer_id: int | None

    @property
    def uses_swa(self) -> bool:
        return self.compress_ratio == 0

    @property
    def uses_c4(self) -> bool:
        return self.compress_ratio == 4

    @property
    def uses_c128(self) -> bool:
        return self.compress_ratio == 128

    @property
    def uses_indexer(self) -> bool:
        return self.indexer_layer_id is not None

    @property
    def compressed_layer_id(self) -> int | None:
        if self.compress_ratio == 4:
            return self.c4_layer_id
        if self.compress_ratio == 128:
            return self.c128_layer_id
        return None


@dataclass(frozen=True)
class DSV4AllocationCounts:
    full_slots: int
    swa_slots: int
    swa_pages: int
    c4_slots: int
    c128_slots: int
    c4_indexer_slots: int
    c4_state_slots: int = 0
    c128_state_slots: int = 0
    c4_indexer_state_slots: int = 0

    @property
    def any_allocated(self) -> bool:
        return any(
            (
                self.full_slots,
                self.swa_slots,
                self.swa_pages,
                self.c4_slots,
                self.c128_slots,
                self.c4_indexer_slots,
                self.c4_state_slots,
                self.c128_state_slots,
                self.c4_indexer_state_slots,
            )
        )


@dataclass(frozen=True)
class DSV4SWAPageHandles:
    length: int
    page_size: int
    swa_pages: torch.Tensor | None = None

    @property
    def num_pages(self) -> int:
        return div_ceil(self.length, self.page_size) if self.length else 0

    @property
    def live_pages(self) -> int:
        if self.swa_pages is None:
            return 0
        return int(torch.count_nonzero(self.swa_pages >= 0).item())

    @property
    def tombstoned_pages(self) -> int:
        if self.swa_pages is None:
            return self.num_pages
        return int(torch.count_nonzero(self.swa_pages < 0).item())

    @property
    def has_live_tail(self) -> bool:
        return bool(
            self.swa_pages is not None
            and self.swa_pages.numel() > 0
            and int(self.swa_pages[-1].item()) >= 0
        )

    def slice_tokens(self, start: int, end: int) -> DSV4SWAPageHandles:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.length:
            raise ValueError(
                f"Invalid DSV4 SWA handle slice: start={start}, end={end}, length={self.length}"
            )
        if start % self.page_size != 0 or end % self.page_size != 0:
            raise ValueError(
                "DSV4 SWA handles can only be sliced on page boundaries: "
                f"start={start}, end={end}, page_size={self.page_size}"
            )
        page_start = start // self.page_size
        page_end = end // self.page_size
        pages = None if self.swa_pages is None else self.swa_pages[page_start:page_end].clone()
        return DSV4SWAPageHandles(
            length=end - start,
            page_size=self.page_size,
            swa_pages=pages,
        )

    def tombstone_tokens(
        self, start: int, end: int
    ) -> tuple[DSV4SWAPageHandles, DSV4SWAPageHandles]:
        released = self.slice_tokens(start, end)
        if self.swa_pages is None or released.num_pages == 0:
            return self, released
        pages = self.swa_pages.clone()
        page_start = start // self.page_size
        page_end = end // self.page_size
        pages[page_start:page_end] = -1
        return (
            DSV4SWAPageHandles(
                length=self.length,
                page_size=self.page_size,
                swa_pages=pages,
            ),
            released,
        )

    def tombstone_pages(self, pages_to_tombstone: torch.Tensor) -> tuple[DSV4SWAPageHandles, int]:
        if self.swa_pages is None or self.swa_pages.numel() == 0:
            return self, 0
        if pages_to_tombstone.numel() == 0:
            return self, 0
        pages_to_tombstone = pages_to_tombstone.to(
            device=self.swa_pages.device,
            dtype=self.swa_pages.dtype,
        )
        pages_to_tombstone = pages_to_tombstone[pages_to_tombstone >= 0]
        if pages_to_tombstone.numel() == 0:
            return self, 0
        live = self.swa_pages >= 0
        mask = live & torch.isin(self.swa_pages, torch.unique(pages_to_tombstone))
        count = int(torch.count_nonzero(mask).item())
        if count == 0:
            return self, 0
        pages = self.swa_pages.clone()
        pages[mask] = -1
        return (
            DSV4SWAPageHandles(
                length=self.length,
                page_size=self.page_size,
                swa_pages=pages,
            ),
            count,
        )

    @staticmethod
    def concat(handles: list[DSV4SWAPageHandles]) -> DSV4SWAPageHandles | None:
        handles = [h for h in handles if h.length > 0]
        if not handles:
            return None
        page_size = handles[0].page_size
        if any(h.page_size != page_size for h in handles):
            raise ValueError("Cannot concatenate DSV4 SWA handles with mixed page sizes")
        if any(h.swa_pages is None for h in handles):
            if all(h.swa_pages is None for h in handles):
                pages = None
            else:
                raise ValueError("Mixed DSV4 SWA handle field presence")
        else:
            pages = torch.cat([h.swa_pages for h in handles if h.swa_pages is not None])
        return DSV4SWAPageHandles(
            length=sum(h.length for h in handles),
            page_size=page_size,
            swa_pages=pages,
        )


@dataclass(frozen=True)
class DSV4ComponentPageHandles:
    length: int
    page_size: int
    c4_pages: torch.Tensor | None = None
    c128_pages: torch.Tensor | None = None
    c4_indexer_pages: torch.Tensor | None = None
    c4_state_pages: torch.Tensor | None = None
    c128_state_pages: torch.Tensor | None = None
    c4_indexer_state_pages: torch.Tensor | None = None

    @property
    def num_pages(self) -> int:
        return div_ceil(self.length, self.page_size) if self.length else 0

    @property
    def has_required_state_pages(self) -> bool:
        return (
            (self.c4_pages is None or self.c4_state_pages is not None)
            and (self.c128_pages is None or self.c128_state_pages is not None)
            and (self.c4_indexer_pages is None or self.c4_indexer_state_pages is not None)
        )

    def slice_tokens(self, start: int, end: int) -> DSV4ComponentPageHandles:
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > self.length:
            raise ValueError(
                "Invalid DSV4 component handle slice: "
                f"start={start}, end={end}, length={self.length}"
            )
        if start % self.page_size != 0 or end % self.page_size != 0:
            raise ValueError(
                "DSV4 component handles can only be sliced on page boundaries: "
                f"start={start}, end={end}, page_size={self.page_size}"
            )
        page_start = start // self.page_size
        page_end = end // self.page_size

        def _slice(x: torch.Tensor | None) -> torch.Tensor | None:
            return None if x is None else x[page_start:page_end].clone()

        return DSV4ComponentPageHandles(
            length=end - start,
            page_size=self.page_size,
            c4_pages=_slice(self.c4_pages),
            c128_pages=_slice(self.c128_pages),
            c4_indexer_pages=_slice(self.c4_indexer_pages),
            c4_state_pages=_slice(self.c4_state_pages),
            c128_state_pages=_slice(self.c128_state_pages),
            c4_indexer_state_pages=_slice(self.c4_indexer_state_pages),
        )

    @staticmethod
    def concat(handles: list[DSV4ComponentPageHandles]) -> DSV4ComponentPageHandles | None:
        handles = [h for h in handles if h.length > 0]
        if not handles:
            return None
        page_size = handles[0].page_size
        if any(h.page_size != page_size for h in handles):
            raise ValueError("Cannot concatenate DSV4 component handles with mixed page sizes")

        def _cat(attr: str) -> torch.Tensor | None:
            chunks = [getattr(h, attr) for h in handles]
            present = [x for x in chunks if x is not None]
            if not present:
                return None
            if len(present) != len(handles):
                raise ValueError(f"Mixed DSV4 component handle field presence for {attr}")
            return torch.cat(present)

        return DSV4ComponentPageHandles(
            length=sum(h.length for h in handles),
            page_size=page_size,
            c4_pages=_cat("c4_pages"),
            c128_pages=_cat("c128_pages"),
            c4_indexer_pages=_cat("c4_indexer_pages"),
            c4_state_pages=_cat("c4_state_pages"),
            c128_state_pages=_cat("c128_state_pages"),
            c4_indexer_state_pages=_cat("c4_indexer_state_pages"),
        )


class DSV4KVAndScore:
    def __init__(self, kv_score: torch.Tensor) -> None:
        self.kv_score = kv_score
        self._item_size = kv_score.shape[-1] // 2

    @property
    def kv(self) -> torch.Tensor:
        return self.kv_score[..., : self._item_size]

    @property
    def score(self) -> torch.Tensor:
        return self.kv_score[..., self._item_size :]

    def clear(self) -> None:
        self.kv.zero_()
        self.score.fill_(float("-inf"))

    def __getitem__(self, index) -> DSV4KVAndScore:
        return DSV4KVAndScore(self.kv_score[index])

    def __setitem__(self, index, value: DSV4KVAndScore) -> None:
        self.kv_score[index] = value.kv_score


class DSV4CompressStatePool:
    def __init__(
        self,
        *,
        size: int,
        ring_size: int,
        overlap: bool,
        head_dim: int,
        ratio: Literal[4, 128],
        dtype: torch.dtype,
        device: torch.device,
        page_size: int,
    ) -> None:
        self.ratio = ratio
        self.ring_size = ring_size
        self.overlap = overlap
        self.head_dim = head_dim
        self.page_size = page_size
        last_dim = 2 * (1 + int(overlap)) * head_dim
        padded_size = _align_up(size + ring_size + 1, _lcm(ratio, page_size))
        self.last_dim = last_dim
        self.logical_size = size
        self.kv_score_buffer = DSV4KVAndScore(
            torch.empty((padded_size, last_dim), dtype=dtype, device=device)
        )
        self.kv_score_buffer[-1].clear()

    def translate_from_swa_loc_to_state_loc(self, swa_loc: torch.Tensor) -> torch.Tensor:
        page_size = max(self.page_size, 1)
        swa_pages = swa_loc // page_size
        state_loc = swa_pages * self.ring_size + (swa_loc % self.ring_size)
        return torch.where(swa_loc < 0, -1, state_loc)

    def get_state_by_state_loc(self, state_loc: torch.Tensor) -> DSV4KVAndScore:
        return self.kv_score_buffer[state_loc]

    def set_state_by_state_loc(self, state_loc: torch.Tensor, value: DSV4KVAndScore) -> None:
        self.kv_score_buffer[state_loc] = value
        self.kv_score_buffer[-1].clear()

    def clear_state_locs(self, state_locs: torch.Tensor) -> None:
        if state_locs.numel() == 0:
            return
        buffer = self.kv_score_buffer.kv_score
        locs = torch.unique(state_locs.to(device=buffer.device, dtype=torch.long))
        locs = locs[(locs >= 0) & (locs < self.kv_score_buffer.kv_score.shape[0])]
        if locs.numel() == 0:
            return
        item_size = self.kv_score_buffer._item_size
        buffer.index_fill_(0, locs, 0)
        buffer[:, item_size:].index_fill_(0, locs, float("-inf"))
        self.kv_score_buffer[-1].clear()


class DeepSeekV4KVCache(BaseKVCachePool):
    """DSV4-specific KV pool with radix prefix reuse intentionally disabled.

    The scheduler owns a single full-token page table.  This pool derives all
    DSV4 component slots from that namespace and tracks page-level allocation
    with refcounts, so C4/C128/indexer/cache-state buffers are released when
    the owning full-token pages are released.
    """

    C4_STATE_RING_SIZE = 8
    C128_STATE_RING_SIZE = 128

    def __init__(
        self,
        *,
        model_config,
        num_pages: int,
        page_size: int,
        device: torch.device,
        policy: DSV4CacheLayoutPolicy | None = None,
        enable_component_loc_ownership: bool = False,
        enable_swa_independent_lifecycle: bool = False,
        max_running_req: int | None = None,
        swa_num_pages: int | None = None,
        dummy_token_start: int | None = None,
    ) -> None:
        self._policy = policy or DSV4CacheLayoutPolicy()
        self._device = device
        self._dtype = self._policy.storage_dtype
        self._num_layers = model_config.num_layers
        self._head_dim = model_config.head_dim
        self._index_head_dim = model_config.index_head_dim or model_config.head_dim
        self._num_pages = num_pages
        self._page_size = page_size
        self._num_tokens = num_pages * page_size
        self._dummy_token_start = (
            self._num_tokens if dummy_token_start is None else int(dummy_token_start)
        )
        if self._dummy_token_start < 0 or self._dummy_token_start > self._num_tokens:
            raise ValueError(
                "DSV4 dummy token start must be within the allocated full-token pool: "
                f"dummy_token_start={self._dummy_token_start}, num_tokens={self._num_tokens}"
            )
        if self._dummy_token_start % page_size != 0:
            raise ValueError(
                "DSV4 dummy token start must be page-aligned: "
                f"dummy_token_start={self._dummy_token_start}, page_size={page_size}"
            )
        self._window_size = int(getattr(model_config, "window_size", 128) or 128)
        self._component_loc_ownership_enabled = bool(enable_component_loc_ownership)
        self._swa_independent_lifecycle_enabled = bool(enable_swa_independent_lifecycle)
        if self._swa_independent_lifecycle_enabled and not self._component_loc_ownership_enabled:
            raise ValueError(
                "DSV4 SWA independent lifecycle requires Route B component loc ownership."
            )
        self._c4_slots = div_ceil(self._num_tokens, 4)
        self._c128_slots = div_ceil(self._num_tokens, 128)
        self._c4_component_page_size = max(div_ceil(page_size, 4), 1)
        self._c128_component_page_size = max(div_ceil(page_size, 128), 1)
        self._c4_state_page_size = self.C4_STATE_RING_SIZE
        self._c128_state_page_size = self.C128_STATE_RING_SIZE
        self._c4_component_pages = div_ceil(self._c4_slots, self._c4_component_page_size)
        self._c128_component_pages = div_ceil(
            self._c128_slots,
            self._c128_component_page_size,
        )

        self._layer_mapping = _build_layer_mapping(model_config.compress_ratios, self._num_layers)
        self._normal_layer_count = sum(m.compress_ratio == 0 for m in self._layer_mapping)
        self._c4_layer_count = sum(m.compress_ratio == 4 for m in self._layer_mapping)
        self._c128_layer_count = sum(m.compress_ratio == 128 for m in self._layer_mapping)

        self._swa_tail_pages_per_req = max(div_ceil(self._window_size, page_size), 1)
        if swa_num_pages is not None:
            planned_swa_pages = int(swa_num_pages)
        elif self._swa_independent_lifecycle_enabled:
            running_req = max(int(max_running_req or 1), 1)
            planned_swa_pages = running_req * (self._swa_tail_pages_per_req + 1) + 1
        else:
            planned_swa_pages = num_pages
        self._swa_num_pages = max(1, min(num_pages, planned_swa_pages))
        self._swa_num_tokens = self._swa_num_pages * page_size
        self._swa_dummy_page = self._swa_num_pages - 1

        shape = (self._num_layers, self._swa_num_pages, page_size, self._head_dim)
        self._swa_buffer = torch.empty(shape, dtype=self._dtype, device=device)
        self._c4_buffer = torch.empty(
            (self._c4_layer_count, self._c4_slots, self._head_dim),
            dtype=self._dtype,
            device=device,
        )
        self._c128_buffer = torch.empty(
            (self._c128_layer_count, self._c128_slots, self._head_dim),
            dtype=self._dtype,
            device=device,
        )
        self._c4_indexer_buffer = torch.empty(
            (self._c4_layer_count, self._c4_slots, self._index_head_dim),
            dtype=self._dtype,
            device=device,
        )
        self._use_indexer_fp8_cache = _indexer_fp8_cache_enabled()
        self._c4_indexer_fp8_page_size = max(page_size // 4, 1)
        self._c4_indexer_fp8_num_pages = div_ceil(self._c4_slots, self._c4_indexer_fp8_page_size)
        if self._use_indexer_fp8_cache and self._c4_layer_count:
            self._c4_indexer_fp8_paged_cache = torch.empty(
                (
                    self._c4_layer_count,
                    self._c4_indexer_fp8_num_pages,
                    self._c4_indexer_fp8_page_size * (self._index_head_dim + 4),
                ),
                dtype=torch.uint8,
                device=device,
            )
            if device.type == "cuda":
                self._c4_indexer_fp8_values = None
                self._c4_indexer_fp8_scales = None
            else:
                self._c4_indexer_fp8_values = torch.empty(
                    (self._c4_layer_count, self._c4_slots, self._index_head_dim),
                    dtype=torch.uint8,
                    device=device,
                )
                self._c4_indexer_fp8_scales = torch.empty(
                    (self._c4_layer_count, self._c4_slots, 4),
                    dtype=torch.uint8,
                    device=device,
                )
        else:
            self._c4_indexer_fp8_paged_cache = None
            self._c4_indexer_fp8_values = None
            self._c4_indexer_fp8_scales = None

        self._full_refcount = torch.zeros(self._num_tokens, dtype=torch.int16, device=device)
        self._swa_page_refcount = torch.zeros(
            self._swa_num_pages,
            dtype=torch.int16,
            device=device,
        )
        self._full_to_swa_page = torch.full(
            (self._num_pages,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        if self._swa_independent_lifecycle_enabled:
            self._swa_page_refcount[self._swa_dummy_page] = 1
            self._free_swa_pages = torch.arange(
                max(self._swa_num_pages - 1, 0),
                dtype=torch.int32,
                device=device,
            )
        else:
            self._swa_page_refcount.fill_(1)
            self._full_to_swa_page.copy_(
                torch.arange(self._num_pages, dtype=torch.int32, device=device)
            )
            self._free_swa_pages = torch.empty(0, dtype=torch.int32, device=device)
        self._swa_pages_allocated_total = 0
        self._swa_pages_freed_total = 0
        self._swa_pages_tombstoned_total = 0
        self._swa_ownership_version = 0
        self._c4_refcount = torch.zeros(self._c4_slots, dtype=torch.int16, device=device)
        self._c128_refcount = torch.zeros(self._c128_slots, dtype=torch.int16, device=device)
        self._c4_indexer_refcount = torch.zeros(self._c4_slots, dtype=torch.int16, device=device)
        self._c4_state_refcount = torch.zeros(
            self._num_pages * self._c4_state_page_size,
            dtype=torch.int16,
            device=device,
        )
        self._c128_state_refcount = torch.zeros(
            self._num_pages * self._c128_state_page_size,
            dtype=torch.int16,
            device=device,
        )
        self._c4_indexer_state_refcount = torch.zeros_like(self._c4_state_refcount)
        self._full_to_c4_page = torch.full(
            (self._num_pages,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._full_to_c128_page = torch.full_like(self._full_to_c4_page, -1)
        self._full_to_c4_indexer_page = torch.full_like(self._full_to_c4_page, -1)
        self._full_to_c4_state_page = torch.full_like(self._full_to_c4_page, -1)
        self._full_to_c128_state_page = torch.full_like(self._full_to_c4_page, -1)
        self._full_to_c4_indexer_state_page = torch.full_like(self._full_to_c4_page, -1)
        self._free_c4_pages = torch.arange(
            self._c4_component_pages,
            dtype=torch.int32,
            device=device,
        )
        self._free_c128_pages = torch.arange(
            self._c128_component_pages,
            dtype=torch.int32,
            device=device,
        )
        self._free_c4_indexer_pages = torch.arange(
            self._c4_component_pages,
            dtype=torch.int32,
            device=device,
        )
        self._free_c4_state_pages = torch.arange(
            self._num_pages,
            dtype=torch.int32,
            device=device,
        )
        self._free_c128_state_pages = torch.arange(
            self._num_pages,
            dtype=torch.int32,
            device=device,
        )
        self._free_c4_indexer_state_pages = torch.arange(
            self._num_pages,
            dtype=torch.int32,
            device=device,
        )

        self._compress_state_pools: list[DSV4CompressStatePool | None] = [None] * self._num_layers
        self._indexer_compress_state_pools: list[DSV4CompressStatePool | None] = [
            None
        ] * self._num_layers
        for mapping in self._layer_mapping:
            if mapping.compress_ratio == 4:
                self._compress_state_pools[mapping.layer_id] = DSV4CompressStatePool(
                    size=self._num_pages * self.C4_STATE_RING_SIZE,
                    ring_size=self.C4_STATE_RING_SIZE,
                    overlap=True,
                    head_dim=self._head_dim,
                    ratio=4,
                    dtype=self._policy.compress_state_dtype,
                    device=device,
                    page_size=page_size,
                )
                self._indexer_compress_state_pools[mapping.layer_id] = DSV4CompressStatePool(
                    size=self._num_pages * self.C4_STATE_RING_SIZE,
                    ring_size=self.C4_STATE_RING_SIZE,
                    overlap=True,
                    head_dim=self._index_head_dim,
                    ratio=4,
                    dtype=self._policy.compress_state_dtype,
                    device=device,
                    page_size=page_size,
                )
            elif mapping.compress_ratio == 128:
                self._compress_state_pools[mapping.layer_id] = DSV4CompressStatePool(
                    size=self._num_pages * self.C128_STATE_RING_SIZE,
                    ring_size=self.C128_STATE_RING_SIZE,
                    overlap=False,
                    head_dim=self._head_dim,
                    ratio=128,
                    dtype=self._policy.compress_state_dtype,
                    device=device,
                    page_size=page_size,
                )

    @property
    def indexer_fp8_page_size(self) -> int:
        return self._c4_indexer_fp8_page_size

    @property
    def indexer_fp8_num_pages(self) -> int:
        return self._c4_indexer_fp8_num_pages

    @property
    def policy(self) -> DSV4CacheLayoutPolicy:
        return self._policy

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def component_loc_ownership_enabled(self) -> bool:
        return self._component_loc_ownership_enabled

    @property
    def swa_independent_lifecycle_enabled(self) -> bool:
        return self._swa_independent_lifecycle_enabled

    @property
    def swa_ownership_version(self) -> int:
        return int(self._swa_ownership_version)

    @property
    def c4_component_page_size(self) -> int:
        return self._c4_component_page_size

    @property
    def c128_component_page_size(self) -> int:
        return self._c128_component_page_size

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def dummy_token_start(self) -> int:
        return self._dummy_token_start

    @property
    def layer_mapping(self) -> tuple[DSV4LayerCacheMapping, ...]:
        return tuple(self._layer_mapping)

    @property
    def allocation_counts(self) -> DSV4AllocationCounts:
        if self._swa_independent_lifecycle_enabled:
            tail_refcount = self._swa_page_refcount[: self._swa_dummy_page]
            swa_pages = int(torch.count_nonzero(tail_refcount > 0).item())
        else:
            full_page_refcount = self._full_refcount.view(self._num_pages, self._page_size)
            swa_pages = int(torch.count_nonzero(full_page_refcount.sum(dim=1) > 0).item())
        return DSV4AllocationCounts(
            full_slots=int(torch.count_nonzero(self._full_refcount).item()),
            swa_slots=swa_pages * self._page_size,
            swa_pages=swa_pages,
            c4_slots=(
                int(torch.count_nonzero(self._c4_refcount).item()) if self._c4_layer_count else 0
            ),
            c128_slots=(
                int(torch.count_nonzero(self._c128_refcount).item())
                if self._c128_layer_count
                else 0
            ),
            c4_indexer_slots=(
                int(torch.count_nonzero(self._c4_indexer_refcount).item())
                if self._c4_layer_count
                else 0
            ),
            c4_state_slots=(
                int(torch.count_nonzero(self._c4_state_refcount).item())
                if self._c4_layer_count
                else 0
            ),
            c128_state_slots=(
                int(torch.count_nonzero(self._c128_state_refcount).item())
                if self._c128_layer_count
                else 0
            ),
            c4_indexer_state_slots=(
                int(torch.count_nonzero(self._c4_indexer_state_refcount).item())
                if self._c4_layer_count
                else 0
            ),
        )

    def get_layer_mapping(self, layer_id: int) -> DSV4LayerCacheMapping:
        return self._layer_mapping[layer_id]

    def swa_cache(self, layer_id: int) -> torch.Tensor:
        return self._swa_buffer[layer_id].view(self._swa_num_tokens, self._head_dim)

    def c4_cache(self, layer_id: int) -> torch.Tensor:
        mapping = self.get_layer_mapping(layer_id)
        assert mapping.c4_layer_id is not None, f"Layer {layer_id} is not a C4 layer."
        return self._c4_buffer[mapping.c4_layer_id]

    def c128_cache(self, layer_id: int) -> torch.Tensor:
        mapping = self.get_layer_mapping(layer_id)
        assert mapping.c128_layer_id is not None, f"Layer {layer_id} is not a C128 layer."
        return self._c128_buffer[mapping.c128_layer_id]

    def indexer_cache(self, layer_id: int) -> torch.Tensor:
        mapping = self.get_layer_mapping(layer_id)
        assert mapping.indexer_layer_id is not None, f"Layer {layer_id} has no C4 indexer."
        return self._c4_indexer_buffer[mapping.indexer_layer_id]

    def has_indexer_fp8_cache(self) -> bool:
        return self._c4_indexer_fp8_paged_cache is not None

    def has_indexer_fp8_paged_cache(self) -> bool:
        return self._c4_indexer_fp8_paged_cache is not None

    def indexer_fp8_paged_cache(self, layer_id: int) -> torch.Tensor:
        mapping = self.get_layer_mapping(layer_id)
        assert mapping.indexer_layer_id is not None, f"Layer {layer_id} has no C4 indexer."
        if self._c4_indexer_fp8_paged_cache is None:
            raise RuntimeError(
                f"DSV4 paged FP8 indexer cache was requested for layer {layer_id}, "
                "but the paged FP8 indexer cache was not enabled at cache allocation."
            )
        return self._c4_indexer_fp8_paged_cache[mapping.indexer_layer_id]

    def indexer_fp8_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        mapping = self.get_layer_mapping(layer_id)
        assert mapping.indexer_layer_id is not None, f"Layer {layer_id} has no C4 indexer."
        if self._c4_indexer_fp8_values is None or self._c4_indexer_fp8_scales is None:
            raise RuntimeError(
                f"DSV4 legacy two-tensor FP8 indexer cache was requested for layer {layer_id}, "
                "but this allocation uses the paged vLLM-style FP8 indexer cache layout."
            )
        return (
            self._c4_indexer_fp8_values[mapping.indexer_layer_id],
            self._c4_indexer_fp8_scales[mapping.indexer_layer_id],
        )

    def attention_compress_state(self, layer_id: int) -> DSV4CompressStatePool:
        pool = self._compress_state_pools[layer_id]
        assert pool is not None, f"Layer {layer_id} has no attention compress state."
        return pool

    def indexer_compress_state(self, layer_id: int) -> DSV4CompressStatePool:
        pool = self._indexer_compress_state_pools[layer_id]
        assert pool is not None, f"Layer {layer_id} has no indexer compress state."
        return pool

    def component_cache(self, layer_id: int) -> torch.Tensor:
        mapping = self.get_layer_mapping(layer_id)
        if mapping.compress_ratio == 4:
            return self.c4_cache(layer_id)
        if mapping.compress_ratio == 128:
            return self.c128_cache(layer_id)
        return self.swa_cache(layer_id)

    def compressed_locs_from_full_locs(
        self,
        full_locs: torch.Tensor,
        ratio: Literal[4, 128],
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        full_locs = full_locs.to(device=self.device, dtype=torch.long)
        if positions is not None:
            positions = positions.to(device=self.device, dtype=torch.long)
            full_locs = full_locs[(positions + 1) % ratio == 0]
        if full_locs.numel() == 0:
            return full_locs
        if not self._component_loc_ownership_enabled:
            return torch.unique_consecutive(full_locs // ratio)
        return torch.unique_consecutive(
            self._component_locs_from_full_locs(full_locs, ratio, component="compressed")
        )

    def indexer_locs_from_full_locs(
        self,
        full_locs: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._component_loc_ownership_enabled:
            return self.compressed_locs_from_full_locs(full_locs, 4, positions)
        full_locs = full_locs.to(device=self.device, dtype=torch.long)
        if positions is not None:
            positions = positions.to(device=self.device, dtype=torch.long)
            full_locs = full_locs[(positions + 1) % 4 == 0]
        if full_locs.numel() == 0:
            return full_locs
        return torch.unique_consecutive(
            self._component_locs_from_full_locs(full_locs, 4, component="indexer")
        )

    def store_swa(self, layer_id: int, kv: torch.Tensor, out_loc: torch.Tensor) -> None:
        if kv.numel() == 0:
            return
        swa_loc = self.translate_full_locs_to_swa_locs(out_loc)
        valid = swa_loc >= 0
        if not bool(torch.all(valid)):
            raise RuntimeError("DSV4 SWA write requested for full loc without live SWA mapping")
        self.swa_cache(layer_id)[swa_loc.long()] = kv.reshape(-1, self._head_dim).to(self._dtype)

    def store_compressed(
        self,
        layer_id: int,
        kv: torch.Tensor,
        loc: torch.Tensor,
    ) -> None:
        if kv.numel() == 0:
            return
        cache = self.component_cache(layer_id)
        cache[loc.long()] = kv.reshape(-1, self._head_dim).to(self._dtype)

    def store_indexer(self, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
        if kv.numel() == 0:
            return
        self.indexer_cache(layer_id)[loc.long()] = kv.reshape(-1, self._index_head_dim).to(
            self._dtype
        )

    def _clear_full_locs(self, locs: torch.Tensor) -> None:
        locs = self.translate_full_locs_to_swa_locs(locs)
        locs = self._sanitize_locs(locs, self._swa_num_tokens)
        if locs.numel() == 0:
            return
        self._swa_buffer.view(self._num_layers, self._swa_num_tokens, self._head_dim).index_fill_(
            1,
            locs,
            0,
        )

    def _clear_c4_component_locs(self, locs: torch.Tensor) -> None:
        locs = self._sanitize_locs(locs, self._c4_slots)
        if locs.numel() == 0:
            return
        self._c4_buffer.index_fill_(1, locs, 0)

    def _clear_c4_indexer_component_locs(
        self,
        locs: torch.Tensor,
        pages: torch.Tensor | None = None,
    ) -> None:
        locs = self._sanitize_locs(locs, self._c4_slots)
        if locs.numel() == 0:
            return
        self._c4_indexer_buffer.index_fill_(1, locs, 0)
        if self._c4_indexer_fp8_paged_cache is not None:
            if pages is None:
                pages = torch.unique(locs // max(self._c4_indexer_fp8_page_size, 1))
            pages = self._sanitize_locs(pages, self._c4_indexer_fp8_num_pages)
            if pages.numel() > 0:
                self._c4_indexer_fp8_paged_cache.index_fill_(1, pages, 0)
        if self._c4_indexer_fp8_values is not None:
            self._c4_indexer_fp8_values.index_fill_(1, locs, 0)
        if self._c4_indexer_fp8_scales is not None:
            self._c4_indexer_fp8_scales.index_fill_(1, locs, 0)

    def _clear_c128_component_locs(self, locs: torch.Tensor) -> None:
        locs = self._sanitize_locs(locs, self._c128_slots)
        if locs.numel() == 0:
            return
        self._c128_buffer.index_fill_(1, locs, 0)

    def _clear_c4_state_locs(self, locs: torch.Tensor) -> None:
        locs = self._sanitize_locs(locs, self._num_pages * self._c4_state_page_size)
        if locs.numel() == 0:
            return
        for pool in self._compress_state_pools:
            if pool is not None and pool.ratio == 4:
                pool.clear_state_locs(locs)

    def _clear_c4_indexer_state_locs(self, locs: torch.Tensor) -> None:
        locs = self._sanitize_locs(locs, self._num_pages * self._c4_state_page_size)
        if locs.numel() == 0:
            return
        for pool in self._indexer_compress_state_pools:
            if pool is not None and pool.ratio == 4:
                pool.clear_state_locs(locs)

    def _clear_c128_state_locs(self, locs: torch.Tensor) -> None:
        locs = self._sanitize_locs(locs, self._num_pages * self._c128_state_page_size)
        if locs.numel() == 0:
            return
        for pool in self._compress_state_pools:
            if pool is not None and pool.ratio == 128:
                pool.clear_state_locs(locs)

    def _sanitize_locs(self, locs: torch.Tensor, upper_bound: int) -> torch.Tensor:
        if locs.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        locs = torch.unique(locs.to(device=self.device, dtype=torch.long))
        return locs[(locs >= 0) & (locs < int(upper_bound))]

    def on_pages_allocated(self, page_starts: torch.Tensor, page_size: int) -> None:
        full_locs = self._expand_page_starts(page_starts, page_size)
        if full_locs.numel() == 0:
            return
        clear_modes = _clear_allocated_kv_modes()
        self._full_refcount[full_locs] += 1
        if self._swa_independent_lifecycle_enabled:
            self._allocate_swa_pages_for_full_pages(page_starts, page_size)
        if "full" in clear_modes:
            self._clear_full_locs(full_locs)
        if self._component_loc_ownership_enabled:
            self._allocate_component_pages_for_full_pages(
                page_starts,
                page_size,
                clear_modes=clear_modes,
            )
            return
        if self._c4_layer_count:
            c4_locs = torch.unique(full_locs // 4)
            self._c4_refcount[c4_locs] += 1
            self._c4_indexer_refcount[c4_locs] += 1
            if "component" in clear_modes:
                self._clear_c4_component_locs(c4_locs)
                self._clear_c4_indexer_component_locs(c4_locs)
            if "state" in clear_modes:
                state_locs = self.state_locs_from_full_locs(
                    full_locs,
                    4,
                    component="attention",
                )
                indexer_state_locs = self.state_locs_from_full_locs(
                    full_locs,
                    4,
                    component="indexer",
                )
                self._clear_c4_state_locs(state_locs)
                self._clear_c4_indexer_state_locs(indexer_state_locs)
        if self._c128_layer_count:
            c128_locs = torch.unique(full_locs // 128)
            self._c128_refcount[c128_locs] += 1
            if "component" in clear_modes:
                self._clear_c128_component_locs(c128_locs)
            if "state" in clear_modes:
                state_locs = self.state_locs_from_full_locs(
                    full_locs,
                    128,
                    component="attention",
                )
                self._clear_c128_state_locs(state_locs)

    def on_token_indices_freed(
        self,
        indices: torch.Tensor,
        page_size: int,
        *,
        free_components: bool = True,
        free_swa: bool = True,
    ) -> None:
        if indices.numel() == 0:
            return
        {
            "tokens": int(indices.numel()),
            "page_size": int(page_size),
            "free_components": bool(free_components),
            "free_swa": bool(free_swa),
            "component_loc_ownership": bool(self._component_loc_ownership_enabled),
            "swa_independent_lifecycle": bool(self._swa_independent_lifecycle_enabled),
        }
        page_starts = self._valid_page_starts(indices, page_size)
        if page_starts.numel() == 0:
            return
        full_locs = self._expand_page_starts(page_starts, page_size)
        if full_locs.numel() == 0:
            return
        self._decrement_refcount(self._full_refcount, full_locs, "full token")
        if self._swa_independent_lifecycle_enabled:
            self._release_swa_pages_for_full_pages(
                page_starts,
                page_size,
                free_swa=free_swa,
            )
        if self._component_loc_ownership_enabled:
            self._release_component_pages_for_full_pages(
                page_starts,
                page_size,
                free_components=free_components,
            )
            return
        if self._c4_layer_count:
            c4_locs = torch.unique(full_locs // 4)
            self._decrement_refcount(self._c4_refcount, c4_locs, "C4")
            self._decrement_refcount(self._c4_indexer_refcount, c4_locs, "C4 indexer")
        if self._c128_layer_count:
            self._decrement_refcount(self._c128_refcount, torch.unique(full_locs // 128), "C128")

    def check_allocation_integrity(self, allocated_pages: int, page_size: int) -> None:
        expected_full_slots = allocated_pages * page_size
        actual_full_slots = self.allocation_counts.full_slots
        if actual_full_slots != expected_full_slots:
            raise RuntimeError(
                "DSV4 KV cache allocation mismatch:"
                f" full_slots={actual_full_slots}, expected={expected_full_slots}"
            )
        self._assert_no_negative_refcounts()

    def assert_no_leak(self) -> None:
        counts = self.allocation_counts
        if counts.any_allocated:
            raise RuntimeError(f"DSV4 KV cache slot leak: {counts}")

    def estimate_prefix_retention(
        self,
        retained_full_tokens: int,
        page_size: int | None = None,
    ) -> dict[str, int | bool]:
        """Estimate DSV4 component residency for retained prefix pages.

        The scheduler/radix cache owns full-token pages. DSV4 compressed and
        indexer slots are derived from those full-token pages, so metrics can
        report component residency without introducing a second owner path.
        """

        page_size = self._page_size if page_size is None else int(page_size)
        retained_full_tokens = int(retained_full_tokens)
        retained_pages = div_ceil(retained_full_tokens, page_size) if retained_full_tokens else 0
        c4_slots = retained_pages * div_ceil(page_size, 4) if self._c4_layer_count else 0
        c128_slots = retained_pages * div_ceil(page_size, 128) if self._c128_layer_count else 0
        c4_indexer_slots = c4_slots if self._c4_layer_count else 0
        c4_state_slots = retained_pages * self.C4_STATE_RING_SIZE if self._c4_layer_count else 0
        c128_state_slots = (
            retained_pages * self.C128_STATE_RING_SIZE if self._c128_layer_count else 0
        )
        c4_indexer_state_slots = c4_state_slots if self._c4_layer_count else 0

        dtype_size = self._dtype.itemsize
        state_dtype_size = self._policy.compress_state_dtype.itemsize
        legacy_swa_bytes = self._num_layers * retained_full_tokens * self._head_dim * dtype_size
        if self._swa_independent_lifecycle_enabled:
            runtime_swa_pages = self.runtime_swa_counters()["current_swa_tail_pages"]
            swa_tokens = int(runtime_swa_pages) * page_size
            swa_bytes = self._num_layers * swa_tokens * self._head_dim * dtype_size
        else:
            swa_tokens = retained_full_tokens
            swa_bytes = legacy_swa_bytes
        c4_bytes = self._c4_layer_count * c4_slots * self._head_dim * dtype_size
        c128_bytes = self._c128_layer_count * c128_slots * self._head_dim * dtype_size
        c4_indexer_bytes = (
            self._c4_layer_count * c4_indexer_slots * self._index_head_dim * dtype_size
        )
        c4_indexer_fp8_bytes = (
            self._c4_layer_count * c4_indexer_slots * (self._index_head_dim + 4)
            if self._use_indexer_fp8_cache
            else 0
        )
        c4_state_bytes = (
            self._c4_layer_count
            * retained_pages
            * self.C4_STATE_RING_SIZE
            * 4
            * self._head_dim
            * state_dtype_size
        )
        c4_indexer_state_bytes = (
            self._c4_layer_count
            * retained_pages
            * self.C4_STATE_RING_SIZE
            * 4
            * self._index_head_dim
            * state_dtype_size
        )
        c128_state_bytes = (
            self._c128_layer_count
            * retained_pages
            * self.C128_STATE_RING_SIZE
            * 2
            * self._head_dim
            * state_dtype_size
        )
        retained_memory_bytes = (
            swa_bytes
            + c4_bytes
            + c128_bytes
            + c4_indexer_bytes
            + c4_indexer_fp8_bytes
            + c4_state_bytes
            + c4_indexer_state_bytes
            + c128_state_bytes
        )
        return {
            "retained_pages": retained_pages,
            "full_slots": retained_full_tokens,
            "c4_slots": c4_slots,
            "c128_slots": c128_slots,
            "c4_indexer_slots": c4_indexer_slots,
            "c4_state_slots": c4_state_slots,
            "c128_state_slots": c128_state_slots,
            "c4_indexer_state_slots": c4_indexer_state_slots,
            "swa_independent_lifecycle": bool(self._swa_independent_lifecycle_enabled),
            "swa_tail_tokens": swa_tokens,
            "swa_bytes": swa_bytes,
            "legacy_swa_bytes": legacy_swa_bytes,
            "c4_bytes": c4_bytes,
            "c128_bytes": c128_bytes,
            "c4_indexer_bytes": c4_indexer_bytes,
            "c4_indexer_fp8_bytes": c4_indexer_fp8_bytes,
            "c4_state_bytes": c4_state_bytes,
            "c4_indexer_state_bytes": c4_indexer_state_bytes,
            "c128_state_bytes": c128_state_bytes,
            "retained_memory_bytes": retained_memory_bytes,
            "page_size_c128_aligned": page_size % 128 == 0,
        }

    def runtime_swa_counters(self) -> dict[str, int | bool]:
        current_pages = (
            int(torch.count_nonzero(self._swa_page_refcount > 0).item())
            if self._swa_independent_lifecycle_enabled
            else int(
                torch.count_nonzero(
                    self._full_refcount.view(self._num_pages, self._page_size).sum(dim=1) > 0
                ).item()
            )
        )
        if (
            self._swa_independent_lifecycle_enabled
            and self._swa_page_refcount[self._swa_dummy_page] > 0
        ):
            current_tail_pages = max(current_pages - 1, 0)
        else:
            current_tail_pages = current_pages
        return {
            "enabled": bool(self._swa_independent_lifecycle_enabled),
            "swa_capacity_pages": int(self._swa_num_pages),
            "swa_tail_capacity_pages": int(
                max(self._swa_num_pages - (1 if self._swa_independent_lifecycle_enabled else 0), 0)
            ),
            "current_swa_pages": int(current_pages),
            "current_swa_tail_pages": int(current_tail_pages),
            "available_swa_pages": int(self.available_swa_pages()),
            "swa_pages_allocated_total": int(self._swa_pages_allocated_total),
            "swa_pages_freed_total": int(self._swa_pages_freed_total),
            "swa_pages_tombstoned_total": int(self._swa_pages_tombstoned_total),
            "swa_ownership_version": int(self._swa_ownership_version),
            "swa_tail_pages_per_req": int(self._swa_tail_pages_per_req),
            "sliding_window": int(self._window_size),
            "page_size": int(self._page_size),
            "dummy_token_start": int(self._dummy_token_start),
        }

    def debug_validate_swa_lifecycle(self, *, stage: str = "") -> dict[str, int | bool | str]:
        if not self._swa_independent_lifecycle_enabled:
            return {"enabled": False, "stage": stage}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        refcount = self._swa_page_refcount
        free_pages = self._free_swa_pages.to(device=self.device, dtype=torch.long)
        if torch.any(refcount < 0):
            raise RuntimeError(f"DSV4 SWA lifecycle debug found negative refcounts at {stage}")
        if int(refcount[self._swa_dummy_page].item()) <= 0:
            raise RuntimeError(f"DSV4 SWA lifecycle debug found unpinned dummy page at {stage}")
        if free_pages.numel() > 0:
            if torch.any(free_pages < 0) or torch.any(free_pages >= self._swa_dummy_page):
                raise RuntimeError(
                    f"DSV4 SWA lifecycle debug found out-of-range free page at {stage}"
                )
            if torch.unique(free_pages).numel() != free_pages.numel():
                raise RuntimeError(
                    f"DSV4 SWA lifecycle debug found duplicate free pages at {stage}"
                )
            if torch.any(refcount[free_pages] != 0):
                raise RuntimeError(
                    f"DSV4 SWA lifecycle debug found live refcount on free page at {stage}"
                )
        mapped = self._full_to_swa_page[self._full_to_swa_page >= 0].to(torch.long)
        if mapped.numel() > 0:
            if torch.any(mapped >= self._swa_dummy_page):
                raise RuntimeError(
                    f"DSV4 SWA lifecycle debug found dummy/out-of-range mapping at {stage}"
                )
            if torch.any(refcount[mapped] <= 0):
                raise RuntimeError(
                    f"DSV4 SWA lifecycle debug found zero-refcount mapping at {stage}"
                )
            if free_pages.numel() > 0 and torch.any(torch.isin(mapped, free_pages)):
                raise RuntimeError(
                    f"DSV4 SWA lifecycle debug found mapping to free page at {stage}"
                )
        return {
            **self.runtime_swa_counters(),
            "stage": stage,
            "mapped_full_pages": int(mapped.numel()),
            "free_swa_pages": int(free_pages.numel()),
        }

    def make_component_page_handles(
        self,
        full_indices: torch.Tensor,
        page_size: int,
    ) -> DSV4ComponentPageHandles | None:
        if not self._component_loc_ownership_enabled:
            return None
        if full_indices.numel() == 0:
            return DSV4ComponentPageHandles(length=0, page_size=page_size)
        if full_indices.numel() % page_size != 0:
            raise ValueError(
                "DSV4 component handles require page-aligned full indices, "
                f"got {full_indices.numel()} tokens for page_size={page_size}"
            )
        page_starts = full_indices[::page_size].to(device=self.device, dtype=torch.long)
        full_pages = torch.where(
            page_starts >= 0,
            page_starts.div(page_size, rounding_mode="floor"),
            torch.full_like(page_starts, -1),
        )

        def _gather(mapping: torch.Tensor, enabled: bool, name: str) -> torch.Tensor | None:
            if not enabled:
                return None
            out = torch.full_like(full_pages, -1, dtype=torch.int32)
            valid = full_pages >= 0
            if bool(torch.any(valid)):
                gathered = mapping[full_pages[valid]]
                if torch.any(gathered < 0):
                    raise RuntimeError(
                        f"DSV4 component mapping is missing for active {name} full pages"
                    )
                out[valid] = gathered.to(torch.int32)
            return out

        return DSV4ComponentPageHandles(
            length=int(full_indices.numel()),
            page_size=page_size,
            c4_pages=_gather(self._full_to_c4_page, self._c4_layer_count > 0, "C4"),
            c128_pages=_gather(self._full_to_c128_page, self._c128_layer_count > 0, "C128"),
            c4_indexer_pages=_gather(
                self._full_to_c4_indexer_page,
                self._c4_layer_count > 0,
                "C4 indexer",
            ),
            c4_state_pages=_gather(
                self._full_to_c4_state_page,
                self._c4_layer_count > 0,
                "C4 state",
            ),
            c128_state_pages=_gather(
                self._full_to_c128_state_page,
                self._c128_layer_count > 0,
                "C128 state",
            ),
            c4_indexer_state_pages=_gather(
                self._full_to_c4_indexer_state_page,
                self._c4_layer_count > 0,
                "C4 indexer state",
            ),
        )

    def component_pages_from_full_page_starts(
        self,
        page_starts: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        full_pages = torch.where(
            page_starts >= 0,
            page_starts.div(page_size, rounding_mode="floor"),
            torch.full_like(page_starts, -1),
        )

        def _gather(mapping: torch.Tensor, enabled: bool) -> torch.Tensor | None:
            if not enabled:
                return None
            out = torch.full_like(full_pages, -1, dtype=torch.int32)
            valid = (full_pages >= 0) & (full_pages < self._num_pages)
            if bool(torch.any(valid)):
                out[valid] = mapping[full_pages[valid]].to(torch.int32)
            return out

        if not self._component_loc_ownership_enabled:
            full_page_i32 = full_pages.to(torch.int32)
            return (
                full_page_i32 if self._c4_layer_count else None,
                full_page_i32 if self._c128_layer_count else None,
                full_page_i32 if self._c4_layer_count else None,
            )
        return (
            _gather(self._full_to_c4_page, self._c4_layer_count > 0),
            _gather(self._full_to_c128_page, self._c128_layer_count > 0),
            _gather(self._full_to_c4_indexer_page, self._c4_layer_count > 0),
        )

    def swa_pages_from_full_page_starts(
        self,
        page_starts: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor | None:
        if not self._swa_independent_lifecycle_enabled:
            return None
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        full_pages = torch.where(
            page_starts >= 0,
            page_starts.div(page_size, rounding_mode="floor"),
            torch.full_like(page_starts, -1),
        )
        out = torch.full_like(full_pages, -1, dtype=torch.int32)
        valid = (full_pages >= 0) & (full_pages < self._num_pages)
        if bool(torch.any(valid)):
            out[valid] = self._full_to_swa_page[full_pages[valid]].to(torch.int32)
        dummy = page_starts == self._dummy_token_start
        if bool(torch.any(dummy)):
            out[dummy] = int(self._swa_dummy_page)
        return out

    def make_swa_page_handles(
        self,
        full_indices: torch.Tensor,
        page_size: int,
    ) -> DSV4SWAPageHandles | None:
        if not self._swa_independent_lifecycle_enabled:
            return None
        if full_indices.numel() == 0:
            return DSV4SWAPageHandles(length=0, page_size=page_size)
        if full_indices.numel() % page_size != 0:
            raise ValueError(
                "DSV4 SWA handles require page-aligned full indices, "
                f"got {full_indices.numel()} tokens for page_size={page_size}"
            )
        page_starts = full_indices[::page_size].to(device=self.device, dtype=torch.long)
        swa_pages = self.swa_pages_from_full_page_starts(page_starts, page_size)
        assert swa_pages is not None
        return DSV4SWAPageHandles(
            length=int(full_indices.numel()),
            page_size=page_size,
            swa_pages=swa_pages,
        )

    def state_locs_from_full_locs(
        self,
        full_locs: torch.Tensor,
        ratio: Literal[4, 128],
        *,
        component: Literal["attention", "indexer"] = "attention",
    ) -> torch.Tensor:
        if ratio == 4:
            mapping = (
                self._full_to_c4_indexer_state_page
                if component == "indexer"
                else self._full_to_c4_state_page
            )
            state_page_size = self._c4_state_page_size
        else:
            if component == "indexer":
                raise ValueError("DSV4 C128 has no indexer compression state")
            mapping = self._full_to_c128_state_page
            state_page_size = self._c128_state_page_size

        full_locs = full_locs.to(device=self.device, dtype=torch.long)
        if full_locs.numel() == 0:
            return full_locs
        if not self._component_loc_ownership_enabled:
            page_size = max(self._page_size, 1)
            pages = full_locs.div(page_size, rounding_mode="floor")
            state_locs = pages * state_page_size + (full_locs % state_page_size)
            return torch.where(full_locs < 0, torch.full_like(state_locs, -1), state_locs)

        full_pages = full_locs.div(self._page_size, rounding_mode="floor")
        offsets = full_locs % state_page_size
        valid = (full_locs >= 0) & (full_pages >= 0) & (full_pages < self._num_pages)
        out = torch.full_like(full_locs, -1)
        if bool(torch.any(valid)):
            state_pages = mapping[full_pages[valid]]
            if torch.any(state_pages < 0):
                raise RuntimeError(
                    "DSV4 state loc requested for full locs without active state mapping"
                )
            out[valid] = state_pages.to(torch.long) * state_page_size + offsets[valid]
        return out

    def release_component_page_handles(self, handles: DSV4ComponentPageHandles | None) -> None:
        if handles is None or handles.length == 0:
            return
        if not self._component_loc_ownership_enabled:
            return
        if handles.c4_pages is not None:
            self._free_component_pages(
                handles.c4_pages,
                refcount=self._c4_refcount,
                page_size=self._c4_component_page_size,
                free_attr="_free_c4_pages",
                name="C4",
            )
        if handles.c128_pages is not None:
            self._free_component_pages(
                handles.c128_pages,
                refcount=self._c128_refcount,
                page_size=self._c128_component_page_size,
                free_attr="_free_c128_pages",
                name="C128",
            )
        if handles.c4_indexer_pages is not None:
            self._free_component_pages(
                handles.c4_indexer_pages,
                refcount=self._c4_indexer_refcount,
                page_size=self._c4_component_page_size,
                free_attr="_free_c4_indexer_pages",
                name="C4 indexer",
            )
        if handles.c4_state_pages is not None:
            self._free_component_pages(
                handles.c4_state_pages,
                refcount=self._c4_state_refcount,
                page_size=self._c4_state_page_size,
                free_attr="_free_c4_state_pages",
                name="C4 state",
            )
        if handles.c128_state_pages is not None:
            self._free_component_pages(
                handles.c128_state_pages,
                refcount=self._c128_state_refcount,
                page_size=self._c128_state_page_size,
                free_attr="_free_c128_state_pages",
                name="C128 state",
            )
        if handles.c4_indexer_state_pages is not None:
            self._free_component_pages(
                handles.c4_indexer_state_pages,
                refcount=self._c4_indexer_state_refcount,
                page_size=self._c4_state_page_size,
                free_attr="_free_c4_indexer_state_pages",
                name="C4 indexer state",
            )

    def release_swa_page_handles(
        self,
        handles: DSV4SWAPageHandles | None,
        *,
        tombstone: bool = False,
    ) -> None:
        if handles is None or handles.length == 0 or handles.swa_pages is None:
            return
        if not self._swa_independent_lifecycle_enabled:
            return
        {
            "tombstone": bool(tombstone),
            "handle_length": int(handles.length),
        }
        pages = handles.swa_pages.to(device=self.device, dtype=torch.long)
        pages = pages[(pages >= 0) & (pages != self._swa_dummy_page)]
        if pages.numel() == 0:
            return
        pages = torch.unique(pages)
        self._decrement_refcount(self._swa_page_refcount, pages, "SWA page")
        self._bump_swa_ownership_version()
        freed = pages[self._swa_page_refcount[pages] == 0].to(torch.int32)
        if freed.numel() > 0:
            self._clear_full_to_swa_mappings_for_swa_pages(freed)
            self._free_swa_pages = torch.cat([self._free_swa_pages, freed])
            self._swa_pages_freed_total += int(freed.numel())
        if tombstone:
            self._swa_pages_tombstoned_total += int(pages.numel())

    def release_swa_for_full_indices(
        self,
        full_indices: torch.Tensor,
        page_size: int,
        *,
        tombstone: bool = True,
    ) -> None:
        if not self._swa_independent_lifecycle_enabled or full_indices.numel() == 0:
            return
        if full_indices.numel() % page_size != 0:
            raise ValueError(
                "DSV4 active SWA release requires page-aligned full indices, "
                f"got {full_indices.numel()} tokens for page_size={page_size}"
            )
        pages = full_indices.to(device=self.device, dtype=torch.long).view(-1, page_size)
        valid = pages[:, 0] >= 0
        if not bool(torch.any(valid)):
            return
        self._release_swa_pages_for_full_pages(
            pages[valid, 0],
            page_size,
            free_swa=True,
            tombstone=tombstone,
        )

    def available_component_pages(self) -> int:
        if not self._component_loc_ownership_enabled:
            return self._num_pages
        counts: list[int] = []
        if self._c4_layer_count:
            counts.append(int(self._free_c4_pages.numel()))
            counts.append(int(self._free_c4_indexer_pages.numel()))
            counts.append(int(self._free_c4_state_pages.numel()))
            counts.append(int(self._free_c4_indexer_state_pages.numel()))
        if self._c128_layer_count:
            counts.append(int(self._free_c128_pages.numel()))
            counts.append(int(self._free_c128_state_pages.numel()))
        return min(counts) if counts else self._num_pages

    def available_swa_pages(self) -> int:
        if not self._swa_independent_lifecycle_enabled:
            return self._num_pages
        return int(self._free_swa_pages.numel())

    def translate_full_locs_to_swa_locs(self, full_locs: torch.Tensor) -> torch.Tensor:
        full_locs = full_locs.to(device=self.device, dtype=torch.long)
        if full_locs.numel() == 0:
            return full_locs
        if not self._swa_independent_lifecycle_enabled:
            return torch.where(full_locs < 0, torch.full_like(full_locs, -1), full_locs)
        full_pages = full_locs.div(self._page_size, rounding_mode="floor")
        offsets = full_locs % self._page_size
        valid = (full_locs >= 0) & (full_pages >= 0) & (full_pages < self._num_pages)
        safe_pages = full_pages.clamp(min=0, max=max(self._num_pages - 1, 0))
        swa_pages = self._full_to_swa_page[safe_pages].to(torch.long)
        mapped = swa_pages * self._page_size + offsets
        out = torch.where(valid & (swa_pages >= 0), mapped, torch.full_like(mapped, -1))
        dummy = full_locs == self._dummy_token_start
        dummy_loc = torch.full_like(out, int(self._swa_dummy_page * self._page_size))
        return torch.where(dummy, dummy_loc, out)

    def _expand_page_starts(self, page_starts: torch.Tensor, page_size: int) -> torch.Tensor:
        if page_starts.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        offsets = torch.arange(page_size, dtype=torch.long, device=self.device)
        full_locs = (page_starts.unsqueeze(1) + offsets).flatten()
        return full_locs[full_locs < self._num_tokens]

    def _valid_page_starts(self, indices: torch.Tensor, page_size: int) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        page_starts = indices.to(device=self.device, dtype=torch.long)[::page_size]
        return page_starts[page_starts >= 0]

    def _component_locs_from_pages(
        self,
        component_pages: torch.Tensor,
        *,
        page_size: int,
    ) -> torch.Tensor:
        if component_pages.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        pages = component_pages.to(device=self.device, dtype=torch.long)
        offsets = torch.arange(page_size, dtype=torch.long, device=self.device)
        locs = (pages.unsqueeze(1) * page_size + offsets.unsqueeze(0)).flatten()
        return locs[locs >= 0]

    def _alloc_component_pages(self, attr: str, count: int, name: str) -> torch.Tensor:
        free_pages = getattr(self, attr)
        if count > int(free_pages.numel()):
            raise RuntimeError(
                f"DSV4 component allocator exhausted for {name}: "
                f"need={count}, available={int(free_pages.numel())}"
            )
        pages = free_pages[:count].clone()
        setattr(self, attr, free_pages[count:])
        return pages

    def _allocate_swa_pages_for_full_pages(
        self,
        page_starts: torch.Tensor,
        page_size: int,
    ) -> None:
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        if page_starts.numel() == 0:
            return
        full_pages = page_starts.div(page_size, rounding_mode="floor")
        count = int(full_pages.numel())
        if torch.any(full_pages < 0) or torch.any(full_pages >= self._num_pages):
            raise RuntimeError("DSV4 SWA allocation received out-of-range full pages")
        if torch.any(self._full_to_swa_page[full_pages] >= 0):
            raise RuntimeError("DSV4 SWA allocation found stale full-to-SWA mapping")
        if count > int(self._free_swa_pages.numel()):
            raise RuntimeError(
                "DSV4 SWA allocator exhausted: "
                f"need={count}, available={int(self._free_swa_pages.numel())}"
            )
        swa_pages = self._free_swa_pages[:count].clone()
        self._free_swa_pages = self._free_swa_pages[count:]
        self._swa_page_refcount[swa_pages.long()] += 1
        self._full_to_swa_page[full_pages] = swa_pages
        self._swa_pages_allocated_total += count
        self._bump_swa_ownership_version()

    def _allocate_component_pages_for_full_pages(
        self,
        page_starts: torch.Tensor,
        page_size: int,
        *,
        clear_modes: set[str] | None = None,
    ) -> None:
        clear_modes = clear_modes or set()
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        if page_starts.numel() == 0:
            return
        full_pages = page_starts.div(page_size, rounding_mode="floor")
        count = int(full_pages.numel())
        if torch.any(full_pages < 0) or torch.any(full_pages >= self._num_pages):
            raise RuntimeError("DSV4 component allocation received out-of-range full pages")
        if (
            torch.any(self._full_to_c4_page[full_pages] >= 0)
            or torch.any(self._full_to_c128_page[full_pages] >= 0)
            or torch.any(self._full_to_c4_indexer_page[full_pages] >= 0)
            or torch.any(self._full_to_c4_state_page[full_pages] >= 0)
            or torch.any(self._full_to_c128_state_page[full_pages] >= 0)
            or torch.any(self._full_to_c4_indexer_state_page[full_pages] >= 0)
        ):
            raise RuntimeError("DSV4 component allocation found stale full-to-component mapping")

        if self._c4_layer_count:
            c4_pages = self._alloc_component_pages("_free_c4_pages", count, "C4")
            c4_locs = self._component_locs_from_pages(
                c4_pages,
                page_size=self._c4_component_page_size,
            )
            self._c4_refcount[c4_locs] += 1
            self._full_to_c4_page[full_pages] = c4_pages
            if "component" in clear_modes:
                self._clear_c4_component_locs(c4_locs)

            indexer_pages = self._alloc_component_pages(
                "_free_c4_indexer_pages",
                count,
                "C4 indexer",
            )
            indexer_locs = self._component_locs_from_pages(
                indexer_pages,
                page_size=self._c4_component_page_size,
            )
            self._c4_indexer_refcount[indexer_locs] += 1
            self._full_to_c4_indexer_page[full_pages] = indexer_pages
            if "component" in clear_modes:
                self._clear_c4_indexer_component_locs(indexer_locs, indexer_pages)

            c4_state_pages = self._alloc_component_pages(
                "_free_c4_state_pages",
                count,
                "C4 state",
            )
            c4_state_locs = self._component_locs_from_pages(
                c4_state_pages,
                page_size=self._c4_state_page_size,
            )
            self._c4_state_refcount[c4_state_locs] += 1
            self._full_to_c4_state_page[full_pages] = c4_state_pages
            if "state" in clear_modes:
                self._clear_c4_state_locs(c4_state_locs)

            indexer_state_pages = self._alloc_component_pages(
                "_free_c4_indexer_state_pages",
                count,
                "C4 indexer state",
            )
            indexer_state_locs = self._component_locs_from_pages(
                indexer_state_pages,
                page_size=self._c4_state_page_size,
            )
            self._c4_indexer_state_refcount[indexer_state_locs] += 1
            self._full_to_c4_indexer_state_page[full_pages] = indexer_state_pages
            if "state" in clear_modes:
                self._clear_c4_indexer_state_locs(indexer_state_locs)

        if self._c128_layer_count:
            c128_pages = self._alloc_component_pages("_free_c128_pages", count, "C128")
            c128_locs = self._component_locs_from_pages(
                c128_pages,
                page_size=self._c128_component_page_size,
            )
            self._c128_refcount[c128_locs] += 1
            self._full_to_c128_page[full_pages] = c128_pages
            if "component" in clear_modes:
                self._clear_c128_component_locs(c128_locs)

            c128_state_pages = self._alloc_component_pages(
                "_free_c128_state_pages",
                count,
                "C128 state",
            )
            c128_state_locs = self._component_locs_from_pages(
                c128_state_pages,
                page_size=self._c128_state_page_size,
            )
            self._c128_state_refcount[c128_state_locs] += 1
            self._full_to_c128_state_page[full_pages] = c128_state_pages
            if "state" in clear_modes:
                self._clear_c128_state_locs(c128_state_locs)

    def _release_component_pages_for_full_pages(
        self,
        page_starts: torch.Tensor,
        page_size: int,
        *,
        free_components: bool,
    ) -> None:
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        if page_starts.numel() == 0:
            return
        full_pages = page_starts.div(page_size, rounding_mode="floor")
        if torch.any(full_pages < 0) or torch.any(full_pages >= self._num_pages):
            raise RuntimeError("DSV4 component free received out-of-range full pages")

        if free_components:
            self.release_component_page_handles(
                DSV4ComponentPageHandles(
                    length=int(full_pages.numel()) * page_size,
                    page_size=page_size,
                    c4_pages=(
                        self._full_to_c4_page[full_pages].clone() if self._c4_layer_count else None
                    ),
                    c128_pages=(
                        self._full_to_c128_page[full_pages].clone()
                        if self._c128_layer_count
                        else None
                    ),
                    c4_indexer_pages=(
                        self._full_to_c4_indexer_page[full_pages].clone()
                        if self._c4_layer_count
                        else None
                    ),
                    c4_state_pages=(
                        self._full_to_c4_state_page[full_pages].clone()
                        if self._c4_layer_count
                        else None
                    ),
                    c128_state_pages=(
                        self._full_to_c128_state_page[full_pages].clone()
                        if self._c128_layer_count
                        else None
                    ),
                    c4_indexer_state_pages=(
                        self._full_to_c4_indexer_state_page[full_pages].clone()
                        if self._c4_layer_count
                        else None
                    ),
                )
            )

        self._full_to_c4_page[full_pages] = -1
        self._full_to_c128_page[full_pages] = -1
        self._full_to_c4_indexer_page[full_pages] = -1
        self._full_to_c4_state_page[full_pages] = -1
        self._full_to_c128_state_page[full_pages] = -1
        self._full_to_c4_indexer_state_page[full_pages] = -1

    def _release_swa_pages_for_full_pages(
        self,
        page_starts: torch.Tensor,
        page_size: int,
        *,
        free_swa: bool,
        tombstone: bool = False,
    ) -> None:
        {
            "page_starts": int(page_starts.numel()),
            "page_size": int(page_size),
            "free_swa": bool(free_swa),
            "tombstone": bool(tombstone),
        }
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        if page_starts.numel() == 0:
            return
        full_pages = page_starts.div(page_size, rounding_mode="floor")
        if torch.any(full_pages < 0) or torch.any(full_pages >= self._num_pages):
            raise RuntimeError("DSV4 SWA free received out-of-range full pages")
        swa_pages = self._full_to_swa_page[full_pages].clone()
        if free_swa:
            self.release_swa_page_handles(
                DSV4SWAPageHandles(
                    length=int(full_pages.numel()) * page_size,
                    page_size=page_size,
                    swa_pages=swa_pages,
                ),
                tombstone=tombstone,
            )
        if torch.any(self._full_to_swa_page[full_pages] >= 0):
            self._full_to_swa_page[full_pages] = -1
            self._bump_swa_ownership_version()
        else:
            self._full_to_swa_page[full_pages] = -1

    def _clear_full_to_swa_mappings_for_swa_pages(self, swa_pages: torch.Tensor) -> None:
        if swa_pages.numel() == 0:
            return
        pages = swa_pages.to(device=self.device, dtype=self._full_to_swa_page.dtype)
        stale = torch.isin(self._full_to_swa_page, pages)
        if bool(torch.any(stale)):
            self._full_to_swa_page[stale] = -1
            self._bump_swa_ownership_version()

    def _bump_swa_ownership_version(self) -> None:
        if self._swa_independent_lifecycle_enabled:
            self._swa_ownership_version += 1

    def _free_component_pages(
        self,
        pages: torch.Tensor,
        *,
        refcount: torch.Tensor,
        page_size: int,
        free_attr: str,
        name: str,
    ) -> None:
        pages = pages.to(device=self.device, dtype=torch.long)
        pages = pages[pages >= 0]
        if pages.numel() == 0:
            return
        pages = torch.unique(pages)
        locs = self._component_locs_from_pages(pages, page_size=page_size)
        self._decrement_refcount(refcount, locs, name)
        freed_pages = pages.to(torch.int32)
        current_free = getattr(self, free_attr)
        setattr(self, free_attr, torch.cat([current_free, freed_pages]))

    def _component_locs_from_full_locs(
        self,
        full_locs: torch.Tensor,
        ratio: Literal[4, 128],
        *,
        component: Literal["compressed", "indexer"],
    ) -> torch.Tensor:
        if ratio == 4:
            component_page_size = self._c4_component_page_size
            mapping = (
                self._full_to_c4_indexer_page if component == "indexer" else self._full_to_c4_page
            )
        else:
            component_page_size = self._c128_component_page_size
            mapping = self._full_to_c128_page
        full_locs = full_locs.to(device=self.device, dtype=torch.long)
        full_pages = full_locs.div(self._page_size, rounding_mode="floor")
        offsets = (full_locs % self._page_size).div(ratio, rounding_mode="floor")
        valid = (full_locs >= 0) & (full_pages >= 0) & (full_pages < self._num_pages)
        out = torch.full_like(full_locs, -1)
        if bool(torch.any(valid)):
            component_pages = mapping[full_pages[valid]]
            if torch.any(component_pages < 0):
                raise RuntimeError(
                    f"DSV4 component loc requested for full locs without active {component} mapping"
                )
            out[valid] = component_pages.to(torch.long) * component_page_size + offsets[valid]
        return out

    def _decrement_refcount(self, refcount: torch.Tensor, locs: torch.Tensor, name: str) -> None:
        if locs.numel() == 0:
            return
        if torch.any(refcount[locs] <= 0):
            raise RuntimeError(f"DSV4 KV cache double free detected in {name} slots")
        refcount[locs] -= 1

    def _assert_no_negative_refcounts(self) -> None:
        for name, refcount in (
            ("full token", self._full_refcount),
            ("SWA page", self._swa_page_refcount),
            ("C4", self._c4_refcount),
            ("C128", self._c128_refcount),
            ("C4 indexer", self._c4_indexer_refcount),
            ("C4 state", self._c4_state_refcount),
            ("C128 state", self._c128_state_refcount),
            ("C4 indexer state", self._c4_indexer_state_refcount),
        ):
            if torch.any(refcount < 0):
                raise RuntimeError(f"DSV4 KV cache has negative {name} refcounts")
        if self._component_loc_ownership_enabled:
            self._assert_component_free_lists_unique()
        if self._swa_independent_lifecycle_enabled:
            self._assert_swa_free_list_unique()
            self._assert_swa_mapping_integrity()

    def _assert_swa_free_list_unique(self) -> None:
        pages = self._free_swa_pages
        total = max(self._swa_num_pages - 1, 0)
        if pages.numel() == 0:
            return
        if torch.any(pages < 0) or torch.any(pages >= total):
            raise RuntimeError("DSV4 SWA free list contains out-of-range pages")
        if torch.unique(pages).numel() != pages.numel():
            raise RuntimeError("DSV4 SWA free list contains duplicate pages")

    def _assert_swa_mapping_integrity(self) -> None:
        mapped = self._full_to_swa_page[self._full_to_swa_page >= 0].to(torch.long)
        if mapped.numel() == 0:
            return
        total = max(self._swa_num_pages - 1, 0)
        if torch.any(mapped >= total):
            raise RuntimeError("DSV4 SWA mapping points outside non-dummy SWA pages")
        if torch.any(self._swa_page_refcount[mapped] <= 0):
            raise RuntimeError("DSV4 SWA mapping points to a free SWA page")
        if self._free_swa_pages.numel() > 0 and bool(
            torch.any(torch.isin(mapped.to(self._free_swa_pages.dtype), self._free_swa_pages))
        ):
            raise RuntimeError("DSV4 SWA mapping points to a page on the free list")

    def _assert_component_free_lists_unique(self) -> None:
        for name, pages, total in (
            ("C4", self._free_c4_pages, self._c4_component_pages),
            ("C128", self._free_c128_pages, self._c128_component_pages),
            ("C4 indexer", self._free_c4_indexer_pages, self._c4_component_pages),
            ("C4 state", self._free_c4_state_pages, self._num_pages),
            ("C128 state", self._free_c128_state_pages, self._num_pages),
            ("C4 indexer state", self._free_c4_indexer_state_pages, self._num_pages),
        ):
            if pages.numel() == 0:
                continue
            if torch.any(pages < 0) or torch.any(pages >= total):
                raise RuntimeError(f"DSV4 {name} free list contains out-of-range pages")
            if torch.unique(pages).numel() != pages.numel():
                raise RuntimeError(f"DSV4 {name} free list contains duplicate pages")

    def k_cache(self, index: int) -> torch.Tensor:
        return self.swa_cache(index)

    def v_cache(self, index: int) -> torch.Tensor:
        return self.swa_cache(index)

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        del v
        self.store_swa(layer_id, k, out_loc)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


def _build_layer_mapping(
    compress_ratios: list[int],
    num_layers: int,
) -> list[DSV4LayerCacheMapping]:
    if not compress_ratios:
        compress_ratios = [0] * num_layers
    if len(compress_ratios) < num_layers:
        compress_ratios = compress_ratios + [0] * (num_layers - len(compress_ratios))

    mappings: list[DSV4LayerCacheMapping] = []
    normal_count = 0
    c4_count = 0
    c128_count = 0
    for layer_id, ratio in enumerate(compress_ratios[:num_layers]):
        if ratio == 0:
            mappings.append(
                DSV4LayerCacheMapping(
                    layer_id=layer_id,
                    compress_ratio=0,
                    normal_layer_id=normal_count,
                    c4_layer_id=None,
                    c128_layer_id=None,
                    indexer_layer_id=None,
                )
            )
            normal_count += 1
        elif ratio == 4:
            mappings.append(
                DSV4LayerCacheMapping(
                    layer_id=layer_id,
                    compress_ratio=4,
                    normal_layer_id=None,
                    c4_layer_id=c4_count,
                    c128_layer_id=None,
                    indexer_layer_id=c4_count,
                )
            )
            c4_count += 1
        elif ratio == 128:
            mappings.append(
                DSV4LayerCacheMapping(
                    layer_id=layer_id,
                    compress_ratio=128,
                    normal_layer_id=None,
                    c4_layer_id=None,
                    c128_layer_id=c128_count,
                    indexer_layer_id=None,
                )
            )
            c128_count += 1
        else:
            raise ValueError(f"Unsupported DSV4 compression ratio: {ratio}")
    return mappings


def estimate_deepseek_v4_kvcache_bytes_per_page(model_config, page_size: int) -> int:
    dtype_size = torch.bfloat16.itemsize
    head_dim = model_config.head_dim
    index_head_dim = model_config.index_head_dim or head_dim
    ratios = model_config.compress_ratios or [0] * model_config.num_layers
    if len(ratios) < model_config.num_layers:
        ratios = ratios + [0] * (model_config.num_layers - len(ratios))
    c4_layers = sum(r == 4 for r in ratios[: model_config.num_layers])
    c128_layers = sum(r == 128 for r in ratios[: model_config.num_layers])

    def compressed_bytes(layers: int, dim: int, ratio: int, multiplier: int = 1) -> int:
        return div_ceil(layers * page_size * dim * multiplier * dtype_size, ratio)

    swa_bytes = model_config.num_layers * page_size * head_dim * dtype_size
    c4_bytes = compressed_bytes(c4_layers, head_dim, 4)
    c128_bytes = compressed_bytes(c128_layers, head_dim, 128)
    indexer_bytes = compressed_bytes(c4_layers, index_head_dim, 4)
    indexer_fp8_extra_bytes = (
        div_ceil(c4_layers * page_size * (index_head_dim + 4), 4)
        if _indexer_fp8_cache_enabled()
        else 0
    )
    c4_state_bytes = c4_layers * DeepSeekV4KVCache.C4_STATE_RING_SIZE * 4 * head_dim * dtype_size
    c4_indexer_state_bytes = (
        c4_layers * DeepSeekV4KVCache.C4_STATE_RING_SIZE * 4 * index_head_dim * dtype_size
    )
    c128_state_bytes = (
        c128_layers * DeepSeekV4KVCache.C128_STATE_RING_SIZE * 2 * head_dim * dtype_size
    )
    return (
        swa_bytes
        + c4_bytes
        + c128_bytes
        + indexer_bytes
        + indexer_fp8_extra_bytes
        + c4_state_bytes
        + c4_indexer_state_bytes
        + c128_state_bytes
    )


__all__ = [
    "DeepSeekV4KVCache",
    "DSV4AllocationCounts",
    "DSV4CacheLayoutPolicy",
    "DSV4ComponentPageHandles",
    "DSV4CompressStatePool",
    "DSV4KVAndScore",
    "DSV4LayerCacheMapping",
    "DSV4SWAPageHandles",
    "estimate_deepseek_v4_kvcache_bytes_per_page",
]

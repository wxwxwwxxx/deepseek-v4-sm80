from __future__ import annotations

import os
from dataclasses import dataclass
from math import gcd
from typing import Literal

import torch
from minisgl.utils import div_ceil

from .base import BaseKVCachePool

DSV4CacheLayout = Literal["bf16_flat", "flashmla_fp8_packed"]
DSV4_INDEXER_FP8_CACHE_ENV = "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _indexer_fp8_cache_enabled() -> bool:
    try:
        from minisgl.kernel import deepseek_v4 as dsv4_kernel

        return bool(dsv4_kernel.dsv4_env_flag(DSV4_INDEXER_FP8_CACHE_ENV))
    except Exception:
        return os.environ.get(DSV4_INDEXER_FP8_CACHE_ENV, "").strip().lower() in _TRUE_ENV_VALUES


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def _align_up(value: int, alignment: int) -> int:
    return div_ceil(value, alignment) * alignment


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
    c4_slots: int
    c128_slots: int
    c4_indexer_slots: int

    @property
    def any_allocated(self) -> bool:
        return any((self.full_slots, self.c4_slots, self.c128_slots, self.c4_indexer_slots))


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
        dtype: torch.dtype | None = None,
        policy: DSV4CacheLayoutPolicy | None = None,
    ) -> None:
        del dtype
        self._policy = policy or DSV4CacheLayoutPolicy()
        self._device = device
        self._dtype = self._policy.storage_dtype
        self._num_layers = model_config.num_layers
        self._head_dim = model_config.head_dim
        self._index_head_dim = model_config.index_head_dim or model_config.head_dim
        self._num_pages = num_pages
        self._page_size = page_size
        self._num_tokens = num_pages * page_size
        self._c4_slots = div_ceil(self._num_tokens, 4)
        self._c128_slots = div_ceil(self._num_tokens, 128)

        self._layer_mapping = _build_layer_mapping(model_config.compress_ratios, self._num_layers)
        self._normal_layer_count = sum(m.compress_ratio == 0 for m in self._layer_mapping)
        self._c4_layer_count = sum(m.compress_ratio == 4 for m in self._layer_mapping)
        self._c128_layer_count = sum(m.compress_ratio == 128 for m in self._layer_mapping)

        shape = (self._num_layers, num_pages, page_size, self._head_dim)
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
        self._c4_refcount = torch.zeros(self._c4_slots, dtype=torch.int16, device=device)
        self._c128_refcount = torch.zeros(self._c128_slots, dtype=torch.int16, device=device)
        self._c4_indexer_refcount = torch.zeros(self._c4_slots, dtype=torch.int16, device=device)

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
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def layer_mapping(self) -> tuple[DSV4LayerCacheMapping, ...]:
        return tuple(self._layer_mapping)

    @property
    def allocation_counts(self) -> DSV4AllocationCounts:
        return DSV4AllocationCounts(
            full_slots=int(torch.count_nonzero(self._full_refcount).item()),
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
        )

    def get_layer_mapping(self, layer_id: int) -> DSV4LayerCacheMapping:
        return self._layer_mapping[layer_id]

    def swa_cache(self, layer_id: int) -> torch.Tensor:
        return self._swa_buffer[layer_id].view(self._num_tokens, self._head_dim)

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
                f"but {DSV4_INDEXER_FP8_CACHE_ENV}=1 was not active at cache allocation."
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
        return torch.unique_consecutive(full_locs // ratio)

    def store_swa(self, layer_id: int, kv: torch.Tensor, out_loc: torch.Tensor) -> None:
        if kv.numel() == 0:
            return
        self.swa_cache(layer_id)[out_loc.long()] = kv.reshape(-1, self._head_dim).to(self._dtype)

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

    def on_pages_allocated(self, page_starts: torch.Tensor, page_size: int) -> None:
        full_locs = self._expand_page_starts(page_starts, page_size)
        if full_locs.numel() == 0:
            return
        self._full_refcount[full_locs] += 1
        if self._c4_layer_count:
            c4_locs = torch.unique(full_locs // 4)
            self._c4_refcount[c4_locs] += 1
            self._c4_indexer_refcount[c4_locs] += 1
        if self._c128_layer_count:
            self._c128_refcount[torch.unique(full_locs // 128)] += 1

    def on_token_indices_freed(self, indices: torch.Tensor, page_size: int) -> None:
        if indices.numel() == 0:
            return
        page_starts = indices[::page_size]
        full_locs = self._expand_page_starts(page_starts, page_size)
        if full_locs.numel() == 0:
            return
        self._decrement_refcount(self._full_refcount, full_locs, "full token")
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
        swa_bytes = self._num_layers * retained_full_tokens * self._head_dim * dtype_size
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
            "swa_bytes": swa_bytes,
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

    def _expand_page_starts(self, page_starts: torch.Tensor, page_size: int) -> torch.Tensor:
        if page_starts.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        page_starts = page_starts.to(device=self.device, dtype=torch.long)
        offsets = torch.arange(page_size, dtype=torch.long, device=self.device)
        full_locs = (page_starts.unsqueeze(1) + offsets).flatten()
        return full_locs[full_locs < self._num_tokens]

    def _decrement_refcount(self, refcount: torch.Tensor, locs: torch.Tensor, name: str) -> None:
        if locs.numel() == 0:
            return
        if torch.any(refcount[locs] <= 0):
            raise RuntimeError(f"DSV4 KV cache double free detected in {name} slots")
        refcount[locs] -= 1

    def _assert_no_negative_refcounts(self) -> None:
        for name, refcount in (
            ("full token", self._full_refcount),
            ("C4", self._c4_refcount),
            ("C128", self._c128_refcount),
            ("C4 indexer", self._c4_indexer_refcount),
        ):
            if torch.any(refcount < 0):
                raise RuntimeError(f"DSV4 KV cache has negative {name} refcounts")

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
    "DSV4CompressStatePool",
    "DSV4_INDEXER_FP8_CACHE_ENV",
    "DSV4KVAndScore",
    "DSV4LayerCacheMapping",
    "estimate_deepseek_v4_kvcache_bytes_per_page",
]

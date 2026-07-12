from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx
from minisgl.dsv4_runtime import get_dsv4_runtime_config
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kvcache.deepseek_v4_pool import DeepSeekV4KVCache
from minisgl.utils import (
    div_ceil,
)

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


DSV4CompressRatio = Literal[0, 4, 128]
_PAGE_INDEX_ALIGNMENT = 64
_DIRECT_GRAPH_METADATA_ALL_GROUPS = frozenset({"swa", "c4", "c128"})


def _direct_graph_metadata_groups() -> frozenset[str]:
    return get_dsv4_runtime_config().direct_graph_metadata_groups


def _swa_metadata_page_table_cache_enabled() -> bool:
    return dsv4_kernel.dsv4_optimized_enabled()


def _swa_direct_token_metadata_enabled() -> bool:
    return dsv4_kernel.dsv4_optimized_enabled()




def _swa_direct_replay_metadata_fused_enabled() -> bool:
    return dsv4_kernel.dsv4_optimized_enabled()


def _pad_last_dim(
    x: torch.Tensor,
    multiple: int = _PAGE_INDEX_ALIGNMENT,
    value: int = -1,
) -> torch.Tensor:
    size = x.shape[-1]
    target_size = div_ceil(size, multiple) * multiple
    if target_size == size:
        return x
    out = torch.full(
        (*x.shape[:-1], target_size),
        value,
        dtype=x.dtype,
        device=x.device,
    )
    out[..., :size] = x
    return out


def _to_int32(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device=device, dtype=torch.int32)










def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False














@dataclass
class DSV4CoreAttentionMetadata(BaseAttnMetadata):
    raw_out_loc: torch.Tensor
    page_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    req_seq_lens: torch.Tensor
    extend_lens: torch.Tensor
    positions: torch.Tensor
    req_table_indices: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_out_loc: torch.Tensor | None
    c128_out_loc: torch.Tensor | None
    c4_indexer_out_loc: torch.Tensor | None
    c4_topk_lengths_raw: torch.Tensor
    c4_topk_lengths_clamp1: torch.Tensor
    c4_sparse_topk_lengths: torch.Tensor
    c4_sparse_raw_indices: torch.Tensor
    c4_sparse_page_indices: torch.Tensor
    c4_sparse_full_indices: torch.Tensor
    c128_topk_lengths_clamp1: torch.Tensor
    c128_raw_indices: torch.Tensor
    c128_page_indices: torch.Tensor
    c128_full_indices: torch.Tensor
    swa_out_loc: torch.Tensor | None = None
    component_loc_ownership: bool = False
    c4_page_table: torch.Tensor | None = None
    c128_page_table: torch.Tensor | None = None
    c4_indexer_page_table: torch.Tensor | None = None
    swa_source_elided_for_graph: bool = False
    c4_sparse_source_elided_for_graph: bool = False
    c128_source_elided_for_graph: bool = False
    swa_ownership_version: int = 0
    materialized_seq_lens: torch.Tensor | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


@dataclass
class DSV4IndexerMetadata:
    page_size: int
    page_table: torch.Tensor
    c4_seq_lens: torch.Tensor

    @property
    def c4_page_size(self) -> int:
        return max(self.page_size // 4, 1)


@dataclass
class DSV4CompressMetadata:
    ratio: Literal[4, 128]
    write_loc: torch.Tensor | None
    seq_lens: torch.Tensor
    positions: torch.Tensor


@dataclass
class DSV4AttentionMetadata(BaseAttnMetadata):
    core_attn_metadata: DSV4CoreAttentionMetadata
    indexer_metadata: DSV4IndexerMetadata | None = None
    c4_compress_metadata: DSV4CompressMetadata | None = None
    c128_compress_metadata: DSV4CompressMetadata | None = None

    @property
    def core_metadata(self) -> DSV4CoreAttentionMetadata:
        return self.core_attn_metadata

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.core_attn_metadata.get_last_indices(bs)


@dataclass
class DSV4RawDecodeGraphMetadata(BaseAttnMetadata):
    raw_out_loc: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    req_seq_lens: torch.Tensor
    extend_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    req_table_indices: torch.Tensor
    page_table: torch.Tensor
    materialized_seq_lens: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    component_loc_ownership: bool
    c4_page_table: torch.Tensor | None
    c128_page_table: torch.Tensor | None
    c4_indexer_page_table: torch.Tensor | None
    swa_ownership_version: int = 0
    oracle_metadata: DSV4AttentionMetadata | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class DSV4AttentionBackend(BaseAttnBackend):
    """Correctness-first DSV4 attention backend.

    This backend deliberately materializes SGLang-style DSV4 metadata while the
    actual attention path stays in PyTorch.  Fused FlashMLA/indexer kernels can
    later consume the same metadata fields without changing the model call site.
    """

    def __init__(self, config: ModelConfig) -> None:
        ctx = get_global_ctx()
        assert isinstance(ctx.kv_cache, DeepSeekV4KVCache)
        self.config = config
        self.kvcache = ctx.kv_cache
        self.device = self.kvcache.device
        self.page_size = ctx.page_size
        self.window_size = int(config.window_size or 128)
        self.index_topk = int(config.index_topk or 512)
        self.softmax_scale = config.head_dim**-0.5
        self.capture: DSV4AttentionMetadata | None = None
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self._capture_graph_inputs_bound = False
        self._capture_compressed_locs_in_graph = False
        self._capture_compressed_locs_in_graph_component_guarded = False
        self._component_page_table_cache_width = 0
        self._component_page_table_cache_rows = 0
        self._component_page_table_cache_has_c4 = False
        self._component_page_table_cache_has_c128 = False
        self._component_page_table_cache_c4: torch.Tensor | None = None
        self._component_page_table_cache_c128: torch.Tensor | None = None
        self._component_page_table_cache_indexer: torch.Tensor | None = None
        self._component_page_table_cache_signatures: dict[int, tuple[int, ...]] = {}
        self._swa_page_table_cache_width = 0
        self._swa_page_table_cache_rows = 0
        self._swa_page_table_cache: torch.Tensor | None = None
        self._swa_page_table_cache_signatures: dict[int, tuple[int, ...]] = {}
        self._prep_metadata_in_graph = False
        self._prep_metadata_in_graph_requested = False
        self._prep_metadata_in_graph_unsupported_reason: str | None = None
        self._pending_prep_metadata_oracle: DSV4AttentionMetadata | None = None
        self._pending_prep_metadata_oracle_rows = 0
        self._prep_metadata_in_graph_oracle_replay_step = 0
        self._materializing_prep_metadata_oracle = False
        self._c128_prefill_one_surface_status: dict[str, int | str] = {
            "calls": 0,
            "backend": "not_run",
            "last_rows": 0,
            "last_width": 0,
            "last_surface_bytes": 0,
            "last_raw_placeholder_bytes": 0,
            "last_full_placeholder_bytes": 0,
            "max_width": 0,
            "max_surface_bytes": 0,
        }
        dsv4_kernel.warmup_indexer_fp8_backend(self.device)

    @property
    def capture_compressed_locs_in_graph(self) -> bool:
        return self._capture_compressed_locs_in_graph

    @property
    def capture_compressed_locs_in_graph_component_guarded(self) -> bool:
        return self._capture_compressed_locs_in_graph_component_guarded

    @property
    def prep_metadata_in_graph(self) -> bool:
        return self._prep_metadata_in_graph

    @property
    def prep_metadata_in_graph_requested(self) -> bool:
        return self._prep_metadata_in_graph_requested

    @property
    def prep_metadata_in_graph_unsupported_reason(self) -> str | None:
        return self._prep_metadata_in_graph_unsupported_reason

    def c128_prefill_one_surface_status(self) -> dict[str, int | str]:
        return dict(self._c128_prefill_one_surface_status)

    def get_layer_compress_ratio(self, layer_id: int) -> DSV4CompressRatio:
        return self.kvcache.get_layer_mapping(layer_id).compress_ratio

    def prepare_metadata(self, batch: Batch) -> None:
        if self._should_prepare_raw_decode_metadata_in_graph(batch):
            batch.attn_metadata = self._build_raw_decode_graph_metadata(batch)
        else:
            batch.attn_metadata = self._build_metadata(batch)


    def _compute_prep_metadata_in_graph_supported(
        self,
        core: DSV4CoreAttentionMetadata,
    ) -> bool:
        if not self._prep_metadata_in_graph_requested:
            self._prep_metadata_in_graph_unsupported_reason = None
            return False

        def unsupported(reason: str) -> bool:
            self._prep_metadata_in_graph_unsupported_reason = reason
            return False

        if self.device.type != "cuda":
            return unsupported("non_cuda_device")
        if self.page_size != 256:
            return unsupported("unsupported_page_size")
        if not self._capture_graph_inputs_bound:
            return unsupported("graph_inputs_not_bound")
        if not bool(getattr(self.kvcache, "component_loc_ownership_enabled", False)):
            return unsupported("component_loc_ownership_required")
        if bool(getattr(self.kvcache, "swa_independent_lifecycle_enabled", False)):
            full_to_swa_page = getattr(self.kvcache, "_full_to_swa_page", None)
            if full_to_swa_page is None:
                return unsupported("missing_swa_full_to_swa_page")
            if (
                not isinstance(full_to_swa_page, torch.Tensor)
                or not full_to_swa_page.is_cuda
                or full_to_swa_page.dtype is not torch.int32
                or not full_to_swa_page.is_contiguous()
                or int(getattr(self.kvcache, "_dummy_token_start", -1)) < 0
                or int(getattr(self.kvcache, "_swa_dummy_page", -1)) < 0
            ):
                return unsupported("invalid_swa_independent_mapping_surface")
        if (
            core.c4_page_table is None
            or core.c128_page_table is None
            or core.c4_indexer_page_table is None
            or core.c4_out_loc is None
            or core.c128_out_loc is None
            or core.c4_indexer_out_loc is None
            or core.materialized_seq_lens is None
        ):
            return unsupported("missing_capture_surfaces")
        if not dsv4_kernel.dsv4_optimized_triton_enabled():
            return unsupported("triton_unavailable_or_not_sm80")
        self._prep_metadata_in_graph_unsupported_reason = None
        return True

    def _should_prepare_raw_decode_metadata_in_graph(self, batch: Batch) -> bool:
        if not self._prep_metadata_in_graph:
            return False
        if not batch.is_decode:
            return False
        if self.capture is None:
            return False
        padded_size = int(getattr(batch, "padded_size", batch.size))
        if padded_size <= 0 or padded_size not in self.capture_bs:
            return False
        return True

    def _build_raw_decode_graph_metadata(self, batch: Batch) -> DSV4RawDecodeGraphMetadata:
        reqs = batch.padded_reqs
        if not reqs:
            raise ValueError("DSV4 raw graph metadata requires at least one request")
        device = self.device
        rows = int(sum(req.extend_len for req in reqs))
        {
            "phase": batch.phase,
            "batch_size": int(batch.size),
            "padded_size": int(getattr(batch, "padded_size", batch.size)),
            "rows": rows,
            "prep_in_graph": True,
        }
        positions = _to_int32(batch.positions, device)
        raw_out_loc = _to_int32(batch.out_loc, device)
        extend_lens_list = [req.extend_len for req in reqs]
        req_seq_lens_list = [req.device_len for req in reqs]
        materialized_seq_lens_list = self._graph_replay_materialized_seq_len_values(
            batch,
            rows,
        )
        if materialized_seq_lens_list is None:
            materialized_seq_lens_list = req_seq_lens_list
        max_seqlen_q = max(extend_lens_list)
        max_seqlen_k = max(req_seq_lens_list)
        extend_lens = torch.tensor(extend_lens_list, dtype=torch.int32, device=device)
        req_seq_lens = torch.tensor(req_seq_lens_list, dtype=torch.int32, device=device)
        materialized_seq_lens = torch.tensor(
            materialized_seq_lens_list[:rows],
            dtype=torch.int32,
            device=device,
        )
        if int(materialized_seq_lens.numel()) < rows:
            materialized_seq_lens = F.pad(
                materialized_seq_lens,
                (0, rows - int(materialized_seq_lens.numel())),
            )
        cu_seqlens_q = F.pad(extend_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
        seq_lens = positions + 1
        table_indices = torch.empty(positions.numel(), dtype=torch.int32, device=device)
        offset = 0
        for req, length in zip(reqs, extend_lens_list):
            table_indices[offset : offset + length].fill_(req.table_idx)
            offset += length
        assert offset == positions.numel()
        page_table = self._make_page_table(table_indices, max_seqlen_k)
        if dsv4_kernel.dsv4_optimized_enabled():
            component_tables = self._make_component_page_tables_cached(
                reqs,
                max_seqlen_k,
                table_indices,
                has_c4=True,
                has_c128=True,
            )
        else:
            component_tables = self._make_component_page_tables(reqs, max_seqlen_k)
        oracle = None
        return DSV4RawDecodeGraphMetadata(
            raw_out_loc=raw_out_loc,
            positions=positions,
            seq_lens=seq_lens,
            req_seq_lens=req_seq_lens,
            extend_lens=extend_lens,
            cu_seqlens_q=cu_seqlens_q,
            req_table_indices=table_indices,
            page_table=page_table,
            materialized_seq_lens=materialized_seq_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            component_loc_ownership=True,
            c4_page_table=component_tables[0],
            c128_page_table=component_tables[1],
            c4_indexer_page_table=component_tables[2],
            swa_ownership_version=self._current_swa_ownership_version(),
            oracle_metadata=oracle,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        *,
        compress_ratio: DSV4CompressRatio | None = None,
        attn_sink: torch.Tensor | None = None,
        swa_cache_written: bool = False,
    ) -> torch.Tensor:
        del v
        metadata = batch.attn_metadata
        if not isinstance(metadata, DSV4AttentionMetadata):
            raise TypeError(
                "DSV4AttentionBackend requires DSV4AttentionMetadata. "
                "Call prepare_metadata before DSV4 model forward."
            )
        ratio = compress_ratio
        if ratio is None:
            ratio = self.get_layer_compress_ratio(layer_id)
        if not swa_cache_written:
            swa_out_loc = metadata.core_metadata.swa_out_loc
            rows = int(batch.out_loc.shape[0])
            if swa_out_loc is not None and int(swa_out_loc.shape[0]) >= rows:
                dsv4_kernel.store_swa_fallback(
                    self.kvcache,
                    layer_id,
                    k,
                    swa_out_loc[:rows],
                    out_loc_is_swa=True,
                )
            else:
                dsv4_kernel.store_swa_fallback(
                    self.kvcache, layer_id, k, batch.out_loc
                )
        {0: "swa_attention", 4: "c4_attention", 128: "c128_attention"}[ratio]
        return self._fallback_attention(
            q, layer_id, metadata.core_metadata, ratio, attn_sink
        )

    def store_compressed(
        self,
        layer_id: int,
        kv: torch.Tensor,
        batch: Batch,
        compress_ratio: Literal[4, 128],
        *,
        norm_weight: torch.Tensor | None = None,
        rms_norm_eps: float | None = None,
        rotary_dim: int = 0,
        base: float = 10000.0,
        original_seq_len: int = 0,
        factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        metadata = batch.attn_metadata
        if not isinstance(metadata, DSV4AttentionMetadata):
            return
        compress_metadata = (
            metadata.c4_compress_metadata
            if compress_ratio == 4
            else metadata.c128_compress_metadata
        )
        if compress_metadata is None:
            return
        loc = compress_metadata.write_loc
        if loc is None or loc.numel() == 0 or kv.numel() == 0:
            return
        positions = compress_metadata.positions
        if loc.numel() == positions.numel():
            n = min(loc.numel(), positions.numel(), kv.shape[0])
            compressed_positions = positions[:n]
        else:
            n = min(loc.numel(), kv.shape[0])
            boundary = (positions + 1) % compress_ratio == 0
            compressed_positions = positions[boundary]
            if compressed_positions.numel() < n:
                n = compressed_positions.numel()
        if n == 0:
            return
        dsv4_kernel.compress_norm_rope_store_fallback(
            self.kvcache,
            layer_id,
            kv[:n],
            loc[:n],
            positions=compressed_positions[:n],
            norm_weight=norm_weight,
            rms_norm_eps=rms_norm_eps,
            rotary_dim=rotary_dim,
            base=base,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            cache_type="compressed",
        )

    def store_indexer(
        self,
        layer_id: int,
        kv: torch.Tensor,
        batch: Batch,
        *,
        norm_weight: torch.Tensor | None = None,
        rms_norm_eps: float | None = None,
        rotary_dim: int = 0,
        base: float = 10000.0,
        original_seq_len: int = 0,
        factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
        apply_hadamard: bool = False,
    ) -> None:
        metadata = batch.attn_metadata
        if not isinstance(metadata, DSV4AttentionMetadata):
            return
        compress_metadata = metadata.c4_compress_metadata
        if compress_metadata is None:
            return
        loc = metadata.core_metadata.c4_indexer_out_loc
        if loc is None or loc.numel() == 0 or kv.numel() == 0:
            return
        positions = compress_metadata.positions
        if loc.numel() == positions.numel():
            n = min(loc.numel(), positions.numel(), kv.shape[0])
            compressed_positions = positions[:n]
        else:
            n = min(loc.numel(), kv.shape[0])
            boundary = (positions + 1) % compress_metadata.ratio == 0
            compressed_positions = positions[boundary]
            if compressed_positions.numel() < n:
                n = compressed_positions.numel()
        if n == 0:
            return
        dsv4_kernel.compress_norm_rope_store_fallback(
            self.kvcache,
            layer_id,
            kv[:n],
            loc[:n],
            positions=compressed_positions[:n],
            norm_weight=norm_weight,
            rms_norm_eps=rms_norm_eps,
            rotary_dim=rotary_dim,
            base=base,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            cache_type="indexer",
            apply_hadamard=apply_hadamard,
        )

    def select_indexer(
        self,
        layer_id: int,
        q: torch.Tensor,
        weights: torch.Tensor,
        batch: Batch,
    ) -> dsv4_kernel.DSV4IndexerSelectOutput | None:
        metadata = batch.attn_metadata
        if not isinstance(metadata, DSV4AttentionMetadata):
            return None
        indexer_metadata = metadata.indexer_metadata
        if indexer_metadata is None or q.numel() == 0:
            return None
        rows = min(
            q.shape[0],
            weights.shape[0],
            indexer_metadata.c4_seq_lens.shape[0],
            indexer_metadata.page_table.shape[0],
        )
        if rows == 0:
            return None
        q = q[:rows]
        weights = weights[:rows]
        seq_lens = indexer_metadata.c4_seq_lens[:rows]
        page_table = indexer_metadata.page_table[:rows]
        out = dsv4_kernel.indexer_select_bf16_fallback(
            q,
            weights,
            self.kvcache.indexer_cache(layer_id),
            seq_lens,
            page_table,
            page_size=indexer_metadata.c4_page_size,
            width=max(self.index_topk, 1),
            ratio=4,
            layer_id=layer_id,
        )
        core = metadata.core_metadata
        raw_indices, page_indices, full_indices = (
            self._remap_indexer_topk_for_attention(core, out)
        )
        self._merge_indexer_rows_in_place(core.c4_sparse_raw_indices, raw_indices)
        self._merge_indexer_rows_in_place(core.c4_sparse_page_indices, page_indices)
        self._merge_indexer_rows_in_place(core.c4_sparse_full_indices, full_indices)
        if out.topk.topk_lens is not None:
            self._merge_indexer_lengths_in_place(
                core.c4_sparse_topk_lengths,
                out.topk.topk_lens,
            )
        return out

    def select_indexer_fp8(
        self,
        layer_id: int,
        q_values: torch.Tensor,
        weights: torch.Tensor,
        batch: Batch,
    ) -> dsv4_kernel.DSV4IndexerSelectOutput | None:
        metadata = batch.attn_metadata
        if not isinstance(metadata, DSV4AttentionMetadata):
            return None
        indexer_metadata = metadata.indexer_metadata
        if indexer_metadata is None or q_values.numel() == 0:
            return None
        rows = min(
            q_values.shape[0],
            weights.shape[0],
            indexer_metadata.c4_seq_lens.shape[0],
            indexer_metadata.page_table.shape[0],
        )
        if rows == 0:
            return None
        q_values = q_values[:rows]
        weights = weights[:rows]
        seq_lens = indexer_metadata.c4_seq_lens[:rows]
        page_table = indexer_metadata.page_table[:rows]
        if (
            hasattr(self.kvcache, "has_indexer_fp8_paged_cache")
            and self.kvcache.has_indexer_fp8_paged_cache()
        ):
            out = dsv4_kernel.indexer_select_fp8_paged_fallback(
                q_values,
                weights,
                self.kvcache.indexer_fp8_paged_cache(layer_id),
                seq_lens,
                page_table,
                page_size=indexer_metadata.c4_page_size,
                width=max(self.index_topk, 1),
                ratio=4,
                layer_id=layer_id,
            )
        else:
            cache_values, cache_scales = self.kvcache.indexer_fp8_cache(layer_id)
            out = dsv4_kernel.indexer_select_fp8_fallback(
                q_values,
                weights,
                cache_values,
                cache_scales,
                seq_lens,
                page_table,
                page_size=indexer_metadata.c4_page_size,
                width=max(self.index_topk, 1),
                ratio=4,
                layer_id=layer_id,
        )
        core = metadata.core_metadata
        raw_indices, page_indices, full_indices = (
            self._remap_indexer_topk_for_attention(core, out)
        )
        self._merge_indexer_rows_in_place(core.c4_sparse_raw_indices, raw_indices)
        self._merge_indexer_rows_in_place(core.c4_sparse_page_indices, page_indices)
        self._merge_indexer_rows_in_place(core.c4_sparse_full_indices, full_indices)
        if out.topk.topk_lens is not None:
            self._merge_indexer_lengths_in_place(
                core.c4_sparse_topk_lengths,
                out.topk.topk_lens,
            )
        return out


    def _remap_indexer_topk_for_attention(
        self,
        core: DSV4CoreAttentionMetadata,
        out: dsv4_kernel.DSV4IndexerSelectOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_indices = out.topk.raw_indices
        page_indices = out.topk.page_indices
        full_indices = out.topk.full_indices
        if core.component_loc_ownership and core.c4_page_table is not None:
            component_page_size = self.kvcache.c4_component_page_size
            remapped = dsv4_kernel.remap_indexer_topk_locs(
                raw_indices,
                core.c4_page_table[: raw_indices.shape[0]],
                core.page_table[: raw_indices.shape[0]],
                component_page_size=component_page_size,
                full_page_size=self.page_size,
                ratio=4,
            )
            if remapped is not None:
                page_indices, full_indices = remapped
                return raw_indices, page_indices, full_indices
            page_indices = self._compressed_raw_to_component_locs(
                core.c4_page_table[: raw_indices.shape[0]],
                raw_indices,
                4,
            ).to(torch.int32)
            full_indices = self._compressed_raw_to_full_locs_from_page_table(
                core.page_table[: raw_indices.shape[0]],
                raw_indices,
                4,
            ).to(torch.int32)
        return raw_indices, page_indices, full_indices

    def _merge_indexer_rows(self, current: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        padded = _pad_last_dim(values, value=-1)
        if current.shape == padded.shape:
            return padded
        rows = current.shape[0]
        width = max(current.shape[1], padded.shape[1])
        out = torch.full((rows, width), -1, dtype=current.dtype, device=current.device)
        copy_rows = min(rows, padded.shape[0])
        copy_width = min(width, padded.shape[1])
        out[:copy_rows, :copy_width] = padded[:copy_rows, :copy_width].to(
            device=current.device,
            dtype=current.dtype,
        )
        return out

    def _merge_indexer_rows_in_place(self, current: torch.Tensor, values: torch.Tensor) -> None:
        padded = _pad_last_dim(values, value=-1).to(device=current.device, dtype=current.dtype)
        current.fill_(-1)
        rows = min(current.shape[0], padded.shape[0])
        width = min(current.shape[1], padded.shape[1])
        if rows > 0 and width > 0:
            current[:rows, :width].copy_(padded[:rows, :width])

    def _merge_indexer_lengths(self, current: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        lens = values.to(device=current.device, dtype=current.dtype).reshape(-1)
        if current.shape == lens.shape:
            return lens
        out = torch.zeros_like(current)
        rows = min(out.numel(), lens.numel())
        if rows > 0:
            out[:rows] = lens[:rows]
        return out

    def _merge_indexer_lengths_in_place(self, current: torch.Tensor, values: torch.Tensor) -> None:
        lens = values.to(device=current.device, dtype=current.dtype).reshape(-1)
        current.zero_()
        rows = min(current.numel(), lens.numel())
        if rows > 0:
            current[:rows].copy_(lens[:rows])

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        self.max_graph_bs = max(bs_list) if bs_list else 0
        self._capture_graph_inputs_bound = False
        self._capture_compressed_locs_in_graph = False
        self._capture_compressed_locs_in_graph_component_guarded = False
        self._prep_metadata_in_graph_requested = dsv4_kernel.dsv4_optimized_enabled()
        self._prep_metadata_in_graph = False
        self._prep_metadata_in_graph_unsupported_reason = None
        self._pending_prep_metadata_oracle = None
        self._pending_prep_metadata_oracle_rows = 0
        self._prep_metadata_in_graph_oracle_replay_step = 0
        if self.max_graph_bs == 0:
            if self._prep_metadata_in_graph_requested:
                self._prep_metadata_in_graph_unsupported_reason = "cuda_graph_disabled"
            return
        self.capture = self._empty_decode_metadata(self.max_graph_bs, max_seq_len)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert self.capture is not None
        assert batch.size in self.capture_bs
        self.capture.core_metadata.swa_ownership_version = self._current_swa_ownership_version()
        batch.attn_metadata = self.capture

    def bind_capture_graph_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        out_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        del input_ids
        if self.capture is None:
            return
        disable_capture_locs = False
        component_guarded = bool(getattr(self.kvcache, "component_loc_ownership_enabled", False))
        self._capture_compressed_locs_in_graph_component_guarded = component_guarded
        core = self.capture.core_metadata
        if out_loc.shape != core.raw_out_loc.shape or positions.shape != core.positions.shape:
            self._capture_graph_inputs_bound = False
            self._capture_compressed_locs_in_graph = False
            return
        core.raw_out_loc = out_loc
        core.positions = positions
        if self.capture.c4_compress_metadata is not None:
            self.capture.c4_compress_metadata.positions = positions
        if self.capture.c128_compress_metadata is not None:
            self.capture.c128_compress_metadata.positions = positions
        self._capture_graph_inputs_bound = True
        self._capture_compressed_locs_in_graph = (
            not disable_capture_locs
            and not component_guarded
            and out_loc.is_cuda
            and positions.is_cuda
            and dsv4_kernel.dsv4_optimized_triton_enabled()
        )
        self._prep_metadata_in_graph = self._compute_prep_metadata_in_graph_supported(core)

    def stage_capture_metadata_for_graph(self, batch: Batch) -> None:
        if self._prep_metadata_in_graph:
            self._stage_prep_metadata_in_graph(batch)
            return
        if not self._capture_compressed_locs_in_graph or self.capture is None:
            return
        if getattr(batch, "attn_metadata", None) is not self.capture:
            return
        core = self.capture.core_metadata
        dsv4_kernel.copy_masked_compressed_locs(
            core.raw_out_loc,
            core.positions,
            core.c4_out_loc,
            core.c128_out_loc,
            batch.padded_size,
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        assert self.capture is not None
        metadata = batch.attn_metadata
        assert isinstance(metadata, (DSV4AttentionMetadata, DSV4RawDecodeGraphMetadata))
        {
            "phase": "decode" if batch.is_decode else batch.phase,
            "batch_size": int(batch.size),
            "padded_size": int(batch.padded_size),
            "rows": int(batch.padded_size),
        }
        if isinstance(metadata, DSV4RawDecodeGraphMetadata):
            if not self._prep_metadata_in_graph:
                metadata = self._build_metadata(batch)
                batch.attn_metadata = metadata
            else:
                self._copy_raw_decode_graph_metadata_for_replay(
                    metadata,
                    batch.padded_size,
                )
                if metadata.oracle_metadata is not None:
                    self._clamp_graph_replay_compressed_read_metadata(
                        batch,
                        metadata.oracle_metadata.core_metadata,
                        batch.padded_size,
                    )
                    ok = self._run_prep_metadata_in_graph_kernel(int(batch.padded_size))
                    if not ok:
                        raise RuntimeError(
                            "Optimized DSV4 graph metadata prep could not materialize "
                            "the pre-forward metadata surface."
                        )
                self._pending_prep_metadata_oracle = metadata.oracle_metadata
                self._pending_prep_metadata_oracle_rows = int(batch.padded_size)
                return
        if metadata is self.capture:
            return
        assert isinstance(metadata, DSV4AttentionMetadata)
        if self._swa_version_guard_required():
            src_core = metadata.core_metadata
            if src_core.swa_ownership_version != self._current_swa_ownership_version():
                metadata = self._build_metadata(batch)
                batch.attn_metadata = metadata
                self._ensure_swa_metadata_current(
                    metadata.core_metadata,
                    context="CUDA graph replay metadata rebuild",
                )
        self._clamp_graph_replay_compressed_read_metadata(
            batch,
            metadata.core_metadata,
            batch.padded_size,
        )
        self._copy_metadata_for_replay(self.capture, metadata, batch.padded_size)

    def _copy_raw_decode_graph_metadata_for_replay(
        self,
        src: DSV4RawDecodeGraphMetadata,
        rows: int,
    ) -> None:
        assert self.capture is not None
        dst_core = self.capture.core_metadata
        if self._swa_version_guard_required():
            current = self._current_swa_ownership_version()
            if int(src.swa_ownership_version) != current:
                raise RuntimeError(
                    "DSV4 independent SWA raw graph metadata ownership version is stale: "
                    f"context=CUDA graph replay raw metadata source, "
                    f"metadata_version={int(src.swa_ownership_version)}, "
                    f"current_version={current}"
                )
        dst_core.component_loc_ownership = bool(src.component_loc_ownership)
        dst_core.swa_ownership_version = int(src.swa_ownership_version)
        dst_core.max_seqlen_q = int(src.max_seqlen_q)
        dst_core.max_seqlen_k = int(src.max_seqlen_k)
        rows = max(0, min(int(rows), int(dst_core.req_table_indices.shape[0])))
        scalar_fields = (
            ("req_table_indices", src.req_table_indices, dst_core.req_table_indices),
            ("req_seq_lens", src.req_seq_lens, dst_core.req_seq_lens),
            ("extend_lens", src.extend_lens, dst_core.extend_lens),
            ("materialized_seq_lens", src.materialized_seq_lens, dst_core.materialized_seq_lens),
        )
        for name, src_tensor, dst_tensor in scalar_fields:
            if dst_tensor is None:
                continue
            dst_tensor[:rows].copy_(src_tensor[:rows])
        dst_core.cu_seqlens_q[: rows + 1].copy_(src.cu_seqlens_q[: rows + 1])
        self._copy_2d(dst_core.page_table, src.page_table, rows, fill=0)
        for name, src_tensor in (
            ("c4_page_table", src.c4_page_table),
            ("c128_page_table", src.c128_page_table),
            ("c4_indexer_page_table", src.c4_indexer_page_table),
        ):
            dst_tensor = getattr(dst_core, name)
            if dst_tensor is None:
                continue
            if src_tensor is None:
                dst_tensor[:rows].fill_(-1)
            else:
                self._copy_2d(dst_tensor, src_tensor, rows, fill=-1)

    def _stage_prep_metadata_in_graph(self, batch: Batch) -> None:
        if self.capture is None or getattr(batch, "attn_metadata", None) is not self.capture:
            return
        rows = int(getattr(batch, "padded_size", batch.size))
        if rows <= 0:
            return
        ok = self._run_prep_metadata_in_graph_kernel(rows)
        if not ok:
            raise RuntimeError(
                "Optimized DSV4 graph metadata prep kernel was unavailable for "
                "this capture surface."
            )

    def _run_prep_metadata_in_graph_kernel(self, rows: int) -> bool:
        if self.capture is None:
            return False
        core = self.capture.core_metadata
        if core.materialized_seq_lens is None:
            return False
        return dsv4_kernel.prep_decode_metadata_in_graph(
            ctx_page_table=get_global_ctx().page_table,
            table_indices=core.req_table_indices,
            positions=core.positions,
            raw_out_loc=core.raw_out_loc,
            materialized_seq_lens=core.materialized_seq_lens,
            c4_page_table=core.c4_page_table,
            c128_page_table=core.c128_page_table,
            c4_indexer_page_table=core.c4_indexer_page_table,
            dst_seq_lens=core.seq_lens,
            dst_swa_topk_lengths=core.swa_topk_lengths,
            dst_c4_topk_lengths_raw=core.c4_topk_lengths_raw,
            dst_c4_topk_lengths_clamp1=core.c4_topk_lengths_clamp1,
            dst_c4_sparse_topk_lengths=core.c4_sparse_topk_lengths,
            dst_c128_topk_lengths_clamp1=core.c128_topk_lengths_clamp1,
            dst_swa_page_indices=core.swa_page_indices,
            dst_c4_sparse_raw_indices=core.c4_sparse_raw_indices,
            dst_c4_sparse_page_indices=core.c4_sparse_page_indices,
            dst_c4_sparse_full_indices=core.c4_sparse_full_indices,
            dst_c128_raw_indices=core.c128_raw_indices,
            dst_c128_page_indices=core.c128_page_indices,
            dst_c128_full_indices=core.c128_full_indices,
            dst_c4_out_loc=core.c4_out_loc,
            dst_c128_out_loc=core.c128_out_loc,
            dst_c4_indexer_out_loc=core.c4_indexer_out_loc,
            dst_swa_out_loc=core.swa_out_loc,
            rows=rows,
            page_size=self.page_size,
            window_size=self.window_size,
            index_topk=self.index_topk,
            swa_full_to_swa_page=getattr(self.kvcache, "_full_to_swa_page", None),
            swa_dummy_token_start=int(getattr(self.kvcache, "_dummy_token_start", -1)),
            swa_dummy_page=int(getattr(self.kvcache, "_swa_dummy_page", -1)),
            swa_independent=bool(
                getattr(self.kvcache, "swa_independent_lifecycle_enabled", False)
            ),
        )


    def validate_after_replay(self, batch: Batch) -> None:
        oracle = self._pending_prep_metadata_oracle
        rows = self._pending_prep_metadata_oracle_rows
        self._pending_prep_metadata_oracle = None
        self._pending_prep_metadata_oracle_rows = 0
        if oracle is None or self.capture is None:
            return
        self._prep_metadata_in_graph_oracle_replay_step += 1
        rows = max(0, min(int(rows), int(batch.padded_size)))
        if rows <= 0:
            return



    def _clamp_graph_replay_compressed_read_metadata(
        self,
        batch: Batch,
        metadata: DSV4CoreAttentionMetadata,
        rows: int,
    ) -> None:
        if not batch.is_decode or rows <= 0:
            return
        if metadata.max_seqlen_q != 1:
            return
        {
            "phase": "decode",
            "batch_size": int(batch.size),
            "padded_size": int(batch.padded_size),
            "rows": int(rows),
        }
        materialized_values = self._graph_replay_materialized_seq_len_values(batch, rows)
        if materialized_values is None:
            return
        rows = min(int(rows), len(materialized_values), int(metadata.seq_lens.shape[0]))
        if rows <= 0:
            return
        clamp_needed = self._graph_replay_compressed_read_clamp_needed(
            batch,
            materialized_values,
            rows,
        )
        if not clamp_needed:
            return
        materialized_seq_lens = torch.tensor(
            materialized_values[:rows],
            dtype=torch.long,
            device=self.device,
        )
        seq_lens = metadata.seq_lens[:rows].to(device=self.device, dtype=torch.long)
        capped_seq_lens = torch.minimum(seq_lens, materialized_seq_lens[:rows])

        table_indices = metadata.req_table_indices[:rows]
        has_c4 = any(mapping.compress_ratio == 4 for mapping in self.kvcache.layer_mapping)
        if has_c4:
            c4_lengths = capped_seq_lens.div(4, rounding_mode="floor").to(torch.int32)
            metadata.c4_topk_lengths_raw[:rows].copy_(c4_lengths)
            metadata.c4_topk_lengths_clamp1[:rows].copy_(c4_lengths.clamp_min(1))
            metadata.c4_sparse_topk_lengths[:rows].copy_(
                c4_lengths.clamp(min=0, max=self.index_topk)
            )
            raw, page, full = self._make_sparse_compressed_indices(
                table_indices,
                c4_lengths,
                4,
                component_page_table=metadata.c4_page_table,
            )
            self._copy_2d(metadata.c4_sparse_raw_indices, raw, rows, fill=-1)
            self._copy_2d(metadata.c4_sparse_page_indices, page, rows, fill=-1)
            self._copy_2d(metadata.c4_sparse_full_indices, full, rows, fill=-1)

        has_c128 = any(mapping.compress_ratio == 128 for mapping in self.kvcache.layer_mapping)
        if has_c128:
            c128_lengths = capped_seq_lens.div(128, rounding_mode="floor").to(torch.int32)
            metadata.c128_topk_lengths_clamp1[:rows].copy_(c128_lengths.clamp_min(1))
            raw, page, full = self._make_all_compressed_indices(
                table_indices,
                c128_lengths,
                128,
                component_page_table=metadata.c128_page_table,
            )
            self._copy_2d(metadata.c128_raw_indices, raw, rows, fill=-1)
            self._copy_2d(metadata.c128_page_indices, page, rows, fill=-1)
            self._copy_2d(metadata.c128_full_indices, full, rows, fill=-1)

    def _graph_replay_materialized_seq_len_values(
        self,
        batch: Batch,
        rows: int,
    ) -> list[int] | None:
        if rows <= 0:
            return None
        values: list[int] = []
        for req in getattr(batch, "padded_reqs", batch.reqs):
            extend_len = max(int(getattr(req, "extend_len", 1)), 1)
            if int(getattr(req, "uid", 0)) < 0:
                materialized = 0
            else:
                max_device_len = int(getattr(req, "max_device_len", getattr(req, "device_len", 0)))
                output_len = int(getattr(req, "output_len", 0))
                device_len = int(getattr(req, "device_len", 0))
                materialized = max(0, min(device_len, max_device_len - output_len))
            values.extend([materialized] * extend_len)
            if len(values) >= rows:
                break
        if not values:
            return None
        if len(values) < rows:
            values.extend([0] * (rows - len(values)))
        return values[:rows]

    def _graph_replay_compressed_read_clamp_needed(
        self,
        batch: Batch,
        materialized_values: list[int],
        rows: int,
    ) -> bool:
        seq_len_values: list[int] = []
        for req in getattr(batch, "padded_reqs", batch.reqs):
            extend_len = max(int(getattr(req, "extend_len", 1)), 1)
            if int(getattr(req, "uid", 0)) < 0:
                seq_len = 0
            else:
                seq_len = int(getattr(req, "device_len", 0))
            seq_len_values.extend([seq_len] * extend_len)
            if len(seq_len_values) >= rows:
                break
        if len(seq_len_values) < rows:
            seq_len_values.extend([0] * (rows - len(seq_len_values)))
        return any(
            int(materialized_values[idx]) < int(seq_len_values[idx])
            for idx in range(min(rows, len(materialized_values), len(seq_len_values)))
        )

    def _swa_version_guard_required(self) -> bool:
        return bool(getattr(self.kvcache, "swa_independent_lifecycle_enabled", False))

    def _current_swa_ownership_version(self) -> int:
        return int(getattr(self.kvcache, "swa_ownership_version", 0))

    def _ensure_swa_metadata_current(
        self,
        metadata: DSV4CoreAttentionMetadata,
        *,
        context: str,
    ) -> None:
        if not self._swa_version_guard_required():
            return
        current = self._current_swa_ownership_version()
        if int(metadata.swa_ownership_version) == current:
            return
        raise RuntimeError(
            "DSV4 independent SWA metadata ownership version is stale: "
            f"context={context}, metadata_version={int(metadata.swa_ownership_version)}, "
            f"current_version={current}"
        )

    def _should_elide_index_source_for_graph(
        self,
        batch: Batch,
        *,
        component_ownership: bool,
        enabled: bool,
        group: Literal["swa", "c4", "c128"],
    ) -> bool:
        if not (
            batch.is_decode
            and component_ownership
            and enabled
            and self.capture is not None
            and dsv4_kernel.dsv4_optimized_enabled()
            and group in _direct_graph_metadata_groups()
        ):
            return False
        if self._materializing_prep_metadata_oracle:
            return False
        if group == "swa" and bool(
            getattr(self.kvcache, "swa_independent_lifecycle_enabled", False)
        ) and not _swa_direct_replay_metadata_fused_enabled():
            return False
        padded_size = int(getattr(batch, "padded_size", batch.size))
        if padded_size <= 0 or padded_size not in self.capture_bs:
            return False
        return self.device.type == "cuda" and dsv4_kernel.dsv4_optimized_triton_enabled()

    def _empty_index_source_placeholder(self, rows: int) -> torch.Tensor:
        return torch.full((int(rows), 1), -1, dtype=torch.int32, device=self.device)

    def _explicit_c128_raw_full_oracle_requested(self) -> bool:
        """Return whether this metadata build explicitly needs C128 raw/full.

        Prefix-debug snapshots serialize the complete metadata family, and the
        decode graph oracle compares it. Both are explicit diagnostic modes;
        ordinary release eager prefill must keep raw/full lazy.
        """
        return None

    def _release_eager_c128_one_surface_configured(
        self,
        batch: Batch,
        *,
        has_c128: bool,
        component_ownership: bool,
    ) -> bool:
        """Identify the release eager path whose ABI is page indices + lengths."""
        return bool(
            not batch.is_decode
            and has_c128
            and component_ownership
            and self.page_size == 256
            and dsv4_kernel.dsv4_optimized_enabled()
            and not self._explicit_c128_raw_full_oracle_requested()
        )

    @staticmethod
    def _aligned_c128_prefill_width(max_seqlen_k: int) -> int:
        raw_width = max(int(max_seqlen_k) // 128, 1)
        return div_ceil(raw_width, _PAGE_INDEX_ALIGNMENT) * _PAGE_INDEX_ALIGNMENT

    def _build_release_eager_c128_one_surface(
        self,
        c128_page_table: torch.Tensor | None,
        c128_lengths_raw: torch.Tensor,
        *,
        max_seqlen_k: int,
        rows: int,
        phase: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the mandatory release eager C128 surface or fail closed."""
        failure_prefix = (
            "DSV4 release eager-prefill C128 one-surface metadata is mandatory"
        )
        if c128_page_table is None:
            raise RuntimeError(
                f"{failure_prefix}, but the Route-B c128_page_table is missing; "
                "refusing legacy raw/page/full materialization."
            )
        cap = dsv4_kernel.detect_dsv4_kernel_capabilities()
        if self.device.type != "cuda" or not (cap.is_sm80 and cap.triton_available):
            raise RuntimeError(
                f"{failure_prefix}, but CUDA sm80/Triton is unavailable "
                f"(device={self.device}, capability={cap.cuda_capability}, "
                f"triton={cap.triton_available}); refusing legacy raw/page/full "
                "materialization."
            )
        width = self._aligned_c128_prefill_width(max_seqlen_k)
        backend: list[str] = []
        page_indices = dsv4_kernel.c128_prefill_page_indices_one_surface(
            c128_page_table,
            c128_lengths_raw,
            width=width,
            component_page_size=self.kvcache.c128_component_page_size,
            _backend=backend,
        )
        if page_indices is None:
            raise RuntimeError(
                f"{failure_prefix}, but c128_prefill_page_indices_one_surface "
                f"rejected rows={rows}, width={width}, "
                f"page_table_shape={tuple(c128_page_table.shape)}; refusing legacy "
                "raw/page/full materialization."
            )
        marker = backend[0] if backend else "triton_c128_prefill_one_surface"
        raw_placeholder = self._empty_index_source_placeholder(rows)
        full_placeholder = self._empty_index_source_placeholder(rows)
        surface_bytes = page_indices.numel() * page_indices.element_size()
        raw_placeholder_bytes = raw_placeholder.numel() * raw_placeholder.element_size()
        full_placeholder_bytes = full_placeholder.numel() * full_placeholder.element_size()
        status = self._c128_prefill_one_surface_status
        status.update(
            calls=int(status["calls"]) + 1,
            backend=marker,
            last_rows=int(rows),
            last_width=int(width),
            last_surface_bytes=surface_bytes,
            last_raw_placeholder_bytes=raw_placeholder_bytes,
            last_full_placeholder_bytes=full_placeholder_bytes,
            max_width=max(int(status["max_width"]), int(width)),
            max_surface_bytes=max(int(status["max_surface_bytes"]), surface_bytes),
        )
        return raw_placeholder, page_indices, full_placeholder

    def _build_metadata(self, batch: Batch) -> DSV4AttentionMetadata:
        reqs = batch.padded_reqs
        if not reqs:
            raise ValueError("DSV4 attention metadata requires at least one request")

        device = self.device
        rows = int(sum(req.extend_len for req in reqs))
        has_c4 = any(m.compress_ratio == 4 for m in self.kvcache.layer_mapping)
        has_c128 = any(m.compress_ratio == 128 for m in self.kvcache.layer_mapping)
        timing_base = {
            "phase": batch.phase,
            "batch_size": int(batch.size),
            "padded_size": int(getattr(batch, "padded_size", batch.size)),
            "rows": rows,
            "component_loc_ownership": bool(
                getattr(self.kvcache, "component_loc_ownership_enabled", False)
            ),
        }
        positions = _to_int32(batch.positions, device)
        raw_out_loc = _to_int32(batch.out_loc, device)
        extend_lens_list = [req.extend_len for req in reqs]
        req_seq_lens_list = [req.device_len for req in reqs]
        max_seqlen_q = max(extend_lens_list)
        max_seqlen_k = max(req_seq_lens_list)

        extend_lens = torch.tensor(extend_lens_list, dtype=torch.int32, device=device)
        req_seq_lens = torch.tensor(req_seq_lens_list, dtype=torch.int32, device=device)
        cu_seqlens_q = F.pad(extend_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
        component_ownership = bool(
            getattr(self.kvcache, "component_loc_ownership_enabled", False)
        )
        swa_independent = bool(
            getattr(self.kvcache, "swa_independent_lifecycle_enabled", False)
        )
        swa_direct_token_metadata = bool(
            swa_independent and batch.is_decode and _swa_direct_token_metadata_enabled()
        )
        table_indices = torch.empty(positions.numel(), dtype=torch.int32, device=device)
        offset = 0
        for req, length in zip(reqs, extend_lens_list):
            table_indices[offset : offset + length].fill_(req.table_idx)
            offset += length
        assert offset == positions.numel()

        seq_lens = positions + 1
        swa_page_table = (
            self._make_swa_page_tables(
                reqs,
                max_seqlen_k,
                table_indices=table_indices,
                use_cache=batch.is_decode
                and _swa_metadata_page_table_cache_enabled(),
                timing_base=timing_base,
            )
            if swa_independent and not swa_direct_token_metadata
            else None
        )

        if component_ownership and batch.is_decode and dsv4_kernel.dsv4_optimized_enabled():
            component_tables = self._make_component_page_tables_cached(
                reqs,
                max_seqlen_k,
                table_indices,
                has_c4=has_c4,
                has_c128=has_c128,
            )
        else:
            component_tables = (
                self._make_component_page_tables(reqs, max_seqlen_k)
                if component_ownership
                else None
            )
        c4_page_table = None if component_tables is None else component_tables[0]
        c128_page_table = None if component_tables is None else component_tables[1]
        c4_indexer_page_table = None if component_tables is None else component_tables[2]
        swa_source_elided = False
        c4_sparse_source_elided = False
        c128_source_elided = False
        deforested = None
        if deforested is not None:
            page_table = deforested.page_table
            swa_page_indices = deforested.swa_page_indices
            swa_topk_lengths = deforested.swa_topk_lengths
            c4_topk_lengths_raw = deforested.c4_topk_lengths_raw
            c4_topk_lengths_clamp1 = deforested.c4_topk_lengths_clamp1
            c4_sparse_topk_lengths = deforested.c4_sparse_topk_lengths
            c4_sparse_raw_indices = deforested.c4_sparse_raw_indices
            c4_sparse_page_indices = deforested.c4_sparse_page_indices
            c4_sparse_full_indices = deforested.c4_sparse_full_indices
            c128_topk_lengths_clamp1 = deforested.c128_topk_lengths_clamp1
            c128_raw_indices = deforested.c128_raw_indices
            c128_page_indices = deforested.c128_page_indices
            c128_full_indices = deforested.c128_full_indices
        else:
            page_table = self._make_page_table(table_indices, max_seqlen_k)
            swa_source_elided = self._should_elide_index_source_for_graph(
                batch,
                component_ownership=component_ownership,
                enabled=True,
                group="swa",
            )
            if swa_source_elided:
                swa_page_indices = self._empty_index_source_placeholder(rows)
            else:
                if swa_direct_token_metadata:
                    swa_page_indices = self._make_swa_indices_direct_token_metadata(
                        table_indices,
                        positions,
                    )
                elif swa_page_table is not None:
                    swa_page_indices = self._make_swa_indices_from_page_table(
                        swa_page_table,
                        positions,
                    )
                else:
                    swa_page_indices = self._make_swa_indices(
                        table_indices, positions
                    )
            swa_topk_lengths = torch.clamp(seq_lens, max=self.window_size)

            c4_topk_lengths_raw = torch.div(seq_lens, 4, rounding_mode="floor")
            c4_topk_lengths_clamp1 = c4_topk_lengths_raw.clamp_min(1)
            c4_sparse_topk_lengths = c4_topk_lengths_raw.clamp(min=0, max=self.index_topk)
            c4_sparse_source_elided = self._should_elide_index_source_for_graph(
                batch,
                component_ownership=component_ownership,
                enabled=has_c4,
                group="c4",
            )
            if c4_sparse_source_elided:
                c4_sparse_raw_indices = self._empty_index_source_placeholder(rows)
                c4_sparse_page_indices = self._empty_index_source_placeholder(rows)
                c4_sparse_full_indices = self._empty_index_source_placeholder(rows)
            else:
                (
                    c4_sparse_raw_indices,
                    c4_sparse_page_indices,
                    c4_sparse_full_indices,
                ) = self._make_sparse_compressed_indices(
                    table_indices,
                    c4_topk_lengths_raw,
                    4,
                    component_page_table=c4_page_table,
                )

            c128_lengths_raw = torch.div(seq_lens, 128, rounding_mode="floor")
            c128_topk_lengths_clamp1 = c128_lengths_raw.clamp_min(1)
            c128_source_elided = self._should_elide_index_source_for_graph(
                batch,
                component_ownership=component_ownership,
                enabled=has_c128,
                group="c128",
            )
            if c128_source_elided:
                c128_raw_indices = self._empty_index_source_placeholder(rows)
                c128_page_indices = self._empty_index_source_placeholder(rows)
                c128_full_indices = self._empty_index_source_placeholder(rows)
            elif self._release_eager_c128_one_surface_configured(
                batch,
                has_c128=has_c128,
                component_ownership=component_ownership,
            ):
                (
                    c128_raw_indices,
                    c128_page_indices,
                    c128_full_indices,
                ) = self._build_release_eager_c128_one_surface(
                    c128_page_table,
                    c128_lengths_raw,
                    max_seqlen_k=max_seqlen_k,
                    rows=rows,
                    phase=batch.phase,
                )
            else:
                c128_raw_indices, c128_page_indices, c128_full_indices = (
                    self._materialize_c128_raw_page_full_oracle(
                        table_indices,
                        c128_lengths_raw,
                        component_page_table=c128_page_table,
                    )
                )

        if component_ownership and component_tables is not None:
            c4_out_loc = self._component_write_locs_from_page_table(
                c4_page_table,
                positions,
                4,
            )
            c128_out_loc = self._component_write_locs_from_page_table(
                c128_page_table,
                positions,
                128,
            )
            c4_indexer_out_loc = self._component_write_locs_from_page_table(
                c4_indexer_page_table,
                positions,
                4,
            )
        else:
            c4_out_loc = self.kvcache.compressed_locs_from_full_locs(
                raw_out_loc,
                4,
                positions,
            )
            c128_out_loc = self.kvcache.compressed_locs_from_full_locs(
                raw_out_loc,
                128,
                positions,
            )
            c4_indexer_out_loc = self.kvcache.indexer_locs_from_full_locs(
                raw_out_loc,
                positions,
            )
        swa_out_loc = self._make_swa_out_loc_for_store(raw_out_loc)
        if c4_out_loc.numel() == 0:
            c4_out_loc = None
        if c128_out_loc.numel() == 0:
            c128_out_loc = None
        if c4_indexer_out_loc.numel() == 0:
            c4_indexer_out_loc = None

        core = DSV4CoreAttentionMetadata(
            raw_out_loc=raw_out_loc,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            seq_lens=seq_lens,
            req_seq_lens=req_seq_lens,
            extend_lens=extend_lens,
            positions=positions,
            req_table_indices=table_indices,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_out_loc=c4_out_loc,
            c128_out_loc=c128_out_loc,
            c4_indexer_out_loc=c4_indexer_out_loc,
            c4_topk_lengths_raw=c4_topk_lengths_raw,
            c4_topk_lengths_clamp1=c4_topk_lengths_clamp1,
            c4_sparse_topk_lengths=c4_sparse_topk_lengths,
            c4_sparse_raw_indices=c4_sparse_raw_indices,
            c4_sparse_page_indices=c4_sparse_page_indices,
            c4_sparse_full_indices=c4_sparse_full_indices,
            c128_topk_lengths_clamp1=c128_topk_lengths_clamp1,
            c128_raw_indices=c128_raw_indices,
            c128_page_indices=c128_page_indices,
            c128_full_indices=c128_full_indices,
            swa_out_loc=swa_out_loc,
            component_loc_ownership=component_ownership,
            c4_page_table=c4_page_table,
            c128_page_table=c128_page_table,
            c4_indexer_page_table=c4_indexer_page_table,
            swa_source_elided_for_graph=swa_source_elided,
            c4_sparse_source_elided_for_graph=c4_sparse_source_elided,
            c128_source_elided_for_graph=c128_source_elided,
            swa_ownership_version=self._current_swa_ownership_version(),
        )

        indexer_metadata = (
            DSV4IndexerMetadata(
                page_size=self.page_size,
                page_table=(
                    core.c4_indexer_page_table
                    if component_ownership and core.c4_indexer_page_table is not None
                    else core.page_table
                ),
                c4_seq_lens=core.c4_topk_lengths_raw,
            )
            if has_c4
            else None
        )
        return DSV4AttentionMetadata(
            core_attn_metadata=core,
            indexer_metadata=indexer_metadata,
            c4_compress_metadata=DSV4CompressMetadata(
                ratio=4,
                write_loc=core.c4_out_loc,
                seq_lens=core.seq_lens,
                positions=core.positions,
            ),
            c128_compress_metadata=DSV4CompressMetadata(
                ratio=128,
                write_loc=core.c128_out_loc,
                seq_lens=core.seq_lens,
                positions=core.positions,
            ),
        )

    def _make_page_table(self, table_indices: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        ctx_page_table = get_global_ctx().page_table
        offsets = torch.arange(
            0,
            max(max_seq_len, 1),
            self.page_size,
            dtype=torch.long,
            device=self.device,
        )
        rows = table_indices.to(torch.long)
        page_table = ctx_page_table[rows[:, None], offsets[None, :]]
        if self.page_size > 1:
            page_table = torch.where(
                page_table >= 0,
                page_table.div(self.page_size, rounding_mode="floor"),
                page_table,
            )
        return page_table.to(torch.int32)

    def _make_component_page_tables(
        self,
        reqs,
        max_seq_len: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return self._make_component_page_tables_uncached(reqs, max_seq_len)

    def _make_component_page_tables_uncached(
        self,
        reqs,
        max_seq_len: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        table_len = div_ceil(max(max_seq_len, 1), self.page_size)
        rows = sum(req.extend_len for req in reqs)
        c4_rows: list[torch.Tensor] = []
        c128_rows: list[torch.Tensor] = []
        indexer_rows: list[torch.Tensor] = []
        has_c4 = any(m.compress_ratio == 4 for m in self.kvcache.layer_mapping)
        has_c128 = any(m.compress_ratio == 128 for m in self.kvcache.layer_mapping)

        for req in reqs:
            c4_table, c128_table, indexer_table = self._build_component_page_table_row(
                req,
                table_len,
                has_c4=has_c4,
                has_c128=has_c128,
            )

            for _ in range(req.extend_len):
                if c4_table is not None:
                    c4_rows.append(c4_table)
                if c128_table is not None:
                    c128_rows.append(c128_table)
                if indexer_table is not None:
                    indexer_rows.append(indexer_table)

        def _stack(chunks: list[torch.Tensor], enabled: bool) -> torch.Tensor | None:
            if not enabled:
                return None
            if not chunks:
                return torch.full((rows, table_len), -1, dtype=torch.int32, device=self.device)
            return torch.stack(chunks).to(torch.int32)

        return _stack(c4_rows, has_c4), _stack(c128_rows, has_c128), _stack(indexer_rows, has_c4)

    def _build_component_page_table_row(
        self,
        req,
        table_len: int,
        *,
        has_c4: bool,
        has_c128: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        c4_table = (
            torch.full((table_len,), -1, dtype=torch.int32, device=self.device)
            if has_c4
            else None
        )
        c128_table = (
            torch.full((table_len,), -1, dtype=torch.int32, device=self.device)
            if has_c128
            else None
        )
        indexer_table = (
            torch.full((table_len,), -1, dtype=torch.int32, device=self.device)
            if has_c4
            else None
        )

        handle_getter = getattr(req.cache_handle, "get_dsv4_component_pages", None)
        handle_pages = handle_getter() if callable(handle_getter) else None
        if handle_pages is not None:
            prefix_pages = min(handle_pages.num_pages, table_len)
            if c4_table is not None and handle_pages.c4_pages is not None:
                c4_table[:prefix_pages] = handle_pages.c4_pages[:prefix_pages].to(
                    device=self.device,
                    dtype=torch.int32,
                )
            if c128_table is not None and handle_pages.c128_pages is not None:
                c128_table[:prefix_pages] = handle_pages.c128_pages[:prefix_pages].to(
                    device=self.device,
                    dtype=torch.int32,
                )
            if indexer_table is not None and handle_pages.c4_indexer_pages is not None:
                indexer_table[:prefix_pages] = handle_pages.c4_indexer_pages[
                    :prefix_pages
                ].to(device=self.device, dtype=torch.int32)

        logical_pages = min(div_ceil(req.device_len, self.page_size), table_len)
        if logical_pages > 0:
            page_offsets = (
                torch.arange(logical_pages, dtype=torch.long, device=self.device)
                * self.page_size
            )
            full_page_starts = get_global_ctx().page_table[req.table_idx, page_offsets]
            active_c4, active_c128, active_indexer = (
                self.kvcache.component_pages_from_full_page_starts(
                    full_page_starts,
                    self.page_size,
                )
            )

            def _fill_missing(dst: torch.Tensor | None, src: torch.Tensor | None) -> None:
                if dst is None or src is None:
                    return
                n = min(dst.numel(), src.numel())
                if n <= 0:
                    return
                missing = dst[:n] < 0
                valid = src[:n] >= 0
                mask = missing & valid
                dst_view = dst[:n]
                dst_view[mask] = src[:n][mask].to(device=dst.device, dtype=dst.dtype)

            _fill_missing(c4_table, active_c4)
            _fill_missing(c128_table, active_c128)
            _fill_missing(indexer_table, active_indexer)

        return c4_table, c128_table, indexer_table

    def _make_component_page_tables_cached(
        self,
        reqs,
        max_seq_len: int,
        table_indices: torch.Tensor,
        *,
        has_c4: bool,
        has_c128: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        table_len = div_ceil(max(max_seq_len, 1), self.page_size)
        rows = sum(req.extend_len for req in reqs)
        {"phase": "decode", "rows": int(rows), "table_width": int(table_len)}
        self._ensure_component_page_table_cache(
            table_len,
            has_c4=has_c4,
            has_c128=has_c128,
        )
        dirty = 0
        clean = 0
        for req in reqs:
            if self._refresh_component_page_table_cache_row(
                req,
                has_c4=has_c4,
                has_c128=has_c128,
            ):
                dirty += 1
            else:
                clean += 1

        row_indices = table_indices.to(device=self.device, dtype=torch.long)

        def _select(src: torch.Tensor | None, enabled: bool) -> torch.Tensor | None:
            if not enabled:
                return None
            assert src is not None
            if rows == 0:
                return torch.full(
                    (0, table_len),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
            return src.index_select(0, row_indices)[:, :table_len].contiguous()

        result = (
            _select(self._component_page_table_cache_c4, has_c4),
            _select(self._component_page_table_cache_c128, has_c128),
            _select(self._component_page_table_cache_indexer, has_c4),
        )


        return result

    def _ensure_component_page_table_cache(
        self,
        table_len: int,
        *,
        has_c4: bool,
        has_c128: bool,
    ) -> None:
        ctx_page_table = get_global_ctx().page_table
        rows = int(ctx_page_table.shape[0])
        width = max(
            int(table_len),
            div_ceil(max(int(ctx_page_table.shape[1]), 1), self.page_size),
        )
        needs_alloc = (
            self._component_page_table_cache_c4 is None and has_c4
        ) or (
            self._component_page_table_cache_c128 is None and has_c128
        ) or (
            self._component_page_table_cache_indexer is None and has_c4
        ) or (
            self._component_page_table_cache_rows != rows
            or self._component_page_table_cache_width < width
            or self._component_page_table_cache_has_c4 != has_c4
            or self._component_page_table_cache_has_c128 != has_c128
        )
        if not needs_alloc:
            return
        self._component_page_table_cache_rows = rows
        self._component_page_table_cache_width = width
        self._component_page_table_cache_has_c4 = has_c4
        self._component_page_table_cache_has_c128 = has_c128
        self._component_page_table_cache_c4 = (
            torch.full((rows, width), -1, dtype=torch.int32, device=self.device)
            if has_c4
            else None
        )
        self._component_page_table_cache_c128 = (
            torch.full((rows, width), -1, dtype=torch.int32, device=self.device)
            if has_c128
            else None
        )
        self._component_page_table_cache_indexer = (
            torch.full((rows, width), -1, dtype=torch.int32, device=self.device)
            if has_c4
            else None
        )
        self._component_page_table_cache_signatures.clear()

    def _component_page_table_cache_signature(self, req) -> tuple[int, ...]:
        handle = getattr(req, "cache_handle", None)
        node = getattr(handle, "node", None)
        node_uuid = int(getattr(node, "uuid", -1))
        handle_cached_len = int(getattr(handle, "cached_len", -1))
        logical_pages = div_ceil(int(req.device_len), self.page_size)
        return (
            int(id(req)),
            int(getattr(req, "uid", -1)),
            int(req.table_idx),
            int(handle_cached_len),
            int(node_uuid),
            int(logical_pages),
        )

    def _refresh_component_page_table_cache_row(
        self,
        req,
        *,
        has_c4: bool,
        has_c128: bool,
    ) -> bool:
        table_idx = int(req.table_idx)
        signature = self._component_page_table_cache_signature(req)
        if self._component_page_table_cache_signatures.get(table_idx) == signature:
            return False

        width = self._component_page_table_cache_width
        c4_row, c128_row, indexer_row = self._build_component_page_table_row(
            req,
            width,
            has_c4=has_c4,
            has_c128=has_c128,
        )

        def _copy_row(dst: torch.Tensor | None, src: torch.Tensor | None) -> None:
            if dst is None:
                return
            dst[table_idx].fill_(-1)
            if src is not None and src.numel() > 0:
                dst[table_idx, : src.numel()].copy_(src)

        _copy_row(self._component_page_table_cache_c4, c4_row)
        _copy_row(self._component_page_table_cache_c128, c128_row)
        _copy_row(self._component_page_table_cache_indexer, indexer_row)
        self._component_page_table_cache_signatures[table_idx] = signature
        return True

    def _make_swa_page_tables(
        self,
        reqs,
        max_seq_len: int,
        *,
        table_indices: torch.Tensor,
        use_cache: bool,
        timing_base: dict[str, int | str | bool],
    ) -> torch.Tensor:
        if use_cache:
            return self._make_swa_page_tables_cached(
                reqs,
                max_seq_len,
                table_indices,
                timing_base=timing_base,
            )
        return self._make_swa_page_tables_uncached(
            reqs,
            max_seq_len,
            timing_base=timing_base,
        )

    def _make_swa_page_tables_uncached(
        self,
        reqs,
        max_seq_len: int,
        *,
        timing_base: dict[str, int | str | bool],
    ) -> torch.Tensor:
        table_len = div_ceil(max(max_seq_len, 1), self.page_size)
        rows = sum(req.extend_len for req in reqs)
        chunks: list[torch.Tensor] = []
        profile = None
        for req in reqs:
            row = self._build_swa_page_table_row(req, table_len, profile=profile)
            for _ in range(req.extend_len):
                chunks.append(row)
        if not chunks:
            return torch.full((rows, table_len), -1, dtype=torch.int32, device=self.device)
        table = torch.stack(chunks).to(torch.int32)
        return table

    def _make_swa_page_tables_cached(
        self,
        reqs,
        max_seq_len: int,
        table_indices: torch.Tensor,
        *,
        timing_base: dict[str, int | str | bool],
    ) -> torch.Tensor:
        table_len = div_ceil(max(max_seq_len, 1), self.page_size)
        rows = sum(req.extend_len for req in reqs)
        profile = None
        self._ensure_swa_page_table_cache(table_len)

        dirty = 0
        clean = 0
        for req in reqs:
            if self._refresh_swa_page_table_cache_row(req, profile=profile):
                dirty += 1
            else:
                clean += 1

        row_indices = table_indices.to(device=self.device, dtype=torch.long)
        if rows == 0:
            table = torch.full(
                (0, table_len),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            assert self._swa_page_table_cache is not None
            table = (
                self._swa_page_table_cache.index_select(0, row_indices)[:, :table_len]
                .contiguous()
            )

        return table

    def _ensure_swa_page_table_cache(self, table_len: int) -> None:
        ctx_page_table = get_global_ctx().page_table
        rows = int(ctx_page_table.shape[0])
        width = max(
            int(table_len),
            div_ceil(max(int(ctx_page_table.shape[1]), 1), self.page_size),
        )
        needs_alloc = (
            self._swa_page_table_cache is None
            or self._swa_page_table_cache_rows != rows
            or self._swa_page_table_cache_width < width
        )
        if not needs_alloc:
            return
        self._swa_page_table_cache_rows = rows
        self._swa_page_table_cache_width = width
        self._swa_page_table_cache = torch.full(
            (rows, width),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self._swa_page_table_cache_signatures.clear()

    def _swa_page_table_cache_signature(self, req) -> tuple[int, ...]:
        handle = getattr(req, "cache_handle", None)
        node = getattr(handle, "node", None)
        node_uuid = int(getattr(node, "uuid", -1))
        handle_cached_len = int(getattr(handle, "cached_len", -1))
        device_len = int(req.device_len)
        logical_pages = div_ceil(device_len, self.page_size)
        live_window_start = max(device_len - int(self.window_size), 0)
        live_page_start = live_window_start // self.page_size
        live_page_end = (device_len - 1) // self.page_size if device_len > 0 else -1
        return (
            int(getattr(req, "uid", -1)),
            int(req.table_idx),
            int(handle_cached_len),
            int(node_uuid),
            int(logical_pages),
            int(live_page_start),
            int(live_page_end),
            int(getattr(req, "swa_evicted_seqlen", 0)),
        )

    def _refresh_swa_page_table_cache_row(
        self,
        req,
        *,
        profile: dict[str, float] | None,
    ) -> bool:
        table_idx = int(req.table_idx)
        signature = self._swa_page_table_cache_signature(req)
        if self._swa_page_table_cache_signatures.get(table_idx) == signature:
            return False
        assert self._swa_page_table_cache is not None
        width = self._swa_page_table_cache_width
        row = self._build_swa_page_table_row(req, width, profile=profile)
        dst = self._swa_page_table_cache[table_idx]
        dst.fill_(-1)
        if row.numel() > 0:
            dst[: row.numel()].copy_(row)
        self._swa_page_table_cache_signatures[table_idx] = signature
        return True

    def _build_swa_page_table_row(
        self,
        req,
        table_len: int,
        *,
        profile: dict[str, float] | None = None,
    ) -> torch.Tensor:
        table = torch.full((table_len,), -1, dtype=torch.int32, device=self.device)
        if int(getattr(req, "uid", 0)) < 0:
            swa_pages = int(self.kvcache.swa_cache(0).shape[0]) // self.page_size
            table.fill_(max(swa_pages - 1, 0))
            return table
        handle_getter = getattr(req.cache_handle, "get_dsv4_swa_pages", None)
        handle_pages = handle_getter() if callable(handle_getter) else None
        if handle_pages is not None and handle_pages.swa_pages is not None:
            prefix_pages = min(handle_pages.num_pages, table_len)
            table[:prefix_pages] = handle_pages.swa_pages[:prefix_pages].to(
                device=self.device,
                dtype=torch.int32,
            )

        logical_pages = min(div_ceil(req.device_len, self.page_size), table_len)
        if logical_pages <= 0:
            return table
        page_offsets = (
            torch.arange(logical_pages, dtype=torch.long, device=self.device)
            * self.page_size
        )
        full_page_starts = get_global_ctx().page_table[req.table_idx, page_offsets]
        active_swa = self.kvcache.swa_pages_from_full_page_starts(
            full_page_starts,
            self.page_size,
        )
        if active_swa is None or active_swa.numel() == 0:
            return table
        n = min(table.numel(), active_swa.numel())
        live_window_start = max(int(req.device_len) - int(self.window_size), 0)
        live_page_start = min(live_window_start // self.page_size, n)
        live_missing = (table[live_page_start:n] < 0) & (active_swa[live_page_start:n] < 0)
        if bool(torch.any(live_missing).item()):
            rel = torch.where(live_missing)[0]
            page = int((rel[0] + live_page_start).item())
            raise RuntimeError(
                "DSV4 independent SWA missing active page mapping: "
                f"uid={getattr(req, 'uid', None)}, table_idx={getattr(req, 'table_idx', None)}, "
                f"device_len={int(req.device_len)}, cached_len={int(req.cached_len)}, "
                f"page={page}, table_len={table_len}"
            )
        missing = table[:n] < 0
        valid = active_swa[:n] >= 0
        mask = missing & valid
        table_view = table[:n]
        table_view[mask] = active_swa[:n][mask].to(device=table.device, dtype=table.dtype)
        return table


    def _make_swa_indices(
        self,
        table_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        offsets = (
            positions[:, None]
            - torch.arange(
                self.window_size,
                dtype=torch.int32,
                device=self.device,
            )[None, :]
        )
        return self._gather_full_locs(table_indices, offsets)

    def _make_swa_indices_direct_token_metadata(
        self,
        table_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        translate = getattr(self.kvcache, "translate_full_locs_to_swa_locs", None)
        if not callable(translate):
            raise RuntimeError(
                f"{DSV4_SWA_DIRECT_TOKEN_METADATA_ENV}=1 requires "
                "DeepSeekV4KVCache.translate_full_locs_to_swa_locs."
            )
        rows = int(positions.numel())
        {
            "phase": "decode",
            "rows": rows,
            "window_size": int(self.window_size),
            "direct_token_metadata": True,
        }
        full_locs = self._make_swa_indices(table_indices, positions)
        return translate(full_locs).to(device=self.device, dtype=torch.int32)

    def _make_swa_out_loc_for_store(self, raw_out_loc: torch.Tensor) -> torch.Tensor | None:
        if not (
            _swa_direct_replay_metadata_fused_enabled()
            and bool(getattr(self.kvcache, "swa_independent_lifecycle_enabled", False))
        ):
            return None
        translate = getattr(self.kvcache, "translate_full_locs_to_swa_locs", None)
        if not callable(translate):
            raise RuntimeError(
                f"{dsv4_kernel.DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED_TOGGLE}=1 requires "
                "DeepSeekV4KVCache.translate_full_locs_to_swa_locs."
            )
        return translate(raw_out_loc).to(device=self.device, dtype=torch.int32)

    def _copy_swa_out_loc_for_replay(
        self,
        dst_core: DSV4CoreAttentionMetadata,
        src_core: DSV4CoreAttentionMetadata,
        rows: int,
    ) -> None:
        if dst_core.swa_out_loc is None or src_core.swa_out_loc is None:
            return
        rows = max(0, min(int(rows), int(dst_core.swa_out_loc.shape[0])))
        rows = min(rows, int(src_core.swa_out_loc.shape[0]))
        if rows <= 0:
            return
        dst_core.swa_out_loc[:rows].copy_(src_core.swa_out_loc[:rows])

    def _make_swa_indices_from_page_table(
        self,
        swa_page_table: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        offsets = (
            positions[:, None]
            - torch.arange(
                self.window_size,
                dtype=torch.int32,
                device=self.device,
            )[None, :]
        )
        valid = offsets >= 0
        clamped = offsets.clamp_min(0).to(torch.long)
        logical_pages = clamped.div(self.page_size, rounding_mode="floor")
        page_offsets = clamped % self.page_size
        rows = torch.arange(clamped.shape[0], dtype=torch.long, device=self.device)[:, None]
        rows = rows.expand_as(clamped)
        max_page = max(swa_page_table.shape[1] - 1, 0)
        gathered_pages = swa_page_table[
            rows,
            logical_pages.clamp(max=max_page),
        ].to(torch.long)
        locs = gathered_pages * self.page_size + page_offsets
        valid = valid & (logical_pages < swa_page_table.shape[1]) & (gathered_pages >= 0)
        return torch.where(valid, locs.to(torch.int32), torch.full_like(offsets, -1))

    def _make_sparse_compressed_indices(
        self,
        table_indices: torch.Tensor,
        lengths: torch.Tensor,
        ratio: Literal[4, 128],
        *,
        component_page_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(self.index_topk, 1)
        raw = torch.full((lengths.numel(), width), -1, dtype=torch.int32, device=self.device)
        for row, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            start = max(0, int(length) - self.index_topk)
            values = torch.arange(start, int(length), dtype=torch.int32, device=self.device)
            raw[row, : values.numel()] = values
        full = self._compressed_raw_to_full_locs(table_indices, raw, ratio)
        if component_page_table is None:
            page = torch.where(full >= 0, full.div(ratio, rounding_mode="floor"), full)
        else:
            page = self._compressed_raw_to_component_locs(component_page_table, raw, ratio)
        return (
            _pad_last_dim(raw, value=-1),
            _pad_last_dim(page.to(torch.int32), value=-1),
            _pad_last_dim(full.to(torch.int32), value=-1),
        )

    def _make_all_compressed_indices(
        self,
        table_indices: torch.Tensor,
        lengths: torch.Tensor,
        ratio: Literal[4, 128],
        *,
        component_page_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(int(lengths.max().item()) if lengths.numel() else 0, 1)
        raw = torch.full((lengths.numel(), width), -1, dtype=torch.int32, device=self.device)
        for row, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            values = torch.arange(int(length), dtype=torch.int32, device=self.device)
            raw[row, : values.numel()] = values
        full = self._compressed_raw_to_full_locs(table_indices, raw, ratio)
        if component_page_table is None:
            page = torch.where(full >= 0, full.div(ratio, rounding_mode="floor"), full)
        else:
            page = self._compressed_raw_to_component_locs(component_page_table, raw, ratio)
        return (
            _pad_last_dim(raw, value=-1),
            _pad_last_dim(page.to(torch.int32), value=-1),
            _pad_last_dim(full.to(torch.int32), value=-1),
        )

    def _materialize_c128_raw_page_full_oracle(
        self,
        table_indices: torch.Tensor,
        lengths: torch.Tensor,
        *,
        component_page_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Explicit legacy/debug materialization; never a release eager fallback."""
        return self._make_all_compressed_indices(
            table_indices,
            lengths,
            128,
            component_page_table=component_page_table,
        )

    def _compressed_raw_to_component_locs(
        self,
        component_page_table: torch.Tensor,
        raw_indices: torch.Tensor,
        ratio: Literal[4, 128],
    ) -> torch.Tensor:
        component_page_size = (
            self.kvcache.c4_component_page_size
            if ratio == 4
            else self.kvcache.c128_component_page_size
        )
        valid = raw_indices >= 0
        raw = raw_indices.clamp_min(0).to(torch.long)
        logical_pages = raw.div(component_page_size, rounding_mode="floor")
        offsets = raw % component_page_size
        rows = torch.arange(raw.shape[0], dtype=torch.long, device=self.device)[:, None]
        rows = rows.expand_as(raw)
        max_page = max(component_page_table.shape[1] - 1, 0)
        gathered_pages = component_page_table[
            rows,
            logical_pages.clamp(max=max_page),
        ].to(torch.long)
        locs = gathered_pages * component_page_size + offsets
        valid = valid & (logical_pages < component_page_table.shape[1]) & (gathered_pages >= 0)
        return torch.where(valid, locs, torch.full_like(locs, -1))

    def _compressed_raw_to_full_locs_from_page_table(
        self,
        page_table: torch.Tensor,
        raw_indices: torch.Tensor,
        ratio: Literal[4, 128],
    ) -> torch.Tensor:
        valid = raw_indices >= 0
        raw = raw_indices.clamp_min(0).to(torch.long)
        full_positions = raw * int(ratio) + (int(ratio) - 1)
        logical_pages = full_positions.div(self.page_size, rounding_mode="floor")
        offsets = full_positions % self.page_size
        rows = torch.arange(raw.shape[0], dtype=torch.long, device=self.device)[:, None]
        rows = rows.expand_as(raw)
        max_page = max(page_table.shape[1] - 1, 0)
        gathered_pages = page_table[
            rows,
            logical_pages.clamp(max=max_page),
        ].to(torch.long)
        locs = gathered_pages * self.page_size + offsets
        valid = valid & (logical_pages < page_table.shape[1]) & (gathered_pages >= 0)
        return torch.where(valid, locs, torch.full_like(locs, -1))

    def _component_write_locs_from_page_table(
        self,
        component_page_table: torch.Tensor | None,
        positions: torch.Tensor,
        ratio: Literal[4, 128],
    ) -> torch.Tensor:
        if component_page_table is None or positions.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        component_page_size = (
            self.kvcache.c4_component_page_size
            if ratio == 4
            else self.kvcache.c128_component_page_size
        )
        positions_long = positions.to(device=self.device, dtype=torch.long)
        boundary = (positions_long + 1) % ratio == 0
        if not bool(torch.any(boundary)):
            return torch.empty(0, dtype=torch.long, device=self.device)
        raw = positions_long[boundary].div(ratio, rounding_mode="floor")
        logical_pages = raw.div(component_page_size, rounding_mode="floor")
        offsets = raw % component_page_size
        source_rows = torch.arange(positions.numel(), dtype=torch.long, device=self.device)[
            boundary
        ]
        valid = logical_pages < component_page_table.shape[1]
        out = torch.full_like(raw, -1)
        if bool(torch.any(valid)):
            component_pages = component_page_table[
                source_rows[valid],
                logical_pages[valid],
            ].to(torch.long)
            locs = component_pages * component_page_size + offsets[valid]
            out[valid] = torch.where(
                component_pages >= 0,
                locs,
                torch.full_like(locs, -1),
            )
        return out

    def _compressed_raw_to_full_locs(
        self,
        table_indices: torch.Tensor,
        raw_indices: torch.Tensor,
        ratio: Literal[4, 128],
    ) -> torch.Tensor:
        raw_positions = raw_indices * ratio + (ratio - 1)
        return self._gather_full_locs(table_indices, raw_positions)

    def _gather_full_locs(
        self,
        table_indices: torch.Tensor,
        logical_positions: torch.Tensor,
    ) -> torch.Tensor:
        ctx_page_table = get_global_ctx().page_table
        valid = logical_positions >= 0
        clamped = logical_positions.clamp_min(0).to(torch.long)
        rows = table_indices.to(torch.long)[:, None].expand_as(clamped)
        out = ctx_page_table[rows, clamped].to(torch.int32)
        return torch.where(valid, out, torch.full_like(out, -1))









    def _fallback_attention(
        self,
        q: torch.Tensor,
        layer_id: int,
        metadata: DSV4CoreAttentionMetadata,
        compress_ratio: DSV4CompressRatio,
        attn_sink: torch.Tensor | None,
    ) -> torch.Tensor:
        fast = self._sparse_attention_two_source(
            q,
            layer_id,
            metadata,
            compress_ratio,
            attn_sink,
        )
        if fast is not None:
            return fast
        cache = self.kvcache.swa_cache(layer_id).to(q.dtype)
        context_indices = self._context_metadata_for_queries(metadata, q.shape[0], compress_ratio)
        return dsv4_kernel.paged_mqa_attention_fallback(
            q,
            cache,
            context_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
        )

    def _two_source_attention_torch(
        self,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        *,
        compressed_cache: torch.Tensor | None,
        compressed_indices: torch.Tensor | None,
        compressed_lengths: torch.Tensor | None,
        attn_sink: torch.Tensor | None,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        sink = (
            attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)
            if attn_sink is not None
            else None
        )
        for row in range(q.shape[0]):
            parts: list[torch.Tensor] = []
            if (
                compressed_cache is not None
                and compressed_indices is not None
                and compressed_lengths is not None
            ):
                comp_len = max(0, int(compressed_lengths[row].item()))
                comp_idx = compressed_indices[row, :comp_len].to(device=q.device, dtype=torch.long)
                comp_idx = comp_idx[comp_idx >= 0]
                if comp_idx.numel() > 0:
                    parts.append(compressed_cache[comp_idx].to(device=q.device, dtype=q.dtype))

            swa_len = max(0, int(swa_lengths[row].item()))
            swa_idx = swa_indices[row, :swa_len].to(device=q.device, dtype=torch.long)
            swa_idx = swa_idx[swa_idx >= 0]
            if swa_idx.numel() > 0:
                parts.append(swa_cache[swa_idx].to(device=q.device, dtype=q.dtype))

            if not parts:
                out[row].zero_()
                continue

            candidates = torch.cat(parts, dim=0).float()
            scores = torch.einsum("hd,td->ht", q[row].float(), candidates) * self.softmax_scale
            if sink is None:
                attn = torch.softmax(scores, dim=-1)
            else:
                max_score = torch.maximum(scores.max(dim=-1).values, sink)
                exp_scores = torch.exp(scores - max_score[:, None])
                denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
                attn = exp_scores / denom[:, None]
            out[row] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
        return out













    def _sparse_attention_two_source(
        self,
        q: torch.Tensor,
        layer_id: int,
        metadata: DSV4CoreAttentionMetadata,
        compress_ratio: DSV4CompressRatio,
        attn_sink: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rows = q.shape[0]
        if rows == 0:
            return q.new_empty(q.shape)
        self._ensure_swa_metadata_current(metadata, context="sparse attention launch")
        swa_indices_view = metadata.swa_page_indices[:rows]
        swa_lengths_view = metadata.swa_topk_lengths[:rows]
        use_swa_boundary_fast_path = (
            _swa_direct_replay_metadata_fused_enabled()
            and swa_indices_view.device == q.device
            and swa_lengths_view.device == q.device
            and swa_indices_view.dtype == torch.int32
            and swa_lengths_view.dtype == torch.int32
        )
        if use_swa_boundary_fast_path:
            swa_indices = swa_indices_view
            swa_lengths = swa_lengths_view
        else:
            swa_indices = swa_indices_view.to(device=q.device, dtype=torch.int32)
            swa_lengths = swa_lengths_view.to(device=q.device, dtype=torch.int32)
            swa_lengths = swa_lengths.clamp(max=swa_indices.shape[-1])

        compressed_cache = None
        compressed_indices = None
        compressed_lengths = None
        use_compressed_boundary_fast_path = _swa_direct_replay_metadata_fused_enabled()
        if compress_ratio == 4:
            compressed_cache = self.kvcache.c4_cache(layer_id).to(q.dtype)
            compressed_indices_view = metadata.c4_sparse_page_indices[:rows]
            compressed_lengths_view = metadata.c4_sparse_topk_lengths[:rows]
            if (
                use_compressed_boundary_fast_path
                and compressed_indices_view.device == q.device
                and compressed_lengths_view.device == q.device
                and compressed_indices_view.dtype == torch.int32
                and compressed_lengths_view.dtype == torch.int32
                and compressed_indices_view.stride(-1) == 1
                and compressed_lengths_view.is_contiguous()
                and compressed_lengths_view.numel() == rows
            ):
                compressed_indices = compressed_indices_view
                compressed_lengths = compressed_lengths_view
            else:
                compressed_indices = compressed_indices_view.to(
                    device=q.device,
                    dtype=torch.int32,
                )
                if dsv4_kernel.dsv4_optimized_enabled():
                    compressed_lengths = compressed_lengths_view.to(
                        device=q.device,
                        dtype=torch.int32,
                    )
                    compressed_lengths = compressed_lengths.clamp(max=compressed_indices.shape[-1])
                else:
                    compressed_lengths = (compressed_indices >= 0).sum(dim=-1).to(torch.int32)
        elif compress_ratio == 128:
            compressed_cache = self.kvcache.c128_cache(layer_id).to(q.dtype)
            compressed_indices_view = metadata.c128_page_indices[:rows]
            compressed_lengths_view = metadata.c128_topk_lengths_clamp1[:rows]
            if (
                use_compressed_boundary_fast_path
                and compressed_indices_view.device == q.device
                and compressed_lengths_view.device == q.device
                and compressed_indices_view.dtype == torch.int32
                and compressed_lengths_view.dtype == torch.int32
                and compressed_indices_view.stride(-1) == 1
                and compressed_lengths_view.is_contiguous()
                and compressed_lengths_view.numel() == rows
            ):
                compressed_indices = compressed_indices_view
                compressed_lengths = compressed_lengths_view
            else:
                compressed_indices = compressed_indices_view.to(
                    device=q.device,
                    dtype=torch.int32,
                )
                compressed_lengths = (compressed_indices >= 0).sum(dim=-1).to(torch.int32)

        if metadata.max_seqlen_q <= 1:
            splitk_swa_cache = self.kvcache.swa_cache(layer_id).to(q.dtype)
            fast = dsv4_kernel.dsv4_sparse_attention_two_source_splitk_bf16(
                q,
                splitk_swa_cache,
                swa_indices,
                swa_lengths,
                compressed_cache=compressed_cache,
                compressed_indices=compressed_indices,
                compressed_lengths=compressed_lengths,
                softmax_scale=self.softmax_scale,
                attn_sink=attn_sink,
            )
            if fast is not None:
                return fast

        base_swa_cache = self.kvcache.swa_cache(layer_id).to(q.dtype)
        fast = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
            q,
            base_swa_cache,
            swa_indices,
            swa_lengths,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            compressed_lengths=compressed_lengths,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
        )
        if fast is not None:
            return fast
        if compressed_cache is None:
            return None
        fallback_swa_cache = self.kvcache.swa_cache(layer_id).to(q.dtype)
        out = self._two_source_attention_torch(
            q,
            fallback_swa_cache,
            swa_indices,
            swa_lengths,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            compressed_lengths=compressed_lengths,
            attn_sink=attn_sink,
        )
        return out

    def _context_metadata_for_queries(
        self,
        metadata: DSV4CoreAttentionMetadata,
        rows: int,
        compress_ratio: DSV4CompressRatio,
    ) -> dsv4_kernel.DSV4PagedMQAMetadata:
        context_indices = [
            self._context_indices_for_query(metadata, row, compress_ratio) for row in range(rows)
        ]
        return dsv4_kernel.get_paged_mqa_logits_metadata_fallback(
            context_indices,
            device=self.device,
        )

    def _context_indices_for_query(
        self,
        metadata: DSV4CoreAttentionMetadata,
        row: int,
        compress_ratio: DSV4CompressRatio,
    ) -> torch.Tensor:
        pieces = []
        if compress_ratio == 4:
            pieces.append(metadata.c4_sparse_full_indices[row])
        elif compress_ratio == 128:
            pieces.append(metadata.c128_full_indices[row])
        pieces.append(metadata.swa_page_indices[row])
        values = torch.cat([x.reshape(-1) for x in pieces])
        values = values[values >= 0]
        if values.numel() <= 1:
            return values
        seen: set[int] = set()
        ordered = []
        for value in values.tolist():
            ivalue = int(value)
            if ivalue not in seen:
                seen.add(ivalue)
                ordered.append(ivalue)
        return torch.tensor(ordered, dtype=torch.int32, device=values.device)

    def _empty_decode_metadata(self, max_bs: int, max_seq_len: int) -> DSV4AttentionMetadata:
        device = self.device
        table_len = div_ceil(max(max_seq_len, 1), self.page_size)
        topk_width = (
            div_ceil(max(self.index_topk, 1), _PAGE_INDEX_ALIGNMENT) * _PAGE_INDEX_ALIGNMENT
        )
        c128_width = div_ceil(max(div_ceil(max_seq_len, 128), 1), _PAGE_INDEX_ALIGNMENT)
        c128_width *= _PAGE_INDEX_ALIGNMENT
        component_ownership = bool(getattr(self.kvcache, "component_loc_ownership_enabled", False))
        swa_independent = bool(getattr(self.kvcache, "swa_independent_lifecycle_enabled", False))
        has_c4 = any(m.compress_ratio == 4 for m in self.kvcache.layer_mapping)
        has_c128 = any(m.compress_ratio == 128 for m in self.kvcache.layer_mapping)

        def empty_index(width: int) -> torch.Tensor:
            return torch.full((max_bs, width), -1, dtype=torch.int32, device=device)

        swa_index_width = div_ceil(self.window_size, _PAGE_INDEX_ALIGNMENT) * _PAGE_INDEX_ALIGNMENT
        swa_page_indices = empty_index(swa_index_width)
        dummy_loc = 0
        if swa_independent:
            swa_rows = int(self.kvcache.swa_cache(0).shape[0])
            dummy_loc = max(swa_rows - self.page_size, 0)
        swa_page_indices.fill_(dummy_loc)

        c4_sparse_raw_indices = empty_index(topk_width)
        c4_sparse_page_indices = empty_index(topk_width)
        c4_sparse_full_indices = empty_index(topk_width)
        if topk_width > 0:
            c4_sparse_raw_indices.fill_(0)
            c4_sparse_page_indices.fill_(0)
            c4_sparse_full_indices.fill_(0)

        c128_raw_indices = empty_index(c128_width)
        c128_page_indices = empty_index(c128_width)
        c128_full_indices = empty_index(c128_width)
        if c128_width > 0:
            c128_raw_indices.fill_(0)
            c128_page_indices.fill_(0)
            c128_full_indices.fill_(0)

        capture_swa_len = max(min(int(self.window_size), int(swa_index_width)), 1)
        capture_c4_sparse_len = max(min(int(self.index_topk), int(topk_width)), 1)
        capture_c4_raw_len = max(div_ceil(max_seq_len, 4), 1)
        capture_c128_len = max(min(div_ceil(max_seq_len, 128), int(c128_width)), 1)

        c4_out_loc = torch.full((max_bs,), -1, dtype=torch.int32, device=device)
        c128_out_loc = torch.full((max_bs,), -1, dtype=torch.int32, device=device)
        c4_indexer_out_loc = (
            torch.full((max_bs,), -1, dtype=torch.int32, device=device)
            if component_ownership
            else c4_out_loc
        )
        swa_out_loc = (
            torch.full((max_bs,), dummy_loc, dtype=torch.int32, device=device)
            if swa_independent and _swa_direct_replay_metadata_fused_enabled()
            else None
        )
        c4_page_table = (
            torch.zeros((max_bs, table_len), dtype=torch.int32, device=device)
            if component_ownership and has_c4
            else None
        )
        c128_page_table = (
            torch.zeros((max_bs, table_len), dtype=torch.int32, device=device)
            if component_ownership and has_c128
            else None
        )
        c4_indexer_page_table = (
            torch.zeros((max_bs, table_len), dtype=torch.int32, device=device)
            if component_ownership and has_c4
            else None
        )
        core = DSV4CoreAttentionMetadata(
            raw_out_loc=torch.zeros(max_bs, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, table_len), dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(max_bs + 1, dtype=torch.int32, device=device),
            seq_lens=torch.ones(max_bs, dtype=torch.int32, device=device),
            req_seq_lens=torch.ones(max_bs, dtype=torch.int32, device=device),
            extend_lens=torch.ones(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
            req_table_indices=torch.zeros(max_bs, dtype=torch.int32, device=device),
            max_seqlen_q=1,
            max_seqlen_k=max_seq_len,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=torch.full(
                (max_bs,), capture_swa_len, dtype=torch.int32, device=device
            ),
            c4_out_loc=c4_out_loc,
            c128_out_loc=c128_out_loc,
            c4_indexer_out_loc=c4_indexer_out_loc,
            c4_topk_lengths_raw=torch.full(
                (max_bs,), capture_c4_raw_len, dtype=torch.int32, device=device
            ),
            c4_topk_lengths_clamp1=torch.full(
                (max_bs,), capture_c4_raw_len, dtype=torch.int32, device=device
            ),
            c4_sparse_topk_lengths=torch.full(
                (max_bs,), capture_c4_sparse_len, dtype=torch.int32, device=device
            ),
            c4_sparse_raw_indices=c4_sparse_raw_indices,
            c4_sparse_page_indices=c4_sparse_page_indices,
            c4_sparse_full_indices=c4_sparse_full_indices,
            c128_topk_lengths_clamp1=torch.full(
                (max_bs,), capture_c128_len, dtype=torch.int32, device=device
            ),
            c128_raw_indices=c128_raw_indices,
            c128_page_indices=c128_page_indices,
            c128_full_indices=c128_full_indices,
            swa_out_loc=swa_out_loc,
            component_loc_ownership=component_ownership,
            c4_page_table=c4_page_table,
            c128_page_table=c128_page_table,
            c4_indexer_page_table=c4_indexer_page_table,
            swa_ownership_version=self._current_swa_ownership_version(),
            materialized_seq_lens=torch.ones(max_bs, dtype=torch.int32, device=device),
        )
        return DSV4AttentionMetadata(
            core_attn_metadata=core,
            indexer_metadata=DSV4IndexerMetadata(
                self.page_size,
                (
                    core.c4_indexer_page_table
                    if component_ownership and core.c4_indexer_page_table is not None
                    else core.page_table
                ),
                core.c4_topk_lengths_raw,
            ),
            c4_compress_metadata=DSV4CompressMetadata(
                4, core.c4_out_loc, core.seq_lens, core.positions
            ),
            c128_compress_metadata=DSV4CompressMetadata(
                128, core.c128_out_loc, core.seq_lens, core.positions
            ),
        )

    def _copy_metadata_for_replay(
        self,
        dst: DSV4AttentionMetadata,
        src: DSV4AttentionMetadata,
        bs: int,
    ) -> None:
        dst_core = dst.core_metadata
        src_core = src.core_metadata
        self._ensure_swa_metadata_current(src_core, context="CUDA graph replay metadata copy")
        dst_core.component_loc_ownership = src_core.component_loc_ownership
        direct_swa_requested, direct_c4_requested, direct_c128_requested = (
            self._direct_index_groups_for_replay(dst_core, src_core, bs)
        )
        {
            "phase": "decode",
            "rows": int(bs),
            "component_loc_ownership": bool(src_core.component_loc_ownership),
            "direct_swa": bool(direct_swa_requested),
            "direct_c4": bool(direct_c4_requested),
            "direct_c128": bool(direct_c128_requested),
        }
        copied_by_helper = dsv4_kernel.copy_decode_metadata_for_replay(
            dst_raw_out_loc=dst_core.raw_out_loc,
            src_raw_out_loc=src_core.raw_out_loc,
            dst_seq_lens=dst_core.seq_lens,
            src_seq_lens=src_core.seq_lens,
            dst_req_seq_lens=dst_core.req_seq_lens,
            src_req_seq_lens=src_core.req_seq_lens,
            dst_extend_lens=dst_core.extend_lens,
            src_extend_lens=src_core.extend_lens,
            dst_positions=dst_core.positions,
            src_positions=src_core.positions,
            dst_req_table_indices=dst_core.req_table_indices,
            src_req_table_indices=src_core.req_table_indices,
            dst_swa_topk_lengths=dst_core.swa_topk_lengths,
            src_swa_topk_lengths=src_core.swa_topk_lengths,
            dst_c4_topk_lengths_raw=dst_core.c4_topk_lengths_raw,
            src_c4_topk_lengths_raw=src_core.c4_topk_lengths_raw,
            dst_c4_topk_lengths_clamp1=dst_core.c4_topk_lengths_clamp1,
            src_c4_topk_lengths_clamp1=src_core.c4_topk_lengths_clamp1,
            dst_c4_sparse_topk_lengths=dst_core.c4_sparse_topk_lengths,
            src_c4_sparse_topk_lengths=src_core.c4_sparse_topk_lengths,
            dst_c128_topk_lengths_clamp1=dst_core.c128_topk_lengths_clamp1,
            src_c128_topk_lengths_clamp1=src_core.c128_topk_lengths_clamp1,
            dst_cu_seqlens_q=dst_core.cu_seqlens_q,
            src_cu_seqlens_q=src_core.cu_seqlens_q,
            dst_page_table=dst_core.page_table,
            src_page_table=src_core.page_table,
            dst_swa_page_indices=dst_core.swa_page_indices,
            src_swa_page_indices=src_core.swa_page_indices,
            dst_c4_sparse_raw_indices=dst_core.c4_sparse_raw_indices,
            src_c4_sparse_raw_indices=src_core.c4_sparse_raw_indices,
            dst_c4_sparse_page_indices=dst_core.c4_sparse_page_indices,
            src_c4_sparse_page_indices=src_core.c4_sparse_page_indices,
            dst_c4_sparse_full_indices=dst_core.c4_sparse_full_indices,
            src_c4_sparse_full_indices=src_core.c4_sparse_full_indices,
            dst_c128_raw_indices=dst_core.c128_raw_indices,
            src_c128_raw_indices=src_core.c128_raw_indices,
            dst_c128_page_indices=dst_core.c128_page_indices,
            src_c128_page_indices=src_core.c128_page_indices,
            dst_c128_full_indices=dst_core.c128_full_indices,
            src_c128_full_indices=src_core.c128_full_indices,
            rows=bs,
            graph_inputs_bound=self._capture_graph_inputs_bound,
            skip_swa_page_indices=direct_swa_requested,
            skip_c4_sparse_indices=direct_c4_requested,
            skip_c128_indices=direct_c128_requested,
        )
        self._copy_component_page_tables_for_replay(dst_core, src_core, bs)
        direct_swa_done = False
        direct_c4_done = False
        direct_c128_done = False
        if direct_swa_requested or direct_c4_requested or direct_c128_requested:
            direct_done = self._direct_index_metadata_for_replay(
                dst_core,
                src_core,
                bs,
                direct_swa=direct_swa_requested,
                direct_c4=direct_c4_requested,
                direct_c128=direct_c128_requested,
            )
            if direct_done:
                direct_swa_done = direct_swa_requested
                direct_c4_done = direct_c4_requested
                direct_c128_done = direct_c128_requested
            elif (
                src_core.swa_source_elided_for_graph
                or src_core.c4_sparse_source_elided_for_graph
                or src_core.c128_source_elided_for_graph
            ):
                raise RuntimeError(
                    "Optimized DSV4 elided eager index source metadata, but direct "
                    "graph-buffer "
                    "generation failed."
                )
            else:
                self._copy_direct_index_fallback_from_source(
                    dst_core,
                    src_core,
                    bs,
                    copy_swa=direct_swa_requested,
                    copy_c4=direct_c4_requested,
                    copy_c128=direct_c128_requested,
                )
        self._copy_swa_out_loc_for_replay(dst_core, src_core, bs)
        dst_core.swa_ownership_version = int(src_core.swa_ownership_version)
        if copied_by_helper:
            if not self._capture_compressed_locs_in_graph or src_core.component_loc_ownership:
                self._copy_decode_write_locs_for_replay(dst_core, src_core, bs)
            return
        scalar_names = (
            "raw_out_loc",
            "seq_lens",
            "req_seq_lens",
            "extend_lens",
            "positions",
            "req_table_indices",
            "swa_topk_lengths",
            "c4_topk_lengths_raw",
            "c4_topk_lengths_clamp1",
            "c4_sparse_topk_lengths",
            "c128_topk_lengths_clamp1",
        )
        for name in scalar_names:
            if self._capture_graph_inputs_bound and name in {"raw_out_loc", "positions"}:
                continue
            getattr(dst_core, name)[:bs].copy_(getattr(src_core, name)[:bs])
        dst_core.cu_seqlens_q[: bs + 1].copy_(src_core.cu_seqlens_q[: bs + 1])
        self._copy_2d(dst_core.page_table, src_core.page_table, bs, fill=0)
        for name in (
            "swa_page_indices",
            "c4_sparse_raw_indices",
            "c4_sparse_page_indices",
            "c4_sparse_full_indices",
            "c128_raw_indices",
            "c128_page_indices",
            "c128_full_indices",
        ):
            if direct_swa_done and name == "swa_page_indices":
                continue
            if direct_c4_done and name.startswith("c4_sparse_"):
                continue
            if direct_c128_done and name.startswith("c128_"):
                continue
            self._copy_2d(getattr(dst_core, name), getattr(src_core, name), bs, fill=-1)
        if not self._capture_compressed_locs_in_graph:
            self._copy_decode_write_locs_for_replay(dst_core, src_core, bs)

    def _direct_index_groups_for_replay(
        self,
        dst_core: DSV4CoreAttentionMetadata,
        src_core: DSV4CoreAttentionMetadata,
        rows: int,
    ) -> tuple[bool, bool, bool]:
        if not (
            rows > 0
            and src_core.component_loc_ownership
            and dsv4_kernel.dsv4_optimized_enabled()
        ):
            return False, False, False
        if self.device.type != "cuda" or not dsv4_kernel.dsv4_optimized_triton_enabled():
            return False, False, False
        groups = _direct_graph_metadata_groups()
        direct_swa = (
            "swa" in groups
            and src_core.swa_page_indices is not None
            and (
                not bool(getattr(self.kvcache, "swa_independent_lifecycle_enabled", False))
                or _swa_direct_replay_metadata_fused_enabled()
            )
        )
        direct_c4 = (
            "c4" in groups
            and dst_core.c4_page_table is not None
            and src_core.c4_page_table is not None
            and dst_core.c4_sparse_raw_indices is not None
        )
        direct_c128 = (
            "c128" in groups
            and dst_core.c128_page_table is not None
            and src_core.c128_page_table is not None
            and dst_core.c128_raw_indices is not None
        )
        return direct_swa, direct_c4, direct_c128

    def _direct_index_metadata_for_replay(
        self,
        dst_core: DSV4CoreAttentionMetadata,
        src_core: DSV4CoreAttentionMetadata,
        rows: int,
        *,
        direct_swa: bool,
        direct_c4: bool,
        direct_c128: bool,
    ) -> bool:
        approx_bytes = rows * 4 * (
            (dst_core.swa_page_indices.shape[1] if direct_swa else 0)
            + (
                dst_core.c4_sparse_raw_indices.shape[1]
                + dst_core.c4_sparse_page_indices.shape[1]
                + dst_core.c4_sparse_full_indices.shape[1]
                if direct_c4
                else 0
            )
            + (
                dst_core.c128_raw_indices.shape[1]
                + dst_core.c128_page_indices.shape[1]
                + dst_core.c128_full_indices.shape[1]
                if direct_c128
                else 0
            )
        )
        {
            "phase": "decode",
            "rows": int(rows),
            "direct_swa": bool(direct_swa),
            "direct_c4": bool(direct_c4),
            "direct_c128": bool(direct_c128),
            "approx_dst_bytes": int(approx_bytes),
        }
        ok = dsv4_kernel.direct_decode_index_metadata_for_replay(
            ctx_page_table=get_global_ctx().page_table,
            table_indices=src_core.req_table_indices,
            positions=src_core.positions,
            c4_page_table=src_core.c4_page_table,
            c128_page_table=src_core.c128_page_table,
            dst_swa_page_indices=dst_core.swa_page_indices,
            dst_c4_sparse_raw_indices=dst_core.c4_sparse_raw_indices,
            dst_c4_sparse_page_indices=dst_core.c4_sparse_page_indices,
            dst_c4_sparse_full_indices=dst_core.c4_sparse_full_indices,
            dst_c128_raw_indices=dst_core.c128_raw_indices,
            dst_c128_page_indices=dst_core.c128_page_indices,
            dst_c128_full_indices=dst_core.c128_full_indices,
            rows=rows,
            page_size=self.page_size,
            window_size=self.window_size,
            index_topk=self.index_topk,
            direct_swa=direct_swa,
            direct_c4=direct_c4,
            direct_c128=direct_c128,
            swa_full_to_swa_page=getattr(self.kvcache, "_full_to_swa_page", None),
            swa_dummy_token_start=int(
                getattr(self.kvcache, "_dummy_token_start", -1)
            ),
            swa_dummy_page=int(getattr(self.kvcache, "_swa_dummy_page", -1)),
            swa_independent=bool(
                getattr(self.kvcache, "swa_independent_lifecycle_enabled", False)
            ),
        )
        return ok

    def _copy_direct_index_fallback_from_source(
        self,
        dst_core: DSV4CoreAttentionMetadata,
        src_core: DSV4CoreAttentionMetadata,
        rows: int,
        *,
        copy_swa: bool,
        copy_c4: bool,
        copy_c128: bool,
    ) -> None:
        names = []
        if copy_swa:
            names.append("swa_page_indices")
        if copy_c4:
            names.extend(
                (
                    "c4_sparse_raw_indices",
                    "c4_sparse_page_indices",
                    "c4_sparse_full_indices",
                )
            )
        if copy_c128:
            names.extend(
                (
                    "c128_raw_indices",
                    "c128_page_indices",
                    "c128_full_indices",
                )
            )
        for name in names:
            self._copy_2d(getattr(dst_core, name), getattr(src_core, name), rows, fill=-1)

    def _copy_component_page_tables_for_replay(
        self,
        dst_core: DSV4CoreAttentionMetadata,
        src_core: DSV4CoreAttentionMetadata,
        rows: int,
    ) -> None:
        for name in ("c4_page_table", "c128_page_table", "c4_indexer_page_table"):
            dst = getattr(dst_core, name)
            src = getattr(src_core, name)
            if dst is None:
                continue
            if src is None:
                dst[:rows].fill_(-1)
            else:
                self._copy_2d(dst, src, rows, fill=-1)

    def _copy_decode_write_locs_for_replay(
        self,
        dst_core: DSV4CoreAttentionMetadata,
        src_core: DSV4CoreAttentionMetadata,
        rows: int,
    ) -> None:
        {
            "phase": "decode",
            "rows": int(rows),
            "component_loc_ownership": bool(src_core.component_loc_ownership),
        }
        if src_core.component_loc_ownership:
            copied = dsv4_kernel.copy_component_write_locs_for_replay(
                c4_page_table=dst_core.c4_page_table,
                c128_page_table=dst_core.c128_page_table,
                c4_indexer_page_table=dst_core.c4_indexer_page_table,
                positions=dst_core.positions,
                c4_out_loc=dst_core.c4_out_loc,
                c128_out_loc=dst_core.c128_out_loc,
                c4_indexer_out_loc=dst_core.c4_indexer_out_loc,
                rows=rows,
                page_size=self.page_size,
            )
            rows * 4 * 3
            if copied:
                return
            self._copy_masked_compact_write_locs(
                dst_core.c4_out_loc,
                src_core.c4_out_loc,
                src_core.positions,
                rows,
                ratio=4,
            )
            self._copy_masked_compact_write_locs(
                dst_core.c128_out_loc,
                src_core.c128_out_loc,
                src_core.positions,
                rows,
                ratio=128,
            )
            self._copy_masked_compact_write_locs(
                dst_core.c4_indexer_out_loc,
                src_core.c4_indexer_out_loc,
                src_core.positions,
                rows,
                ratio=4,
            )
            return
        dsv4_kernel.copy_masked_compressed_locs(
            src_core.raw_out_loc,
            src_core.positions,
            dst_core.c4_out_loc,
            dst_core.c128_out_loc,
            rows,
        )
        rows * 4 * 2
        if (
            dst_core.c4_indexer_out_loc is not None
            and dst_core.c4_indexer_out_loc is not dst_core.c4_out_loc
            and dst_core.c4_out_loc is not None
        ):
            dst_core.c4_indexer_out_loc[:rows].copy_(dst_core.c4_out_loc[:rows])
            if dst_core.c4_indexer_out_loc.shape[0] > rows:
                dst_core.c4_indexer_out_loc[rows:].fill_(-1)

    def _copy_masked_compact_write_locs(
        self,
        dst: torch.Tensor | None,
        src: torch.Tensor | None,
        positions: torch.Tensor,
        rows: int,
        *,
        ratio: Literal[4, 128],
    ) -> None:
        if dst is None:
            return
        rows = min(rows, dst.shape[0], positions.numel())
        if rows <= 0:
            dst.fill_(-1)
            return
        dst[:rows].fill_(-1)
        if dst.shape[0] > rows:
            dst[rows:].fill_(-1)
        if src is None or src.numel() == 0:
            return
        mask = (positions[:rows] + 1) % ratio == 0
        count = min(int(mask.sum().item()), src.numel())
        if count <= 0:
            return
        target_rows = torch.nonzero(mask, as_tuple=False).flatten()[:count]
        dst[target_rows] = src[:count].to(device=dst.device, dtype=dst.dtype)

    def _copy_2d(self, dst: torch.Tensor, src: torch.Tensor, rows: int, *, fill: int) -> None:
        rows = min(rows, dst.shape[0], src.shape[0])
        if rows <= 0:
            return
        width = min(dst.shape[1], src.shape[1])
        if width > 0:
            dst[:rows, :width].copy_(src[:rows, :width])
        if width < dst.shape[1]:
            dst[:rows, width:].fill_(fill)


__all__ = [
    "DSV4AttentionBackend",
    "DSV4AttentionMetadata",
    "DSV4CompressMetadata",
    "DSV4CoreAttentionMetadata",
    "DSV4IndexerMetadata",
    "DSV4RawDecodeGraphMetadata",
]

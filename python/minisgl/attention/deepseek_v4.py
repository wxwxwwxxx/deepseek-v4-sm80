from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kvcache.deepseek_v4_pool import DeepSeekV4KVCache
from minisgl.utils import (
    div_ceil,
    dsv4_direct_copy_nvtx,
    dsv4_memory_debug,
    dsv4_owner_timing,
    dsv4_prefix_debug,
)

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


DSV4CompressRatio = Literal[0, 4, 128]
_PAGE_INDEX_ALIGNMENT = 64
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_DIRECT_GRAPH_METADATA_ALL_GROUPS = frozenset({"swa", "c4", "c128"})


def _direct_graph_metadata_groups() -> frozenset[str]:
    raw = os.environ.get(
        dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS_ENV,
        "",
    ).strip()
    if not raw or raw.lower() in {"all", "1", "true", "yes", "on"}:
        return _DIRECT_GRAPH_METADATA_ALL_GROUPS
    if raw.lower() in {"none", "0", "false", "no", "off"}:
        return frozenset()
    normalized = raw.replace("+", ",").replace(";", ",").replace(" ", ",")
    groups = frozenset(token for token in normalized.lower().split(",") if token)
    invalid = groups - _DIRECT_GRAPH_METADATA_ALL_GROUPS
    if invalid:
        invalid_text = ",".join(sorted(invalid))
        raise RuntimeError(
            f"Unsupported {dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS_ENV} "
            f"value {raw!r}; invalid groups: {invalid_text}."
        )
    return groups


DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH_ENV = (
    "MINISGL_DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH"
)
DSV4_DEBUG_ATTENTION_COMPONENTS_ENV = "MINISGL_DSV4_PREFIX_DEBUG_ATTENTION_COMPONENTS"
DSV4_SWA_INDEX_BOUNDS_DEBUG_ENV = "MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG"
DSV4_SPARSE_SYNC_DEBUG_ENV = "MINISGL_DSV4_SPARSE_SYNC_DEBUG"


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


def _tensor_nbytes(x: torch.Tensor | None, rows: int | None = None) -> int:
    if x is None:
        return 0
    if rows is None or x.dim() == 0:
        return int(x.numel() * x.element_size())
    rows = max(0, min(int(rows), int(x.shape[0])))
    if x.dim() == 1:
        return int(rows * x.element_size())
    return int(rows * x[0].numel() * x.element_size())


def _record_metadata_counter(
    label: str,
    *,
    value: int = 1,
    phase: str,
    rows: int,
    padded_rows: int | None = None,
    field: str | None = None,
    stable: str | None = None,
) -> None:
    metadata: dict[str, int | str] = {"phase": phase, "rows": int(rows)}
    if padded_rows is not None:
        metadata["padded_rows"] = int(padded_rows)
    if field is not None:
        metadata["field"] = field
    if stable is not None:
        metadata["stable"] = stable
    dsv4_owner_timing.record_counter(label, metadata, value=int(value))


def _debug_activations_enabled() -> bool:
    recorder = dsv4_prefix_debug.get_dsv4_prefix_debug_recorder()
    return recorder is not None and bool(getattr(recorder, "capture_activations", False))


def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _debug_attention_components_enabled() -> bool:
    return (
        _debug_activations_enabled()
        and os.environ.get(DSV4_DEBUG_ATTENTION_COMPONENTS_ENV, "").strip().lower()
        in _TRUE_ENV_VALUES
    )


def _capture_debug_activation(
    name: str,
    tensor: torch.Tensor,
    row_indices: torch.Tensor | None = None,
) -> None:
    try:
        batch = get_global_ctx().batch
    except Exception:
        batch = None
    dsv4_prefix_debug.capture_dsv4_activation(name, tensor, batch, row_indices=row_indices)


def _debug_topk_scores(logits: torch.Tensor, raw_indices: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or raw_indices.numel() == 0:
        return logits.new_empty(raw_indices.shape)
    max_col = max(logits.shape[1] - 1, 0)
    gather = raw_indices.to(device=logits.device, dtype=torch.long).clamp(min=0, max=max_col)
    scores = torch.gather(logits, 1, gather)
    return torch.where(
        raw_indices.to(device=logits.device) >= 0, scores, torch.full_like(scores, float("-inf"))
    )


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
    component_loc_ownership: bool = False
    c4_page_table: torch.Tensor | None = None
    c128_page_table: torch.Tensor | None = None
    c4_indexer_page_table: torch.Tensor | None = None
    swa_source_elided_for_graph: bool = False
    c4_sparse_source_elided_for_graph: bool = False
    c128_source_elided_for_graph: bool = False

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
        self._capture_compressed_locs_in_graph_disabled_by_env = False
        self._capture_compressed_locs_in_graph_component_guarded = False
        self._component_page_table_cache_width = 0
        self._component_page_table_cache_rows = 0
        self._component_page_table_cache_has_c4 = False
        self._component_page_table_cache_has_c128 = False
        self._component_page_table_cache_c4: torch.Tensor | None = None
        self._component_page_table_cache_c128: torch.Tensor | None = None
        self._component_page_table_cache_indexer: torch.Tensor | None = None
        self._component_page_table_cache_signatures: dict[int, tuple[int, ...]] = {}
        dsv4_kernel.warmup_indexer_fp8_backend(self.device)

    @property
    def capture_compressed_locs_in_graph(self) -> bool:
        return self._capture_compressed_locs_in_graph

    @property
    def capture_compressed_locs_in_graph_disabled_by_env(self) -> bool:
        return self._capture_compressed_locs_in_graph_disabled_by_env

    @property
    def capture_compressed_locs_in_graph_component_guarded(self) -> bool:
        return self._capture_compressed_locs_in_graph_component_guarded

    def get_layer_compress_ratio(self, layer_id: int) -> DSV4CompressRatio:
        return self.kvcache.get_layer_mapping(layer_id).compress_ratio

    def prepare_metadata(self, batch: Batch) -> None:
        batch.attn_metadata = self._build_metadata(batch)

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
            dsv4_kernel.store_swa_fallback(self.kvcache, layer_id, k, batch.out_loc)
        return self._fallback_attention(q, layer_id, metadata.core_metadata, ratio, attn_sink)

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
        self._capture_indexer_select_debug(layer_id, out, seq_lens, page_table)
        core = metadata.core_metadata
        core.c4_sparse_raw_indices = self._merge_indexer_rows(
            core.c4_sparse_raw_indices,
            out.topk.raw_indices,
        )
        core.c4_sparse_page_indices = self._merge_indexer_rows(
            core.c4_sparse_page_indices,
            out.topk.page_indices,
        )
        core.c4_sparse_full_indices = self._merge_indexer_rows(
            core.c4_sparse_full_indices,
            out.topk.full_indices,
        )
        if out.topk.topk_lens is not None:
            core.c4_sparse_topk_lengths = self._merge_indexer_lengths(
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
        self._capture_indexer_select_debug(layer_id, out, seq_lens, page_table)
        core = metadata.core_metadata
        core.c4_sparse_raw_indices = self._merge_indexer_rows(
            core.c4_sparse_raw_indices,
            out.topk.raw_indices,
        )
        core.c4_sparse_page_indices = self._merge_indexer_rows(
            core.c4_sparse_page_indices,
            out.topk.page_indices,
        )
        core.c4_sparse_full_indices = self._merge_indexer_rows(
            core.c4_sparse_full_indices,
            out.topk.full_indices,
        )
        if out.topk.topk_lens is not None:
            core.c4_sparse_topk_lengths = self._merge_indexer_lengths(
                core.c4_sparse_topk_lengths,
                out.topk.topk_lens,
            )
        return out

    def _capture_indexer_select_debug(
        self,
        layer_id: int,
        out: dsv4_kernel.DSV4IndexerSelectOutput,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        if (
            dsv4_memory_debug.marlin_wna16_layer2_owner_probe_enabled()
            and int(layer_id) == 2
            and not _cuda_graph_capture_active()
        ):
            stage = "layer2_indexer_select"
            tensors = {
                "seq_lens": seq_lens,
                "page_table": page_table,
                "logits": out.logits,
                "topk_raw_indices": out.topk.raw_indices,
                "topk_page_indices": out.topk.page_indices,
                "topk_full_indices": out.topk.full_indices,
                "topk_lens": out.topk.topk_lens,
                "topk_scores": _debug_topk_scores(out.logits, out.topk.raw_indices),
            }
            dsv4_memory_debug.record_owner_tensors(
                owner_prefix="dsv4.layer2_owner_probe.indexer_select",
                stage=stage,
                tensors=tensors,
                include_integrity=True,
                extra={"layer_id": int(layer_id), "backend": out.backend},
            )
        if not _debug_activations_enabled():
            return
        prefix = f"layer{layer_id}.indexer_select"
        _capture_debug_activation(f"{prefix}.seq_lens", seq_lens)
        _capture_debug_activation(f"{prefix}.page_table", page_table)
        _capture_debug_activation(f"{prefix}.logits", out.logits)
        _capture_debug_activation(f"{prefix}.topk_raw_indices", out.topk.raw_indices)
        _capture_debug_activation(f"{prefix}.topk_page_indices", out.topk.page_indices)
        _capture_debug_activation(f"{prefix}.topk_full_indices", out.topk.full_indices)
        if out.topk.topk_lens is not None:
            _capture_debug_activation(f"{prefix}.topk_lens", out.topk.topk_lens)
        _capture_debug_activation(
            f"{prefix}.topk_scores",
            _debug_topk_scores(out.logits, out.topk.raw_indices),
        )

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

    def _merge_indexer_lengths(self, current: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        lens = values.to(device=current.device, dtype=current.dtype).reshape(-1)
        if current.shape == lens.shape:
            return lens
        out = torch.zeros_like(current)
        rows = min(out.numel(), lens.numel())
        if rows > 0:
            out[:rows] = lens[:rows]
        return out

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        self.max_graph_bs = max(bs_list) if bs_list else 0
        self._capture_graph_inputs_bound = False
        self._capture_compressed_locs_in_graph = False
        self._capture_compressed_locs_in_graph_disabled_by_env = False
        self._capture_compressed_locs_in_graph_component_guarded = False
        if self.max_graph_bs == 0:
            return
        self.capture = self._empty_decode_metadata(self.max_graph_bs, max_seq_len)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert self.capture is not None
        assert batch.size in self.capture_bs
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
        disable_capture_locs = (
            os.environ.get(DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH_ENV, "").strip().lower()
            in _TRUE_ENV_VALUES
        )
        self._capture_compressed_locs_in_graph_disabled_by_env = disable_capture_locs
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
            and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_COMPRESS_STORE")
        )

    def stage_capture_metadata_for_graph(self, batch: Batch) -> None:
        if not self._capture_compressed_locs_in_graph or self.capture is None:
            return
        if getattr(batch, "attn_metadata", None) is not self.capture:
            return
        core = self.capture.core_metadata
        with dsv4_direct_copy_nvtx(
            f"static_graph_input_updates.capture_compressed_locs.bs{batch.size}.padded{batch.padded_size}",
            raw_out_loc=core.raw_out_loc,
            positions=core.positions,
        ):
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
        assert isinstance(metadata, DSV4AttentionMetadata)
        if metadata is self.capture:
            return
        self._copy_metadata_for_replay(self.capture, metadata, batch.padded_size)

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
            and dsv4_kernel.dsv4_env_flag(
                dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE
            )
            and group in _direct_graph_metadata_groups()
        ):
            return False
        if group == "swa" and bool(
            getattr(self.kvcache, "swa_independent_lifecycle_enabled", False)
        ):
            return False
        padded_size = int(getattr(batch, "padded_size", batch.size))
        if padded_size <= 0 or padded_size not in self.capture_bs:
            return False
        return self.device.type == "cuda" and dsv4_kernel.dsv4_sm80_triton_enabled(
            dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE
        )

    def _empty_index_source_placeholder(self, rows: int) -> torch.Tensor:
        return torch.full((int(rows), 1), -1, dtype=torch.int32, device=self.device)

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
        cu_seqlens_q = F.pad(extend_lens.cumsum(dim=0), (1, 0))
        component_ownership = bool(getattr(self.kvcache, "component_loc_ownership_enabled", False))
        swa_independent = bool(getattr(self.kvcache, "swa_independent_lifecycle_enabled", False))
        swa_page_table = (
            self._make_swa_page_tables(reqs, max_seqlen_k) if swa_independent else None
        )

        with dsv4_owner_timing.maybe_host_range(
            "dsv4.metadata.build.table_indices",
            {**timing_base, "max_seqlen_k": int(max_seqlen_k)},
        ):
            table_indices = torch.empty(positions.numel(), dtype=torch.int32, device=device)
            offset = 0
            for req, length in zip(reqs, extend_lens_list):
                table_indices[offset : offset + length].fill_(req.table_idx)
                offset += length
        assert offset == positions.numel()

        seq_lens = positions + 1
        with dsv4_owner_timing.maybe_cuda_range(
            "dsv4.metadata.decode.make_component_page_tables",
            {**timing_base, "max_seqlen_k": int(max_seqlen_k)},
        ):
            if component_ownership and batch.is_decode and dsv4_kernel.dsv4_env_flag(
                dsv4_kernel.DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_TOGGLE
            ):
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
        if (
            batch.is_decode
            and not swa_independent
            and dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_DECODE_METADATA_DEFOREST_TOGGLE)
            and not self._should_elide_index_source_for_graph(
                batch,
                component_ownership=component_ownership,
                enabled=has_c4,
                group="c4",
            )
        ):
            with dsv4_owner_timing.maybe_cuda_range(
                "dsv4.metadata.decode.deforest",
                {**timing_base, "max_seqlen_k": int(max_seqlen_k)},
            ):
                deforested = dsv4_kernel.decode_metadata_deforest_fallback(
                    get_global_ctx().page_table,
                    table_indices,
                    positions,
                    page_size=self.page_size,
                    max_seqlen_k=max_seqlen_k,
                    window_size=self.window_size,
                    index_topk=self.index_topk,
                    alignment=_PAGE_INDEX_ALIGNMENT,
                    c4_page_table=c4_page_table if component_ownership and has_c4 else None,
                    c128_page_table=(c128_page_table if component_ownership and has_c128 else None),
                    component_loc_ownership=component_ownership,
                )
            if deforested is None:
                raise RuntimeError(
                    f"{dsv4_kernel.DSV4_SM80_DECODE_METADATA_DEFOREST_TOGGLE}=1 "
                    "requires the SM80 Triton decode metadata helper for decode batches."
                )

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
            with dsv4_owner_timing.maybe_cuda_range(
                "dsv4.metadata.decode.make_page_table",
                {**timing_base, "max_seqlen_k": int(max_seqlen_k)},
            ):
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
                with dsv4_owner_timing.maybe_cuda_range(
                    "dsv4.metadata.decode.make_swa_indices",
                    {**timing_base, "window_size": int(self.window_size)},
                ):
                    if swa_page_table is not None:
                        swa_page_indices = self._make_swa_indices_from_page_table(
                            swa_page_table,
                            positions,
                        )
                    else:
                        swa_page_indices = self._make_swa_indices(table_indices, positions)
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
                with dsv4_owner_timing.maybe_cuda_range(
                    "dsv4.metadata.decode.make_c4_sparse_indices",
                    {**timing_base, "index_topk": int(self.index_topk)},
                ):
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
            else:
                with dsv4_owner_timing.maybe_cuda_range(
                    "dsv4.metadata.decode.make_c128_indices",
                    {**timing_base, "max_seqlen_k": int(max_seqlen_k)},
                ):
                    c128_raw_indices, c128_page_indices, c128_full_indices = (
                        self._make_all_compressed_indices(
                            table_indices,
                            c128_lengths_raw,
                            128,
                            component_page_table=c128_page_table,
                        )
                    )

        with dsv4_owner_timing.maybe_cuda_range(
            "dsv4.metadata.decode.make_write_locs",
            {**timing_base, "component_tables": bool(component_tables is not None)},
        ):
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
            component_loc_ownership=component_ownership,
            c4_page_table=c4_page_table,
            c128_page_table=c128_page_table,
            c4_indexer_page_table=c4_indexer_page_table,
            swa_source_elided_for_graph=swa_source_elided,
            c4_sparse_source_elided_for_graph=c4_sparse_source_elided,
            c128_source_elided_for_graph=c128_source_elided,
        )
        self._record_metadata_build_bytes(batch, core)
        self._record_marlin_wna16_metadata_owners(batch, core)

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
        if dsv4_owner_timing.enabled():
            base = {"phase": "decode", "rows": int(rows), "table_width": int(table_len)}
            dsv4_owner_timing.record_counter(
                "dsv4.component_page_table_cache.rows",
                {**base, "status": "dirty"},
                value=dirty,
            )
            dsv4_owner_timing.record_counter(
                "dsv4.component_page_table_cache.rows",
                {**base, "status": "clean"},
                value=clean,
            )

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

        if dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY_TOGGLE
        ):
            oracle = self._make_component_page_tables_uncached(reqs, max_seq_len)
            for name, got, expected in zip(
                ("c4_page_table", "c128_page_table", "c4_indexer_page_table"),
                result,
                oracle,
            ):
                if got is None or expected is None:
                    if got is not expected:
                        raise RuntimeError(
                            "DSV4 component page-table cache verify mismatch: "
                            f"{name} got={got is not None}, expected={expected is not None}"
                        )
                    continue
                if got.shape != expected.shape or not bool(torch.equal(got, expected)):
                    raise RuntimeError(
                        "DSV4 component page-table cache verify mismatch: "
                        f"{name} got_shape={tuple(got.shape)} "
                        f"expected_shape={tuple(expected.shape)}"
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
        dsv4_memory_debug.record_owner_tensors(
            owner_prefix="attention.dsv4.component_page_table_cache",
            stage="ensure_component_page_table_cache",
            tensors={
                "c4": self._component_page_table_cache_c4,
                "c128": self._component_page_table_cache_c128,
                "indexer": self._component_page_table_cache_indexer,
            },
            extra={
                "rows": int(rows),
                "width": int(width),
                "has_c4": bool(has_c4),
                "has_c128": bool(has_c128),
            },
        )

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

    def _make_swa_page_tables(self, reqs, max_seq_len: int) -> torch.Tensor:
        table_len = div_ceil(max(max_seq_len, 1), self.page_size)
        rows = sum(req.extend_len for req in reqs)
        chunks: list[torch.Tensor] = []
        for req in reqs:
            row = self._build_swa_page_table_row(req, table_len)
            for _ in range(req.extend_len):
                chunks.append(row)
        if not chunks:
            return torch.full((rows, table_len), -1, dtype=torch.int32, device=self.device)
        return torch.stack(chunks).to(torch.int32)

    def _build_swa_page_table_row(self, req, table_len: int) -> torch.Tensor:
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

    def _record_metadata_build_bytes(
        self,
        batch: Batch,
        core: DSV4CoreAttentionMetadata,
    ) -> None:
        if not dsv4_owner_timing.enabled():
            return
        phase = batch.phase
        rows = int(core.positions.numel())
        field_stability = {
            "raw_out_loc": "per-token",
            "positions": "per-token",
            "page_table": "per-request",
            "cu_seqlens_q": "per-bucket",
            "seq_lens": "per-token",
            "req_seq_lens": "per-request",
            "extend_lens": "per-request",
            "req_table_indices": "per-token",
            "swa_page_indices": "per-token",
            "swa_topk_lengths": "per-token",
            "c4_topk_lengths_raw": "per-token",
            "c4_topk_lengths_clamp1": "per-token",
            "c4_sparse_topk_lengths": "per-token",
            "c4_sparse_raw_indices": "per-token",
            "c4_sparse_page_indices": "per-token",
            "c4_sparse_full_indices": "per-token",
            "c128_topk_lengths_clamp1": "per-token",
            "c128_raw_indices": "per-prefix-hit",
            "c128_page_indices": "per-prefix-hit",
            "c128_full_indices": "per-prefix-hit",
            "c4_page_table": "per-request",
            "c128_page_table": "per-request",
            "c4_indexer_page_table": "per-request",
            "c4_out_loc": "per-token",
            "c128_out_loc": "per-token",
            "c4_indexer_out_loc": "per-token",
        }
        for field, stable in field_stability.items():
            tensor = getattr(core, field)
            byte_count = _tensor_nbytes(tensor)
            if byte_count <= 0:
                continue
            _record_metadata_counter(
                "dsv4.metadata_build.bytes",
                value=byte_count,
                phase=phase,
                rows=rows,
                field=field,
                stable=stable,
            )
            _record_metadata_counter(
                "dsv4.metadata_build.calls",
                phase=phase,
                rows=rows,
                field=field,
                stable=stable,
            )

    def _record_marlin_wna16_metadata_owners(
        self,
        batch: Batch,
        core: DSV4CoreAttentionMetadata,
    ) -> None:
        if not dsv4_memory_debug.marlin_wna16_release_ledger_enabled():
            return
        stage = (
            f"attention_metadata_{batch.phase}"
            f"_bs{int(batch.size)}"
            f"_padded{int(getattr(batch, 'padded_size', batch.size))}"
        )
        dsv4_memory_debug.record_owner_tensors(
            owner_prefix="attention.dsv4.metadata",
            stage=stage,
            tensors={
                "raw_out_loc": core.raw_out_loc,
                "page_table": core.page_table,
                "cu_seqlens_q": core.cu_seqlens_q,
                "seq_lens": core.seq_lens,
                "req_seq_lens": core.req_seq_lens,
                "extend_lens": core.extend_lens,
                "positions": core.positions,
                "req_table_indices": core.req_table_indices,
                "swa_page_indices": core.swa_page_indices,
                "swa_topk_lengths": core.swa_topk_lengths,
                "c4_out_loc": core.c4_out_loc,
                "c128_out_loc": core.c128_out_loc,
                "c4_indexer_out_loc": core.c4_indexer_out_loc,
                "c4_topk_lengths_raw": core.c4_topk_lengths_raw,
                "c4_topk_lengths_clamp1": core.c4_topk_lengths_clamp1,
                "c4_sparse_topk_lengths": core.c4_sparse_topk_lengths,
                "c4_sparse_raw_indices": core.c4_sparse_raw_indices,
                "c4_sparse_page_indices": core.c4_sparse_page_indices,
                "c4_sparse_full_indices": core.c4_sparse_full_indices,
                "c128_topk_lengths_clamp1": core.c128_topk_lengths_clamp1,
                "c128_raw_indices": core.c128_raw_indices,
                "c128_page_indices": core.c128_page_indices,
                "c128_full_indices": core.c128_full_indices,
                "c4_page_table": core.c4_page_table,
                "c128_page_table": core.c128_page_table,
                "c4_indexer_page_table": core.c4_indexer_page_table,
            },
            extra={
                "component_loc_ownership": bool(core.component_loc_ownership),
                "max_seqlen_q": int(core.max_seqlen_q),
                "max_seqlen_k": int(core.max_seqlen_k),
                "swa_source_elided_for_graph": bool(core.swa_source_elided_for_graph),
                "c4_sparse_source_elided_for_graph": bool(
                    core.c4_sparse_source_elided_for_graph
                ),
                "c128_source_elided_for_graph": bool(core.c128_source_elided_for_graph),
            },
        )

    def _record_replay_copy_bytes(
        self,
        core: DSV4CoreAttentionMetadata,
        rows: int,
        *,
        fused: bool,
        skip_swa: bool = False,
        skip_c4_sparse: bool = False,
        skip_c128: bool = False,
    ) -> None:
        if not dsv4_owner_timing.enabled():
            return
        field_stability = {
            "raw_out_loc": "per-token",
            "seq_lens": "per-token",
            "req_seq_lens": "per-request",
            "extend_lens": "per-request",
            "positions": "per-token",
            "req_table_indices": "per-token",
            "swa_topk_lengths": "per-token",
            "c4_topk_lengths_raw": "per-token",
            "c4_topk_lengths_clamp1": "per-token",
            "c4_sparse_topk_lengths": "per-token",
            "c128_topk_lengths_clamp1": "per-token",
            "cu_seqlens_q": "per-bucket",
            "page_table": "per-request",
            "swa_page_indices": "per-token",
            "c4_sparse_raw_indices": "per-token",
            "c4_sparse_page_indices": "per-token",
            "c4_sparse_full_indices": "per-token",
            "c128_raw_indices": "per-prefix-hit",
            "c128_page_indices": "per-prefix-hit",
            "c128_full_indices": "per-prefix-hit",
        }
        for field, stable in field_stability.items():
            if skip_swa and field == "swa_page_indices":
                continue
            if skip_c4_sparse and field in {
                "c4_sparse_raw_indices",
                "c4_sparse_page_indices",
                "c4_sparse_full_indices",
            }:
                continue
            if skip_c128 and field in {
                "c128_raw_indices",
                "c128_page_indices",
                "c128_full_indices",
            }:
                continue
            tensor = getattr(core, field)
            byte_count = _tensor_nbytes(tensor, rows)
            if field == "cu_seqlens_q":
                byte_count = min(rows + 1, tensor.numel()) * tensor.element_size()
            if byte_count <= 0:
                continue
            _record_metadata_counter(
                "dsv4.replay_metadata_copy.bytes",
                value=byte_count,
                phase="decode",
                rows=rows,
                field=field,
                stable=stable,
            )
            _record_metadata_counter(
                "dsv4.replay_metadata_copy.calls",
                phase="decode",
                rows=rows,
                field=field,
                stable=f"{stable};{'fused' if fused else 'fallback'}",
            )

    def _record_direct_graph_metadata_bytes(
        self,
        rows: int,
        *,
        direct_swa: bool,
        direct_c4: bool,
        direct_c128: bool,
    ) -> None:
        if not dsv4_owner_timing.enabled():
            return
        fields = []
        if direct_swa:
            fields.append("swa_page_indices")
        if direct_c4:
            fields.extend(
                (
                    "c4_sparse_raw_indices",
                    "c4_sparse_page_indices",
                    "c4_sparse_full_indices",
                )
            )
        if direct_c128:
            fields.extend(
                (
                    "c128_raw_indices",
                    "c128_page_indices",
                    "c128_full_indices",
                )
            )
        for field in fields:
            tensor = (
                getattr(self.capture.core_metadata, field) if self.capture is not None else None
            )
            byte_count = _tensor_nbytes(tensor, rows)
            if byte_count <= 0:
                continue
            _record_metadata_counter(
                "dsv4.direct_graph_metadata.bytes",
                value=byte_count,
                phase="decode",
                rows=rows,
                field=field,
                stable="per-token;direct_dst",
            )
            _record_metadata_counter(
                "dsv4.direct_graph_metadata.calls",
                phase="decode",
                rows=rows,
                field=field,
                stable="per-token;direct_dst",
            )

    def _record_component_write_loc_copy(self, rows: int, *, backend: str) -> None:
        if not dsv4_owner_timing.enabled():
            return
        for field in ("c4_out_loc", "c128_out_loc", "c4_indexer_out_loc"):
            _record_metadata_counter(
                "dsv4.replay_metadata_copy.bytes",
                value=rows * 4,
                phase="decode",
                rows=rows,
                field=field,
                stable="per-token",
            )
            _record_metadata_counter(
                "dsv4.replay_metadata_copy.calls",
                phase="decode",
                rows=rows,
                field=field,
                stable=f"per-token;{backend}",
            )

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

    def _debug_attention_rows(
        self,
        metadata: DSV4CoreAttentionMetadata,
        rows: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not _debug_activations_enabled() or rows <= 0:
            return None
        try:
            batch_rows = int(metadata.extend_lens.numel())
            row_indices = metadata.get_last_indices(batch_rows).to(device=device, dtype=torch.long)
            row_indices = row_indices[(row_indices >= 0) & (row_indices < rows)]
            if row_indices.numel() > 0:
                return row_indices
        except Exception:
            pass
        count = min(rows, 4)
        return torch.arange(count, dtype=torch.long, device=device)

    def _debug_check_swa_index_bounds(
        self,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        cache_rows: int,
        *,
        layer_id: int,
    ) -> None:
        if _cuda_graph_capture_active():
            return
        if (
            os.environ.get(DSV4_SWA_INDEX_BOUNDS_DEBUG_ENV, "").strip().lower()
            not in _TRUE_ENV_VALUES
        ):
            return
        if swa_indices.numel() == 0:
            return
        width = int(swa_indices.shape[-1])
        lengths = swa_lengths.to(device=swa_indices.device, dtype=torch.long).clamp(
            min=0,
            max=width,
        )
        cols = torch.arange(width, device=swa_indices.device, dtype=torch.long)
        active = cols[None, :] < lengths[:, None]
        bad = active & ((swa_indices < 0) | (swa_indices >= int(cache_rows)))
        if not bool(torch.any(bad).item()):
            return
        rows, cols = torch.where(bad)
        row = int(rows[0].item())
        col = int(cols[0].item())
        value = int(swa_indices[row, col].item())
        length = int(lengths[row].item())
        raise RuntimeError(
            "DSV4 SWA index out of bounds before sparse attention: "
            f"layer={layer_id}, row={row}, col={col}, value={value}, "
            f"length={length}, cache_rows={int(cache_rows)}"
        )

    def _debug_check_cache_index_bounds(
        self,
        indices: torch.Tensor | None,
        lengths: torch.Tensor | None,
        cache_rows: int,
        *,
        layer_id: int,
        label: str,
    ) -> None:
        if _cuda_graph_capture_active():
            return
        if (
            os.environ.get(DSV4_SWA_INDEX_BOUNDS_DEBUG_ENV, "").strip().lower()
            not in _TRUE_ENV_VALUES
        ):
            return
        if indices is None or indices.numel() == 0:
            return
        width = int(indices.shape[-1])
        if lengths is None:
            active = indices >= 0
            length_values = active.sum(dim=-1).to(dtype=torch.long)
        else:
            length_values = lengths.to(device=indices.device, dtype=torch.long).clamp(
                min=0,
                max=width,
            )
            cols = torch.arange(width, device=indices.device, dtype=torch.long)
            active = cols[None, :] < length_values[:, None]
        bad = active & ((indices < 0) | (indices >= int(cache_rows)))
        if not bool(torch.any(bad).item()):
            return
        rows, cols = torch.where(bad)
        row = int(rows[0].item())
        col = int(cols[0].item())
        value = int(indices[row, col].item())
        length = int(length_values[row].item())
        raise RuntimeError(
            "DSV4 cache index out of bounds before sparse attention: "
            f"label={label}, layer={layer_id}, row={row}, col={col}, "
            f"value={value}, length={length}, cache_rows={int(cache_rows)}"
        )

    def _debug_sync_sparse_attention(
        self,
        *,
        backend: str,
        layer_id: int,
        rows: int,
        metadata: DSV4CoreAttentionMetadata,
        compress_ratio: DSV4CompressRatio,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        compressed_indices: torch.Tensor | None,
        compressed_lengths: torch.Tensor | None,
    ) -> None:
        if _cuda_graph_capture_active():
            return
        if (
            os.environ.get(DSV4_SPARSE_SYNC_DEBUG_ENV, "").strip().lower()
            not in _TRUE_ENV_VALUES
        ):
            return
        try:
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            def _range(tensor: torch.Tensor | None) -> tuple[int | None, int | None]:
                if tensor is None or tensor.numel() == 0:
                    return None, None
                valid = tensor[tensor >= 0]
                if valid.numel() == 0:
                    return None, None
                return int(valid.min().item()), int(valid.max().item())

            swa_min, swa_max = _range(swa_indices)
            comp_min, comp_max = _range(compressed_indices)
            max_swa_len = (
                int(swa_lengths.max().item()) if swa_lengths.numel() else 0
            )
            max_comp_len = (
                int(compressed_lengths.max().item())
                if compressed_lengths is not None and compressed_lengths.numel()
                else 0
            )
            raise RuntimeError(
                "DSV4 sparse attention failed during debug synchronize: "
                f"backend={backend}, layer={layer_id}, ratio={compress_ratio}, "
                f"rows={rows}, max_seqlen_q={int(metadata.max_seqlen_q)}, "
                f"max_seqlen_k={int(metadata.max_seqlen_k)}, "
                f"swa_len_max={max_swa_len}, swa_range=({swa_min}, {swa_max}), "
                f"compressed_len_max={max_comp_len}, "
                f"compressed_range=({comp_min}, {comp_max})"
            ) from exc

    def _capture_attention_debug(
        self,
        layer_id: int,
        q: torch.Tensor,
        metadata: DSV4CoreAttentionMetadata,
        rows: int,
        *,
        swa_indices: torch.Tensor,
        swa_lengths: torch.Tensor,
        swa_cache: torch.Tensor,
        compressed_cache: torch.Tensor | None,
        compressed_indices: torch.Tensor | None,
        compressed_lengths: torch.Tensor | None,
        compress_ratio: DSV4CompressRatio,
        attn_sink: torch.Tensor | None,
        merged_output: torch.Tensor,
    ) -> None:
        if not _debug_activations_enabled():
            return
        row_indices = self._debug_attention_rows(metadata, rows, q.device)
        prefix = f"layer{layer_id}.attention_backend"
        _capture_debug_activation(f"{prefix}.swa_selected_full_indices", swa_indices, row_indices)
        _capture_debug_activation(f"{prefix}.swa_lengths", swa_lengths, row_indices)
        if compressed_indices is not None:
            name = "c4" if compress_ratio == 4 else "c128"
            _capture_debug_activation(
                f"{prefix}.{name}_selected_page_indices",
                compressed_indices,
                row_indices,
            )
            if compressed_lengths is not None:
                _capture_debug_activation(
                    f"{prefix}.{name}_lengths",
                    compressed_lengths,
                    row_indices,
                )
        _capture_debug_activation(
            f"{prefix}.merged_attention_output_before_wo",
            merged_output,
            row_indices,
        )
        if not _debug_attention_components_enabled() or row_indices is None:
            return
        if row_indices.numel() == 0:
            return
        q_sel = q.index_select(0, row_indices)
        swa_indices_sel = swa_indices.index_select(0, row_indices)
        swa_lengths_sel = swa_lengths.index_select(0, row_indices)
        swa_out = self._two_source_attention_torch(
            q_sel,
            swa_cache,
            swa_indices_sel,
            swa_lengths_sel,
            compressed_cache=None,
            compressed_indices=None,
            compressed_lengths=None,
            attn_sink=attn_sink,
        )
        _capture_debug_activation(f"{prefix}.swa_attention_output", swa_out)
        if (
            compressed_cache is not None
            and compressed_indices is not None
            and compressed_lengths is not None
        ):
            compressed_indices_sel = compressed_indices.index_select(0, row_indices)
            compressed_lengths_sel = compressed_lengths.index_select(0, row_indices)
            empty_swa_indices = swa_indices_sel[:, :1]
            empty_swa_lengths = torch.zeros_like(swa_lengths_sel)
            compressed_out = self._two_source_attention_torch(
                q_sel,
                swa_cache,
                empty_swa_indices,
                empty_swa_lengths,
                compressed_cache=compressed_cache,
                compressed_indices=compressed_indices_sel,
                compressed_lengths=compressed_lengths_sel,
                attn_sink=attn_sink,
            )
            name = "c4_sparse" if compress_ratio == 4 else "c128"
            _capture_debug_activation(f"{prefix}.{name}_attention_output", compressed_out)

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
        with dsv4_direct_copy_nvtx(
            f"attention_boundary.swa_indices_to_i32.layer{layer_id}.rows{rows}",
            src=metadata.swa_page_indices[:rows],
        ):
            swa_indices = metadata.swa_page_indices[:rows].to(device=q.device, dtype=torch.int32)
        with dsv4_direct_copy_nvtx(
            f"attention_boundary.swa_lengths_to_i32.layer{layer_id}.rows{rows}",
            src=metadata.swa_topk_lengths[:rows],
        ):
            swa_lengths = metadata.swa_topk_lengths[:rows].to(device=q.device, dtype=torch.int32)
        swa_lengths = swa_lengths.clamp(max=swa_indices.shape[-1])
        self._debug_check_swa_index_bounds(
            swa_indices,
            swa_lengths,
            self.kvcache.swa_cache(layer_id).shape[0],
            layer_id=layer_id,
        )

        compressed_cache = None
        compressed_indices = None
        compressed_lengths = None
        if compress_ratio == 4:
            with dsv4_direct_copy_nvtx(
                f"attention_boundary.c4_cache_to_q_dtype.layer{layer_id}.rows{rows}",
                src=self.kvcache.c4_cache(layer_id),
            ):
                compressed_cache = self.kvcache.c4_cache(layer_id).to(q.dtype)
            with dsv4_direct_copy_nvtx(
                f"attention_boundary.c4_sparse_page_indices_to_i32.layer{layer_id}.rows{rows}",
                src=metadata.c4_sparse_page_indices[:rows],
            ):
                compressed_indices = metadata.c4_sparse_page_indices[:rows].to(
                    device=q.device,
                    dtype=torch.int32,
                )
            if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE):
                with dsv4_direct_copy_nvtx(
                    f"attention_boundary.c4_sparse_topk_lengths_to_i32.layer{layer_id}.rows{rows}",
                    src=metadata.c4_sparse_topk_lengths[:rows],
                ):
                    compressed_lengths = metadata.c4_sparse_topk_lengths[:rows].to(
                        device=q.device,
                        dtype=torch.int32,
                    )
                compressed_lengths = compressed_lengths.clamp(max=compressed_indices.shape[-1])
            else:
                compressed_lengths = (compressed_indices >= 0).sum(dim=-1).to(torch.int32)
            self._debug_check_cache_index_bounds(
                compressed_indices,
                compressed_lengths,
                compressed_cache.shape[0],
                layer_id=layer_id,
                label="c4",
            )
        elif compress_ratio == 128:
            with dsv4_direct_copy_nvtx(
                f"attention_boundary.c128_cache_to_q_dtype.layer{layer_id}.rows{rows}",
                src=self.kvcache.c128_cache(layer_id),
            ):
                compressed_cache = self.kvcache.c128_cache(layer_id).to(q.dtype)
            with dsv4_direct_copy_nvtx(
                f"attention_boundary.c128_page_indices_to_i32.layer{layer_id}.rows{rows}",
                src=metadata.c128_page_indices[:rows],
            ):
                compressed_indices = metadata.c128_page_indices[:rows].to(
                    device=q.device,
                    dtype=torch.int32,
                )
            compressed_lengths = (compressed_indices >= 0).sum(dim=-1).to(torch.int32)
            self._debug_check_cache_index_bounds(
                compressed_indices,
                compressed_lengths,
                compressed_cache.shape[0],
                layer_id=layer_id,
                label="c128",
            )

        if metadata.max_seqlen_q <= 1:
            with dsv4_direct_copy_nvtx(
                f"attention_boundary.swa_cache_to_q_dtype.splitk.layer{layer_id}.rows{rows}",
                src=self.kvcache.swa_cache(layer_id),
            ):
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
                self._debug_sync_sparse_attention(
                    backend="splitk",
                    layer_id=layer_id,
                    rows=rows,
                    metadata=metadata,
                    compress_ratio=compress_ratio,
                    swa_indices=swa_indices,
                    swa_lengths=swa_lengths,
                    compressed_indices=compressed_indices,
                    compressed_lengths=compressed_lengths,
                )
                self._capture_attention_debug(
                    layer_id,
                    q,
                    metadata,
                    rows,
                    swa_indices=swa_indices,
                    swa_lengths=swa_lengths,
                    swa_cache=splitk_swa_cache,
                    compressed_cache=compressed_cache,
                    compressed_indices=compressed_indices,
                    compressed_lengths=compressed_lengths,
                    compress_ratio=compress_ratio,
                    attn_sink=attn_sink,
                    merged_output=fast,
                )
                return fast

        with dsv4_direct_copy_nvtx(
            f"attention_boundary.swa_cache_to_q_dtype.base.layer{layer_id}.rows{rows}",
            src=self.kvcache.swa_cache(layer_id),
        ):
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
            self._debug_sync_sparse_attention(
                backend="base",
                layer_id=layer_id,
                rows=rows,
                metadata=metadata,
                compress_ratio=compress_ratio,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                compressed_indices=compressed_indices,
                compressed_lengths=compressed_lengths,
            )
            self._capture_attention_debug(
                layer_id,
                q,
                metadata,
                rows,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                swa_cache=base_swa_cache,
                compressed_cache=compressed_cache,
                compressed_indices=compressed_indices,
                compressed_lengths=compressed_lengths,
                compress_ratio=compress_ratio,
                attn_sink=attn_sink,
                merged_output=fast,
            )
            return fast
        if compressed_cache is None:
            return None
        with dsv4_direct_copy_nvtx(
            f"attention_boundary.swa_cache_to_q_dtype.torch_fallback.layer{layer_id}.rows{rows}",
            src=self.kvcache.swa_cache(layer_id),
        ):
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
        self._capture_attention_debug(
            layer_id,
            q,
            metadata,
            rows,
            swa_indices=swa_indices,
            swa_lengths=swa_lengths,
            swa_cache=fallback_swa_cache,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            compressed_lengths=compressed_lengths,
            compress_ratio=compress_ratio,
            attn_sink=attn_sink,
            merged_output=out,
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
        if swa_independent:
            swa_rows = int(self.kvcache.swa_cache(0).shape[0])
            dummy_loc = max(swa_rows - self.page_size, 0)
            swa_page_indices.fill_(dummy_loc)

        c4_out_loc = torch.full((max_bs,), -1, dtype=torch.int32, device=device)
        c128_out_loc = torch.full((max_bs,), -1, dtype=torch.int32, device=device)
        c4_indexer_out_loc = (
            torch.full((max_bs,), -1, dtype=torch.int32, device=device)
            if component_ownership
            else c4_out_loc
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
            swa_topk_lengths=torch.ones(max_bs, dtype=torch.int32, device=device),
            c4_out_loc=c4_out_loc,
            c128_out_loc=c128_out_loc,
            c4_indexer_out_loc=c4_indexer_out_loc,
            c4_topk_lengths_raw=torch.zeros(max_bs, dtype=torch.int32, device=device),
            c4_topk_lengths_clamp1=torch.ones(max_bs, dtype=torch.int32, device=device),
            c4_sparse_topk_lengths=torch.zeros(max_bs, dtype=torch.int32, device=device),
            c4_sparse_raw_indices=empty_index(topk_width),
            c4_sparse_page_indices=empty_index(topk_width),
            c4_sparse_full_indices=empty_index(topk_width),
            c128_topk_lengths_clamp1=torch.ones(max_bs, dtype=torch.int32, device=device),
            c128_raw_indices=empty_index(c128_width),
            c128_page_indices=empty_index(c128_width),
            c128_full_indices=empty_index(c128_width),
            component_loc_ownership=component_ownership,
            c4_page_table=c4_page_table,
            c128_page_table=c128_page_table,
            c4_indexer_page_table=c4_indexer_page_table,
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
        dst_core.component_loc_ownership = src_core.component_loc_ownership
        direct_swa_requested, direct_c4_requested, direct_c128_requested = (
            self._direct_index_groups_for_replay(dst_core, src_core, bs)
        )
        timing_metadata = {
            "phase": "decode",
            "rows": int(bs),
            "component_loc_ownership": bool(src_core.component_loc_ownership),
            "direct_swa": bool(direct_swa_requested),
            "direct_c4": bool(direct_c4_requested),
            "direct_c128": bool(direct_c128_requested),
        }
        with dsv4_direct_copy_nvtx(
            f"replay_metadata_copy.fused_decode_metadata.bs{bs}",
            page_table=src_core.page_table,
            swa_page_indices=src_core.swa_page_indices,
            c4_sparse_page_indices=src_core.c4_sparse_page_indices,
        ):
            with dsv4_owner_timing.maybe_cuda_range(
                "dsv4.replay_metadata.decode.fused_copy",
                timing_metadata,
            ):
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
                self._record_direct_graph_metadata_bytes(
                    bs,
                    direct_swa=direct_swa_done,
                    direct_c4=direct_c4_done,
                    direct_c128=direct_c128_done,
                )
            elif (
                src_core.swa_source_elided_for_graph
                or src_core.c4_sparse_source_elided_for_graph
                or src_core.c128_source_elided_for_graph
            ):
                raise RuntimeError(
                    f"{dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE}=1 "
                    "elided eager index source metadata, but direct graph-buffer "
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
        self._record_replay_copy_bytes(
            src_core,
            bs,
            fused=copied_by_helper,
            skip_swa=direct_swa_done,
            skip_c4_sparse=direct_c4_done,
            skip_c128=direct_c128_done,
        )
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
        with dsv4_direct_copy_nvtx(
            f"replay_metadata_copy.fallback_scalar_vectors.bs{bs}",
            positions=src_core.positions,
            seq_lens=src_core.seq_lens,
        ):
            for name in scalar_names:
                if self._capture_graph_inputs_bound and name in {"raw_out_loc", "positions"}:
                    continue
                getattr(dst_core, name)[:bs].copy_(getattr(src_core, name)[:bs])
            dst_core.cu_seqlens_q[: bs + 1].copy_(src_core.cu_seqlens_q[: bs + 1])
        with dsv4_direct_copy_nvtx(
            f"replay_metadata_copy.fallback_page_table.bs{bs}",
            page_table=src_core.page_table,
        ):
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
            with dsv4_direct_copy_nvtx(
                f"replay_metadata_copy.fallback_{name}.bs{bs}",
                src=getattr(src_core, name),
            ):
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
            and dsv4_kernel.dsv4_env_flag(
                dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE
            )
        ):
            return False, False, False
        if self.device.type != "cuda" or not dsv4_kernel.dsv4_sm80_triton_enabled(
            dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE
        ):
            return False, False, False
        groups = _direct_graph_metadata_groups()
        direct_swa = (
            "swa" in groups
            and src_core.swa_page_indices is not None
            and not bool(getattr(self.kvcache, "swa_independent_lifecycle_enabled", False))
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
        with dsv4_direct_copy_nvtx(
            f"replay_metadata_direct.index_buffers.bs{rows}",
            positions=src_core.positions,
            c4_page_table=src_core.c4_page_table,
            c128_page_table=src_core.c128_page_table,
        ):
            with dsv4_owner_timing.maybe_cuda_range(
                "dsv4.direct_graph_metadata.decode.index_buffers",
                {
                    "phase": "decode",
                    "rows": int(rows),
                    "direct_swa": bool(direct_swa),
                    "direct_c4": bool(direct_c4),
                    "direct_c128": bool(direct_c128),
                },
            ):
                return dsv4_kernel.direct_decode_index_metadata_for_replay(
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
                )

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
            with dsv4_direct_copy_nvtx(
                f"replay_metadata_copy.direct_index_fallback_{name}.bs{rows}",
                src=getattr(src_core, name),
            ):
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
            with dsv4_direct_copy_nvtx(
                f"replay_metadata_copy.component_{name}.bs{rows}",
                src=src,
            ):
                with dsv4_owner_timing.maybe_cuda_range(
                    f"dsv4.replay_metadata.decode.component_page_table.{name}",
                    {"phase": "decode", "rows": int(rows), "field": name},
                ):
                    _record_metadata_counter(
                        "dsv4.replay_metadata_copy.bytes",
                        value=(
                            _tensor_nbytes(src, rows)
                            if src is not None
                            else _tensor_nbytes(dst, rows)
                        ),
                        phase="decode",
                        rows=rows,
                        field=name,
                        stable="per-request",
                    )
                    _record_metadata_counter(
                        "dsv4.replay_metadata_copy.calls",
                        phase="decode",
                        rows=rows,
                        field=name,
                        stable="per-request",
                    )
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
        if src_core.component_loc_ownership:
            with dsv4_direct_copy_nvtx(
                f"replay_metadata_copy.component_write_locs.bs{rows}",
                positions=src_core.positions,
            ):
                with dsv4_owner_timing.maybe_cuda_range(
                    "dsv4.replay_metadata.decode.component_write_locs",
                    {"phase": "decode", "rows": int(rows)},
                ):
                    if dsv4_kernel.copy_component_write_locs_for_replay(
                        c4_page_table=dst_core.c4_page_table,
                        c128_page_table=dst_core.c128_page_table,
                        c4_indexer_page_table=dst_core.c4_indexer_page_table,
                        positions=dst_core.positions,
                        c4_out_loc=dst_core.c4_out_loc,
                        c128_out_loc=dst_core.c128_out_loc,
                        c4_indexer_out_loc=dst_core.c4_indexer_out_loc,
                        rows=rows,
                        page_size=self.page_size,
                    ):
                        self._record_component_write_loc_copy(
                            rows, backend="triton_component_table"
                        )
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
                    self._record_component_write_loc_copy(rows, backend="python_compact_fallback")
            return
        with dsv4_direct_copy_nvtx(
            f"replay_metadata_copy.masked_compressed_locs.bs{rows}",
            raw_out_loc=src_core.raw_out_loc,
            positions=src_core.positions,
        ):
            dsv4_kernel.copy_masked_compressed_locs(
                src_core.raw_out_loc,
                src_core.positions,
                dst_core.c4_out_loc,
                dst_core.c128_out_loc,
                rows,
            )
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
]

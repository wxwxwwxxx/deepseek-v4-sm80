from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kvcache.deepseek_v4_pool import DeepSeekV4KVCache
from minisgl.utils import div_ceil

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


DSV4CompressRatio = Literal[0, 4, 128]
_PAGE_INDEX_ALIGNMENT = 64


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
        dsv4_kernel.store_swa_fallback(self.kvcache, layer_id, k, batch.out_loc)
        return self._fallback_attention(q, layer_id, metadata.core_metadata, ratio, attn_sink)

    def store_compressed(
        self,
        layer_id: int,
        kv: torch.Tensor,
        batch: Batch,
        compress_ratio: Literal[4, 128],
    ) -> None:
        metadata = batch.attn_metadata
        if not isinstance(metadata, DSV4AttentionMetadata):
            return
        loc = (
            metadata.core_metadata.c4_out_loc
            if compress_ratio == 4
            else metadata.core_metadata.c128_out_loc
        )
        if loc is None or loc.numel() == 0 or kv.numel() == 0:
            return
        n = min(loc.numel(), kv.shape[0])
        dsv4_kernel.compress_norm_rope_store_fallback(self.kvcache, layer_id, kv[:n], loc[:n])

    def store_indexer(self, layer_id: int, kv: torch.Tensor, batch: Batch) -> None:
        metadata = batch.attn_metadata
        if not isinstance(metadata, DSV4AttentionMetadata):
            return
        loc = metadata.core_metadata.c4_indexer_out_loc
        if loc is None or loc.numel() == 0 or kv.numel() == 0:
            return
        n = min(loc.numel(), kv.shape[0])
        dsv4_kernel.store_indexer_fallback(self.kvcache, layer_id, kv[:n], loc[:n])

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        self.capture_bs = sorted(bs_list)
        self.max_graph_bs = max(bs_list) if bs_list else 0
        if self.max_graph_bs == 0:
            return
        self.capture = self._empty_decode_metadata(self.max_graph_bs, max_seq_len)

    def prepare_for_capture(self, batch: Batch) -> None:
        assert self.capture is not None
        assert batch.size in self.capture_bs
        batch.attn_metadata = self.capture

    def prepare_for_replay(self, batch: Batch) -> None:
        assert self.capture is not None
        metadata = batch.attn_metadata
        assert isinstance(metadata, DSV4AttentionMetadata)
        self._copy_metadata_for_replay(self.capture, metadata, batch.padded_size)

    def _build_metadata(self, batch: Batch) -> DSV4AttentionMetadata:
        reqs = batch.padded_reqs
        if not reqs:
            raise ValueError("DSV4 attention metadata requires at least one request")

        device = self.device
        positions = _to_int32(batch.positions, device)
        raw_out_loc = _to_int32(batch.out_loc, device)
        extend_lens_list = [req.extend_len for req in reqs]
        req_seq_lens_list = [req.device_len for req in reqs]
        max_seqlen_q = max(extend_lens_list)
        max_seqlen_k = max(req_seq_lens_list)

        extend_lens = torch.tensor(extend_lens_list, dtype=torch.int32, device=device)
        req_seq_lens = torch.tensor(req_seq_lens_list, dtype=torch.int32, device=device)
        cu_seqlens_q = F.pad(extend_lens.cumsum(dim=0), (1, 0))

        table_indices = torch.empty(positions.numel(), dtype=torch.int32, device=device)
        offset = 0
        for req, length in zip(reqs, extend_lens_list):
            table_indices[offset : offset + length].fill_(req.table_idx)
            offset += length
        assert offset == positions.numel()

        seq_lens = positions + 1
        page_table = self._make_page_table(table_indices, max_seqlen_k)
        swa_page_indices = self._make_swa_indices(table_indices, positions)
        swa_topk_lengths = torch.clamp(seq_lens, max=self.window_size)

        c4_out_loc = self.kvcache.compressed_locs_from_full_locs(raw_out_loc, 4, positions)
        c128_out_loc = self.kvcache.compressed_locs_from_full_locs(raw_out_loc, 128, positions)
        if c4_out_loc.numel() == 0:
            c4_out_loc = None
        if c128_out_loc.numel() == 0:
            c128_out_loc = None

        c4_topk_lengths_raw = torch.div(seq_lens, 4, rounding_mode="floor")
        c4_topk_lengths_clamp1 = c4_topk_lengths_raw.clamp_min(1)
        c4_sparse_topk_lengths = c4_topk_lengths_clamp1.clamp(max=self.index_topk)
        (
            c4_sparse_raw_indices,
            c4_sparse_page_indices,
            c4_sparse_full_indices,
        ) = self._make_sparse_compressed_indices(table_indices, c4_topk_lengths_raw, 4)

        c128_lengths_raw = torch.div(seq_lens, 128, rounding_mode="floor")
        c128_topk_lengths_clamp1 = c128_lengths_raw.clamp_min(1)
        c128_raw_indices, c128_page_indices, c128_full_indices = (
            self._make_all_compressed_indices(table_indices, c128_lengths_raw, 128)
        )

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
            c4_indexer_out_loc=c4_out_loc,
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
        )

        has_c4 = any(m.compress_ratio == 4 for m in self.kvcache.layer_mapping)
        indexer_metadata = (
            DSV4IndexerMetadata(
                page_size=self.page_size,
                page_table=core.page_table,
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

    def _make_swa_indices(
        self,
        table_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        offsets = positions[:, None] - torch.arange(
            self.window_size,
            dtype=torch.int32,
            device=self.device,
        )[None, :]
        return self._gather_full_locs(table_indices, offsets)

    def _make_sparse_compressed_indices(
        self,
        table_indices: torch.Tensor,
        lengths: torch.Tensor,
        ratio: Literal[4, 128],
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
        page = torch.where(full >= 0, full.div(ratio, rounding_mode="floor"), full)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(int(lengths.max().item()) if lengths.numel() else 0, 1)
        raw = torch.full((lengths.numel(), width), -1, dtype=torch.int32, device=self.device)
        for row, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            values = torch.arange(int(length), dtype=torch.int32, device=self.device)
            raw[row, : values.numel()] = values
        full = self._compressed_raw_to_full_locs(table_indices, raw, ratio)
        page = torch.where(full >= 0, full.div(ratio, rounding_mode="floor"), full)
        return (
            _pad_last_dim(raw, value=-1),
            _pad_last_dim(page.to(torch.int32), value=-1),
            _pad_last_dim(full.to(torch.int32), value=-1),
        )

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
        cache = self.kvcache.swa_cache(layer_id).to(q.dtype)
        context_indices = [
            self._context_indices_for_query(metadata, row, compress_ratio)
            for row in range(q.shape[0])
        ]
        return dsv4_kernel.paged_mqa_attention_fallback(
            q,
            cache,
            context_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
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
        topk_width = div_ceil(max(self.index_topk, 1), _PAGE_INDEX_ALIGNMENT) * _PAGE_INDEX_ALIGNMENT
        c128_width = div_ceil(max(div_ceil(max_seq_len, 128), 1), _PAGE_INDEX_ALIGNMENT)
        c128_width *= _PAGE_INDEX_ALIGNMENT

        def empty_index(width: int) -> torch.Tensor:
            return torch.full((max_bs, width), -1, dtype=torch.int32, device=device)

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
            swa_page_indices=empty_index(
                div_ceil(self.window_size, _PAGE_INDEX_ALIGNMENT) * _PAGE_INDEX_ALIGNMENT
            ),
            swa_topk_lengths=torch.ones(max_bs, dtype=torch.int32, device=device),
            c4_out_loc=None,
            c128_out_loc=None,
            c4_indexer_out_loc=None,
            c4_topk_lengths_raw=torch.zeros(max_bs, dtype=torch.int32, device=device),
            c4_topk_lengths_clamp1=torch.ones(max_bs, dtype=torch.int32, device=device),
            c4_sparse_topk_lengths=torch.ones(max_bs, dtype=torch.int32, device=device),
            c4_sparse_raw_indices=empty_index(topk_width),
            c4_sparse_page_indices=empty_index(topk_width),
            c4_sparse_full_indices=empty_index(topk_width),
            c128_topk_lengths_clamp1=torch.ones(max_bs, dtype=torch.int32, device=device),
            c128_raw_indices=empty_index(c128_width),
            c128_page_indices=empty_index(c128_width),
            c128_full_indices=empty_index(c128_width),
        )
        return DSV4AttentionMetadata(
            core_attn_metadata=core,
            indexer_metadata=DSV4IndexerMetadata(self.page_size, core.page_table, core.c4_topk_lengths_raw),
            c4_compress_metadata=DSV4CompressMetadata(4, core.c4_out_loc, core.seq_lens, core.positions),
            c128_compress_metadata=DSV4CompressMetadata(128, core.c128_out_loc, core.seq_lens, core.positions),
        )

    def _copy_metadata_for_replay(
        self,
        dst: DSV4AttentionMetadata,
        src: DSV4AttentionMetadata,
        bs: int,
    ) -> None:
        dst_core = dst.core_metadata
        src_core = src.core_metadata
        for name in (
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
        ):
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
            self._copy_2d(getattr(dst_core, name), getattr(src_core, name), bs, fill=-1)

    def _copy_2d(self, dst: torch.Tensor, src: torch.Tensor, rows: int, *, fill: int) -> None:
        dst[:rows].fill_(fill)
        width = min(dst.shape[1], src.shape[1])
        dst[:rows, :width].copy_(src[:rows, :width])


__all__ = [
    "DSV4AttentionBackend",
    "DSV4AttentionMetadata",
    "DSV4CompressMetadata",
    "DSV4CoreAttentionMetadata",
    "DSV4IndexerMetadata",
]

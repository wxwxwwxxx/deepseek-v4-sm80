#!/usr/bin/env python3
"""Synthetic CUDA graph private-pool attribution probes for TARGET 08.32.

The script deliberately avoids loading DSV4 checkpoints.  Parent mode
(`--suite ...`) launches each case in a fresh Python process so allocator,
cuBLAS/cuBLASLt, and graph-pool state do not leak across comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

try:
    from minisgl.kernel import deepseek_v4 as dsv4_kernel
except Exception:  # pragma: no cover - the controls can still run without mini imports.
    dsv4_kernel = None  # type: ignore[assignment]


MILESTONE_DIR = REPO_ROOT / "performance_milestones" / "target08_cuda_graph_private_pool_micro_attribution"
DEFAULT_OUTPUT_DIR = MILESTONE_DIR


DTYPE_BY_NAME = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "i32": torch.int32,
}


def _tensor_nbytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def _sum_nbytes(items: list[torch.Tensor]) -> int:
    return int(sum(_tensor_nbytes(x) for x in items))


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1 << 30)


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().float().sum().item())


def _randn(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    out = torch.empty(shape, device=device, dtype=dtype)
    if dtype.is_floating_point:
        out.normal_(mean=0.0, std=0.02)
    else:
        out.zero_()
    return out


def _arange_i32(n: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(n, device=device, dtype=torch.int32)


def _mem_stats(device: torch.device, prefix: str) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        f"{prefix}_free_bytes": int(free),
        f"{prefix}_total_bytes": int(total),
        f"{prefix}_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        f"{prefix}_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _peak_stats(device: torch.device) -> dict[str, int]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


@dataclass
class BuiltCase:
    name: str
    category: str
    description: str
    owner_count: int
    projection_count: int
    run: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]
    groups: dict[str, list[torch.Tensor]]
    notes: list[str]

    def explicit_bytes(self) -> dict[str, int]:
        input_bytes = _sum_nbytes(self.groups.get("inputs", []))
        output_bytes = _sum_nbytes(self.groups.get("outputs", []))
        workspace_bytes = _sum_nbytes(self.groups.get("workspaces", []))
        weight_bytes = _sum_nbytes(self.groups.get("weights", []))
        cache_bytes = _sum_nbytes(self.groups.get("caches", []))
        metadata_bytes = _sum_nbytes(self.groups.get("metadata", []))
        return {
            "explicit_input_bytes": input_bytes,
            "explicit_output_bytes": output_bytes,
            "explicit_workspace_bytes": workspace_bytes,
            "explicit_weight_bytes": weight_bytes,
            "explicit_cache_bytes": cache_bytes,
            "explicit_metadata_bytes": metadata_bytes,
            "explicit_total_bytes": input_bytes
            + output_bytes
            + workspace_bytes
            + weight_bytes
            + cache_bytes
            + metadata_bytes,
        }


def _case_empty_graph(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    marker = torch.zeros((1,), device=device, dtype=torch.float32)

    def run() -> torch.Tensor:
        return marker

    return BuiltCase(
        name=args.case,
        category="control",
        description="Empty CUDAGraph body with one static marker tensor.",
        owner_count=1,
        projection_count=1,
        run=run,
        groups={"inputs": [marker], "outputs": [], "workspaces": [], "weights": [], "caches": [], "metadata": []},
        notes=[],
    )


def _case_copy_staging(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    rows = int(args.bs)
    src_ids = _arange_i32(rows, device=device)
    src_loc = _arange_i32(rows, device=device) + 1024
    src_pos = _arange_i32(rows, device=device) + 2048
    dst_ids = torch.empty_like(src_ids)
    dst_loc = torch.empty_like(src_loc)
    dst_pos = torch.empty_like(src_pos)

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dst_ids.copy_(src_ids)
        dst_loc.copy_(src_loc)
        dst_pos.copy_(src_pos)
        return dst_ids, dst_loc, dst_pos

    return BuiltCase(
        name=args.case,
        category="control",
        description="Graph input staging copies for input_ids/out_loc/positions.",
        owner_count=1,
        projection_count=1,
        run=run,
        groups={
            "inputs": [src_ids, src_loc, src_pos],
            "outputs": [dst_ids, dst_loc, dst_pos],
            "workspaces": [],
            "weights": [],
            "caches": [],
            "metadata": [],
        },
        notes=[],
    )


def _case_elementwise(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    x = _randn((int(args.bs), int(args.hidden)), device=device, dtype=torch.bfloat16)
    bias = _randn((int(args.hidden),), device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    def run() -> torch.Tensor:
        y = x.float()
        for scale in (1.001, 0.999, 1.003, 0.997, 1.005, 0.995):
            y = F.silu(y * scale + bias)
        if args.variant == "prealloc":
            out.copy_(y.to(out.dtype))
            return out
        return y.to(torch.bfloat16)

    return BuiltCase(
        name=args.case,
        category="control",
        description=f"Out-of-place elementwise chain variant={args.variant}.",
        owner_count=1,
        projection_count=1,
        run=run,
        groups={
            "inputs": [x, bias],
            "outputs": [out] if args.variant == "prealloc" else [],
            "workspaces": [],
            "weights": [],
            "caches": [],
            "metadata": [],
        },
        notes=[],
    )


def _case_repeated_matmul(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    n = int(args.layers)
    hidden = int(args.hidden)
    x = _randn((int(args.bs), hidden), device=device, dtype=torch.bfloat16)
    w = _randn((hidden, hidden), device=device, dtype=torch.bfloat16)
    buf0 = torch.empty_like(x)
    buf1 = torch.empty_like(x)
    keep_all = args.variant == "keep_all"

    def run() -> torch.Tensor:
        y = x
        kept: list[torch.Tensor] = []
        if args.variant == "prealloc":
            a = x
            b = buf0
            c = buf1
            for _ in range(n):
                torch.mm(a, w, out=b)
                a, b, c = b, c, a
            return a
        for _ in range(n):
            y = torch.mm(y, w)
            if keep_all:
                kept.append(y)
        if keep_all and kept:
            return torch.stack([t.flatten()[0] for t in kept]).sum().reshape(1)
        return y

    desc = f"Repeated BF16 square matmul N={n}, variant={args.variant}, shape=[{args.bs},{hidden}]x[{hidden},{hidden}]."
    return BuiltCase(
        name=args.case,
        category="control" if args.case.startswith("control_") else "scaling",
        description=desc,
        owner_count=n,
        projection_count=n,
        run=run,
        groups={
            "inputs": [x],
            "outputs": [buf0, buf1] if args.variant == "prealloc" else [],
            "workspaces": [],
            "weights": [w],
            "caches": [],
            "metadata": [],
        },
        notes=["keep_all intentionally retains every matmul output inside the captured body."] if keep_all else [],
    )


def _make_indices(
    rows: int,
    width: int,
    slots: int,
    *,
    device: torch.device,
    row_stride: int = 17,
) -> torch.Tensor:
    base = torch.arange(width, device=device, dtype=torch.int64)[None, :]
    row = torch.arange(rows, device=device, dtype=torch.int64)[:, None] * row_stride
    return ((base + row) % max(slots, 1)).to(torch.int32)


def _synthetic_two_source_attention(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    *,
    compressed_cache: torch.Tensor | None = None,
    compressed_indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [swa_cache[swa_indices.to(torch.long)].to(q.dtype)]
    if compressed_cache is not None and compressed_indices is not None:
        parts.insert(0, compressed_cache[compressed_indices.to(torch.long)].to(q.dtype))
    candidates = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    qf = q.float()
    cf = candidates.float()
    scores = torch.einsum("bhd,btd->bht", qf, cf) * (q.shape[-1] ** -0.5)
    if attn_sink is not None:
        sink = attn_sink[None, :, None].to(scores.dtype).expand(scores.shape[0], -1, -1)
        scores = torch.cat([scores, sink], dim=-1)
        attn = torch.softmax(scores, dim=-1)[..., :-1]
    else:
        attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bht,btd->bhd", attn, cf).to(q.dtype)


def _case_attention(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    rows = int(args.bs)
    heads = int(args.local_heads)
    dim = int(args.head_dim)
    page_size = int(args.page_size)
    num_pages = int(args.num_pages)
    seq_len = int(args.seq_len)
    q = _randn((rows, heads, dim), device=device, dtype=torch.bfloat16)
    swa_slots = num_pages * page_size
    swa_cache = _randn((swa_slots, dim), device=device, dtype=torch.bfloat16)
    swa_width = int(args.swa_width)
    swa_indices = _make_indices(rows, swa_width, swa_slots, device=device)
    sink = _randn((heads,), device=device, dtype=torch.float32)

    ratio = 0
    compressed_cache: torch.Tensor | None = None
    compressed_indices: torch.Tensor | None = None
    if args.owner == "c4":
        ratio = 4
        c4_slots = max(swa_slots // 4, 1)
        compressed_cache = _randn((c4_slots, dim), device=device, dtype=torch.bfloat16)
        compressed_indices = _make_indices(rows, int(args.index_topk), c4_slots, device=device, row_stride=23)
    elif args.owner == "c128":
        ratio = 128
        c128_slots = max(math.ceil(seq_len / 128), 1)
        c128_width = max(int(args.c128_width), c128_slots)
        compressed_cache = _randn((max(swa_slots // 128, c128_width), dim), device=device, dtype=torch.bfloat16)
        compressed_indices = _make_indices(rows, c128_width, compressed_cache.shape[0], device=device, row_stride=5)

    def run() -> torch.Tensor:
        return _synthetic_two_source_attention(
            q,
            swa_cache,
            swa_indices,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            attn_sink=sink,
        )

    caches = [swa_cache]
    metadata = [swa_indices]
    if compressed_cache is not None:
        caches.append(compressed_cache)
    if compressed_indices is not None:
        metadata.append(compressed_indices)
    owner_name = "SWA" if ratio == 0 else f"C{ratio}"
    return BuiltCase(
        name=args.case,
        category="dsv4_subgraph",
        description=f"Synthetic {owner_name} two-source decode attention, bs={rows}.",
        owner_count=1,
        projection_count=43 if ratio == 0 else (21 if ratio == 4 else 20),
        run=run,
        groups={
            "inputs": [q, sink],
            "outputs": [],
            "workspaces": [],
            "weights": [],
            "caches": caches,
            "metadata": metadata,
        },
        notes=["Vectorized synthetic equivalent; does not instantiate DeepSeekV4AttentionBackend."],
    )


def _case_indexer_topk(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    rows = int(args.bs)
    heads = int(args.index_heads)
    dim = int(args.index_head_dim)
    page_size = max(int(args.page_size) // 4, 1)
    max_c4_len = int(args.num_pages) * page_size
    q = _randn((rows, heads, dim), device=device, dtype=torch.bfloat16)
    weights = _randn((rows, heads), device=device, dtype=torch.float32)
    cache = _randn((max_c4_len, dim), device=device, dtype=torch.bfloat16)
    page_table = torch.arange(int(args.num_pages), device=device, dtype=torch.int32).expand(rows, -1).contiguous()
    seq_lens = torch.full((rows,), min(max_c4_len, max(int(args.seq_len) // 4, 1)), device=device, dtype=torch.int32)

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qf = q.float()
        cf = cache.float()
        scores_by_head = torch.relu(torch.einsum("bhd,td->bth", qf, cf))
        logits = (scores_by_head * weights[:, None, :]).sum(dim=-1)
        width = min(int(args.index_topk), logits.shape[-1])
        values, raw = torch.topk(logits, k=width, dim=-1, largest=True, sorted=False)
        del values
        page = torch.gather(
            page_table,
            dim=1,
            index=(raw.to(torch.long) // page_size).clamp(max=page_table.shape[1] - 1),
        )
        full = (page.to(torch.long) * int(args.page_size) + (raw.to(torch.long) % page_size) * 4 + 3).to(torch.int32)
        return logits, raw.to(torch.int32), full

    return BuiltCase(
        name=args.case,
        category="dsv4_subgraph",
        description=f"Synthetic C4 indexer logits + topk, static c4_len={max_c4_len}, width={args.index_topk}.",
        owner_count=1,
        projection_count=21,
        run=run,
        groups={
            "inputs": [q, weights],
            "outputs": [],
            "workspaces": [],
            "weights": [],
            "caches": [cache],
            "metadata": [page_table, seq_lens],
        },
        notes=["Uses full num_pages*c4_page_size static logits shape to mimic graph capture metadata width."],
    )


def _case_indexer_topk_repeated(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    n = int(args.layers)
    rows = int(args.bs)
    heads = int(args.index_heads)
    dim = int(args.index_head_dim)
    page_size = max(int(args.page_size) // 4, 1)
    max_c4_len = int(args.num_pages) * page_size
    q = _randn((rows, heads, dim), device=device, dtype=torch.bfloat16)
    weights = _randn((rows, heads), device=device, dtype=torch.float32)
    cache = _randn((max_c4_len, dim), device=device, dtype=torch.bfloat16)
    page_table = torch.arange(int(args.num_pages), device=device, dtype=torch.int32).expand(rows, -1).contiguous()
    seq_lens = torch.full((rows,), min(max_c4_len, max(int(args.seq_len) // 4, 1)), device=device, dtype=torch.int32)

    def _one(q_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qf = q_in.float()
        cf = cache.float()
        scores_by_head = torch.relu(torch.einsum("bhd,td->bth", qf, cf))
        logits = (scores_by_head * weights[:, None, :]).sum(dim=-1)
        width = min(int(args.index_topk), logits.shape[-1])
        _values, raw = torch.topk(logits, k=width, dim=-1, largest=True, sorted=False)
        page = torch.gather(
            page_table,
            dim=1,
            index=(raw.to(torch.long) // page_size).clamp(max=page_table.shape[1] - 1),
        )
        full = (
            page.to(torch.long) * int(args.page_size)
            + (raw.to(torch.long) % page_size) * 4
            + 3
        ).to(torch.int32)
        return logits, full

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        q_cur = q
        logits = torch.empty((rows, max_c4_len), device=device, dtype=torch.float32)
        full = torch.empty((rows, min(int(args.index_topk), max_c4_len)), device=device, dtype=torch.int32)
        for _ in range(n):
            logits, full = _one(q_cur)
            q_cur = q_cur + logits[:, :1, None].to(q_cur.dtype) * 0.0
        return logits, full

    return BuiltCase(
        name=args.case,
        category="scaling",
        description=f"Repeated synthetic C4 indexer logits + topk skeleton, N={n}.",
        owner_count=n,
        projection_count=21,
        run=run,
        groups={
            "inputs": [q, weights],
            "outputs": [],
            "workspaces": [],
            "weights": [],
            "caches": [cache],
            "metadata": [page_table, seq_lens],
        },
        notes=["Repeated validation for the single C4 indexer/topk projection; DSV4 has 21 C4/indexer layers."],
    )


def _rope_tail_inplace(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    rotary_dim: int,
) -> torch.Tensor:
    freqs = torch.outer(positions.float(), inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    while cos.ndim < x[..., -rotary_dim:].ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rope = x[..., -rotary_dim:].float().unflatten(-1, (-1, 2))
    a = rope[..., 0]
    b = rope[..., 1]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    x[..., -rotary_dim:] = rotated.to(x.dtype)
    return x


def _case_qkv_norm_rope_store(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    rows = int(args.bs)
    heads = int(args.local_heads)
    dim = int(args.head_dim)
    rotary_dim = int(args.rotary_dim)
    q = _randn((rows, heads, dim), device=device, dtype=torch.bfloat16)
    kv = _randn((rows, dim), device=device, dtype=torch.bfloat16)
    kv_weight = _randn((dim,), device=device, dtype=torch.float32)
    cache = _randn((int(args.num_pages) * int(args.page_size), dim), device=device, dtype=torch.bfloat16)
    out_loc = torch.arange(rows, device=device, dtype=torch.int64)
    positions = torch.arange(rows, device=device, dtype=torch.int64) + 1024
    inv_freq = 1.0 / (
        float(args.rope_base)
        ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        qf = q.float()
        q.copy_((qf * torch.rsqrt(qf.square().mean(-1, keepdim=True) + float(args.rms_eps))).to(q.dtype))
        _rope_tail_inplace(q, positions, inv_freq, rotary_dim=rotary_dim)
        kvf = kv.float()
        kv.copy_((kvf * torch.rsqrt(kvf.square().mean(-1, keepdim=True) + float(args.rms_eps)) * kv_weight).to(kv.dtype))
        _rope_tail_inplace(kv, positions, inv_freq, rotary_dim=rotary_dim)
        cache.index_copy_(0, out_loc, kv)
        return q, kv

    return BuiltCase(
        name=args.case,
        category="dsv4_subgraph",
        description="Synthetic q/kv RMSNorm + RoPE + SWA cache store boundary.",
        owner_count=1,
        projection_count=43,
        run=run,
        groups={
            "inputs": [q, kv, kv_weight, positions, out_loc, inv_freq],
            "outputs": [],
            "workspaces": [],
            "weights": [],
            "caches": [cache],
            "metadata": [],
        },
        notes=["Uses capture-safe index_copy_ instead of fallback bool(torch.any(...)) branches."],
    )


def _case_metadata_deforest(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    rows = int(args.bs)
    page_size = int(args.page_size)
    pages = int(args.num_pages)
    window = int(args.swa_width)
    c4_width = int(args.index_topk)
    c128_width = int(args.c128_width)
    ctx_page_table = torch.arange(pages, device=device, dtype=torch.int32).expand(rows, -1).contiguous()
    table_indices = torch.arange(rows, device=device, dtype=torch.int64)
    positions = (torch.arange(rows, device=device, dtype=torch.int64) * 7 + int(args.seq_len)).clamp(max=pages * page_size - 1)
    swa_offsets = torch.arange(window, device=device, dtype=torch.int64)
    c4_offsets = torch.arange(c4_width, device=device, dtype=torch.int64)
    c128_offsets = torch.arange(c128_width, device=device, dtype=torch.int64)

    def _map_raw(raw: torch.Tensor, ratio: int) -> torch.Tensor:
        logical = raw.clamp(min=0)
        page_idx = (logical // page_size).clamp(max=ctx_page_table.shape[1] - 1)
        page = torch.gather(ctx_page_table, 1, page_idx.to(torch.long))
        return (page.to(torch.long) * page_size + logical % page_size) // ratio

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = table_indices
        start = (positions - window + 1).clamp(min=0)
        swa_raw = start[:, None] + swa_offsets[None, :]
        swa_valid = swa_raw <= positions[:, None]
        swa = torch.where(swa_valid, _map_raw(swa_raw, 1).to(torch.int32), torch.full_like(swa_raw, -1).to(torch.int32))
        c4_raw = ((positions[:, None] // 4) - c4_offsets[None, :]).clamp(min=0) * 4 + 3
        c4 = _map_raw(c4_raw, 4).to(torch.int32)
        c128_raw = ((positions[:, None] // 128) - c128_offsets[None, :]).clamp(min=0) * 128 + 127
        c128 = _map_raw(c128_raw, 128).to(torch.int32)
        lengths = torch.stack(
            [
                swa_valid.sum(dim=1).to(torch.int32),
                torch.full((rows,), c4_width, device=device, dtype=torch.int32),
                torch.full((rows,), c128_width, device=device, dtype=torch.int32),
            ],
            dim=1,
        )
        return swa, c4, c128, lengths

    return BuiltCase(
        name=args.case,
        category="dsv4_subgraph",
        description="Synthetic direct graph metadata/deforest helper for SWA/C4/C128 indices.",
        owner_count=1,
        projection_count=1,
        run=run,
        groups={
            "inputs": [ctx_page_table, positions, table_indices, swa_offsets, c4_offsets, c128_offsets],
            "outputs": [],
            "workspaces": [],
            "weights": [],
            "caches": [],
            "metadata": [ctx_page_table, positions, table_indices],
        },
        notes=["Synthetic torch version; local JIT deforest helper is not required for this attribution run."],
    )


def _projection_block(
    h: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    local_groups: int,
    d_per_group: int,
) -> torch.Tensor:
    q_lora = torch.mm(h, weights["qwa"])
    q = torch.mm(q_lora, weights["qwb"])
    kv = torch.mm(h, weights["wkv"])
    del kv
    o = q.view(h.shape[0], local_groups, d_per_group)
    o = torch.bmm(o.transpose(0, 1), weights["woa"]).transpose(0, 1).reshape(h.shape[0], -1)
    return torch.mm(o, weights["wob"])


def _make_projection_weights(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    hidden = int(args.hidden)
    q_lora = int(args.q_lora_rank)
    local_heads = int(args.local_heads)
    head_dim = int(args.head_dim)
    local_groups = int(args.local_groups)
    o_rank = int(args.o_lora_rank)
    d_per_group = local_heads * head_dim // local_groups
    return {
        "qwa": _randn((hidden, q_lora), device=device, dtype=torch.bfloat16),
        "qwb": _randn((q_lora, local_heads * head_dim), device=device, dtype=torch.bfloat16),
        "wkv": _randn((hidden, head_dim), device=device, dtype=torch.bfloat16),
        "woa": _randn((local_groups, d_per_group, o_rank), device=device, dtype=torch.bfloat16),
        "wob": _randn((local_groups * o_rank, hidden), device=device, dtype=torch.bfloat16),
    }


def _case_projection_skeleton(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    n = int(args.layers)
    x = _randn((int(args.bs), int(args.hidden)), device=device, dtype=torch.bfloat16)
    weights = _make_projection_weights(args, device)
    local_groups = int(args.local_groups)
    d_per_group = int(args.local_heads) * int(args.head_dim) // local_groups

    def run() -> torch.Tensor:
        h = x
        for _ in range(n):
            h = _projection_block(h, weights, local_groups=local_groups, d_per_group=d_per_group)
        return h

    return BuiltCase(
        name=args.case,
        category="scaling",
        description=f"Projection-only one-layer/repeated-layer skeleton, N={n}.",
        owner_count=n,
        projection_count=n,
        run=run,
        groups={
            "inputs": [x],
            "outputs": [],
            "workspaces": [],
            "weights": list(weights.values()),
            "caches": [],
            "metadata": [],
        },
        notes=["Weights are synthetic and reused across repetitions; this isolates captured temporaries/workspace."],
    )


def _case_moe_skeleton(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    n = int(args.layers)
    bs = int(args.bs)
    hidden = int(args.hidden)
    topk = int(args.topk)
    experts = int(args.n_routed_experts)
    inter = int(args.moe_intermediate_per_rank)
    x = _randn((bs, hidden), device=device, dtype=torch.bfloat16)
    gate_w = _randn((hidden, experts), device=device, dtype=torch.bfloat16)
    up_w = _randn((hidden, inter), device=device, dtype=torch.bfloat16)
    down_w = _randn((inter, hidden), device=device, dtype=torch.bfloat16)
    shared_up_w = _randn((hidden, inter * 2), device=device, dtype=torch.bfloat16)
    shared_down_w = _randn((inter, hidden), device=device, dtype=torch.bfloat16)

    def _one(h: torch.Tensor) -> torch.Tensor:
        scores = torch.mm(h, gate_w).float()
        vals, idx = torch.topk(scores, k=topk, dim=-1)
        route_w = torch.softmax(vals, dim=-1).to(h.dtype)
        del idx
        expanded = h[:, None, :].expand(-1, topk, -1).reshape(bs * topk, hidden)
        mid = F.silu(torch.mm(expanded, up_w))
        routed = torch.mm(mid, down_w).view(bs, topk, hidden)
        routed = (routed * route_w[:, :, None]).sum(dim=1)
        shared_gate, shared_up = torch.mm(h, shared_up_w).chunk(2, dim=-1)
        shared = torch.mm(F.silu(shared_gate.float()).to(h.dtype) * shared_up, shared_down_w)
        return (routed + shared).to(torch.bfloat16)

    def run() -> torch.Tensor:
        h = x
        for _ in range(n):
            h = _one(h)
        return h

    return BuiltCase(
        name=args.case,
        category="scaling",
        description=f"Synthetic MoE-only route/topk/expand/reduce skeleton, N={n}.",
        owner_count=n,
        projection_count=n,
        run=run,
        groups={
            "inputs": [x],
            "outputs": [],
            "workspaces": [],
            "weights": [gate_w, up_w, down_w, shared_up_w, shared_down_w],
            "caches": [],
            "metadata": [],
        },
        notes=["Does not use full expert checkpoint weights; route tensor sizes match decode topk pressure."],
    )


def _case_attention_skeleton(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    n = int(args.layers)
    rows = int(args.bs)
    heads = int(args.local_heads)
    dim = int(args.head_dim)
    q_seed = _randn((rows, heads, dim), device=device, dtype=torch.bfloat16)
    swa_slots = int(args.num_pages) * int(args.page_size)
    swa_cache = _randn((swa_slots, dim), device=device, dtype=torch.bfloat16)
    c4_cache = _randn((max(swa_slots // 4, 1), dim), device=device, dtype=torch.bfloat16)
    c128_cache = _randn((max(swa_slots // 128, int(args.c128_width)), dim), device=device, dtype=torch.bfloat16)
    swa_indices = _make_indices(rows, int(args.swa_width), swa_slots, device=device)
    c4_indices = _make_indices(rows, int(args.index_topk), c4_cache.shape[0], device=device, row_stride=23)
    c128_indices = _make_indices(rows, int(args.c128_width), c128_cache.shape[0], device=device, row_stride=5)
    sink = _randn((heads,), device=device, dtype=torch.float32)

    ratios = [0, 0] + [4, 128] * 20 + [4, 0]

    def run() -> torch.Tensor:
        q = q_seed
        for i in range(n):
            ratio = ratios[i % len(ratios)]
            if ratio == 4:
                out = _synthetic_two_source_attention(
                    q,
                    swa_cache,
                    swa_indices,
                    compressed_cache=c4_cache,
                    compressed_indices=c4_indices,
                    attn_sink=sink,
                )
            elif ratio == 128:
                out = _synthetic_two_source_attention(
                    q,
                    swa_cache,
                    swa_indices,
                    compressed_cache=c128_cache,
                    compressed_indices=c128_indices,
                    attn_sink=sink,
                )
            else:
                out = _synthetic_two_source_attention(q, swa_cache, swa_indices, attn_sink=sink)
            q = out
        return q

    return BuiltCase(
        name=args.case,
        category="scaling",
        description=f"Attention-only repeated-layer skeleton following DSV4 ratio pattern, N={n}.",
        owner_count=n,
        projection_count=n,
        run=run,
        groups={
            "inputs": [q_seed, sink],
            "outputs": [],
            "workspaces": [],
            "weights": [],
            "caches": [swa_cache, c4_cache, c128_cache],
            "metadata": [swa_indices, c4_indices, c128_indices],
        },
        notes=["Synthetic vectorized attention with DSV4 ratio pattern; no checkpoint or module construction."],
    )


def _case_attention_mlp_skeleton(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    n = int(args.layers)
    x = _randn((int(args.bs), int(args.hidden)), device=device, dtype=torch.bfloat16)
    proj_weights = _make_projection_weights(args, device)
    local_groups = int(args.local_groups)
    d_per_group = int(args.local_heads) * int(args.head_dim) // local_groups
    inter = int(args.moe_intermediate_per_rank)
    up_w = _randn((int(args.hidden), inter * 2), device=device, dtype=torch.bfloat16)
    down_w = _randn((inter, int(args.hidden)), device=device, dtype=torch.bfloat16)

    def run() -> torch.Tensor:
        h = x
        for _ in range(n):
            attn = _projection_block(
                h,
                proj_weights,
                local_groups=local_groups,
                d_per_group=d_per_group,
            )
            gate, up = torch.mm(h, up_w).chunk(2, dim=-1)
            mlp = torch.mm(F.silu(gate.float()).to(up.dtype) * up, down_w)
            h = (attn + mlp).to(torch.bfloat16)
        return h

    return BuiltCase(
        name=args.case,
        category="scaling",
        description=f"Attention projection + shared-MLP repeated-layer skeleton, N={n}.",
        owner_count=n,
        projection_count=n,
        run=run,
        groups={
            "inputs": [x],
            "outputs": [],
            "workspaces": [],
            "weights": [*proj_weights.values(), up_w, down_w],
            "caches": [],
            "metadata": [],
        },
        notes=["A compact composition probe; routed MoE is covered separately by moe_only."],
    )


def build_case(args: argparse.Namespace, device: torch.device) -> BuiltCase:
    case = args.case
    if case == "empty_graph":
        return _case_empty_graph(args, device)
    if case.startswith("copy_staging"):
        return _case_copy_staging(args, device)
    if case.startswith("elementwise"):
        return _case_elementwise(args, device)
    if case.startswith("bf16_matmul") or case.startswith("repeated_bf16_matmul"):
        return _case_repeated_matmul(args, device)
    if case.startswith("swa_attention"):
        args.owner = "swa"
        return _case_attention(args, device)
    if case.startswith("c4_sparse_attention") or case.startswith("c4a_attention"):
        args.owner = "c4"
        return _case_attention(args, device)
    if case.startswith("c128_attention"):
        args.owner = "c128"
        return _case_attention(args, device)
    if case.startswith("c4_indexer_topk"):
        return _case_indexer_topk(args, device)
    if case.startswith("indexer_topk_only"):
        return _case_indexer_topk_repeated(args, device)
    if case.startswith("qkv_norm_rope_cache_store"):
        return _case_qkv_norm_rope_store(args, device)
    if case.startswith("metadata_deforest"):
        return _case_metadata_deforest(args, device)
    if case.startswith("attention_only"):
        return _case_attention_skeleton(args, device)
    if case.startswith("moe_only"):
        return _case_moe_skeleton(args, device)
    if case.startswith("projection_only"):
        return _case_projection_skeleton(args, device)
    if case.startswith("attention_mlp"):
        return _case_attention_mlp_skeleton(args, device)
    raise ValueError(f"Unknown case: {case}")


def _normalize_case_args(args: argparse.Namespace) -> None:
    case = args.case
    if case.startswith("bf16_matmul"):
        args.layers = 1
    if "_prealloc" in case or case.endswith("_preallocated"):
        args.variant = "prealloc"
    elif "_keep_all" in case:
        args.variant = "keep_all"
    elif args.variant is None:
        args.variant = "out_of_place"
    for prefix in (
        "repeated_bf16_matmul_n",
        "attention_only_n",
        "moe_only_n",
        "projection_only_n",
        "attention_mlp_n",
        "indexer_topk_only_n",
    ):
        if case.startswith(prefix):
            suffix = case[len(prefix) :].split("_", 1)[0]
            if suffix.isdigit():
                args.layers = int(suffix)
    if "_bs16" in case:
        args.bs = 16
    elif "_bs1" in case:
        args.bs = 1


def _output_sanity(output: torch.Tensor | tuple[torch.Tensor, ...]) -> dict[str, Any]:
    tensors = list(output) if isinstance(output, tuple) else [output]
    first = tensors[0]
    finite = True
    checksums = []
    for item in tensors[:4]:
        if item.is_floating_point():
            finite = finite and bool(torch.isfinite(item).all().item())
        checksums.append(_safe_float(item.reshape(-1)[: min(item.numel(), 1024)]))
    return {
        "output_count": len(tensors),
        "first_output_shape": list(first.shape),
        "first_output_dtype": str(first.dtype),
        "finite": finite,
        "checksums": checksums,
    }


def run_single_case(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA graph private-pool micro attribution.")
    _normalize_case_args(args)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(f"cuda:{int(args.device)}")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    built = build_case(args, device)
    torch.cuda.synchronize(device)
    warmup_error = None
    warmup_sanity: dict[str, Any] | None = None
    warmup_start = time.perf_counter()
    try:
        with torch.inference_mode():
            warmup_out = built.run()
        torch.cuda.synchronize(device)
        warmup_sanity = _output_sanity(warmup_out)
    except Exception as exc:
        warmup_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
    warmup_elapsed = time.perf_counter() - warmup_start
    if warmup_error is not None:
        raise RuntimeError(f"Warmup failed for {built.name}: {warmup_error['message']}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    before = _mem_stats(device, "before")
    graph = torch.cuda.CUDAGraph()
    capture_error = None
    capture_elapsed = None
    captured_out: torch.Tensor | tuple[torch.Tensor, ...] | None = None
    try:
        capture_start = time.perf_counter()
        with torch.inference_mode(), torch.cuda.graph(graph):
            captured_out = built.run()
        torch.cuda.synchronize(device)
        capture_elapsed = time.perf_counter() - capture_start
    except Exception as exc:
        capture_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
    after = _mem_stats(device, "after")
    peaks = _peak_stats(device)

    replay_error = None
    replay_sanity: dict[str, Any] | None = None
    replay_elapsed = None
    if capture_error is None:
        try:
            replay_start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize(device)
            replay_elapsed = time.perf_counter() - replay_start
            assert captured_out is not None
            replay_sanity = _output_sanity(captured_out)
        except Exception as exc:
            replay_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }

    free_delta = before["before_free_bytes"] - after["after_free_bytes"]
    allocated_delta = after["after_allocated_bytes"] - before["before_allocated_bytes"]
    reserved_delta = after["after_reserved_bytes"] - before["before_reserved_bytes"]
    result: dict[str, Any] = {
        "case": built.name,
        "category": built.category,
        "description": built.description,
        "notes": built.notes,
        "device": int(args.device),
        "shape": {
            "bs": int(args.bs),
            "layers": int(args.layers),
            "variant": str(args.variant),
            "hidden": int(args.hidden),
            "local_heads": int(args.local_heads),
            "head_dim": int(args.head_dim),
            "seq_len": int(args.seq_len),
            "page_size": int(args.page_size),
            "num_pages": int(args.num_pages),
        },
        "owner_count": int(built.owner_count),
        "projection_count": int(built.projection_count),
        "projected_full_model_delta_bytes": int(free_delta * int(built.projection_count)),
        "warmup_elapsed_s": warmup_elapsed,
        "warmup_sanity": warmup_sanity,
        "capture_elapsed_s": capture_elapsed,
        "replay_elapsed_s": replay_elapsed,
        "replay_sanity": replay_sanity,
        "capture_error": capture_error,
        "replay_error": replay_error,
        **before,
        **after,
        **peaks,
        "free_delta_bytes": int(free_delta),
        "allocated_delta_bytes": int(allocated_delta),
        "reserved_delta_bytes": int(reserved_delta),
        **built.explicit_bytes(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "cwd": str(REPO_ROOT),
            "pid": os.getpid(),
        },
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def default_cases_for_suite(suite: str) -> list[str]:
    controls = [
        "empty_graph",
        "copy_staging_bs16",
        "elementwise_bs16",
        "elementwise_bs16_prealloc",
        "bf16_matmul_bs16",
        "bf16_matmul_bs16_prealloc",
        *[f"repeated_bf16_matmul_n{n}_bs16" for n in (1, 2, 4, 8, 16, 43)],
        *[f"repeated_bf16_matmul_n{n}_bs16_prealloc" for n in (1, 2, 4, 8, 16, 43)],
        "repeated_bf16_matmul_n43_bs16_keep_all",
    ]
    dsv4 = [
        "swa_attention_bs1",
        "swa_attention_bs16",
        "c4_sparse_attention_bs1",
        "c4_sparse_attention_bs16",
        "c128_attention_bs1",
        "c128_attention_bs16",
        "c4_indexer_topk_bs1",
        "c4_indexer_topk_bs16",
        "qkv_norm_rope_cache_store_bs1",
        "qkv_norm_rope_cache_store_bs16",
        "metadata_deforest_bs1",
        "metadata_deforest_bs16",
    ]
    scaling = []
    for family in ("attention_only", "projection_only", "moe_only", "attention_mlp"):
        scaling.extend([f"{family}_n{n}_bs16" for n in (1, 2, 4, 8, 16, 43)])
    scaling.extend([f"indexer_topk_only_n{n}_bs16" for n in (1, 2, 4, 8, 16, 21)])
    if suite == "controls":
        return controls
    if suite == "dsv4":
        return dsv4
    if suite == "scaling":
        return scaling
    if suite == "quick":
        return [
            "empty_graph",
            "copy_staging_bs16",
            "bf16_matmul_bs16",
            "repeated_bf16_matmul_n43_bs16",
            "swa_attention_bs16",
            "c4_sparse_attention_bs16",
            "c4_indexer_topk_bs16",
            "projection_only_n43_bs16",
            "moe_only_n43_bs16",
        ]
    if suite == "all":
        return controls + dsv4 + scaling
    raise ValueError(f"Unknown suite: {suite}")


def run_suite(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cases = args.cases or default_cases_for_suite(args.suite)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath = str(PYTHON_ROOT)
    if existing_pythonpath:
        pythonpath = pythonpath + os.pathsep + existing_pythonpath
    env["PYTHONPATH"] = pythonpath
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    failures = []
    for index, case in enumerate(cases, start=1):
        json_out = raw_dir / f"{case}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            case,
            "--json-out",
            str(json_out),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--bs",
            str(args.bs),
            "--hidden",
            str(args.hidden),
            "--seq-len",
            str(args.seq_len),
            "--page-size",
            str(args.page_size),
            "--num-pages",
            str(args.num_pages),
        ]
        print(f"[{index}/{len(cases)}] {case}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, capture_output=True)
        log_path = raw_dir / f"{case}.log"
        log_path.write_text(
            "COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            failures.append({"case": case, "returncode": proc.returncode, "log": str(log_path)})
            print(f"  failed rc={proc.returncode}; log={log_path}", flush=True)
        elif not json_out.exists():
            failures.append({"case": case, "returncode": proc.returncode, "log": str(log_path), "missing_json": True})
            print(f"  missing json; log={log_path}", flush=True)

    summary_script = Path(__file__).with_name("summarize_graph_private_pool_micro.py")
    if summary_script.exists():
        subprocess.run(
            [sys.executable, str(summary_script), "--milestone-dir", str(output_dir)],
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        )
    if failures:
        failure_path = output_dir / "summaries" / f"{args.suite}_failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} cases failed; see {failure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["quick", "controls", "dsv4", "scaling", "all"], help="Run a fresh-process suite.")
    parser.add_argument("--cases", nargs="*", help="Explicit cases for suite mode.")
    parser.add_argument("--case", help="Run a single case in this process.")
    parser.add_argument("--json-out")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--variant", choices=["out_of_place", "prealloc", "keep_all"], default=None)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-lora-rank", type=int, default=1024)
    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--index-heads", type=int, default=64)
    parser.add_argument("--index-head-dim", type=int, default=128)
    parser.add_argument("--local-groups", type=int, default=1)
    parser.add_argument("--o-lora-rank", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--n-routed-experts", type=int, default=256)
    parser.add_argument("--moe-intermediate-per-rank", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--swa-width", type=int, default=128)
    parser.add_argument("--index-topk", type=int, default=512)
    parser.add_argument("--c128-width", type=int, default=64)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    args = parser.parse_args()
    if bool(args.suite) == bool(args.case):
        parser.error("Specify exactly one of --suite or --case.")
    return args


def main() -> None:
    args = parse_args()
    if args.suite:
        run_suite(args)
        return
    result = run_single_case(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""TARGET 07.51 vLLM DeepSeek V4 FP8 backend isolation probe.

Run from the vLLM virtualenv:

    source /workspace/venvs/vllm-dsv4/bin/activate
    source /workspace/mini-sglang/performance_milestones/vllm/scripts/vllm_env.sh
    setup_vllm_runtime_env
    python /workspace/mini-sglang/performance_milestones/target07_vllm_fp8_backend_parity/scripts/vllm_fp8_backend_microbench.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

MINI_ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = Path(os.environ.get("TARGET0751_VLLM_ROOT", "/workspace/vllm-dsv4-docker"))
if VLLM_ROOT.exists() and str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

from vllm import _custom_ops as vllm_ops  # noqa: E402
from vllm.v1.attention.ops.deepseek_v4_ops import (  # noqa: E402
    dequantize_and_gather_k_cache,
    gather_dequant_two_scopes_with_mask,
    quantize_and_insert_k_cache,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (  # noqa: E402
    _fused_indexer_q_rope_quant_torch,
    fused_indexer_q_rope_quant,
)
from vllm.v1.attention.ops.mqa_logits_triton import (  # noqa: E402
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return ordered[idx]


def _time_cuda(fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": _p90(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _rounded_blocks(tokens: int, block_size: int) -> int:
    return max(1, math.ceil(max(tokens, 1) / block_size))


def _seq_lens(batch: int, history: int, *, device: torch.device) -> torch.Tensor:
    if batch == 1:
        return torch.tensor([history], device=device, dtype=torch.int32)
    return torch.linspace(
        history // 2,
        history,
        batch,
        device=device,
        dtype=torch.float32,
    ).to(torch.int32)


def _cos_sin_cache(max_pos: int, rope_dim: int, *, device: torch.device) -> torch.Tensor:
    half = rope_dim // 2
    pos = torch.arange(max_pos, dtype=torch.float32, device=device).unsqueeze(1)
    freq = torch.arange(half, dtype=torch.float32, device=device).unsqueeze(0)
    angles = pos / torch.pow(10000.0, (2.0 * freq) / max(rope_dim, 1))
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(torch.bfloat16)


def _block_table(batch: int, blocks_per_req: int, *, device: torch.device) -> torch.Tensor:
    return torch.arange(
        batch * blocks_per_req,
        device=device,
        dtype=torch.int32,
    ).reshape(batch, blocks_per_req)


def _prefix_sums(seq_lens: torch.Tensor) -> torch.Tensor:
    out = torch.empty(seq_lens.numel() + 1, device=seq_lens.device, dtype=torch.int32)
    out[0] = 0
    out[1:] = torch.cumsum(seq_lens, dim=0)
    return out


def _expected_rows_from_blocks(
    rows: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for req in range(seq_lens.numel()):
        length = int(seq_lens[req].item())
        if length <= 0:
            continue
        n_blocks = _rounded_blocks(length, block_size)
        phys = block_table[req, :n_blocks].long()
        pieces.append(rows.index_select(0, phys).reshape(-1, rows.shape[-1])[:length])
    if not pieces:
        return rows.new_empty((0, rows.shape[-1]))
    return torch.cat(pieces, dim=0)


def _finite_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    finite = torch.isfinite(reference) & torch.isfinite(actual)
    if not bool(finite.any()):
        return {"max_abs": 0.0, "mean_abs": 0.0}
    diff = (reference[finite].float() - actual[finite].float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def _topk_overlap(
    reference: torch.Tensor,
    actual: torch.Tensor,
    seq_lens: torch.Tensor,
    width: int,
) -> dict[str, float]:
    overlaps: list[float] = []
    rows = min(reference.shape[0], actual.shape[0], seq_lens.numel())
    for row in range(rows):
        valid = int(seq_lens[row].item())
        if valid <= 0:
            continue
        k = min(width, valid, reference.shape[1], actual.shape[1])
        ref_idx = torch.topk(reference[row, :valid], k=k, dim=-1).indices.detach().cpu()
        act_idx = torch.topk(actual[row, :valid], k=k, dim=-1).indices.detach().cpu()
        ref_set = {int(x) for x in ref_idx.tolist()}
        act_set = {int(x) for x in act_idx.tolist()}
        overlaps.append(len(ref_set & act_set) / max(len(ref_set), 1))
    if not overlaps:
        return {"mean": 1.0, "min": 1.0}
    return {"mean": float(statistics.fmean(overlaps)), "min": float(min(overlaps))}


def _rotate_q_exact(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    num_tokens, _, head_dim = q.shape
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    nope_dim = head_dim - rope_dim
    qf = q.to(torch.float32)
    cos_sin = cos_sin_cache.index_select(0, positions.long())
    cos = cos_sin[:, :half].view(num_tokens, 1, half).to(torch.float32)
    sin = cos_sin[:, half : 2 * half].view(num_tokens, 1, half).to(torch.float32)
    rope = qf[..., nope_dim:]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = (even * cos - odd * sin).to(torch.bfloat16).to(torch.float32)
    rotated[..., 1::2] = (odd * cos + even * sin).to(torch.bfloat16).to(torch.float32)
    if nope_dim > 0:
        return torch.cat([qf[..., :nope_dim], rotated], dim=-1)
    return rotated


def _bf16_paged_indexer_logits(
    q_exact: torch.Tensor,
    k_blocks: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_model_len: int,
) -> torch.Tensor:
    batch, heads, _ = q_exact.shape
    out = torch.full(
        (batch, max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device=q_exact.device,
    )
    for req in range(batch):
        length = int(seq_lens[req].item())
        if length <= 0:
            continue
        n_blocks = _rounded_blocks(length, block_size)
        phys = block_table[req, :n_blocks].long()
        k = k_blocks.index_select(0, phys).reshape(-1, k_blocks.shape[-1])[:length].float()
        score = torch.einsum("hd,nd->hn", q_exact[req], k).float()
        logits = (score.relu() * weights[req, :heads].unsqueeze(-1)).sum(dim=0)
        out[req, :length] = logits
    return out


def _bf16_prefill_indexer_logits(
    q_exact: torch.Tensor,
    gathered_k: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    batch, heads, _ = q_exact.shape
    total = gathered_k.shape[0]
    out = torch.full(
        (batch, total),
        float("-inf"),
        dtype=torch.float32,
        device=q_exact.device,
    )
    for req in range(batch):
        start = int(ks[req].item())
        end = int(ke[req].item())
        if end <= start:
            continue
        k = gathered_k[start:end].float()
        score = torch.einsum("hd,nd->hn", q_exact[req], k).float()
        logits = (score.relu() * weights[req, :heads].unsqueeze(-1)).sum(dim=0)
        out[req, start:end] = logits
    return out


def _run_persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    width: int,
    max_model_len: int,
) -> tuple[str, torch.Tensor]:
    out = torch.empty((logits.shape[0], width), device=logits.device, dtype=torch.int32)
    try:
        workspace = torch.empty(1024 * 1024, device=logits.device, dtype=torch.uint8)
        torch.ops._C.persistent_topk(logits, seq_lens, out, workspace, width, max_model_len)
        return "vllm_persistent_topk", out
    except Exception:
        values = torch.topk(logits, k=width, dim=-1).indices.to(torch.int32)
        out.copy_(values)
        return "torch_topk_fallback", out


def _pack_and_insert_indexer_cache(
    k_blocks: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> torch.Tensor:
    num_blocks = k_blocks.shape[0]
    cache_stride = head_dim + head_dim * 4 // head_dim
    kv_cache = torch.zeros(
        (num_blocks, block_size, cache_stride),
        dtype=torch.uint8,
        device=k_blocks.device,
    )
    k_flat = k_blocks.reshape(num_blocks * block_size, head_dim)
    slot_mapping = torch.arange(k_flat.shape[0], device=k_blocks.device, dtype=torch.int64)
    vllm_ops.indexer_k_quant_and_cache(k_flat, kv_cache, slot_mapping, head_dim, "ue8m0")
    return kv_cache


def _insert_indexer_cache_inplace(
    k_blocks: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    head_dim: int,
) -> None:
    vllm_ops.indexer_k_quant_and_cache(
        k_blocks.reshape(-1, head_dim),
        kv_cache,
        slot_mapping,
        head_dim,
        "ue8m0",
    )


def _indexer_case(
    *,
    batch: int,
    history: int,
    heads: int,
    head_dim: int,
    topk_width: int,
    block_size: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    blocks_per_req = _rounded_blocks(history, block_size)
    total_blocks = batch * blocks_per_req
    max_model_len = blocks_per_req * block_size
    seq_lens = _seq_lens(batch, history, device=device)
    block_table = _block_table(batch, blocks_per_req, device=device)
    positions = seq_lens.long() - 1
    rope_dim = 64

    q_bf16 = torch.randn(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
    k_blocks = torch.randn(
        total_blocks,
        block_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    weights_raw = torch.randn(batch, heads, device=device, dtype=torch.float32)
    softmax_scale = head_dim**-0.5
    head_scale = heads**-0.5
    weights_bf16 = weights_raw * softmax_scale * head_scale
    cos_sin = _cos_sin_cache(history + block_size + 1, rope_dim, device=device)

    q_ref, weights_ref = _fused_indexer_q_rope_quant_torch(
        positions,
        q_bf16,
        cos_sin,
        weights_raw,
        softmax_scale,
        head_scale,
    )
    q_quant, weights_fp8 = fused_indexer_q_rope_quant(
        positions,
        q_bf16,
        cos_sin,
        weights_raw,
        softmax_scale,
        head_scale,
    )
    kv_cache = _pack_and_insert_indexer_cache(k_blocks, block_size, head_dim)
    k_flat = k_blocks.reshape(total_blocks * block_size, head_dim)
    slot_mapping = torch.arange(k_flat.shape[0], device=device, dtype=torch.int64)
    kv_cache_for_store = torch.empty_like(kv_cache)

    cu_seq_lens = _prefix_sums(seq_lens)
    total_seq_len = int(cu_seq_lens[-1].item())
    gathered_k = torch.empty((total_seq_len, head_dim), dtype=torch.uint8, device=device)
    gathered_scale = torch.empty((total_seq_len, 4), dtype=torch.uint8, device=device)
    vllm_ops.cp_gather_indexer_k_quant_cache(
        kv_cache,
        gathered_k,
        gathered_scale,
        block_table,
        cu_seq_lens,
    )
    gathered_dequant = gathered_k.view(torch.float8_e4m3fn).float() * gathered_scale.view(
        torch.float32
    )
    expected_gathered = _expected_rows_from_blocks(k_blocks, block_table, seq_lens, block_size)

    q_exact = _rotate_q_exact(q_bf16, positions, cos_sin)
    bf16_logits = _bf16_paged_indexer_logits(
        q_exact,
        k_blocks,
        weights_bf16,
        seq_lens,
        block_table,
        block_size,
        max_model_len,
    )

    paged_kv = kv_cache.unsqueeze(-2)
    fp8_decode_logits = fp8_paged_mqa_logits_triton(
        q_quant.reshape(batch, 1, heads, head_dim),
        paged_kv,
        weights_fp8,
        seq_lens,
        block_table,
        max_model_len=max_model_len,
    )

    ks = cu_seq_lens[:-1]
    ke = cu_seq_lens[1:]
    fp8_prefill_logits = fp8_mqa_logits_triton(
        q_quant,
        (gathered_k.view(torch.float8_e4m3fn), gathered_scale.view(torch.float32)),
        weights_fp8,
        ks,
        ke,
    )
    bf16_prefill_logits = _bf16_prefill_indexer_logits(
        q_exact,
        expected_gathered,
        weights_bf16,
        ks,
        ke,
    )

    topk_backend, _ = _run_persistent_topk(
        fp8_decode_logits,
        seq_lens,
        topk_width,
        max_model_len,
    )

    def _k_store() -> torch.Tensor:
        _insert_indexer_cache_inplace(k_blocks, kv_cache_for_store, slot_mapping, head_dim)
        return kv_cache_for_store

    def _k_gather() -> None:
        vllm_ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            gathered_k,
            gathered_scale,
            block_table,
            cu_seq_lens,
        )

    def _decode_logits() -> torch.Tensor:
        return fp8_paged_mqa_logits_triton(
            q_quant.reshape(batch, 1, heads, head_dim),
            paged_kv,
            weights_fp8,
            seq_lens,
            block_table,
            max_model_len=max_model_len,
        )

    def _decode_select() -> tuple[str, torch.Tensor]:
        logits = _decode_logits()
        return _run_persistent_topk(logits, seq_lens, topk_width, max_model_len)

    def _prefill_logits() -> torch.Tensor:
        return fp8_mqa_logits_triton(
            q_quant,
            (gathered_k.view(torch.float8_e4m3fn), gathered_scale.view(torch.float32)),
            weights_fp8,
            ks,
            ke,
        )

    return {
        "case": {"batch": batch, "history": history},
        "shape": {
            "heads": heads,
            "head_dim": head_dim,
            "topk_width": topk_width,
            "block_size": block_size,
            "blocks_per_request": blocks_per_req,
            "max_model_len": max_model_len,
            "seq_lens": [int(x) for x in seq_lens.detach().cpu().tolist()],
            "total_seq_len": total_seq_len,
        },
        "backends": {
            "q_path": "fused_indexer_q_rope_quant",
            "k_store": "vllm._custom_ops.indexer_k_quant_and_cache",
            "k_gather": "vllm._custom_ops.cp_gather_indexer_k_quant_cache",
            "decode_logits": "fp8_paged_mqa_logits_triton",
            "prefill_logits": "fp8_mqa_logits_triton",
            "topk": topk_backend,
        },
        "timings": {
            "q_rope_quant_ms": _time_cuda(
                lambda: fused_indexer_q_rope_quant(
                    positions,
                    q_bf16,
                    cos_sin,
                    weights_raw,
                    softmax_scale,
                    head_scale,
                ),
                warmup=warmup,
                iters=iters,
            ),
            "k_quant_store_full_cache_ms": _time_cuda(
                _k_store,
                warmup=warmup,
                iters=iters,
            ),
            "k_cp_gather_cache_ms": _time_cuda(_k_gather, warmup=warmup, iters=iters),
            "decode_paged_logits_ms": _time_cuda(
                _decode_logits,
                warmup=warmup,
                iters=iters,
            ),
            "decode_paged_logits_plus_topk_ms": _time_cuda(
                _decode_select,
                warmup=warmup,
                iters=iters,
            ),
            "prefill_gathered_logits_ms": _time_cuda(
                _prefill_logits,
                warmup=warmup,
                iters=iters,
            ),
        },
        "quality": {
            "q_vs_torch_ref": {
                "q_byte_exact": bool(
                    torch.equal(q_quant.view(torch.uint8), q_ref.view(torch.uint8))
                ),
                "weights": _finite_error(weights_ref, weights_fp8),
            },
            "k_cache_dequant_vs_bf16": _finite_error(expected_gathered, gathered_dequant),
            "decode_logits_vs_bf16": _finite_error(bf16_logits, fp8_decode_logits),
            "decode_topk_overlap_vs_bf16": _topk_overlap(
                bf16_logits,
                fp8_decode_logits,
                seq_lens,
                topk_width,
            ),
            "prefill_logits_vs_decode_finite": _finite_error(
                bf16_prefill_logits,
                fp8_prefill_logits,
            ),
        },
    }


def _pack_fp8_ds_mla_cache_ref(
    k_blocks: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    num_blocks = k_blocks.shape[0]
    k_cache = torch.zeros(
        num_blocks,
        block_size,
        584,
        dtype=torch.uint8,
        device=k_blocks.device,
    )

    nope = k_blocks[..., :448].float().reshape(num_blocks, block_size, 7, 64)
    rope = k_blocks[..., 448:].contiguous()
    amax = nope.abs().amax(dim=-1).clamp_min(1e-4)
    exponents = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scales = torch.exp2(exponents).unsqueeze(-1)
    fp8_nope = (nope / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    scale_u8 = (exponents + 127.0).to(torch.uint8)

    flat = k_cache.view(num_blocks, -1)
    data = flat[:, : block_size * 576].view(num_blocks, block_size, 576)
    scale_region = flat[:, block_size * 576 :].view(num_blocks, block_size, 8)
    data[:, :, :448] = fp8_nope.reshape(num_blocks, block_size, 448).view(torch.uint8)
    data[:, :, 448:576] = rope.view(torch.uint8).reshape(num_blocks, block_size, 128)
    scale_region[:, :, :7] = scale_u8
    return k_cache


def _insert_fp8_ds_mla_cache_inplace(
    k_blocks: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    quantize_and_insert_k_cache(
        k_blocks.reshape(-1, 512),
        k_cache.view(k_cache.shape[0], -1),
        slot_mapping,
        block_size=block_size,
    )


def _flat_indices(
    *,
    batch: int,
    blocks_per_req: int,
    block_size: int,
    topk: int,
    seq_lens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    out = torch.empty((batch, topk), dtype=torch.int64, device=device)
    for req in range(batch):
        length = int(seq_lens[req].item())
        base = req * blocks_per_req * block_size
        out[req] = base + torch.randint(0, max(length, 1), (topk,), device=device)
    return out


def _expected_from_flat_indices(
    k_blocks: torch.Tensor,
    indices: torch.Tensor,
    block_size: int,
    topk_lens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = k_blocks.reshape(-1, k_blocks.shape[-1])
    safe = indices.clamp_min(0).clamp_max(flat.shape[0] - 1)
    gathered = flat.index_select(0, safe.reshape(-1)).reshape(
        indices.shape[0],
        indices.shape[1],
        flat.shape[-1],
    )
    invalid = indices < 0
    if topk_lens is not None:
        slots = torch.arange(indices.shape[1], device=indices.device).view(1, -1)
        invalid = invalid | (slots >= topk_lens.view(-1, 1))
    return gathered, invalid


def _fp8_ds_mla_case(
    *,
    batch: int,
    history: int,
    block_size: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    swa_topk = 128
    extra_topk = 512
    seq_lens = _seq_lens(batch, history, device=device)

    swa_blocks_per_req = _rounded_blocks(history, block_size)
    c4_history = max(1, history // 4)
    c4_lens = torch.clamp(seq_lens // 4, min=1)
    c4_blocks_per_req = _rounded_blocks(c4_history, block_size)

    swa_blocks = batch * swa_blocks_per_req
    c4_blocks = batch * c4_blocks_per_req
    swa_k_blocks = torch.randn(
        swa_blocks,
        block_size,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    c4_k_blocks = torch.randn(
        c4_blocks,
        block_size,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    swa_cache = _pack_fp8_ds_mla_cache_ref(swa_k_blocks, block_size)
    c4_cache = _pack_fp8_ds_mla_cache_ref(c4_k_blocks, block_size)
    swa_cache_for_insert = torch.empty_like(swa_cache)
    swa_slot_mapping = torch.arange(
        swa_blocks * block_size,
        dtype=torch.int64,
        device=device,
    )
    swa_indices = _flat_indices(
        batch=batch,
        blocks_per_req=swa_blocks_per_req,
        block_size=block_size,
        topk=swa_topk,
        seq_lens=seq_lens,
        device=device,
    )
    extra_indices = _flat_indices(
        batch=batch,
        blocks_per_req=c4_blocks_per_req,
        block_size=block_size,
        topk=extra_topk,
        seq_lens=c4_lens,
        device=device,
    )
    swa_lens = torch.full((batch,), swa_topk, dtype=torch.int32, device=device)
    extra_lens = torch.full((batch,), extra_topk, dtype=torch.int32, device=device)

    gathered, invalid = gather_dequant_two_scopes_with_mask(
        swa_cache,
        block_size,
        swa_indices,
        swa_lens,
        c4_cache,
        block_size,
        extra_indices,
        extra_lens,
        448,
        64,
        512,
    )
    exp_swa, inv_swa = _expected_from_flat_indices(swa_k_blocks, swa_indices, block_size, swa_lens)
    exp_c4, inv_c4 = _expected_from_flat_indices(c4_k_blocks, extra_indices, block_size, extra_lens)
    expected = torch.cat([exp_swa, exp_c4], dim=1)
    expected_invalid = torch.cat([inv_swa, inv_c4], dim=1)
    valid = ~expected_invalid

    full_out = torch.empty(
        batch,
        int(seq_lens.max().item()),
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    swa_block_table = _block_table(batch, swa_blocks_per_req, device=device)

    def _two_scope_gather() -> tuple[torch.Tensor, torch.Tensor]:
        return gather_dequant_two_scopes_with_mask(
            swa_cache,
            block_size,
            swa_indices,
            swa_lens,
            c4_cache,
            block_size,
            extra_indices,
            extra_lens,
            448,
            64,
            512,
        )

    def _full_dequant_gather() -> None:
        dequantize_and_gather_k_cache(
            full_out,
            swa_cache,
            seq_lens,
            None,
            swa_block_table,
            block_size,
            offset=0,
        )

    def _quant_insert_swa() -> torch.Tensor:
        _insert_fp8_ds_mla_cache_inplace(
            swa_k_blocks,
            swa_cache_for_insert,
            swa_slot_mapping,
            block_size,
        )
        return swa_cache_for_insert

    nope_diff = (gathered[..., :448][valid] - expected[..., :448][valid]).float().abs()
    rope_diff = (gathered[..., 448:][valid] - expected[..., 448:][valid]).float().abs()

    full_status: dict[str, Any]
    try:
        _full_dequant_gather()
        full_status = {
            "status": "pass",
            "timing": _time_cuda(_full_dequant_gather, warmup=warmup, iters=iters),
        }
    except Exception as exc:
        full_status = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    try:
        _quant_insert_swa()
        quant_insert_status = {
            "status": "pass",
            "timing": _time_cuda(_quant_insert_swa, warmup=warmup, iters=iters),
        }
    except Exception as exc:
        quant_insert_status = {
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "case": {"batch": batch, "history": history},
        "shape": {
            "block_size": block_size,
            "head_dim": 512,
            "nope_dim": 448,
            "rope_dim": 64,
            "token_bytes": 584,
            "swa_topk": swa_topk,
            "extra_topk": extra_topk,
            "seq_lens": [int(x) for x in seq_lens.detach().cpu().tolist()],
            "swa_blocks_per_request": swa_blocks_per_req,
            "c4_blocks_per_request": c4_blocks_per_req,
        },
        "backends": {
            "quant_insert": "quantize_and_insert_k_cache",
            "two_scope_gather": "gather_dequant_two_scopes_with_mask",
            "full_gather": "dequantize_and_gather_k_cache",
        },
        "timings": {
            "kv_quant_insert_swa_full_cache": quant_insert_status,
            "two_scope_gather_dequant_topk_ms": _time_cuda(
                _two_scope_gather,
                warmup=warmup,
                iters=iters,
            ),
            "full_dequantize_and_gather_k_cache": full_status,
        },
        "quality": {
            "invalid_mask_exact": bool(torch.equal(invalid, expected_invalid)),
            "nope_dequant_error": {
                "max_abs": float(nope_diff.max().item()) if nope_diff.numel() else 0.0,
                "mean_abs": float(nope_diff.mean().item()) if nope_diff.numel() else 0.0,
            },
            "rope_error": {
                "max_abs": float(rope_diff.max().item()) if rope_diff.numel() else 0.0,
                "mean_abs": float(rope_diff.mean().item()) if rope_diff.numel() else 0.0,
            },
        },
    }


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    mini_0750 = {
        (1, 1024): {
            "bf16_logits_ms": 0.1270751953125,
            "fp8_logits_ms": 0.1572543978691101,
            "bf16_select_ms": 0.30928959846496584,
            "fp8_select_ms": 0.3181663990020752,
        },
        (4, 2048): {
            "bf16_logits_ms": 0.12463359832763672,
            "fp8_logits_ms": 0.2654752016067505,
            "bf16_select_ms": 0.3036511898040771,
            "fp8_select_ms": 0.3162751913070679,
        },
        (16, 4096): {
            "bf16_logits_ms": 0.30763840675354004,
            "fp8_logits_ms": 1.3072352409362793,
            "bf16_select_ms": 0.35858879089355467,
            "fp8_select_ms": 1.7368160247802735,
        },
    }
    rows: list[dict[str, Any]] = []
    for case in payload["indexer_cases"]:
        key = (case["case"]["batch"], case["case"]["history"])
        mini = mini_0750[key]
        vllm_logits = case["timings"]["decode_paged_logits_ms"]["mean_ms"]
        vllm_select = case["timings"]["decode_paged_logits_plus_topk_ms"]["mean_ms"]
        rows.append(
            {
                "batch": key[0],
                "history": key[1],
                "mini_bf16_logits_ms": mini["bf16_logits_ms"],
                "mini_fp8_logits_ms": mini["fp8_logits_ms"],
                "vllm_fp8_decode_logits_ms": vllm_logits,
                "mini_bf16_select_ms": mini["bf16_select_ms"],
                "mini_fp8_select_ms": mini["fp8_select_ms"],
                "vllm_fp8_decode_logits_plus_topk_ms": vllm_select,
                "vllm_logits_vs_mini_bf16_speedup": mini["bf16_logits_ms"] / vllm_logits,
                "vllm_select_vs_mini_bf16_speedup": mini["bf16_select_ms"] / vllm_select,
                "topk_overlap_mean": case["quality"]["decode_topk_overlap_vs_bf16"]["mean"],
                "logits_mean_abs": case["quality"]["decode_logits_vs_bf16"]["mean_abs"],
            }
        )
    gather_rows: list[dict[str, Any]] = []
    for case in payload["fp8_ds_mla_cases"]:
        full = case["timings"]["full_dequantize_and_gather_k_cache"]
        gather_rows.append(
            {
                "batch": case["case"]["batch"],
                "history": case["case"]["history"],
                "two_scope_gather_dequant_topk_ms": case["timings"][
                    "two_scope_gather_dequant_topk_ms"
                ]["mean_ms"],
                "full_dequantize_and_gather_status": full["status"],
                "full_dequantize_and_gather_ms": (
                    full["timing"]["mean_ms"] if full["status"] == "pass" else None
                ),
                "nope_mean_abs": case["quality"]["nope_dequant_error"]["mean_abs"],
                "rope_max_abs": case["quality"]["rope_error"]["max_abs"],
            }
        )
    return {
        "suite": payload["suite"] + "_summary",
        "mini_0750_baseline_source": "performance_milestones/target07_fp8_cache_indexer_precision/summaries/target0750_indexer_fp8_summary.json",
        "indexer_comparison": rows,
        "fp8_ds_mla_gather": gather_rows,
        "decision_hint": (
            "port/adapt indexer only if vLLM logits/select beats mini bf16 by >=20%; "
            "probe fp8_ds_mla only if gather/dequant is clearly below mini bf16 sparse boundary."
        ),
    }


def _env_info() -> dict[str, Any]:
    ablation_env = {
        "VLLM_DSV4_ABLATE_AUX_STREAM": os.environ.get("VLLM_DSV4_ABLATE_AUX_STREAM"),
        "VLLM_DSV4_ABLATE_PERSISTENT_TOPK": os.environ.get("VLLM_DSV4_ABLATE_PERSISTENT_TOPK"),
    }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "vllm_root": str(VLLM_ROOT),
        "ablation_env": ablation_env,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=MINI_ROOT
        / "performance_milestones"
        / "target07_vllm_fp8_backend_parity"
        / "raw"
        / "vllm_fp8_backend_microbench.json",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=MINI_ROOT
        / "performance_milestones"
        / "target07_vllm_fp8_backend_parity"
        / "summaries"
        / "vllm_fp8_backend_microbench_summary.json",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability(0) != (8, 0):
        raise SystemExit(f"sm80/A100 is required, got {torch.cuda.get_device_capability(0)}")
    for key in ("VLLM_DSV4_ABLATE_AUX_STREAM", "VLLM_DSV4_ABLATE_PERSISTENT_TOPK"):
        if os.environ.get(key):
            raise SystemExit(f"{key} is set; unset ablation envs for TARGET 07.51")

    torch.cuda.set_device(0)
    torch.manual_seed(20260751)
    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 3)
    iters = args.iters if args.iters is not None else (3 if args.quick else 10)
    cases = [(1, 1024), (4, 2048), (16, 4096)]
    if args.quick:
        cases = cases[:2]

    started = time.time()
    payload = {
        "suite": "target07_51_vllm_fp8_backend_microbench",
        "started_at_unix": started,
        "env": _env_info(),
        "parameters": {
            "warmup": warmup,
            "iters": iters,
            "block_size": args.block_size,
            "indexer_heads": 64,
            "indexer_head_dim": 128,
            "topk_width": 512,
        },
        "indexer_cases": [
            _indexer_case(
                batch=batch,
                history=history,
                heads=64,
                head_dim=128,
                topk_width=512,
                block_size=args.block_size,
                warmup=warmup,
                iters=iters,
            )
            for batch, history in cases
        ],
        "fp8_ds_mla_cases": [
            _fp8_ds_mla_case(
                batch=batch,
                history=history,
                block_size=args.block_size,
                warmup=warmup,
                iters=iters,
            )
            for batch, history in cases
        ],
        "elapsed_s": time.time() - started,
    }
    summary = _summarize(payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

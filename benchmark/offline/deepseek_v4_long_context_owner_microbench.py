#!/usr/bin/env python3
"""Production-shape microbench for the two leading DSV4 long-prefill owners.

This benchmark deliberately measures the v0.0.0 release boundaries.  It does
not introduce a candidate kernel: the indexer case exercises the packed FP8
paged cache plus bounded logits/top-k path, and the C128 case exercises the
BF16 two-source sparse-attention path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


def _pctl(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _time_cuda(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, Any]:
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
        "p10_ms": _pctl(samples, 0.10),
        "p90_ms": _pctl(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _memory_probe(fn: Callable[[], Any]) -> dict[str, int]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    output = fn()
    torch.cuda.synchronize()
    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    del output
    return {
        "before_allocated_bytes": before_allocated,
        "after_allocated_bytes": after_allocated,
        "peak_allocated_bytes": peak_allocated,
        "gross_call_high_water_bytes": max(0, peak_allocated - before_allocated),
        "temporary_high_water_bytes": max(
            0, peak_allocated - max(before_allocated, after_allocated)
        ),
        "before_reserved_bytes": before_reserved,
        "after_reserved_bytes": after_reserved,
    }


def _profile_launch_count(fn: Callable[[], Any]) -> dict[str, Any]:
    # Profiler timing is intentionally discarded; it is only a launch census.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    cuda_events = [
        event
        for event in prof.events()
        if str(getattr(event, "device_type", "")).lower().endswith("cuda")
    ]
    names: dict[str, int] = {}
    for event in cuda_events:
        names[event.name] = names.get(event.name, 0) + 1
    return {
        "cuda_launches": len(cuda_events),
        "cuda_kernel_launches": sum(
            count for name, count in names.items() if not name.startswith("Memcpy ")
        ),
        "cuda_kernel_kinds": len(names),
        "top_cuda_kernels": sorted(names.items(), key=lambda item: (-item[1], item[0]))[:12],
    }


def _exact_topk_sets(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape:
        return False
    a_sorted = torch.sort(a, dim=-1).values
    b_sorted = torch.sort(b, dim=-1).values
    return bool(torch.equal(a_sorted, b_sorted))


def _indexer_case(
    context: int, *, query_rows: int, warmup: int, iters: int, profile: bool
) -> dict[str, Any]:
    model_page_size = 256
    ratio = 4
    page_size = model_page_size // ratio
    c4_context = context // ratio
    prefix = context - query_rows
    c4_lens = torch.div(
        torch.arange(prefix + 1, context + 1, device="cuda", dtype=torch.int64),
        ratio,
        rounding_mode="floor",
    ).clamp_min(1).to(torch.int32)
    pages = math.ceil(c4_context / page_size)
    page_row = torch.arange(pages, device="cuda", dtype=torch.int32)
    page_table = page_row.unsqueeze(0).expand(query_rows, -1).contiguous()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(1261000 + context)
    q_bf16 = torch.randn(
        query_rows, 64, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    raw_weights = torch.randn(
        query_rows, 64, device="cuda", dtype=torch.float32, generator=generator
    )
    positions = torch.arange(prefix, context, device="cuda", dtype=torch.int64)
    q = dsv4_kernel.indexer_q_rope_fp8_fallback(
        q_bf16,
        raw_weights,
        positions,
        rotary_dim=0,
        base=10000.0,
        softmax_scale=128**-0.5,
        head_scale=64**-0.5,
    )
    cache_bf16 = torch.randn(
        pages * page_size, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    cache_values, cache_scales = dsv4_kernel.quantize_indexer_fp8_cache_ref(cache_bf16)
    packed_cache = dsv4_kernel.pack_indexer_fp8_paged_cache_ref(
        cache_values, cache_scales, page_size=page_size
    )
    del cache_bf16, cache_values, cache_scales, q_bf16, raw_weights

    def run():
        return dsv4_kernel.indexer_select_fp8_paged_fallback(
            q.q_values,
            q.weights,
            packed_cache,
            c4_lens,
            page_table,
            page_size=page_size,
            width=512,
            ratio=ratio,
        )

    actual = run()
    # The independent oracle uses the same paged logits and top-k primitives,
    # but on 16 rows so it cannot enter bounded slicing.  This checks that the
    # production bounded composition preserves exact index sets.
    oracle_rows = 16
    oracle = dsv4_kernel.indexer_select_fp8_paged_fallback(
        q.q_values[-oracle_rows:],
        q.weights[-oracle_rows:],
        packed_cache,
        c4_lens[-oracle_rows:],
        page_table[-oracle_rows:],
        page_size=page_size,
        width=512,
        ratio=ratio,
    )
    parity = _exact_topk_sets(
        actual.topk.raw_indices[-oracle_rows:], oracle.topk.raw_indices
    )
    backend = actual.backend
    del actual, oracle

    timing = _time_cuda(run, warmup, iters)
    memory = _memory_probe(run)
    launches = _profile_launch_count(run) if profile else {"cuda_launches": None}

    max_logits_bytes = 512 * 1024 * 1024
    is_bounded = query_rows * c4_context * 4 > max_logits_bytes
    slice_rows = (
        max(1, max_logits_bytes // (c4_context * 4)) if is_bounded else query_rows
    )
    slices = math.ceil(query_rows / slice_rows)
    valid_pairs = int(c4_lens.to(torch.int64).sum().item())
    flops = valid_pairs * 64 * (2 * 128 + 2)
    # Algorithmic traffic floor with perfect K reuse across index heads: one
    # packed K vector and one FP32 logits write+read per query/key pair.
    # Hardware traffic can be higher; this is not an HBM measurement.
    traffic_floor = valid_pairs * (132 + 8)
    seconds = timing["median_ms"] / 1000.0
    return {
        "context_tokens": context,
        "query_rows": query_rows,
        "compressed_context": c4_context,
        "page_size_model": model_page_size,
        "page_size_indexer": page_size,
        "page_table_shape": list(page_table.shape),
        "page_table_bytes": page_table.numel() * page_table.element_size(),
        "packed_cache_shape": list(packed_cache.shape),
        "packed_cache_bytes": packed_cache.numel(),
        "topk": 512,
        "logits_dtype": "float32",
        "full_logits_bytes_if_materialized": query_rows * c4_context * 4,
        "workspace_budget_bytes": max_logits_bytes,
        "bounded": is_bounded,
        "bounded_slice_rows": slice_rows,
        "bounded_slices": slices,
        "backend": backend,
        "correctness": {"oracle_rows": oracle_rows, "exact_topk_set_parity": parity},
        "timing": timing,
        "launch_census": launches,
        "memory": memory,
        "roofline": {
            "logical_flops": flops,
            "traffic_floor_bytes": traffic_floor,
            "arithmetic_intensity_flop_per_byte": flops / traffic_floor,
            "achieved_tflop_s": flops / seconds / 1e12,
            "achieved_traffic_floor_gb_s": traffic_floor / seconds / 1e9,
            "note": "algorithmic lower bound with perfect cross-head reuse; not measured DRAM bytes",
        },
    }


def _c128_case(
    context: int, *, query_rows: int, warmup: int, iters: int, profile: bool
) -> dict[str, Any]:
    heads = 8
    dim = 512
    swa_width = 128
    c128_width = max(1, context // 128)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1262000 + context)
    q = torch.randn(
        query_rows, heads, dim, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    swa_cache = torch.randn(
        swa_width, dim, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    compressed_cache = torch.randn(
        c128_width, dim, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    swa_indices = torch.arange(swa_width, device="cuda", dtype=torch.int32).expand(
        query_rows, -1
    ).contiguous()
    compressed_indices = torch.arange(
        c128_width, device="cuda", dtype=torch.int32
    ).expand(query_rows, -1).contiguous()
    swa_lengths = torch.full((query_rows,), swa_width, device="cuda", dtype=torch.int32)
    compressed_lengths = torch.full(
        (query_rows,), c128_width, device="cuda", dtype=torch.int32
    )
    attn_sink = torch.randn(heads, device="cuda", dtype=torch.float32, generator=generator)

    def run():
        out = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
            q,
            swa_cache,
            swa_indices,
            swa_lengths,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            compressed_lengths=compressed_lengths,
            softmax_scale=dim**-0.5,
            attn_sink=attn_sink,
        )
        if out is None:
            raise RuntimeError("sm80 two-source attention backend did not dispatch")
        return out

    actual = run()
    # Full manual reference is intentionally limited to one row: it has the
    # exact production key width without materializing a second 8192-row graph.
    ref_q = q[:1].float()
    candidates = torch.cat((compressed_cache, swa_cache), dim=0).float()
    scores = torch.einsum("bhd,td->bht", ref_q, candidates) * (dim**-0.5)
    sink = attn_sink.float()[None, :]
    max_score = torch.maximum(scores.max(dim=-1).values, sink)
    exp_scores = torch.exp(scores - max_score[..., None])
    denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
    reference = torch.einsum("bht,td->bhd", exp_scores / denom[..., None], candidates)
    error = (actual[:1].float() - reference).abs()
    correctness = {
        "oracle_rows": 1,
        "allclose_atol_rtol_6e_2": bool(
            torch.allclose(actual[:1].float(), reference, atol=6e-2, rtol=6e-2)
        ),
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }
    del actual, reference, scores, exp_scores, candidates, ref_q
    timing = _time_cuda(run, warmup, iters)
    memory = _memory_probe(run)
    launches = _profile_launch_count(run) if profile else {"cuda_launches": None}
    keys = c128_width + swa_width
    flops = query_rows * heads * keys * (4 * dim + 5)
    # Perfect reuse across heads: read each BF16 candidate for score and value.
    traffic_floor = query_rows * keys * dim * 2 * 2
    seconds = timing["median_ms"] / 1000.0
    return {
        "context_tokens": context,
        "query_rows": query_rows,
        "heads": heads,
        "head_dim": dim,
        "c128_key_count": c128_width,
        "swa_key_count": swa_width,
        "total_key_count": keys,
        "metadata_shape": list(compressed_indices.shape),
        "metadata_bytes": compressed_indices.numel() * compressed_indices.element_size(),
        "backend": "dsv4_sparse_attention_two_source_bf16",
        "correctness": correctness,
        "timing": timing,
        "launch_census": launches,
        "memory": memory,
        "roofline": {
            "logical_flops": flops,
            "traffic_floor_bytes": traffic_floor,
            "arithmetic_intensity_flop_per_byte": flops / traffic_floor,
            "achieved_tflop_s": flops / seconds / 1e12,
            "achieved_traffic_floor_gb_s": traffic_floor / seconds / 1e9,
            "note": "algorithmic lower bound with perfect cross-head reuse; not measured DRAM bytes",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="16384,65536,131072,262144,524288")
    parser.add_argument("--query-rows", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--owner", choices=("both", "indexer", "c128"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This production-shape benchmark requires an sm80 CUDA device")
    if args.query_rows != 8192:
        raise SystemExit("The release contract fixes --query-rows at 8192")

    os.environ[dsv4_kernel.DSV4_SM80_INDEXER_FP8_CACHE_TOGGLE] = "1"
    os.environ[dsv4_kernel.DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE] = "1"
    os.environ["MINISGL_DSV4_SM80_TOPK"] = "1"
    os.environ["MINISGL_DSV4_SM80_SPARSE_ATTN_BF16"] = "1"
    os.environ[dsv4_kernel.DSV4_INDEXER_MAX_LOGITS_MB_ENV] = "512"

    contexts = [int(value) for value in args.contexts.split(",") if value]
    started = time.time()
    output: dict[str, Any] = {
        "suite": "target12_61_long_context_owner_production_shape",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "contract": {
            "model_page_size": 256,
            "chunk_size": 8192,
            "contexts": contexts,
            "timing": "per-iteration CUDA events after warmup",
            "profiler": "launch census only; profiler latency is discarded",
        },
    }
    if args.owner in ("both", "indexer"):
        output["indexer"] = [
            _indexer_case(
                context,
                query_rows=args.query_rows,
                warmup=args.warmup,
                iters=args.iters,
                profile=not args.skip_profiler,
            )
            for context in contexts
        ]
    if args.owner in ("both", "c128"):
        output["c128_attention"] = [
            _c128_case(
                context,
                query_rows=args.query_rows,
                warmup=args.warmup,
                iters=args.iters,
                profile=not args.skip_profiler,
            )
            for context in contexts
        ]
    output["elapsed_s"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

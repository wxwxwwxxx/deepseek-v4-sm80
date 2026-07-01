#!/usr/bin/env python3
"""Mini-sglang DeepSeek V4 sm80 subgraph probes.

The goal is not to reproduce the full engine path.  Each probe keeps the
shape/dtype boundary close enough to compare with vLLM or to document why the
boundary does not match cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


DEFAULT_TOGGLES = {
    "MINISGL_DSV4_SM80_RMSNORM": "1",
    "MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE": "1",
    "MINISGL_DSV4_SM80_KV_BF16": "1",
    "MINISGL_DSV4_SM80_SPARSE_ATTN_BF16": "1",
    "MINISGL_DSV4_SM80_INDEXER_BF16": "1",
    "MINISGL_DSV4_SM80_TOPK": "1",
    "MINISGL_DSV4_SM80_MOE_ROUTE": "1",
    "MINISGL_DSV4_SM80_SWIGLU": "1",
    "MINISGL_DSV4_SM80_HC": "1",
}

for key, value in DEFAULT_TOGGLES.items():
    os.environ.setdefault(key, value)

import torch
import torch.nn.functional as F

from minisgl.kernel.deepseek_v4 import (
    dsv4_sparse_attention_two_source_bf16,
    e8m0_dtype,
    hc_head_fallback,
    indexer_select_bf16_fallback,
    k_norm_rope_cache_fallback,
    moe_route_dispatch_bf16_grouped,
    q_kv_norm_rope_cache_fallback,
    q_norm_rope_fallback,
    rms_norm_fallback,
    silu_and_mul_clamp_fallback,
)


def p90(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return ordered[idx]


def cuda_event_bench(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "name": name,
        "status": "pass",
        "warmup": warmup,
        "repeat": repeat,
        "cuda_sync": "torch.cuda.synchronize after every measured iteration",
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": p90(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def blocker(name: str, reason: str, **extra: Any) -> dict[str, Any]:
    out = {
        "name": name,
        "status": "blocked",
        "reason": reason,
    }
    out.update(extra)
    return out


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def attention_front_cache_insert(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    fused_weight = torch.randn(1536, 4096, device=device, dtype=torch.bfloat16) * 0.01
    q_a_weight = torch.ones(1024, device=device, dtype=torch.bfloat16)
    kv_weight = torch.ones(512, device=device, dtype=torch.bfloat16)
    wq_b_weight = torch.randn(4096, 1024, device=device, dtype=torch.bfloat16) * 0.01
    positions = torch.arange(tokens, device=device, dtype=torch.long) + 4096
    cache = torch.empty(max(tokens + 4096, 8192), 512, device=device, dtype=torch.bfloat16)
    out_loc = torch.arange(tokens, device=device, dtype=torch.int32)

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        qr_kv = F.linear(hidden, fused_weight)
        qr, kv = qr_kv.split([1024, 512], dim=-1)
        qr = rms_norm_fallback(qr, q_a_weight, eps=1.0e-6)
        q = F.linear(qr, wq_b_weight).view(tokens, 8, 512)
        fused = q_kv_norm_rope_cache_fallback(
            q,
            kv,
            positions,
            norm_weight=kv_weight,
            rms_norm_eps=1.0e-6,
            cache=cache,
            out_loc=out_loc,
            rotary_dim=64,
            base=10000.0,
        )
        if not fused:
            q_norm_rope_fallback(
                q,
                positions,
                rms_norm_eps=1.0e-6,
                rotary_dim=64,
                base=10000.0,
            )
            k_norm_rope_cache_fallback(
                kv,
                positions,
                norm_weight=kv_weight,
                rms_norm_eps=1.0e-6,
                cache=cache,
                out_loc=out_loc,
                rotary_dim=64,
                base=10000.0,
            )
        return q, kv

    result = cuda_event_bench(
        f"attention_front_cache_insert_tokens{tokens}",
        fn,
        warmup=warmup,
        repeat=repeat,
    )
    result.update(
        {
            "subgraph": "attention_front_projection_and_cache_insert",
            "tokens": tokens,
            "shape": {
                "hidden": [tokens, 4096],
                "fused_wqa_wkv_weight": [1536, 4096],
                "wq_b_weight": [4096, 1024],
                "q": [tokens, 8, 512],
                "kv_cache": list(cache.shape),
            },
            "dtype": "bf16 direct, bf16 KV cache",
            "boundary_note": "fused wq_a+wkv F.linear, q_a RMSNorm, wq_b, q head norm/RoPE, kv norm/RoPE/cache store",
        }
    )
    return result


def sparse_attention(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    cache_slots = max(8192, tokens + 4096)
    q = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16)
    swa_cache = torch.randn(cache_slots, 512, device=device, dtype=torch.bfloat16)
    compressed_cache = torch.randn(cache_slots, 512, device=device, dtype=torch.bfloat16)
    swa_indices = torch.randint(0, cache_slots, (tokens, 128), device=device, dtype=torch.int32)
    compressed_indices = torch.randint(
        0, cache_slots, (tokens, 512), device=device, dtype=torch.int32
    )
    swa_lengths = torch.full((tokens,), 128, device=device, dtype=torch.int32)
    compressed_lengths = torch.full((tokens,), 512, device=device, dtype=torch.int32)
    attn_sink = torch.zeros(8, device=device, dtype=torch.float32)

    def fn() -> torch.Tensor:
        out = dsv4_sparse_attention_two_source_bf16(
            q,
            swa_cache,
            swa_indices,
            swa_lengths,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            compressed_lengths=compressed_lengths,
            softmax_scale=512.0**-0.5,
            attn_sink=attn_sink,
        )
        if out is None:
            raise RuntimeError("dsv4_sparse_attention_two_source_bf16 returned None")
        return out

    try:
        result = cuda_event_bench(
            f"sparse_attention_two_source_tokens{tokens}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path
        return blocker(
            f"sparse_attention_two_source_tokens{tokens}",
            f"{type(exc).__name__}: {exc}",
            subgraph="sparse_attention_and_indexer",
            tokens=tokens,
        )
    result.update(
        {
            "subgraph": "sparse_attention_and_indexer",
            "tokens": tokens,
            "shape": {
                "q": [tokens, 8, 512],
                "swa_indices": [tokens, 128],
                "compressed_indices": [tokens, 512],
                "cache": [cache_slots, 512],
            },
            "dtype": "bf16 cache/output",
            "boundary_note": "mini exact sparse attention kernel, two-source SWA+C4/C128; excludes indexer/topk construction",
        }
    )
    return result


def indexer_select(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    page_size = 64
    seq_len = 4096
    pages = seq_len // page_size
    q = torch.randn(tokens, 64, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, 64, device=device, dtype=torch.float32)
    cache = torch.randn(seq_len, 128, device=device, dtype=torch.bfloat16)
    seq_lens = torch.full((tokens,), seq_len, device=device, dtype=torch.int32)
    page_table = torch.arange(pages, device=device, dtype=torch.int32).repeat(tokens, 1)

    def fn() -> Any:
        return indexer_select_bf16_fallback(
            q,
            weights,
            cache,
            seq_lens,
            page_table,
            page_size=page_size,
            width=512,
            ratio=4,
        )

    try:
        result = cuda_event_bench(
            f"indexer_select_bf16_tokens{tokens}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path
        return blocker(
            f"indexer_select_bf16_tokens{tokens}",
            f"{type(exc).__name__}: {exc}",
            subgraph="sparse_attention_and_indexer",
            tokens=tokens,
        )
    result.update(
        {
            "subgraph": "sparse_attention_and_indexer",
            "tokens": tokens,
            "shape": {
                "q": [tokens, 64, 128],
                "weights": [tokens, 64],
                "cache": [seq_len, 128],
                "page_table": [tokens, pages],
                "topk_width": 512,
            },
            "dtype": "bf16 indexer cache/query, fp32 weights/logits",
            "boundary_note": "mini C4 indexer logits + topk transform; deterministic synthetic page table",
        }
    )
    return result


def moe_routed(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, 6, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    indices = torch.randint(0, 256, (tokens, 6), device=device, dtype=torch.int64)
    scale_dtype = e8m0_dtype()
    w13 = torch.randint(-8, 8, (256, 2, 256, 2048), device=device, dtype=torch.int8)
    w13_scale = torch.ones((256, 2, 256, 128), device=device, dtype=scale_dtype)
    w2 = torch.randint(-8, 8, (256, 4096, 128), device=device, dtype=torch.int8)
    w2_scale = torch.ones((256, 4096, 8), device=device, dtype=scale_dtype)

    def fn() -> torch.Tensor:
        out = moe_route_dispatch_bf16_grouped(
            hidden,
            weights,
            indices,
            w13,
            w13_scale,
            w2,
            w2_scale,
            swiglu_limit=10.0,
        )
        if out is None:
            raise RuntimeError("moe_route_dispatch_bf16_grouped returned None")
        return out

    try:
        result = cuda_event_bench(
            f"moe_routed_grouped_fp4_tokens{tokens}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path
        return blocker(
            f"moe_routed_grouped_fp4_tokens{tokens}",
            f"{type(exc).__name__}: {exc}",
            subgraph="moe_route_and_routed_experts",
            tokens=tokens,
        )
    result.update(
        {
            "subgraph": "moe_route_and_routed_experts",
            "tokens": tokens,
            "shape": {
                "hidden": [tokens, 4096],
                "topk": 6,
                "w13": [256, 2, 256, 2048],
                "w2": [256, 4096, 128],
            },
            "dtype": "bf16 activations, packed fp4 expert weights, ue8m0 scales, fp32 route weights",
            "boundary_note": "mini grouped routed expert kernel only; excludes gate and TP all-reduce",
        }
    )
    return result


def shared_experts(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    gate_up = torch.randn(512, 4096, device=device, dtype=torch.bfloat16) * 0.01
    down = torch.randn(4096, 256, device=device, dtype=torch.bfloat16) * 0.01

    def fn() -> torch.Tensor:
        gu = F.linear(hidden, gate_up)
        gate, up = gu.chunk(2, dim=-1)
        act = silu_and_mul_clamp_fallback(gate, up, swiglu_limit=10.0)
        return F.linear(act.to(torch.bfloat16), down)

    result = cuda_event_bench(
        f"shared_experts_local_tokens{tokens}",
        fn,
        warmup=warmup,
        repeat=repeat,
    )
    result.update(
        {
            "subgraph": "shared_experts",
            "tokens": tokens,
            "shape": {
                "hidden": [tokens, 4096],
                "gate_up_local_weight": [512, 4096],
                "down_local_weight": [4096, 256],
            },
            "dtype": "bf16 activations/weights, fp32 activation math",
            "boundary_note": "local TP shard of shared experts; excludes reduce-once combine",
        }
    )
    return result


def hc_rmsnorm_final(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    x = torch.randn(tokens, 4, 4096, device=device, dtype=torch.bfloat16)
    fn_weight = torch.randn(4, 4 * 4096, device=device, dtype=torch.bfloat16) * 0.01
    scale = torch.ones(4, device=device, dtype=torch.float32)
    base = torch.zeros(4, device=device, dtype=torch.float32)
    norm_weight = torch.ones(4096, device=device, dtype=torch.bfloat16)

    def fn() -> torch.Tensor:
        y = hc_head_fallback(x, fn_weight, scale, base, eps=1.0e-6, norm_eps=1.0e-6)
        return rms_norm_fallback(y, norm_weight, eps=1.0e-6)

    result = cuda_event_bench(
        f"hc_head_rmsnorm_tokens{tokens}",
        fn,
        warmup=warmup,
        repeat=repeat,
    )
    result.update(
        {
            "subgraph": "hc_rmsnorm_final_layers",
            "tokens": tokens,
            "shape": {"hc_input": [tokens, 4, 4096], "fn": [4, 16384]},
            "dtype": "bf16 activations, fp32 HC mix math",
            "boundary_note": "HC head + final RMSNorm stand-in; layer HC pre/post appears in macro profile",
        }
    )
    return result


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "toggles": {k: os.environ.get(k) for k in sorted(DEFAULT_TOGGLES)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="*", default=[4, 4096])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(7)
    torch.cuda.set_device(0)

    warmup = 2 if args.quick else 5
    repeats_by_tokens = {
        4: 10 if args.quick else 50,
        4096: 3 if args.quick else 10,
    }
    started = time.time()
    results: list[dict[str, Any]] = []
    for tokens in args.tokens:
        repeat = repeats_by_tokens.get(tokens, 3 if args.quick else 10)
        for probe in (
            attention_front_cache_insert,
            sparse_attention,
            indexer_select,
            moe_routed,
            shared_experts,
            hc_rmsnorm_final,
        ):
            results.append(probe(tokens, warmup, repeat))
            torch.cuda.empty_cache()

    output = {
        "suite": "mini_subgraph_microbench",
        "scope": "DeepSeek V4 Flash, A100/sm80, synthetic single-rank subgraphs",
        "env": env_info(),
        "started_at_unix": started,
        "elapsed_s": time.time() - started,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

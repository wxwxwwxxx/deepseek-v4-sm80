#!/usr/bin/env python3
"""vLLM DeepSeek V4 sm80 subgraph probes.

This script deliberately avoids starting vLLM as a serving/runtime dependency
for mini.  It imports vLLM kernels in the vLLM Python environment and records
blockers where an exact engine-managed boundary cannot be reconstructed safely.
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

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.deepseek_v4_attention import (
    _dsv4_sm80_sparse_attn_decode_triton,
)
from vllm.v1.attention.ops.deepseek_v4_ops import fused_indexer_q_rope_quant
from vllm.v1.attention.ops.deepseek_v4_ops import fused_q_kv_rmsnorm


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


def make_cos_sin(max_pos: int, rotary_dim: int, *, device: torch.device) -> torch.Tensor:
    half = rotary_dim // 2
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device) / rotary_dim))
    freqs = torch.outer(pos, inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def attention_front_projection(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    fused_weight = torch.randn(1536, 4096, device=device, dtype=torch.bfloat16) * 0.01
    q_weight = torch.ones(1024, device=device, dtype=torch.bfloat16)
    kv_weight = torch.ones(512, device=device, dtype=torch.bfloat16)
    wq_b_weight = torch.randn(4096, 1024, device=device, dtype=torch.bfloat16) * 0.01

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        qr_kv = F.linear(hidden, fused_weight)
        qr, kv = qr_kv.split([1024, 512], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(qr.contiguous(), kv.contiguous(), q_weight, kv_weight, 1.0e-6)
        q = F.linear(qr, wq_b_weight).view(tokens, 8, 512)
        return q, kv

    result = cuda_event_bench(
        f"attention_front_projection_tokens{tokens}",
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
            },
            "dtype": "bf16 activations here; engine path then quant-inserts fp8_ds_mla KV cache",
            "boundary_note": "vLLM fused_wqa_wkv + fused_q_kv_rmsnorm + wq_b only; exact cache insert requires engine metadata/cache layer",
            "mismatch": "does not include fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert because it needs vLLM cache metadata",
        }
    )
    return result


def sparse_attention_core(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    topk = 640
    q = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16)
    gathered = torch.randn(tokens, topk, 512, device=device, dtype=torch.bfloat16)
    invalid = torch.zeros(tokens, topk, device=device, dtype=torch.bool)
    attn_sink = torch.zeros(8, device=device, dtype=torch.float32)

    def fn() -> torch.Tensor:
        return _dsv4_sm80_sparse_attn_decode_triton(
            q,
            gathered,
            invalid,
            attn_sink,
            512.0**-0.5,
            512,
        )

    try:
        result = cuda_event_bench(
            f"sparse_attention_decode_core_tokens{tokens}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path
        return blocker(
            f"sparse_attention_decode_core_tokens{tokens}",
            f"{type(exc).__name__}: {exc}",
            subgraph="sparse_attention_and_indexer",
            tokens=tokens,
        )
    result.update(
        {
            "subgraph": "sparse_attention_and_indexer",
            "tokens": tokens,
            "shape": {"q": [tokens, 8, 512], "gathered_kv": [tokens, topk, 512]},
            "dtype": "bf16 gathered KV/output; vLLM engine stores packed fp8_ds_mla cache before gather",
            "boundary_note": "vLLM sm80 split-K sparse attention decode core; excludes gather/dequant and topk/indexer",
            "mismatch": "mini sparse benchmark reads bf16 cache directly; vLLM engine benchmark here starts after gather/dequant",
        }
    )
    return result


def indexer_q_rope_quant(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    positions = torch.arange(tokens, device=device, dtype=torch.long) + 4096
    q = torch.randn(tokens, 64, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, 64, device=device, dtype=torch.float32)
    cos_sin = make_cos_sin(tokens + 4096 + 1, 128, device=device)

    def fn() -> Any:
        return fused_indexer_q_rope_quant(
            positions,
            q,
            cos_sin,
            weights,
            512.0**-0.5,
            64.0**-0.5,
            use_fp4=False,
        )

    try:
        result = cuda_event_bench(
            f"indexer_q_rope_quant_tokens{tokens}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path
        return blocker(
            f"indexer_q_rope_quant_tokens{tokens}",
            f"{type(exc).__name__}: {exc}",
            subgraph="sparse_attention_and_indexer",
            tokens=tokens,
        )
    result.update(
        {
            "subgraph": "sparse_attention_and_indexer",
            "tokens": tokens,
            "shape": {"q": [tokens, 64, 128], "weights": [tokens, 64]},
            "dtype": "bf16 q input, fp8 q output on SM80 reference path, fp32 folded weights",
            "boundary_note": "vLLM fused indexer Q RoPE + quant/weight fold; excludes SparseAttnIndexer topk/cache state",
        }
    )
    return result


def overlap_probe(tokens: int, warmup: int, repeat: int) -> dict[str, Any]:
    device = torch.device("cuda")
    a = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    w0 = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16) * 0.01
    w1 = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16) * 0.01
    aux = torch.cuda.Stream()
    ev0 = torch.cuda.Event()
    ev1 = torch.cuda.Event()

    def serial() -> tuple[torch.Tensor, torch.Tensor]:
        return F.linear(a, w0), F.linear(a, w1)

    def overlapped() -> tuple[torch.Tensor, torch.Tensor]:
        ev0.record()
        y0 = F.linear(a, w0)
        with torch.cuda.stream(aux):
            ev0.wait()
            y1 = F.linear(a, w1)
            ev1.record()
        ev1.wait()
        return y0, y1

    serial_result = cuda_event_bench(
        f"stream_overlap_probe_serial_tokens{tokens}",
        serial,
        warmup=warmup,
        repeat=repeat,
    )
    overlap_result = cuda_event_bench(
        f"stream_overlap_probe_aux_tokens{tokens}",
        overlapped,
        warmup=warmup,
        repeat=repeat,
    )
    return {
        "name": f"stream_overlap_probe_tokens{tokens}",
        "status": "pass",
        "subgraph": "scheduler_graph_stream_overlap",
        "tokens": tokens,
        "shape": {"two_independent_bf16_gemm": [[tokens, 4096], [4096, 4096]]},
        "dtype": "bf16",
        "serial_mean_ms": serial_result["mean_ms"],
        "overlap_mean_ms": overlap_result["mean_ms"],
        "estimated_speedup": serial_result["mean_ms"] / max(overlap_result["mean_ms"], 1.0e-9),
        "serial": serial_result,
        "overlap": overlap_result,
        "boundary_note": "generic two-stream probe mirroring maybe_execute_in_parallel topology; not a vLLM engine subgraph timing",
    }


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "env_toggles": {
            key: os.environ.get(key)
            for key in (
                "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
                "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
                "VLLM_USE_V1",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tokens", type=int, nargs="*", default=[4, 4096])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(11)
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
        results.append(attention_front_projection(tokens, warmup, repeat))
        results.append(sparse_attention_core(tokens, warmup, repeat))
        results.append(indexer_q_rope_quant(tokens, warmup, repeat))
        if tokens in (4, 4096):
            results.append(overlap_probe(tokens, warmup, repeat))
        results.append(
            blocker(
                f"moe_routed_fused_moe_tokens{tokens}",
                "Exact vLLM routed-MoE boundary requires FusedMoE layer construction, transformed MXFP4 weights, static forward context, and runner-owned router/shared-expert state. Running the quant_method kernel standalone would not preserve the engine boundary.",
                subgraph="moe_route_and_routed_experts",
                tokens=tokens,
                evidence="vllm/model_executor/layers/fused_moe/runner/moe_runner.py and quantization/mxfp4.py",
            )
        )
        results.append(
            blocker(
                f"shared_experts_vllm_tokens{tokens}",
                "Exact shared-experts boundary is scheduled by vLLM SharedExperts wrapper on an aux stream and depends on the DeepseekV2MLP quantized linear modules plus MoE runner ordering.",
                subgraph="shared_experts",
                tokens=tokens,
                evidence="vllm/model_executor/layers/fused_moe/runner/shared_experts.py",
            )
        )
        torch.cuda.empty_cache()

    output = {
        "suite": "vllm_subgraph_microbench",
        "scope": "DeepSeek V4 Flash, A100/sm80, synthetic vLLM kernel probes",
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

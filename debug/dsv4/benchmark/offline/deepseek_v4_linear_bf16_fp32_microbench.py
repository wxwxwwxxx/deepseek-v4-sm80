from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


SHAPES = {
    "hc_pre": (16384, 24),
    "hc_head": (16384, 4),
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _bench_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _with_env(name: str, value: str | None):
    class Guard:
        def __enter__(self):
            self.prev = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

        def __exit__(self, exc_type, exc, tb):
            if self.prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = self.prev

    return Guard()


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    error = (actual.float() - expected.float()).abs()
    rel = error / expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "max_rel_error": float(rel.max().item()),
        "mean_rel_error": float(rel.mean().item()),
        "allclose_1e_0": bool(torch.allclose(actual, expected, atol=1.0, rtol=1e-2)),
        "allclose_5e_1": bool(torch.allclose(actual, expected, atol=0.5, rtol=1e-2)),
    }


def _bench_case(
    *,
    label: str,
    m: int,
    k: int,
    n: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight_fp32 = torch.randn(n, k, device=device, dtype=torch.float32)
    weight_bf16 = weight_fp32.to(torch.bfloat16).contiguous()

    with _with_env(dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE, None):
        expected_fp32 = dsv4_kernel.linear_bf16_fp32_fallback(x, weight_fp32)
        fallback_fp32_ms = _bench_cuda(
            lambda: dsv4_kernel.linear_bf16_fp32_fallback(x, weight_fp32),
            warmup=warmup,
            iters=iters,
        )
        expected_bf16_weight = dsv4_kernel.linear_bf16_fp32_fallback(x, weight_bf16)
        fallback_bf16_weight_ms = _bench_cuda(
            lambda: dsv4_kernel.linear_bf16_fp32_fallback(x, weight_bf16),
            warmup=warmup,
            iters=iters,
        )

    with _with_env(dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE, "1"):
        actual = dsv4_kernel.linear_bf16_fp32_fallback(x, weight_bf16)
        opt_in_ms = _bench_cuda(
            lambda: dsv4_kernel.linear_bf16_fp32_fallback(x, weight_bf16),
            warmup=warmup,
            iters=iters,
        )

    cast_once_ms = _bench_cuda(
        lambda: weight_fp32.to(torch.bfloat16).contiguous(),
        warmup=warmup,
        iters=iters,
    )

    return {
        "label": label,
        "m": m,
        "k": k,
        "n": n,
        "fallback_fp32_weight_ms": fallback_fp32_ms,
        "fallback_bf16_weight_ms": fallback_bf16_weight_ms,
        "opt_in_bf16_weight_mm_ms": opt_in_ms,
        "weight_fp32_to_bf16_once_ms": cast_once_ms,
        "speedup_vs_current_fp32_weight": fallback_fp32_ms / opt_in_ms
        if opt_in_ms > 0
        else None,
        "speedup_vs_bf16_weight_fallback": fallback_bf16_weight_ms / opt_in_ms
        if opt_in_ms > 0
        else None,
        "error_vs_current_fp32_weight": _error_stats(actual, expected_fp32),
        "error_vs_bf16_weight_fallback": _error_stats(actual, expected_bf16_weight),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DeepSeek V4 linear_bf16_fp32 upstream/cuBLAS path."
    )
    parser.add_argument("--m-values", default="1,8,128,2048")
    parser.add_argument(
        "--shapes",
        default="hc_pre,hc_head",
        help=f"Comma-separated shape names from: {','.join(SHAPES)}",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--output",
        default="/tmp/dsv4_linear_bf16_fp32_upstream_microbench.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(123)
    m_values = _parse_csv_ints(args.m_values)
    shape_names = [name for name in args.shapes.split(",") if name]

    results = []
    started = time.time()
    for shape_name in shape_names:
        k, n = SHAPES[shape_name]
        for m in m_values:
            result = _bench_case(
                label=shape_name,
                m=m,
                k=k,
                n=n,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            results.append(result)
            print(
                f"{shape_name} m={m}: fallback_fp32={result['fallback_fp32_weight_ms']:.4f} ms, "
                f"opt_in={result['opt_in_bf16_weight_mm_ms']:.4f} ms, "
                f"speedup={result['speedup_vs_current_fp32_weight']:.2f}x, "
                f"max_abs={result['error_vs_current_fp32_weight']['max_abs_error']:.3f}"
            )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "toggle": dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE,
        "mode": "opt-in cached HC fp32->bf16 weight copy + torch.mm out_dtype=torch.float32",
        "capabilities": asdict(dsv4_kernel.detect_dsv4_kernel_capabilities()),
        "m_values": m_values,
        "shapes": shape_names,
        "results": results,
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

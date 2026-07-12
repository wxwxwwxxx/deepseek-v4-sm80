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
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


SHAPES = {
    "fp8_wq_a": ("fp8", 4096, 1024),
    "fp8_wkv": ("fp8", 4096, 512),
    "fp8_wq_b": ("fp8", 1024, 32768),
    "fp8_indexer_wq_b": ("fp8", 1024, 8192),
    "fp8_wo_b": ("fp8", 1024, 4096),
    "fp8_shared_gate_up": ("fp8", 4096, 4096),
    "fp8_shared_down": ("fp8", 256, 4096),
    "fp4_w13": ("fp4", 4096, 2048),
    "fp4_w2": ("fp4", 2048, 4096),
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _sync() -> None:
    torch.cuda.synchronize()


def _bench_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
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


def _make_case(
    *,
    kind: str,
    m: int,
    k: int,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    scale_base = torch.rand(device=device, dtype=torch.float32, size=(1,)) + 0.5
    if kind == "fp8":
        weight = torch.randn(n, k, device=device, dtype=torch.float32).clamp(-4, 4).to(
            dsv4_kernel.fp8_dtype()
        )
        scale = (
            torch.rand(
                dsv4_kernel.scale_dim(n),
                dsv4_kernel.scale_dim(k),
                device=device,
                dtype=torch.float32,
            )
            + scale_base
        ).to(dsv4_kernel.e8m0_dtype())
    elif kind == "fp4":
        weight = torch.randint(-128, 127, (n, k // 2), device=device, dtype=torch.int8)
        scale = (
            torch.rand(n, (k + 31) // 32, device=device, dtype=torch.float32) + scale_base
        ).to(dsv4_kernel.e8m0_dtype())
    else:
        raise ValueError(f"unknown kind: {kind}")
    return x, weight, scale


def _bench_case(
    *,
    label: str,
    kind: str,
    m: int,
    k: int,
    n: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    x, weight, scale = _make_case(kind=kind, m=m, k=k, n=n, device=device)
    toggle = (
        "MINISGL_DSV4_SM80_FP8_GEMM"
        if kind == "fp8"
        else "MINISGL_DSV4_SM80_FP4_GEMM"
    )

    with _with_env(toggle, None):
        expected = dsv4_kernel.quantized_linear_ref(x, weight, scale, weight_kind=kind)
        fallback_ms = _bench_cuda(
            lambda: dsv4_kernel.quantized_linear_ref(x, weight, scale, weight_kind=kind),
            warmup=warmup,
            iters=iters,
        )

    with _with_env(toggle, "1"):
        actual = dsv4_kernel.quantized_linear_ref(x, weight, scale, weight_kind=kind)
        triton_ms = _bench_cuda(
            lambda: dsv4_kernel.quantized_linear_ref(x, weight, scale, weight_kind=kind),
            warmup=warmup,
            iters=iters,
        )

    x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
    if kind == "fp8":
        dequant_fn = lambda: dsv4_kernel.dequant_fp8_weight(
            weight,
            scale,
            out_dtype=x.dtype,
        )
    else:
        dequant_fn = lambda: dsv4_kernel.dequant_fp4_weight(
            weight,
            scale,
            out_dtype=x.dtype,
        )
    w_dequant = dequant_fn()

    act_quant_ms = _bench_cuda(
        lambda: dsv4_kernel.quantize_fp8_activation_ref(x),
        warmup=warmup,
        iters=iters,
    )
    dequant_ms = _bench_cuda(dequant_fn, warmup=warmup, iters=iters)
    matmul_ms = _bench_cuda(lambda: F.linear(x_quant, w_dequant), warmup=warmup, iters=iters)

    error = (actual.float() - expected.float()).abs()
    return {
        "label": label,
        "kind": kind,
        "m": m,
        "k": k,
        "n": n,
        "fallback_total_ms": fallback_ms,
        "act_quant_ms": act_quant_ms,
        "weight_dequant_ms": dequant_ms,
        "matmul_after_dequant_ms": matmul_ms,
        "triton_total_ms": triton_ms,
        "speedup": fallback_ms / triton_ms if triton_ms > 0 else None,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "allclose_3e_2": bool(torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DeepSeek V4 quantized_linear_ref weight-dequant paths."
    )
    parser.add_argument("--m-values", default="1,4,8,16,64,256")
    parser.add_argument(
        "--shapes",
        default="fp8_wq_a,fp8_wkv,fp4_w13,fp4_w2",
        help=f"Comma-separated shape names from: {','.join(SHAPES)}",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output",
        default="/tmp/dsv4_quantized_linear_ref_bf16_weight_dequant_microbench.json",
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
        kind, k, n = SHAPES[shape_name]
        for m in m_values:
            result = _bench_case(
                label=shape_name,
                kind=kind,
                m=m,
                k=k,
                n=n,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            results.append(result)
            print(
                f"{shape_name} m={m}: fallback={result['fallback_total_ms']:.3f} ms, "
                f"triton={result['triton_total_ms']:.3f} ms, "
                f"speedup={result['speedup']:.2f}x, "
                f"ok={result['allclose_3e_2']}"
            )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "capabilities": asdict(dsv4_kernel.detect_dsv4_kernel_capabilities()),
        "m_values": m_values,
        "shapes": shape_names,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

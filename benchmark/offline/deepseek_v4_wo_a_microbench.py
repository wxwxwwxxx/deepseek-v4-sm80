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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


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
    tokens: int,
    groups: int,
    rank: int,
    d_per_group: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o = torch.randn(tokens, groups, d_per_group, device=device, dtype=torch.bfloat16)
    weight = torch.randn(
        groups * rank,
        d_per_group,
        device=device,
        dtype=torch.float32,
    ).clamp(-4, 4).to(dsv4_kernel.fp8_dtype())
    scale = (
        torch.rand(
            dsv4_kernel.scale_dim(groups * rank),
            dsv4_kernel.scale_dim(d_per_group),
            device=device,
            dtype=torch.float32,
        )
        + 0.5
    ).to(dsv4_kernel.e8m0_dtype())
    return o, weight, scale


def _fallback_einsum(
    o: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    groups: int,
    rank: int,
) -> torch.Tensor:
    d_per_group = o.shape[-1]
    wo_a = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=o.dtype)
    wo_a = wo_a.view(groups, rank, d_per_group)
    return torch.einsum("tgd,grd->tgr", o, wo_a).reshape(o.shape[0], -1)


def _bench_case(
    *,
    tokens: int,
    groups: int,
    rank: int,
    d_per_group: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    o, weight, scale = _make_case(
        tokens=tokens,
        groups=groups,
        rank=rank,
        d_per_group=d_per_group,
        device=device,
    )

    with _with_env("MINISGL_DSV4_SM80_WO_A_BF16", None):
        expected = dsv4_kernel.wo_a_grouped_projection_fallback(
            o,
            weight,
            scale,
            num_local_groups=groups,
            o_lora_rank=rank,
        )
        fallback_ms = _bench_cuda(
            lambda: dsv4_kernel.wo_a_grouped_projection_fallback(
                o,
                weight,
                scale,
                num_local_groups=groups,
                o_lora_rank=rank,
            ),
            warmup=warmup,
            iters=iters,
        )

    with _with_env("MINISGL_DSV4_SM80_WO_A_BF16", "1"):
        actual = dsv4_kernel.wo_a_grouped_projection_fallback(
            o,
            weight,
            scale,
            num_local_groups=groups,
            o_lora_rank=rank,
        )
        triton_ms = _bench_cuda(
            lambda: dsv4_kernel.wo_a_grouped_projection_fallback(
                o,
                weight,
                scale,
                num_local_groups=groups,
                o_lora_rank=rank,
            ),
            warmup=warmup,
            iters=iters,
        )

    wo_a = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=o.dtype).view(
        groups,
        rank,
        d_per_group,
    )
    dequant_ms = _bench_cuda(
        lambda: dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=o.dtype),
        warmup=warmup,
        iters=iters,
    )
    einsum_ms = _bench_cuda(
        lambda: torch.einsum("tgd,grd->tgr", o, wo_a).reshape(tokens, -1),
        warmup=warmup,
        iters=iters,
    )
    fallback_split_ms = _bench_cuda(
        lambda: _fallback_einsum(o, weight, scale, groups=groups, rank=rank),
        warmup=warmup,
        iters=iters,
    )

    error = (actual.float() - expected.float()).abs()
    return {
        "tokens": tokens,
        "groups": groups,
        "rank": rank,
        "d_per_group": d_per_group,
        "weight_shape": list(weight.shape),
        "scale_shape": list(scale.shape),
        "fallback_total_ms": fallback_ms,
        "weight_dequant_ms": dequant_ms,
        "einsum_after_dequant_ms": einsum_ms,
        "fallback_split_total_ms": fallback_split_ms,
        "triton_total_ms": triton_ms,
        "opt_in_expected_path": "triton" if tokens <= 16 else "fallback_gated",
        "speedup": fallback_ms / triton_ms if triton_ms > 0 else None,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "allclose_4e_2": bool(torch.allclose(actual, expected, atol=4e-2, rtol=4e-2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DeepSeek V4 wo_a grouped fp8 weight-dequant projection."
    )
    parser.add_argument("--tokens", default="1,8,64,512")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--d-per-group", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output",
        default="/tmp/dsv4_wo_a_grouped_projection_bf16_weight_dequant_microbench.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(123)
    token_values = _parse_csv_ints(args.tokens)

    results = []
    started = time.time()
    for tokens in token_values:
        result = _bench_case(
            tokens=tokens,
            groups=args.groups,
            rank=args.rank,
            d_per_group=args.d_per_group,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        results.append(result)
        print(
            f"tokens={tokens}: fallback={result['fallback_total_ms']:.3f} ms, "
            f"triton={result['triton_total_ms']:.3f} ms, "
            f"speedup={result['speedup']:.2f}x, "
            f"ok={result['allclose_4e_2']}"
        )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "capabilities": asdict(dsv4_kernel.detect_dsv4_kernel_capabilities()),
        "toggle": "MINISGL_DSV4_SM80_WO_A_BF16",
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402
from minisgl.kernel.triton import deepseek_v4 as triton_dsv4  # noqa: E402


OWNER_TENSORS: dict[str, tuple[str, str]] = {
    "attn.wo_b.layer0": ("layers.0.attn.wo_b.weight", "layers.0.attn.wo_b.scale"),
}

OWNER_TP_SHARD_DIM: dict[str, int | None] = {
    "attn.wo_b.layer0": 1,
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


def _load_tensor(model_path: Path, index: dict[str, str], key: str, device: torch.device) -> torch.Tensor:
    file = model_path / index[key]
    with safe_open(file, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    return tensor.to(device=device).contiguous()


def _load_owner(
    model_path: Path,
    index: dict[str, str],
    owner: str,
    device: torch.device,
    *,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_key, scale_key = OWNER_TENSORS[owner]
    weight = _load_tensor(model_path, index, weight_key, device)
    scale = _load_tensor(model_path, index, scale_key, device)
    shard_dim = OWNER_TP_SHARD_DIM[owner]
    if tp_size > 1 and shard_dim is not None:
        weight = weight.chunk(tp_size, dim=shard_dim)[tp_rank].contiguous()
        scale = scale.chunk(tp_size, dim=shard_dim)[tp_rank].contiguous()
    return weight, scale


def _error_row(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1.0e-6)
    rel = diff / denom
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_rel_error": float(rel.max().item()),
        "mean_rel_error": float(rel.mean().item()),
        "allclose_3e_2": bool(torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)),
    }


def _bench_owner_m(
    *,
    owner: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    m: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = weight.device
    k = weight.shape[-1]
    n = weight.shape[0]
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    scale_fp32 = scale.float().contiguous()

    with _with_env(dsv4_kernel.DSV4_SM80_FP8_ACT_QUANT_TRITON_TOGGLE, "1"):
        x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
        current = dsv4_kernel.quantized_linear_ref(
            x,
            weight,
            scale,
            weight_kind="fp8",
            fp8_gemm=True,
        )
        fallback = dsv4_kernel.quantized_linear_ref(
            x,
            weight,
            scale,
            weight_kind="fp8",
            fp8_gemm=False,
        )
        cached_weight = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=x.dtype).contiguous()
        cached = F.linear(x_quant, cached_weight)

        current_wrapper_ms = _bench_cuda(
            lambda: dsv4_kernel.quantized_linear_ref(
                x,
                weight,
                scale,
                weight_kind="fp8",
                fp8_gemm=True,
            ),
            warmup=warmup,
            iters=iters,
        )
        activation_quant_ms = _bench_cuda(
            lambda: dsv4_kernel.quantize_fp8_activation_ref(x),
            warmup=warmup,
            iters=iters,
        )
        current_intrinsic_cached_scale_ms = _bench_cuda(
            lambda: triton_dsv4.quantized_linear_fp8(x_quant, weight, scale_fp32),
            warmup=warmup,
            iters=iters,
        )
        fallback_total_ms = _bench_cuda(
            lambda: dsv4_kernel.quantized_linear_ref(
                x,
                weight,
                scale,
                weight_kind="fp8",
                fp8_gemm=False,
            ),
            warmup=max(1, warmup // 2),
            iters=max(3, min(iters, 10)),
        )
        cache_build_ms = _bench_cuda(
            lambda: dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=x.dtype).contiguous(),
            warmup=max(1, warmup // 2),
            iters=max(3, min(iters, 10)),
        )
        cached_bf16_flinear_ms = _bench_cuda(
            lambda: F.linear(x_quant, cached_weight),
            warmup=warmup,
            iters=iters,
        )
        cached_bf16_total_ms = _bench_cuda(
            lambda: F.linear(dsv4_kernel.quantize_fp8_activation_ref(x), cached_weight),
            warmup=warmup,
            iters=iters,
        )

    return {
        "owner": owner,
        "m": m,
        "k": k,
        "n": n,
        "weight_shape": list(weight.shape),
        "scale_shape": list(scale.shape),
        "cached_weight_bytes": int(cached_weight.numel() * cached_weight.element_size()),
        "current_wrapper_ms": current_wrapper_ms,
        "activation_quant_ms": activation_quant_ms,
        "current_intrinsic_cached_scale_ms": current_intrinsic_cached_scale_ms,
        "fallback_dequant_total_ms": fallback_total_ms,
        "cache_build_dequant_ms": cache_build_ms,
        "cached_bf16_flinear_ms": cached_bf16_flinear_ms,
        "cached_bf16_total_local_projection_ms": cached_bf16_total_ms,
        "cached_vs_current_speedup": (
            current_wrapper_ms / cached_bf16_total_ms if cached_bf16_total_ms > 0 else None
        ),
        "cached_flinear_vs_current_intrinsic_speedup": (
            current_intrinsic_cached_scale_ms / cached_bf16_flinear_ms
            if cached_bf16_flinear_ms > 0
            else None
        ),
        "current_vs_fallback_error": _error_row(current, fallback),
        "cached_vs_current_error": _error_row(cached, current),
        "all_reduce_included": False,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# wo_b Cached BF16 Projection Microbench")
    lines.append("")
    lines.append(f"- Model path: `{report['model_path']}`")
    lines.append(f"- CUDA device: `{report['cuda_device']}`")
    lines.append(f"- TP shard: rank `{report['tp_rank']}` / size `{report['tp_size']}`")
    lines.append(f"- M values: `{report['m_values']}`")
    lines.append(f"- Warmup/iters: `{report['warmup']}` / `{report['iters']}`")
    lines.append("- Scope: local row-parallel projection only; all-reduce is not included here.")
    lines.append("")
    lines.append(
        "| Owner | M | K | N | current FP8 wrapper ms | current intrinsic ms | fallback dequant ms | cached BF16 local F.linear ms | cached total local projection ms | speedup vs wrapper | max abs err | max rel err | ok |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for row in report["results"]:
        err = row["cached_vs_current_error"]
        lines.append(
            "| `{owner}` | {m} | {k} | {n} | `{current:.4f}` | `{intrinsic:.4f}` | `{fallback:.4f}` | `{cached_linear:.4f}` | `{cached_total:.4f}` | `{speedup:.2f}x` | `{max_abs:.6f}` | `{max_rel:.6f}` | `{ok}` |".format(
                owner=row["owner"],
                m=row["m"],
                k=row["k"],
                n=row["n"],
                current=row["current_wrapper_ms"],
                intrinsic=row["current_intrinsic_cached_scale_ms"],
                fallback=row["fallback_dequant_total_ms"],
                cached_linear=row["cached_bf16_flinear_ms"],
                cached_total=row["cached_bf16_total_local_projection_ms"],
                speedup=row["cached_vs_current_speedup"] or 0.0,
                max_abs=err["max_abs_error"],
                max_rel=err["max_rel_error"],
                ok=err["allclose_3e_2"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/models/DeepSeek-V4-Flash"))
    parser.add_argument("--owners", default="attn.wo_b.layer0")
    parser.add_argument("--m-values", default="1,4,8,16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=759)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_cached_bf16_wo_b_projection_backend/raw/wob_cached_bf16_microbench.json"
        ),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_cached_bf16_wo_b_projection_backend/summaries/wob_cached_bf16_microbench.md"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    index_path = args.model_path / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    device = torch.device("cuda")
    owners = [owner for owner in args.owners.split(",") if owner]
    if args.tp_size <= 0:
        raise SystemExit("--tp-size must be positive")
    if not 0 <= args.tp_rank < args.tp_size:
        raise SystemExit("--tp-rank must be in [0, tp_size)")
    m_values = _parse_csv_ints(args.m_values)
    started = time.time()
    results: list[dict[str, Any]] = []
    for owner_idx, owner in enumerate(owners):
        weight, scale = _load_owner(
            args.model_path,
            weight_map,
            owner,
            device,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
        )
        for m in m_values:
            row = _bench_owner_m(
                owner=owner,
                weight=weight,
                scale=scale,
                m=m,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + owner_idx * 100 + m,
            )
            results.append(row)
            print(
                f"{owner} M={m}: current={row['current_wrapper_ms']:.4f} ms, "
                f"cached_total_local={row['cached_bf16_total_local_projection_ms']:.4f} ms, "
                f"cached_linear={row['cached_bf16_flinear_ms']:.4f} ms, "
                f"ok={row['cached_vs_current_error']['allclose_3e_2']}"
            )
        del weight, scale
        torch.cuda.empty_cache()

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "model_path": str(args.model_path),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "m_values": m_values,
        "owners": owners,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md_out.write_text(_render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()

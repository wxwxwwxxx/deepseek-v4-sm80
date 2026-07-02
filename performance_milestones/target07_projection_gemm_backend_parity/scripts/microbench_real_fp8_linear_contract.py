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
    "attn.q_wqb.layer0": ("layers.0.attn.wq_b.weight", "layers.0.attn.wq_b.scale"),
    "attn.wo_b.layer0": ("layers.0.attn.wo_b.weight", "layers.0.attn.wo_b.scale"),
    "indexer.wq_b.layer2": (
        "layers.2.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.wq_b.scale",
    ),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_key, scale_key = OWNER_TENSORS[owner]
    return (
        _load_tensor(model_path, index, weight_key, device),
        _load_tensor(model_path, index, scale_key, device),
    )


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

    with _with_env("MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON", "1"):
        expected = dsv4_kernel.quantized_linear_ref(x, weight, scale, weight_kind="fp8")
        actual = dsv4_kernel.quantized_linear_ref(
            x,
            weight,
            scale,
            weight_kind="fp8",
            fp8_gemm=True,
        )
        x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)

        wrapper_raw_ms = _bench_cuda(
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
        triton_raw_scale_ms = _bench_cuda(
            lambda: triton_dsv4.quantized_linear_fp8(x_quant, weight, scale),
            warmup=warmup,
            iters=iters,
        )
        triton_cached_scale_ms = _bench_cuda(
            lambda: triton_dsv4.quantized_linear_fp8(x_quant, weight, scale_fp32),
            warmup=warmup,
            iters=iters,
        )

    with _with_env("MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON", "1"):
        with _with_env("MINISGL_DSV4_SM80_FP8_GEMM", None):
            fallback_ms = _bench_cuda(
                lambda: dsv4_kernel.quantized_linear_ref(x, weight, scale, weight_kind="fp8"),
                warmup=max(1, warmup // 2),
                iters=max(3, min(iters, 10)),
            )

    w_dequant = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=x.dtype)
    dequant_ms = _bench_cuda(
        lambda: dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=x.dtype),
        warmup=max(1, warmup // 2),
        iters=max(3, min(iters, 10)),
    )
    matmul_after_dequant_ms = _bench_cuda(
        lambda: F.linear(x_quant, w_dequant),
        warmup=warmup,
        iters=iters,
    )

    error = (actual.float() - expected.float()).abs()
    return {
        "owner": owner,
        "m": m,
        "k": k,
        "n": n,
        "wrapper_raw_ms": wrapper_raw_ms,
        "activation_quant_ms": activation_quant_ms,
        "triton_raw_scale_ms": triton_raw_scale_ms,
        "triton_cached_scale_ms": triton_cached_scale_ms,
        "fallback_total_ms": fallback_ms,
        "weight_dequant_ms": dequant_ms,
        "matmul_after_dequant_ms": matmul_after_dequant_ms,
        "cached_scale_delta_ms": triton_cached_scale_ms - triton_raw_scale_ms,
        "intrinsic_share_of_wrapper": (
            triton_cached_scale_ms / wrapper_raw_ms if wrapper_raw_ms > 0 else None
        ),
        "activation_share_of_wrapper": (
            activation_quant_ms / wrapper_raw_ms if wrapper_raw_ms > 0 else None
        ),
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "allclose_3e_2": bool(torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Real-Weight FP8 Linear Contract Microbench")
    lines.append("")
    lines.append(f"- Model path: `{report['model_path']}`")
    lines.append(f"- CUDA device: `{report['cuda_device']}`")
    lines.append(f"- M values: `{report['m_values']}`")
    lines.append(f"- Warmup/iters: `{report['warmup']}` / `{report['iters']}`")
    lines.append("")
    lines.append(
        "| Owner | M | K | N | wrapper ms | activation ms | intrinsic cached-scale ms | fallback ms | intrinsic share | ok |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in report["results"]:
        lines.append(
            "| `{owner}` | {m} | {k} | {n} | `{wrapper:.4f}` | `{act:.4f}` | `{intr:.4f}` | `{fb:.4f}` | `{share:.2%}` | `{ok}` |".format(
                owner=row["owner"],
                m=row["m"],
                k=row["k"],
                n=row["n"],
                wrapper=row["wrapper_raw_ms"],
                act=row["activation_quant_ms"],
                intr=row["triton_cached_scale_ms"],
                fb=row["fallback_total_ms"],
                share=row["intrinsic_share_of_wrapper"] or 0.0,
                ok=row["allclose_3e_2"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/models/DeepSeek-V4-Flash"))
    parser.add_argument("--owners", default="attn.q_wqb.layer0,attn.wo_b.layer0,indexer.wq_b.layer2")
    parser.add_argument("--m-values", default="1,4,8,16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=757)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_projection_gemm_backend_parity/raw/real_fp8_linear_microbench.json"
        ),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_projection_gemm_backend_parity/summaries/real_fp8_linear_microbench.md"
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
    m_values = _parse_csv_ints(args.m_values)
    started = time.time()
    results: list[dict[str, Any]] = []
    for owner_idx, owner in enumerate(owners):
        weight, scale = _load_owner(args.model_path, weight_map, owner, device)
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
                f"{owner} M={m}: wrapper={row['wrapper_raw_ms']:.4f} ms, "
                f"intrinsic={row['triton_cached_scale_ms']:.4f} ms, "
                f"act={row['activation_quant_ms']:.4f} ms, ok={row['allclose_3e_2']}"
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

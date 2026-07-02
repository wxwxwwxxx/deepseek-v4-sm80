#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Callable

from safetensors import safe_open
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


def load_index(model_path: Path) -> dict[str, str]:
    with (model_path / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def load_tensor(model_path: Path, index: dict[str, str], name: str, device: torch.device) -> torch.Tensor:
    shard = index.get(name)
    if shard is None:
        raise KeyError(f"tensor not found in safetensors index: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    return tensor.to(device=device)


def timed_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    # Keep the result alive until after synchronization.
    _ = out
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def benchmark(model_path: Path, layer: int, tokens: list[int], warmup: int, iters: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("focused projection microbench requires CUDA")
    device = torch.device("cuda:0")
    index = load_index(model_path)

    hc_attn_fn = load_tensor(model_path, index, f"layers.{layer}.hc_attn_fn", device)
    hc_ffn_fn = load_tensor(model_path, index, f"layers.{layer}.hc_ffn_fn", device)
    gate_weight = load_tensor(model_path, index, f"layers.{layer}.ffn.gate.weight", device)
    wq_a = load_tensor(model_path, index, f"layers.{layer}.attn.wq_a.weight", device)
    wq_a_scale = load_tensor(model_path, index, f"layers.{layer}.attn.wq_a.scale", device)
    wkv = load_tensor(model_path, index, f"layers.{layer}.attn.wkv.weight", device)
    wkv_scale = load_tensor(model_path, index, f"layers.{layer}.attn.wkv.scale", device)

    fused_wqa = dsv4_kernel.dequant_fp8_weight(
        wq_a,
        wq_a_scale,
        out_dtype=torch.bfloat16,
    )
    fused_wkv = dsv4_kernel.dequant_fp8_weight(
        wkv,
        wkv_scale,
        out_dtype=torch.bfloat16,
    )
    fused_wqa_wkv = torch.cat((fused_wqa, fused_wkv), dim=0).contiguous()
    torch.cuda.synchronize()

    results: list[dict[str, object]] = []
    for m in tokens:
        x_hc = torch.randn(m, hc_attn_fn.shape[1], device=device, dtype=torch.bfloat16)
        x_hidden = torch.randn(m, gate_weight.shape[1], device=device, dtype=torch.bfloat16)
        x_q = torch.randn(m, wq_a.shape[1], device=device, dtype=torch.bfloat16)
        x_q_quant = dsv4_kernel.quantize_fp8_activation_ref(x_q)

        cases: list[tuple[str, str, Callable[[], torch.Tensor], dict[str, object]]] = [
            (
                "hc_attn_pre_linear_bf16_fp32_fallback",
                "HC pre linear",
                lambda x=x_hc, w=hc_attn_fn: F.linear(x.float(), w.float()),
                {
                    "shape": f"[{m}, {hc_attn_fn.shape[1]}] x [{hc_attn_fn.shape[0]}, {hc_attn_fn.shape[1]}]",
                    "backend_family_from_profile": "cuBLAS SGEMM/FP32 GEMM + cuBLASLt splitK/reduce",
                    "source_boundary": "DeepseekV4DecoderLayer._hc_pre -> linear_bf16_fp32_fallback",
                    "weight_bytes": tensor_bytes(hc_attn_fn),
                },
            ),
            (
                "hc_ffn_pre_linear_bf16_fp32_fallback",
                "HC pre linear",
                lambda x=x_hc, w=hc_ffn_fn: F.linear(x.float(), w.float()),
                {
                    "shape": f"[{m}, {hc_ffn_fn.shape[1]}] x [{hc_ffn_fn.shape[0]}, {hc_ffn_fn.shape[1]}]",
                    "backend_family_from_profile": "cuBLAS SGEMM/FP32 GEMM + cuBLASLt splitK/reduce",
                    "source_boundary": "DeepseekV4DecoderLayer._hc_pre -> linear_bf16_fp32_fallback",
                    "weight_bytes": tensor_bytes(hc_ffn_fn),
                },
            ),
            (
                "moe_router_gate_linear",
                "MoE router / route projection",
                lambda x=x_hidden, w=gate_weight: F.linear(x.float(), w.float()),
                {
                    "shape": f"[{m}, {gate_weight.shape[1]}] x [{gate_weight.shape[0]}, {gate_weight.shape[1]}]",
                    "backend_family_from_profile": "cuBLAS SGEMM/FP32 GEMM + cuBLASLt splitK/reduce",
                    "source_boundary": "DSV4FusedMoERunner.route -> DSV4MoEGate.forward -> moe_gate_fallback",
                    "weight_bytes": tensor_bytes(gate_weight),
                },
            ),
            (
                "attn_qproj_fused_wqa_wkv_cached_bf16_gemm_only",
                "attention WQA/WKV/compress",
                lambda x=x_q_quant, w=fused_wqa_wkv: F.linear(x, w),
                {
                    "shape": f"[{m}, {fused_wqa_wkv.shape[1]}] x [{fused_wqa_wkv.shape[0]}, {fused_wqa_wkv.shape[1]}]",
                    "backend_family_from_profile": "CUTLASS BF16 GEMM + cuBLASLt splitK/reduce",
                    "source_boundary": "DSV4Attention.forward q_proj cached fused WQA/WKV GEMM",
                    "weight_bytes": tensor_bytes(fused_wqa_wkv),
                },
            ),
            (
                "attn_qproj_fused_wqa_wkv_cached_bf16_with_act_quant",
                "attention WQA/WKV/compress",
                lambda x=x_q, w=fused_wqa_wkv: F.linear(dsv4_kernel.quantize_fp8_activation_ref(x), w),
                {
                    "shape": f"[{m}, {fused_wqa_wkv.shape[1]}] x [{fused_wqa_wkv.shape[0]}, {fused_wqa_wkv.shape[1]}]",
                    "backend_family_from_profile": "FP8 activation quant helper + CUTLASS BF16 GEMM",
                    "source_boundary": "DSV4Attention.forward q_proj activation quant + cached fused WQA/WKV GEMM",
                    "weight_bytes": tensor_bytes(fused_wqa_wkv),
                },
            ),
        ]

        for case_name, owner, fn, meta in cases:
            timed = timed_ms(fn, warmup=warmup, iters=iters)
            results.append(
                {
                    "case": case_name,
                    "owner": owner,
                    "tokens_m": m,
                    **meta,
                    **timed,
                }
            )

    return {
        "model_path": str(model_path),
        "layer": layer,
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "torch_version": torch.__version__,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "warmup": warmup,
        "iters": iters,
        "results": results,
    }


def render_markdown(data: dict[str, object]) -> str:
    lines = [
        "# Focused Projection/GEMM Microbench",
        "",
        f"- model: `{data['model_path']}`",
        f"- layer: `{data['layer']}`",
        f"- device: `{data['device']}` capability `{data['capability']}`",
        f"- warmup/iters: `{data['warmup']}` / `{data['iters']}`",
        f"- torch: `{data['torch_version']}`, matmul allow_tf32: `{data['matmul_allow_tf32']}`",
        "",
        "| Case | Owner | M | Shape | Mean ms | Median ms | Min ms | Backend family from profile |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in data["results"]:  # type: ignore[index]
        lines.append(
            "| `{}` | `{}` | {} | `{}` | `{:.6f}` | `{:.6f}` | `{:.6f}` | `{}` |".format(
                row["case"],
                row["owner"],
                row["tokens_m"],
                row["shape"],
                float(row["mean_ms"]),
                float(row["median_ms"]),
                float(row["min_ms"]),
                row["backend_family_from_profile"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/models/DeepSeek-V4-Flash"))
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = benchmark(args.model_path, args.layer, args.tokens, args.warmup, args.iters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True))
    args.output.with_suffix(".md").write_text(render_markdown(data))


if __name__ == "__main__":
    main()

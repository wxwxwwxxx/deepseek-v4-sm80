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


def error_stats(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    cand = candidate.float()
    base = baseline.float()
    diff = (cand - base).abs()
    denom = base.abs().clamp_min(1e-6)
    rel = diff / denom
    flat = diff.flatten()
    rel_flat = rel.flatten()
    p99 = 0.0 if flat.numel() == 0 else float(torch.quantile(flat, 0.99).item())
    rel_p99 = 0.0 if rel_flat.numel() == 0 else float(torch.quantile(rel_flat, 0.99).item())
    return {
        "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_err": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs_err": p99,
        "max_rel_err": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel_err": float(rel.mean().item()) if rel.numel() else 0.0,
        "p99_rel_err": rel_p99,
    }


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def shard_tensor(tensor: torch.Tensor, *, dim: int | None, tp_size: int, rank: int = 0) -> torch.Tensor:
    if dim is None or tp_size == 1:
        return tensor
    return tensor.chunk(tp_size, dim=dim)[rank].contiguous()


def dequant_fp8_local(
    model_path: Path,
    index: dict[str, str],
    weight_name: str,
    scale_name: str,
    device: torch.device,
    *,
    tp_size: int,
    split_dim: int | None,
) -> torch.Tensor:
    weight = shard_tensor(
        load_tensor(model_path, index, weight_name, device),
        dim=split_dim,
        tp_size=tp_size,
    )
    scale = shard_tensor(
        load_tensor(model_path, index, scale_name, device),
        dim=split_dim,
        tp_size=tp_size,
    )
    return dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=torch.bfloat16).contiguous()


def make_linear_variants(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, tuple[Callable[[], torch.Tensor], int, str]]:
    weight_t = weight.t().contiguous()
    addmm_bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    return {
        "baseline_f_linear": (
            lambda x=x, w=weight: F.linear(x, w),
            0,
            "current promoted cached-BF16 F.linear route",
        ),
        "pretransposed_mm": (
            lambda x=x, wt=weight_t: torch.mm(x, wt),
            tensor_bytes(weight_t),
            "pretransposed BF16 weight_t + torch.mm",
        ),
        "addmm_pretransposed": (
            lambda x=x, wt=weight_t, b=addmm_bias: torch.addmm(b, x, wt),
            tensor_bytes(weight_t) + tensor_bytes(addmm_bias),
            "pretransposed BF16 weight_t + torch.addmm",
        ),
    }


def make_bmm_variants(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, tuple[Callable[[], torch.Tensor], int, str]]:
    tokens, groups, d_per_group = x.shape
    baseline_x = x.transpose(0, 1).contiguous()
    return {
        "baseline_grouped_bmm": (
            lambda x=baseline_x, w=weight, t=tokens, g=groups: torch.bmm(x, w)
            .transpose(0, 1)
            .reshape(t, g * w.shape[2]),
            0,
            "current promoted cached-BF16 grouped torch.bmm route",
        ),
        "einsum_grouped": (
            lambda x=x, w=weight, t=tokens, g=groups: torch.einsum("tgd,gdr->tgr", x, w).reshape(
                t, g * w.shape[2]
            ),
            0,
            "equivalent grouped contraction through torch.einsum",
        ),
    }


def benchmark(model_path: Path, layer: int, tokens: list[int], warmup: int, iters: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("focused projection microbench requires CUDA")
    device = torch.device("cuda:0")
    index = load_index(model_path)

    tp_size = 8
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
    q_wqb = dequant_fp8_local(
        model_path,
        index,
        f"layers.{layer}.attn.wq_b.weight",
        f"layers.{layer}.attn.wq_b.scale",
        device,
        tp_size=tp_size,
        split_dim=0,
    )
    wo_b = dequant_fp8_local(
        model_path,
        index,
        f"layers.{layer}.attn.wo_b.weight",
        f"layers.{layer}.attn.wo_b.scale",
        device,
        tp_size=tp_size,
        split_dim=1,
    )
    indexer_wq_b = None
    indexer_weight_name = f"layers.{layer}.attn.indexer.wq_b.weight"
    indexer_scale_name = f"layers.{layer}.attn.indexer.wq_b.scale"
    if indexer_weight_name in index and indexer_scale_name in index:
        indexer_wq_b = dequant_fp8_local(
            model_path,
            index,
            indexer_weight_name,
            indexer_scale_name,
            device,
            tp_size=tp_size,
            split_dim=None,
        )
    shared_gate = dequant_fp8_local(
        model_path,
        index,
        f"layers.{layer}.ffn.shared_experts.w1.weight",
        f"layers.{layer}.ffn.shared_experts.w1.scale",
        device,
        tp_size=tp_size,
        split_dim=0,
    )
    shared_up = dequant_fp8_local(
        model_path,
        index,
        f"layers.{layer}.ffn.shared_experts.w3.weight",
        f"layers.{layer}.ffn.shared_experts.w3.scale",
        device,
        tp_size=tp_size,
        split_dim=0,
    )
    shared_gate_up = torch.cat((shared_gate, shared_up), dim=0).contiguous()
    shared_down = dequant_fp8_local(
        model_path,
        index,
        f"layers.{layer}.ffn.shared_experts.w2.weight",
        f"layers.{layer}.ffn.shared_experts.w2.scale",
        device,
        tp_size=tp_size,
        split_dim=1,
    )
    wo_a_weight = dequant_fp8_local(
        model_path,
        index,
        f"layers.{layer}.attn.wo_a.weight",
        f"layers.{layer}.attn.wo_a.scale",
        device,
        tp_size=tp_size,
        split_dim=0,
    )
    num_local_groups = 2
    o_lora_rank = wo_a_weight.shape[0] // num_local_groups
    d_per_group = wo_a_weight.shape[1]
    wo_a_bmm_weight = (
        wo_a_weight.view(num_local_groups, o_lora_rank, d_per_group)
        .transpose(1, 2)
        .contiguous()
    )
    torch.cuda.synchronize()

    results: list[dict[str, object]] = []
    for m in tokens:
        x_hc = torch.randn(m, hc_attn_fn.shape[1], device=device, dtype=torch.bfloat16)
        x_hidden = torch.randn(m, gate_weight.shape[1], device=device, dtype=torch.bfloat16)
        x_q = torch.randn(m, wq_a.shape[1], device=device, dtype=torch.bfloat16)
        x_q_quant = dsv4_kernel.quantize_fp8_activation_ref(x_q)
        x_lora = torch.randn(m, q_wqb.shape[1], device=device, dtype=torch.bfloat16)
        x_lora_quant = dsv4_kernel.quantize_fp8_activation_ref(x_lora)
        x_wo_b = torch.randn(m, wo_b.shape[1], device=device, dtype=torch.bfloat16)
        x_wo_b_quant = dsv4_kernel.quantize_fp8_activation_ref(x_wo_b)
        x_shared = torch.randn(m, shared_gate_up.shape[1], device=device, dtype=torch.bfloat16)
        x_shared_quant = dsv4_kernel.quantize_fp8_activation_ref(x_shared)
        x_shared_down = torch.randn(m, shared_down.shape[1], device=device, dtype=torch.bfloat16)
        x_shared_down_quant = dsv4_kernel.quantize_fp8_activation_ref(x_shared_down)
        x_wo_a = torch.randn(m, num_local_groups, d_per_group, device=device, dtype=torch.bfloat16)

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

        linear_groups: list[tuple[str, str, torch.Tensor, torch.Tensor, str, str]] = [
            (
                "attn_qproj_fused_wqa_wkv_cached_bf16_gemm_only",
                "attention WQA/WKV/compress",
                x_q_quant,
                fused_wqa_wkv,
                "CUTLASS BF16 GEMM + cuBLASLt splitK/reduce",
                "DSV4Attention.forward q_proj cached fused WQA/WKV GEMM",
            ),
            (
                "attn_q_wqb_cached_bf16_gemm_only",
                "attention q_wqb",
                x_lora_quant,
                q_wqb,
                "cuBLASLt BF16 GEMM",
                "DSV4Attention.forward q_wqb cached BF16",
            ),
            (
                "attn_wo_b_cached_bf16_gemm_only",
                "attention wo_b local",
                x_wo_b_quant,
                wo_b,
                "cuBLASLt BF16 GEMM",
                "DSV4Attention.forward wo_b cached BF16 local GEMM",
            ),
            (
                "shared_experts_gate_up_cached_bf16_gemm_only",
                "shared experts cached BF16",
                x_shared_quant,
                shared_gate_up,
                "CUTLASS BF16 GEMM + cuBLASLt splitK/reduce",
                "DSV4SharedExperts.forward gate/up cached BF16",
            ),
            (
                "shared_experts_down_cached_bf16_gemm_only",
                "shared experts cached BF16",
                x_shared_down_quant,
                shared_down,
                "CUTLASS BF16 GEMM",
                "DSV4SharedExperts.forward down cached BF16",
            ),
        ]
        if indexer_wq_b is not None:
            linear_groups.append(
                (
                    "indexer_wq_b_cached_bf16_gemm_only",
                    "indexer wq_b",
                    x_lora_quant,
                    indexer_wq_b,
                    "cuBLASLt BF16 GEMM",
                    "DSV4Indexer._wq_b_forward cached BF16",
                )
            )

        for case_name, owner, x, weight, backend, source_boundary in linear_groups:
            variants = make_linear_variants(x, weight)
            baseline = variants["baseline_f_linear"][0]()
            torch.cuda.synchronize()
            for variant_name, (fn, extra_bytes, description) in variants.items():
                candidate = fn()
                torch.cuda.synchronize()
                errors = error_stats(candidate, baseline)
                timed = timed_ms(fn, warmup=warmup, iters=iters)
                baseline_ms = None
                speedup = 0.0
                if variant_name != "baseline_f_linear":
                    baseline_timed = timed_ms(
                        variants["baseline_f_linear"][0],
                        warmup=max(5, warmup // 5),
                        iters=max(20, iters // 5),
                    )
                    baseline_ms = baseline_timed["mean_ms"]
                    speedup = (float(baseline_ms) - float(timed["mean_ms"])) / float(baseline_ms)
                results.append(
                    {
                        "case": case_name,
                        "owner": owner,
                        "tokens_m": m,
                        "variant": variant_name,
                        "variant_description": description,
                        "shape": f"[{m}, {weight.shape[1]}] x [{weight.shape[0]}, {weight.shape[1]}]",
                        "backend_family_from_profile": backend,
                        "source_boundary": source_boundary,
                        "weight_bytes": tensor_bytes(weight),
                        "extra_bytes": int(extra_bytes),
                        "baseline_mean_ms_for_speedup": baseline_ms,
                        "speedup_vs_baseline": speedup,
                        "graph_safe_expected": True,
                        **errors,
                        **timed,
                    }
                )

        bmm_variants = make_bmm_variants(x_wo_a, wo_a_bmm_weight)
        bmm_baseline = bmm_variants["baseline_grouped_bmm"][0]()
        torch.cuda.synchronize()
        for variant_name, (fn, extra_bytes, description) in bmm_variants.items():
            candidate = fn()
            torch.cuda.synchronize()
            errors = error_stats(candidate, bmm_baseline)
            timed = timed_ms(fn, warmup=warmup, iters=iters)
            baseline_ms = None
            speedup = 0.0
            if variant_name != "baseline_grouped_bmm":
                baseline_timed = timed_ms(
                    bmm_variants["baseline_grouped_bmm"][0],
                    warmup=max(5, warmup // 5),
                    iters=max(20, iters // 5),
                )
                baseline_ms = baseline_timed["mean_ms"]
                speedup = (float(baseline_ms) - float(timed["mean_ms"])) / float(baseline_ms)
            results.append(
                {
                    "case": "attn_wo_a_cached_bf16_grouped_bmm",
                    "owner": "attention wo_a",
                    "tokens_m": m,
                    "variant": variant_name,
                    "variant_description": description,
                    "shape": (
                        f"o=[{m}, {num_local_groups}, {d_per_group}] x "
                        f"weight=[{num_local_groups}, {d_per_group}, {o_lora_rank}]"
                    ),
                    "backend_family_from_profile": "cuBLASLt BF16 GEMM + cuBLASLt splitK/reduce",
                    "source_boundary": "DSV4Attention.forward wo_a cached BF16 grouped BMM",
                    "weight_bytes": tensor_bytes(wo_a_bmm_weight),
                    "extra_bytes": int(extra_bytes),
                    "baseline_mean_ms_for_speedup": baseline_ms,
                    "speedup_vs_baseline": speedup,
                    "graph_safe_expected": variant_name == "baseline_grouped_bmm",
                    **errors,
                    **timed,
                }
            )

    return {
        "model_path": str(model_path),
        "layer": layer,
        "tp_size_simulated": tp_size,
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
        f"- simulated TP size/rank: `{data.get('tp_size_simulated', 1)}` / `0`",
        f"- device: `{data['device']}` capability `{data['capability']}`",
        f"- warmup/iters: `{data['warmup']}` / `{data['iters']}`",
        f"- torch: `{data['torch_version']}`, matmul allow_tf32: `{data['matmul_allow_tf32']}`",
        "",
        "| Case | Variant | Owner | M | Shape | Mean ms | Median ms | Min ms | Speedup | Max abs err | Extra bytes | Backend family from profile |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["results"]:  # type: ignore[index]
        lines.append(
            "| `{}` | `{}` | `{}` | {} | `{}` | `{:.6f}` | `{:.6f}` | `{:.6f}` | `{:.2%}` | `{:.6g}` | `{}` | `{}` |".format(
                row["case"],
                row.get("variant", "legacy_baseline"),
                row["owner"],
                row["tokens_m"],
                row["shape"],
                float(row["mean_ms"]),
                float(row["median_ms"]),
                float(row["min_ms"]),
                float(row.get("speedup_vs_baseline", 0.0)),
                float(row.get("max_abs_err", 0.0)),
                int(row.get("extra_bytes", 0)),
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

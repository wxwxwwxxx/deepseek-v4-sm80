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


TimedFn = Callable[[], torch.Tensor]


def load_index(model_path: Path) -> dict[str, str]:
    with (model_path / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def load_tensor(
    model_path: Path,
    index: dict[str, str],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    shard = index.get(name)
    if shard is None:
        raise KeyError(f"tensor not found in safetensors index: {name}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    return tensor.to(device=device)


def shard_tensor(
    tensor: torch.Tensor,
    *,
    dim: int | None,
    tp_size: int,
    rank: int = 0,
) -> torch.Tensor:
    if dim is None or tp_size == 1:
        return tensor.contiguous()
    return tensor.chunk(tp_size, dim=dim)[rank].contiguous()


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def timed_ms(fn: TimedFn, *, warmup: int, iters: int) -> dict[str, float]:
    out = None
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
    cosine = torch.nn.functional.cosine_similarity(
        cand.flatten().unsqueeze(0),
        base.flatten().unsqueeze(0),
    )
    return {
        "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_err": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs_err": p99,
        "max_rel_err": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel_err": float(rel.mean().item()) if rel.numel() else 0.0,
        "p99_rel_err": rel_p99,
        "cosine": float(cosine.item()) if diff.numel() else 1.0,
    }


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    return dsv4_kernel.dequant_fp8_weight(
        weight,
        scale,
        out_dtype=torch.bfloat16,
    ).contiguous()


def triton_fp8_linear_or_raise(
    x_quant: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    out = dsv4_kernel._triton_dsv4_ops().quantized_linear_fp8(x_quant, weight, scale)
    if out is None:
        raise RuntimeError(
            "mini Triton FP8 linear rejected the tensor layout/dtype/shape for this case"
        )
    return out


def linear_case_rows(
    *,
    case: str,
    owner: str,
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale: torch.Tensor | None,
    cached_bf16_weight: torch.Tensor,
    tokens_m: int,
    warmup: int,
    iters: int,
    source_boundary: str,
    vllm_analogue: str,
    extra_fp8_cache_bytes: int = 0,
) -> list[dict[str, object]]:
    x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
    baseline_fn = lambda xq=x_quant, w=cached_bf16_weight: F.linear(xq, w)
    triton_fn = lambda xq=x_quant, w=weight_fp8, s=scale: triton_fp8_linear_or_raise(xq, w, s)
    on_the_fly_fn = lambda x=x, w=weight_fp8, s=scale: dsv4_kernel.quantized_linear_ref(
        x,
        w,
        s,
        weight_kind="fp8",
        fp8_gemm=False,
    )

    baseline = baseline_fn()
    torch.cuda.synchronize()
    baseline_timed = timed_ms(baseline_fn, warmup=warmup, iters=iters)
    rows: list[dict[str, object]] = []

    variants: list[tuple[str, str, TimedFn, int, bool]] = [
        (
            "promoted_cached_bf16",
            "Promoted path: FP8-style activation rounding, cached BF16 dequantized weight, F.linear.",
            baseline_fn,
            tensor_bytes(cached_bf16_weight),
            True,
        ),
        (
            "fp8_weight_triton_direct",
            "Candidate: original FP8 weight/scales decoded inside mini Triton dot kernel.",
            triton_fn,
            int(extra_fp8_cache_bytes),
            True,
        ),
        (
            "dequant_on_the_fly_ref",
            "Diagnostic: activation rounding plus per-call FP8 weight dequantization then F.linear.",
            on_the_fly_fn,
            0,
            False,
        ),
    ]

    for variant, description, fn, extra_bytes, graph_safe in variants:
        candidate = fn()
        torch.cuda.synchronize()
        timed = baseline_timed if variant == "promoted_cached_bf16" else timed_ms(
            fn,
            warmup=warmup,
            iters=iters,
        )
        speedup = (
            (baseline_timed["mean_ms"] - timed["mean_ms"]) / baseline_timed["mean_ms"]
            if baseline_timed["mean_ms"] > 0
            else 0.0
        )
        rows.append(
            {
                "case": case,
                "owner": owner,
                "tokens_m": tokens_m,
                "variant": variant,
                "variant_description": description,
                "shape": f"[{tokens_m}, {weight_fp8.shape[1]}] x [{weight_fp8.shape[0]}, {weight_fp8.shape[1]}]",
                "source_boundary": source_boundary,
                "vllm_analogue": vllm_analogue,
                "baseline_mean_ms": baseline_timed["mean_ms"],
                "speedup_vs_promoted_cached_bf16": speedup,
                "weight_fp8_bytes": tensor_bytes(weight_fp8),
                "weight_scale_bytes": tensor_bytes(scale),
                "cached_bf16_weight_bytes": tensor_bytes(cached_bf16_weight),
                "extra_cache_or_workspace_bytes": int(extra_bytes),
                "graph_safe_expected": bool(graph_safe),
                **error_stats(candidate, baseline),
                **timed,
            }
        )
    return rows


def wo_a_case_rows(
    *,
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale: torch.Tensor | None,
    cached_bmm_weight: torch.Tensor,
    num_local_groups: int,
    o_lora_rank: int,
    tokens_m: int,
    warmup: int,
    iters: int,
) -> list[dict[str, object]]:
    tokens, groups, d_per_group = x.shape
    baseline_x = x.transpose(0, 1).contiguous()
    baseline_fn = (
        lambda x=baseline_x, w=cached_bmm_weight, t=tokens, g=groups: torch.bmm(x, w)
        .transpose(0, 1)
        .reshape(t, g * w.shape[2])
    )
    triton_fn = lambda x=x, w=weight_fp8, s=scale: dsv4_kernel._triton_dsv4_ops().wo_a_grouped_projection_fp8(
        x,
        w,
        s,
        num_local_groups=num_local_groups,
        o_lora_rank=o_lora_rank,
    )
    on_the_fly_fn = lambda x=x, w=weight_fp8, s=scale: dsv4_kernel.wo_a_grouped_projection_fallback(
        x,
        w,
        s,
        num_local_groups=num_local_groups,
        o_lora_rank=o_lora_rank,
    )

    baseline = baseline_fn()
    torch.cuda.synchronize()
    baseline_timed = timed_ms(baseline_fn, warmup=warmup, iters=iters)
    rows: list[dict[str, object]] = []
    variants: list[tuple[str, str, TimedFn, int, bool]] = [
        (
            "promoted_cached_bf16_grouped_bmm",
            "Promoted path: cached BF16 grouped BMM weight and torch.bmm.",
            baseline_fn,
            tensor_bytes(cached_bmm_weight),
            True,
        ),
        (
            "fp8_weight_triton_grouped",
            "Candidate: original FP8 wo_a weight/scales decoded inside grouped Triton dot kernel.",
            triton_fn,
            0,
            True,
        ),
        (
            "dequant_on_the_fly_ref",
            "Diagnostic: per-call FP8 weight dequantization then torch.einsum.",
            on_the_fly_fn,
            0,
            False,
        ),
    ]
    for variant, description, fn, extra_bytes, graph_safe in variants:
        candidate = fn()
        if candidate is None:
            raise RuntimeError("mini Triton wo_a grouped projection rejected this shape")
        torch.cuda.synchronize()
        timed = baseline_timed if variant.startswith("promoted") else timed_ms(
            fn,
            warmup=warmup,
            iters=iters,
        )
        speedup = (
            (baseline_timed["mean_ms"] - timed["mean_ms"]) / baseline_timed["mean_ms"]
            if baseline_timed["mean_ms"] > 0
            else 0.0
        )
        rows.append(
            {
                "case": "attn_wo_a_grouped_projection",
                "owner": "attention wo_a",
                "tokens_m": tokens_m,
                "variant": variant,
                "variant_description": description,
                "shape": f"o=[{tokens_m}, {num_local_groups}, {d_per_group}] x weight=[{num_local_groups * o_lora_rank}, {d_per_group}]",
                "source_boundary": "DSV4Attention.forward wo_a grouped projection",
                "vllm_analogue": "fused_inv_rope_fp8_quant + deepseek_v4_fp8_einsum",
                "baseline_mean_ms": baseline_timed["mean_ms"],
                "speedup_vs_promoted_cached_bf16": speedup,
                "weight_fp8_bytes": tensor_bytes(weight_fp8),
                "weight_scale_bytes": tensor_bytes(scale),
                "cached_bf16_weight_bytes": tensor_bytes(cached_bmm_weight),
                "extra_cache_or_workspace_bytes": int(extra_bytes),
                "graph_safe_expected": bool(graph_safe),
                **error_stats(candidate, baseline),
                **timed,
            }
        )
    return rows


def benchmark(
    model_path: Path,
    layer: int,
    tokens: list[int],
    warmup: int,
    iters: int,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("focused FP8 projection-cache microbench requires CUDA")
    device = torch.device("cuda:0")
    index = load_index(model_path)
    tp_size = 8

    wq_a = load_tensor(model_path, index, f"layers.{layer}.attn.wq_a.weight", device)
    wq_a_scale = load_tensor(model_path, index, f"layers.{layer}.attn.wq_a.scale", device)
    wkv = load_tensor(model_path, index, f"layers.{layer}.attn.wkv.weight", device)
    wkv_scale = load_tensor(model_path, index, f"layers.{layer}.attn.wkv.scale", device)
    fused_wqa_wkv_fp8 = torch.cat((wq_a, wkv), dim=0).contiguous()
    fused_wqa_wkv_scale = torch.cat((wq_a_scale, wkv_scale), dim=0).contiguous()
    fused_wqa_wkv_bf16 = dequant_fp8(fused_wqa_wkv_fp8, fused_wqa_wkv_scale)

    q_wqb_fp8 = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wq_b.weight", device),
        dim=0,
        tp_size=tp_size,
    )
    q_wqb_scale = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wq_b.scale", device),
        dim=0,
        tp_size=tp_size,
    )
    q_wqb_bf16 = dequant_fp8(q_wqb_fp8, q_wqb_scale)

    wo_b_fp8 = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wo_b.weight", device),
        dim=1,
        tp_size=tp_size,
    )
    wo_b_scale = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wo_b.scale", device),
        dim=1,
        tp_size=tp_size,
    )
    wo_b_bf16 = dequant_fp8(wo_b_fp8, wo_b_scale)

    indexer_wq_b_fp8 = None
    indexer_wq_b_scale = None
    indexer_wq_b_bf16 = None
    indexer_weight_name = f"layers.{layer}.attn.indexer.wq_b.weight"
    indexer_scale_name = f"layers.{layer}.attn.indexer.wq_b.scale"
    if indexer_weight_name in index and indexer_scale_name in index:
        indexer_wq_b_fp8 = load_tensor(model_path, index, indexer_weight_name, device).contiguous()
        indexer_wq_b_scale = load_tensor(model_path, index, indexer_scale_name, device).contiguous()
        indexer_wq_b_bf16 = dequant_fp8(indexer_wq_b_fp8, indexer_wq_b_scale)

    shared_gate_fp8 = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.ffn.shared_experts.w1.weight", device),
        dim=0,
        tp_size=tp_size,
    )
    shared_gate_scale = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.ffn.shared_experts.w1.scale", device),
        dim=0,
        tp_size=tp_size,
    )
    shared_up_fp8 = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.ffn.shared_experts.w3.weight", device),
        dim=0,
        tp_size=tp_size,
    )
    shared_up_scale = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.ffn.shared_experts.w3.scale", device),
        dim=0,
        tp_size=tp_size,
    )
    shared_gate_up_fp8 = torch.cat((shared_gate_fp8, shared_up_fp8), dim=0).contiguous()
    shared_gate_up_scale = torch.cat((shared_gate_scale, shared_up_scale), dim=0).contiguous()
    shared_gate_up_bf16 = dequant_fp8(shared_gate_up_fp8, shared_gate_up_scale)

    shared_down_fp8 = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.ffn.shared_experts.w2.weight", device),
        dim=1,
        tp_size=tp_size,
    )
    shared_down_scale = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.ffn.shared_experts.w2.scale", device),
        dim=1,
        tp_size=tp_size,
    )
    shared_down_bf16 = dequant_fp8(shared_down_fp8, shared_down_scale)

    wo_a_fp8 = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wo_a.weight", device),
        dim=0,
        tp_size=tp_size,
    )
    wo_a_scale = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wo_a.scale", device),
        dim=0,
        tp_size=tp_size,
    )
    wo_a_bf16 = dequant_fp8(wo_a_fp8, wo_a_scale)
    num_local_groups = 2
    o_lora_rank = wo_a_bf16.shape[0] // num_local_groups
    d_per_group = wo_a_bf16.shape[1]
    wo_a_bmm_weight = (
        wo_a_bf16.view(num_local_groups, o_lora_rank, d_per_group)
        .transpose(1, 2)
        .contiguous()
    )

    torch.cuda.synchronize()
    results: list[dict[str, object]] = []
    for m in tokens:
        x_hidden = torch.randn(m, fused_wqa_wkv_fp8.shape[1], device=device, dtype=torch.bfloat16)
        x_lora = torch.randn(m, q_wqb_fp8.shape[1], device=device, dtype=torch.bfloat16)
        x_wo_b = torch.randn(m, wo_b_fp8.shape[1], device=device, dtype=torch.bfloat16)
        x_shared = torch.randn(
            m,
            shared_gate_up_fp8.shape[1],
            device=device,
            dtype=torch.bfloat16,
        )
        x_shared_down = torch.randn(
            m,
            shared_down_fp8.shape[1],
            device=device,
            dtype=torch.bfloat16,
        )
        x_wo_a = torch.randn(
            m,
            num_local_groups,
            d_per_group,
            device=device,
            dtype=torch.bfloat16,
        )

        results.extend(
            linear_case_rows(
                case="attn_qproj_fused_wqa_wkv",
                owner="attention WQA/WKV/compress",
                x=x_hidden,
                weight_fp8=fused_wqa_wkv_fp8,
                scale=fused_wqa_wkv_scale,
                cached_bf16_weight=fused_wqa_wkv_bf16,
                tokens_m=m,
                warmup=warmup,
                iters=iters,
                source_boundary="DSV4Attention.forward q_proj fused WQA/WKV",
                vllm_analogue="DeepseekV4Attention.fused_wqa_wkv MergedColumnParallelLinear",
                extra_fp8_cache_bytes=tensor_bytes(fused_wqa_wkv_fp8)
                + tensor_bytes(fused_wqa_wkv_scale),
            )
        )
        results.extend(
            linear_case_rows(
                case="attn_q_wqb",
                owner="attention q_wqb",
                x=x_lora,
                weight_fp8=q_wqb_fp8,
                scale=q_wqb_scale,
                cached_bf16_weight=q_wqb_bf16,
                tokens_m=m,
                warmup=warmup,
                iters=iters,
                source_boundary="DSV4Attention.forward q_wqb",
                vllm_analogue="DeepseekV4Attention.wq_b ColumnParallelLinear",
            )
        )
        results.extend(
            linear_case_rows(
                case="attn_wo_b_local",
                owner="attention wo_b local",
                x=x_wo_b,
                weight_fp8=wo_b_fp8,
                scale=wo_b_scale,
                cached_bf16_weight=wo_b_bf16,
                tokens_m=m,
                warmup=warmup,
                iters=iters,
                source_boundary="DSV4Attention.forward wo_b local projection",
                vllm_analogue="DeepseekV4Attention.wo_b RowParallelLinear",
            )
        )
        if indexer_wq_b_fp8 is not None and indexer_wq_b_scale is not None and indexer_wq_b_bf16 is not None:
            results.extend(
                linear_case_rows(
                    case="indexer_wq_b",
                    owner="indexer wq_b",
                    x=x_lora,
                    weight_fp8=indexer_wq_b_fp8,
                    scale=indexer_wq_b_scale,
                    cached_bf16_weight=indexer_wq_b_bf16,
                    tokens_m=m,
                    warmup=warmup,
                    iters=iters,
                    source_boundary="DSV4Indexer._wq_b_forward",
                    vllm_analogue="DeepseekV4Indexer fused_indexer_q_rope_quant input projection",
                )
            )
        results.extend(
            linear_case_rows(
                case="shared_experts_gate_up",
                owner="shared experts gate/up",
                x=x_shared,
                weight_fp8=shared_gate_up_fp8,
                scale=shared_gate_up_scale,
                cached_bf16_weight=shared_gate_up_bf16,
                tokens_m=m,
                warmup=warmup,
                iters=iters,
                source_boundary="DSV4SharedExperts.forward gate_up_proj",
                vllm_analogue="DeepseekV4MLP shared gate/up ColumnParallelLinear under deepseek_v4_fp8",
                extra_fp8_cache_bytes=tensor_bytes(shared_gate_up_fp8)
                + tensor_bytes(shared_gate_up_scale),
            )
        )
        results.extend(
            linear_case_rows(
                case="shared_experts_down",
                owner="shared experts down",
                x=x_shared_down,
                weight_fp8=shared_down_fp8,
                scale=shared_down_scale,
                cached_bf16_weight=shared_down_bf16,
                tokens_m=m,
                warmup=warmup,
                iters=iters,
                source_boundary="DSV4SharedExperts.forward down_proj",
                vllm_analogue="DeepseekV4MLP shared down RowParallelLinear under deepseek_v4_fp8",
            )
        )
        results.extend(
            wo_a_case_rows(
                x=x_wo_a,
                weight_fp8=wo_a_fp8,
                scale=wo_a_scale,
                cached_bmm_weight=wo_a_bmm_weight,
                num_local_groups=num_local_groups,
                o_lora_rank=o_lora_rank,
                tokens_m=m,
                warmup=warmup,
                iters=iters,
            )
        )

    candidate_rows = [
        row
        for row in results
        if str(row["variant"]).startswith("fp8_weight_triton")
    ]
    owners_passing = sorted(
        {
            str(row["owner"])
            for row in candidate_rows
            if int(row["tokens_m"]) == 4
            and float(row["speedup_vs_promoted_cached_bf16"]) >= 0.15
            and float(row["max_abs_err"]) < 1.0
        }
    )
    gate_pass = len(owners_passing) >= 2
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
        "gate": {
            "name": "focused runtime implementation gate",
            "pass": gate_pass,
            "criterion": "at least two representative owners improve >=15% at M=4 with no obvious output error",
            "owners_passing_m4": owners_passing,
        },
        "results": results,
    }


def render_markdown(data: dict[str, object]) -> str:
    lines = [
        "# Focused FP8 Projection-Cache Microbench",
        "",
        f"- model: `{data['model_path']}`",
        f"- layer: `{data['layer']}`",
        f"- simulated TP size/rank: `{data.get('tp_size_simulated', 1)}` / `0`",
        f"- device: `{data['device']}` capability `{data['capability']}`",
        f"- warmup/iters: `{data['warmup']}` / `{data['iters']}`",
        f"- torch: `{data['torch_version']}`, matmul allow_tf32: `{data['matmul_allow_tf32']}`",
        f"- gate pass: `{data['gate']['pass']}`",
        f"- owners passing at M=4: `{', '.join(data['gate']['owners_passing_m4']) or 'none'}`",
        "",
        "| Case | Variant | Owner | M | Mean ms | Median ms | Min ms | Speedup | Max abs err | P99 abs err | Mean rel err | Cosine | Extra bytes | Graph safe |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in data["results"]:  # type: ignore[index]
        lines.append(
            "| `{}` | `{}` | `{}` | {} | `{:.6f}` | `{:.6f}` | `{:.6f}` | `{:.2%}` | `{:.6g}` | `{:.6g}` | `{:.6g}` | `{:.8f}` | `{}` | `{}` |".format(
                row["case"],
                row["variant"],
                row["owner"],
                row["tokens_m"],
                float(row["mean_ms"]),
                float(row["median_ms"]),
                float(row["min_ms"]),
                float(row["speedup_vs_promoted_cached_bf16"]),
                float(row["max_abs_err"]),
                float(row["p99_abs_err"]),
                float(row["mean_rel_err"]),
                float(row["cosine"]),
                int(row["extra_cache_or_workspace_bytes"]),
                row["graph_safe_expected"],
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

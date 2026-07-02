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
    _ = out
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def error_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = (candidate.float() - reference.float()).abs().flatten()
    denom = reference.float().abs().flatten().clamp_min(1e-8)
    rel = diff / denom
    return {
        "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_err": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs_err": float(torch.quantile(diff, 0.99).item()) if diff.numel() else 0.0,
        "max_rel_err": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel_err": float(rel.mean().item()) if rel.numel() else 0.0,
    }


def topk_overlap(candidate: torch.Tensor, reference: torch.Tensor, k: int) -> dict[str, float]:
    cand_topk = candidate.topk(k, dim=-1).indices
    ref_topk = reference.topk(k, dim=-1).indices
    matches = (cand_topk[:, :, None] == ref_topk[:, None, :]).any(dim=-1).float()
    exact = (cand_topk == ref_topk).all(dim=-1).float()
    return {
        "topk_set_overlap": float(matches.mean().item()),
        "topk_exact_order_match": float(exact.mean().item()),
        "changed_rows": float((1.0 - exact).sum().item()),
        "rows": float(reference.shape[0]),
    }


def set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    torch.set_float32_matmul_precision("high" if enabled else "highest")


def benchmark(
    model_path: Path,
    layer: int,
    tokens: list[int],
    topk: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("HC/router precision probe requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    index = load_index(model_path)

    hc_attn_fn = load_tensor(model_path, index, f"layers.{layer}.hc_attn_fn", device)
    hc_ffn_fn = load_tensor(model_path, index, f"layers.{layer}.hc_ffn_fn", device)
    gate_weight = load_tensor(model_path, index, f"layers.{layer}.ffn.gate.weight", device)

    rows: list[dict[str, object]] = []
    original_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        for m in tokens:
            x_hc = torch.randn(m, hc_attn_fn.shape[1], device=device, dtype=torch.bfloat16)
            x_hidden = torch.randn(m, gate_weight.shape[1], device=device, dtype=torch.bfloat16)

            owner_inputs = [
                ("hc_attn_pre_linear", "HC pre linear", x_hc, hc_attn_fn),
                ("hc_ffn_pre_linear", "HC pre linear", x_hc, hc_ffn_fn),
                ("moe_router_gate_linear", "MoE router / route projection", x_hidden, gate_weight),
            ]
            for case, owner, x, weight in owner_inputs:
                set_tf32(False)
                ref = F.linear(x.float(), weight.float())
                baseline = timed_ms(
                    lambda x=x, weight=weight: F.linear(x.float(), weight.float()),
                    warmup=warmup,
                    iters=iters,
                )

                set_tf32(True)
                tf32_out = F.linear(x.float(), weight.float())
                tf32 = timed_ms(
                    lambda x=x, weight=weight: F.linear(x.float(), weight.float()),
                    warmup=warmup,
                    iters=iters,
                )

                set_tf32(False)
                weight_bf16 = weight.to(torch.bfloat16)
                bf16_out = F.linear(x, weight_bf16)
                bf16 = timed_ms(
                    lambda x=x, weight_bf16=weight_bf16: F.linear(x, weight_bf16),
                    warmup=warmup,
                    iters=iters,
                )

                row: dict[str, object] = {
                    "case": case,
                    "owner": owner,
                    "tokens_m": m,
                    "shape": f"[{m}, {weight.shape[1]}] x [{weight.shape[0]}, {weight.shape[1]}]",
                    "baseline_fp32_sgemm": baseline,
                    "tf32_enabled": tf32,
                    "bf16_like": bf16,
                    "tf32_error": error_stats(tf32_out, ref),
                    "bf16_like_error": error_stats(bf16_out, ref),
                }
                if case == "moe_router_gate_linear":
                    row["router_tf32_topk"] = topk_overlap(tf32_out, ref, topk)
                    row["router_bf16_like_topk"] = topk_overlap(bf16_out, ref, topk)
                rows.append(row)
    finally:
        set_tf32(original_tf32)

    return {
        "model_path": str(model_path),
        "layer": layer,
        "device": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "torch_version": torch.__version__,
        "topk": topk,
        "warmup": warmup,
        "iters": iters,
        "seed": seed,
        "results": rows,
        "profile_reference_seconds": {
            "hc_pre_linear": 0.178373,
            "moe_router_route_projection": 0.097109,
            "combined_fp32_sgemm_cluster": 0.257269,
        },
        "notes": {
            "tf32_control": "torch.backends.cuda.matmul.allow_tf32 is process-global; this probe toggles it locally around calls, but a runtime implementation would need a scoped custom owner or accept global policy risk.",
            "bf16_like": "BF16-like uses BF16 input and BF16-cast weight with F.linear; it is a risk probe only, not a proposed promoted contract.",
        },
    }


def pct_delta(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0 if baseline else 0.0


def render_markdown(data: dict[str, object]) -> str:
    lines = [
        "# HC/Router Precision Probe",
        "",
        f"- model: `{data['model_path']}`",
        f"- layer: `{data['layer']}`",
        f"- device: `{data['device']}` capability `{data['capability']}`",
        f"- warmup/iters: `{data['warmup']}` / `{data['iters']}`",
        f"- torch: `{data['torch_version']}`",
        f"- router top-k: `{data['topk']}`",
        "",
        "| Case | Owner | M | FP32 mean ms | TF32 mean ms | TF32 delta | BF16-like mean ms | BF16-like delta | TF32 max abs | BF16 max abs | Router TF32 overlap | Router BF16 overlap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["results"]:  # type: ignore[index]
        base = float(row["baseline_fp32_sgemm"]["mean_ms"])
        tf32 = float(row["tf32_enabled"]["mean_ms"])
        bf16 = float(row["bf16_like"]["mean_ms"])
        router_tf32 = row.get("router_tf32_topk", {})
        router_bf16 = row.get("router_bf16_like_topk", {})
        lines.append(
            "| `{}` | `{}` | {} | `{:.6f}` | `{:.6f}` | `{:+.2f}%` | `{:.6f}` | `{:+.2f}%` | `{:.6g}` | `{:.6g}` | `{}` | `{}` |".format(
                row["case"],
                row["owner"],
                row["tokens_m"],
                base,
                tf32,
                pct_delta(tf32, base),
                bf16,
                pct_delta(bf16, base),
                float(row["tf32_error"]["max_abs_err"]),
                float(row["bf16_like_error"]["max_abs_err"]),
                f"{float(router_tf32.get('topk_set_overlap', 0.0)):.6f}" if router_tf32 else "",
                f"{float(router_bf16.get('topk_set_overlap', 0.0)):.6f}" if router_bf16 else "",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/models/DeepSeek-V4-Flash"))
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7171)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = benchmark(
        args.model_path,
        args.layer,
        args.tokens,
        args.topk,
        args.warmup,
        args.iters,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True))
    args.output.with_suffix(".md").write_text(render_markdown(data))


if __name__ == "__main__":
    main()

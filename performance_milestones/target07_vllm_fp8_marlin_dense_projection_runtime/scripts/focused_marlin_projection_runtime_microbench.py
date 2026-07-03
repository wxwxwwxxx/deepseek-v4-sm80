#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402
from minisgl.kernel import vllm_fp8_marlin  # noqa: E402

TP_SIZE = 8


@dataclass(frozen=True)
class DenseOwner:
    case: str
    owner: str
    weight_key: str
    scale_key: str
    shard_dim: int
    source_boundary: str


PHASE_A_OWNERS: tuple[DenseOwner, ...] = (
    DenseOwner(
        case="attn_q_wqb",
        owner="attention q_wqb",
        weight_key="layers.{layer}.attn.wq_b.weight",
        scale_key="layers.{layer}.attn.wq_b.scale",
        shard_dim=0,
        source_boundary="DSV4Attention.forward q_wqb",
    ),
    DenseOwner(
        case="attn_wo_b_local",
        owner="attention wo_b local",
        weight_key="layers.{layer}.attn.wo_b.weight",
        scale_key="layers.{layer}.attn.wo_b.scale",
        shard_dim=1,
        source_boundary="DSV4Attention.forward wo_b local projection before all-reduce",
    ),
    DenseOwner(
        case="shared_experts_down",
        owner="shared experts down",
        weight_key="layers.{layer}.ffn.shared_experts.w2.weight",
        scale_key="layers.{layer}.ffn.shared_experts.w2.scale",
        shard_dim=1,
        source_boundary="DSV4SharedExperts.forward down_proj before all-reduce",
    ),
)


def load_index(model_path: Path) -> dict[str, str]:
    with (model_path / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def load_tensor(
    model_path: Path,
    index: dict[str, str],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    shard = index.get(key)
    if shard is None:
        raise KeyError(f"tensor not found in safetensors index: {key}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(device=device)


def shard_tensor(tensor: torch.Tensor, *, dim: int, tp_size: int = TP_SIZE) -> torch.Tensor:
    return tensor.chunk(tp_size, dim=dim)[0].contiguous()


def sync() -> None:
    torch.cuda.synchronize()


def timed_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        sync()
        samples.append(float(start.elapsed_time(end)))
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
    flat = diff.flatten()
    cosine = F.cosine_similarity(cand.flatten().unsqueeze(0), base.flatten().unsqueeze(0))
    return {
        "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_err": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs_err": float(torch.quantile(flat, 0.99).item()) if flat.numel() else 0.0,
        "cosine": float(cosine.item()) if diff.numel() else 1.0,
    }


def make_input(m: int, k: int, *, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=gen)


def bench_owner(
    *,
    model_path: Path,
    index: dict[str, str],
    layer: int,
    owner: DenseOwner,
    tokens: list[int],
    warmup: int,
    iters: int,
    device: torch.device,
    owner_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weight = shard_tensor(
        load_tensor(model_path, index, owner.weight_key.format(layer=layer), device),
        dim=owner.shard_dim,
    )
    scale = shard_tensor(
        load_tensor(model_path, index, owner.scale_key.format(layer=layer), device),
        dim=owner.shard_dim,
    )
    n, k = weight.shape

    sync()
    prep_start = time.perf_counter()
    cached_bf16 = dsv4_kernel.dequant_fp8_weight(
        weight,
        scale,
        out_dtype=torch.bfloat16,
    ).contiguous()
    sync()
    cached_bf16_prep_ms = (time.perf_counter() - prep_start) * 1000.0

    before_marlin_allocated = torch.cuda.memory_allocated(device)
    sync()
    prep_start = time.perf_counter()
    marlin = vllm_fp8_marlin.prepare_linear(
        weight,
        scale,
        owner_label=owner.owner,
    )
    sync()
    marlin_prep_ms = (time.perf_counter() - prep_start) * 1000.0
    after_marlin_allocated = torch.cuda.memory_allocated(device)

    prep_rows = [
        {
            "case": owner.case,
            "owner": owner.owner,
            "backend": "promoted_cached_bf16",
            "prep_ms": cached_bf16_prep_ms,
            "persistent_bytes": cached_bf16.numel() * cached_bf16.element_size(),
            "workspace_bytes": 0,
            "original_weight_bytes": weight.numel() * weight.element_size(),
            "original_scale_bytes": scale.numel() * scale.element_size(),
        },
        {
            "case": owner.case,
            "owner": owner.owner,
            "backend": "mini_runtime_vllm_fp8_marlin_w8a16_block",
            "prep_ms": marlin_prep_ms,
            "persistent_bytes": marlin.persistent_bytes,
            "prepared_weight_bytes": marlin.prepared_weight_bytes,
            "prepared_scale_bytes": marlin.prepared_scale_bytes,
            "workspace_bytes": marlin.workspace_bytes,
            "original_weight_bytes": marlin.original_weight_bytes,
            "original_scale_bytes": marlin.original_scale_bytes,
            "allocated_delta_bytes": int(after_marlin_allocated - before_marlin_allocated),
        },
    ]

    rows: list[dict[str, Any]] = []
    for m in tokens:
        x = make_input(m, k, seed=7740 + owner_index * 100 + m, device=device)
        baseline_fn = lambda x=x: F.linear(dsv4_kernel.quantize_fp8_activation_ref(x), cached_bf16)
        marlin_fn = lambda x=x: vllm_fp8_marlin.apply_linear(x, marlin)
        baseline = baseline_fn()
        candidate = marlin_fn()
        sync()
        baseline_timing = timed_ms(baseline_fn, warmup=warmup, iters=iters)
        marlin_timing = timed_ms(marlin_fn, warmup=warmup, iters=iters)
        err = error_stats(candidate, baseline)
        rows.append(
            {
                "case": owner.case,
                "owner": owner.owner,
                "tokens_m": m,
                "n": n,
                "k": k,
                "source_boundary": owner.source_boundary,
                "baseline_backend": "promoted_cached_bf16_total",
                "candidate_backend": "mini_runtime_vllm_fp8_marlin_w8a16_block",
                "baseline_mean_ms": baseline_timing["mean_ms"],
                "candidate_mean_ms": marlin_timing["mean_ms"],
                "speedup_vs_promoted_cached_bf16_pct": (
                    (baseline_timing["mean_ms"] - marlin_timing["mean_ms"])
                    / baseline_timing["mean_ms"]
                    * 100.0
                ),
                "activation_policy": "BF16 activations for Marlin; no replay-time activation quantization",
                "repack_required_per_decode": False,
                "per_decode_workspace_allocation": False,
                **err,
            }
        )

    return rows, prep_rows


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Focused Marlin Projection Runtime Microbench",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- python: `{report['python']}`",
        f"- torch: `{report['torch']}`",
        f"- device: `{report['device']}`",
        f"- model_path: `{report['model_path']}`",
        "",
        "## Latency",
        "",
        "| Owner | M | Baseline ms | Marlin ms | Speedup | Mean abs | P99 abs | Cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {owner} | {tokens_m} | `{baseline_mean_ms:.6f}` | "
            "`{candidate_mean_ms:.6f}` | `{speedup_vs_promoted_cached_bf16_pct:.2f}%` | "
            "`{mean_abs_err:.6f}` | `{p99_abs_err:.6f}` | `{cosine:.8f}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Prep And Persistent Bytes",
            "",
            "| Owner | Backend | Prep ms | Persistent bytes | Workspace bytes | Original weight bytes | Original scale bytes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["prep_rows"]:
        lines.append(
            "| {owner} | `{backend}` | `{prep_ms:.3f}` | `{persistent_bytes}` | "
            "`{workspace_bytes}` | `{original_weight_bytes}` | `{original_scale_bytes}` |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("focused Marlin projection microbench requires CUDA")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    index = load_index(args.model_path)
    rows: list[dict[str, Any]] = []
    prep_rows: list[dict[str, Any]] = []
    for owner_index, owner in enumerate(PHASE_A_OWNERS):
        owner_rows, owner_prep_rows = bench_owner(
            model_path=args.model_path,
            index=index,
            layer=args.layer,
            owner=owner,
            tokens=args.tokens,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            owner_index=owner_index,
        )
        rows.extend(owner_rows)
        prep_rows.extend(owner_prep_rows)

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "model_path": str(args.model_path),
        "layer": args.layer,
        "tokens": args.tokens,
        "rows": rows,
        "prep_rows": prep_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown_output, report)


if __name__ == "__main__":
    main()

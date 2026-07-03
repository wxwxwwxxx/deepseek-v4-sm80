#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

TP_SIZE = 8
FP8_MAX = 448.0
TimedFn = Callable[[], torch.Tensor]


@dataclass(frozen=True)
class DenseOwner:
    case: str
    owner: str
    weight_keys: tuple[str, ...]
    scale_keys: tuple[str, ...]
    shard_dim: int | None
    source_boundary: str
    vllm_analogue: str


@dataclass
class PreparedBackend:
    backend: str
    apply_fn: Callable[[torch.Tensor], torch.Tensor]
    prep_ms: float
    original_weight_bytes: int
    original_scale_bytes: int
    prepared_weight_bytes: int
    prepared_scale_bytes: int
    workspace_bytes: int
    persistent_bytes: int
    notes: str


DENSE_OWNERS: tuple[DenseOwner, ...] = (
    DenseOwner(
        case="attn_q_wqb",
        owner="attn.q_wqb",
        weight_keys=("layers.{layer}.attn.wq_b.weight",),
        scale_keys=("layers.{layer}.attn.wq_b.scale",),
        shard_dim=0,
        source_boundary="DSV4Attention.forward q_wqb",
        vllm_analogue="DeepseekV4Attention.wq_b ColumnParallelLinear",
    ),
    DenseOwner(
        case="attn_wo_b_local",
        owner="attn.wo_b local",
        weight_keys=("layers.{layer}.attn.wo_b.weight",),
        scale_keys=("layers.{layer}.attn.wo_b.scale",),
        shard_dim=1,
        source_boundary="DSV4Attention.forward wo_b local projection before all-reduce",
        vllm_analogue="DeepseekV4Attention.wo_b RowParallelLinear local GEMM",
    ),
    DenseOwner(
        case="shared_experts_down",
        owner="shared experts down",
        weight_keys=("layers.{layer}.ffn.shared_experts.w2.weight",),
        scale_keys=("layers.{layer}.ffn.shared_experts.w2.scale",),
        shard_dim=1,
        source_boundary="DSV4SharedExperts.forward down_proj before all-reduce",
        vllm_analogue="shared experts down quantized RowParallelLinear local GEMM",
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


def shard_tensor(tensor: torch.Tensor, *, dim: int | None, tp_size: int = TP_SIZE) -> torch.Tensor:
    if dim is None:
        return tensor.contiguous()
    return tensor.chunk(tp_size, dim=dim)[0].contiguous()


def load_dense_owner(
    model_path: Path,
    index: dict[str, str],
    layer: int,
    owner: DenseOwner,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = [
        shard_tensor(
            load_tensor(model_path, index, key.format(layer=layer), device),
            dim=owner.shard_dim,
        )
        for key in owner.weight_keys
    ]
    scales = [
        shard_tensor(
            load_tensor(model_path, index, key.format(layer=layer), device),
            dim=owner.shard_dim,
        )
        for key in owner.scale_keys
    ]
    return torch.cat(weights, dim=0).contiguous(), torch.cat(scales, dim=0).contiguous()


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape
    expanded = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expanded = expanded[:out_features, :in_features]
    return (weight.float() * expanded).to(torch.bfloat16).contiguous()


def fp8_activation_round(x: torch.Tensor, *, block_size: int = 128) -> torch.Tensor:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or x.numel() == 0 or x.shape[-1] % block_size != 0:
        return x
    dtype = x.dtype
    flat = x.contiguous().view(-1, x.shape[-1]).float()
    groups = flat.view(flat.shape[0], flat.shape[1] // block_size, block_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / FP8_MAX
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    y = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(fp8).float() * scale
    return y.reshape_as(flat).reshape_as(x).to(dtype)


def sync() -> None:
    torch.cuda.synchronize()


def wall_cuda_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - start) * 1000.0


def timed_ms(fn: TimedFn, *, warmup: int, iters: int) -> dict[str, float]:
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
        "p20_ms": sorted(samples)[max(0, int(0.2 * (len(samples) - 1)))],
        "p80_ms": sorted(samples)[min(len(samples) - 1, int(0.8 * (len(samples) - 1)))],
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def make_input(m: int, k: int, *, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=gen)


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


def prepare_mini_backend(weight: torch.Tensor, scale: torch.Tensor, owner: DenseOwner) -> PreparedBackend:
    from minisgl.kernel import dense_fp8_marlin

    prepared, prep_ms = wall_cuda_ms(
        lambda: dense_fp8_marlin.prepare_dense_fp8_marlin_weight(
            weight,
            scale,
            owner_label=owner.owner,
        )
    )
    return PreparedBackend(
        backend="mini_dense_fp8_marlin_w8a16_block",
        apply_fn=lambda x, prepared=prepared: dense_fp8_marlin.apply_dense_fp8_marlin_linear(
            x,
            prepared,
        ),
        prep_ms=prep_ms,
        original_weight_bytes=dense_fp8_marlin.tensor_bytes(weight),
        original_scale_bytes=dense_fp8_marlin.tensor_bytes(scale),
        prepared_weight_bytes=dense_fp8_marlin.tensor_bytes(prepared.weight),
        prepared_scale_bytes=dense_fp8_marlin.tensor_bytes(prepared.weight_scale),
        workspace_bytes=dense_fp8_marlin.tensor_bytes(prepared.workspace),
        persistent_bytes=prepared.persistent_bytes,
        notes="Mini-owned torch extension bridge, no vLLM runtime import.",
    )


def prepare_vllm_backend(weight: torch.Tensor, scale: torch.Tensor, owner: DenseOwner) -> PreparedBackend:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    def build() -> nn.Module:
        n, k = weight.shape
        layer = nn.Module().to(weight.device)
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.orig_dtype = torch.bfloat16
        layer.logical_widths = [n]
        layer.weight_block_size = [128, 128]
        processed_weight, processed_scale = process_fp8_weight_block_strategy(weight, scale)
        layer.weight = nn.Parameter(processed_weight, requires_grad=False)
        layer.weight_scale_inv = nn.Parameter(processed_scale, requires_grad=False)
        layer.input_scale = None
        prepare_fp8_layer_for_marlin(layer, size_k_first=False, input_dtype=None)
        return layer

    layer, prep_ms = wall_cuda_ms(build)
    n, k = weight.shape
    return PreparedBackend(
        backend="vllm_helper_fp8_marlin_w8a16_block",
        apply_fn=lambda x, layer=layer, n=n, k=k: apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale_inv,
            workspace=layer.workspace,
            size_n=n,
            size_k=k,
            input_dtype=None,
            bias=None,
        ),
        prep_ms=prep_ms,
        original_weight_bytes=tensor_bytes(weight),
        original_scale_bytes=tensor_bytes(scale),
        prepared_weight_bytes=tensor_bytes(layer.weight),
        prepared_scale_bytes=tensor_bytes(layer.weight_scale_inv),
        workspace_bytes=tensor_bytes(layer.workspace),
        persistent_bytes=(
            tensor_bytes(layer.weight) + tensor_bytes(layer.weight_scale_inv) + tensor_bytes(layer.workspace)
        ),
        notes="Offline vLLM helper path reference.",
    )


def build_backends(
    backend_mode: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
    owner: DenseOwner,
) -> tuple[list[PreparedBackend], list[dict[str, Any]]]:
    backends: list[PreparedBackend] = []
    errors: list[dict[str, Any]] = []
    builders: list[tuple[str, Callable[[], PreparedBackend]]] = []
    if backend_mode in {"mini", "both"}:
        builders.append(("mini_dense_fp8_marlin_w8a16_block", lambda: prepare_mini_backend(weight, scale, owner)))
    if backend_mode in {"vllm", "both"}:
        builders.append(("vllm_helper_fp8_marlin_w8a16_block", lambda: prepare_vllm_backend(weight, scale, owner)))

    for name, builder in builders:
        try:
            backends.append(builder())
        except Exception as exc:  # noqa: BLE001 - probe records bridge failures
            errors.append(
                {
                    "owner": owner.owner,
                    "backend": name,
                    "stage": "prepare",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
    return backends, errors


def bench_owner(
    *,
    model_path: Path,
    index: dict[str, str],
    layer: int,
    owner: DenseOwner,
    owner_index: int,
    backend_mode: str,
    tokens: list[int],
    warmup: int,
    iters: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    weight, scale = load_dense_owner(model_path, index, layer, owner, device)
    n, k = weight.shape
    cached_bf16, cached_prep_ms = wall_cuda_ms(lambda: dequant_fp8_block(weight, scale))
    backends, errors = build_backends(backend_mode, weight, scale, owner)

    prep_rows: list[dict[str, Any]] = [
        {
            "case": owner.case,
            "owner": owner.owner,
            "backend": "cached_bf16_f_linear",
            "prep_ms": cached_prep_ms,
            "original_weight_bytes": tensor_bytes(weight),
            "original_scale_bytes": tensor_bytes(scale),
            "prepared_weight_bytes": tensor_bytes(cached_bf16),
            "prepared_scale_bytes": 0,
            "workspace_bytes": 0,
            "persistent_bytes": tensor_bytes(cached_bf16),
            "original_needed_after_packing": False,
            "notes": "BF16 dequantized F.linear baseline.",
        }
    ]
    for backend in backends:
        prep_rows.append(
            {
                "case": owner.case,
                "owner": owner.owner,
                "backend": backend.backend,
                "prep_ms": backend.prep_ms,
                "original_weight_bytes": backend.original_weight_bytes,
                "original_scale_bytes": backend.original_scale_bytes,
                "prepared_weight_bytes": backend.prepared_weight_bytes,
                "prepared_scale_bytes": backend.prepared_scale_bytes,
                "workspace_bytes": backend.workspace_bytes,
                "persistent_bytes": backend.persistent_bytes,
                "original_needed_after_packing": False,
                "notes": backend.notes,
            }
        )

    rows: list[dict[str, Any]] = []
    for m in tokens:
        x = make_input(m, k, seed=7750 + owner_index * 100 + m, device=device)
        exact_fn = lambda x=x, w=cached_bf16: F.linear(x, w)
        promoted_fn = lambda x=x, w=cached_bf16: F.linear(fp8_activation_round(x), w)
        exact = exact_fn()
        promoted = promoted_fn()
        exact_timing = timed_ms(exact_fn, warmup=warmup, iters=iters)
        promoted_timing = timed_ms(promoted_fn, warmup=warmup, iters=iters)
        promoted_err = error_stats(promoted, exact)
        rows.append(
            {
                "case": owner.case,
                "owner": owner.owner,
                "tokens_m": m,
                "n": n,
                "k": k,
                "backend": "cached_bf16_f_linear_exact",
                "source_boundary": owner.source_boundary,
                "vllm_analogue": owner.vllm_analogue,
                "activation_policy": "BF16 activations; no runtime activation FP8 quantization",
                "quant_dequant_included": "weight dequant is one-time BF16 cache prep",
                "mean_ms": exact_timing["mean_ms"],
                "median_ms": exact_timing["median_ms"],
                "p20_ms": exact_timing["p20_ms"],
                "p80_ms": exact_timing["p80_ms"],
                "min_ms": exact_timing["min_ms"],
                "max_ms": exact_timing["max_ms"],
                "baseline_mean_ms": promoted_timing["mean_ms"],
                "exact_bf16_mean_ms": exact_timing["mean_ms"],
                "speedup_vs_promoted_cached_bf16": (
                    (promoted_timing["mean_ms"] - exact_timing["mean_ms"])
                    / promoted_timing["mean_ms"]
                ),
                "max_abs_err": 0.0,
                "mean_abs_err": 0.0,
                "p99_abs_err": 0.0,
                "cosine": 1.0,
            }
        )
        rows.append(
            {
                "case": owner.case,
                "owner": owner.owner,
                "tokens_m": m,
                "n": n,
                "k": k,
                "backend": "promoted_cached_bf16_activation_round_f_linear",
                "source_boundary": owner.source_boundary,
                "vllm_analogue": owner.vllm_analogue,
                "activation_policy": "Mini FP8 activation rounding before cached BF16 F.linear",
                "quant_dequant_included": "runtime activation rounding included; weight dequant is one-time BF16 cache prep",
                "mean_ms": promoted_timing["mean_ms"],
                "median_ms": promoted_timing["median_ms"],
                "p20_ms": promoted_timing["p20_ms"],
                "p80_ms": promoted_timing["p80_ms"],
                "min_ms": promoted_timing["min_ms"],
                "max_ms": promoted_timing["max_ms"],
                "baseline_mean_ms": promoted_timing["mean_ms"],
                "exact_bf16_mean_ms": exact_timing["mean_ms"],
                "speedup_vs_promoted_cached_bf16": 0.0,
                **promoted_err,
            }
        )
        for backend in backends:
            try:
                candidate = backend.apply_fn(x)
                sync()
                timing = timed_ms(lambda backend=backend, x=x: backend.apply_fn(x), warmup=warmup, iters=iters)
                err = error_stats(candidate, exact)
                promoted_err = error_stats(candidate, promoted)
                rows.append(
                    {
                        "case": owner.case,
                        "owner": owner.owner,
                        "tokens_m": m,
                        "n": n,
                        "k": k,
                        "backend": backend.backend,
                        "source_boundary": owner.source_boundary,
                        "vllm_analogue": owner.vllm_analogue,
                        "activation_policy": "BF16 activations; no runtime activation FP8 quantization",
                        "quant_dequant_included": "one-time pack/repack; steady GEMM only",
                        "mean_ms": timing["mean_ms"],
                        "median_ms": timing["median_ms"],
                        "p20_ms": timing["p20_ms"],
                        "p80_ms": timing["p80_ms"],
                        "min_ms": timing["min_ms"],
                        "max_ms": timing["max_ms"],
                        "baseline_mean_ms": promoted_timing["mean_ms"],
                        "exact_bf16_mean_ms": exact_timing["mean_ms"],
                        "speedup_vs_promoted_cached_bf16": (
                            (promoted_timing["mean_ms"] - timing["mean_ms"])
                            / promoted_timing["mean_ms"]
                        ),
                        "max_abs_err_vs_promoted": promoted_err["max_abs_err"],
                        "mean_abs_err_vs_promoted": promoted_err["mean_abs_err"],
                        "p99_abs_err_vs_promoted": promoted_err["p99_abs_err"],
                        "cosine_vs_promoted": promoted_err["cosine"],
                        **err,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "case": owner.case,
                        "owner": owner.owner,
                        "tokens_m": m,
                        "n": n,
                        "k": k,
                        "backend": backend.backend,
                        "stage": "apply_or_timing",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
    return rows, prep_rows, errors


def quality_ok(row: dict[str, Any]) -> bool:
    return float(row.get("cosine", 0.0)) >= 0.999 and float(row.get("mean_abs_err", math.inf)) <= 0.01


def compute_gates(results: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    gates: dict[str, Any] = {"errors": errors}
    for backend in sorted(
        {
            row["backend"]
            for row in results
            if row["backend"]
            not in {"cached_bf16_f_linear_exact", "promoted_cached_bf16_activation_round_f_linear"}
        }
    ):
        rows = [row for row in results if row["backend"] == backend]
        speedups = [float(row["speedup_vs_promoted_cached_bf16"]) for row in rows]
        gates[backend] = {
            "all_quality_ok": bool(rows) and all(quality_ok(row) for row in rows),
            "all_faster_than_promoted_cached_bf16": bool(rows)
            and all(speedup > 0.0 for speedup in speedups),
            "min_speedup_vs_promoted_cached_bf16": min(speedups) if speedups else None,
            "mean_speedup_vs_promoted_cached_bf16": statistics.fmean(speedups)
            if speedups
            else None,
            "covers_owners": sorted({row["owner"] for row in rows}),
            "covers_m_values": sorted({int(row["tokens_m"]) for row in rows}),
        }
    return gates


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Focused Dense FP8 Marlin Bridge Microbench",
        "",
        f"- backend mode: `{data['backend_mode']}`",
        f"- model: `{data['model_path']}`",
        f"- layer: `{data['layer']}`",
        f"- device: `{data['device']}` capability `{data['capability']}`",
        f"- torch: `{data['torch_version']}`",
        f"- warmup/iters: `{data['warmup']}` / `{data['iters']}`",
        "",
        "## Latency And Quality",
        "",
        "| Owner | M | Backend | Median ms | p20 ms | p80 ms | Speedup vs promoted | Max abs vs exact | Mean abs vs exact | p99 abs vs exact | Cosine vs exact |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["results"]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{:.6f}` | `{:.6f}` | `{:.6f}` | `{:.2%}` | `{:.6g}` | `{:.6g}` | `{:.6g}` | `{:.8f}` |".format(
                row["owner"],
                row["tokens_m"],
                row["backend"],
                float(row["median_ms"]),
                float(row["p20_ms"]),
                float(row["p80_ms"]),
                float(row["speedup_vs_promoted_cached_bf16"]),
                float(row.get("max_abs_err", 0.0)),
                float(row.get("mean_abs_err", 0.0)),
                float(row.get("p99_abs_err", 0.0)),
                float(row.get("cosine", 1.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Preparation / Memory",
            "",
            "| Owner | Backend | Prep ms | Original weight bytes | Original scale bytes | Prepared weight bytes | Prepared scale bytes | Workspace bytes | Persistent bytes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in data["prep"]:
        lines.append(
            "| `{}` | `{}` | `{:.3f}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["owner"],
                row["backend"],
                float(row["prep_ms"]),
                int(row["original_weight_bytes"]),
                int(row["original_scale_bytes"]),
                int(row["prepared_weight_bytes"]),
                int(row["prepared_scale_bytes"]),
                int(row["workspace_bytes"]),
                int(row["persistent_bytes"]),
            )
        )
    if data["errors"]:
        lines.extend(["", "## Errors", "", "```json", json.dumps(data["errors"], indent=2), "```"])
    lines.extend(["", "## Gates", "", "```json", json.dumps(data["gates"], indent=2), "```"])
    return "\n".join(lines) + "\n"


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    index = load_index(args.model_path)

    results: list[dict[str, Any]] = []
    prep: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for owner_index, owner in enumerate(DENSE_OWNERS):
        owner_results, owner_prep, owner_errors = bench_owner(
            model_path=args.model_path,
            index=index,
            layer=args.layer,
            owner=owner,
            owner_index=owner_index,
            backend_mode=args.backend,
            tokens=args.tokens,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        results.extend(owner_results)
        prep.extend(owner_prep)
        errors.extend(owner_errors)
        torch.cuda.empty_cache()

    data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend_mode": args.backend,
        "model_path": str(args.model_path),
        "layer": args.layer,
        "tokens": args.tokens,
        "tp_size_simulated": TP_SIZE,
        "rank_simulated": 0,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "results": results,
        "prep": prep,
        "errors": errors,
    }
    data["gates"] = compute_gates(results, errors)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/models/DeepSeek-V4-Flash"))
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=775)
    parser.add_argument("--backend", choices=["mini", "vllm", "both"], default="mini")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_mini_owned_dense_fp8_marlin_bridge/raw/focused_dense_fp8_marlin_bridge_microbench.json",
    )
    args = parser.parse_args()

    data = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(data))
    print(json.dumps(data["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[3]
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


class _EnvGuard:
    def __init__(self, updates: dict[str, str | None]):
        self._updates = updates
        self._previous: dict[str, str | None] = {}

    def __enter__(self):
        for name, value in self._updates.items():
            self._previous[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def __exit__(self, exc_type, exc, tb):
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_weight_map(model_path: Path) -> dict[str, str]:
    with open(model_path / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _load_config(model_path: Path) -> dict[str, Any]:
    with open(model_path / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_tensor(
    model_path: Path,
    weight_map: dict[str, str],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    with safe_open(model_path / weight_map[key], framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
    return tensor.to(device=device).contiguous()


def _load_wo_a(
    model_path: Path,
    weight_map: dict[str, str],
    *,
    layer: int,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = _load_tensor(model_path, weight_map, f"layers.{layer}.attn.wo_a.weight", device)
    scale = _load_tensor(model_path, weight_map, f"layers.{layer}.attn.wo_a.scale", device)
    if tp_size > 1:
        weight = weight.chunk(tp_size, dim=0)[tp_rank].contiguous()
        scale = scale.chunk(tp_size, dim=0)[tp_rank].contiguous()
    return weight, scale


def _build_bmm_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    groups: int,
    rank: int,
    d_per_group: int,
) -> torch.Tensor:
    dequant = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=torch.bfloat16)
    return dequant.view(groups, rank, d_per_group).transpose(1, 2).contiguous()


def _cached_bmm_projection(o: torch.Tensor, bmm_weight: torch.Tensor) -> torch.Tensor:
    tokens, groups, _ = o.shape
    out = torch.bmm(o.transpose(0, 1).contiguous(), bmm_weight)
    return out.transpose(0, 1).reshape(tokens, groups * bmm_weight.shape[-1])


def _fallback_projection(
    o: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    groups: int,
    rank: int,
) -> torch.Tensor:
    return dsv4_kernel.wo_a_grouped_projection_fallback(
        o,
        weight,
        scale,
        num_local_groups=groups,
        o_lora_rank=rank,
    )


def _error_row(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1.0e-6)
    rel = diff / denom
    return {
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "max_rel_error": float(rel.max().item()),
        "mean_rel_error": float(rel.mean().item()),
        "allclose_5e_2": bool(torch.allclose(actual, expected, atol=5e-2, rtol=5e-2)),
        "allclose_1e_1": bool(torch.allclose(actual, expected, atol=1e-1, rtol=1e-1)),
    }


def _bench_m(
    *,
    weight: torch.Tensor,
    scale: torch.Tensor,
    m: int,
    groups: int,
    rank: int,
    d_per_group: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    o = torch.randn(m, groups, d_per_group, device=weight.device, dtype=torch.bfloat16)

    with _EnvGuard({"MINISGL_DSV4_SM80_WO_A_BF16": None}):
        expected = _fallback_projection(o, weight, scale, groups=groups, rank=rank)
        fallback_total_ms = _bench_cuda(
            lambda: _fallback_projection(o, weight, scale, groups=groups, rank=rank),
            warmup=warmup,
            iters=iters,
        )

    bmm_weight = _build_bmm_weight(
        weight,
        scale,
        groups=groups,
        rank=rank,
        d_per_group=d_per_group,
    )
    actual = _cached_bmm_projection(o, bmm_weight)
    cache_build_ms = _bench_cuda(
        lambda: _build_bmm_weight(
            weight,
            scale,
            groups=groups,
            rank=rank,
            d_per_group=d_per_group,
        ),
        warmup=max(1, warmup // 2),
        iters=max(3, min(iters, 10)),
    )
    cached_total_ms = _bench_cuda(
        lambda: _cached_bmm_projection(o, bmm_weight),
        warmup=warmup,
        iters=iters,
    )
    x_bmm = o.transpose(0, 1).contiguous()
    cached_bmm_only_ms = _bench_cuda(
        lambda: torch.bmm(x_bmm, bmm_weight),
        warmup=warmup,
        iters=iters,
    )
    cached_einsum_weight = bmm_weight.transpose(1, 2).contiguous()
    cached_einsum_ms = _bench_cuda(
        lambda: torch.einsum("tgd,grd->tgr", o, cached_einsum_weight).reshape(m, -1),
        warmup=warmup,
        iters=iters,
    )
    dequant_ms = _bench_cuda(
        lambda: dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=o.dtype),
        warmup=max(1, warmup // 2),
        iters=max(3, min(iters, 10)),
    )

    return {
        "m": m,
        "groups": groups,
        "rank": rank,
        "d_per_group": d_per_group,
        "input_shape": list(o.shape),
        "weight_shape": list(weight.shape),
        "scale_shape": list(scale.shape),
        "bmm_weight_shape": list(bmm_weight.shape),
        "cached_weight_bytes": int(bmm_weight.numel() * bmm_weight.element_size()),
        "fallback_total_ms": fallback_total_ms,
        "fallback_dequant_ms": dequant_ms,
        "cache_build_ms": cache_build_ms,
        "cached_bf16_bmm_total_ms": cached_total_ms,
        "cached_bmm_only_ms": cached_bmm_only_ms,
        "cached_bf16_einsum_no_dequant_ms": cached_einsum_ms,
        "cached_vs_fallback_speedup": (
            fallback_total_ms / cached_total_ms if cached_total_ms > 0 else None
        ),
        "cached_vs_fallback_improvement_pct": (
            (1.0 - cached_total_ms / fallback_total_ms) * 100.0
            if fallback_total_ms > 0
            else None
        ),
        "cached_vs_fallback_error": _error_row(actual, expected),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# wo_a BF16 BMM Cache Microbench")
    lines.append("")
    lines.append(f"- Model path: `{report['model_path']}`")
    lines.append(f"- Layer: `{report['layer']}`")
    lines.append(f"- CUDA device: `{report['cuda_device']}`")
    lines.append(f"- TP shard: rank `{report['tp_rank']}` / size `{report['tp_size']}`")
    lines.append(f"- M values: `{report['m_values']}`")
    lines.append(f"- Warmup/iters: `{report['warmup']}` / `{report['iters']}`")
    lines.append(
        "- Scope: current `wo_a_grouped_projection_fallback` vs replay-time cached BF16 `torch.bmm`; cache build is reported separately."
    )
    lines.append("")
    lines.append(
        "| M | Groups | K/group | Rank | fallback total ms | cache build ms | cached BMM total ms | BMM only ms | speedup | improvement | max abs err | max rel err | ok 5e-2 |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for row in report["results"]:
        err = row["cached_vs_fallback_error"]
        lines.append(
            "| {m} | {groups} | {k} | {rank} | `{fallback:.4f}` | `{build:.4f}` | `{cached:.4f}` | `{bmm:.4f}` | `{speedup:.2f}x` | `{impr:.2f}%` | `{max_abs:.6f}` | `{max_rel:.6f}` | `{ok}` |".format(
                m=row["m"],
                groups=row["groups"],
                k=row["d_per_group"],
                rank=row["rank"],
                fallback=row["fallback_total_ms"],
                build=row["cache_build_ms"],
                cached=row["cached_bf16_bmm_total_ms"],
                bmm=row["cached_bmm_only_ms"],
                speedup=row["cached_vs_fallback_speedup"] or 0.0,
                impr=row["cached_vs_fallback_improvement_pct"] or 0.0,
                max_abs=err["max_abs_error"],
                max_rel=err["max_rel_error"],
                ok=err["allclose_5e_2"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/models/DeepSeek-V4-Flash"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--m-values", default="1,4,8,16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=762)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_wo_a_attention_boundary_parity/raw/wo_a_bf16_bmm_cache_microbench.json"
        ),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_wo_a_attention_boundary_parity/summaries/wo_a_bf16_bmm_cache_microbench.md"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.tp_size <= 0:
        raise SystemExit("--tp-size must be positive")
    if not 0 <= args.tp_rank < args.tp_size:
        raise SystemExit("--tp-rank must be in [0, tp_size)")

    device = torch.device("cuda")
    config = _load_config(args.model_path)
    rank = int(config["o_lora_rank"])
    weight_map = _load_weight_map(args.model_path)
    weight, scale = _load_wo_a(
        args.model_path,
        weight_map,
        layer=args.layer,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        device=device,
    )
    if weight.shape[0] % rank != 0:
        raise SystemExit(f"local wo_a output rows {weight.shape[0]} are not divisible by rank {rank}")
    groups = weight.shape[0] // rank
    d_per_group = weight.shape[1]
    m_values = _parse_csv_ints(args.m_values)

    started = time.time()
    results = []
    for m in m_values:
        row = _bench_m(
            weight=weight,
            scale=scale,
            m=m,
            groups=groups,
            rank=rank,
            d_per_group=d_per_group,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed + m,
        )
        results.append(row)
        print(
            f"M={m}: fallback={row['fallback_total_ms']:.4f} ms, "
            f"cached_bmm_total={row['cached_bf16_bmm_total_ms']:.4f} ms, "
            f"speedup={row['cached_vs_fallback_speedup']:.2f}x, "
            f"ok={row['cached_vs_fallback_error']['allclose_5e_2']}"
        )

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "model_path": str(args.model_path),
        "layer": args.layer,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "toggle": dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE,
        "m_values": m_values,
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

#!/usr/bin/env python3
"""TARGET 07.52 mini vLLM-aligned FP8 indexer backend microbench."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.kernel import deepseek_v4 as dsv4_kernel


class _FakePagedFP8Cache:
    def __init__(self, cache: torch.Tensor, page_size: int) -> None:
        self._cache = cache
        self.indexer_fp8_page_size = page_size

    def has_indexer_fp8_cache(self) -> bool:
        return True

    def has_indexer_fp8_paged_cache(self) -> bool:
        return True

    def indexer_fp8_paged_cache(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self._cache


class _FakeLegacyFP8Cache:
    def __init__(self, values: torch.Tensor, scales: torch.Tensor) -> None:
        self._values = values
        self._scales = scales

    def has_indexer_fp8_cache(self) -> bool:
        return True

    def has_indexer_fp8_paged_cache(self) -> bool:
        return False

    def indexer_fp8_cache(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        del layer_id
        return self._values, self._scales


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]


def _time_cuda(fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": _p90(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _make_inputs(
    batch: int,
    history: int,
    *,
    num_heads: int,
    head_dim: int,
    page_size: int,
) -> tuple[torch.Tensor, ...]:
    q = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    blocks_per_req = (history + page_size - 1) // page_size
    cache = torch.randn(
        batch * blocks_per_req * page_size,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn(batch, num_heads, device="cuda", dtype=torch.float32)
    if batch == 1:
        seq_lens = torch.tensor([history], device="cuda", dtype=torch.int32)
    else:
        seq_lens = torch.linspace(
            history // 2,
            history,
            batch,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.int32)
    page_table = torch.arange(
        batch * blocks_per_req,
        device="cuda",
        dtype=torch.int32,
    ).reshape(batch, blocks_per_req)
    positions = seq_lens.to(torch.int64) - 1
    return q, weights, cache, seq_lens, page_table, positions


def _finite_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    finite = torch.isfinite(reference) & torch.isfinite(actual)
    if not bool(finite.any()):
        return {"max_abs": 0.0, "mean_abs": 0.0}
    diff = (reference[finite].float() - actual[finite].float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def _topk_overlap(
    reference: dsv4_kernel.DSV4IndexerSelectOutput,
    actual: dsv4_kernel.DSV4IndexerSelectOutput,
) -> dict[str, float]:
    ref = reference.topk.raw_indices.detach().cpu()
    act = actual.topk.raw_indices.detach().cpu()
    rows = min(ref.shape[0], act.shape[0])
    overlaps: list[float] = []
    for row in range(rows):
        ref_set = {int(x) for x in ref[row].tolist() if int(x) >= 0}
        act_set = {int(x) for x in act[row].tolist() if int(x) >= 0}
        overlaps.append(len(ref_set & act_set) / max(len(ref_set), 1))
    if not overlaps:
        return {"mean": 1.0, "min": 1.0}
    return {"mean": float(statistics.fmean(overlaps)), "min": float(min(overlaps))}


def _mean_ms(timing: dict[str, Any]) -> float:
    return float(timing["mean_ms"])


def run_case(
    *,
    batch: int,
    history: int,
    num_heads: int,
    head_dim: int,
    topk_width: int,
    page_size: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    q, weights, cache, seq_lens, page_table, positions = _make_inputs(
        batch,
        history,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    weight_scale = (head_dim**-0.5) * (num_heads**-0.5)
    bf16_weights = weights * weight_scale

    os.environ["MINISGL_DSV4_SM80_INDEXER_BF16"] = "1"
    os.environ["MINISGL_DSV4_SM80_TOPK"] = "1"
    os.environ[dsv4_kernel.DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE] = "1"
    os.environ[dsv4_kernel.DSV4_SM80_INDEXER_FP8_CACHE_TOGGLE] = "1"

    bf16_backend: list[str] = []
    bf16_logits = dsv4_kernel.indexer_bf16_logits_fallback(
        q,
        cache,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=bf16_weights,
        _backend=bf16_backend,
    )
    bf16_select = dsv4_kernel.indexer_select_bf16_fallback(
        q,
        bf16_weights,
        cache,
        seq_lens,
        page_table,
        page_size=page_size,
        width=topk_width,
        ratio=4,
    )
    query = dsv4_kernel.indexer_q_rope_fp8_fallback(
        q,
        weights,
        positions,
        rotary_dim=0,
        base=10000.0,
        softmax_scale=head_dim**-0.5,
        head_scale=num_heads**-0.5,
    )
    cache_values, cache_scales = dsv4_kernel.quantize_indexer_fp8_cache_ref(cache)
    packed_cache = dsv4_kernel.pack_indexer_fp8_paged_cache_ref(
        cache_values,
        cache_scales,
        page_size=page_size,
    )

    legacy_backend: list[str] = []
    legacy_logits = dsv4_kernel.indexer_fp8_logits_fallback(
        query.q_values,
        cache_values,
        cache_scales,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=query.weights,
        _backend=legacy_backend,
    )
    legacy_select = dsv4_kernel.indexer_select_fp8_fallback(
        query.q_values,
        query.weights,
        cache_values,
        cache_scales,
        seq_lens,
        page_table,
        page_size=page_size,
        width=topk_width,
        ratio=4,
    )
    paged_backend: list[str] = []
    paged_logits = dsv4_kernel.indexer_fp8_paged_logits_fallback(
        query.q_values,
        packed_cache,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=query.weights,
        _backend=paged_backend,
    )
    paged_select = dsv4_kernel.indexer_select_fp8_paged_fallback(
        query.q_values,
        query.weights,
        packed_cache,
        seq_lens,
        page_table,
        page_size=page_size,
        width=topk_width,
        ratio=4,
    )

    slot_mapping = torch.arange(cache.shape[0], device=cache.device, dtype=torch.int64)
    legacy_store_values = torch.empty_like(cache_values)
    legacy_store_scales = torch.empty_like(cache_scales)
    paged_store_cache = torch.empty_like(packed_cache)
    legacy_fake = _FakeLegacyFP8Cache(legacy_store_values, legacy_store_scales)
    paged_fake = _FakePagedFP8Cache(paged_store_cache, page_size)

    timings = {
        "bf16_logits_ms": _time_cuda(
            lambda: dsv4_kernel.indexer_bf16_logits_fallback(
                q,
                cache,
                seq_lens,
                page_table,
                page_size=page_size,
                weights=bf16_weights,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "bf16_select_ms": _time_cuda(
            lambda: dsv4_kernel.indexer_select_bf16_fallback(
                q,
                bf16_weights,
                cache,
                seq_lens,
                page_table,
                page_size=page_size,
                width=topk_width,
                ratio=4,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "fp8_q_quant_fold_ms": _time_cuda(
            lambda: dsv4_kernel.indexer_q_rope_fp8_fallback(
                q,
                weights,
                positions,
                rotary_dim=0,
                base=10000.0,
                softmax_scale=head_dim**-0.5,
                head_scale=num_heads**-0.5,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "legacy_fp8_store_ms": _time_cuda(
            lambda: dsv4_kernel.store_indexer_fp8_cache_fallback(
                legacy_fake,
                0,
                cache,
                slot_mapping,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "paged_fp8_store_ms": _time_cuda(
            lambda: dsv4_kernel.store_indexer_fp8_cache_fallback(
                paged_fake,
                0,
                cache,
                slot_mapping,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "legacy_fp8_logits_ms": _time_cuda(
            lambda: dsv4_kernel.indexer_fp8_logits_fallback(
                query.q_values,
                cache_values,
                cache_scales,
                seq_lens,
                page_table,
                page_size=page_size,
                weights=query.weights,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "legacy_fp8_select_ms": _time_cuda(
            lambda: dsv4_kernel.indexer_select_fp8_fallback(
                query.q_values,
                query.weights,
                cache_values,
                cache_scales,
                seq_lens,
                page_table,
                page_size=page_size,
                width=topk_width,
                ratio=4,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "paged_fp8_logits_ms": _time_cuda(
            lambda: dsv4_kernel.indexer_fp8_paged_logits_fallback(
                query.q_values,
                packed_cache,
                seq_lens,
                page_table,
                page_size=page_size,
                weights=query.weights,
            ),
            warmup=warmup,
            iters=iters,
        ),
        "paged_fp8_select_ms": _time_cuda(
            lambda: dsv4_kernel.indexer_select_fp8_paged_fallback(
                query.q_values,
                query.weights,
                packed_cache,
                seq_lens,
                page_table,
                page_size=page_size,
                width=topk_width,
                ratio=4,
            ),
            warmup=warmup,
            iters=iters,
        ),
    }

    cache_dequant = dsv4_kernel.dequantize_indexer_fp8_paged_cache_ref(
        packed_cache,
        page_size=page_size,
        dim=head_dim,
        slots=cache.shape[0],
        out_dtype=torch.float32,
    )
    return {
        "case": {"batch": batch, "history": history},
        "shape": {
            "num_heads": num_heads,
            "head_dim": head_dim,
            "topk_width": topk_width,
            "indexer_page_size": page_size,
            "blocks_per_request": page_table.shape[1],
            "seq_lens": [int(x) for x in seq_lens.detach().cpu().tolist()],
            "cache_slots": int(cache.shape[0]),
            "fp8_paged_cache_bytes": int(packed_cache.numel()),
        },
        "backends": {
            "bf16_logits": bf16_backend[0] if bf16_backend else bf16_select.backend,
            "legacy_fp8_logits": legacy_backend[0] if legacy_backend else legacy_select.backend,
            "paged_fp8_logits": paged_backend[0] if paged_backend else paged_select.backend,
        },
        "timings": timings,
        "speedups": {
            "paged_logits_vs_bf16": _mean_ms(timings["bf16_logits_ms"])
            / _mean_ms(timings["paged_fp8_logits_ms"]),
            "paged_select_vs_bf16": _mean_ms(timings["bf16_select_ms"])
            / _mean_ms(timings["paged_fp8_select_ms"]),
            "paged_logits_vs_legacy_fp8": _mean_ms(timings["legacy_fp8_logits_ms"])
            / _mean_ms(timings["paged_fp8_logits_ms"]),
            "paged_select_vs_legacy_fp8": _mean_ms(timings["legacy_fp8_select_ms"])
            / _mean_ms(timings["paged_fp8_select_ms"]),
        },
        "quality": {
            "paged_logits_vs_bf16": _finite_error(bf16_logits, paged_logits),
            "legacy_logits_vs_bf16": _finite_error(bf16_logits, legacy_logits),
            "paged_topk_overlap_vs_bf16": _topk_overlap(bf16_select, paged_select),
            "legacy_topk_overlap_vs_bf16": _topk_overlap(bf16_select, legacy_select),
            "paged_cache_dequant_vs_bf16": _finite_error(cache.float(), cache_dequant),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones"
        / "target07_vllm_fp8_indexer_backend_port"
        / "raw"
        / "mini_vllm_fp8_indexer_backend_microbench.json",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT
        / "performance_milestones"
        / "target07_vllm_fp8_indexer_backend_port"
        / "summaries"
        / "mini_vllm_fp8_indexer_backend_microbench_summary.json",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--model-page-size", type=int, default=256)
    parser.add_argument("--topk-width", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if getattr(torch, "float8_e4m3fn", None) is None:
        raise SystemExit("torch.float8_e4m3fn is required")
    if args.model_page_size % 4 != 0:
        raise SystemExit("--model-page-size must be divisible by 4 for the C4 indexer")

    indexer_page_size = args.model_page_size // 4
    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 5)
    iters = args.iters if args.iters is not None else (3 if args.quick else 20)
    cases = [
        {"batch": 1, "history": 1024, "num_heads": 64, "head_dim": 128},
        {"batch": 4, "history": 2048, "num_heads": 64, "head_dim": 128},
        {"batch": 16, "history": 4096, "num_heads": 64, "head_dim": 128},
    ]
    if args.quick:
        cases = cases[:2]

    torch.manual_seed(20260752)
    results = [
        run_case(
            **case,
            topk_width=args.topk_width,
            page_size=indexer_page_size,
            warmup=warmup,
            iters=iters,
        )
        for case in cases
    ]
    output = {
        "suite": "target07_52_mini_vllm_fp8_indexer_backend_microbench",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "warmup": warmup,
        "iters": iters,
        "model_page_size": args.model_page_size,
        "indexer_page_size": indexer_page_size,
        "cases": results,
    }
    summary = {
        key: output[key]
        for key in (
            "suite",
            "device",
            "capability",
            "torch",
            "warmup",
            "iters",
            "model_page_size",
            "indexer_page_size",
        )
    }
    summary["cases"] = [
        {
            "case": case["case"],
            "shape": case["shape"],
            "backends": case["backends"],
            "mean_ms": {name: timing["mean_ms"] for name, timing in case["timings"].items()},
            "speedups": case["speedups"],
            "quality": case["quality"],
        }
        for case in results
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

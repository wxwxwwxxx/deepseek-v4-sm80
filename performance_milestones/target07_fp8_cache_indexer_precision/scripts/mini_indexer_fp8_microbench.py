from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.kernel import deepseek_v4 as dsv4_kernel


def _time_cuda(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(iters, 1))


def _make_inputs(
    batch: int,
    max_seq_len: int,
    *,
    num_heads: int,
    head_dim: int,
    page_size: int,
) -> tuple[torch.Tensor, ...]:
    q = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    num_pages = (max_seq_len + page_size - 1) // page_size
    cache = torch.randn(
        batch * num_pages * page_size,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn(batch, num_heads, device="cuda", dtype=torch.float32)
    if batch == 1:
        seq_lens = torch.tensor([max_seq_len], device="cuda", dtype=torch.int32)
    else:
        seq_lens = torch.linspace(
            max_seq_len // 2,
            max_seq_len,
            batch,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.int32)
    page_table = torch.arange(
        batch * num_pages,
        device="cuda",
        dtype=torch.int32,
    ).reshape(batch, num_pages)
    positions = torch.arange(batch, device="cuda", dtype=torch.int64)
    return q, weights, cache, seq_lens, page_table, positions


def _valid_logit_diff(
    expected: torch.Tensor,
    actual: torch.Tensor,
    seq_lens: torch.Tensor,
) -> dict[str, float]:
    rows = min(expected.shape[0], actual.shape[0], seq_lens.numel())
    diffs: list[torch.Tensor] = []
    for row in range(rows):
        length = min(int(seq_lens[row].item()), expected.shape[1], actual.shape[1])
        if length > 0:
            diffs.append((expected[row, :length] - actual[row, :length]).abs())
    if not diffs:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    diff = torch.cat(diffs)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _topk_overlap(
    expected: dsv4_kernel.DSV4IndexerSelectOutput,
    actual: dsv4_kernel.DSV4IndexerSelectOutput,
) -> dict[str, float]:
    exp = expected.topk.raw_indices.detach().cpu()
    act = actual.topk.raw_indices.detach().cpu()
    rows = min(exp.shape[0], act.shape[0])
    if rows == 0:
        return {"mean": 1.0, "min": 1.0}
    overlaps = []
    for row in range(rows):
        exp_set = {int(x) for x in exp[row].tolist() if int(x) >= 0}
        act_set = {int(x) for x in act[row].tolist() if int(x) >= 0}
        denom = max(len(exp_set), 1)
        overlaps.append(len(exp_set & act_set) / denom)
    return {"mean": float(sum(overlaps) / len(overlaps)), "min": float(min(overlaps))}


def run_case(
    *,
    batch: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    width: int,
    page_size: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    q, weights, cache, seq_lens, page_table, positions = _make_inputs(
        batch,
        max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    weight_scale = (head_dim**-0.5) * (num_heads**-0.5)
    bf16_weights = weights * weight_scale

    os.environ["MINISGL_DSV4_SM80_INDEXER_BF16"] = "1"
    os.environ["MINISGL_DSV4_SM80_TOPK"] = "1"
    os.environ.pop(dsv4_kernel.DSV4_SM80_INDEXER_FP8_CACHE_TOGGLE, None)
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
        width=width,
        ratio=4,
    )
    bf16_logits_ms = _time_cuda(
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
    )
    bf16_select_ms = _time_cuda(
        lambda: dsv4_kernel.indexer_select_bf16_fallback(
            q,
            bf16_weights,
            cache,
            seq_lens,
            page_table,
            page_size=page_size,
            width=width,
            ratio=4,
        ),
        warmup=warmup,
        iters=iters,
    )

    os.environ[dsv4_kernel.DSV4_SM80_INDEXER_FP8_CACHE_TOGGLE] = "1"
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
    cache_dequant = dsv4_kernel.dequantize_indexer_fp8_cache_ref(
        cache_values,
        cache_scales,
        out_dtype=torch.float32,
    )
    cache_diff = (cache.float() - cache_dequant).abs()
    fp8_backend: list[str] = []
    fp8_logits = dsv4_kernel.indexer_fp8_logits_fallback(
        query.q_values,
        cache_values,
        cache_scales,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=query.weights,
        _backend=fp8_backend,
    )
    fp8_select = dsv4_kernel.indexer_select_fp8_fallback(
        query.q_values,
        query.weights,
        cache_values,
        cache_scales,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=4,
    )
    fp8_query_ms = _time_cuda(
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
    )
    fp8_cache_quant_ms = _time_cuda(
        lambda: dsv4_kernel.quantize_indexer_fp8_cache_ref(cache),
        warmup=warmup,
        iters=iters,
    )
    fp8_logits_ms = _time_cuda(
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
    )
    fp8_select_ms = _time_cuda(
        lambda: dsv4_kernel.indexer_select_fp8_fallback(
            query.q_values,
            query.weights,
            cache_values,
            cache_scales,
            seq_lens,
            page_table,
            page_size=page_size,
            width=width,
            ratio=4,
        ),
        warmup=warmup,
        iters=iters,
    )

    os.environ.pop("MINISGL_DSV4_SM80_INDEXER_BF16", None)
    os.environ.pop("MINISGL_DSV4_SM80_TOPK", None)
    os.environ.pop(dsv4_kernel.DSV4_SM80_INDEXER_FP8_CACHE_TOGGLE, None)

    cache_slots = cache.shape[0]
    return {
        "batch": batch,
        "max_seq_len": max_seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "width": width,
        "cache_slots": cache_slots,
        "bf16_cache_bytes": int(cache_slots * head_dim * torch.bfloat16.itemsize),
        "fp8_indexer_cache_bytes": int(cache_slots * (head_dim + 4)),
        "bf16_backend": bf16_backend[0] if bf16_backend else bf16_select.backend,
        "fp8_backend": fp8_backend[0] if fp8_backend else fp8_select.backend,
        "bf16_logits_ms": bf16_logits_ms,
        "bf16_select_ms": bf16_select_ms,
        "fp8_query_quant_ms": fp8_query_ms,
        "fp8_cache_quant_full_ms": fp8_cache_quant_ms,
        "fp8_logits_ms": fp8_logits_ms,
        "fp8_select_ms": fp8_select_ms,
        "logits_speedup_bf16_over_fp8": (
            bf16_logits_ms / fp8_logits_ms if fp8_logits_ms > 0 else None
        ),
        "select_speedup_bf16_over_fp8": (
            bf16_select_ms / fp8_select_ms if fp8_select_ms > 0 else None
        ),
        "fp8_vs_bf16_logits": _valid_logit_diff(bf16_logits, fp8_logits, seq_lens),
        "fp8_vs_bf16_topk_overlap": _topk_overlap(bf16_select, fp8_select),
        "cache_dequant_error": {
            "max_abs": float(cache_diff.max().item()),
            "mean_abs": float(cache_diff.mean().item()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones"
        / "target07_fp8_cache_indexer_precision"
        / "raw"
        / "mini_indexer_fp8_microbench.json",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if getattr(torch, "float8_e4m3fn", None) is None:
        raise SystemExit("torch.float8_e4m3fn is required for FP8 indexer cache benchmarking")

    warmup = args.warmup if args.warmup is not None else (1 if args.quick else 5)
    iters = args.iters if args.iters is not None else (3 if args.quick else 20)
    torch.manual_seed(20260701)
    cases = [
        {"batch": 1, "max_seq_len": 1024, "num_heads": 64, "head_dim": 128},
        {"batch": 4, "max_seq_len": 2048, "num_heads": 64, "head_dim": 128},
        {"batch": 16, "max_seq_len": 4096, "num_heads": 64, "head_dim": 128},
    ]
    if args.quick:
        cases = cases[:2]

    results = [
        run_case(
            **case,
            width=args.width,
            page_size=args.page_size,
            warmup=warmup,
            iters=iters,
        )
        for case in cases
    ]
    result = {
        "suite": "target07_50_mini_indexer_fp8_microbench",
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "warmup": warmup,
        "iters": iters,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

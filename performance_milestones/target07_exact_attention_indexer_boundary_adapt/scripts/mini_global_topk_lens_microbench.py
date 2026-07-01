#!/usr/bin/env python3
"""TARGET 07.394 mini exact bf16 global-topk/lens boundary probe.

This keeps the 07.393 synthetic decode shape and compares the legacy mini
topk boundary against the opt-in exact-bf16 global topk/lens consolidation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

DEFAULT_TOGGLES = {
    "MINISGL_DSV4_SM80_V0_BF16": "1",
    "MINISGL_DSV4_SM80_SPARSE_ATTN_BF16": "1",
    "MINISGL_DSV4_SM80_INDEXER_BF16": "1",
    "MINISGL_DSV4_SM80_TOPK": "1",
}

for key, value in DEFAULT_TOGGLES.items():
    os.environ.setdefault(key, value)

import torch

from minisgl.kernel.deepseek_v4 import (  # noqa: E402
    DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE,
    dsv4_sparse_attention_two_source_bf16,
    indexer_select_bf16_fallback,
    topk_transform_512_full_fallback,
)


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return ordered[idx]


def cuda_event_bench(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {
        "name": name,
        "status": "pass",
        "warmup": warmup,
        "repeat": repeat,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": p90(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


@contextlib.contextmanager
def global_topk_lens_enabled(enabled: bool) -> Iterator[None]:
    old = os.environ.get(DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE)
    if enabled:
        os.environ[DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE] = "1"
    else:
        os.environ.pop(DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE, None)
        else:
            os.environ[DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE] = old


def rounded_pages(length: int, page_size: int) -> int:
    return max(1, math.ceil(max(length, 1) / page_size))


def make_page_table(rows: int, pages: int, *, device: torch.device) -> torch.Tensor:
    base = torch.arange(pages, dtype=torch.int32, device=device)
    return base.repeat(rows, 1).contiguous()


def timed_pair(
    name: str,
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    results = {}
    backends = {}
    for label, enabled in (("legacy", False), ("global_topk_lens", True)):
        with global_topk_lens_enabled(enabled):
            bench = cuda_event_bench(f"{name}_{label}", fn, warmup=warmup, repeat=repeat)
            out = fn()
            if hasattr(out, "backend"):
                backends[label] = out.backend
            elif hasattr(out, "topk") and hasattr(out.topk, "backend"):
                backends[label] = out.topk.backend
            results[label] = bench
    before = results["legacy"]["mean_ms"]
    after = results["global_topk_lens"]["mean_ms"]
    improvement = (before - after) / before if before > 0 else float("nan")
    return {
        "name": name,
        "status": "pass",
        "legacy": results["legacy"],
        "global_topk_lens": results["global_topk_lens"],
        "backend": backends,
        "improvement_fraction": improvement,
        "speedup": before / after if after > 0 else float("nan"),
    }


def topk_full_transform(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    c4_page_size = max(page_size // 4, 1)
    c4_len = max(history // 4, 1)
    c4_pages = rounded_pages(c4_len, c4_page_size)
    scores = torch.randn(tokens, c4_len, device=device, dtype=torch.float32)
    seq_lens = torch.full((tokens,), c4_len, device=device, dtype=torch.int32)
    page_table = make_page_table(tokens, c4_pages, device=device)

    def fn() -> Any:
        return topk_transform_512_full_fallback(
            scores,
            seq_lens,
            page_table,
            page_size=c4_page_size,
            width=512,
            ratio=4,
        )

    result = timed_pair("mini_topk_full_transform", fn, warmup=warmup, repeat=repeat)
    result.update(
        {
            "subgraph": "global_topk_lens_consolidation",
            "tokens": tokens,
            "history": history,
            "shape": {
                "scores": [tokens, c4_len],
                "page_table": [tokens, c4_pages],
                "c4_page_size": c4_page_size,
                "width": 512,
            },
            "boundary_note": "topk raw selection plus page/full-index mapping and topk_lens",
        }
    )
    return result


def indexer_select(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    c4_page_size = max(page_size // 4, 1)
    c4_len = max(history // 4, 1)
    c4_pages = rounded_pages(c4_len, c4_page_size)
    q = torch.randn(tokens, 64, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, 64, device=device, dtype=torch.float32)
    cache = torch.randn(c4_pages * c4_page_size, 128, device=device, dtype=torch.bfloat16)
    seq_lens = torch.full((tokens,), c4_len, device=device, dtype=torch.int32)
    page_table = make_page_table(tokens, c4_pages, device=device)

    def fn() -> Any:
        return indexer_select_bf16_fallback(
            q,
            weights,
            cache,
            seq_lens,
            page_table,
            page_size=c4_page_size,
            width=512,
            ratio=4,
        )

    result = timed_pair("mini_indexer_select_bf16", fn, warmup=warmup, repeat=repeat)
    result.update(
        {
            "subgraph": "indexer_logits_select_topk",
            "tokens": tokens,
            "history": history,
            "shape": {
                "q": [tokens, 64, 128],
                "indexer_cache": list(cache.shape),
                "page_table": [tokens, c4_pages],
                "c4_page_size": c4_page_size,
            },
            "boundary_note": "full bf16 indexer logits plus topk/global-index mapping",
        }
    )
    return result


def combined_decode_boundary(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    c4_page_size = max(page_size // 4, 1)
    c4_len = max(history // 4, 1)
    c4_pages = rounded_pages(c4_len, c4_page_size)
    q_attn = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16)
    q_indexer = torch.randn(tokens, 64, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, 64, device=device, dtype=torch.float32)
    idx_cache = torch.randn(c4_pages * c4_page_size, 128, device=device, dtype=torch.bfloat16)
    c4_cache = torch.randn(c4_pages * c4_page_size, 512, device=device, dtype=torch.bfloat16)
    swa_cache = torch.randn(history + tokens + 256, 512, device=device, dtype=torch.bfloat16)
    page_table = make_page_table(tokens, c4_pages, device=device)
    seq_lens = torch.full((tokens,), c4_len, device=device, dtype=torch.int32)
    swa_indices = torch.randint(0, history, (tokens, 128), device=device, dtype=torch.int32)
    swa_lengths = torch.full((tokens,), 128, device=device, dtype=torch.int32)
    attn_sink = torch.zeros(8, device=device, dtype=torch.float32)

    def fn() -> torch.Tensor:
        selected = indexer_select_bf16_fallback(
            q_indexer,
            weights,
            idx_cache,
            seq_lens,
            page_table,
            page_size=c4_page_size,
            width=512,
            ratio=4,
        )
        c4_indices = selected.topk.full_indices.to(torch.int32)
        if selected.topk.topk_lens is not None:
            c4_lengths = selected.topk.topk_lens
        else:
            c4_lengths = (c4_indices >= 0).sum(dim=-1).to(torch.int32)
        out = dsv4_sparse_attention_two_source_bf16(
            q_attn,
            swa_cache,
            swa_indices,
            swa_lengths,
            compressed_cache=c4_cache,
            compressed_indices=c4_indices,
            compressed_lengths=c4_lengths,
            softmax_scale=512.0**-0.5,
            attn_sink=attn_sink,
        )
        if out is None:
            raise RuntimeError("dsv4_sparse_attention_two_source_bf16 returned None")
        return out

    result = timed_pair("mini_combined_indexer_sparse_decode", fn, warmup=warmup, repeat=repeat)
    result.update(
        {
            "subgraph": "combined_indexer_sparse_decode",
            "tokens": tokens,
            "history": history,
            "shape": {
                "q_attn": [tokens, 8, 512],
                "q_indexer": [tokens, 64, 128],
                "swa_indices": [tokens, 128],
                "c4_topk": [tokens, 512],
            },
            "boundary_note": "indexer_select -> topk/global-index/lens -> exact bf16 sparse attention",
        }
    )
    return result


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "toggles": {
            **{k: os.environ.get(k) for k in sorted(DEFAULT_TOGGLES)},
            DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE: os.environ.get(
                DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--history", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(394)
    torch.cuda.set_device(0)

    warmup = 2 if args.quick else 5
    repeat = 10 if args.quick else 50
    started = time.time()
    probes = (
        topk_full_transform,
        indexer_select,
        combined_decode_boundary,
    )
    results = [
        probe(args.tokens, args.history, args.page_size, warmup, repeat) for probe in probes
    ]
    output = {
        "suite": "target07_394_mini_global_topk_lens_microbench",
        "scope": "DeepSeek V4 sm80 exact bf16 global topk/lens boundary adaptation",
        "env": env_info(),
        "parameters": {
            "tokens": args.tokens,
            "history": args.history,
            "page_size": args.page_size,
            "local_heads": 8,
            "head_dim": 512,
            "indexer_heads": 64,
            "indexer_head_dim": 128,
            "index_topk": 512,
            "swa_window": 128,
        },
        "started_at_unix": started,
        "elapsed_s": time.time() - started,
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

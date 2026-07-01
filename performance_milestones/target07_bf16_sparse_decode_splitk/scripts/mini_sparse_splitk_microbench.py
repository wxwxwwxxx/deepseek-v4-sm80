#!/usr/bin/env python3
"""TARGET 07.395 exact bf16 sparse decode split-K probe."""

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
    "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS": "1",
}

for key, value in DEFAULT_TOGGLES.items():
    os.environ.setdefault(key, value)

import torch

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


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
def splitk_enabled(enabled: bool) -> Iterator[None]:
    name = dsv4_kernel.DSV4_SM80_SPARSE_SPLITK_BF16_TOGGLE
    old = os.environ.get(name)
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def rounded_pages(length: int, page_size: int) -> int:
    return max(1, math.ceil(max(length, 1) / page_size))


def make_page_table(rows: int, pages: int, *, device: torch.device) -> torch.Tensor:
    base = torch.arange(pages, dtype=torch.int32, device=device)
    return base.repeat(rows, 1).contiguous()


def compare_outputs(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    error = (reference.float() - actual.float()).abs()
    return {
        "max_abs_error": float(error.max().item()) if error.numel() else 0.0,
        "mean_abs_error": float(error.mean().item()) if error.numel() else 0.0,
        "allclose_7e_2": bool(torch.allclose(reference, actual, atol=7e-2, rtol=7e-2)),
    }


def run_legacy_sparse(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    c4_cache: torch.Tensor,
    c4_indices: torch.Tensor,
    c4_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    out = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
        q,
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
        raise RuntimeError("legacy dsv4_sparse_attention_two_source_bf16 returned None")
    return out


def run_splitk_sparse(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    c4_cache: torch.Tensor,
    c4_indices: torch.Tensor,
    c4_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    out = dsv4_kernel.dsv4_sparse_attention_two_source_splitk_bf16(
        q,
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
        raise RuntimeError("split-K dsv4_sparse_attention_two_source_splitk_bf16 returned None")
    return out


def sparse_only_boundary(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    c4_slots = max(1, history // 4)
    q = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16).contiguous()
    swa_cache = torch.randn(history + tokens + 256, 512, device=device, dtype=torch.bfloat16)
    c4_cache = torch.randn(c4_slots + tokens + 64, 512, device=device, dtype=torch.bfloat16)
    swa_indices = torch.randint(0, history, (tokens, 128), device=device, dtype=torch.int32)
    c4_indices = torch.randint(0, c4_slots, (tokens, 512), device=device, dtype=torch.int32)
    swa_lengths = torch.full((tokens,), 128, device=device, dtype=torch.int32)
    c4_lengths = torch.full((tokens,), 512, device=device, dtype=torch.int32)
    attn_sink = torch.zeros(8, device=device, dtype=torch.float32)

    kwargs = {
        "c4_cache": c4_cache,
        "c4_indices": c4_indices,
        "c4_lengths": c4_lengths,
        "attn_sink": attn_sink,
    }

    with splitk_enabled(False):
        legacy_out = run_legacy_sparse(q, swa_cache, swa_indices, swa_lengths, **kwargs)
        legacy = cuda_event_bench(
            "mini_sparse_attention_two_source_legacy_bf16",
            lambda: run_legacy_sparse(q, swa_cache, swa_indices, swa_lengths, **kwargs),
            warmup=warmup,
            repeat=repeat,
        )
    with splitk_enabled(True):
        splitk_out = run_splitk_sparse(q, swa_cache, swa_indices, swa_lengths, **kwargs)
        splitk = cuda_event_bench(
            "mini_sparse_attention_two_source_splitk_bf16",
            lambda: run_splitk_sparse(q, swa_cache, swa_indices, swa_lengths, **kwargs),
            warmup=warmup,
            repeat=repeat,
        )

    before = legacy["mean_ms"]
    after = splitk["mean_ms"]
    return {
        "name": "mini_sparse_attention_two_source",
        "subgraph": "sparse_decode_attention",
        "tokens": tokens,
        "history": history,
        "shape": {
            "q": [tokens, 8, 512],
            "swa_indices": [tokens, 128],
            "c4_indices": [tokens, 512],
        },
        "legacy": legacy,
        "splitk_bf16": splitk,
        "improvement_fraction": (before - after) / before if before > 0 else float("nan"),
        "speedup": before / after if after > 0 else float("nan"),
        "correctness": compare_outputs(legacy_out, splitk_out),
    }


def combined_indexer_sparse_boundary(
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
    q_attn = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16).contiguous()
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

    def selected_sparse_inputs() -> tuple[torch.Tensor, torch.Tensor]:
        selected = dsv4_kernel.indexer_select_bf16_fallback(
            q_indexer,
            weights,
            idx_cache,
            seq_lens,
            page_table,
            page_size=c4_page_size,
            width=512,
            ratio=4,
        )
        c4_indices = selected.topk.page_indices.to(torch.int32)
        if selected.topk.topk_lens is not None:
            c4_lengths = selected.topk.topk_lens.to(torch.int32)
        else:
            c4_lengths = (c4_indices >= 0).sum(dim=-1).to(torch.int32)
        return c4_indices, c4_lengths

    def run_combined(splitk: bool) -> torch.Tensor:
        c4_indices, c4_lengths = selected_sparse_inputs()
        if splitk:
            return run_splitk_sparse(
                q_attn,
                swa_cache,
                swa_indices,
                swa_lengths,
                c4_cache=c4_cache,
                c4_indices=c4_indices,
                c4_lengths=c4_lengths,
                attn_sink=attn_sink,
            )
        return run_legacy_sparse(
            q_attn,
            swa_cache,
            swa_indices,
            swa_lengths,
            c4_cache=c4_cache,
            c4_indices=c4_indices,
            c4_lengths=c4_lengths,
            attn_sink=attn_sink,
        )

    with splitk_enabled(False):
        legacy_out = run_combined(False)
        legacy = cuda_event_bench(
            "mini_combined_indexer_sparse_decode_legacy_bf16",
            lambda: run_combined(False),
            warmup=warmup,
            repeat=repeat,
        )
    with splitk_enabled(True):
        splitk_out = run_combined(True)
        splitk = cuda_event_bench(
            "mini_combined_indexer_sparse_decode_splitk_bf16",
            lambda: run_combined(True),
            warmup=warmup,
            repeat=repeat,
        )

    before = legacy["mean_ms"]
    after = splitk["mean_ms"]
    return {
        "name": "mini_combined_indexer_sparse_decode",
        "subgraph": "globaltopk_indexer_plus_sparse_decode",
        "tokens": tokens,
        "history": history,
        "shape": {
            "q_attn": [tokens, 8, 512],
            "q_indexer": [tokens, 64, 128],
            "swa_indices": [tokens, 128],
            "c4_topk": [tokens, 512],
            "c4_page_size": c4_page_size,
        },
        "legacy": legacy,
        "splitk_bf16": splitk,
        "improvement_fraction": (before - after) / before if before > 0 else float("nan"),
        "speedup": before / after if after > 0 else float("nan"),
        "correctness": compare_outputs(legacy_out, splitk_out),
    }


def env_info() -> dict[str, Any]:
    toggles = {
        **{key: os.environ.get(key) for key in sorted(DEFAULT_TOGGLES)},
        dsv4_kernel.DSV4_SM80_SPARSE_SPLITK_BF16_TOGGLE: os.environ.get(
            dsv4_kernel.DSV4_SM80_SPARSE_SPLITK_BF16_TOGGLE
        ),
    }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "toggles": toggles,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--history", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")
    torch.manual_seed(395)
    torch.cuda.set_device(0)

    warmup = 2 if args.quick else 5
    repeat = 10 if args.quick else 50
    started = time.time()
    results = [
        sparse_only_boundary(args.tokens, args.history, args.page_size, warmup, repeat),
        combined_indexer_sparse_boundary(
            args.tokens,
            args.history,
            args.page_size,
            warmup,
            repeat,
        ),
    ]
    payload = {
        "suite": "target07_395_mini_sparse_splitk_bf16_microbench",
        "scope": "DeepSeek V4 sm80 exact bf16 sparse decode split-K prototype",
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
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

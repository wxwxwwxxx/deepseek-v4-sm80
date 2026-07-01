#!/usr/bin/env python3
"""mini-sglang DeepSeek V4 sm80 attention/indexer/cache/runtime probes.

These probes are deliberately paired around the boundaries found in the
dispatch report:

* sparse attention reads mini's exact bf16 flat caches directly;
* indexer select uses the C4-compressed bf16 cache length (history // 4);
* cache update probes keep SWA, compressed, and indexer stores separate;
* metadata replay copies approximate the graph-replay buffers visible in the
  post-Marlin Nsight trace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

DEFAULT_TOGGLES = {
    "MINISGL_DSV4_SM80_V0_BF16": "1",
    "MINISGL_DSV4_SM80_RMSNORM": "1",
    "MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE": "1",
    "MINISGL_DSV4_SM80_KV_BF16": "1",
    "MINISGL_DSV4_SM80_COMPRESS": "1",
    "MINISGL_DSV4_SM80_COMPRESS_STORE": "1",
    "MINISGL_DSV4_SM80_SPARSE_ATTN_BF16": "1",
    "MINISGL_DSV4_SM80_INDEXER_BF16": "1",
    "MINISGL_DSV4_SM80_TOPK": "1",
}

for key, value in DEFAULT_TOGGLES.items():
    os.environ.setdefault(key, value)

import torch
import torch.nn.functional as F

from minisgl.kernel.deepseek_v4 import (
    compress_norm_rope_store_fallback,
    copy_masked_compressed_locs,
    dsv4_sparse_attention_two_source_bf16,
    indexer_select_bf16_fallback,
    q_kv_norm_rope_cache_fallback,
    q_norm_rope_fallback,
    k_norm_rope_cache_fallback,
    rms_norm_fallback,
)


def p90(values: list[float]) -> float:
    if not values:
        return float("nan")
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
        "cuda_sync": "torch.cuda.synchronize after every measured iteration",
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": p90(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def blocker(name: str, reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "status": "blocked",
        "reason": reason,
    }
    out.update(extra)
    return out


class SyntheticKVCache:
    def __init__(self, compressed: torch.Tensor, indexer: torch.Tensor) -> None:
        self.compressed = compressed
        self.indexer = indexer

    def component_cache(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self.compressed

    def indexer_cache(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self.indexer

    def store_compressed(self, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
        del layer_id
        self.compressed[loc.to(torch.long)] = kv.to(self.compressed.dtype)

    def store_indexer(self, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
        del layer_id
        self.indexer[loc.to(torch.long)] = kv.to(self.indexer.dtype)


def rounded_pages(length: int, page_size: int) -> int:
    return max(1, math.ceil(max(length, 1) / page_size))


def make_page_table(rows: int, pages: int, *, device: torch.device) -> torch.Tensor:
    base = torch.arange(pages, dtype=torch.int32, device=device)
    return base.repeat(rows, 1).contiguous()


def attention_front_swa_store(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    fused_weight = torch.randn(1536, 4096, device=device, dtype=torch.bfloat16) * 0.01
    q_a_weight = torch.ones(1024, device=device, dtype=torch.bfloat16)
    kv_weight = torch.ones(512, device=device, dtype=torch.bfloat16)
    wq_b_weight = torch.randn(4096, 1024, device=device, dtype=torch.bfloat16) * 0.01
    positions = torch.arange(tokens, device=device, dtype=torch.long) + history
    cache = torch.empty(history + tokens + 256, 512, device=device, dtype=torch.bfloat16)
    out_loc = torch.arange(history, history + tokens, device=device, dtype=torch.int32)

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        qr_kv = F.linear(hidden, fused_weight)
        qr, kv = qr_kv.split([1024, 512], dim=-1)
        qr = rms_norm_fallback(qr, q_a_weight, eps=1.0e-6)
        q = F.linear(qr, wq_b_weight).view(tokens, 8, 512)
        fused = q_kv_norm_rope_cache_fallback(
            q,
            kv,
            positions,
            norm_weight=kv_weight,
            rms_norm_eps=1.0e-6,
            cache=cache,
            out_loc=out_loc,
            rotary_dim=64,
            base=10000.0,
        )
        if not fused:
            q_norm_rope_fallback(
                q,
                positions,
                rms_norm_eps=1.0e-6,
                rotary_dim=64,
                base=10000.0,
            )
            k_norm_rope_cache_fallback(
                kv,
                positions,
                norm_weight=kv_weight,
                rms_norm_eps=1.0e-6,
                cache=cache,
                out_loc=out_loc,
                rotary_dim=64,
                base=10000.0,
            )
        return q, kv

    try:
        result = cuda_event_bench(
            f"mini_attention_front_swa_store_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"mini_attention_front_swa_store_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="attention_front_qkv_cache_insert",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "mini",
            "subgraph": "attention_front_qkv_cache_insert",
            "tokens": tokens,
            "history": history,
            "shape": {
                "hidden": [tokens, 4096],
                "q": [tokens, 8, 512],
                "swa_cache": list(cache.shape),
            },
            "dtype_layout": "bf16 q/kv, bf16 SWA cache",
            "boundary_note": "fused wq_a+wkv, q_a RMSNorm, wq_b, q/KV norm+RoPE, SWA cache insert",
        }
    )
    return result


def sparse_attention_two_source(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    c4_slots = max(1, history // 4)
    q = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16)
    swa_cache = torch.randn(history + tokens + 256, 512, device=device, dtype=torch.bfloat16)
    c4_cache = torch.randn(c4_slots + tokens + 64, 512, device=device, dtype=torch.bfloat16)
    swa_indices = torch.randint(0, history, (tokens, 128), device=device, dtype=torch.int32)
    c4_indices = torch.randint(0, c4_slots, (tokens, 512), device=device, dtype=torch.int32)
    swa_lengths = torch.full((tokens,), 128, device=device, dtype=torch.int32)
    c4_lengths = torch.full((tokens,), 512, device=device, dtype=torch.int32)
    attn_sink = torch.zeros(8, device=device, dtype=torch.float32)

    def fn() -> torch.Tensor:
        out = dsv4_sparse_attention_two_source_bf16(
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
            raise RuntimeError("dsv4_sparse_attention_two_source_bf16 returned None")
        return out

    try:
        result = cuda_event_bench(
            f"mini_sparse_attention_two_source_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"mini_sparse_attention_two_source_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="sparse_decode_attention",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "mini",
            "subgraph": "sparse_decode_attention",
            "tokens": tokens,
            "history": history,
            "shape": {
                "q": [tokens, 8, 512],
                "swa_indices": [tokens, 128],
                "c4_indices": [tokens, 512],
                "swa_cache": list(swa_cache.shape),
                "c4_cache": list(c4_cache.shape),
            },
            "dtype_layout": "bf16 flat caches; direct read inside attention kernel",
            "boundary_note": "mini exact sparse attention kernel; paired with vLLM gather/dequant+split-K, not only vLLM split-K core",
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

    try:
        result = cuda_event_bench(
            f"mini_indexer_select_bf16_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
        backend = fn().backend
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"mini_indexer_select_bf16_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="indexer_logits_select_topk",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "mini",
            "subgraph": "indexer_logits_select_topk",
            "tokens": tokens,
            "history": history,
            "backend": backend,
            "shape": {
                "q": [tokens, 64, 128],
                "weights": [tokens, 64],
                "indexer_cache": list(cache.shape),
                "c4_seq_lens": [tokens],
                "page_table": [tokens, c4_pages],
                "c4_page_size": c4_page_size,
            },
            "dtype_layout": "bf16 indexer q/cache, fp32 weights/logits",
            "boundary_note": "mini C4 indexer logits plus topk transform over compressed history length history//4",
        }
    )
    return result


def compressed_cache_store(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    rows_c4 = max(1, math.ceil(tokens / 4))
    rows_idx = rows_c4
    rows_c128 = max(1, math.ceil(tokens / 128))
    c4_slots = max(1, history // 4) + rows_c4 + 64
    c128_slots = max(1, history // 128) + rows_c128 + 8
    idx_slots = c4_slots
    c4_cache = torch.empty(c4_slots, 512, device=device, dtype=torch.bfloat16)
    c128_cache = torch.empty(c128_slots, 512, device=device, dtype=torch.bfloat16)
    idx_cache = torch.empty(idx_slots, 128, device=device, dtype=torch.bfloat16)
    kv_c4 = torch.randn(rows_c4, 512, device=device, dtype=torch.bfloat16)
    kv_c128 = torch.randn(rows_c128, 512, device=device, dtype=torch.bfloat16)
    kv_idx = torch.randn(rows_idx, 128, device=device, dtype=torch.bfloat16)
    norm_weight = torch.ones(512, device=device, dtype=torch.bfloat16)
    pos_c4 = torch.arange(history + 3, history + 3 + rows_c4 * 4, 4, device=device, dtype=torch.long)
    pos_c128 = torch.arange(history + 127, history + 127 + rows_c128 * 128, 128, device=device, dtype=torch.long)
    loc_c4 = torch.arange(history // 4, history // 4 + rows_c4, device=device, dtype=torch.int32)
    loc_c128 = torch.arange(history // 128, history // 128 + rows_c128, device=device, dtype=torch.int32)
    loc_idx = loc_c4.clone()
    c4_kv = SyntheticKVCache(c4_cache, idx_cache)
    c128_kv = SyntheticKVCache(c128_cache, idx_cache)

    def fn() -> None:
        compress_norm_rope_store_fallback(
            c4_kv,
            0,
            kv_c4,
            loc_c4,
            positions=pos_c4,
            norm_weight=norm_weight,
            rms_norm_eps=1.0e-6,
            rotary_dim=64,
            base=10000.0,
            cache_type="compressed",
        )
        compress_norm_rope_store_fallback(
            c128_kv,
            0,
            kv_c128,
            loc_c128,
            positions=pos_c128,
            norm_weight=norm_weight,
            rms_norm_eps=1.0e-6,
            rotary_dim=64,
            base=10000.0,
            cache_type="compressed",
        )
        compress_norm_rope_store_fallback(
            c4_kv,
            0,
            kv_idx,
            loc_idx,
            cache_type="indexer",
            apply_hadamard=True,
        )

    try:
        result = cuda_event_bench(
            f"mini_compressed_indexer_cache_store_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"mini_compressed_indexer_cache_store_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="cache_store_update",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "mini",
            "subgraph": "cache_store_update",
            "tokens": tokens,
            "history": history,
            "shape": {
                "c4_rows": rows_c4,
                "c128_rows": rows_c128,
                "indexer_rows": rows_idx,
                "c4_cache": list(c4_cache.shape),
                "c128_cache": list(c128_cache.shape),
                "indexer_cache": list(idx_cache.shape),
            },
            "dtype_layout": "bf16 compressed cache and bf16 indexer cache",
            "boundary_note": "mini exact compressed C4/C128 norm+RoPE stores plus indexer hadamard store",
        }
    )
    return result


def metadata_replay_copy(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    pages = rounded_pages(history + tokens, page_size)
    c4_width = 512
    c128_width = 64
    window = 128
    src_1d = [
        torch.arange(tokens, device=device, dtype=torch.int32),
        torch.full((tokens,), history + tokens, device=device, dtype=torch.int32),
        torch.full((tokens,), min(history + tokens, window), device=device, dtype=torch.int32),
        torch.full((tokens,), min(history // 4, c4_width), device=device, dtype=torch.int32),
        torch.full((tokens,), max(history // 128, 1), device=device, dtype=torch.int32),
    ]
    dst_1d = [torch.empty_like(t) for t in src_1d]
    src_2d = [
        make_page_table(tokens, pages, device=device),
        torch.randint(0, history, (tokens, window), device=device, dtype=torch.int32),
        torch.randint(0, max(history // 4, 1), (tokens, c4_width), device=device, dtype=torch.int32),
        torch.randint(0, max(history // 128, 1), (tokens, c128_width), device=device, dtype=torch.int32),
    ]
    dst_2d = [torch.empty_like(t) for t in src_2d]
    raw_out_loc = torch.arange(history, history + tokens, device=device, dtype=torch.int32)
    positions = torch.arange(history, history + tokens, device=device, dtype=torch.int32)
    c4_out_loc = torch.empty(tokens, device=device, dtype=torch.int32)
    c128_out_loc = torch.empty(tokens, device=device, dtype=torch.int32)

    def fn() -> None:
        for dst, src in zip(dst_1d, src_1d):
            dst.copy_(src)
        for dst, src in zip(dst_2d, src_2d):
            dst.copy_(src)
        copy_masked_compressed_locs(raw_out_loc, positions, c4_out_loc, c128_out_loc, tokens)

    try:
        result = cuda_event_bench(
            f"mini_graph_replay_metadata_copy_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"mini_graph_replay_metadata_copy_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="graph_runtime_metadata",
            tokens=tokens,
            history=history,
        )
    bytes_copied = sum(t.numel() * t.element_size() for t in src_1d + src_2d)
    result.update(
        {
            "engine": "mini",
            "subgraph": "graph_runtime_metadata",
            "tokens": tokens,
            "history": history,
            "page_size": page_size,
            "shape": {
                "page_table": [tokens, pages],
                "swa_indices": [tokens, window],
                "c4_indices": [tokens, c4_width],
                "c128_indices": [tokens, c128_width],
                "compressed_loc_buffers": [tokens],
            },
            "bytes_copied_device_to_device": bytes_copied,
            "dtype_layout": "int32 replay metadata buffers",
            "boundary_note": "stand-in for DSV4AttentionBackend._copy_metadata_for_replay plus graph-stage compressed loc copy",
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
        c4_lengths = torch.full((tokens,), min(512, c4_len), device=device, dtype=torch.int32)
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

    try:
        result = cuda_event_bench(
            f"mini_combined_indexer_sparse_decode_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"mini_combined_indexer_sparse_decode_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="combined_indexer_sparse_decode",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "mini",
            "subgraph": "combined_indexer_sparse_decode",
            "tokens": tokens,
            "history": history,
            "shape": {
                "q_attn": [tokens, 8, 512],
                "q_indexer": [tokens, 64, 128],
                "swa_indices": [tokens, 128],
                "c4_topk": [tokens, 512],
            },
            "dtype_layout": "bf16 exact indexer cache and bf16 exact attention caches",
            "boundary_note": "synthetic combined decode of indexer_select -> sparse_attention; excludes model projections and engine graph replay",
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
        "toggles": {k: os.environ.get(k) for k in sorted(DEFAULT_TOGGLES)},
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
    torch.manual_seed(393)
    torch.cuda.set_device(0)

    warmup = 2 if args.quick else 5
    repeat = 10 if args.quick else 50
    started = time.time()
    probes = (
        attention_front_swa_store,
        compressed_cache_store,
        metadata_replay_copy,
        indexer_select,
        sparse_attention_two_source,
        combined_decode_boundary,
    )
    results: list[dict[str, Any]] = []
    for probe in probes:
        results.append(probe(args.tokens, args.history, args.page_size, warmup, repeat))
        torch.cuda.empty_cache()

    output = {
        "suite": "mini_attention_indexer_cache_runtime_microbench",
        "scope": "DeepSeek V4 sm80 exact mini bf16 cache/indexer/attention boundaries",
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

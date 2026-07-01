#!/usr/bin/env python3
"""vLLM DeepSeek V4 sm80 attention/indexer/cache/runtime probes.

Run this inside the vLLM environment, for example:

  PYTHONPATH=/workspace/vllm-dsv4-docker /workspace/venvs/vllm-dsv4/bin/python \
    performance_milestones/target07_attention_indexer_cache_runtime/scripts/vllm_attention_indexer_cache_microbench.py \
    --quick --output performance_milestones/target07_attention_indexer_cache_runtime/raw/vllm_microbench.json

The probes are intentionally synthetic but boundary-aware.  They measure the
SM80 pieces vLLM actually dispatches after backend selection:
packed fp8_ds_mla gather/dequant, split-K sparse attention, global topk mapping,
and fused indexer Q RoPE/quant.  Full engine-owned cache metadata and CUDA graph
capture are recorded as blockers rather than faked.
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


VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", "/workspace/vllm-dsv4-docker"))
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))

os.environ.setdefault("VLLM_USE_V1", "1")

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.deepseek_v4_attention import (
    _dsv4_sm80_sparse_attn_decode_triton,
)
from vllm.v1.attention.ops.deepseek_v4_ops import (
    compute_global_topk_indices_and_lens,
    fused_indexer_q_rope_quant,
    fused_q_kv_rmsnorm,
    gather_dequant_two_scopes_with_mask,
    quantize_and_insert_k_cache,
)


TOKEN_FP8_DIM = 448
TOKEN_BF16_DIM = 64
TOKEN_SCALE_DIM = 8
TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2
HEAD_BYTES = TOKEN_DATA_SIZE + TOKEN_SCALE_DIM


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


def make_cos_sin(max_pos: int, rotary_dim: int, *, device: torch.device) -> torch.Tensor:
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device) / rotary_dim))
    freqs = torch.outer(pos, inv)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def make_fp8_ds_mla_cache(num_blocks: int, block_size: int, *, device: torch.device) -> torch.Tensor:
    """Make a flat block buffer matching vLLM's fp8_ds_mla raw pointer layout."""
    cache = torch.zeros(
        (num_blocks, block_size * HEAD_BYTES),
        device=device,
        dtype=torch.uint8,
    )
    scale_start = block_size * TOKEN_DATA_SIZE
    scales = cache[:, scale_start : scale_start + block_size * TOKEN_SCALE_DIM]
    scales = scales.view(num_blocks, block_size, TOKEN_SCALE_DIM)
    scales[..., : TOKEN_FP8_DIM // 64] = 127
    return cache


def attention_front_projection(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del history, page_size
    device = torch.device("cuda")
    hidden = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16)
    fused_weight = torch.randn(1536, 4096, device=device, dtype=torch.bfloat16) * 0.01
    q_weight = torch.ones(1024, device=device, dtype=torch.bfloat16)
    kv_weight = torch.ones(512, device=device, dtype=torch.bfloat16)
    wq_b_weight = torch.randn(4096, 1024, device=device, dtype=torch.bfloat16) * 0.01

    def fn() -> tuple[torch.Tensor, torch.Tensor]:
        qr_kv = F.linear(hidden, fused_weight)
        qr, kv = qr_kv.split([1024, 512], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(qr.contiguous(), kv.contiguous(), q_weight, kv_weight, 1.0e-6)
        q = F.linear(qr, wq_b_weight).view(tokens, 8, 512)
        return q, kv

    try:
        result = cuda_event_bench(
            f"vllm_attention_front_projection_t{tokens}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"vllm_attention_front_projection_t{tokens}",
            f"{type(exc).__name__}: {exc}",
            subgraph="attention_front_qkv_cache_insert",
            tokens=tokens,
        )
    result.update(
        {
            "engine": "vllm",
            "subgraph": "attention_front_qkv_cache_insert",
            "tokens": tokens,
            "shape": {
                "hidden": [tokens, 4096],
                "fused_wqa_wkv_weight": [1536, 4096],
                "q": [tokens, 8, 512],
            },
            "dtype_layout": "bf16 projection inputs; engine path then quant-inserts fp8_ds_mla cache",
            "boundary_note": "excludes fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert because that op needs engine-owned SWA cache metadata",
        }
    )
    return result


def fp8_swa_cache_quant_insert(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    block_size = 64
    num_blocks = math.ceil((history + tokens + block_size) / block_size)
    cache = make_fp8_ds_mla_cache(num_blocks, block_size, device=device)
    k = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16)
    slot_mapping = torch.arange(history, history + tokens, device=device, dtype=torch.int64)

    def fn() -> None:
        quantize_and_insert_k_cache(k, cache, slot_mapping, block_size=block_size)

    try:
        result = cuda_event_bench(
            f"vllm_fp8_swa_quant_insert_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"vllm_fp8_swa_quant_insert_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="cache_store_update",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "vllm",
            "subgraph": "cache_store_update",
            "tokens": tokens,
            "history": history,
            "shape": {"k": [tokens, 512], "fp8_ds_mla_cache_flat": list(cache.shape)},
            "dtype_layout": "uint8 fp8_ds_mla cache, 448 fp8 bytes + 64 bf16 values + 8 scales per token",
            "boundary_note": "quantize_and_insert_k_cache standalone proxy; full engine uses fused qnorm/RoPE/quant/insert custom op",
        }
    )
    return result


def global_topk_mapping(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    c4_page_size = max(page_size // 4, 1)
    c4_len = max(history // 4, 1)
    c4_pages = math.ceil(c4_len / c4_page_size)
    topk_indices = torch.randint(0, c4_len, (tokens, 512), device=device, dtype=torch.int32)
    token_to_req = torch.arange(tokens, device=device, dtype=torch.int32)
    block_table = torch.arange(tokens * c4_pages, device=device, dtype=torch.int32).view(tokens, c4_pages)
    is_valid = torch.ones(tokens, device=device, dtype=torch.bool)

    def fn() -> Any:
        return compute_global_topk_indices_and_lens(
            topk_indices,
            token_to_req,
            block_table,
            c4_page_size,
            is_valid,
        )

    try:
        result = cuda_event_bench(
            f"vllm_global_topk_indices_lens_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"vllm_global_topk_indices_lens_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="graph_runtime_metadata",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "vllm",
            "subgraph": "graph_runtime_metadata",
            "tokens": tokens,
            "history": history,
            "shape": {
                "topk_indices": [tokens, 512],
                "block_table": [tokens, c4_pages],
                "c4_page_size": c4_page_size,
            },
            "dtype_layout": "int32 topk/block metadata",
            "boundary_note": "vLLM fused local-topk to global-slot mapping and valid-lens count",
        }
    )
    return result


def indexer_q_rope_quant(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    positions = torch.arange(history, history + tokens, device=device, dtype=torch.long)
    q = torch.randn(tokens, 64, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(tokens, 64, device=device, dtype=torch.float32)
    cos_sin = make_cos_sin(history + tokens + 1, 128, device=device)

    def fn() -> Any:
        return fused_indexer_q_rope_quant(
            positions,
            q,
            cos_sin,
            weights,
            512.0**-0.5,
            64.0**-0.5,
            use_fp4=False,
        )

    try:
        result = cuda_event_bench(
            f"vllm_indexer_q_rope_quant_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"vllm_indexer_q_rope_quant_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="indexer_logits_select_topk",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "vllm",
            "subgraph": "indexer_logits_select_topk",
            "tokens": tokens,
            "history": history,
            "shape": {"q": [tokens, 64, 128], "weights": [tokens, 64]},
            "dtype_layout": "bf16 q input, fp8 q output on SM80 reference path, fp32 folded weights",
            "boundary_note": "vLLM fused indexer Q RoPE + FP8 quant/weight fold; full SparseAttnIndexer logits/topk requires engine attention metadata and FP8 indexer cache",
        }
    )
    return result


def gather_dequant_dual_scope(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    block_size = 64
    num_blocks = math.ceil((history + tokens + block_size) / block_size)
    c4_blocks = math.ceil((history // 4 + tokens + block_size) / block_size)
    swa_cache = make_fp8_ds_mla_cache(num_blocks, block_size, device=device)
    c4_cache = make_fp8_ds_mla_cache(c4_blocks, block_size, device=device)
    swa_indices = torch.randint(0, history, (tokens, 128), device=device, dtype=torch.int64)
    c4_indices = torch.randint(0, max(history // 4, 1), (tokens, 512), device=device, dtype=torch.int64)
    swa_lens = torch.full((tokens,), 128, device=device, dtype=torch.int32)
    c4_lens = torch.full((tokens,), 512, device=device, dtype=torch.int32)

    def fn() -> Any:
        return gather_dequant_two_scopes_with_mask(
            swa_cache,
            block_size,
            swa_indices,
            swa_lens,
            c4_cache,
            block_size,
            c4_indices,
            c4_lens,
            TOKEN_FP8_DIM,
            TOKEN_BF16_DIM,
            512,
        )

    try:
        result = cuda_event_bench(
            f"vllm_gather_dequant_two_scopes_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"vllm_gather_dequant_two_scopes_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="sparse_decode_attention",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "vllm",
            "subgraph": "sparse_decode_attention",
            "tokens": tokens,
            "history": history,
            "shape": {
                "swa_indices": [tokens, 128],
                "c4_indices": [tokens, 512],
                "gathered_output": [tokens, 640, 512],
                "fp8_ds_mla_cache_flat": list(swa_cache.shape),
            },
            "dtype_layout": "uint8 fp8_ds_mla cache -> bf16 gathered KV + bool invalid mask",
            "boundary_note": "vLLM sm80 decode gather/dequant/mask production, one launch per scope",
        }
    )
    return result


def sparse_attention_core(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del history, page_size
    device = torch.device("cuda")
    total_topk = 640
    q = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16)
    gathered = torch.randn(tokens, total_topk, 512, device=device, dtype=torch.bfloat16)
    invalid = torch.zeros(tokens, total_topk, device=device, dtype=torch.bool)
    attn_sink = torch.zeros(8, device=device, dtype=torch.float32)

    def fn() -> torch.Tensor:
        return _dsv4_sm80_sparse_attn_decode_triton(
            q,
            gathered,
            invalid,
            attn_sink,
            512.0**-0.5,
            512,
        )

    try:
        result = cuda_event_bench(
            f"vllm_sparse_attention_splitk_core_t{tokens}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"vllm_sparse_attention_splitk_core_t{tokens}",
            f"{type(exc).__name__}: {exc}",
            subgraph="sparse_decode_attention_core_only",
            tokens=tokens,
        )
    result.update(
        {
            "engine": "vllm",
            "subgraph": "sparse_decode_attention_core_only",
            "tokens": tokens,
            "shape": {"q": [tokens, 8, 512], "gathered_kv": [tokens, total_topk, 512]},
            "dtype_layout": "bf16 gathered KV/output; starts after fp8 cache gather/dequant",
            "boundary_note": "not directly comparable to mini bf16-cache sparse attention by itself",
        }
    )
    return result


def combined_gather_sparse_decode(
    tokens: int,
    history: int,
    page_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    del page_size
    device = torch.device("cuda")
    block_size = 64
    num_blocks = math.ceil((history + tokens + block_size) / block_size)
    c4_blocks = math.ceil((history // 4 + tokens + block_size) / block_size)
    swa_cache = make_fp8_ds_mla_cache(num_blocks, block_size, device=device)
    c4_cache = make_fp8_ds_mla_cache(c4_blocks, block_size, device=device)
    swa_indices = torch.randint(0, history, (tokens, 128), device=device, dtype=torch.int64)
    c4_indices = torch.randint(0, max(history // 4, 1), (tokens, 512), device=device, dtype=torch.int64)
    swa_lens = torch.full((tokens,), 128, device=device, dtype=torch.int32)
    c4_lens = torch.full((tokens,), 512, device=device, dtype=torch.int32)
    q = torch.randn(tokens, 8, 512, device=device, dtype=torch.bfloat16)
    attn_sink = torch.zeros(8, device=device, dtype=torch.float32)

    def fn() -> torch.Tensor:
        gathered, invalid = gather_dequant_two_scopes_with_mask(
            swa_cache,
            block_size,
            swa_indices,
            swa_lens,
            c4_cache,
            block_size,
            c4_indices,
            c4_lens,
            TOKEN_FP8_DIM,
            TOKEN_BF16_DIM,
            512,
        )
        return _dsv4_sm80_sparse_attn_decode_triton(
            q,
            gathered,
            invalid,
            attn_sink,
            512.0**-0.5,
            512,
        )

    try:
        result = cuda_event_bench(
            f"vllm_combined_gather_sparse_decode_t{tokens}_h{history}",
            fn,
            warmup=warmup,
            repeat=repeat,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return blocker(
            f"vllm_combined_gather_sparse_decode_t{tokens}_h{history}",
            f"{type(exc).__name__}: {exc}",
            subgraph="combined_indexer_sparse_decode",
            tokens=tokens,
            history=history,
        )
    result.update(
        {
            "engine": "vllm",
            "subgraph": "combined_indexer_sparse_decode",
            "tokens": tokens,
            "history": history,
            "shape": {
                "q": [tokens, 8, 512],
                "swa_indices": [tokens, 128],
                "c4_indices": [tokens, 512],
                "merged_topk": [tokens, 640],
            },
            "dtype_layout": "uint8 fp8_ds_mla cache -> bf16 gathered KV -> bf16 attention output",
            "boundary_note": "vLLM sm80 sparse decode after topk indices are known; excludes SparseAttnIndexer logits/topk",
        }
    )
    return result


def engine_boundary_blockers(tokens: int, history: int) -> list[dict[str, Any]]:
    return [
        blocker(
            f"vllm_sparse_attn_indexer_full_logits_topk_t{tokens}_h{history}",
            "Full SparseAttnIndexer decode requires vLLM attention metadata, kv-cache manager, FP8 indexer cache pages, and custom op registration state. Standalone synthetic q/cache timings would not preserve the engine boundary.",
            engine="vllm",
            subgraph="indexer_logits_select_topk",
            tokens=tokens,
            history=history,
            evidence="vllm/model_executor/layers/sparse_attn_indexer.py",
        ),
        blocker(
            f"vllm_deepseek_v4_attention_custom_op_full_t{tokens}_h{history}",
            "torch.ops.vllm.deepseek_v4_attention mutates engine-owned output/cache buffers and depends on the active forward context. The paired probes measure its callable pieces instead.",
            engine="vllm",
            subgraph="attention_full_custom_op",
            tokens=tokens,
            history=history,
            evidence="vllm/model_executor/layers/deepseek_v4_attention.py",
        ),
        blocker(
            f"vllm_cudagraph_dispatcher_runtime_t{tokens}_h{history}",
            "CUDA graph capture/replay requires GPUModelRunner, CudagraphDispatcher descriptors, persistent input buffers, and attention backend builders. This target records dispatch evidence and macro logs rather than faking graph replay.",
            engine="vllm",
            subgraph="graph_runtime_metadata",
            tokens=tokens,
            history=history,
            evidence="vllm/v1/worker/gpu_model_runner.py",
        ),
    ]


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "vllm_root": str(VLLM_ROOT),
        "env_toggles": {
            key: os.environ.get(key)
            for key in (
                "VLLM_USE_V1",
                "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
                "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
            )
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
    torch.manual_seed(7393)
    torch.cuda.set_device(0)

    warmup = 2 if args.quick else 5
    repeat = 10 if args.quick else 50
    started = time.time()
    probes = (
        attention_front_projection,
        fp8_swa_cache_quant_insert,
        global_topk_mapping,
        indexer_q_rope_quant,
        gather_dequant_dual_scope,
        sparse_attention_core,
        combined_gather_sparse_decode,
    )
    results: list[dict[str, Any]] = []
    for probe in probes:
        results.append(probe(args.tokens, args.history, args.page_size, warmup, repeat))
        torch.cuda.empty_cache()
    results.extend(engine_boundary_blockers(args.tokens, args.history))

    output = {
        "suite": "vllm_attention_indexer_cache_runtime_microbench",
        "scope": "DeepSeek V4 sm80 vLLM packed-cache/indexer/attention boundaries",
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
            "fp8_ds_mla_block_size": 64,
            "fp8_ds_mla_head_bytes": HEAD_BYTES,
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

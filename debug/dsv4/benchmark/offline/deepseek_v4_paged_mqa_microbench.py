from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _with_env(name: str, value: str | None):
    class Guard:
        def __enter__(self):
            self.prev = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

        def __exit__(self, exc_type, exc, tb):
            if self.prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = self.prev

    return Guard()


def _time_cuda_wall(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _make_case(
    *,
    tokens: int,
    heads: int,
    dim: int,
    candidates: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(1000 + candidates)
    cache_rows = max(candidates + tokens * 17 + 256, 512)
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16, generator=generator)
    cache = torch.randn(cache_rows, dim, device=device, dtype=torch.bfloat16, generator=generator)
    contexts = []
    for row in range(tokens):
        start = row * 17
        row_indices = (
            torch.arange(candidates, device=device, dtype=torch.int32) + start
        ) % cache_rows
        contexts.append(row_indices)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32, generator=generator)
    return q, cache, contexts, attn_sink


def _bench_case(
    *,
    candidates: int,
    tokens: int,
    heads: int,
    dim: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    q, cache, contexts, attn_sink = _make_case(
        tokens=tokens,
        heads=heads,
        dim=dim,
        candidates=candidates,
        device=device,
    )

    with _with_env("MINISGL_DSV4_SM80_PAGED_MQA_BF16", None):
        metadata = dsv4_kernel.get_paged_mqa_logits_metadata_fallback(contexts, device=device)
        expected = dsv4_kernel.paged_mqa_attention_fallback(
            q,
            cache,
            contexts,
            softmax_scale=dim**-0.5,
            attn_sink=attn_sink,
        )
        fallback_ms = _time_cuda_wall(
            lambda: dsv4_kernel.paged_mqa_attention_fallback(
                q,
                cache,
                contexts,
                softmax_scale=dim**-0.5,
                attn_sink=attn_sink,
            ),
            warmup=warmup,
            iters=iters,
        )

    metadata_build_ms = _time_cuda_wall(
        lambda: dsv4_kernel.get_paged_mqa_logits_metadata_fallback(contexts, device=device),
        warmup=warmup,
        iters=iters,
    )

    with _with_env("MINISGL_DSV4_SM80_PAGED_MQA_BF16", "1"):
        actual = dsv4_kernel.paged_mqa_attention_fallback(
            q,
            cache,
            metadata,
            softmax_scale=dim**-0.5,
            attn_sink=attn_sink,
        )
        triton_metadata_ms = _time_cuda_wall(
            lambda: dsv4_kernel.paged_mqa_attention_fallback(
                q,
                cache,
                metadata,
                softmax_scale=dim**-0.5,
                attn_sink=attn_sink,
            ),
            warmup=warmup,
            iters=iters,
        )
        triton_list_ms = _time_cuda_wall(
            lambda: dsv4_kernel.paged_mqa_attention_fallback(
                q,
                cache,
                contexts,
                softmax_scale=dim**-0.5,
                attn_sink=attn_sink,
            ),
            warmup=warmup,
            iters=iters,
        )

    error = (actual.float() - expected.float()).abs()
    return {
        "tokens": tokens,
        "heads": heads,
        "dim": dim,
        "candidates": candidates,
        "metadata_total_indices": int(metadata.indices.numel()),
        "fallback_list_wall_ms": fallback_ms,
        "metadata_build_wall_ms": metadata_build_ms,
        "triton_metadata_wall_ms": triton_metadata_ms,
        "triton_list_wall_ms": triton_list_ms,
        "speedup_vs_fallback": fallback_ms / triton_metadata_ms
        if triton_metadata_ms > 0
        else None,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "allclose_3e_2": bool(torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DeepSeek V4 bf16 flat-cache paged MQA attention."
    )
    parser.add_argument("--candidates", default="32,128,640,1024")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dsv4_paged_mqa_attention_bf16_microbench.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")

    device = torch.device("cuda")
    started = time.time()
    results = []
    for candidates in _parse_csv_ints(args.candidates):
        result = _bench_case(
            candidates=candidates,
            tokens=args.tokens,
            heads=args.heads,
            dim=args.dim,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        results.append(result)
        print(
            f"candidates={candidates}: fallback={result['fallback_list_wall_ms']:.3f} ms, "
            f"triton={result['triton_metadata_wall_ms']:.3f} ms, "
            f"speedup={result['speedup_vs_fallback']:.2f}x, "
            f"ok={result['allclose_3e_2']}"
        )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "capabilities": asdict(dsv4_kernel.detect_dsv4_kernel_capabilities()),
        "toggle": "MINISGL_DSV4_SM80_PAGED_MQA_BF16",
        "timer": "wall_time_with_cuda_synchronize",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

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


def _manual_two_source_attention(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    compressed_cache: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    compressed_lengths: torch.Tensor | None,
    softmax_scale: float,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q)
    sink = attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)

    def _candidates(
        cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        row: int,
    ) -> torch.Tensor | None:
        row_len = max(0, min(int(lengths[row].item()), indices.shape[-1]))
        if row_len == 0:
            return None
        row_indices = indices[row, :row_len]
        row_indices = row_indices[row_indices >= 0]
        if row_indices.numel() == 0:
            return None
        return cache[row_indices.to(torch.long)].float()

    for row in range(q.shape[0]):
        sources = []
        if (
            compressed_cache is not None
            and compressed_indices is not None
            and compressed_lengths is not None
        ):
            compressed = _candidates(
                compressed_cache,
                compressed_indices,
                compressed_lengths,
                row,
            )
            if compressed is not None:
                sources.append(compressed)
        swa = _candidates(swa_cache, swa_indices, swa_lengths, row)
        if swa is not None:
            sources.append(swa)
        if not sources:
            out[row].zero_()
            continue
        candidates = torch.cat(sources, dim=0)
        scores = torch.einsum("hd,td->ht", q[row].float(), candidates) * softmax_scale
        max_score = torch.maximum(scores.max(dim=-1).values, sink)
        exp_scores = torch.exp(scores - max_score[:, None])
        denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
        attn = exp_scores / denom[:, None]
        out[row] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
    return out


def _make_case(
    *,
    total_candidates: int,
    tokens: int,
    heads: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(5000 + total_candidates)
    swa_candidates = max(1, total_candidates // 4)
    compressed_candidates = max(0, total_candidates - swa_candidates)
    swa_rows = max(swa_candidates + tokens * 19 + 256, 512)
    compressed_rows = max(compressed_candidates + tokens * 23 + 256, 512)
    q = torch.randn(tokens, heads, 512, device=device, dtype=torch.bfloat16, generator=generator)
    swa_cache = torch.randn(
        swa_rows,
        512,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    compressed_cache = torch.randn(
        compressed_rows,
        512,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    swa_indices = torch.empty(tokens, swa_candidates, device=device, dtype=torch.int32)
    compressed_indices = torch.empty(
        tokens,
        max(1, compressed_candidates),
        device=device,
        dtype=torch.int32,
    )
    for row in range(tokens):
        swa_indices[row] = (
            torch.arange(swa_candidates, device=device, dtype=torch.int32) + row * 19
        ) % swa_rows
        if compressed_candidates:
            compressed_indices[row, :compressed_candidates] = (
                torch.arange(compressed_candidates, device=device, dtype=torch.int32) + row * 23
            ) % compressed_rows
        else:
            compressed_indices[row, 0] = -1
    return {
        "q": q,
        "swa_cache": swa_cache,
        "compressed_cache": compressed_cache,
        "swa_indices": swa_indices,
        "swa_lengths": torch.full((tokens,), swa_candidates, device=device, dtype=torch.int32),
        "compressed_indices": compressed_indices,
        "compressed_lengths": torch.full(
            (tokens,),
            compressed_candidates,
            device=device,
            dtype=torch.int32,
        ),
        "attn_sink": torch.randn(heads, device=device, dtype=torch.float32, generator=generator),
    }


def _bench_case(
    *,
    total_candidates: int,
    tokens: int,
    heads: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    case = _make_case(
        total_candidates=total_candidates,
        tokens=tokens,
        heads=heads,
        device=device,
    )
    softmax_scale = 512**-0.5

    with _with_env("MINISGL_DSV4_SM80_SPARSE_ATTN_BF16", None):
        expected = _manual_two_source_attention(
            **case,
            softmax_scale=softmax_scale,
        )
        reference_ms = _time_cuda_wall(
            lambda: _manual_two_source_attention(**case, softmax_scale=softmax_scale),
            warmup=warmup,
            iters=iters,
        )

    def run_kernel(compressed_lengths: torch.Tensor):
        out = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
            case["q"],
            case["swa_cache"],
            case["swa_indices"],
            case["swa_lengths"],
            compressed_cache=case["compressed_cache"],
            compressed_indices=case["compressed_indices"],
            compressed_lengths=compressed_lengths,
            softmax_scale=softmax_scale,
            attn_sink=case["attn_sink"],
        )
        if out is None:
            raise RuntimeError("dsv4_sparse_attention_two_source_bf16 returned None")
        return out

    with _with_env("MINISGL_DSV4_SM80_SPARSE_ATTN_BF16", "1"):
        actual = run_kernel(case["compressed_lengths"])
        kernel_ms = _time_cuda_wall(
            lambda: run_kernel(case["compressed_lengths"]),
            warmup=warmup,
            iters=iters,
        )
        kernel_with_length_scan_ms = _time_cuda_wall(
            lambda: run_kernel((case["compressed_indices"] >= 0).sum(dim=-1).to(torch.int32)),
            warmup=warmup,
            iters=iters,
        )

    error = (actual.float() - expected.float()).abs()
    compressed_candidates = int(case["compressed_lengths"][0].item())
    swa_candidates = int(case["swa_lengths"][0].item())
    return {
        "tokens": tokens,
        "heads": heads,
        "dim": 512,
        "total_candidates": total_candidates,
        "compressed_candidates": compressed_candidates,
        "swa_candidates": swa_candidates,
        "reference_wall_ms": reference_ms,
        "kernel_direct_wall_ms": kernel_ms,
        "kernel_with_length_scan_wall_ms": kernel_with_length_scan_ms,
        "speedup_vs_reference": reference_ms / kernel_ms if kernel_ms > 0 else None,
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "allclose_6e_2": bool(torch.allclose(actual, expected, atol=6e-2, rtol=6e-2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DSV4 sm80 bf16 two-source sparse attention."
    )
    parser.add_argument("--candidates", default="128,640,1024")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dsv4_sparse_attention_two_source_bf16_microbench.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")

    device = torch.device("cuda")
    started = time.time()
    results = []
    for total_candidates in _parse_csv_ints(args.candidates):
        result = _bench_case(
            total_candidates=total_candidates,
            tokens=args.tokens,
            heads=args.heads,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        results.append(result)
        print(
            f"candidates={total_candidates}: reference={result['reference_wall_ms']:.3f} ms, "
            f"kernel={result['kernel_direct_wall_ms']:.3f} ms, "
            f"adapter={result['kernel_with_length_scan_wall_ms']:.3f} ms, "
            f"speedup={result['speedup_vs_reference']:.2f}x, "
            f"ok={result['allclose_6e_2']}"
        )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "capabilities": asdict(dsv4_kernel.detect_dsv4_kernel_capabilities()),
        "toggle": "MINISGL_DSV4_SM80_SPARSE_ATTN_BF16",
        "timer": "wall_time_with_cuda_synchronize",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

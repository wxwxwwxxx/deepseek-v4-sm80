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
    dim: int,
    cache_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(5000 + tokens)
    kv = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16, generator=generator)
    norm_weight = torch.randn(dim, device=device, dtype=torch.bfloat16, generator=generator)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    out_loc = ((torch.arange(tokens, device=device, dtype=torch.int32) * 17 + 3) % cache_rows).to(
        torch.int32
    )
    return kv, norm_weight, positions, out_loc


def _bench_case(
    *,
    tokens: int,
    dim: int,
    rotary_dim: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    cache_rows = max(tokens * 32 + 256, 4096)
    kv, norm_weight, positions, out_loc = _make_case(
        tokens=tokens,
        dim=dim,
        cache_rows=cache_rows,
        device=device,
    )
    fallback_kv = torch.empty_like(kv)
    fallback_cache = torch.empty(cache_rows, dim, device=device, dtype=torch.bfloat16)
    triton_kv = torch.empty_like(kv)
    triton_cache = torch.empty_like(fallback_cache)

    def run_once(kv_buf: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        kv_buf.copy_(kv)
        cache.zero_()
        return dsv4_kernel.k_norm_rope_cache_fallback(
            kv_buf,
            positions,
            norm_weight=norm_weight,
            rms_norm_eps=1e-6,
            cache=cache,
            out_loc=out_loc,
            rotary_dim=rotary_dim,
            base=10000.0,
            original_seq_len=4096,
            factor=2.0,
        )

    with _with_env("MINISGL_DSV4_SM80_KV_BF16", None):
        expected = run_once(fallback_kv, fallback_cache).clone()
        expected_cache = fallback_cache.clone()
        fallback_ms = _time_cuda_wall(
            lambda: run_once(fallback_kv, fallback_cache),
            warmup=warmup,
            iters=iters,
        )

    with _with_env("MINISGL_DSV4_SM80_KV_BF16", "1"):
        actual = run_once(triton_kv, triton_cache).clone()
        actual_cache = triton_cache.clone()
        triton_ms = _time_cuda_wall(
            lambda: run_once(triton_kv, triton_cache),
            warmup=warmup,
            iters=iters,
        )

    kv_error = (actual.float() - expected.float()).abs()
    cache_error = (actual_cache.float() - expected_cache.float()).abs()
    return {
        "tokens": tokens,
        "dim": dim,
        "rotary_dim": rotary_dim,
        "cache_rows": cache_rows,
        "fallback_wall_ms": fallback_ms,
        "triton_wall_ms": triton_ms,
        "speedup_vs_fallback": fallback_ms / triton_ms if triton_ms > 0 else None,
        "kv_max_abs_error": float(kv_error.max().item()) if kv_error.numel() else 0.0,
        "cache_max_abs_error": float(cache_error.max().item()) if cache_error.numel() else 0.0,
        "allclose_2e_2": bool(
            torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
            and torch.allclose(actual_cache, expected_cache, atol=2e-2, rtol=2e-2)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DeepSeek V4 bf16 K RMSNorm + RoPE + flat cache store."
    )
    parser.add_argument("--tokens", default="1,8,64,512")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dsv4_k_norm_rope_cache_bf16_microbench.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")

    device = torch.device("cuda")
    started = time.time()
    results = []
    for tokens in _parse_csv_ints(args.tokens):
        result = _bench_case(
            tokens=tokens,
            dim=args.dim,
            rotary_dim=args.rotary_dim,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        results.append(result)
        print(
            f"tokens={tokens}: fallback={result['fallback_wall_ms']:.3f} ms, "
            f"triton={result['triton_wall_ms']:.3f} ms, "
            f"speedup={result['speedup_vs_fallback']:.2f}x, "
            f"ok={result['allclose_2e_2']}"
        )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_capability": torch.cuda.get_device_capability(),
        "capabilities": asdict(dsv4_kernel.detect_dsv4_kernel_capabilities()),
        "toggle": "MINISGL_DSV4_SM80_KV_BF16",
        "timer": "wall_time_with_cuda_synchronize",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


Mode = Literal["store_only", "norm_store", "norm_rope_store"]


class _FakeCompressedCache:
    def __init__(self, cache: torch.Tensor) -> None:
        self.cache = cache

    def component_cache(self, layer_id: int) -> torch.Tensor:
        assert layer_id == 0
        return self.cache

    def store_compressed(self, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
        assert layer_id == 0
        flat = kv.reshape(-1, self.cache.shape[-1])
        valid = loc.reshape(-1) >= 0
        self.cache[loc.reshape(-1)[valid].long()] = flat[valid].to(self.cache.dtype)


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
    generator.manual_seed(6100 + tokens)
    kv = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16, generator=generator)
    norm_weight = torch.randn(dim, device=device, dtype=torch.bfloat16, generator=generator)
    positions = torch.arange(tokens, device=device, dtype=torch.int64) * 4 + 3
    out_loc = ((torch.arange(tokens, device=device, dtype=torch.int32) * 19 + 5) % cache_rows).to(
        torch.int32
    )
    return kv, norm_weight, positions, out_loc


def _run_once(
    *,
    mode: Mode,
    kv_src: torch.Tensor,
    kv_buf: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    rotary_dim: int,
) -> None:
    kv_buf.copy_(kv_src)
    cache.zero_()
    dsv4_kernel.compress_norm_rope_store_fallback(
        _FakeCompressedCache(cache),
        0,
        kv_buf,
        loc,
        positions=None if mode == "store_only" else positions,
        norm_weight=norm_weight if mode != "store_only" else None,
        rms_norm_eps=1e-6 if mode != "store_only" else None,
        rotary_dim=rotary_dim if mode == "norm_rope_store" else 0,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )


def _bench_case(
    *,
    tokens: int,
    dim: int,
    rotary_dim: int,
    mode: Mode,
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

    with _with_env("MINISGL_DSV4_SM80_COMPRESS_STORE", None):
        _run_once(
            mode=mode,
            kv_src=kv,
            kv_buf=fallback_kv,
            cache=fallback_cache,
            loc=out_loc,
            positions=positions,
            norm_weight=norm_weight,
            rotary_dim=rotary_dim,
        )
        expected_kv = fallback_kv.clone()
        expected_cache = fallback_cache.clone()
        fallback_ms = _time_cuda_wall(
            lambda: _run_once(
                mode=mode,
                kv_src=kv,
                kv_buf=fallback_kv,
                cache=fallback_cache,
                loc=out_loc,
                positions=positions,
                norm_weight=norm_weight,
                rotary_dim=rotary_dim,
            ),
            warmup=warmup,
            iters=iters,
        )

    with _with_env("MINISGL_DSV4_SM80_COMPRESS_STORE", "1"):
        _run_once(
            mode=mode,
            kv_src=kv,
            kv_buf=triton_kv,
            cache=triton_cache,
            loc=out_loc,
            positions=positions,
            norm_weight=norm_weight,
            rotary_dim=rotary_dim,
        )
        actual_kv = triton_kv.clone()
        actual_cache = triton_cache.clone()
        triton_ms = _time_cuda_wall(
            lambda: _run_once(
                mode=mode,
                kv_src=kv,
                kv_buf=triton_kv,
                cache=triton_cache,
                loc=out_loc,
                positions=positions,
                norm_weight=norm_weight,
                rotary_dim=rotary_dim,
            ),
            warmup=warmup,
            iters=iters,
        )

    kv_error = (actual_kv.float() - expected_kv.float()).abs()
    cache_error = (actual_cache.float() - expected_cache.float()).abs()
    return {
        "mode": mode,
        "tokens": tokens,
        "dim": dim,
        "rotary_dim": rotary_dim if mode == "norm_rope_store" else 0,
        "cache_rows": cache_rows,
        "fallback_wall_ms": fallback_ms,
        "triton_wall_ms": triton_ms,
        "speedup_vs_fallback": fallback_ms / triton_ms if triton_ms > 0 else None,
        "kv_max_abs_error": float(kv_error.max().item()) if kv_error.numel() else 0.0,
        "cache_max_abs_error": float(cache_error.max().item()) if cache_error.numel() else 0.0,
        "allclose_2e_2": bool(
            torch.allclose(actual_kv, expected_kv, atol=2e-2, rtol=2e-2)
            and torch.allclose(actual_cache, expected_cache, atol=2e-2, rtol=2e-2)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark DeepSeek V4 compressed bf16 norm/RoPE/cache store."
    )
    parser.add_argument("--tokens", default="1,8,64,512")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--modes", default="store_only,norm_store,norm_rope_store")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dsv4_compress_norm_rope_store_bf16_microbench.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")

    device = torch.device("cuda")
    started = time.time()
    results = []
    for mode in args.modes.split(","):
        if mode not in ("store_only", "norm_store", "norm_rope_store"):
            raise SystemExit(f"Unsupported mode: {mode}")
        for tokens in _parse_csv_ints(args.tokens):
            result = _bench_case(
                tokens=tokens,
                dim=args.dim,
                rotary_dim=args.rotary_dim,
                mode=mode,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            results.append(result)
            print(
                f"{mode} tokens={tokens}: fallback={result['fallback_wall_ms']:.3f} ms, "
                f"triton={result['triton_wall_ms']:.3f} ms, "
                f"speedup={result['speedup_vs_fallback']:.2f}x, "
                f"allclose={result['allclose_2e_2']}"
            )

    payload = {
        "benchmark": "deepseek_v4_compress_norm_rope_store_bf16",
        "started_at": started,
        "elapsed_s": time.time() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

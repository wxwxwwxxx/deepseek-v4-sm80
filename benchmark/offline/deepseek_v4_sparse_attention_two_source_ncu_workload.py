from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


def _make_case(
    *,
    total_candidates: int,
    tokens: int,
    heads: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(7000 + total_candidates)
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
        "q": q.contiguous(),
        "swa_cache": swa_cache.contiguous(),
        "compressed_cache": compressed_cache.contiguous(),
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


def _run_kernel(case: dict[str, torch.Tensor]) -> torch.Tensor:
    out = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
        case["q"],
        case["swa_cache"],
        case["swa_indices"],
        case["swa_lengths"],
        compressed_cache=case["compressed_cache"],
        compressed_indices=case["compressed_indices"],
        compressed_lengths=case["compressed_lengths"],
        softmax_scale=512**-0.5,
        attn_sink=case["attn_sink"],
    )
    if out is None:
        raise RuntimeError("dsv4_sparse_attention_two_source_bf16 returned None")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused NCU workload for DSV4 sm80 two-source sparse attention."
    )
    parser.add_argument("--candidates", type=int, default=640)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This workload requires an sm80 CUDA device.")

    os.environ["MINISGL_DSV4_SM80_SPARSE_ATTN_BF16"] = "1"
    device = torch.device("cuda")
    case = _make_case(
        total_candidates=args.candidates,
        tokens=args.tokens,
        heads=args.heads,
        device=device,
    )

    # Build/load the JIT module and run non-profiled warmup launches first.
    for _ in range(args.warmup):
        _run_kernel(case)
    torch.cuda.synchronize()

    out = None
    for _ in range(args.iters):
        out = _run_kernel(case)
    torch.cuda.synchronize()
    assert out is not None
    print(
        "profiled "
        f"candidates={args.candidates} tokens={args.tokens} heads={args.heads} "
        f"checksum={float(out.float().sum().item()):.6f}"
    )


if __name__ == "__main__":
    main()

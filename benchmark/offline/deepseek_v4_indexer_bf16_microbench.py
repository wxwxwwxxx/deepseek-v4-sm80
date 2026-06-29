from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from minisgl.kernel import deepseek_v4 as dsv4_kernel


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end) / iters)


def _assert_same_raw_sets(
    expected: dsv4_kernel.DSV4IndexerSelectOutput,
    actual: dsv4_kernel.DSV4IndexerSelectOutput,
) -> None:
    exp_raw = expected.topk.raw_indices.detach().cpu()
    act_raw = actual.topk.raw_indices.detach().cpu()
    if exp_raw.shape != act_raw.shape:
        raise AssertionError(f"raw shape mismatch: {exp_raw.shape} != {act_raw.shape}")
    for row in range(exp_raw.shape[0]):
        if sorted(exp_raw[row].tolist()) != sorted(act_raw[row].tolist()):
            raise AssertionError(f"raw top-k set mismatch in row {row}")


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
    return q, weights, cache, seq_lens, page_table


def run_case(
    batch: int,
    max_seq_len: int,
    *,
    num_heads: int,
    head_dim: int,
    width: int,
    iters: int,
    warmup: int,
) -> dict:
    page_size = 64
    q, weights, cache, seq_lens, page_table = _make_inputs(
        batch,
        max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
    )

    os.environ.pop("MINISGL_DSV4_SM80_INDEXER_BF16", None)
    os.environ.pop("MINISGL_DSV4_SM80_TOPK", None)
    expected = dsv4_kernel.indexer_select_bf16_fallback(
        q,
        weights,
        cache,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=4,
    )
    default_ms = _time_cuda(
        lambda: dsv4_kernel.indexer_select_bf16_fallback(
            q,
            weights,
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

    os.environ["MINISGL_DSV4_SM80_INDEXER_BF16"] = "1"
    os.environ["MINISGL_DSV4_SM80_TOPK"] = "1"
    actual = dsv4_kernel.indexer_select_bf16_fallback(
        q,
        weights,
        cache,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=4,
    )
    _assert_same_raw_sets(expected, actual)
    opt_in_ms = _time_cuda(
        lambda: dsv4_kernel.indexer_select_bf16_fallback(
            q,
            weights,
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
    os.environ.pop("MINISGL_DSV4_SM80_INDEXER_BF16", None)
    os.environ.pop("MINISGL_DSV4_SM80_TOPK", None)

    return {
        "batch": batch,
        "max_seq_len": max_seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "width": width,
        "page_size": page_size,
        "default_backend": expected.backend,
        "opt_in_backend": actual.backend,
        "default_ms": default_ms,
        "opt_in_ms": opt_in_ms,
        "speedup": default_ms / opt_in_ms if opt_in_ms > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dsv4_indexer_bf16_microbench.json"),
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    torch.manual_seed(20260629)
    cases = [
        run_case(1, 1024, num_heads=64, head_dim=128, width=512, iters=args.iters, warmup=args.warmup),
        run_case(4, 2048, num_heads=64, head_dim=128, width=512, iters=args.iters, warmup=args.warmup),
        run_case(16, 4096, num_heads=64, head_dim=128, width=512, iters=args.iters, warmup=args.warmup),
    ]
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

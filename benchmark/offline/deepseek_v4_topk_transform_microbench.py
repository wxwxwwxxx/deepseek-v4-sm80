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
    expected: dsv4_kernel.DSV4TopKTransformOutput,
    actual: dsv4_kernel.DSV4TopKTransformOutput,
) -> None:
    exp_raw = expected.raw_indices.detach().cpu()
    act_raw = actual.raw_indices.detach().cpu()
    if exp_raw.shape != act_raw.shape:
        raise AssertionError(f"raw shape mismatch: {exp_raw.shape} != {act_raw.shape}")
    for row in range(exp_raw.shape[0]):
        if sorted(exp_raw[row].tolist()) != sorted(act_raw[row].tolist()):
            raise AssertionError(f"raw top-k set mismatch in row {row}")
    valid = actual.page_indices >= 0
    expected_full = actual.page_indices.to(torch.long) * 4 + 3
    if not torch.equal(
        actual.full_indices,
        torch.where(valid, expected_full.to(torch.int32), torch.full_like(actual.full_indices, -1)),
    ):
        raise AssertionError("full_indices are not derived from page_indices")


def _make_inputs(batch: int, max_seq_len: int, page_size: int) -> tuple[torch.Tensor, ...]:
    scores = torch.randn(batch, max_seq_len, device="cuda", dtype=torch.float32)
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
    num_pages = (max_seq_len + page_size - 1) // page_size
    page_table = (
        torch.arange(batch * num_pages, device="cuda", dtype=torch.int32).reshape(batch, num_pages)
        + 1000
    )
    return scores, seq_lens, page_table


def run_case(batch: int, max_seq_len: int, width: int, *, iters: int, warmup: int) -> dict:
    page_size = 64
    scores, seq_lens, page_table = _make_inputs(batch, max_seq_len, page_size)

    os.environ.pop("MINISGL_DSV4_SM80_TOPK", None)
    expected = dsv4_kernel.topk_transform_512_full_fallback(
        scores,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=4,
    )
    default_ms = _time_cuda(
        lambda: dsv4_kernel.topk_transform_512_full_fallback(
            scores,
            seq_lens,
            page_table,
            page_size=page_size,
            width=width,
            ratio=4,
        ),
        warmup=warmup,
        iters=iters,
    )

    os.environ["MINISGL_DSV4_SM80_TOPK"] = "1"
    actual = dsv4_kernel.topk_transform_512_full_fallback(
        scores,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=4,
    )
    _assert_same_raw_sets(expected, actual)
    opt_in_ms = _time_cuda(
        lambda: dsv4_kernel.topk_transform_512_full_fallback(
            scores,
            seq_lens,
            page_table,
            page_size=page_size,
            width=width,
            ratio=4,
        ),
        warmup=warmup,
        iters=iters,
    )
    os.environ.pop("MINISGL_DSV4_SM80_TOPK", None)

    return {
        "batch": batch,
        "max_seq_len": max_seq_len,
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
        default=Path("/tmp/dsv4_topk_transform_full_microbench.json"),
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    torch.manual_seed(20260629)
    cases = [
        run_case(1, 1024, 512, iters=args.iters, warmup=args.warmup),
        run_case(4, 2048, 512, iters=args.iters, warmup=args.warmup),
        run_case(16, 4096, 512, iters=args.iters, warmup=args.warmup),
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

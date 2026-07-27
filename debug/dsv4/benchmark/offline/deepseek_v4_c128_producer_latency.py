#!/usr/bin/env python3
"""No-weight C128 producer latency probe for old and sequence-owned signatures."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
from pathlib import Path

import torch

from minisgl.kernel.triton.deepseek_v4 import c128_online_pool_and_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="1,8,128,256")
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("C128 producer latency requires CUDA")
    device = torch.device("cuda")
    signature = inspect.signature(c128_online_pool_and_update)
    sequence_owned = "raw_out_loc" not in signature.parameters
    generator = torch.Generator(device="cpu").manual_seed(161)
    ape = torch.randn(
        128, args.head_dim, generator=generator, dtype=torch.float32
    ).to(device)
    cases: dict[str, object] = {}

    for rows in (int(value) for value in args.rows.split(",")):
        projected = torch.randn(
            rows,
            2 * args.head_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        positions = torch.arange(rows, dtype=torch.int32, device=device)
        table_indices = torch.zeros(rows, dtype=torch.int32, device=device)
        if sequence_owned:
            state = torch.empty(
                128, 2 * args.head_dim, dtype=torch.float32, device=device
            )

            def invoke() -> None:
                result = c128_online_pool_and_update(
                    projected, state, ape, positions, table_indices
                )
                assert result is not None

        else:
            page_size = 256
            page_count = max((rows + page_size - 1) // page_size, 1)
            state = torch.empty(
                page_count * 128,
                2 * args.head_dim,
                dtype=torch.float32,
                device=device,
            )
            raw_out_loc = positions.clone()
            ctx_page_table = torch.arange(
                page_count, dtype=torch.int32, device=device
            ).unsqueeze(0)
            state_page_mapping = torch.arange(
                page_count, dtype=torch.int32, device=device
            )

            def invoke() -> None:
                result = c128_online_pool_and_update(
                    projected,
                    state,
                    ape,
                    positions,
                    table_indices,
                    raw_out_loc,
                    ctx_page_table,
                    state_page_mapping,
                    page_size=page_size,
                )
                assert result is not None

        for _ in range(args.warmup):
            invoke()
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                invoke()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0 / args.iterations)
        cases[str(rows)] = {
            "median_us": statistics.median(samples),
            "min_us": min(samples),
            "max_us": max(samples),
            "samples_us": samples,
        }

    report = {
        "ownership": "sequence" if sequence_owned else "page",
        "device": torch.cuda.get_device_name(device),
        "head_dim": args.head_dim,
        "warmup": args.warmup,
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

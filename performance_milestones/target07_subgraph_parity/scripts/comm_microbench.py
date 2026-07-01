#!/usr/bin/env python3
"""Torch distributed NCCL all-reduce/all-gather probes for DSV4 shapes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return ordered[idx]


def bench(fn: Callable[[], Any], warmup: int, repeat: int) -> dict[str, Any]:
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
        "warmup": warmup,
        "repeat": repeat,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": p90(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    warmup = 5 if args.quick else 20
    repeat = 20 if args.quick else 100
    shapes = [
        ("decode_hidden_bf16", [4, 4096], torch.bfloat16),
        ("decode_hidden_f32", [4, 4096], torch.float32),
        ("prefill_hidden_bf16", [4096, 4096], torch.bfloat16),
        ("prefill_hidden_f32", [4096, 4096], torch.float32),
        ("lm_head_logits_shard_bf16", [4, 129280 // max(world, 1)], torch.bfloat16),
    ]
    results = []
    for name, shape, dtype in shapes:
        x = torch.randn(*shape, device=device, dtype=dtype)

        def ar() -> torch.Tensor:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x

        ar_result = bench(ar, warmup, repeat)
        ar_result.update(
            {
                "name": f"all_reduce_{name}",
                "collective": "all_reduce_sum",
                "shape": shape,
                "dtype": str(dtype).replace("torch.", ""),
                "bytes_per_rank": x.numel() * x.element_size(),
            }
        )
        results.append(ar_result)

        if "logits" in name:
            gathered = [torch.empty_like(x) for _ in range(world)]

            def ag() -> list[torch.Tensor]:
                dist.all_gather(gathered, x)
                return gathered

            ag_result = bench(ag, warmup, repeat)
            ag_result.update(
                {
                    "name": f"all_gather_{name}",
                    "collective": "all_gather",
                    "shape": shape,
                    "dtype": str(dtype).replace("torch.", ""),
                    "bytes_per_rank": x.numel() * x.element_size(),
                }
            )
            results.append(ag_result)
        torch.cuda.empty_cache()

    output = {
        "suite": "comm_microbench",
        "rank": rank,
        "world_size": world,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "elapsed_s": None,
        "results": results,
    }
    started = time.time()
    dist.barrier()
    output["elapsed_s"] = time.time() - started
    if rank == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(out_path)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

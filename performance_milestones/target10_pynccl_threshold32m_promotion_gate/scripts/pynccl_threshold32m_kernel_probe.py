#!/usr/bin/env python3
"""Small PyNCCL threshold32m kernel/profile probe for TARGET 10.26."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from minisgl.distributed import set_tp_info
from minisgl.kernel import init_pynccl


THRESHOLD32M_BYTES = 32 * 1024 * 1024


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= dim
    return out


def _bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return _numel(shape) * torch.empty((), dtype=dtype).element_size()


def _sync(sync_group: object) -> None:
    torch.cuda.synchronize()
    dist.barrier(group=sync_group)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_tp_info(rank, world_size)
    gloo_group = dist.new_group(backend="gloo")
    comm = init_pynccl(
        tp_rank=rank,
        tp_size=world_size,
        tp_cpu_group=gloo_group,
        max_size_bytes=THRESHOLD32M_BYTES,
    )

    cases: list[dict[str, object]] = []
    expected_sum = world_size * (world_size + 1) / 2.0

    def run_all_reduce(label: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        size_bytes = _bytes(shape, dtype)
        tensor = torch.full(shape, float(rank + 1), dtype=dtype, device="cuda")
        _sync(gloo_group)
        for _ in range(args.warmup):
            comm.all_reduce(tensor, "sum")
        _sync(gloo_group)
        torch.cuda.nvtx.range_push(label)
        for _ in range(args.iterations):
            tensor.fill_(float(rank + 1))
            comm.all_reduce(tensor, "sum")
        torch.cuda.nvtx.range_pop()
        _sync(gloo_group)
        expected = torch.full_like(tensor, expected_sum)
        ok = bool(torch.allclose(tensor, expected, rtol=0.0, atol=0.0))
        cases.append(
            {
                "label": label,
                "op": "all_reduce",
                "shape": list(shape),
                "dtype": str(dtype).replace("torch.", ""),
                "input_bytes": size_bytes,
                "threshold_path": "symmetric" if size_bytes <= THRESHOLD32M_BYTES else "direct",
                "iterations": args.iterations,
                "correct": ok,
            }
        )

    def run_all_gather(label: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        src = torch.full(shape, float(rank), dtype=dtype, device="cuda")
        dst = torch.empty((shape[0] * world_size, *shape[1:]), dtype=dtype, device="cuda")
        _sync(gloo_group)
        for _ in range(args.warmup):
            comm.all_gather(dst, src)
        _sync(gloo_group)
        torch.cuda.nvtx.range_push(label)
        for _ in range(args.iterations):
            comm.all_gather(dst, src)
        torch.cuda.nvtx.range_pop()
        _sync(gloo_group)
        expected = torch.cat(
            [
                torch.full(shape, float(r), dtype=dtype, device="cuda")
                for r in range(world_size)
            ],
            dim=0,
        )
        ok = bool(torch.allclose(dst, expected, rtol=0.0, atol=0.0))
        cases.append(
            {
                "label": label,
                "op": "all_gather",
                "shape": list(shape),
                "output_shape": list(dst.shape),
                "dtype": str(dtype).replace("torch.", ""),
                "input_bytes": _bytes(shape, dtype),
                "output_bytes": int(dst.numel() * dst.element_size()),
                "threshold_path": "direct_output",
                "iterations": args.iterations,
                "correct": ok,
            }
        )

    run_all_reduce("pynccl_threshold32m_symm_all_reduce_2496x4096_bf16", (2496, 4096), torch.bfloat16)
    run_all_reduce("pynccl_threshold32m_direct_all_reduce_16384x4096_bf16", (16384, 4096), torch.bfloat16)
    run_all_gather("pynccl_threshold32m_all_gather_16x16160_fp32", (16, 16160), torch.float32)

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "world_size": world_size,
                    "threshold_bytes": THRESHOLD32M_BYTES,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "cases": cases,
                },
                indent=2,
            )
            + "\n"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

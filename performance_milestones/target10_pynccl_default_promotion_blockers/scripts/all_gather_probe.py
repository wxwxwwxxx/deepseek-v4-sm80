#!/usr/bin/env python3
"""TARGET 10.27 lm-head all-gather probe through DistributedCommunicator."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from minisgl.distributed import (
    DistributedCommunicator,
    enable_pynccl_distributed,
    get_tp_info,
    reset_communication_stats,
    set_tp_info,
    snapshot_communication_stats,
)


LABEL = "dsv4.lm_head_all_gather"


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.lower().replace(",", "x").split("x") if part)


def _sync(group: dist.ProcessGroup) -> None:
    torch.cuda.synchronize()
    dist.barrier(group=group)


def _timed_all_gather(
    comm: DistributedCommunicator,
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, float, float]:
    _sync(group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    y = comm.all_gather(x, label=LABEL)
    end.record()
    end.synchronize()
    torch.cuda.synchronize()
    host_ms = (time.perf_counter() - host_start) * 1000.0
    dist.barrier(group=group)
    return y, float(start.elapsed_time(end)), float(host_ms)


def _correct(y: torch.Tensor, shape: tuple[int, ...], world_size: int) -> bool:
    expected = torch.cat(
        [torch.full(shape, float(rank), dtype=y.dtype, device=y.device) for rank in range(world_size)],
        dim=0,
    )
    return bool(torch.allclose(y, expected, rtol=0.0, atol=0.0))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _run_eager_case(
    *,
    comm: DistributedCommunicator,
    shape: tuple[int, ...],
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    x = torch.full(shape, float(rank), dtype=torch.float32, device="cuda")
    first_y, first_event_ms, first_host_ms = _timed_all_gather(comm, x, group)
    first_correct = _correct(first_y, shape, world_size)

    for _ in range(warmup):
        _timed_all_gather(comm, x, group)

    event_ms: list[float] = []
    host_ms: list[float] = []
    y = first_y
    for _ in range(iterations):
        y, event_value, host_value = _timed_all_gather(comm, x, group)
        event_ms.append(event_value)
        host_ms.append(host_value)

    return {
        "shape": list(shape),
        "output_shape": list(y.shape),
        "dtype": "float32",
        "first_call_event_ms": first_event_ms,
        "first_call_host_sync_ms": first_host_ms,
        "first_call_correct": first_correct,
        "warmup": warmup,
        "iterations": iterations,
        "event_ms": _stats(event_ms),
        "host_sync_ms": _stats(host_ms),
        "last_correct": _correct(y, shape, world_size),
    }


def _run_graph_case(
    *,
    comm: DistributedCommunicator,
    shape: tuple[int, ...],
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    iterations: int,
) -> dict[str, Any]:
    x = torch.full(shape, float(rank), dtype=torch.float32, device="cuda")
    for _ in range(3):
        y = comm.all_gather(x, label=LABEL)
    _sync(group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = comm.all_gather(x, label=LABEL)
    _sync(group)

    event_ms: list[float] = []
    host_ms: list[float] = []
    for _ in range(iterations):
        _sync(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        torch.cuda.synchronize()
        host_ms.append(float((time.perf_counter() - host_start) * 1000.0))
        event_ms.append(float(start.elapsed_time(end)))
        dist.barrier(group=group)

    return {
        "shape": list(shape),
        "output_shape": list(y.shape),
        "dtype": "float32",
        "iterations": iterations,
        "event_ms": _stats(event_ms),
        "host_sync_ms": _stats(host_ms),
        "last_correct": _correct(y, shape, world_size),
    }


def _gather_rank_payload(payload: dict[str, Any], group: dist.ProcessGroup) -> list[dict[str, Any]]:
    world_size = dist.get_world_size(group=group)
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload, group=group)
    return list(gathered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "pynccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["16x16160", "8x16160", "4x16160", "2x16160", "1x16160"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--graph-iterations", type=int, default=20)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--pynccl-max-buffer-size", default="32M")
    args = parser.parse_args()

    if args.backend == "pynccl":
        os.environ.setdefault("MINISGL_PYNCCL_MAX_BUFFER_SIZE", args.pynccl_max_buffer_size)

    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_tp_info(rank, world_size)
    cpu_group = dist.new_group(backend="gloo")

    if args.backend == "pynccl":
        from minisgl.env import ENV

        enable_pynccl_distributed(
            get_tp_info(),
            cpu_group,
            max_bytes=int(ENV.PYNCCL_MAX_BUFFER_SIZE.value),
        )

    comm = DistributedCommunicator()
    reset_communication_stats()

    shapes = [_parse_shape(value) for value in args.shapes]
    eager = [
        _run_eager_case(
            comm=comm,
            shape=shape,
            rank=rank,
            world_size=world_size,
            group=cpu_group,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for shape in shapes
    ]
    graph: list[dict[str, Any]] = []
    if not args.skip_graph:
        graph = [
            _run_graph_case(
                comm=comm,
                shape=shape,
                rank=rank,
                world_size=world_size,
                group=cpu_group,
                iterations=args.graph_iterations,
            )
            for shape in shapes
        ]

    payload = {
        "rank": rank,
        "backend": args.backend,
        "eager": eager,
        "graph": graph,
        "communication_stats": snapshot_communication_stats(),
    }
    gathered = _gather_rank_payload(payload, cpu_group)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "backend": args.backend,
                    "world_size": world_size,
                    "label": LABEL,
                    "rank_payloads": gathered,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

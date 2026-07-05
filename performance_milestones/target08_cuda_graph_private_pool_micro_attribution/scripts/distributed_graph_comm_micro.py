#!/usr/bin/env python3
"""Tiny NCCL CUDA graph communication control for TARGET 08.32."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _mem_stats(device: torch.device, prefix: str) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        f"{prefix}_free_bytes": int(free),
        f"{prefix}_total_bytes": int(total),
        f"{prefix}_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        f"{prefix}_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _run_rank(args: argparse.Namespace) -> dict[str, Any]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = _dtype(args.dtype)
    tensor = torch.ones((int(args.elements),), device=device, dtype=dtype)
    explicit_bytes = int(tensor.numel() * tensor.element_size())
    dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = _mem_stats(device, "before")
    graph = torch.cuda.CUDAGraph()
    capture_error = None
    capture_elapsed = None
    try:
        start = time.perf_counter()
        with torch.cuda.graph(graph):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        capture_elapsed = time.perf_counter() - start
    except Exception as exc:
        capture_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
    after = _mem_stats(device, "after")
    replay_error = None
    replay_elapsed = None
    checksum = None
    if capture_error is None:
        try:
            start = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize(device)
            replay_elapsed = time.perf_counter() - start
            checksum = float(tensor[: min(tensor.numel(), 1024)].float().sum().item())
        except Exception as exc:
            replay_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
    return {
        "case": f"comm_all_reduce_{args.dtype}_{explicit_bytes // (1 << 20)}mib_tp{world}",
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "dtype": args.dtype,
        "elements": int(args.elements),
        "explicit_input_output_workspace_bytes": explicit_bytes,
        "capture_elapsed_s": capture_elapsed,
        "replay_elapsed_s": replay_elapsed,
        "capture_error": capture_error,
        "replay_error": replay_error,
        "checksum": checksum,
        **before,
        **after,
        "free_delta_bytes": int(before["before_free_bytes"] - after["after_free_bytes"]),
        "allocated_delta_bytes": int(after["after_allocated_bytes"] - before["before_allocated_bytes"]),
        "reserved_delta_bytes": int(after["after_reserved_bytes"] - before["before_reserved_bytes"]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    try:
        result = _run_rank(args)
        gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, result)
        if dist.get_rank() == 0:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "case": result["case"],
                "dtype": args.dtype,
                "elements": int(args.elements),
                "world_size": dist.get_world_size(),
                "ranks": gathered,
                "max_free_delta_bytes": max(int(r["free_delta_bytes"]) for r in gathered if r is not None),
                "max_allocated_delta_bytes": max(int(r["allocated_delta_bytes"]) for r in gathered if r is not None),
                "max_reserved_delta_bytes": max(int(r["reserved_delta_bytes"]) for r in gathered if r is not None),
                "errors": [
                    {
                        "rank": r["rank"],
                        "capture_error": r["capture_error"],
                        "replay_error": r["replay_error"],
                    }
                    for r in gathered
                    if r is not None and (r.get("capture_error") or r.get("replay_error"))
                ],
            }
            out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

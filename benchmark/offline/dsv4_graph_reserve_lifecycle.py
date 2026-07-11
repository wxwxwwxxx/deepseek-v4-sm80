#!/usr/bin/env python3
"""No-weight CUDA graph storage lifecycle harness for TARGET 12.603."""

from __future__ import annotations

import argparse
import gc
import json
import time
import weakref
from pathlib import Path

import torch


def _range(tensor: torch.Tensor) -> tuple[int, int]:
    begin = int(tensor.untyped_storage().data_ptr())
    return begin, begin + int(tensor.untyped_storage().nbytes())


def run(device: torch.device) -> dict[str, object]:
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_before, total = torch.cuda.mem_get_info(device)

    # These stand in for the exact owner classes a future full-model temporary
    # profile must construct.  No model weight or Marlin cache is involved.
    temporary: dict[str, torch.Tensor] = {
        "kv_component": torch.full((2, 256, 128), 0.25, device=device),
        "request_table": torch.zeros((9, 4096), dtype=torch.int32, device=device),
        "page_table": torch.zeros((9, 16), dtype=torch.int32, device=device),
        "attention_metadata": torch.zeros((64, 512), dtype=torch.int32, device=device),
        "moe_route_ids": torch.arange(64 * 8, dtype=torch.int32, device=device).view(64, 8),
        "moe_route_weights": torch.ones((64, 8), dtype=torch.float32, device=device),
        "num_token_non_padded": torch.tensor([57], dtype=torch.int32, device=device),
        "moe_finalize_output": torch.empty((64, 8), dtype=torch.float32, device=device),
    }
    owner_ranges = {name: _range(tensor) for name, tensor in temporary.items()}
    weak_tensors = {name: weakref.ref(tensor) for name, tensor in temporary.items()}

    # Simulate model/backend/global-context attachment, then prove every edge is
    # detached before the storage owners are released.
    holder: dict[str, object | None] = {
        "model_kv": temporary["kv_component"],
        "backend_metadata": temporary["attention_metadata"],
        "global_page_table": temporary["page_table"],
    }
    graph = torch.cuda.CUDAGraph()
    route_rows = torch.arange(64, dtype=torch.int32, device=device).view(64, 1)
    weak_route_rows = weakref.ref(route_rows)
    owner_ranges["moe_route_rows"] = _range(route_rows)

    def body() -> None:
        live = route_rows < temporary["num_token_non_padded"]
        masked = torch.where(live, temporary["moe_route_weights"], 0.0)
        temporary["moe_finalize_output"].copy_(masked)

    with torch.cuda.stream(stream):
        body()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            body()
    graph.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(temporary["moe_finalize_output"][57:], torch.zeros_like(temporary["moe_finalize_output"][57:])):
        raise RuntimeError("captured repaired-MoE finalize sentinel failed")
    graph_pool = graph.pool()
    graph_ref = weakref.ref(graph)
    free_after_capture, _ = torch.cuda.mem_get_info(device)

    # Required cleanup order: sync -> graph exec/wrapper -> attachment edges ->
    # pool handle -> temporary owners -> allocator cache -> reuse probe.
    torch.cuda.synchronize(device)
    del graph
    gc.collect()
    for key in tuple(holder):
        holder[key] = None
    del graph_pool
    route_rows = None  # type: ignore[assignment]
    temporary.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after_cleanup, _ = torch.cuda.mem_get_info(device)

    stale_tensors = [name for name, ref in weak_tensors.items() if ref() is not None]
    if weak_route_rows() is not None:
        stale_tensors.append("moe_route_rows")
    if graph_ref() is not None:
        raise RuntimeError("temporary CUDA graph wrapper survived cleanup")
    if stale_tensors:
        raise RuntimeError(f"temporary storage references survived cleanup: {stale_tensors}")
    if any(value is not None for value in holder.values()):
        raise RuntimeError("model/backend/global-context attachment survived cleanup")

    # Allocator reuse is allowed; a surviving Python tensor/graph edge is not.
    # Writing and synchronizing a similarly sized probe detects stale graph work
    # or an illegal old-address access more strongly than mem_get_info alone.
    probe = torch.full((2, 256, 128), 7.0, device=device)
    probe_range = _range(probe)
    probe.sum().item()
    torch.cuda.synchronize(device)
    reused_owner_ranges = [
        name
        for name, (begin, end) in owner_ranges.items()
        if max(begin, probe_range[0]) < min(end, probe_range[1])
    ]
    del probe
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return {
        "status": "pass",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "free_before_bytes": int(free_before),
        "free_after_capture_bytes": int(free_after_capture),
        "capture_physical_delta_bytes": int(free_before - free_after_capture),
        "free_after_cleanup_bytes": int(free_after_cleanup),
        "total_memory_bytes": int(total),
        "old_owner_ranges": {name: list(bounds) for name, bounds in owner_ranges.items()},
        "reuse_probe_range": list(probe_range),
        "reused_owner_ranges": reused_owner_ranges,
        "stale_tensor_refs": stale_tensors,
        "graph_wrapper_alive": graph_ref() is not None,
        "cleanup_order": [
            "synchronize_profiling_work",
            "destroy_graph_exec_and_wrapper",
            "detach_model_backend_global_context",
            "release_graph_pool_handle",
            "release_kv_component_page_table_metadata_storage",
            "collect_and_empty_allocator_cache",
            "allocator_reuse_write_and_sync_probe",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(torch.device(f"cuda:{args.device}"))
    result["elapsed_s"] = time.perf_counter() - started
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()

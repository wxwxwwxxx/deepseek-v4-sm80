#!/usr/bin/env python3
"""TARGET 16.15 production-shape C128 owner-transition timing and census."""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
from minisgl.kvcache.deepseek_v4_pool import (
    DSV4CompressStatePool,
    DeepSeekV4KVCache,
)


SLOT_COUNTS = (1, 4, 16, 64, 128)
LAYERS = 20
HEAD_DIM = 512
RING_SIZE = 128
MAX_RUNNING_REQ = 128


def _make_lifecycle_only_pool(device: torch.device) -> DeepSeekV4KVCache:
    """Allocate only the production C128 state owned by the lifecycle methods."""

    pool = DeepSeekV4KVCache.__new__(DeepSeekV4KVCache)
    pool._c128_layer_count = LAYERS
    pool._max_running_req = MAX_RUNNING_REQ
    pool._c128_sequence_slots = MAX_RUNNING_REQ + 1
    pool._c128_sequence_owners = [None] * MAX_RUNNING_REQ
    pool._compress_state_pools = [
        DSV4CompressStatePool(
            size=(MAX_RUNNING_REQ + 1) * RING_SIZE,
            overlap=False,
            head_dim=HEAD_DIM,
            ratio=128,
            dtype=torch.float32,
            device=device,
        )
        for _ in range(LAYERS)
    ]
    return pool


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _measure_stage(
    device: torch.device,
    action,
) -> tuple[float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start_event.record()
    host_start = time.perf_counter_ns()
    action()
    host_us = (time.perf_counter_ns() - host_start) / 1_000.0
    end_event.record()
    end_event.synchronize()
    return host_us, float(start_event.elapsed_time(end_event) * 1_000.0)


def _profile_one_transition(
    pool: DeepSeekV4KVCache,
    device: torch.device,
    generation: int,
) -> dict[str, object]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as prof:
        pool.acquire_c128_sequence_slot(0, generation)
        pool.release_c128_sequence_slot(0, generation)
        torch.cuda.synchronize(device)

    key_events = []
    for event in prof.key_averages():
        name = str(event.key)
        if (
            "zero_" in name
            or "fill_" in name
            or "fill_kernel" in name
            or "elementwise_kernel" in name
        ):
            key_events.append(
                {
                    "name": name,
                    "count": int(event.count),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "device_time_total_us": float(event.device_time_total),
                }
            )
    key_events.sort(key=lambda item: (str(item["name"]), int(item["count"])))
    acquire_source = inspect.getsource(DeepSeekV4KVCache.acquire_c128_sequence_slot)
    release_source = inspect.getsource(DeepSeekV4KVCache.release_c128_sequence_slot)
    transition_clear_present = (
        "clear" in acquire_source.lower() or "clear" in release_source.lower()
    )
    operations_per_transition = 2 * LAYERS if transition_clear_present else 0
    return {
        "scope": "one acquire plus one release for one slot",
        "matching_key_averages": key_events,
        "owner_transition_source_clear_present": transition_clear_present,
        "source_expected_per_transition": {
            "layer_pool_iterations": LAYERS if transition_clear_present else 0,
            "aten_zero_calls": LAYERS if transition_clear_present else 0,
            "aten_fill_calls": LAYERS if transition_clear_present else 0,
            "state_clear_fill_operations": operations_per_transition,
        },
        "source_expected_acquire_plus_release": {
            "layer_pool_iterations": 2 * LAYERS if transition_clear_present else 0,
            "aten_zero_calls": 2 * LAYERS if transition_clear_present else 0,
            "aten_fill_calls": 2 * LAYERS if transition_clear_present else 0,
            "state_clear_fill_operations": 2 * operations_per_transition,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("C128 lifecycle timing requires CUDA")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    device = torch.device("cuda")
    pool = _make_lifecycle_only_pool(device)
    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "production_dimensions": {
            "c128_layers": LAYERS,
            "head_dim": HEAD_DIM,
            "ring_rows": RING_SIZE,
            "max_running_req": MAX_RUNNING_REQ,
            "physical_slots_including_graph_dummy": MAX_RUNNING_REQ + 1,
            "allocated_state_bytes": int(pool.c128_sequence_state_bytes),
        },
        "repeats": int(args.repeats),
        "slot_counts": {},
    }

    generation = 1
    acquire_source = inspect.getsource(DeepSeekV4KVCache.acquire_c128_sequence_slot)
    release_source = inspect.getsource(DeepSeekV4KVCache.release_c128_sequence_slot)
    transition_clear_present = (
        "clear" in acquire_source.lower() or "clear" in release_source.lower()
    )
    operations_per_transition = 2 * LAYERS if transition_clear_present else 0
    for slot_count in SLOT_COUNTS:
        acquire_host_us: list[float] = []
        acquire_gpu_us: list[float] = []
        release_host_us: list[float] = []
        release_gpu_us: list[float] = []
        for _ in range(args.repeats):
            current_generation = generation
            generation += slot_count

            host_us, gpu_us = _measure_stage(
                device,
                lambda: [
                    pool.acquire_c128_sequence_slot(
                        slot,
                        current_generation + slot,
                    )
                    for slot in range(slot_count)
                ],
            )
            acquire_host_us.append(host_us)
            acquire_gpu_us.append(gpu_us)

            host_us, gpu_us = _measure_stage(
                device,
                lambda: [
                    pool.release_c128_sequence_slot(
                        slot,
                        current_generation + slot,
                    )
                    for slot in range(slot_count)
                ],
            )
            release_host_us.append(host_us)
            release_gpu_us.append(gpu_us)

        report["slot_counts"][str(slot_count)] = {
            "acquire": {
                "host_enqueue_us": _percentiles(acquire_host_us),
                "cuda_event_us": _percentiles(acquire_gpu_us),
                "source_expected_clear_fill_operations": (
                    operations_per_transition * slot_count
                ),
            },
            "release": {
                "host_enqueue_us": _percentiles(release_host_us),
                "cuda_event_us": _percentiles(release_gpu_us),
                "source_expected_clear_fill_operations": (
                    operations_per_transition * slot_count
                ),
            },
            "acquire_plus_release": {
                "host_enqueue_us": _percentiles(
                    [
                        acquire + release
                        for acquire, release in zip(acquire_host_us, release_host_us)
                    ]
                ),
                "cuda_event_us": _percentiles(
                    [
                        acquire + release
                        for acquire, release in zip(acquire_gpu_us, release_gpu_us)
                    ]
                ),
                "source_expected_clear_fill_operations": (
                    2 * operations_per_transition * slot_count
                ),
            },
        }

    report["profiler_census"] = _profile_one_transition(pool, device, generation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

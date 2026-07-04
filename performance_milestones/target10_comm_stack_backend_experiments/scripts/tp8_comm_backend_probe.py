#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from minisgl.distributed import reset_communication_stats, set_tp_info, snapshot_communication_stats
from minisgl.distributed.impl import (
    DistributedCommunicator,
    PyNCCLDistributedImpl,
    TorchDistributedImpl,
)
from minisgl.kernel import init_pynccl


HIDDEN_SIZE = 4096
LM_HEAD_SHARD = 16160
NUM_LAYERS = 43


@dataclass(frozen=True)
class OwnerCase:
    scenario: str
    owner: str
    op: str
    dtype: str
    shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class TraceCase:
    scenario: str
    hidden_rows: int
    lm_batch: int
    forward_repeats: int


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


OWNER_CASES = [
    OwnerCase(
        "historical_4096_128_bs4",
        "dsv4.attn.wo_b.row_parallel_projection_all_reduce",
        "all_reduce",
        "bf16",
        (16384, HIDDEN_SIZE),
    ),
    OwnerCase(
        "historical_4096_128_bs4",
        "dsv4.v1_moe_reduce_once_all_reduce",
        "all_reduce",
        "bf16",
        (16384, HIDDEN_SIZE),
    ),
    OwnerCase(
        "historical_4096_128_bs4",
        "dsv4.embedding_all_reduce",
        "all_reduce",
        "bf16",
        (16384, HIDDEN_SIZE),
    ),
    OwnerCase(
        "historical_4096_128_bs4",
        "dsv4.lm_head_all_gather",
        "all_gather",
        "fp32",
        (4, LM_HEAD_SHARD),
        (32, LM_HEAD_SHARD),
    ),
    OwnerCase(
        "serving_mixed_112req_wave16",
        "dsv4.attn.wo_b.row_parallel_projection_all_reduce",
        "all_reduce",
        "bf16",
        (2496, HIDDEN_SIZE),
    ),
    OwnerCase(
        "serving_mixed_112req_wave16",
        "dsv4.v1_moe_reduce_once_all_reduce",
        "all_reduce",
        "bf16",
        (2496, HIDDEN_SIZE),
    ),
    OwnerCase(
        "serving_mixed_112req_wave16",
        "dsv4.embedding_all_reduce",
        "all_reduce",
        "bf16",
        (2496, HIDDEN_SIZE),
    ),
    OwnerCase(
        "serving_mixed_112req_wave16",
        "dsv4.lm_head_all_gather",
        "all_gather",
        "fp32",
        (16, LM_HEAD_SHARD),
        (128, LM_HEAD_SHARD),
    ),
]


TRACE_CASES = [
    TraceCase("historical_4096_128_bs4", hidden_rows=16384, lm_batch=4, forward_repeats=16),
    TraceCase("serving_mixed_112req_wave16", hidden_rows=2496, lm_batch=16, forward_repeats=56),
]


@dataclass
class Backend:
    name: str
    pynccl_comm: object | None


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float32:
        return "float32"
    return str(dtype).removeprefix("torch.")


def _shape_list(shape: tuple[int, ...]) -> list[int]:
    return [int(dim) for dim in shape]


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return value


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def _summary_us(values: list[float]) -> dict[str, float]:
    return {
        "median_us": float(statistics.median(values)),
        "p95_us": _percentile(values, 95.0),
        "mean_us": float(statistics.fmean(values)),
        "min_us": float(min(values)),
        "max_us": float(max(values)),
    }


def _time_cuda(fn: Callable[[], None], *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end) * 1000.0))
    torch.cuda.synchronize()
    return times


def _all_reduce_fn(backend: Backend, tensor: torch.Tensor) -> Callable[[], None]:
    if backend.name == "torch_nccl":
        return lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    assert backend.pynccl_comm is not None
    return lambda: backend.pynccl_comm.all_reduce(tensor, "sum")


def _all_gather_fn(
    backend: Backend,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> Callable[[], None]:
    if backend.name == "torch_nccl":
        return lambda: dist.all_gather_into_tensor(dst, src)
    assert backend.pynccl_comm is not None
    return lambda: backend.pynccl_comm.all_gather(dst, src)


def _bench_copy(shape: tuple[int, ...], dtype: torch.dtype, warmup: int, iterations: int) -> dict:
    src = torch.zeros(shape, dtype=dtype, device="cuda")
    dst = torch.empty_like(src)
    times = _time_cuda(lambda: dst.copy_(src), warmup=warmup, iterations=iterations)
    summary = _summary_us(times)
    bytes_value = src.numel() * src.element_size()
    summary.update(
        {
            "shape": _shape_list(shape),
            "dtype": _dtype_name(dtype),
            "bytes": int(bytes_value),
            "bandwidth_gb_s": float(bytes_value / (summary["median_us"] * 1e-6) / 1e9),
        }
    )
    return summary


def _graph_probe(fn: Callable[[], None], *, replays: int) -> dict:
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed_us = float(start.elapsed_time(end) * 1000.0)
        return {
            "ok": True,
            "error": None,
            "replays": int(replays),
            "median_replay_us": elapsed_us / max(replays, 1),
        }
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "replays": 0}


def _correctness_all_reduce(
    backend: Backend,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> dict:
    x = torch.full(shape, float(_rank() + 1), dtype=dtype, device="cuda")
    ref = x.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    _all_reduce_fn(backend, x)()
    torch.cuda.synchronize()
    diff = (x.float() - ref.float()).abs()
    return {
        "ok": bool(torch.allclose(x.float(), ref.float(), rtol=0, atol=0)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _correctness_all_gather(
    backend: Backend,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> dict:
    src = torch.full(shape, float(_rank() + 1), dtype=dtype, device="cuda")
    dst_shape = (shape[0] * _world_size(),) + shape[1:]
    got = torch.empty(dst_shape, dtype=dtype, device="cuda")
    ref = torch.empty_like(got)
    dist.all_gather_into_tensor(ref, src)
    _all_gather_fn(backend, src, got)()
    torch.cuda.synchronize()
    diff = (got.float() - ref.float()).abs()
    return {
        "ok": bool(torch.allclose(got.float(), ref.float(), rtol=0, atol=0)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _bench_collective(
    backend: Backend,
    case: OwnerCase,
    *,
    warmup: int,
    iterations: int,
    graph_replays: int,
    symm_max_bytes: int,
    copy_cache: dict[tuple[tuple[int, ...], str], dict],
) -> dict:
    dtype = DTYPES[case.dtype]
    input_bytes = _numel(case.shape) * torch.empty((), dtype=dtype).element_size()
    output_shape = case.output_shape or case.shape
    output_bytes = _numel(output_shape) * torch.empty((), dtype=dtype).element_size()

    if case.op == "all_reduce":
        tensor = torch.zeros(case.shape, dtype=dtype, device="cuda")
        fn = _all_reduce_fn(backend, tensor)
        correctness = _correctness_all_reduce(backend, case.shape, dtype)
        measured_bytes = input_bytes
    elif case.op == "all_gather":
        src = torch.zeros(case.shape, dtype=dtype, device="cuda")
        dst = torch.empty(output_shape, dtype=dtype, device="cuda")
        fn = _all_gather_fn(backend, src, dst)
        correctness = _correctness_all_gather(backend, case.shape, dtype)
        measured_bytes = output_bytes
    else:
        raise ValueError(f"unsupported op: {case.op}")

    times = _time_cuda(fn, warmup=warmup, iterations=iterations)
    summary = _summary_us(times)
    graph = _graph_probe(fn, replays=graph_replays)

    copy_bytes_per_call = 0
    copy_time_us_per_call = 0.0
    if (
        backend.name == "pynccl_symmetric"
        and case.op == "all_reduce"
        and input_bytes <= symm_max_bytes
    ):
        copy_key = (case.shape, case.dtype)
        if copy_key not in copy_cache:
            copy_cache[copy_key] = _bench_copy(case.shape, dtype, warmup=warmup, iterations=iterations)
        copy_bytes_per_call = 2 * input_bytes
        copy_time_us_per_call = 2.0 * float(copy_cache[copy_key]["median_us"])

    return {
        "scenario": case.scenario,
        "owner": case.owner,
        "op": case.op,
        "dtype": _dtype_name(dtype),
        "shape": _shape_list(case.shape),
        "output_shape": _shape_list(output_shape),
        "backend": backend.name,
        "input_bytes": int(input_bytes),
        "output_bytes": int(output_bytes),
        "message_bytes_for_bw": int(measured_bytes),
        "median_us": summary["median_us"],
        "p95_us": summary["p95_us"],
        "mean_us": summary["mean_us"],
        "achieved_gb_s": float(measured_bytes / (summary["median_us"] * 1e-6) / 1e9),
        "copy_bytes_per_call": int(copy_bytes_per_call),
        "copy_time_us_per_call_estimate": float(copy_time_us_per_call),
        "correctness": correctness,
        "graph_capture": graph,
    }


def _trace_body(backend: Backend, hidden: torch.Tensor, lm_src: torch.Tensor, lm_dst: torch.Tensor):
    all_reduce = _all_reduce_fn(backend, hidden)
    all_gather = _all_gather_fn(backend, lm_src, lm_dst)

    def body() -> None:
        all_reduce()
        for _ in range(NUM_LAYERS):
            all_reduce()
            all_reduce()
        all_gather()

    return body


def _bench_trace(
    backend: Backend,
    trace: TraceCase,
    *,
    repeats: int,
    symm_max_bytes: int,
) -> dict:
    hidden = torch.zeros((trace.hidden_rows, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
    lm_src = torch.zeros((trace.lm_batch, LM_HEAD_SHARD), dtype=torch.float32, device="cuda")
    lm_dst = torch.empty((trace.lm_batch * _world_size(), LM_HEAD_SHARD), dtype=torch.float32, device="cuda")
    body = _trace_body(backend, hidden, lm_src, lm_dst)

    def eager_once() -> None:
        for _ in range(trace.forward_repeats):
            body()

    eager_times = _time_cuda(eager_once, warmup=1, iterations=repeats)
    eager_summary = _summary_us(eager_times)

    graph_result: dict
    try:
        for _ in range(2):
            body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize()

        graph_times: list[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(trace.forward_repeats):
                graph.replay()
            end.record()
            end.synchronize()
            graph_times.append(float(start.elapsed_time(end) * 1000.0))
        graph_summary = _summary_us(graph_times)
        graph_result = {"ok": True, "error": None, **graph_summary}
    except Exception as exc:  # noqa: BLE001
        torch.cuda.synchronize()
        graph_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    hidden_bytes = trace.hidden_rows * HIDDEN_SIZE * torch.bfloat16.itemsize
    all_reduce_calls = trace.forward_repeats * (1 + 2 * NUM_LAYERS)
    all_gather_calls = trace.forward_repeats
    copy_bytes = 0
    if backend.name == "pynccl_symmetric" and hidden_bytes <= symm_max_bytes:
        copy_bytes = all_reduce_calls * 2 * hidden_bytes

    return {
        "scenario": trace.scenario,
        "backend": backend.name,
        "forward_repeats": int(trace.forward_repeats),
        "body_all_reduce_calls": int(1 + 2 * NUM_LAYERS),
        "body_all_gather_calls": 1,
        "total_all_reduce_calls": int(all_reduce_calls),
        "total_all_gather_calls": int(all_gather_calls),
        "hidden_shape": [trace.hidden_rows, HIDDEN_SIZE],
        "lm_head_shape": [trace.lm_batch, LM_HEAD_SHARD],
        "eager": eager_summary,
        "graph_replay": graph_result,
        "symm_copy_bytes_total": int(copy_bytes),
    }


def _partial_runtime_probe(backends: list[Backend]) -> list[dict]:
    results = []
    case = OwnerCase(
        "partial_runtime_probe",
        "dsv4.v1_moe_reduce_once_all_reduce",
        "all_reduce",
        "bf16",
        (1024, HIDDEN_SIZE),
    )
    gather_case = OwnerCase(
        "partial_runtime_probe",
        "dsv4.lm_head_all_gather",
        "all_gather",
        "fp32",
        (4, LM_HEAD_SHARD),
        (4 * _world_size(), LM_HEAD_SHARD),
    )
    for backend in backends:
        if backend.name == "torch_nccl":
            DistributedCommunicator.plugins = [TorchDistributedImpl()]
        else:
            assert backend.pynccl_comm is not None
            DistributedCommunicator.plugins = [PyNCCLDistributedImpl(backend.pynccl_comm)]
        reset_communication_stats()
        comm = DistributedCommunicator()
        dtype = DTYPES[case.dtype]

        x = torch.full(case.shape, float(_rank() + 1), dtype=dtype, device="cuda")
        ref = x.clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        y = comm.all_reduce(x, label=case.owner)
        torch.cuda.synchronize()
        ar_ok = bool(torch.allclose(y.float(), ref.float(), rtol=0, atol=0))

        src = torch.full(gather_case.shape, float(_rank() + 1), dtype=torch.float32, device="cuda")
        ref_gather = torch.empty(gather_case.output_shape, dtype=torch.float32, device="cuda")
        dist.all_gather_into_tensor(ref_gather, src)
        got_gather = comm.all_gather(src, label=gather_case.owner)
        torch.cuda.synchronize()
        ag_ok = bool(torch.allclose(got_gather, ref_gather, rtol=0, atol=0))

        zero = torch.zeros(case.shape, dtype=dtype, device="cuda")
        graph = _graph_probe(lambda: comm.all_reduce(zero, label=case.owner), replays=8)
        stats = snapshot_communication_stats()
        results.append(
            {
                "backend": backend.name,
                "all_reduce_correct": ar_ok,
                "all_gather_correct": ag_ok,
                "graph_capture": graph,
                "stats": stats,
            }
        )
    DistributedCommunicator.plugins = [TorchDistributedImpl()]
    return results


def _p2p_table() -> dict:
    count = torch.cuda.device_count()
    matrix = []
    for src in range(count):
        row = []
        for dst in range(count):
            row.append(bool(src == dst or torch.cuda.can_device_access_peer(src, dst)))
        matrix.append(row)
    return {
        "device_count": int(count),
        "names": [torch.cuda.get_device_properties(i).name for i in range(count)],
        "can_device_access_peer": matrix,
        "all_pairs_accessible": bool(all(matrix[i][j] for i in range(count) for j in range(count))),
        "cuda_ipc_runtime_probe": "not_run",
    }


def _init_distributed() -> tuple[int, int, object]:
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_tp_info(rank, world_size)
    gloo_group = dist.new_group(backend="gloo")
    return rank, world_size, gloo_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trace-iterations", type=int, default=3)
    parser.add_argument("--graph-replays", type=int, default=16)
    args = parser.parse_args()

    rank, world_size, gloo_group = _init_distributed()
    if world_size != 8 and rank == 0:
        print(f"warning: intended TP8 probe, got world_size={world_size}", flush=True)

    max_hidden_bytes = max(
        _numel(case.shape) * torch.empty((), dtype=DTYPES[case.dtype]).element_size()
        for case in OWNER_CASES
        if case.op == "all_reduce"
    )
    direct_comm = init_pynccl(
        tp_rank=rank,
        tp_size=world_size,
        tp_cpu_group=gloo_group,
        max_size_bytes=0,
    )
    symm_comm = init_pynccl(
        tp_rank=rank,
        tp_size=world_size,
        tp_cpu_group=gloo_group,
        max_size_bytes=max_hidden_bytes,
    )
    backends = [
        Backend("torch_nccl", None),
        Backend("pynccl_direct", direct_comm),
        Backend("pynccl_symmetric", symm_comm),
    ]

    started = time.time()
    copy_cache: dict[tuple[tuple[int, ...], str], dict] = {}
    microbench: list[dict] = []
    bench_cache: dict[tuple[str, str, tuple[int, ...], str], dict] = {}
    for backend in backends:
        for case in OWNER_CASES:
            key = (backend.name, case.op, case.shape, case.dtype)
            if key not in bench_cache:
                bench_cache[key] = _bench_collective(
                    backend,
                    case,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    graph_replays=args.graph_replays,
                    symm_max_bytes=max_hidden_bytes,
                    copy_cache=copy_cache,
                )
            row = dict(bench_cache[key])
            row.update(
                {
                    "scenario": case.scenario,
                    "owner": case.owner,
                    "output_shape": _shape_list(case.output_shape or case.shape),
                }
            )
            microbench.append(row)

    traces: list[dict] = []
    for backend in backends:
        for trace in TRACE_CASES:
            traces.append(
                _bench_trace(
                    backend,
                    trace,
                    repeats=args.trace_iterations,
                    symm_max_bytes=max_hidden_bytes,
                )
            )

    partial_probe = _partial_runtime_probe(backends)
    p2p = _p2p_table() if rank == 0 else None

    payload = {
        "metadata": {
            "world_size": int(world_size),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_properties(torch.cuda.current_device()).name,
            "symm_max_bytes": int(max_hidden_bytes),
            "warmup": int(args.warmup),
            "iterations": int(args.iterations),
            "trace_iterations": int(args.trace_iterations),
            "graph_replays": int(args.graph_replays),
            "elapsed_seconds": float(time.time() - started),
        },
        "owner_cases": [asdict(case) for case in OWNER_CASES],
        "trace_cases": [asdict(case) for case in TRACE_CASES],
        "microbench": microbench,
        "d2d_copy_microbench": list(copy_cache.values()),
        "trace_replay": traces,
        "partial_runtime_probe": partial_probe,
        "p2p": p2p,
    }

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "elapsed_seconds": payload["metadata"]["elapsed_seconds"]}))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

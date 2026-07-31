#!/usr/bin/env python3
"""TP8 partial-model timing for discarded intermediate-prefill output work."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.distributed import (  # noqa: E402
    DistributedInfo,
    enable_pynccl_distributed,
    set_tp_info,
)
from minisgl.engine.sample import BatchSamplingArgs, ReasoningSampler  # noqa: E402
from minisgl.models.deepseek_v4 import DSV4VocabParallelEmbedding  # noqa: E402
from minisgl.reasoning import ReasoningState, resolve_reasoning_token_ids  # noqa: E402
from minisgl.utils import cached_load_hf_config, load_tokenizer  # noqa: E402


def _pctl(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p10_ms": _pctl(values, 0.10),
        "p90_ms": _pctl(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _time_cuda(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_ms.append(float(start.elapsed_time(end)))
        del result
    return {
        "gpu": _distribution(gpu_ms),
        "wall_complete": _distribution(wall_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--rows", default="1,4,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise SystemExit("TARGET 19 output-boundary microbench requires TP8")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise SystemExit("TARGET 19 output-boundary microbench requires sm80")

    dist.init_process_group(backend="gloo")
    tp = DistributedInfo(rank=rank, size=world_size)
    set_tp_info(rank=rank, size=world_size)
    enable_pynccl_distributed(tp, dist.group.WORLD, max_bytes=64 * 1024 * 1024)

    hf_config = cached_load_hf_config(args.model_path)
    hidden_size = int(hf_config.hidden_size)
    vocab_size = int(hf_config.vocab_size)
    with torch.device(device):
        lm_head = DSV4VocabParallelEmbedding(vocab_size, hidden_size)
    lm_head.weight.normal_(mean=0.0, std=0.01)
    reasoning_ids = resolve_reasoning_token_ids(load_tokenizer(args.model_path))
    sampler = ReasoningSampler(device, vocab_size, reasoning_ids)
    sampling_args = BatchSamplingArgs(temperatures=None)
    hidden = torch.randn(
        8192,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    rows_list = [int(value) for value in args.rows.split(",") if value]
    cases: list[dict[str, Any]] = []
    for rows in rows_list:
        last_indices = torch.arange(
            8192 // rows - 1,
            8192,
            8192 // rows,
            dtype=torch.long,
            device=device,
        )
        batch = SimpleNamespace(
            reasoning_states=torch.full(
                (rows,),
                int(ReasoningState.THINKING),
                dtype=torch.int32,
                device=device,
            ),
            is_decode=False,
            input_ids=None,
        )

        def gather_hidden():
            return hidden[last_indices].contiguous()

        selected = gather_hidden()

        def project_logits():
            return lm_head.linear(selected)

        logits = project_logits()

        def sample_logits():
            return sampler.sample(logits.clone(), sampling_args, batch)

        next_tokens = sample_logits()

        def copy_tokens():
            cpu = next_tokens.to("cpu", non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            done.synchronize()
            return cpu

        def current_discarded_output_path():
            selected_now = gather_hidden()
            logits_now = lm_head.linear(selected_now)
            tokens_now = sampler.sample(logits_now, sampling_args, batch)
            cpu = tokens_now.to("cpu", non_blocking=True)
            done = torch.cuda.Event()
            done.record()
            done.synchronize()
            return cpu

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        persistent_before = int(torch.cuda.memory_allocated(device))
        combined = _time_cuda(
            current_discarded_output_path,
            warmup=args.warmup,
            iters=args.iters,
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        cases.append(
            {
                "rows": rows,
                "hidden_gather": _time_cuda(
                    gather_hidden,
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                "lm_head_projection_and_tp8_all_gather": _time_cuda(
                    project_logits,
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                "reasoning_greedy_sampler": _time_cuda(
                    sample_logits,
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                "d2h_token_copy_and_completion_wait": _time_cuda(
                    copy_tokens,
                    warmup=args.warmup,
                    iters=args.iters,
                ),
                "combined_discarded_output_path": combined,
                "persistent_allocated_before_bytes": persistent_before,
                "peak_allocated_bytes": peak,
                "temporary_high_water_bytes": max(0, peak - persistent_before),
                "d2h_payload_bytes": rows * torch.int32.itemsize,
            }
        )
    local = {
        "rank": rank,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "cases": cases,
        "persistent_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "persistent_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank == 0:
        output = {
            "suite": "target19_tp8_partial_model_discarded_output",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "world_size": world_size,
            "torch": torch.__version__,
            "contract": {
                "model_path": args.model_path,
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "local_vocab_rows": int(lm_head.weight.shape[0]),
                "total_prefill_forward_budget": 8192,
                "sampling": "release reasoning mask plus greedy argmax",
                "communication": "release PyNCCL all-gather",
                "model_weights_loaded": False,
            },
            "ranks": gathered,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

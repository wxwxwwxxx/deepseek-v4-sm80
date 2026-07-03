#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from benchmark.offline.deepseek_v4_perf_matrix import (  # noqa: E402
    DSV4_A100_VICTORY_BUNDLE_TOGGLE,
    Variant,
    collect_runtime_environment,
    configure_variant,
)
from minisgl.distributed import DistributedInfo  # noqa: E402
from minisgl.kvcache import estimate_kvcache_bytes_per_page  # noqa: E402
from minisgl.llm.llm import LLM  # noqa: E402


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rank_size(args: argparse.Namespace) -> tuple[int, int]:
    rank = int(os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or 0)
    world_size = int(os.environ.get("WORLD_SIZE") or args.tensor_parallel_size)
    return rank, world_size


def _memory_snapshot(torch, device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _gather(torch, llm, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if torch.distributed.is_initialized():
        gathered: list[Any] = [None for _ in range(torch.distributed.get_world_size(group=llm.tp_cpu_group))]
        torch.distributed.all_gather_object(gathered, payload, group=llm.tp_cpu_group)
        return list(gathered)
    return [payload]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe automatic DSV4 KV sizing without running a workload.")
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-running-req", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=5120)
    parser.add_argument("--max-extend-tokens", type=int, default=4096)
    parser.add_argument("--allow-dsv4-cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-bs", nargs="*", type=int, default=None)
    parser.add_argument("--cuda-graph-capture-greedy-sample", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from minisgl.engine.engine import Engine
    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    rank, tp_size = _rank_size(args)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    variant = Variant(
        name="dsv4_sm80_a100_victory",
        env={DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1"},
        description="Capacity probe for the promoted A100 victory bundle.",
        allow_dsv4_cuda_graph=args.allow_dsv4_cuda_graph,
        cuda_graph_capture_greedy_sample=args.cuda_graph_capture_greedy_sample,
    )
    variant_env = configure_variant(dsv4_kernel, variant)

    memory_records: list[dict[str, Any]] = []
    original_sync_get_memory = Engine._sync_get_memory

    def recording_sync_get_memory(self):
        result = original_sync_get_memory(self)
        memory_records.append(
            {
                "call_index": len(memory_records),
                "min_free_bytes": int(result[0]),
                "max_free_bytes": int(result[1]),
            }
        )
        return result

    Engine._sync_get_memory = recording_sync_get_memory
    tic = time.perf_counter()
    llm = None
    try:
        llm = LLM(
            args.model_path,
            dtype=dtype,
            tp_info=DistributedInfo(rank, tp_size),
            max_running_req=args.max_running_req,
            max_seq_len_override=args.max_seq_len,
            max_extend_tokens=args.max_extend_tokens,
            num_page_override=None,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            use_pynccl=False,
            allow_dsv4_cuda_graph=args.allow_dsv4_cuda_graph,
            cuda_graph_bs=args.cuda_graph_bs,
            cuda_graph_capture_greedy_sample=args.cuda_graph_capture_greedy_sample,
            cuda_graph_capture_fail_open=True,
            distributed_init_method="env://",
        )
        torch.cuda.synchronize(llm.device)
        elapsed_s = time.perf_counter() - tic
        model_config = llm.engine.attn_backend.config
        bytes_per_page = estimate_kvcache_bytes_per_page(
            model_config,
            page_size=args.page_size,
            dtype=dtype,
            tp_size=tp_size,
        )
        logical_pages = int(llm.engine.num_pages)
        pool_pages = int(getattr(llm.engine.kv_cache, "_num_pages", logical_pages + 1))
        payload = {
            "rank": rank,
            "status": "pass",
            "elapsed_s": elapsed_s,
            "variant": {
                "name": variant.name,
                "env": variant.env,
                **variant_env,
            },
            "config": {
                "tensor_parallel_size": tp_size,
                "page_size": args.page_size,
                "memory_ratio": args.memory_ratio,
                "max_running_req": args.max_running_req,
                "max_seq_len_override": args.max_seq_len,
                "max_extend_tokens": args.max_extend_tokens,
                "allow_dsv4_cuda_graph": args.allow_dsv4_cuda_graph,
                "cuda_graph_bs": args.cuda_graph_bs,
                "cuda_graph_capture_greedy_sample": args.cuda_graph_capture_greedy_sample,
                "cuda_graph_capture_fail_open": True,
            },
            "capacity": {
                "chosen_num_pages": logical_pages,
                "pool_pages_including_dummy": pool_pages,
                "page_size": args.page_size,
                "logical_kv_token_capacity": logical_pages * args.page_size,
                "pool_token_capacity_including_dummy": pool_pages * args.page_size,
                "engine_max_seq_len": int(llm.engine.max_seq_len),
                "kv_cache_bytes_per_page_per_rank": int(bytes_per_page),
                "logical_kv_cache_bytes_per_rank": int(logical_pages * bytes_per_page),
                "pool_kv_cache_bytes_per_rank": int(pool_pages * bytes_per_page),
            },
            "memory_sync_records": memory_records,
            "memory_after_init": _memory_snapshot(torch, llm.device),
            "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}),
            "model_prepare_report": getattr(llm.engine, "model_prepare_report", {}),
            "runtime_environment": collect_runtime_environment(torch, dsv4_kernel, rank=rank),
        }
        gathered = _gather(torch, llm, payload)
        if rank == 0:
            _write_json(
                args.output,
                {
                    "status": "pass",
                    "ranks": gathered,
                    "rank0": payload,
                },
            )
    except BaseException as exc:
        error_payload = {
            "rank": rank,
            "status": "fail",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "memory_sync_records": memory_records,
        }
        if rank == 0:
            _write_json(args.output, {"status": "fail", "ranks": [error_payload], "rank0": error_payload})
        raise
    finally:
        Engine._sync_get_memory = original_sync_get_memory
        if llm is not None:
            llm.shutdown()


if __name__ == "__main__":
    main()

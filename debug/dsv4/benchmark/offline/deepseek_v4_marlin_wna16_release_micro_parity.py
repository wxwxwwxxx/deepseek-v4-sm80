from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from minisgl.distributed import DistributedInfo


def _configure_variant(name: str) -> dict[str, Any]:
    if name != "typed_marlin_release_oracle":
        raise SystemExit("only variant 'typed_marlin_release_oracle' is supported")
    return {
        "name": name,
        "runtime_mode": "fallback",
        "oracle_backend": "grouped_fp4",
        "candidate_backend": "marlin_wna16",
    }


def _distributed_init_method(args: argparse.Namespace, tp_size: int) -> str | None:
    if args.distributed_init_method is not None:
        return args.distributed_init_method
    if tp_size > 1 and "MASTER_ADDR" in os.environ:
        return "env://"
    return None


def _tp_rank_size(args: argparse.Namespace) -> tuple[int, int, int]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
    tp_size = args.tensor_parallel_size or env_world_size
    tp_rank = args.tp_rank if args.tp_rank is not None else env_local_rank
    if env_world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={env_world_size} does not match tensor parallel size {tp_size}"
        )
    return tp_rank, tp_size, env_world_size


def _summary(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().float()
    finite = torch.isfinite(data)
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "finite_ratio": float(finite.float().mean().item()),
        "abs_max": float(data.abs().max().item()),
        "mean": float(data.mean().item()),
        "checksum": float(data.flatten()[:4096].sum().item()),
    }


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    delta = (a.detach().float() - b.detach().float()).abs()
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "finite_ratio": float(torch.isfinite(delta).float().mean().item()),
    }


def _run_layer(args: argparse.Namespace, llm, layer_id: int) -> dict[str, Any]:
    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    torch.manual_seed(args.seed + layer_id)
    device = llm.device
    layer = llm.engine.model.model.layers.op_list[layer_id]
    experts = layer.mlp.experts
    if experts._marlin_wna16_weights is None:
        experts.prepare_marlin_wna16_weight_cache(release_original=False)
    cache = experts._marlin_wna16_weights
    num_experts = int(cache.w13.shape[0])
    hidden_size = int(experts.w13_weight.shape[-1] * 2)
    hidden = torch.randn(args.tokens, hidden_size, device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        low=0,
        high=num_experts,
        size=(args.tokens, args.topk),
        device=device,
        dtype=torch.long,
    )
    weights = torch.rand(args.tokens, args.topk, device=device, dtype=torch.float32)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

    raw_tensors = experts._raw_expert_weight_tensors()
    grouped_oracle = dsv4_kernel.moe_route_dispatch_bf16_grouped(
        hidden,
        weights,
        indices,
        *raw_tensors,
        swiglu_limit=experts.swiglu_limit,
    )
    raw_present, _ = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16(
        hidden,
        weights,
        indices,
        *raw_tensors,
        swiglu_limit=experts.swiglu_limit,
        cache=cache,
    )
    force_prepacked_raw_present = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16_prepacked(
        hidden,
        weights,
        indices,
        cache,
        swiglu_limit=experts.swiglu_limit,
    )

    release_report = experts.release_marlin_wna16_original_expert_weights()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    pressure = None
    if args.allocator_pressure_mib > 0:
        pressure = torch.empty(
            args.allocator_pressure_mib * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        pressure.fill_(17)
    released_same_cache = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16_prepacked(
        hidden,
        weights,
        indices,
        cache,
        swiglu_limit=experts.swiglu_limit,
    )
    torch.cuda.synchronize(device)
    del pressure

    result: dict[str, Any] = {
        "layer_id": int(layer_id),
        "tokens": int(args.tokens),
        "topk": int(args.topk),
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "allocator_pressure_mib": int(args.allocator_pressure_mib),
        "release_report": release_report,
        "raw_present": _summary(raw_present),
        "force_prepacked_raw_present": _summary(force_prepacked_raw_present),
        "released_same_cache": _summary(released_same_cache),
        "raw_present_vs_force_prepacked": _diff(raw_present, force_prepacked_raw_present),
        "raw_present_vs_released_same_cache": _diff(raw_present, released_same_cache),
        "force_prepacked_vs_released_same_cache": _diff(
            force_prepacked_raw_present,
            released_same_cache,
        ),
    }
    if grouped_oracle is not None:
        result["grouped_oracle"] = _summary(grouped_oracle)
        result["grouped_oracle_vs_raw_present"] = _diff(grouped_oracle, raw_present)
        result["grouped_oracle_vs_released_same_cache"] = _diff(
            grouped_oracle,
            released_same_cache,
        )
    else:
        result["grouped_oracle"] = None
    return result


def run(args: argparse.Namespace) -> int:
    from minisgl.llm import LLM

    rank, tp_size, _ = _tp_rank_size(args)
    variant_env = _configure_variant(args.variant)
    llm_kwargs: dict[str, Any] = {}
    distributed_init_method = _distributed_init_method(args, tp_size)
    if distributed_init_method is not None:
        llm_kwargs["distributed_init_method"] = distributed_init_method
    llm = LLM(
        args.model_path,
        tp_info=DistributedInfo(rank, tp_size),
        dsv4_runtime_mode="fallback",
        max_running_req=1,
        context_length=args.max_seq_len,
        max_extend_tokens=args.max_extend_tokens,
        num_page_override=args.num_pages,
        page_size=args.page_size,
        allow_dsv4_cuda_graph=False,
        **llm_kwargs,
    )
    try:
        started = time.perf_counter()
        layer_results = [_run_layer(args, llm, layer_id) for layer_id in args.layers]
        rank_payload = {
            "rank": rank,
            "elapsed_s": time.perf_counter() - started,
            "layers": layer_results,
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(llm.device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(llm.device)),
        }
        gathered: list[Any] = [None for _ in range(tp_size)]
        torch.distributed.all_gather_object(gathered, rank_payload, group=llm.tp_cpu_group)
        if rank == 0:
            payload = {
                "status": "pass",
                "model_path": args.model_path,
                "variant": variant_env,
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "per_rank": gathered,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    finally:
        llm.shutdown()
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--variant", default="typed_marlin_release_oracle")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--max-extend-tokens", type=int, default=4096)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 21, 42])
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allocator-pressure-mib", type=int, default=768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.page_size <= 0 or args.num_pages <= 1:
        parser.error("--page-size must be positive and --num-pages must be > 1")
    if args.tokens <= 0 or args.topk <= 0:
        parser.error("--tokens and --topk must be positive")
    if args.allocator_pressure_mib < 0:
        parser.error("--allocator-pressure-mib must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()

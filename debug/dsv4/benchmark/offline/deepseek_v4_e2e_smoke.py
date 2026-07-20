from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")

from minisgl.dsv4_release import DSV4_RELEASE  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal DeepSeek V4 offline E2E generation smoke for mini-sglang."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--decode-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--num-pages", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-addr", default=None)
    parser.add_argument(
        "--use-pynccl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PyNCCL in the optimized release.",
    )
    return parser.parse_args(argv)


def _execution_settings(requested_use_pynccl: bool) -> tuple[bool, bool]:
    return requested_use_pynccl, True


def _jsonable_release() -> dict[str, Any]:
    payload = asdict(DSV4_RELEASE)
    payload["direct_graph_metadata_groups"] = sorted(
        payload["direct_graph_metadata_groups"]
    )
    return payload


def _gpu_report() -> dict[str, Any]:
    report: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(device)
        report.update(
            {
                "device": device,
                "device_name": torch.cuda.get_device_name(device),
                "capability": f"sm{cap[0]}{cap[1]}",
            }
        )
    return report


def _make_prompts(*, batch_size: int, prompt_len: int, vocab_size: int) -> list[list[int]]:
    low = 10 if vocab_size > 32 else 1
    usable = max(vocab_size - low, 1)
    return [
        [low + ((row * prompt_len + col) % usable) for col in range(prompt_len)]
        for row in range(batch_size)
    ]


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tp_rank_size(args: argparse.Namespace) -> tuple[int, int]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    tp_size = args.tensor_parallel_size or env_world_size
    tp_rank = args.tp_rank if args.tp_rank is not None else env_local_rank
    if tp_size <= 0:
        raise ValueError("tensor parallel size must be positive")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp rank must satisfy 0 <= rank < size, got rank={tp_rank} size={tp_size}")
    if tp_size > 1 and env_world_size == 1 and args.tp_rank is None:
        raise ValueError(
            "multi-rank smoke requires torchrun or explicit per-process --tp-rank launch"
        )
    return tp_rank, tp_size


def _validate_args(args: argparse.Namespace) -> None:
    if args.prompt_len <= 0:
        raise ValueError("--prompt-len must be positive")
    if args.decode_len <= 0:
        raise ValueError("--decode-len must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_pages is not None and args.num_pages <= 1:
        raise ValueError("--num-pages must be greater than 1")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    tp_rank, tp_size = _tp_rank_size(args)
    is_primary = tp_rank == 0
    use_pynccl, allow_cuda_graph = _execution_settings(args.use_pynccl)
    start = time.perf_counter()
    llm = None
    expected_tokens = args.batch_size * args.decode_len
    max_seq_len = args.prompt_len + args.decode_len
    num_pages = args.num_pages or max(max_seq_len + args.batch_size + 8, 32)
    distributed_addr = args.distributed_addr
    if distributed_addr is None and tp_size > 1 and "MASTER_ADDR" in os.environ:
        distributed_addr = "env://"
    result: dict[str, Any] = {
        "status": "fail",
        "model_path": args.model_path,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "batch_size": args.batch_size,
        "expected_generated_tokens": expected_tokens,
        "dsv4_release": _jsonable_release(),
        "torch_version": torch.__version__,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "is_primary_rank": is_primary,
        "distributed_addr": distributed_addr,
        "requested_use_pynccl": args.use_pynccl,
        "use_pynccl": use_pynccl,
        "allow_cuda_graph": allow_cuda_graph,
    }

    try:
        from minisgl.core import SamplingParams
        from minisgl.distributed import DistributedInfo
        from minisgl.llm import LLM

        llm_kwargs: dict[str, Any] = {}
        if distributed_addr is not None:
            llm_kwargs["distributed_init_method"] = distributed_addr
        llm = LLM(
            args.model_path,
            tp_info=DistributedInfo(tp_rank, tp_size),
            max_running_req=max(args.batch_size, 1),
            context_length=max_seq_len,
            max_extend_tokens=args.prompt_len * args.batch_size,
            num_page_override=num_pages,
            page_size=256,
            memory_ratio=args.memory_ratio,
            use_pynccl=use_pynccl,
            allow_dsv4_cuda_graph=allow_cuda_graph,
            disable_cuda_graph=not allow_cuda_graph,
            **llm_kwargs,
        )
        prompts = _make_prompts(
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            vocab_size=llm.engine.sampler.vocab_size,
        )
        sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.decode_len,
        )
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize(llm.device)
        generated_token_ids = [list(output["token_ids"]) for output in outputs]
        generated_tokens = sum(len(ids) for ids in generated_token_ids)
        result.update(
            {
                "generated_token_count": generated_tokens,
                "generated_token_ids": generated_token_ids,
            }
        )
        if generated_tokens == expected_tokens:
            result["status"] = "pass"
        else:
            result["failure"] = (
                f"expected {expected_tokens} generated tokens, got {generated_tokens}"
            )
    except BaseException as exc:
        result.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )
    finally:
        result["elapsed_s"] = time.perf_counter() - start
        result["gpu"] = _gpu_report()
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException as exc:
                result.setdefault("shutdown_exception_type", type(exc).__name__)
                result.setdefault("shutdown_exception_message", str(exc))
                if result["status"] == "pass":
                    result["status"] = "fail"
        if tp_size > 1:
            _write_json(f"{args.output}.rank{tp_rank}.json", result)
        if is_primary:
            _write_json(args.output, result)
            print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

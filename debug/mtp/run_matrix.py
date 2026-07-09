from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "python" / "minisgl").is_dir() and (
            candidate / "benchmark"
        ).is_dir():
            return candidate
    raise RuntimeError(f"Could not find mini-sglang repo root from {start}")


ROOT = _find_repo_root(Path(__file__).resolve())
PYTHON_ROOT = ROOT / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from benchmark.offline.deepseek_v4_text_smoke import (  # noqa: E402
    _distributed_init_method,
    _tp_rank_size,
    _variant_map,
    configure_variant,
)
from minisgl.core import SamplingParams  # noqa: E402
from minisgl.distributed import DistributedInfo  # noqa: E402
from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402
from minisgl.llm import LLM  # noqa: E402


PROMPTS = (
    "The capital of France is",
    "用一句话解释张量并行：",
    "def fibonacci(n):",
    "List three colors:",
    "Explain CUDA graphs briefly:",
    "Translate hello to Chinese:",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _snapshot_stats(llm: LLM) -> dict[str, Any]:
    return copy.deepcopy(getattr(llm.engine, "mtp_spec_stats", {}))


def _list_delta(before: list[Any], after: list[Any]) -> list[Any]:
    if len(after) >= len(before) and after[: len(before)] == before:
        return copy.deepcopy(after[len(before) :])
    return copy.deepcopy(after)


def _stats_delta(before: Any, after: Any) -> Any:
    if isinstance(after, dict) and isinstance(before, dict):
        out: dict[str, Any] = {}
        for key, value in after.items():
            old = before.get(key)
            if isinstance(value, (int, float)) and isinstance(old, (int, float)):
                out[key] = value - old
            elif isinstance(value, list) and isinstance(old, list):
                out[key] = _list_delta(old, value)
            elif isinstance(value, dict) and isinstance(old, dict):
                out[key] = _stats_delta(old, value)
            else:
                out[key] = copy.deepcopy(value)
        return out
    return copy.deepcopy(after)


def _configure_mode(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")
    variant = _variant_map()[args.variant]
    env_report = configure_variant(dsv4_kernel, variant)
    applied_env = dict(variant.env)

    if args.mode == "mtp_speculative":
        mtp_env = {
            "MINISGL_DSV4_EXPERIMENTAL_MTP": "1",
            "MINISGL_DSV4_MTP_SPECULATIVE": "1",
            "MINISGL_DSV4_MTP_SPEC_DRAFT_LEN": str(args.draft_len),
            "MINISGL_DSV4_TARGET_VERIFY_RUNTIME": args.target_verify_runtime,
            "MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH": "1",
        }
        for name, value in mtp_env.items():
            os.environ[name] = value
        applied_env.update(mtp_env)
    else:
        for name in (
            "MINISGL_DSV4_EXPERIMENTAL_MTP",
            "MINISGL_DSV4_MTP_SPECULATIVE",
            "MINISGL_DSV4_MTP_SPEC_DRAFT_LEN",
            "MINISGL_DSV4_TARGET_VERIFY_RUNTIME",
            "MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH",
        ):
            os.environ.pop(name, None)

    debug_env = {}
    for name in (
        "MINISGL_DSV4_MTP_SPEC_TRACE",
        "MINISGL_DSV4_MTP_STATE_PARITY_TRACE",
        "MINISGL_DSV4_MTP_STATE_PARITY_TRACE_LIMIT",
        "MINISGL_DSV4_MTP_ROW_DEPTH_ORACLE",
        "MINISGL_DSV4_MTP_ROW_DEPTH_ORACLE_COMPACT",
        "MINISGL_DSV4_MTP_ROW0_LAYER_PARITY",
        "MINISGL_DSV4_MTP_OPERATOR_PARITY",
        "MINISGL_DSV4_MTP_OPERATOR_PARITY_OPERATORS",
        "MINISGL_DSV4_MTP_OPERATOR_PARITY_LAYERS",
        "MINISGL_DSV4_MTP_OPERATOR_PARITY_ATOL",
        "MINISGL_DSV4_MTP_OPERATOR_PARITY_RTOL",
        "MINISGL_DSV4_MTP_ROW_TRACE_ROWS",
        "MINISGL_DSV4_MTP_ROW_TENSOR_TRACE",
        "MINISGL_DSV4_MTP_Q_WQB_CONTRACT_ORACLE",
        "MINISGL_DSV4_MTP_Q_WQB_TARGET_NORMAL_SHAPE",
        "MINISGL_DSV4_MTP_Q_WQB_TARGET_FULL_ROWS",
        "MINISGL_DSV4_MTP_Q_WQB_GLOBAL_ROW_INVARIANT",
        "MINISGL_DSV4_MTP_Q_WQB_REFERENCE_GATE",
        "MINISGL_DSV4_MTP_SHARED_EXPERT_MICROBENCH",
        "MINISGL_DSV4_MTP_SHARED_EXPERT_MICROBENCH_RANKS",
        "MINISGL_DSV4_MTP_SHARED_EXPERT_MICROBENCH_POSITION",
        "MINISGL_DSV4_MTP_SHARED_EXPERT_MICROBENCH_TOKEN",
        "MINISGL_DSV4_MTP_SHARED_EXPERT_MICROBENCH_MAX_RECORDS",
        "MINISGL_DSV4_NORMAL_PRODUCER_TRACE",
        "MINISGL_DSV4_NORMAL_PRODUCER_TRACE_ALL_LAYERS",
        "MINISGL_DSV4_NORMAL_PRODUCER_TRACE_LAYERS",
        "MINISGL_DSV4_NORMAL_PRODUCER_TRACE_ALL_BOUNDARIES",
        "MINISGL_DSV4_NORMAL_PRODUCER_TRACE_LAYER2_ATTENTION_SPLIT",
        "MINISGL_DSV4_LAYER2_SWA_LIFECYCLE_TRACE",
        "MINISGL_DSV4_SWA_LIFECYCLE_TRACE_LAYERS",
    ):
        if os.environ.get(name, "").strip():
            debug_env[name] = os.environ[name]
    return {
        **env_report,
        "applied_env": applied_env,
        "debug_env": debug_env,
    }


def _gpu_report(device: torch.device) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free, total = torch.cuda.mem_get_info(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "device": int(device.index or 0),
        "name": props.name,
        "capability": f"sm{props.major}{props.minor}",
        "memory_free_bytes": int(free),
        "memory_total_bytes": int(total),
    }


def run(args: argparse.Namespace) -> int:
    rank, tp_size, env_world_size = _tp_rank_size(args)
    if env_world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={env_world_size} does not match tensor parallel size {tp_size}"
        )
    env_report = _configure_mode(args)
    distributed_init_method = _distributed_init_method(args, tp_size)
    llm = None
    payload: dict[str, Any]
    try:
        kwargs: dict[str, Any] = {}
        if distributed_init_method is not None:
            kwargs["distributed_init_method"] = distributed_init_method
        llm = LLM(
            args.model_path,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(rank, tp_size),
            max_running_req=args.max_running_req,
            max_seq_len_override=args.max_seq_len,
            max_extend_tokens=args.max_extend_tokens,
            num_page_override=args.num_pages,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            use_pynccl=False,
            allow_dsv4_cuda_graph=False,
            cuda_graph_bs=[],
            cuda_graph_capture_greedy_sample=False,
            **kwargs,
        )
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.decode_len,
        )
        runs = []
        for batch_size in args.batch_sizes:
            prompts = list(PROMPTS[: int(batch_size)])
            before = _snapshot_stats(llm)
            tic = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            torch.cuda.synchronize(llm.device)
            elapsed_s = time.perf_counter() - tic
            after = _snapshot_stats(llm)
            runs.append(
                {
                    "batch_size": int(batch_size),
                    "prompts": prompts,
                    "outputs": outputs,
                    "token_ids": [list(item["token_ids"]) for item in outputs],
                    "elapsed_s": float(elapsed_s),
                    "stats_delta": _stats_delta(before, after),
                    "stats_after": after,
                }
            )
        payload = {
            "ok": True,
            "rank": int(rank),
            "world_size": int(tp_size),
            "mode": args.mode,
            "model_path": args.model_path,
            "page_size": int(args.page_size),
            "num_pages": int(args.num_pages),
            "draft_len": int(args.draft_len),
            "decode_len": int(args.decode_len),
            "batch_sizes": [int(x) for x in args.batch_sizes],
            "max_running_req": int(args.max_running_req),
            "env": env_report,
            "gpu": _gpu_report(llm.device),
            "model_prepare_report": getattr(llm.engine, "model_prepare_report", {}),
            "mtp_spec_stats_final": _snapshot_stats(llm),
            "runs": runs,
        }
    except BaseException as exc:
        payload = {
            "ok": False,
            "rank": int(rank),
            "world_size": int(tp_size),
            "mode": args.mode,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=40),
            },
        }
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException:
                traceback.print_exc()

    output = Path(args.output)
    rank_output = output.with_suffix(output.suffix + f".rank{rank}.json")
    _write_json(rank_output, payload)
    if rank == 0:
        _write_json(output, payload)
        if payload.get("ok"):
            summary = {
                "ok": True,
                "mode": payload["mode"],
                "output": str(output),
                "debug_env": payload["env"].get("debug_env", {}),
                "runs": [
                    {
                        "batch_size": run["batch_size"],
                        "token_ids": run["token_ids"],
                        "trace_counts": {
                            "debug_trace": len(run["stats_delta"].get("debug_trace", [])),
                            "state_parity_trace": len(
                                run["stats_delta"].get("state_parity_trace", [])
                            ),
                            "target_verify_contract_trace": len(
                                run["stats_delta"].get("target_verify_contract_trace", [])
                            ),
                            "row_depth_oracle_debug": len(
                                run["stats_delta"].get("row_depth_oracle_debug", [])
                            ),
                        },
                    }
                    for run in payload["runs"]
                ],
            }
        else:
            summary = {
                "ok": False,
                "mode": payload.get("mode"),
                "output": str(output),
                "error": payload.get("error", {}),
            }
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if payload.get("ok") else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "mtp_speculative"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--variant", default="dsv4_sm80_a100_victory")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=16)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-extend-tokens", type=int, default=4096)
    parser.add_argument("--max-running-req", type=int, default=4)
    parser.add_argument("--decode-len", type=int, default=8)
    parser.add_argument("--draft-len", type=int, default=2)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 5, 6])
    parser.add_argument(
        "--target-verify-runtime",
        default="sglang_prefill_extend",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

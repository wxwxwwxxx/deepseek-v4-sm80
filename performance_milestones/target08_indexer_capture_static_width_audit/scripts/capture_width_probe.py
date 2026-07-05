from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine
from minisgl.scheduler import SchedulerConfig

MILESTONE_DIR = ROOT / "performance_milestones" / "target08_indexer_capture_static_width_audit"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _distributed_init_method() -> str | None:
    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        return f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture-only DSV4 indexer width audit probe for one CUDA graph bucket."
    )
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument(
        "--width-mode",
        choices=("current", "table_width", "seq_len_aligned"),
        default="current",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--audit-log-dir", default=str(MILESTONE_DIR / "raw"))
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--max-running-req", type=int, default=16)
    parser.add_argument("--cuda-graph-bs", nargs="+", type=int, default=[16])
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--use-dummy-weight", action="store_true")
    parser.add_argument("--disable-pynccl", action="store_true")
    parser.add_argument("--enable-dsv4-radix-prefix-cache", action="store_true")
    parser.add_argument("--enable-dsv4-component-loc-ownership", action="store_true")
    args = parser.parse_args(argv)
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.num_pages <= 1:
        parser.error("--num-pages must be greater than 1")
    if args.max_seq_len <= 0 or args.max_running_req <= 0:
        parser.error("--max-seq-len and --max-running-req must be positive")
    if any(bs <= 0 for bs in args.cuda_graph_bs):
        parser.error("--cuda-graph-bs values must be positive")
    args.cuda_graph_bs = sorted(set(args.cuda_graph_bs))
    if max(args.cuda_graph_bs) > args.max_running_req:
        parser.error("--max-running-req must be >= max --cuda-graph-bs")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rank = args.tp_rank if args.tp_rank is not None else _env_int("LOCAL_RANK", _env_int("RANK", 0))
    tp_size = (
        args.tensor_parallel_size
        if args.tensor_parallel_size is not None
        else _env_int("WORLD_SIZE", 1)
    )
    run_label = args.run_label or f"{args.width_mode}_bs{'_'.join(map(str, args.cuda_graph_bs))}"
    os.environ["MINISGL_DSV4_INDEXER_CAPTURE_WIDTH_MODE"] = args.width_mode
    os.environ.setdefault("MINISGL_DSV4_INDEXER_CAPTURE_WIDTH_DEBUG", "1")
    os.environ.setdefault("MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG", "1")
    os.environ["MINISGL_DSV4_AUDIT_LOG_DIR"] = args.audit_log_dir
    os.environ["MINISGL_DSV4_AUDIT_RUN_LABEL"] = run_label

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    kwargs: dict[str, Any] = {}
    init_method = _distributed_init_method()
    if init_method is not None:
        kwargs["distributed_init_method"] = init_method

    engine = Engine(
        SchedulerConfig(
            model_path=args.model_path,
            tp_info=DistributedInfo(rank=rank, size=tp_size),
            dtype=dtype,
            max_running_req=args.max_running_req,
            max_extend_tokens=args.max_seq_len,
            cuda_graph_bs=args.cuda_graph_bs,
            allow_dsv4_cuda_graph=True,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            use_dummy_weight=args.use_dummy_weight,
            use_pynccl=not args.disable_pynccl,
            max_seq_len_override=args.max_seq_len,
            num_page_override=args.num_pages,
            enable_dsv4_radix_prefix_cache=args.enable_dsv4_radix_prefix_cache,
            enable_dsv4_component_loc_ownership=args.enable_dsv4_component_loc_ownership,
            **kwargs,
        )
    )
    torch.cuda.synchronize(engine.device)
    graph_status = getattr(engine.graph_runner, "capture_status", {})
    payload = {
        "rank": int(rank),
        "tp_size": int(tp_size),
        "width_mode": args.width_mode,
        "run_label": run_label,
        "page_size": int(args.page_size),
        "num_pages": int(args.num_pages),
        "engine_max_seq_len": int(engine.max_seq_len),
        "global_page_table_shape": list(engine.page_table.shape),
        "graph_runner": graph_status,
    }

    gathered: list[Any]
    if torch.distributed.is_initialized():
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    if rank == 0:
        out = Path(args.json_out or MILESTONE_DIR / "raw" / f"capture_width_probe_{run_label}.json")
        _write_json(
            out,
            {
                "config": {
                    "width_mode": args.width_mode,
                    "run_label": run_label,
                    "cuda_graph_bs": args.cuda_graph_bs,
                    "page_size": args.page_size,
                    "num_pages": args.num_pages,
                    "max_seq_len": args.max_seq_len,
                    "max_running_req": args.max_running_req,
                    "use_dummy_weight": bool(args.use_dummy_weight),
                    "use_pynccl": not args.disable_pynccl,
                    "enable_dsv4_radix_prefix_cache": bool(
                        args.enable_dsv4_radix_prefix_cache
                    ),
                    "enable_dsv4_component_loc_ownership": bool(
                        args.enable_dsv4_component_loc_ownership
                    ),
                    "audit_log_dir": args.audit_log_dir,
                },
                "ranks": gathered,
            },
        )

    engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import inspect
import os
from collections import defaultdict
from typing import Any

_INSTALLED = False
_PHASE = "init"
_RECORDS: dict[tuple[Any, ...], dict[str, Any]] = {}


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.distributed.parallel_state import GroupCoordinator

    _patch_method(GroupCoordinator, "all_reduce", "all_reduce")
    _patch_method(GroupCoordinator, "all_gather", "all_gather")
    _patch_method(GroupCoordinator, "reduce_scatter", "reduce_scatter")
    _patch_method(GroupCoordinator, "gather", "gather")
    _INSTALLED = True


def reset_worker(_worker: object | None = None, phase: str = "unknown") -> dict[str, Any]:
    global _PHASE, _RECORDS
    _PHASE = phase
    _RECORDS = {}
    return {"pid": os.getpid(), "phase": _PHASE, "reset": True}


def set_phase_worker(_worker: object | None = None, phase: str = "unknown") -> dict[str, Any]:
    global _PHASE
    _PHASE = phase
    return {"pid": os.getpid(), "phase": _PHASE}


def snapshot_worker(_worker: object | None = None) -> dict[str, Any]:
    entries = []
    by_label: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_op: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    for record in sorted(
        _RECORDS.values(),
        key=lambda item: (
            str(item["phase"]),
            str(item["label"]),
            str(item["op"]),
            str(item["dtype"]),
            str(item["shape"]),
            str(item["output_shape"]),
        ),
    ):
        entry = dict(record)
        entries.append(entry)
        for target in (by_label[entry["label"]], by_op[entry["op"]]):
            target["count"] += int(entry["count"])
            target["bytes"] += int(entry["bytes"])
    return {
        "pid": os.getpid(),
        "phase": _PHASE,
        "installed": _INSTALLED,
        "entries": entries,
        "by_label": dict(sorted(by_label.items())),
        "by_op": dict(sorted(by_op.items())),
        "total_count": int(sum(entry["count"] for entry in entries)),
        "total_bytes": int(sum(entry["bytes"] for entry in entries)),
    }


def _patch_method(cls: type, method_name: str, op: str) -> None:
    original = getattr(cls, method_name)
    if getattr(original, "_target10_comm_probe_wrapped", False):
        return

    def wrapper(self, input_, *args, **kwargs):
        if _is_torch_compiling():
            return original(self, input_, *args, **kwargs)
        result = original(self, input_, *args, **kwargs)
        dim = kwargs.get("dim")
        if dim is None and args:
            dim = args[0]
        dst = kwargs.get("dst")
        if dst is None and args and op == "gather":
            dst = args[0]
        _record(
            op=op,
            group=getattr(self, "unique_name", None),
            world_size=int(getattr(self, "world_size", 1)),
            input_=input_,
            result=result,
            elapsed_us=0.0,
            dim=dim,
            dst=dst,
        )
        return result

    wrapper._target10_comm_probe_wrapped = True  # type: ignore[attr-defined]
    wrapper._target10_comm_probe_original = original  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapper)


def _is_torch_compiling() -> bool:
    try:
        import torch

        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling):
            return bool(is_compiling())
        import torch._dynamo as torch_dynamo  # type: ignore[import-not-found]

        return bool(torch_dynamo.is_compiling())
    except Exception:
        return False


def _record(
    *,
    op: str,
    group: str | None,
    world_size: int,
    input_: Any,
    result: Any,
    elapsed_us: float,
    dim: Any,
    dst: Any,
) -> None:
    if world_size <= 1:
        return
    shape = _shape(input_)
    output_shape = _shape(result)
    dtype = _dtype(input_)
    input_bytes = _nbytes(input_)
    output_bytes = _nbytes(result)
    if op == "all_reduce":
        bytes_value = input_bytes
    elif output_bytes > 0:
        bytes_value = output_bytes
    else:
        bytes_value = input_bytes
    frames = inspect.stack(context=0)
    label, boundary = _classify(op, frames)
    capture_state = _capture_state()
    key = (
        _PHASE,
        label,
        boundary,
        op,
        dtype,
        tuple(shape),
        tuple(output_shape),
        capture_state,
        str(group),
        str(dim),
        str(dst),
    )
    record = _RECORDS.get(key)
    if record is None:
        record = {
            "phase": _PHASE,
            "label": label,
            "boundary": boundary,
            "op": op,
            "dtype": dtype,
            "shape": shape,
            "output_shape": output_shape,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "bytes": 0,
            "count": 0,
            "elapsed_us": 0.0,
            "capture_state": capture_state,
            "group": group,
            "world_size": world_size,
            "dim": dim,
            "dst": dst,
        }
        _RECORDS[key] = record
    record["count"] += 1
    record["bytes"] += bytes_value
    record["elapsed_us"] += float(elapsed_us)


def _shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    return [int(dim) for dim in shape]


def _dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return "none"
    return str(dtype).removeprefix("torch.")


def _nbytes(value: Any) -> int:
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if not callable(numel) or not callable(element_size):
        return 0
    return int(numel() * element_size())


def _capture_state() -> str:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return "cuda_graph_capture"
    except Exception:
        pass
    return "eager_or_python_capture_call"


def _classify(op: str, frames: list[inspect.FrameInfo]) -> tuple[str, str]:
    stack = [
        (os.path.basename(frame.filename), frame.function, int(frame.lineno))
        for frame in frames
    ]
    filenames = {name for name, _, _ in stack}
    functions = {fn for _, fn, _ in stack}

    if "vocab_parallel_embedding.py" in filenames and op == "all_reduce":
        return "vllm.embedding_all_reduce", "VocabParallelEmbedding.forward"

    if "logits_processor.py" in filenames:
        if op == "all_gather":
            return "vllm.logits_all_gather", "LogitsProcessor._gather_logits"
        if op == "gather":
            return "vllm.logits_gather", "LogitsProcessor._gather_logits"

    if "deepseek_v4_attention.py" in filenames and op == "all_reduce":
        return (
            "vllm.attn.wo_b.row_parallel_projection_all_reduce",
            "DeepseekV4 attention wo_b RowParallelLinear",
        )

    if "moe_runner.py" in filenames and "_maybe_reduce_final_output" in functions:
        return "vllm.moe.reduce_once_all_reduce", "FusedMoE runner final reduce"

    if "moe_runner.py" in filenames and "_maybe_reduce_shared_expert_output" in functions:
        return "vllm.shared_expert_all_reduce", "FusedMoE runner shared reduce"

    if "moe_runner.py" in filenames and op == "all_gather":
        return "vllm.moe.dispatch_all_gather", "FusedMoE runner dispatch"

    if "moe_runner.py" in filenames and op == "reduce_scatter":
        return "vllm.moe.combine_reduce_scatter", "FusedMoE runner combine"

    if "linear.py" in filenames and op == "all_reduce":
        return "vllm.row_parallel_projection_all_reduce.unknown", "RowParallelLinear.forward"

    if "linear.py" in filenames and op == "all_gather":
        return "vllm.column_parallel_all_gather.unknown", "ColumnParallelLinear.forward"

    return f"vllm.{op}.unknown", _format_stack_boundary(stack)


def _format_stack_boundary(stack: list[tuple[str, str, int]]) -> str:
    interesting = []
    for filename, function, lineno in stack:
        if filename.startswith("vllm_comm_probe"):
            continue
        if filename in {"parallel_state.py", "communication_op.py"}:
            continue
        if filename.startswith("threading"):
            continue
        interesting.append(f"{filename}:{function}:{lineno}")
        if len(interesting) >= 4:
            break
    return " <- ".join(interesting) if interesting else "unknown"

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any, Iterator

import torch


TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
DSV4_AUDIT_LOG_DIR_ENV = "MINISGL_DSV4_AUDIT_LOG_DIR"
DSV4_AUDIT_RUN_LABEL_ENV = "MINISGL_DSV4_AUDIT_RUN_LABEL"
DSV4_MARLIN_WNA16_CACHE_DEBUG_ENV = "MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG"
DSV4_WARMUP_FORWARD_MEMORY_DEBUG_ENV = "MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG"
DEFAULT_AUDIT_LOG_DIR = "performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/raw"

_warmup_context: dict[str, Any] | None = None


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_ENV_VALUES


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"present": False}
    return {
        "present": True,
        "shape": [int(dim) for dim in tensor.shape],
        "stride": [int(dim) for dim in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "bytes": tensor_nbytes(tensor),
        "is_contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
    }


def cuda_memory_snapshot(
    device: torch.device | str | int | None = None,
    *,
    synchronize: bool = True,
) -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    cuda_device = torch.device(device) if device is not None else torch.cuda.current_device()
    if synchronize:
        torch.cuda.synchronize(cuda_device)
    free_memory, total_memory = torch.cuda.mem_get_info(cuda_device)
    return {
        "free_memory_bytes": int(free_memory),
        "total_memory_bytes": int(total_memory),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(cuda_device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(cuda_device)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
    }


def append_jsonl(kind: str, payload: dict[str, Any]) -> None:
    try:
        directory = os.environ.get(DSV4_AUDIT_LOG_DIR_ENV, DEFAULT_AUDIT_LOG_DIR)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{kind}_{_run_label()}_rank{_rank()}.jsonl")
        record = {
            "rank": _rank(),
            "pid": os.getpid(),
            "time_s": time.time(),
            **payload,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    except Exception:
        return


@contextlib.contextmanager
def warmup_forward_context(
    *,
    label: str,
    batch_size: int | None,
    device: torch.device | str | int,
) -> Iterator[None]:
    global _warmup_context
    if not env_flag(DSV4_WARMUP_FORWARD_MEMORY_DEBUG_ENV) or not torch.cuda.is_available():
        yield
        return

    previous_context = _warmup_context
    _warmup_context = {
        "label": label,
        "batch_size": None if batch_size is None else int(batch_size),
        "device": torch.device(device),
        "seq": 0,
        "baseline": None,
        "previous": None,
    }
    record_warmup_memory(owner="model.forward", stage="enter", device=device)
    try:
        yield
    finally:
        record_warmup_memory(owner="model.forward", stage="exit", device=device)
        _warmup_context = previous_context


def record_warmup_memory(
    *,
    owner: str,
    stage: str,
    layer_id: int | None = None,
    device: torch.device | str | int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if _warmup_context is None:
        return
    if not env_flag(DSV4_WARMUP_FORWARD_MEMORY_DEBUG_ENV) or not torch.cuda.is_available():
        return

    cuda_device = torch.device(device) if device is not None else _warmup_context["device"]
    snapshot = cuda_memory_snapshot(cuda_device, synchronize=True)
    if not snapshot:
        return

    seq = int(_warmup_context["seq"])
    _warmup_context["seq"] = seq + 1
    baseline = _warmup_context.get("baseline")
    previous = _warmup_context.get("previous")
    if baseline is None:
        baseline = snapshot
        _warmup_context["baseline"] = baseline

    payload: dict[str, Any] = {
        "event": "dsv4_warmup_forward_memory",
        "context_label": _warmup_context["label"],
        "batch_size": _warmup_context["batch_size"],
        "seq": seq,
        "owner": owner,
        "stage": stage,
        "layer_id": layer_id,
        **snapshot,
    }
    if previous is not None:
        payload.update(_memory_deltas(previous, snapshot, suffix="from_previous"))
    if baseline is not None:
        payload.update(_memory_deltas(baseline, snapshot, suffix="from_baseline"))
    if extra:
        payload["extra"] = extra
    _warmup_context["previous"] = snapshot
    append_jsonl("warmup_forward_memory", payload)


def _memory_deltas(
    before: dict[str, int],
    after: dict[str, int],
    *,
    suffix: str,
) -> dict[str, int]:
    return {
        f"free_delta_{suffix}_bytes": int(
            before.get("free_memory_bytes", 0) - after.get("free_memory_bytes", 0)
        ),
        f"memory_allocated_delta_{suffix}_bytes": int(
            after.get("memory_allocated_bytes", 0)
            - before.get("memory_allocated_bytes", 0)
        ),
        f"memory_reserved_delta_{suffix}_bytes": int(
            after.get("memory_reserved_bytes", 0) - before.get("memory_reserved_bytes", 0)
        ),
    }


def _rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            pass
    return 0


def _run_label() -> str:
    raw = os.environ.get(DSV4_AUDIT_RUN_LABEL_ENV, "run").strip()
    return raw or "run"

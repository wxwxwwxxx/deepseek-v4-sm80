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
DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG"
)
DSV4_MARLIN_WNA16_CACHE_INTEGRITY_SAMPLE_SIZE_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_SAMPLE_SIZE"
)
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
        "data_ptr": int(tensor.data_ptr()),
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


def tensor_integrity_summary(
    tensor: torch.Tensor | None,
    *,
    sample_size: int | None = None,
) -> dict[str, Any]:
    summary = tensor_summary(tensor)
    if tensor is None:
        return summary
    if sample_size is None:
        sample_size = _env_int(DSV4_MARLIN_WNA16_CACHE_INTEGRITY_SAMPLE_SIZE_ENV, 4096)
    sample_size = max(1, int(sample_size))
    summary["sample_size_requested"] = sample_size
    if tensor.numel() == 0:
        summary.update(
            {
                "sample_count": 0,
                "finite_ratio": 1.0,
                "sample_checksum": 0,
                "sample_abs_max": 0.0,
            }
        )
        return summary
    if tensor.is_cuda and _cuda_graph_capture_active():
        summary["sample_skipped"] = "cuda_graph_capture_active"
        return summary

    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() <= sample_size:
            sample = flat
        else:
            indices = torch.linspace(
                0,
                flat.numel() - 1,
                steps=sample_size,
                device=flat.device,
                dtype=torch.float64,
            ).to(torch.int64)
            sample = flat.index_select(0, indices)
        sample_cpu = sample.contiguous().cpu()
        sample_for_math = _sample_for_math(sample_cpu)
        if torch.is_floating_point(sample_for_math) or sample_for_math.is_complex():
            finite = torch.isfinite(sample_for_math)
            finite_ratio = float(finite.float().mean().item())
        else:
            finite_ratio = 1.0
        sample_abs_max = float(sample_for_math.float().abs().max().item())
        checksum = _sample_checksum(sample_cpu)
        summary.update(
            {
                "sample_count": int(sample_cpu.numel()),
                "finite_ratio": finite_ratio,
                "sample_checksum": checksum,
                "sample_abs_max": sample_abs_max,
            }
        )
    except Exception as exc:
        summary["sample_error"] = f"{type(exc).__name__}: {exc}"
    return summary


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _sample_for_math(sample: torch.Tensor) -> torch.Tensor:
    if sample.dtype in (getattr(torch, "float8_e8m0fnu", None), getattr(torch, "float8_e4m3fn", None)):
        return sample.float()
    if sample.dtype == torch.uint8 or sample.dtype == torch.int8:
        return sample.to(torch.float32)
    return sample


def _sample_checksum(sample: torch.Tensor) -> int:
    try:
        byte_view = sample.contiguous().view(torch.uint8).reshape(-1).to(torch.int64)
    except Exception:
        byte_view = _sample_for_math(sample).float().reshape(-1).cpu().view(torch.uint8).to(torch.int64)
    if byte_view.numel() == 0:
        return 0
    weights = (torch.arange(byte_view.numel(), dtype=torch.int64) % 251) + 1
    return int(torch.sum(byte_view.cpu() * weights).item() % ((1 << 63) - 1))


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

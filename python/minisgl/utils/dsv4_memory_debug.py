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
DSV4_MARLIN_WNA16_RELEASE_LEDGER_DEBUG_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_RELEASE_LEDGER_DEBUG"
)
DSV4_MARLIN_WNA16_OWNER_LEDGER_INTEGRITY_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_OWNER_LEDGER_INTEGRITY"
)
DSV4_MARLIN_WNA16_LAYER2_OWNER_PROBE_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_LAYER2_OWNER_PROBE"
)
DSV4_WARMUP_FORWARD_MEMORY_DEBUG_ENV = "MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG"
DEFAULT_AUDIT_LOG_DIR = "performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/raw"

_warmup_context: dict[str, Any] | None = None
_marlin_wna16_freed_ranges: list[dict[str, Any]] = []


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_ENV_VALUES


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None:
        return {"present": False}
    start = int(tensor.data_ptr()) if tensor.device.type == "cuda" else 0
    nbytes = tensor_nbytes(tensor)
    return {
        "present": True,
        "data_ptr": start,
        "start": start,
        "end": int(start + nbytes) if start else 0,
        "shape": [int(dim) for dim in tensor.shape],
        "stride": [int(dim) for dim in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "bytes": nbytes,
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


def marlin_wna16_release_ledger_enabled() -> bool:
    return env_flag(DSV4_MARLIN_WNA16_RELEASE_LEDGER_DEBUG_ENV)


def marlin_wna16_layer2_owner_probe_enabled() -> bool:
    return env_flag(DSV4_MARLIN_WNA16_LAYER2_OWNER_PROBE_ENV)


def reset_marlin_wna16_freed_ranges() -> None:
    _marlin_wna16_freed_ranges.clear()


def register_marlin_wna16_freed_tensor(
    *,
    tensor: torch.Tensor,
    layer_id: int | None,
    component: str,
    owner: str,
    released: bool,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = tensor_summary(tensor)
    record: dict[str, Any] = {
        "event": "dsv4_marlin_wna16_freed_range",
        "stage": stage,
        "owner": owner,
        "layer_id": None if layer_id is None else int(layer_id),
        "component": component,
        "released": bool(released),
        **summary,
    }
    if extra:
        record["extra"] = extra
    if released and summary.get("present") and int(summary.get("bytes", 0)) > 0:
        _marlin_wna16_freed_ranges.append(dict(record))
    if marlin_wna16_release_ledger_enabled():
        append_jsonl("marlin_wna16_freed_ranges", record)
    return record


def get_marlin_wna16_freed_ranges() -> list[dict[str, Any]]:
    return list(_marlin_wna16_freed_ranges)


def record_owner_tensor(
    *,
    owner_label: str,
    stage: str,
    tensor: torch.Tensor | None,
    include_integrity: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not marlin_wna16_release_ledger_enabled() and not include_integrity:
        return None
    if tensor is None:
        record = {
            "event": "dsv4_marlin_wna16_owner_allocation",
            "owner": owner_label,
            "stage": stage,
            "tensor": {"present": False},
            "overlaps_freed_range": False,
        }
        if extra:
            record["extra"] = extra
        append_jsonl("marlin_wna16_owner_ledger", record)
        return record

    summary = tensor_integrity_summary(tensor) if include_integrity else tensor_summary(tensor)
    overlap = _find_freed_range_overlap(summary)
    nearest = _nearest_freed_range(summary)
    record = {
        "event": "dsv4_marlin_wna16_owner_allocation",
        "owner": owner_label,
        "stage": stage,
        "tensor": summary,
        "overlaps_freed_range": overlap is not None,
        "overlap_freed_range": overlap,
        "nearest_freed_range": nearest,
    }
    if extra:
        record["extra"] = extra
    append_jsonl("marlin_wna16_owner_ledger", record)
    return record


def record_owner_tensors(
    *,
    owner_prefix: str,
    stage: str,
    tensors: dict[str, torch.Tensor | None],
    include_integrity: bool = False,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not marlin_wna16_release_ledger_enabled() and not include_integrity:
        return records
    for name, tensor in tensors.items():
        record = record_owner_tensor(
            owner_label=f"{owner_prefix}.{name}",
            stage=stage,
            tensor=tensor,
            include_integrity=include_integrity,
            extra=extra,
        )
        if record is not None:
            records.append(record)
    return records


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


def _range_from_summary(summary: dict[str, Any]) -> tuple[int, int] | None:
    if not summary.get("present", False):
        return None
    start = int(summary.get("start") or summary.get("data_ptr") or 0)
    nbytes = int(summary.get("bytes", 0))
    if start <= 0 or nbytes <= 0:
        return None
    return start, start + nbytes


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def _range_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    if _ranges_overlap(a, b):
        return 0
    if a[1] <= b[0]:
        return int(b[0] - a[1])
    return int(a[0] - b[1])


def _freed_range_brief(record: dict[str, Any], *, distance_bytes: int | None = None) -> dict[str, Any]:
    brief: dict[str, Any] = {
        "owner": record.get("owner"),
        "layer_id": record.get("layer_id"),
        "component": record.get("component"),
        "data_ptr": int(record.get("data_ptr", 0) or 0),
        "start": int(record.get("start", 0) or 0),
        "end": int(record.get("end", 0) or 0),
        "bytes": int(record.get("bytes", 0) or 0),
        "dtype": record.get("dtype"),
        "shape": record.get("shape"),
        "stage": record.get("stage"),
    }
    if distance_bytes is not None:
        brief["distance_bytes"] = int(distance_bytes)
    return brief


def _find_freed_range_overlap(summary: dict[str, Any]) -> dict[str, Any] | None:
    tensor_range = _range_from_summary(summary)
    if tensor_range is None:
        return None
    for freed in _marlin_wna16_freed_ranges:
        freed_range = _range_from_summary(freed)
        if freed_range is not None and _ranges_overlap(tensor_range, freed_range):
            overlap_start = max(tensor_range[0], freed_range[0])
            overlap_end = min(tensor_range[1], freed_range[1])
            brief = _freed_range_brief(freed, distance_bytes=0)
            brief["overlap_start"] = int(overlap_start)
            brief["overlap_end"] = int(overlap_end)
            brief["overlap_bytes"] = int(overlap_end - overlap_start)
            return brief
    return None


def _nearest_freed_range(summary: dict[str, Any]) -> dict[str, Any] | None:
    tensor_range = _range_from_summary(summary)
    if tensor_range is None or not _marlin_wna16_freed_ranges:
        return None
    best: tuple[int, dict[str, Any]] | None = None
    for freed in _marlin_wna16_freed_ranges:
        freed_range = _range_from_summary(freed)
        if freed_range is None:
            continue
        distance = _range_distance(tensor_range, freed_range)
        if best is None or distance < best[0]:
            best = (distance, freed)
            if distance == 0:
                break
    if best is None:
        return None
    return _freed_range_brief(best[1], distance_bytes=best[0])


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

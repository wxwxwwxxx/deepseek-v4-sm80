from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENABLE_ENV = "MINISGL_DSV4_OWNER_TIMING"
_CUDA_ENV = "MINISGL_DSV4_OWNER_TIMING_CUDA"
_SYNC_HOST_ENV = "MINISGL_DSV4_OWNER_TIMING_SYNC_HOST"
_MAX_SAMPLES_ENV = "MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES"


@dataclass
class _CudaSample:
    seq: int
    label: str
    metadata: dict[str, Any]
    captured: bool
    start: Any
    end: Any


_cuda_samples: list[_CudaSample] = []
_host_samples: list[dict[str, Any]] = []
_counters: dict[tuple[str, str], dict[str, Any]] = {}
_seq = 0


def enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "").strip().lower() in _TRUE_VALUES


def cuda_enabled() -> bool:
    raw = os.environ.get(_CUDA_ENV)
    if raw is None:
        return True
    return raw.strip().lower() in _TRUE_VALUES


def sync_host_enabled() -> bool:
    return os.environ.get(_SYNC_HOST_ENV, "").strip().lower() in _TRUE_VALUES


def reset() -> None:
    global _cuda_samples, _host_samples, _counters, _seq
    _cuda_samples = []
    _host_samples = []
    _counters = {}
    _seq = 0


def current_seq() -> int:
    return _seq


def tensor_metadata(tensor: Any | None) -> dict[str, Any]:
    if tensor is None:
        return {"present": False}
    shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    stride = tuple(int(dim) for dim in getattr(tensor, "stride", lambda: ())())
    data_ptr = int(tensor.data_ptr()) if hasattr(tensor, "data_ptr") else 0
    return {
        "present": True,
        "shape": list(shape),
        "stride": list(stride),
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "device": str(getattr(tensor, "device", "unknown")),
        "is_contiguous": bool(tensor.is_contiguous()) if hasattr(tensor, "is_contiguous") else None,
        "storage_offset": int(tensor.storage_offset()) if hasattr(tensor, "storage_offset") else 0,
        "data_ptr_mod_16": data_ptr % 16 if data_ptr else None,
        "data_ptr_mod_128": data_ptr % 128 if data_ptr else None,
    }


def record_counter(label: str, metadata: dict[str, Any] | None = None, *, value: int = 1) -> None:
    if not enabled():
        return
    key = (label, _stable_json_key(metadata or {}))
    record = _counters.get(key)
    if record is None:
        record = {
            "label": label,
            "metadata": _jsonable(metadata or {}),
            "count": 0,
        }
        _counters[key] = record
    record["count"] = int(record["count"]) + int(value)


class cuda_range:
    def __init__(self, label: str, metadata: dict[str, Any] | None = None) -> None:
        self.label = label
        self.metadata = metadata or {}
        self.active = False
        self.captured = False
        self.start = None
        self.end = None

    def __enter__(self):
        if not enabled():
            return self
        try:
            import torch

            if not torch.cuda.is_available():
                return self
            self.captured = bool(torch.cuda.is_current_stream_capturing())
            self.start = torch.cuda.Event(enable_timing=True, external=self.captured)
            self.end = torch.cuda.Event(enable_timing=True, external=self.captured)
            self.start.record()
            self.active = True
        except Exception:
            self.active = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.active or self.end is None:
            return
        try:
            self.end.record()
            _append_cuda_sample(
                self.label,
                self.metadata,
                captured=self.captured,
                start=self.start,
                end=self.end,
            )
        except Exception:
            return


class host_range:
    def __init__(
        self,
        label: str,
        metadata: dict[str, Any] | None = None,
        *,
        sync_cuda: bool | None = None,
    ) -> None:
        self.label = label
        self.metadata = metadata or {}
        self.sync_cuda = sync_host_enabled() if sync_cuda is None else sync_cuda
        self.active = False
        self.started_at = 0.0

    def __enter__(self):
        if not enabled():
            return self
        _maybe_synchronize(self.sync_cuda)
        self.started_at = time.perf_counter()
        self.active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.active:
            return
        _maybe_synchronize(self.sync_cuda)
        elapsed_ms = (time.perf_counter() - self.started_at) * 1000.0
        _append_host_sample(self.label, self.metadata, elapsed_ms)


def maybe_cuda_range(label: str, metadata: dict[str, Any] | None = None):
    if not enabled() or not cuda_enabled():
        return nullcontext()
    return cuda_range(label, metadata)


def maybe_host_range(
    label: str,
    metadata: dict[str, Any] | None = None,
    *,
    sync_cuda: bool | None = None,
):
    if not enabled():
        return nullcontext()
    return host_range(label, metadata, sync_cuda=sync_cuda)


def snapshot(
    *,
    resolve_cuda: bool = True,
    captured_shape_filter: list[int] | tuple[int, ...] | set[int] | None = None,
) -> dict[str, Any]:
    captured_shapes = (
        {int(value) for value in captured_shape_filter}
        if captured_shape_filter is not None
        else None
    )
    cuda_entries: list[dict[str, Any]] = []
    cuda_by_label: dict[str, dict[str, Any]] = {}
    cuda_by_label_shape: dict[str, dict[str, Any]] = {}
    for sample in _cuda_samples:
        elapsed_ms: float | None
        error: str | None = None
        if not resolve_cuda:
            elapsed_ms = None
            error = "cuda_elapsed_not_resolved"
        elif sample.captured and not _captured_sample_should_resolve(sample.metadata, captured_shapes):
            elapsed_ms = None
            error = "captured_sample_not_in_replayed_shape_filter"
        else:
            try:
                elapsed_ms = float(sample.start.elapsed_time(sample.end))
            except Exception as exc:
                elapsed_ms = None
                error = f"{type(exc).__name__}: {exc}"
        entry = {
            "seq": int(sample.seq),
            "label": sample.label,
            "metadata": sample.metadata,
            "captured": bool(sample.captured),
            "elapsed_ms": elapsed_ms,
        }
        if error is not None:
            entry["error"] = error
        cuda_entries.append(entry)
        _accumulate_timing(cuda_by_label, sample.label, elapsed_ms, sample.captured, sample.metadata)
        shape_key = _shape_key(sample.label, sample.metadata, sample.captured)
        _accumulate_timing(
            cuda_by_label_shape,
            shape_key,
            elapsed_ms,
            sample.captured,
            sample.metadata,
        )

    host_by_label: dict[str, dict[str, Any]] = {}
    for sample in _host_samples:
        _accumulate_timing(
            host_by_label,
            sample["label"],
            float(sample["elapsed_ms"]),
            False,
            sample.get("metadata", {}),
        )

    return {
        "enabled": enabled(),
        "cuda_enabled": cuda_enabled(),
        "sync_host": sync_host_enabled(),
        "seq": int(_seq),
        "cuda_sample_count": len(cuda_entries),
        "host_sample_count": len(_host_samples),
        "counter_count": len(_counters),
        "cuda_by_label": dict(sorted(cuda_by_label.items())),
        "cuda_by_label_shape": dict(sorted(cuda_by_label_shape.items())),
        "host_by_label": dict(sorted(host_by_label.items())),
        "counters": [dict(value) for value in _counters.values()],
        "cuda_samples": cuda_entries[: _max_samples()],
        "host_samples": list(_host_samples[: _max_samples()]),
        "truncated": len(cuda_entries) > _max_samples() or len(_host_samples) > _max_samples(),
    }


def _captured_sample_should_resolve(
    metadata: dict[str, Any],
    captured_shapes: set[int] | None,
) -> bool:
    if captured_shapes is None:
        return True
    shape = _metadata_shape(metadata)
    if not shape:
        return False
    return int(shape[0]) in captured_shapes


def _append_cuda_sample(
    label: str,
    metadata: dict[str, Any],
    *,
    captured: bool,
    start: Any,
    end: Any,
) -> None:
    global _seq
    _seq += 1
    if len(_cuda_samples) >= _max_samples():
        return
    _cuda_samples.append(
        _CudaSample(
            seq=_seq,
            label=label,
            metadata=_jsonable(metadata),
            captured=captured,
            start=start,
            end=end,
        )
    )


def _append_host_sample(label: str, metadata: dict[str, Any], elapsed_ms: float) -> None:
    global _seq
    _seq += 1
    if len(_host_samples) >= _max_samples():
        return
    _host_samples.append(
        {
            "seq": int(_seq),
            "label": label,
            "metadata": _jsonable(metadata),
            "elapsed_ms": float(elapsed_ms),
        }
    )


def _accumulate_timing(
    target: dict[str, dict[str, Any]],
    key: str,
    elapsed_ms: float | None,
    captured: bool,
    metadata: dict[str, Any],
) -> None:
    bucket = target.setdefault(
        key,
        {
            "count": 0,
            "timed_count": 0,
            "captured_count": 0,
            "total_ms": 0.0,
            "captured_total_ms": 0.0,
            "min_ms": None,
            "max_ms": None,
            "metadata_examples": [],
        },
    )
    bucket["count"] = int(bucket["count"]) + 1
    if captured:
        bucket["captured_count"] = int(bucket["captured_count"]) + 1
    if elapsed_ms is None:
        return
    value = float(elapsed_ms)
    bucket["timed_count"] = int(bucket["timed_count"]) + 1
    bucket["total_ms"] = float(bucket["total_ms"]) + value
    if captured:
        bucket["captured_total_ms"] = float(bucket["captured_total_ms"]) + value
    bucket["min_ms"] = value if bucket["min_ms"] is None else min(float(bucket["min_ms"]), value)
    bucket["max_ms"] = value if bucket["max_ms"] is None else max(float(bucket["max_ms"]), value)
    examples = bucket["metadata_examples"]
    if len(examples) < 4:
        compact = _compact_metadata(metadata)
        if compact not in examples:
            examples.append(compact)


def _shape_key(label: str, metadata: dict[str, Any], captured: bool) -> str:
    shape = _metadata_shape(metadata)
    return f"{label}|captured={int(captured)}|shape={shape}"


def _metadata_shape(metadata: dict[str, Any]) -> list[int]:
    tensor = metadata.get("input") or metadata.get("x") or metadata.get("tensor") or {}
    shape = tensor.get("shape", [])
    return [int(dim) for dim in shape] if isinstance(shape, list) else []


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("owner_label", "comm_label", "phase", "batch_size", "padded_size", "reduce"):
        if key in metadata:
            compact[key] = metadata[key]
    for key in ("input", "x", "tensor", "weight", "output"):
        value = metadata.get(key)
        if isinstance(value, dict) and "shape" in value:
            compact[key] = {
                "shape": value.get("shape"),
                "stride": value.get("stride"),
                "dtype": value.get("dtype"),
                "is_contiguous": value.get("is_contiguous"),
            }
    return compact or _jsonable(metadata)


def _maybe_synchronize(sync_cuda: bool) -> None:
    if not sync_cuda:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _max_samples() -> int:
    try:
        return max(1, int(os.environ.get(_MAX_SAMPLES_ENV, "20000")))
    except ValueError:
        return 20000


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_json_key(value: Any) -> str:
    import json

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))

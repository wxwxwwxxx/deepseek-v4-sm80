from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}
_ENABLE_ENV = "MINISGL_DSV4_LONG_PREFILL_TIMING"
_MAX_EVENTS_ENV = "MINISGL_DSV4_LONG_PREFILL_TIMING_MAX_EVENTS"
_MODE_ENV = "MINISGL_DSV4_LONG_PREFILL_TIMING_MODE"
_EVENT_CONTEXTS_ENV = "MINISGL_DSV4_LONG_PREFILL_EVENT_CONTEXTS"


@dataclass
class _EventSample:
    owner: str
    metadata: dict[str, Any]
    start: Any
    end: Any


_samples: list[_EventSample] = []
_host_samples: list[dict[str, Any]] = []
_counters: dict[tuple[str, str], dict[str, Any]] = {}
_batch_metadata: dict[str, Any] | None = None
_dropped_events = 0


def enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "").strip().lower() in _TRUE_VALUES


def mode() -> str:
    value = os.environ.get(_MODE_ENV, "events").strip().lower()
    return value if value in {"events", "coarse_events", "nvtx"} else "events"


def reset() -> None:
    global _samples, _host_samples, _counters, _batch_metadata, _dropped_events
    _samples = []
    _host_samples = []
    _counters = {}
    _batch_metadata = None
    _dropped_events = 0


@contextmanager
def batch_context(batch):
    """Attach stable CPU-side chunk geometry to all nested owner samples."""
    global _batch_metadata
    if not enabled() or not bool(getattr(batch, "is_prefill", False)):
        yield
        return
    previous = _batch_metadata
    reqs = getattr(batch, "reqs", ())
    padded_reqs = getattr(batch, "padded_reqs", reqs)
    committed_context = max(
        (int(getattr(req, "device_len", 0)) for req in reqs),
        default=0,
    )
    rows = sum(int(getattr(req, "extend_len", 0)) for req in padded_reqs)
    _batch_metadata = {
        "phase": str(getattr(batch, "phase", "prefill")),
        "batch_size": int(getattr(batch, "size", len(reqs))),
        "padded_size": int(getattr(batch, "padded_size", len(padded_reqs))),
        "rows": int(rows),
        "committed_context": int(committed_context),
    }
    try:
        yield
    finally:
        _batch_metadata = previous


class cuda_range:
    def __init__(self, owner: str, metadata: dict[str, Any] | None = None) -> None:
        self.owner = owner
        self.metadata = metadata or {}
        self.start = None
        self.end = None
        self.active = False

    def __enter__(self):
        if not enabled() or _batch_metadata is None:
            return self
        try:
            import torch

            if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
                return self
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            self.active = True
        except Exception:
            self.active = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _dropped_events
        if not self.active or self.end is None:
            return
        try:
            self.end.record()
            if len(_samples) >= _max_events():
                _dropped_events += 1
                return
            _samples.append(
                _EventSample(
                    owner=self.owner,
                    metadata={**(_batch_metadata or {}), **_jsonable(self.metadata)},
                    start=self.start,
                    end=self.end,
                )
            )
        except Exception:
            return


def maybe_cuda_range(owner: str, metadata: dict[str, Any] | None = None):
    if not enabled() or _batch_metadata is None:
        return nullcontext()
    raw_contexts = os.environ.get(_EVENT_CONTEXTS_ENV, "").strip()
    if raw_contexts:
        contexts = {
            int(value.strip()) for value in raw_contexts.split(",") if value.strip()
        }
        if int(_batch_metadata.get("committed_context", 0)) not in contexts:
            return nullcontext()
    if mode() == "coarse_events" and owner in {
        "attention_total",
        "moe_route",
        "moe_route_prepare",
        "moe_marlin_experts",
        "moe_shared_expert",
        "moe_reduce",
    }:
        return nullcontext()
    if mode() == "nvtx":
        try:
            import torch

            merged = {**_batch_metadata, **(metadata or {})}
            fields = [f"owner={owner}"]
            for key in (
                "committed_context",
                "layer_id",
                "compress_ratio",
                "max_c4_seq_len",
                "slice_rows",
            ):
                if key in merged:
                    fields.append(f"{key}={merged[key]}")
            return torch.cuda.nvtx.range("dsv4_long_prefill:" + ":".join(fields))
        except Exception:
            return nullcontext()
    return cuda_range(owner, metadata)


def record_counter(
    label: str,
    metadata: dict[str, Any] | None = None,
    *,
    value: int = 1,
) -> None:
    if not enabled() or (_batch_metadata is None and not metadata):
        return
    merged = {**(_batch_metadata or {}), **_jsonable(metadata or {})}
    key = (label, _stable_json_key(merged))
    entry = _counters.setdefault(
        key,
        {"label": label, "metadata": merged, "value": 0},
    )
    entry["value"] = int(entry["value"]) + int(value)


def record_host(
    owner: str,
    elapsed_ms: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not enabled():
        return
    _host_samples.append(
        {
            "owner": owner,
            "elapsed_ms": float(elapsed_ms),
            "metadata": {**(_batch_metadata or {}), **_jsonable(metadata or {})},
        }
    )


def snapshot(*, resolve_cuda: bool = True) -> dict[str, Any]:
    by_owner: dict[str, dict[str, Any]] = {}
    by_context: dict[str, dict[str, dict[str, Any]]] = {}
    host_by_owner: dict[str, dict[str, Any]] = {}
    host_by_context: dict[str, dict[str, dict[str, Any]]] = {}
    errors = 0
    for sample in _samples:
        elapsed_ms = None
        if resolve_cuda:
            try:
                elapsed_ms = float(sample.start.elapsed_time(sample.end))
            except Exception:
                errors += 1
        _accumulate(by_owner, sample.owner, elapsed_ms, sample.metadata)
        context = str(int(sample.metadata.get("committed_context", 0)))
        _accumulate(by_context.setdefault(context, {}), sample.owner, elapsed_ms, sample.metadata)
    for sample in _host_samples:
        metadata = sample["metadata"]
        owner = sample["owner"]
        elapsed_ms = float(sample["elapsed_ms"])
        _accumulate(host_by_owner, owner, elapsed_ms, metadata)
        context = str(int(metadata.get("committed_context", 0)))
        _accumulate(host_by_context.setdefault(context, {}), owner, elapsed_ms, metadata)
    return {
        "enabled": enabled(),
        "mode": mode(),
        "contract": "cuda_events_resolved_once_after_case_no_per_layer_host_sync",
        "event_count": len(_samples),
        "dropped_event_count": int(_dropped_events),
        "resolve_error_count": int(errors),
        "by_owner": dict(sorted(by_owner.items())),
        "by_committed_context": {
            context: dict(sorted(owners.items()))
            for context, owners in sorted(by_context.items(), key=lambda item: int(item[0]))
        },
        "host_by_owner": dict(sorted(host_by_owner.items())),
        "host_by_committed_context": {
            context: dict(sorted(owners.items()))
            for context, owners in sorted(
                host_by_context.items(), key=lambda item: int(item[0])
            )
        },
        "counters": [dict(value) for value in _counters.values()],
    }


def _accumulate(
    target: dict[str, dict[str, Any]],
    owner: str,
    elapsed_ms: float | None,
    metadata: dict[str, Any],
) -> None:
    bucket = target.setdefault(
        owner,
        {
            "count": 0,
            "timed_count": 0,
            "total_ms": 0.0,
            "min_ms": None,
            "max_ms": None,
            "metadata_examples": [],
        },
    )
    bucket["count"] = int(bucket["count"]) + 1
    if elapsed_ms is not None:
        bucket["timed_count"] = int(bucket["timed_count"]) + 1
        bucket["total_ms"] = float(bucket["total_ms"]) + elapsed_ms
        bucket["min_ms"] = elapsed_ms if bucket["min_ms"] is None else min(bucket["min_ms"], elapsed_ms)
        bucket["max_ms"] = elapsed_ms if bucket["max_ms"] is None else max(bucket["max_ms"], elapsed_ms)
    examples = bucket["metadata_examples"]
    compact = {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "committed_context",
            "rows",
            "layer_id",
            "compress_ratio",
            "max_c4_seq_len",
            "slice_rows",
            "backend",
        }
    }
    if compact not in examples and len(examples) < 8:
        examples.append(compact)


def _max_events() -> int:
    try:
        return max(1, int(os.environ.get(_MAX_EVENTS_ENV, "100000")))
    except ValueError:
        return 100000


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

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch

ROW0_LAYER_PARITY_ENV = "MINISGL_DSV4_MTP_ROW0_LAYER_PARITY"
ROW0_LAYER_PARITY_ATOL_ENV = "MINISGL_DSV4_MTP_ROW0_LAYER_PARITY_ATOL"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def row0_layer_parity_enabled() -> bool:
    return env_flag(ROW0_LAYER_PARITY_ENV)


def row0_layer_parity_atol() -> float:
    raw = os.environ.get(ROW0_LAYER_PARITY_ATOL_ENV, "").strip()
    if not raw:
        return 1.0e-3
    try:
        return float(raw)
    except ValueError:
        return 1.0e-3


def reset_row0_layer_trace(batch: Any, *, mode: str) -> None:
    if not row0_layer_parity_enabled():
        return
    setattr(batch, "_dsv4_mtp_row0_layer_trace", [])
    setattr(batch, "_dsv4_mtp_attention_backend_trace", [])
    setattr(batch, "_dsv4_mtp_row0_layer_trace_mode", mode)


def record_row0_tensor(
    batch: Any,
    name: str,
    tensor: torch.Tensor | None,
    *,
    layer_id: int | None = None,
    boundary: str | None = None,
) -> None:
    if not row0_layer_parity_enabled():
        return
    if batch is None:
        return
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return
    if tensor.numel() == 0 or tensor.ndim == 0:
        return
    if tensor.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return
    try:
        if layer_id is None:
            layer_id = _infer_layer_id(name)
        row0 = tensor.detach()[0].float().contiguous().cpu()
        trace = getattr(batch, "_dsv4_mtp_row0_layer_trace", None)
        if trace is None:
            trace = []
            setattr(batch, "_dsv4_mtp_row0_layer_trace", trace)
        trace.append(
            {
                "name": name,
                "layer_id": None if layer_id is None else int(layer_id),
                "boundary": boundary or name,
                "shape": [int(x) for x in tensor.shape],
                "row0_shape": [int(x) for x in row0.shape],
                "dtype": str(tensor.dtype),
                "summary": _tensor_summary(row0),
                "_row0_tensor": row0,
            }
        )
    except Exception as exc:
        trace = getattr(batch, "_dsv4_mtp_row0_layer_trace", None)
        if trace is None:
            trace = []
            setattr(batch, "_dsv4_mtp_row0_layer_trace", trace)
        trace.append(
            {
                "name": name,
                "layer_id": None if layer_id is None else int(layer_id),
                "boundary": boundary or name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def record_attention_backend(
    batch: Any,
    *,
    layer_id: int,
    backend: str,
    rows: int,
    metadata: Any,
    compress_ratio: int,
) -> None:
    if not row0_layer_parity_enabled():
        return
    if batch is None:
        return
    trace = getattr(batch, "_dsv4_mtp_attention_backend_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_mtp_attention_backend_trace", trace)
    trace.append(
        {
            "layer_id": int(layer_id),
            "backend": str(backend),
            "rows": int(rows),
            "compress_ratio": int(compress_ratio),
            "target_verify_decode_rows": bool(
                getattr(metadata, "target_verify_decode_rows", False)
            ),
            "max_seqlen_q": int(getattr(metadata, "max_seqlen_q", -1)),
            "max_seqlen_k": int(getattr(metadata, "max_seqlen_k", -1)),
            "phase": str(getattr(batch, "phase", "")),
            "is_target_verify": bool(
                getattr(batch, "dsv4_target_verify_metadata", None) is not None
            ),
        }
    )


def get_row0_layer_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_mtp_row0_layer_trace", None)
    if not isinstance(trace, list):
        return []
    return trace


def export_row0_layer_trace(batch_or_trace: Any) -> list[dict[str, Any]]:
    trace = (
        batch_or_trace
        if isinstance(batch_or_trace, list)
        else get_row0_layer_trace(batch_or_trace)
    )
    return [_strip_private(entry) for entry in trace]


def export_attention_backend_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_mtp_attention_backend_trace", None)
    if not isinstance(trace, list):
        return []
    return [dict(entry) for entry in trace]


def compare_row0_layer_traces(
    lhs: list[dict[str, Any]],
    rhs: list[dict[str, Any]],
    *,
    lhs_label: str,
    rhs_label: str,
    atol: float | None = None,
) -> dict[str, Any]:
    threshold = row0_layer_parity_atol() if atol is None else float(atol)
    rhs_by_name: dict[str, dict[str, Any]] = {}
    for entry in rhs:
        name = entry.get("name")
        if isinstance(name, str) and name not in rhs_by_name:
            rhs_by_name[name] = entry

    comparisons: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    missing_rhs: list[str] = []
    for lhs_entry in lhs:
        name = lhs_entry.get("name")
        if not isinstance(name, str):
            continue
        rhs_entry = rhs_by_name.get(name)
        if rhs_entry is None:
            missing_rhs.append(name)
            continue
        comparison = _compare_trace_entry(
            lhs_entry,
            rhs_entry,
            lhs_label=lhs_label,
            rhs_label=rhs_label,
            atol=threshold,
        )
        comparisons.append(comparison)
        if first_mismatch is None and bool(comparison.get("is_mismatch", False)):
            first_mismatch = comparison

    lhs_names = {
        entry.get("name") for entry in lhs if isinstance(entry.get("name"), str)
    }
    missing_lhs = [
        str(entry.get("name"))
        for entry in rhs
        if isinstance(entry.get("name"), str) and entry.get("name") not in lhs_names
    ]
    return {
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "atol": float(threshold),
        "num_lhs_entries": int(len(lhs)),
        "num_rhs_entries": int(len(rhs)),
        "num_compared": int(len(comparisons)),
        "missing_from_rhs": missing_rhs[:32],
        "missing_from_lhs": missing_lhs[:32],
        "first_mismatch": first_mismatch,
        "comparisons": comparisons,
    }


def _compare_trace_entry(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    lhs_label: str,
    rhs_label: str,
    atol: float,
) -> dict[str, Any]:
    lhs_tensor = lhs.get("_row0_tensor")
    rhs_tensor = rhs.get("_row0_tensor")
    base: dict[str, Any] = {
        "name": lhs.get("name"),
        "layer_id": lhs.get("layer_id"),
        "boundary": lhs.get("boundary"),
        "lhs_dtype": lhs.get("dtype"),
        "rhs_dtype": rhs.get("dtype"),
        "lhs_shape": lhs.get("row0_shape"),
        "rhs_shape": rhs.get("row0_shape"),
    }
    if not isinstance(lhs_tensor, torch.Tensor) or not isinstance(rhs_tensor, torch.Tensor):
        base.update(
            {
                "is_mismatch": True,
                "error": "missing row0 tensor for comparison",
            }
        )
        return base
    lhs_flat = lhs_tensor.float().reshape(-1)
    rhs_flat = rhs_tensor.float().reshape(-1)
    if lhs_flat.shape != rhs_flat.shape:
        base.update(
            {
                "is_mismatch": True,
                "error": "row0 tensor shape mismatch",
                "lhs_numel": int(lhs_flat.numel()),
                "rhs_numel": int(rhs_flat.numel()),
            }
        )
        return base
    if lhs_flat.numel() == 0:
        base.update(
            {
                "is_mismatch": False,
                "max_abs_delta": 0.0,
                "mean_abs_delta": 0.0,
                "relative_delta": 0.0,
                "top_diffs": [],
            }
        )
        return base
    delta = (lhs_flat - rhs_flat).abs()
    max_abs = float(delta.max().item())
    mean_abs = float(delta.mean().item())
    denom = max(
        float(lhs_flat.abs().max().item()),
        float(rhs_flat.abs().max().item()),
        1.0e-12,
    )
    k = min(5, int(delta.numel()))
    top_values, top_indices = torch.topk(delta, k=k)
    top_diffs = []
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        idx = int(index)
        top_diffs.append(
            {
                "flat_index": idx,
                f"{lhs_label}_value": float(lhs_flat[idx].item()),
                f"{rhs_label}_value": float(rhs_flat[idx].item()),
                "abs_delta": float(value),
            }
        )
    base.update(
        {
            "is_mismatch": bool(max_abs > float(atol)),
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
            "relative_delta": float(max_abs / denom),
            "top_diffs": top_diffs,
            "lhs_checksum": lhs.get("summary", {}).get("checksum"),
            "rhs_checksum": rhs.get("summary", {}).get("checksum"),
        }
    )
    return base


def _strip_private(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def _infer_layer_id(name: str) -> int | None:
    if not name.startswith("layer"):
        return None
    digits = []
    for char in name[len("layer") :]:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return None
    try:
        return int("".join(digits))
    except ValueError:
        return None


def _tensor_summary(row0: torch.Tensor) -> dict[str, Any]:
    flat = row0.reshape(-1)
    summary: dict[str, Any] = {
        "numel": int(flat.numel()),
        "shape": [int(x) for x in row0.shape],
        "dtype": str(row0.dtype),
    }
    if flat.numel() == 0:
        return summary
    finite = flat[torch.isfinite(flat)]
    if finite.numel() > 0:
        finite_float = finite.float()
        summary.update(
            {
                "min": float(finite_float.min().item()),
                "max": float(finite_float.max().item()),
                "mean": float(finite_float.mean().item()),
                "sum": float(finite_float.sum().item()),
                "abs_sum": float(finite_float.abs().sum().item()),
                "l2": float(torch.linalg.vector_norm(finite_float).item()),
            }
        )
    head = flat[: min(8, int(flat.numel()))].tolist()
    tail = flat[-min(8, int(flat.numel())) :].tolist()
    summary["head"] = _jsonable(head)
    summary["tail"] = _jsonable(tail)
    summary["checksum"] = _checksum(flat)
    return summary


def _checksum(flat: torch.Tensor) -> str:
    contiguous = flat.float().contiguous()
    try:
        payload = contiguous.numpy().tobytes()
    except Exception:
        payload = bytes(str(contiguous.tolist()), encoding="utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _jsonable(values: list[Any]) -> list[Any]:
    out = []
    for value in values:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            out.append(bool(value))
        elif isinstance(value, int):
            out.append(int(value))
        else:
            out.append(float(value))
    return out


__all__ = [
    "ROW0_LAYER_PARITY_ENV",
    "ROW0_LAYER_PARITY_ATOL_ENV",
    "compare_row0_layer_traces",
    "env_flag",
    "export_attention_backend_trace",
    "export_row0_layer_trace",
    "get_row0_layer_trace",
    "record_attention_backend",
    "record_row0_tensor",
    "reset_row0_layer_trace",
    "row0_layer_parity_atol",
    "row0_layer_parity_enabled",
]

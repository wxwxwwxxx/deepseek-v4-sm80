from __future__ import annotations

import hashlib
import os
from typing import Any

import torch

ROW0_LAYER_PARITY_ENV = "MINISGL_DSV4_MTP_ROW0_LAYER_PARITY"
ROW0_LAYER_PARITY_ATOL_ENV = "MINISGL_DSV4_MTP_ROW0_LAYER_PARITY_ATOL"
OPERATOR_PARITY_ENV = "MINISGL_DSV4_MTP_OPERATOR_PARITY"
OPERATOR_PARITY_OPERATORS_ENV = "MINISGL_DSV4_MTP_OPERATOR_PARITY_OPERATORS"
OPERATOR_PARITY_LAYERS_ENV = "MINISGL_DSV4_MTP_OPERATOR_PARITY_LAYERS"
OPERATOR_PARITY_ATOL_ENV = "MINISGL_DSV4_MTP_OPERATOR_PARITY_ATOL"
OPERATOR_PARITY_RTOL_ENV = "MINISGL_DSV4_MTP_OPERATOR_PARITY_RTOL"
ROW_TRACE_ROWS_ENV = "MINISGL_DSV4_MTP_ROW_TRACE_ROWS"
ROW_TENSOR_TRACE_ENV = "MINISGL_DSV4_MTP_ROW_TENSOR_TRACE"
LAYER2_SWA_LIFECYCLE_TRACE_ENV = "MINISGL_DSV4_LAYER2_SWA_LIFECYCLE_TRACE"
SWA_LIFECYCLE_TRACE_LAYERS_ENV = "MINISGL_DSV4_SWA_LIFECYCLE_TRACE_LAYERS"
WO_A_PROJECTION_ORACLE_ENV = "MINISGL_DSV4_MTP_WO_A_PROJECTION_ORACLE"
WO_A_PROJECTION_ORACLE_LAYERS_ENV = "MINISGL_DSV4_MTP_WO_A_PROJECTION_ORACLE_LAYERS"
WO_B_PROJECTION_ORACLE_ENV = "MINISGL_DSV4_MTP_WO_B_PROJECTION_ORACLE"
WO_B_PROJECTION_ORACLE_LAYERS_ENV = "MINISGL_DSV4_MTP_WO_B_PROJECTION_ORACLE_LAYERS"
MOE_CONTRACT_ORACLE_ENV = "MINISGL_DSV4_MTP_MOE_CONTRACT_ORACLE"
MOE_CONTRACT_ORACLE_LAYERS_ENV = "MINISGL_DSV4_MTP_MOE_CONTRACT_ORACLE_LAYERS"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def row0_layer_parity_enabled() -> bool:
    return env_flag(ROW0_LAYER_PARITY_ENV) or operator_parity_enabled()


def row_tensor_trace_enabled() -> bool:
    return (
        row0_layer_parity_enabled()
        or env_flag(ROW_TENSOR_TRACE_ENV)
        or moe_contract_oracle_enabled()
    )


def layer2_swa_lifecycle_trace_enabled() -> bool:
    return env_flag(LAYER2_SWA_LIFECYCLE_TRACE_ENV)


def swa_lifecycle_trace_layers() -> set[int]:
    raw = os.environ.get(SWA_LIFECYCLE_TRACE_LAYERS_ENV, "2").strip()
    try:
        selected = _parse_int_filter(raw)
    except ValueError:
        return {2}
    if selected is None:
        return {2}
    return {int(layer_id) for layer_id in selected}


def swa_lifecycle_trace_enabled(layer_id: int | None = None) -> bool:
    if not layer2_swa_lifecycle_trace_enabled():
        return False
    if layer_id is None:
        return True
    return int(layer_id) in swa_lifecycle_trace_layers()


def row0_layer_parity_atol() -> float:
    raw = os.environ.get(ROW0_LAYER_PARITY_ATOL_ENV, "").strip()
    if not raw:
        return 1.0e-3
    try:
        return float(raw)
    except ValueError:
        return 1.0e-3


def operator_parity_enabled(operator_name: str | None = None) -> bool:
    if not env_flag(OPERATOR_PARITY_ENV):
        return False
    raw = os.environ.get(OPERATOR_PARITY_OPERATORS_ENV, "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return True
    if operator_name is None:
        return True
    selected = {part.strip() for part in raw.split(",") if part.strip()}
    return operator_name in selected


def operator_parity_layer_enabled(layer_id: int | None = None) -> bool:
    if layer_id is None:
        return True
    raw = os.environ.get(OPERATOR_PARITY_LAYERS_ENV, "").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return True
    try:
        selected = _parse_int_filter(raw)
    except ValueError:
        return True
    return selected is None or int(layer_id) in selected


def operator_parity_atol() -> float:
    raw = os.environ.get(OPERATOR_PARITY_ATOL_ENV, "").strip()
    if not raw:
        return row0_layer_parity_atol()
    try:
        return float(raw)
    except ValueError:
        return row0_layer_parity_atol()


def operator_parity_rtol() -> float:
    raw = os.environ.get(OPERATOR_PARITY_RTOL_ENV, "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _parse_int_filter(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw or raw.lower() in {"all", "*"}:
        return None
    selected: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    return selected


def wo_a_projection_oracle_enabled(layer_id: int | None = None) -> bool:
    if not env_flag(WO_A_PROJECTION_ORACLE_ENV):
        return False
    raw = os.environ.get(WO_A_PROJECTION_ORACLE_LAYERS_ENV, "0").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return True
    if layer_id is None:
        return True
    selected = _parse_int_filter(raw)
    return selected is None or int(layer_id) in selected


def wo_b_projection_oracle_enabled(layer_id: int | None = None) -> bool:
    if not env_flag(WO_B_PROJECTION_ORACLE_ENV):
        return False
    raw = os.environ.get(WO_B_PROJECTION_ORACLE_LAYERS_ENV, "0").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return True
    if layer_id is None:
        return True
    selected = _parse_int_filter(raw)
    return selected is None or int(layer_id) in selected


def moe_contract_oracle_enabled(layer_id: int | None = None) -> bool:
    if not env_flag(MOE_CONTRACT_ORACLE_ENV):
        return False
    raw = os.environ.get(MOE_CONTRACT_ORACLE_LAYERS_ENV, "0").strip()
    if not raw or raw.lower() in {"all", "*"}:
        return True
    if layer_id is None:
        return True
    try:
        selected = _parse_int_filter(raw)
    except ValueError:
        return True
    return selected is None or int(layer_id) in selected


def reset_row0_layer_trace(batch: Any, *, mode: str) -> None:
    if not row_tensor_trace_enabled():
        return
    setattr(batch, "_dsv4_mtp_row0_layer_trace", [])
    setattr(batch, "_dsv4_mtp_attention_backend_trace", [])
    setattr(batch, "_dsv4_mtp_operator_trace", [])
    setattr(batch, "_dsv4_mtp_wo_a_projection_oracle_trace", [])
    setattr(batch, "_dsv4_mtp_wo_b_projection_oracle_trace", [])
    setattr(batch, "_dsv4_mtp_moe_contract_oracle_trace", [])
    setattr(batch, "_dsv4_mtp_moe_microbatch_runtime_trace", [])
    setattr(batch, "_dsv4_layer2_swa_store_trace", [])
    setattr(batch, "_dsv4_mtp_row0_layer_trace_mode", mode)


def record_row0_tensor(
    batch: Any,
    name: str,
    tensor: torch.Tensor | None,
    *,
    layer_id: int | None = None,
    boundary: str | None = None,
) -> None:
    if not row_tensor_trace_enabled():
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
        selected_rows = _selected_trace_rows(tensor)
        row_tensors: dict[int, torch.Tensor] = {}
        row_summaries = []
        for row_idx in selected_rows:
            row = tensor.detach()[int(row_idx)].contiguous().cpu()
            row_tensors[int(row_idx)] = row
            row_summaries.append(
                {
                    "row": int(row_idx),
                    "summary": _tensor_summary(row),
                    "raw_sha256": _raw_checksum(row),
                }
            )
        row0_raw = row_tensors.get(0)
        if row0_raw is None:
            row0_raw = tensor.detach()[0].contiguous().cpu()
        row0 = row0_raw.float()
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
                "tensor_metadata": _tensor_metadata(tensor),
                "summary": _tensor_summary(row0),
                "row_summaries": row_summaries,
                "_row0_tensor": row0,
                "_row_tensors": row_tensors,
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


def clone_operator_row0_input(
    operator_name: str,
    tensor: torch.Tensor | None,
) -> torch.Tensor | None:
    if not operator_parity_enabled(operator_name):
        return None
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    if tensor.numel() == 0 or tensor.ndim == 0:
        return None
    if tensor.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return None
        except Exception:
            return None
    try:
        return tensor.detach()[0].contiguous().cpu()
    except Exception:
        return None


def record_operator_capture(
    batch: Any,
    *,
    operator_name: str,
    layer_id: int,
    input_row0: torch.Tensor | None,
    input_tensor: torch.Tensor | None = None,
    output_tensor: torch.Tensor | None,
    positions: torch.Tensor | None,
    path: str,
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    private: dict[str, Any] | None = None,
) -> None:
    if not operator_parity_enabled(operator_name):
        return
    if not operator_parity_layer_enabled(layer_id):
        return
    if batch is None:
        return
    trace = getattr(batch, "_dsv4_mtp_operator_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_mtp_operator_trace", trace)
    entry: dict[str, Any] = {
        "operator_name": str(operator_name),
        "layer_id": int(layer_id),
        "path": str(path),
        "mode": str(getattr(batch, "_dsv4_mtp_row0_layer_trace_mode", "")),
        "is_target_verify": bool(getattr(batch, "dsv4_target_verify_metadata", None) is not None),
        "params": _json_dict(params or {}),
        "batch_context": _operator_batch_context(batch, positions),
    }
    if extra:
        entry["extra"] = _json_dict(extra)
    if private:
        for key, value in private.items():
            entry[f"_{key}"] = value
    try:
        selected_source = output_tensor if isinstance(output_tensor, torch.Tensor) else input_tensor
        selected_rows = (
            _selected_trace_rows(selected_source)
            if isinstance(selected_source, torch.Tensor)
            else [0]
        )
        input_row_tensors: dict[int, torch.Tensor] = {}
        output_row_tensors: dict[int, torch.Tensor] = {}
        row_records: list[dict[str, Any]] = []
        if isinstance(input_row0, torch.Tensor):
            row = input_row0.detach().contiguous().cpu()
            entry["input_tensor_metadata"] = _tensor_metadata(input_row0)
            entry["input_summary"] = _tensor_summary(row.float())
            entry["_input_row0_tensor"] = row
        else:
            entry["input_tensor_metadata"] = {"available": False}
        if isinstance(output_tensor, torch.Tensor) and output_tensor.numel() > 0:
            row = output_tensor.detach()[0].contiguous().cpu()
            entry["output_tensor_metadata"] = _tensor_metadata(output_tensor, row0=True)
            entry["output_summary"] = _tensor_summary(row.float())
            entry["_output_row0_tensor"] = row
        else:
            entry["output_tensor_metadata"] = {"available": False}
        for row_idx in selected_rows:
            row_record: dict[str, Any] = {
                "row": int(row_idx),
                "context": _operator_row_context(batch, positions, int(row_idx)),
            }
            if (
                isinstance(input_tensor, torch.Tensor)
                and input_tensor.ndim > 0
                and 0 <= int(row_idx) < int(input_tensor.shape[0])
            ):
                input_row = input_tensor.detach()[int(row_idx)].contiguous().cpu()
                input_row_tensors[int(row_idx)] = input_row
                row_record["input"] = _row_tensor_record(input_row)
            elif int(row_idx) == 0 and isinstance(input_row0, torch.Tensor):
                input_row = input_row0.detach().contiguous().cpu()
                input_row_tensors[0] = input_row
                row_record["input"] = _row_tensor_record(input_row)
            else:
                row_record["input"] = {"available": False}
            if (
                isinstance(output_tensor, torch.Tensor)
                and output_tensor.ndim > 0
                and 0 <= int(row_idx) < int(output_tensor.shape[0])
            ):
                output_row = output_tensor.detach()[int(row_idx)].contiguous().cpu()
                output_row_tensors[int(row_idx)] = output_row
                row_record["output"] = _row_tensor_record(output_row)
            else:
                row_record["output"] = {"available": False}
            row_records.append(row_record)
        if input_row_tensors:
            entry["_input_row_tensors"] = input_row_tensors
        if output_row_tensors:
            entry["_output_row_tensors"] = output_row_tensors
        entry["selected_rows"] = [int(row_idx) for row_idx in selected_rows]
        entry["row_records"] = row_records
    except Exception as exc:
        entry["error_type"] = type(exc).__name__
        entry["error"] = str(exc)
    trace.append(entry)


def get_operator_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_mtp_operator_trace", None)
    if not isinstance(trace, list):
        return []
    return trace


def export_operator_trace(batch_or_trace: Any) -> list[dict[str, Any]]:
    trace = (
        batch_or_trace
        if isinstance(batch_or_trace, list)
        else get_operator_trace(batch_or_trace)
    )
    return [_strip_private(entry) for entry in trace]


def record_moe_contract_oracle(
    batch: Any,
    *,
    layer_id: int,
    variant: str,
    source_rows: list[int] | tuple[int, ...],
    tensors: dict[str, torch.Tensor | None],
    positions: torch.Tensor | None,
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    reference_tensor: torch.Tensor | None = None,
) -> None:
    if not moe_contract_oracle_enabled(layer_id):
        return
    if batch is None:
        return
    if not tensors:
        return
    try:
        if any(
            isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and torch.cuda.is_current_stream_capturing()
            for tensor in tensors.values()
        ):
            return
    except Exception:
        return

    trace = getattr(batch, "_dsv4_mtp_moe_contract_oracle_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_mtp_moe_contract_oracle_trace", trace)

    source_rows_list = [int(row) for row in source_rows]
    if reference_tensor is None:
        for tensor in tensors.values():
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
                reference_tensor = tensor
                break
    selected_source_rows: set[int] = set()
    if isinstance(reference_tensor, torch.Tensor) and reference_tensor.ndim > 0:
        selected_source_rows.update(_selected_trace_rows(reference_tensor))
    if not selected_source_rows and source_rows_list:
        selected_source_rows.add(source_rows_list[0])

    entry: dict[str, Any] = {
        "variant": str(variant),
        "layer_id": int(layer_id),
        "mode": str(getattr(batch, "_dsv4_mtp_row0_layer_trace_mode", "")),
        "is_target_verify": bool(
            getattr(batch, "dsv4_target_verify_metadata", None) is not None
        ),
        "params": _json_dict(params or {}),
        "extra": _json_dict(extra or {}),
        "source_rows": [int(row) for row in source_rows_list],
        "tensor_metadata": {},
        "row_records": [],
    }
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            entry["tensor_metadata"][str(name)] = _tensor_metadata(tensor)
        else:
            entry["tensor_metadata"][str(name)] = {"available": False}

    try:
        for variant_row, source_row in enumerate(source_rows_list):
            if int(source_row) not in selected_source_rows:
                continue
            record: dict[str, Any] = {
                "row": int(variant_row),
                "source_row": int(source_row),
                "context": _operator_row_context(batch, positions, int(source_row)),
                "boundaries": {},
            }
            for name, tensor in tensors.items():
                if (
                    isinstance(tensor, torch.Tensor)
                    and tensor.ndim > 0
                    and 0 <= int(variant_row) < int(tensor.shape[0])
                ):
                    row_tensor = tensor.detach()[int(variant_row)].contiguous().cpu()
                    record["boundaries"][str(name)] = _row_tensor_record(row_tensor)
                else:
                    record["boundaries"][str(name)] = {"available": False}
            entry["row_records"].append(record)
    except Exception as exc:
        entry["error_type"] = type(exc).__name__
        entry["error"] = str(exc)

    trace.append(entry)


def get_moe_contract_oracle_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_mtp_moe_contract_oracle_trace", None)
    if not isinstance(trace, list):
        return []
    return trace


def export_moe_contract_oracle_trace(batch_or_trace: Any) -> list[dict[str, Any]]:
    trace = (
        batch_or_trace
        if isinstance(batch_or_trace, list)
        else get_moe_contract_oracle_trace(batch_or_trace)
    )
    return [_strip_private(entry) for entry in trace]


def record_moe_microbatch_runtime(batch: Any, record: dict[str, Any]) -> None:
    if batch is None:
        return
    trace = getattr(batch, "_dsv4_mtp_moe_microbatch_runtime_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_mtp_moe_microbatch_runtime_trace", trace)
    trace.append(_json_dict(record))
    if len(trace) > 4096:
        del trace[:-4096]


def get_moe_microbatch_runtime_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_mtp_moe_microbatch_runtime_trace", None)
    if not isinstance(trace, list):
        return []
    return trace


def export_moe_microbatch_runtime_trace(batch_or_trace: Any) -> list[dict[str, Any]]:
    trace = (
        batch_or_trace
        if isinstance(batch_or_trace, list)
        else get_moe_microbatch_runtime_trace(batch_or_trace)
    )
    return [_strip_private(entry) for entry in trace]


def record_attention_backend(
    batch: Any,
    *,
    layer_id: int,
    backend: str,
    rows: int,
    metadata: Any,
    compress_ratio: int,
    query: torch.Tensor | None = None,
    merged_output: torch.Tensor | None = None,
    swa_indices: torch.Tensor | None = None,
    swa_lengths: torch.Tensor | None = None,
    swa_cache: torch.Tensor | None = None,
    compressed_indices: torch.Tensor | None = None,
    compressed_lengths: torch.Tensor | None = None,
    compressed_cache: torch.Tensor | None = None,
) -> None:
    if not row_tensor_trace_enabled():
        return
    if batch is None:
        return
    trace = getattr(batch, "_dsv4_mtp_attention_backend_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_mtp_attention_backend_trace", trace)
    entry = {
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
    detail_enabled = (
        int(layer_id) == 2
        and env_flag("MINISGL_DSV4_NORMAL_PRODUCER_TRACE_LAYER2_ATTENTION_SPLIT")
    )
    if detail_enabled:
        try:
            entry["tensor_metadata"] = _attention_backend_tensor_metadata(
                query=query,
                merged_output=merged_output,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                swa_cache=swa_cache,
                compressed_indices=compressed_indices,
                compressed_lengths=compressed_lengths,
                compressed_cache=compressed_cache,
            )
            entry["row_records"] = _attention_backend_row_records(
                metadata,
                int(rows),
                query=query,
                merged_output=merged_output,
                swa_indices=swa_indices,
                swa_lengths=swa_lengths,
                swa_cache=swa_cache,
                compressed_indices=compressed_indices,
                compressed_lengths=compressed_lengths,
                compressed_cache=compressed_cache,
            )
        except Exception as exc:
            entry["row_record_error_type"] = type(exc).__name__
            entry["row_record_error"] = str(exc)
    trace.append(entry)


def _attention_backend_tensor_metadata(**tensors: torch.Tensor | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor):
            metadata[name] = _tensor_metadata(tensor)
        else:
            metadata[name] = {"available": False}
    return metadata


def _attention_backend_row_records(
    metadata: Any,
    rows: int,
    *,
    query: torch.Tensor | None,
    merged_output: torch.Tensor | None,
    swa_indices: torch.Tensor | None,
    swa_lengths: torch.Tensor | None,
    swa_cache: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    compressed_lengths: torch.Tensor | None,
    compressed_cache: torch.Tensor | None,
) -> list[dict[str, Any]]:
    selector = query
    if not isinstance(selector, torch.Tensor):
        selector = getattr(metadata, "positions", None)
    selected_rows = _selected_trace_rows(selector) if isinstance(selector, torch.Tensor) else [0]
    selected_rows = [idx for idx in selected_rows if 0 <= int(idx) < int(rows)]
    records = []
    for row_idx in selected_rows:
        record: dict[str, Any] = {
            "row": int(row_idx),
            "metadata": _attention_metadata_row(metadata, int(row_idx)),
        }
        if isinstance(query, torch.Tensor) and int(row_idx) < int(query.shape[0]):
            record["query"] = _row_record(query, int(row_idx))
        if (
            isinstance(merged_output, torch.Tensor)
            and int(row_idx) < int(merged_output.shape[0])
        ):
            record["merged_output"] = _row_record(merged_output, int(row_idx))
        record["swa"] = _attention_consumed_cache_record(
            indices=swa_indices,
            lengths=swa_lengths,
            cache=swa_cache,
            row=int(row_idx),
        )
        if isinstance(compressed_indices, torch.Tensor):
            record["compressed"] = _attention_consumed_cache_record(
                indices=compressed_indices,
                lengths=compressed_lengths,
                cache=compressed_cache,
                row=int(row_idx),
            )
        records.append(record)
    return records


def _attention_metadata_row(metadata: Any, row: int) -> dict[str, Any]:
    def scalar(name: str) -> int | None:
        tensor = getattr(metadata, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None
        flat = tensor.detach().reshape(-1)
        if row < 0 or row >= int(flat.numel()):
            return None
        try:
            return int(flat[row].cpu().item())
        except Exception:
            return None

    def vector(name: str, limit: int = 64) -> list[int]:
        tensor = getattr(metadata, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return []
        view = tensor.detach()
        if view.ndim == 1:
            if row >= int(view.numel()):
                return []
            view = view[row : row + 1]
        elif row >= int(view.shape[0]):
            return []
        else:
            view = view[row].reshape(-1)
        try:
            return [int(x) for x in view[:limit].cpu().tolist()]
        except Exception:
            return []

    return {
        "raw_out_loc": scalar("raw_out_loc"),
        "position": scalar("positions"),
        "seq_len": scalar("seq_lens"),
        "req_seq_len": scalar("req_seq_lens"),
        "req_table_index": scalar("req_table_indices"),
        "c4_out_loc": scalar("c4_out_loc"),
        "c128_out_loc": scalar("c128_out_loc"),
        "c4_indexer_out_loc": scalar("c4_indexer_out_loc"),
        "swa_topk_length": scalar("swa_topk_lengths"),
        "c4_topk_length_raw": scalar("c4_topk_lengths_raw"),
        "c4_sparse_topk_length": scalar("c4_sparse_topk_lengths"),
        "c128_topk_length_clamp1": scalar("c128_topk_lengths_clamp1"),
        "page_table_row": vector("page_table"),
        "swa_indices_row": vector("swa_page_indices"),
        "c4_sparse_full_indices_row": vector("c4_sparse_full_indices"),
        "c4_sparse_page_indices_row": vector("c4_sparse_page_indices"),
        "c128_full_indices_row": vector("c128_full_indices"),
        "c128_page_indices_row": vector("c128_page_indices"),
        "c4_page_table_row": vector("c4_page_table"),
        "c4_indexer_page_table_row": vector("c4_indexer_page_table"),
        "c128_page_table_row": vector("c128_page_table"),
    }


def _attention_consumed_cache_record(
    *,
    indices: torch.Tensor | None,
    lengths: torch.Tensor | None,
    cache: torch.Tensor | None,
    row: int,
) -> dict[str, Any]:
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        return {"available": False, "reason": "missing indices"}
    if row < 0 or row >= int(indices.shape[0]):
        return {"available": False, "reason": "row out of range"}
    index_row = indices.detach()[row].reshape(-1).to(dtype=torch.long)
    if isinstance(lengths, torch.Tensor) and lengths.numel() > row:
        try:
            active_len = int(lengths.detach().reshape(-1)[row].cpu().item())
        except Exception:
            active_len = int(index_row.numel())
    else:
        active_len = int((index_row >= 0).sum().cpu().item())
    active_len = max(0, min(active_len, int(index_row.numel())))
    active = index_row[:active_len]
    active = active[active >= 0]
    record: dict[str, Any] = {
        "available": True,
        "active_length": int(active_len),
        "active_indices": [int(x) for x in active[:64].cpu().tolist()],
        "active_index_count": int(active.numel()),
        "indices_row": [int(x) for x in index_row[:64].cpu().tolist()],
    }
    if not isinstance(cache, torch.Tensor):
        record["cache_available"] = False
        return record
    valid = active[(active >= 0) & (active < int(cache.shape[0]))]
    record["valid_index_count"] = int(valid.numel())
    if valid.numel() == 0:
        record["cache_values"] = {"available": False, "reason": "no valid indices"}
        return record
    values = cache.index_select(0, valid.to(device=cache.device, dtype=torch.long))
    values_cpu = values.detach().contiguous().cpu()
    record["cache_values"] = {
        "shape": [int(x) for x in values_cpu.shape],
        "dtype": str(values_cpu.dtype),
        "raw_sha256": _raw_checksum(values_cpu),
        "summary": _tensor_summary(values_cpu.float()),
        "row_checksums": [
            {
                "index": int(index),
                "raw_sha256": _raw_checksum(values_cpu[offset]),
                "summary": _tensor_summary(values_cpu[offset].float()),
            }
            for offset, index in enumerate(valid[:8].cpu().tolist())
        ],
    }
    return record


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


def capture_cache_rows(cache: torch.Tensor | None, locs: torch.Tensor | None) -> torch.Tensor | None:
    if not isinstance(cache, torch.Tensor) or not isinstance(locs, torch.Tensor):
        return None
    if cache.numel() == 0 or locs.numel() == 0:
        return None
    if cache.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return None
        except Exception:
            return None
    try:
        locs_long = locs.detach().reshape(-1).to(device=cache.device, dtype=torch.long)
        if not bool(torch.all((locs_long >= 0) & (locs_long < int(cache.shape[0]))).item()):
            return None
        return cache.index_select(0, locs_long).detach().contiguous().cpu()
    except Exception:
        return None


def record_layer2_swa_store(
    batch: Any,
    *,
    layer_id: int,
    path: str,
    kv: torch.Tensor | None,
    full_out_loc: torch.Tensor | None,
    swa_out_loc: torch.Tensor | None,
    positions: torch.Tensor | None,
    cache_before: torch.Tensor | None,
    cache_after: torch.Tensor | None,
    extra: dict[str, Any] | None = None,
) -> None:
    if not swa_lifecycle_trace_enabled(int(layer_id)):
        return
    if batch is None or not isinstance(kv, torch.Tensor) or kv.numel() == 0:
        return
    if kv.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return
    trace = getattr(batch, "_dsv4_layer2_swa_store_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_layer2_swa_store_trace", trace)

    def _scalar(tensor: torch.Tensor | None, row: int) -> int | None:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return None
        flat = tensor.detach().reshape(-1)
        if row < 0 or row >= int(flat.numel()):
            return None
        try:
            return int(flat[row].cpu().item())
        except Exception:
            return None

    try:
        rows = _selected_trace_rows(kv)
        row_records = []
        for row in rows:
            record: dict[str, Any] = {
                "row": int(row),
                "input_id": _scalar(getattr(batch, "input_ids", None), int(row)),
                "position": _scalar(positions, int(row)),
                "full_out_loc": _scalar(full_out_loc, int(row)),
                "swa_out_loc": _scalar(swa_out_loc, int(row)),
                "store_input": _row_record(kv, int(row)),
            }
            if isinstance(cache_before, torch.Tensor) and int(row) < int(cache_before.shape[0]):
                record["cache_before"] = _row_record(cache_before, int(row))
            else:
                record["cache_before"] = {"available": False}
            if isinstance(cache_after, torch.Tensor) and int(row) < int(cache_after.shape[0]):
                record["cache_after"] = _row_record(cache_after, int(row))
            else:
                record["cache_after"] = {"available": False}
            row_records.append(record)
        trace.append(
            {
                "layer_id": int(layer_id),
                "path": str(path),
                "mode": str(getattr(batch, "_dsv4_mtp_row0_layer_trace_mode", "")),
                "phase": str(getattr(batch, "phase", "")),
                "is_target_verify": bool(
                    getattr(batch, "dsv4_target_verify_metadata", None) is not None
                ),
                "kv_tensor_metadata": _tensor_metadata(kv),
                "cache_before_metadata": (
                    _tensor_metadata(cache_before)
                    if isinstance(cache_before, torch.Tensor)
                    else {"available": False}
                ),
                "cache_after_metadata": (
                    _tensor_metadata(cache_after)
                    if isinstance(cache_after, torch.Tensor)
                    else {"available": False}
                ),
                "batch_context": _operator_batch_context(batch, positions),
                "extra": _json_dict(extra or {}),
                "rows": row_records,
            }
        )
    except Exception as exc:
        trace.append(
            {
                "layer_id": int(layer_id),
                "path": str(path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def export_layer2_swa_store_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_layer2_swa_store_trace", None)
    if not isinstance(trace, list):
        return []
    return [_strip_private(entry) for entry in trace]


def get_wo_a_projection_oracle_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_mtp_wo_a_projection_oracle_trace", None)
    if not isinstance(trace, list):
        return []
    return trace


def export_wo_a_projection_oracle_trace(batch_or_trace: Any) -> list[dict[str, Any]]:
    trace = (
        batch_or_trace
        if isinstance(batch_or_trace, list)
        else get_wo_a_projection_oracle_trace(batch_or_trace)
    )
    return [_strip_private(entry) for entry in trace]


def get_wo_b_projection_oracle_trace(batch: Any) -> list[dict[str, Any]]:
    trace = getattr(batch, "_dsv4_mtp_wo_b_projection_oracle_trace", None)
    if not isinstance(trace, list):
        return []
    return trace


def export_wo_b_projection_oracle_trace(batch_or_trace: Any) -> list[dict[str, Any]]:
    trace = (
        batch_or_trace
        if isinstance(batch_or_trace, list)
        else get_wo_b_projection_oracle_trace(batch_or_trace)
    )
    return [_strip_private(entry) for entry in trace]


def record_wo_a_projection_oracle(
    batch: Any,
    *,
    layer_id: int,
    input_tensor: torch.Tensor,
    cached_weight: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    selected_output: str,
    backend_path: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if not wo_a_projection_oracle_enabled(layer_id):
        return
    if batch is None or not isinstance(input_tensor, torch.Tensor):
        return
    if not isinstance(cached_weight, torch.Tensor):
        return
    if input_tensor.numel() == 0 or input_tensor.ndim == 0:
        return
    if input_tensor.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return
    trace = getattr(batch, "_dsv4_mtp_wo_a_projection_oracle_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_mtp_wo_a_projection_oracle_trace", trace)
    try:
        rows = _selected_trace_rows(input_tensor)
        output_metadata = {
            name: _tensor_metadata(tensor)
            for name, tensor in outputs.items()
            if isinstance(tensor, torch.Tensor)
        }
        row_records = []
        for row_idx in rows:
            row_record: dict[str, Any] = {
                "row": int(row_idx),
                "input": _row_record(input_tensor, row_idx),
                "outputs": {},
                "pairwise": {},
            }
            for name, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    row_record["outputs"][name] = _row_record(tensor, row_idx)
            output_items = [
                (name, tensor)
                for name, tensor in outputs.items()
                if isinstance(tensor, torch.Tensor)
            ]
            for i, (lhs_name, lhs_tensor) in enumerate(output_items):
                for rhs_name, rhs_tensor in output_items[i + 1 :]:
                    row_record["pairwise"][f"{lhs_name}_vs_{rhs_name}"] = (
                        _tensor_allclose_stats(
                            lhs_tensor[int(row_idx)].detach().contiguous().cpu(),
                            rhs_tensor[int(row_idx)].detach().contiguous().cpu(),
                            atol=0.0,
                            rtol=0.0,
                        )
                    )
            row_records.append(row_record)
        trace.append(
            {
                "layer_id": int(layer_id),
                "mode": str(getattr(batch, "_dsv4_mtp_row0_layer_trace_mode", "")),
                "phase": str(getattr(batch, "phase", "")),
                "is_target_verify": bool(
                    getattr(batch, "dsv4_target_verify_metadata", None) is not None
                ),
                "backend_path": str(backend_path),
                "selected_output": str(selected_output),
                "input_tensor_metadata": _tensor_metadata(input_tensor),
                "cached_weight_metadata": _tensor_metadata(cached_weight),
                "cached_weight_raw_sha256": _raw_checksum(cached_weight),
                "output_tensor_metadata": output_metadata,
                "batch_context": _operator_batch_context(
                    batch,
                    getattr(batch, "positions", None),
                ),
                "extra": _json_dict(extra or {}),
                "rows": row_records,
            }
        )
    except Exception as exc:
        trace.append(
            {
                "layer_id": int(layer_id),
                "backend_path": str(backend_path),
                "selected_output": str(selected_output),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


def record_wo_b_projection_oracle(
    batch: Any,
    *,
    layer_id: int,
    input_tensor: torch.Tensor,
    cached_weight: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    selected_output: str,
    backend_path: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if not wo_b_projection_oracle_enabled(layer_id):
        return
    if batch is None or not isinstance(input_tensor, torch.Tensor):
        return
    if not isinstance(cached_weight, torch.Tensor):
        return
    if input_tensor.numel() == 0 or input_tensor.ndim == 0:
        return
    if input_tensor.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return
    trace = getattr(batch, "_dsv4_mtp_wo_b_projection_oracle_trace", None)
    if trace is None:
        trace = []
        setattr(batch, "_dsv4_mtp_wo_b_projection_oracle_trace", trace)
    try:
        rows = _selected_trace_rows(input_tensor)
        output_metadata = {
            name: _tensor_metadata(tensor)
            for name, tensor in outputs.items()
            if isinstance(tensor, torch.Tensor)
        }
        row_records = []
        for row_idx in rows:
            row_record: dict[str, Any] = {
                "row": int(row_idx),
                "input": _row_record(input_tensor, row_idx),
                "outputs": {},
                "pairwise": {},
            }
            for name, tensor in outputs.items():
                if isinstance(tensor, torch.Tensor):
                    row_record["outputs"][name] = _row_record(tensor, row_idx)
            output_items = [
                (name, tensor)
                for name, tensor in outputs.items()
                if isinstance(tensor, torch.Tensor)
            ]
            for i, (lhs_name, lhs_tensor) in enumerate(output_items):
                for rhs_name, rhs_tensor in output_items[i + 1 :]:
                    row_record["pairwise"][f"{lhs_name}_vs_{rhs_name}"] = (
                        _tensor_allclose_stats(
                            lhs_tensor[int(row_idx)].detach().contiguous().cpu(),
                            rhs_tensor[int(row_idx)].detach().contiguous().cpu(),
                            atol=0.0,
                            rtol=0.0,
                        )
                    )
            row_records.append(row_record)
        trace.append(
            {
                "layer_id": int(layer_id),
                "mode": str(getattr(batch, "_dsv4_mtp_row0_layer_trace_mode", "")),
                "phase": str(getattr(batch, "phase", "")),
                "is_target_verify": bool(
                    getattr(batch, "dsv4_target_verify_metadata", None) is not None
                ),
                "backend_path": str(backend_path),
                "selected_output": str(selected_output),
                "input_tensor_metadata": _tensor_metadata(input_tensor),
                "cached_weight_metadata": _tensor_metadata(cached_weight),
                "cached_weight_raw_sha256": _raw_checksum(cached_weight),
                "output_tensor_metadata": output_metadata,
                "batch_context": _operator_batch_context(
                    batch,
                    getattr(batch, "positions", None),
                ),
                "extra": _json_dict(extra or {}),
                "rows": row_records,
            }
        )
    except Exception as exc:
        trace.append(
            {
                "layer_id": int(layer_id),
                "backend_path": str(backend_path),
                "selected_output": str(selected_output),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


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


def compare_layer_trace_rows(
    lhs: list[dict[str, Any]],
    rhs: list[dict[str, Any]],
    *,
    lhs_label: str,
    rhs_label: str,
    lhs_row: int = 0,
    rhs_row: int = 0,
    exact: bool = True,
    max_records: int | None = None,
) -> dict[str, Any]:
    rhs_by_name: dict[str, dict[str, Any]] = {}
    for entry in rhs:
        name = entry.get("name")
        if isinstance(name, str) and name not in rhs_by_name:
            rhs_by_name[name] = entry

    comparisons: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    missing_from_rhs: list[str] = []
    for lhs_entry in lhs:
        name = lhs_entry.get("name")
        if not isinstance(name, str):
            continue
        rhs_entry = rhs_by_name.get(name)
        if rhs_entry is None:
            missing_from_rhs.append(name)
            continue
        comparison = _compare_trace_entry_rows(
            lhs_entry,
            rhs_entry,
            lhs_label=lhs_label,
            rhs_label=rhs_label,
            lhs_row=int(lhs_row),
            rhs_row=int(rhs_row),
            exact=bool(exact),
        )
        comparisons.append(comparison)
        if first_mismatch is None and bool(comparison.get("is_mismatch", False)):
            first_mismatch = comparison
        if max_records is not None and len(comparisons) >= int(max_records):
            break

    lhs_names = {
        entry.get("name") for entry in lhs if isinstance(entry.get("name"), str)
    }
    missing_from_lhs = [
        str(entry.get("name"))
        for entry in rhs
        if isinstance(entry.get("name"), str) and entry.get("name") not in lhs_names
    ]
    return {
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "lhs_row": int(lhs_row),
        "rhs_row": int(rhs_row),
        "exact": bool(exact),
        "num_lhs_entries": int(len(lhs)),
        "num_rhs_entries": int(len(rhs)),
        "num_compared": int(len(comparisons)),
        "missing_from_rhs": missing_from_rhs[:32],
        "missing_from_lhs": missing_from_lhs[:32],
        "first_mismatch": first_mismatch,
        "comparisons": comparisons,
    }


def compare_operator_traces(
    normal_trace: list[dict[str, Any]],
    target_trace: list[dict[str, Any]],
    *,
    case_prefix: str,
    verify_event_id: int,
    rank: int,
) -> dict[str, Any]:
    if not operator_parity_enabled():
        return {"enabled": False, "records": []}
    atol = operator_parity_atol()
    rtol = operator_parity_rtol()
    target_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in target_trace:
        key = (str(entry.get("operator_name", "")), int(entry.get("layer_id", -1)))
        if key[0] and key not in target_by_key:
            target_by_key[key] = entry

    records: list[dict[str, Any]] = []
    for normal_entry in normal_trace:
        operator_name = str(normal_entry.get("operator_name", ""))
        layer_id = int(normal_entry.get("layer_id", -1))
        if not operator_parity_enabled(operator_name):
            continue
        target_entry = target_by_key.get((operator_name, layer_id))
        record = _compare_operator_pair(
            normal_entry,
            target_entry,
            case_prefix=case_prefix,
            verify_event_id=verify_event_id,
            rank=rank,
            atol=atol,
            rtol=rtol,
        )
        records.append(record)
    first_owner = None
    for record in records:
        if record.get("owner_verdict") not in {"operator parity pass"}:
            first_owner = record
            break
    return {
        "enabled": True,
        "rtol": float(rtol),
        "atol": float(atol),
        "num_records": int(len(records)),
        "first_owner": first_owner,
        "records": records,
    }


def compare_operator_trace_rows(
    normal_trace: list[dict[str, Any]],
    target_trace: list[dict[str, Any]],
    *,
    case_prefix: str,
    verify_event_id: int,
    rank: int,
    normal_row: int = 0,
    target_row: int = 0,
) -> dict[str, Any]:
    if not operator_parity_enabled():
        return {"enabled": False, "records": []}
    atol = operator_parity_atol()
    rtol = operator_parity_rtol()
    target_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in target_trace:
        key = (str(entry.get("operator_name", "")), int(entry.get("layer_id", -1)))
        if key[0] and key not in target_by_key:
            target_by_key[key] = entry

    records: list[dict[str, Any]] = []
    for normal_entry in normal_trace:
        operator_name = str(normal_entry.get("operator_name", ""))
        layer_id = int(normal_entry.get("layer_id", -1))
        if not operator_parity_enabled(operator_name):
            continue
        target_entry = target_by_key.get((operator_name, layer_id))
        record = _compare_operator_pair_rows(
            normal_entry,
            target_entry,
            case_prefix=case_prefix,
            verify_event_id=verify_event_id,
            rank=rank,
            normal_row=int(normal_row),
            target_row=int(target_row),
            atol=atol,
            rtol=rtol,
        )
        records.append(record)
    first_owner = None
    for record in records:
        if record.get("owner_verdict") not in {"operator parity pass"}:
            first_owner = record
            break
    return {
        "enabled": True,
        "rtol": float(rtol),
        "atol": float(atol),
        "normal_row": int(normal_row),
        "target_row": int(target_row),
        "num_records": int(len(records)),
        "first_owner": first_owner,
        "records": records,
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


def _compare_trace_entry_rows(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    lhs_label: str,
    rhs_label: str,
    lhs_row: int,
    rhs_row: int,
    exact: bool,
) -> dict[str, Any]:
    lhs_tensor = _trace_row_tensor(lhs, lhs_row)
    rhs_tensor = _trace_row_tensor(rhs, rhs_row)
    base: dict[str, Any] = {
        "name": lhs.get("name"),
        "layer_id": lhs.get("layer_id"),
        "boundary": lhs.get("boundary"),
        "lhs_row": int(lhs_row),
        "rhs_row": int(rhs_row),
        "lhs_dtype": str(lhs_tensor.dtype) if isinstance(lhs_tensor, torch.Tensor) else None,
        "rhs_dtype": str(rhs_tensor.dtype) if isinstance(rhs_tensor, torch.Tensor) else None,
        "lhs_shape": [int(x) for x in lhs_tensor.shape]
        if isinstance(lhs_tensor, torch.Tensor)
        else None,
        "rhs_shape": [int(x) for x in rhs_tensor.shape]
        if isinstance(rhs_tensor, torch.Tensor)
        else None,
        "lhs_tensor_metadata": lhs.get("tensor_metadata"),
        "rhs_tensor_metadata": rhs.get("tensor_metadata"),
    }
    if not isinstance(lhs_tensor, torch.Tensor) or not isinstance(rhs_tensor, torch.Tensor):
        base.update(
            {
                "is_mismatch": True,
                "error": "missing selected row tensor for comparison",
            }
        )
        return base
    stats = _tensor_allclose_stats(lhs_tensor, rhs_tensor, atol=0.0, rtol=0.0)
    bit_exact = bool(stats.get("bit_exact", False))
    allclose_exact = bool(stats.get("allclose", False))
    base.update(
        {
            "is_mismatch": (not bit_exact) if exact else (not allclose_exact),
            "bit_exact_result": bit_exact,
            "allclose_result": allclose_exact,
            "max_delta": stats.get("max_delta"),
            "mean_delta": stats.get("mean_delta"),
            "first_differing_index": stats.get("first_differing_index"),
            "max_delta_index": stats.get("max_delta_index"),
            f"{lhs_label}_raw_sha256": _raw_checksum(lhs_tensor),
            f"{rhs_label}_raw_sha256": _raw_checksum(rhs_tensor),
            f"{lhs_label}_sample": stats.get("lhs_sample"),
            f"{rhs_label}_sample": stats.get("rhs_sample"),
        }
    )
    return base


def _compare_operator_pair_rows(
    normal: dict[str, Any],
    target: dict[str, Any] | None,
    *,
    case_prefix: str,
    verify_event_id: int,
    rank: int,
    normal_row: int,
    target_row: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    operator_name = str(normal.get("operator_name", ""))
    layer_id = int(normal.get("layer_id", -1))
    normal_context = _operator_record_row_context(normal, normal_row)
    target_context = (
        _operator_record_row_context(target, target_row)
        if isinstance(target, dict)
        else {}
    )
    request_id = target_context.get("request_id", normal_context.get("request_id"))
    row_depth = target_context.get("row_depth", normal_context.get("row_depth"))
    position = target_context.get("position", normal_context.get("position"))
    input_token = target_context.get("input_token", normal_context.get("input_token"))
    case_id = (
        f"{case_prefix}.rank{rank}.event{verify_event_id}."
        f"layer{layer_id}.{operator_name}.req{request_id}."
        f"depth{row_depth}.pos{position}.tok{input_token}."
        f"nrow{normal_row}.trow{target_row}"
    )
    base: dict[str, Any] = {
        "case_id": case_id,
        "rank": int(rank),
        "layer": int(layer_id),
        "request_id": request_id,
        "verify_event_id": int(verify_event_id),
        "row_depth": row_depth,
        "position": position,
        "input_token": input_token,
        "operator_name": operator_name,
        "normal_row": int(normal_row),
        "target_row": int(target_row),
        "normal_kernel_or_path": normal.get("path"),
        "target_verify_kernel_or_path": target.get("path") if isinstance(target, dict) else None,
        "normal_params": normal.get("params"),
        "target_verify_params": target.get("params") if isinstance(target, dict) else None,
        "normal_extra": normal.get("extra"),
        "target_verify_extra": target.get("extra") if isinstance(target, dict) else None,
        "normal_context": normal_context,
        "target_verify_context": target_context,
        "input_tensor_metadata": {
            "normal": normal.get("input_tensor_metadata"),
            "target_verify": target.get("input_tensor_metadata") if isinstance(target, dict) else None,
        },
        "output_tensor_metadata": {
            "normal": normal.get("output_tensor_metadata"),
            "target_verify": target.get("output_tensor_metadata") if isinstance(target, dict) else None,
        },
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if target is None:
        base.update(
            {
                "allclose_result": False,
                "input_allclose_result": False,
                "owner_verdict": "insufficient evidence",
                "reason": "missing target-verify operator capture",
            }
        )
        return base

    normal_input = _operator_row_tensor(normal, normal_row, "input")
    target_input = _operator_row_tensor(target, target_row, "input")
    normal_output = _operator_row_tensor(normal, normal_row, "output")
    target_output = _operator_row_tensor(target, target_row, "output")
    input_stats = _tensor_allclose_stats(normal_input, target_input, atol=atol, rtol=rtol)
    output_stats = _tensor_allclose_stats(normal_output, target_output, atol=atol, rtol=rtol)
    base.update(
        {
            "input_allclose_result": bool(input_stats.get("allclose", False)),
            "allclose_result": bool(output_stats.get("allclose", False)),
            "input_bit_exact_result": bool(input_stats.get("bit_exact", False)),
            "bit_exact_result": bool(output_stats.get("bit_exact", False)),
            "input_max_delta": input_stats.get("max_delta"),
            "input_mean_delta": input_stats.get("mean_delta"),
            "max_delta": output_stats.get("max_delta"),
            "mean_delta": output_stats.get("mean_delta"),
            "first_differing_index": output_stats.get("first_differing_index"),
            "normal_raw_sha256": (
                _raw_checksum(normal_output)
                if isinstance(normal_output, torch.Tensor)
                else None
            ),
            "target_verify_raw_sha256": (
                _raw_checksum(target_output)
                if isinstance(target_output, torch.Tensor)
                else None
            ),
            "normal_sample": output_stats.get("lhs_sample"),
            "target_verify_sample": output_stats.get("rhs_sample"),
            "input_comparison": input_stats,
            "output_comparison": output_stats,
        }
    )
    if not input_stats.get("available"):
        verdict = "insufficient evidence"
    elif not bool(input_stats.get("bit_exact", False)):
        verdict = "input already drifted"
    elif bool(output_stats.get("bit_exact", False)):
        verdict = "operator parity pass"
    elif normal.get("path") != target.get("path"):
        verdict = "dispatch/path mismatch"
    elif bool(output_stats.get("allclose", False)):
        verdict = "near-exact precision drift"
    else:
        verdict = "same-kernel output drift"
    base["owner_verdict"] = verdict
    return base


def _compare_operator_pair(
    normal: dict[str, Any],
    target: dict[str, Any] | None,
    *,
    case_prefix: str,
    verify_event_id: int,
    rank: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    operator_name = str(normal.get("operator_name", ""))
    layer_id = int(normal.get("layer_id", -1))
    target_context = target.get("batch_context", {}) if isinstance(target, dict) else {}
    normal_context = normal.get("batch_context", {})
    request_id = target_context.get("request_id", normal_context.get("request_id"))
    row_depth = target_context.get("row_depth", normal_context.get("row_depth"))
    position = target_context.get("position", normal_context.get("position"))
    input_token = target_context.get("input_token", normal_context.get("input_token"))
    case_id = (
        f"{case_prefix}.rank{rank}.event{verify_event_id}."
        f"layer{layer_id}.{operator_name}.req{request_id}."
        f"depth{row_depth}.pos{position}.tok{input_token}"
    )
    base: dict[str, Any] = {
        "case_id": case_id,
        "rank": int(rank),
        "layer": int(layer_id),
        "request_id": request_id,
        "verify_event_id": int(verify_event_id),
        "row_depth": row_depth,
        "position": position,
        "input_token": input_token,
        "operator_name": operator_name,
        "normal_kernel_or_path": normal.get("path"),
        "target_verify_kernel_or_path": target.get("path") if isinstance(target, dict) else None,
        "normal_params": normal.get("params"),
        "target_verify_params": target.get("params") if isinstance(target, dict) else None,
        "normal_extra": normal.get("extra"),
        "target_verify_extra": target.get("extra") if isinstance(target, dict) else None,
        "normal_context": normal_context,
        "target_verify_context": target_context,
        "input_tensor_metadata": {
            "normal": normal.get("input_tensor_metadata"),
            "target_verify": target.get("input_tensor_metadata") if isinstance(target, dict) else None,
        },
        "output_tensor_metadata": {
            "normal": normal.get("output_tensor_metadata"),
            "target_verify": target.get("output_tensor_metadata") if isinstance(target, dict) else None,
        },
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if target is None:
        base.update(
            {
                "allclose_result": False,
                "input_allclose_result": False,
                "owner_verdict": "insufficient evidence",
                "reason": "missing target-verify operator capture",
            }
        )
        return base

    normal_input = normal.get("_input_row0_tensor")
    target_input = target.get("_input_row0_tensor")
    normal_output = normal.get("_output_row0_tensor")
    target_output = target.get("_output_row0_tensor")
    input_stats = _tensor_allclose_stats(normal_input, target_input, atol=atol, rtol=rtol)
    output_stats = _tensor_allclose_stats(normal_output, target_output, atol=atol, rtol=rtol)
    base.update(
        {
            "input_allclose_result": bool(input_stats.get("allclose", False)),
            "allclose_result": bool(output_stats.get("allclose", False)),
            "input_bit_exact_result": bool(input_stats.get("bit_exact", False)),
            "bit_exact_result": bool(output_stats.get("bit_exact", False)),
            "input_max_delta": input_stats.get("max_delta"),
            "input_mean_delta": input_stats.get("mean_delta"),
            "max_delta": output_stats.get("max_delta"),
            "mean_delta": output_stats.get("mean_delta"),
            "first_differing_index": output_stats.get("first_differing_index"),
            "normal_sample": output_stats.get("lhs_sample"),
            "target_verify_sample": output_stats.get("rhs_sample"),
            "input_comparison": input_stats,
            "output_comparison": output_stats,
        }
    )
    if operator_name == "q_norm_rope":
        base["micro_allclose_probe"] = _q_norm_rope_micro_probe(
            normal,
            target,
            atol=atol,
            rtol=rtol,
        )
    elif operator_name == "q_norm":
        base["micro_allclose_probe"] = _q_norm_micro_probe(
            normal,
            target,
            atol=atol,
            rtol=rtol,
        )

    if not input_stats.get("available"):
        verdict = "insufficient evidence"
    elif not bool(input_stats.get("bit_exact", False)):
        verdict = "input already drifted"
    elif bool(output_stats.get("bit_exact", False)):
        verdict = "operator parity pass"
    elif _probe_has_reference_oracle_mismatch(base.get("micro_allclose_probe")):
        verdict = "reference-oracle mismatch"
    elif normal.get("path") != target.get("path"):
        verdict = "dispatch/path mismatch"
    elif bool(output_stats.get("allclose", False)):
        verdict = "near-exact precision drift"
    else:
        verdict = "same-kernel output drift"
    base["owner_verdict"] = verdict
    return base


def tensor_compare_stats(
    lhs: Any,
    rhs: Any,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, Any]:
    return _tensor_allclose_stats(
        lhs,
        rhs,
        atol=operator_parity_atol() if atol is None else float(atol),
        rtol=operator_parity_rtol() if rtol is None else float(rtol),
    )


def _q_norm_rope_micro_probe(
    normal: dict[str, Any],
    target: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    params = dict(normal.get("params", {}))
    normal_input = normal.get("_input_row0_tensor")
    target_input = target.get("_input_row0_tensor")
    normal_output = normal.get("_output_row0_tensor")
    target_output = target.get("_output_row0_tensor")
    position = normal.get("batch_context", {}).get("position")
    if position is None:
        position = target.get("batch_context", {}).get("position")
    if not isinstance(normal_input, torch.Tensor) or not isinstance(target_input, torch.Tensor):
        return {"available": False, "reason": "missing captured input"}
    if not isinstance(normal_output, torch.Tensor) or not isinstance(target_output, torch.Tensor):
        return {"available": False, "reason": "missing captured output"}
    if position is None:
        return {"available": False, "reason": "missing position"}

    probe: dict[str, Any] = {
        "available": True,
        "source": "captured row0 tensors",
        "position": int(position),
    }
    try:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        positions = torch.tensor([int(position)], dtype=torch.long, device=device)
        mini_normal = _run_mini_q_norm_rope_replay(normal_input, positions, params, device)
        mini_target = _run_mini_q_norm_rope_replay(target_input, positions, params, device)
        sglang_normal = _run_sglang_style_q_norm_rope_reference(
            normal_input,
            positions.cpu(),
            params,
        )
        sglang_target = _run_sglang_style_q_norm_rope_reference(
            target_input,
            positions.cpu(),
            params,
        )
        probe.update(
            {
                "mini_runtime_replay": {
                    "normal_input_vs_normal_output": _tensor_allclose_stats(
                        mini_normal, normal_output, atol=atol, rtol=rtol
                    ),
                    "normal_input_vs_target_output": _tensor_allclose_stats(
                        mini_normal, target_output, atol=atol, rtol=rtol
                    ),
                    "target_input_vs_target_output": _tensor_allclose_stats(
                        mini_target, target_output, atol=atol, rtol=rtol
                    ),
                    "normal_input_vs_target_input_replay": _tensor_allclose_stats(
                        mini_normal, mini_target, atol=atol, rtol=rtol
                    ),
                },
                "sglang_style_reference": {
                    "normal_input_vs_normal_output": _tensor_allclose_stats(
                        sglang_normal, normal_output, atol=atol, rtol=rtol
                    ),
                    "normal_input_vs_target_output": _tensor_allclose_stats(
                        sglang_normal, target_output, atol=atol, rtol=rtol
                    ),
                    "target_input_vs_target_output": _tensor_allclose_stats(
                        sglang_target, target_output, atol=atol, rtol=rtol
                    ),
                    "normal_input_vs_target_input_reference": _tensor_allclose_stats(
                        sglang_normal, sglang_target, atol=atol, rtol=rtol
                    ),
                },
            }
        )
    except Exception as exc:
        probe.update(
            {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return probe


def _q_norm_micro_probe(
    normal: dict[str, Any],
    target: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    params = dict(normal.get("params", {}))
    normal_input = normal.get("_input_row0_tensor")
    target_input = target.get("_input_row0_tensor")
    normal_output = normal.get("_output_row0_tensor")
    target_output = target.get("_output_row0_tensor")
    weight = normal.get("_q_norm_weight", target.get("_q_norm_weight"))
    if not isinstance(normal_input, torch.Tensor) or not isinstance(target_input, torch.Tensor):
        return {"available": False, "reason": "missing captured input"}
    if not isinstance(normal_output, torch.Tensor) or not isinstance(target_output, torch.Tensor):
        return {"available": False, "reason": "missing captured output"}
    if not isinstance(weight, torch.Tensor):
        return {"available": False, "reason": "missing q_norm weight"}

    probe: dict[str, Any] = {
        "available": True,
        "source": "captured row0 tensors",
        "rms_norm_eps": float(params.get("rms_norm_eps", 1.0e-6)),
    }
    try:
        device = weight.device
        eps = float(params.get("rms_norm_eps", 1.0e-6))
        normal_replay = dsv4_kernel.rms_norm_fallback(
            normal_input.to(device=device, dtype=normal_input.dtype).unsqueeze(0),
            weight,
            eps=eps,
        ).detach()[0].cpu()
        target_replay = dsv4_kernel.rms_norm_fallback(
            target_input.to(device=device, dtype=target_input.dtype).unsqueeze(0),
            weight,
            eps=eps,
        ).detach()[0].cpu()
        if normal_replay.is_cuda:
            torch.cuda.synchronize(normal_replay.device)
        if target_replay.is_cuda:
            torch.cuda.synchronize(target_replay.device)
        normal_ref = _run_rms_norm_reference(normal_input, weight.detach().cpu(), eps=eps)
        target_ref = _run_rms_norm_reference(target_input, weight.detach().cpu(), eps=eps)
        probe.update(
            {
                "mini_runtime_replay": {
                    "normal_input_vs_normal_output": _tensor_allclose_stats(
                        normal_replay, normal_output, atol=atol, rtol=rtol
                    ),
                    "target_input_vs_target_output": _tensor_allclose_stats(
                        target_replay, target_output, atol=atol, rtol=rtol
                    ),
                    "normal_input_vs_target_input_replay": _tensor_allclose_stats(
                        normal_replay, target_replay, atol=atol, rtol=rtol
                    ),
                },
                "torch_reference": {
                    "normal_input_vs_normal_output": _tensor_allclose_stats(
                        normal_ref, normal_output, atol=atol, rtol=rtol
                    ),
                    "target_input_vs_target_output": _tensor_allclose_stats(
                        target_ref, target_output, atol=atol, rtol=rtol
                    ),
                    "normal_input_vs_target_input_reference": _tensor_allclose_stats(
                        normal_ref, target_ref, atol=atol, rtol=rtol
                    ),
                },
            }
        )
    except Exception as exc:
        probe.update(
            {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return probe


def _run_rms_norm_reference(
    input_row0: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    dtype = input_row0.dtype
    y = input_row0.detach().cpu().float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + float(eps))
    return (y * weight.detach().cpu().float()).to(dtype)


def _run_mini_q_norm_rope_replay(
    input_row0: torch.Tensor,
    positions: torch.Tensor,
    params: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    q = input_row0.to(device=device, dtype=input_row0.dtype).unsqueeze(0).contiguous()
    dsv4_kernel.q_norm_rope_fallback(
        q,
        positions.to(device=device),
        rms_norm_eps=float(params.get("rms_norm_eps", 1.0e-6)),
        rotary_dim=int(params.get("rotary_dim", q.shape[-1])),
        base=float(params.get("base", 10000.0)),
        original_seq_len=int(params.get("original_seq_len", 0)),
        factor=float(params.get("factor", 1.0)),
        beta_fast=int(params.get("beta_fast", 32)),
        beta_slow=int(params.get("beta_slow", 1)),
    )
    if q.is_cuda:
        torch.cuda.synchronize(q.device)
    return q.detach()[0].float().cpu()


def _run_sglang_style_q_norm_rope_reference(
    input_row0: torch.Tensor,
    positions: torch.Tensor,
    params: dict[str, Any],
) -> torch.Tensor:
    q = input_row0.detach().cpu()
    dtype = q.dtype
    rotary_dim = int(params.get("rotary_dim", q.shape[-1]))
    base = float(params.get("base", 10000.0))
    eps = float(params.get("rms_norm_eps", 1.0e-6))
    original_seq_len = int(params.get("original_seq_len", 0))
    factor = float(params.get("factor", 1.0))
    beta_fast = int(params.get("beta_fast", 32))
    beta_slow = int(params.get("beta_slow", 1))

    q_fp32 = q.float()
    scale = torch.rsqrt(q_fp32.square().mean(-1, keepdim=True) + eps)
    out = (q_fp32 * scale).to(dtype)
    if rotary_dim <= 0:
        return out.float()
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32)
            / float(rotary_dim)
        )
    )
    if original_seq_len > 0:

        def correction_dim(num_rotations: float) -> float:
            import math

            return (
                rotary_dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        import math

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rotary_dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(rotary_dim // 2, dtype=torch.float32) - low)
            / max(high - low, 1),
            0,
            1,
        )
        smooth = 1 - ramp
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

    pos = positions.to(dtype=torch.long).reshape(-1)
    if pos.numel() != 1:
        pos = pos[:1]
    freqs = torch.outer(pos.float(), inv_freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)[0]
    cos = freqs_real[0::2]
    sin = freqs_real[1::2]
    rope = out[..., -rotary_dim:].float().unflatten(-1, (-1, 2))
    a, b = rope[..., 0], rope[..., 1]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    out[..., -rotary_dim:] = rotated.to(dtype)
    return out.float()


def _probe_has_reference_oracle_mismatch(probe: Any) -> bool:
    if not isinstance(probe, dict) or not probe.get("available"):
        return False
    ref = probe.get("sglang_style_reference")
    if not isinstance(ref, dict):
        return False
    normal_stats = ref.get("normal_input_vs_normal_output", {})
    target_stats = ref.get("normal_input_vs_target_output", {})
    if not isinstance(normal_stats, dict) or not isinstance(target_stats, dict):
        return False
    normal_ok = bool(normal_stats.get("allclose", False))
    target_ok = bool(target_stats.get("allclose", False))
    return normal_ok != target_ok


def _tensor_allclose_stats(
    lhs: Any,
    rhs: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return {"available": False, "allclose": False, "reason": "missing tensor"}
    lhs_cpu = lhs.detach().contiguous().cpu()
    rhs_cpu = rhs.detach().contiguous().cpu()
    lhs_flat_original = lhs_cpu.reshape(-1)
    rhs_flat_original = rhs_cpu.reshape(-1)
    lhs_flat = lhs_flat_original.float()
    rhs_flat = rhs_flat_original.float()
    if lhs_flat.shape != rhs_flat.shape:
        return {
            "available": False,
            "allclose": False,
            "bit_exact": False,
            "reason": "shape mismatch",
            "lhs_shape": [int(x) for x in lhs_flat.shape],
            "rhs_shape": [int(x) for x in rhs_flat.shape],
        }
    dtype_equal = lhs_cpu.dtype == rhs_cpu.dtype
    bit_exact = bool(dtype_equal and torch.equal(lhs_flat_original, rhs_flat_original))
    if lhs_flat.numel() == 0:
        return {
            "available": True,
            "allclose": True,
            "bit_exact": True,
            "dtype_equal": bool(dtype_equal),
            "max_delta": 0.0,
            "mean_delta": 0.0,
            "first_differing_index": None,
            "lhs_sample": [],
            "rhs_sample": [],
        }
    delta = (lhs_flat - rhs_flat).abs()
    tol = float(atol) + float(rtol) * rhs_flat.abs()
    differing = torch.nonzero(delta > tol, as_tuple=False).flatten()
    first_idx = int(differing[0].item()) if differing.numel() > 0 else None
    max_delta = float(delta.max().item())
    mean_delta = float(delta.mean().item())
    k = min(8, int(lhs_flat.numel()))
    max_index = int(torch.argmax(delta).item())
    return {
        "available": True,
        "allclose": bool(torch.allclose(lhs_flat, rhs_flat, atol=float(atol), rtol=float(rtol))),
        "bit_exact": bit_exact,
        "dtype_equal": bool(dtype_equal),
        "lhs_dtype": str(lhs_cpu.dtype),
        "rhs_dtype": str(rhs_cpu.dtype),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_delta": max_delta,
        "mean_delta": mean_delta,
        "first_differing_index": first_idx,
        "max_delta_index": max_index,
        "lhs_value_at_max_delta": float(lhs_flat[max_index].item()),
        "rhs_value_at_max_delta": float(rhs_flat[max_index].item()),
        "lhs_sample": _jsonable(lhs_flat[:k].tolist()),
        "rhs_sample": _jsonable(rhs_flat[:k].tolist()),
    }


def _operator_batch_context(batch: Any, positions: torch.Tensor | None) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "phase": str(getattr(batch, "phase", "")),
        "batch_size": int(getattr(batch, "size", len(getattr(batch, "reqs", [])) or 0)),
        "is_target_verify": bool(getattr(batch, "dsv4_target_verify_metadata", None) is not None),
    }
    try:
        reqs = getattr(batch, "padded_reqs", getattr(batch, "reqs", []))
        if reqs:
            req = reqs[0]
            ctx["request_id"] = int(getattr(req, "uid", 0))
            ctx["request_table_idx"] = int(getattr(req, "table_idx", -1))
            ctx["request_cached_len"] = int(getattr(req, "cached_len", -1))
    except Exception:
        pass
    try:
        input_ids = getattr(batch, "input_ids", None)
        if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
            ctx["input_token"] = int(input_ids.reshape(-1)[0].detach().cpu().item())
    except Exception:
        pass
    try:
        if isinstance(positions, torch.Tensor) and positions.numel() > 0:
            ctx["position"] = int(positions.reshape(-1)[0].detach().cpu().item())
    except Exception:
        pass
    metadata = getattr(batch, "dsv4_target_verify_metadata", None)
    if isinstance(metadata, dict):
        row_depths = metadata.get("row_depths")
        row_to_batch_index = metadata.get("row_to_batch_index")
        for key, value in (
            ("row_depth", row_depths),
            ("row_to_batch_index", row_to_batch_index),
        ):
            try:
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    ctx[key] = int(value.reshape(-1)[0].detach().cpu().item())
                elif isinstance(value, (list, tuple)) and value:
                    ctx[key] = int(value[0])
            except Exception:
                pass
        for src, dst in (
            ("runtime", "target_verify_runtime"),
            ("attention_mode", "target_verify_attention_mode"),
            ("kv_store_mode", "target_verify_kv_store_mode"),
            ("speculative_num_draft_tokens", "speculative_num_draft_tokens"),
        ):
            if src in metadata:
                ctx[dst] = metadata[src]
    return _json_dict(ctx)


def _operator_row_context(
    batch: Any,
    positions: torch.Tensor | None,
    row: int,
) -> dict[str, Any]:
    ctx = _operator_batch_context(batch, positions)
    ctx["row"] = int(row)

    def _scalar(tensor: Any) -> int | None:
        try:
            if isinstance(tensor, torch.Tensor) and tensor.numel() > int(row) >= 0:
                return int(tensor.detach().reshape(-1)[int(row)].cpu().item())
        except Exception:
            return None
        return None

    input_token = _scalar(getattr(batch, "input_ids", None))
    position = _scalar(positions)
    out_loc = _scalar(getattr(batch, "out_loc", None))
    if input_token is not None:
        ctx["input_token"] = input_token
    if position is not None:
        ctx["position"] = position
    if out_loc is not None:
        ctx["out_cache_loc"] = out_loc

    metadata = getattr(batch, "dsv4_target_verify_metadata", None)
    if isinstance(metadata, dict):
        for src, dst in (
            ("row_depths", "row_depth"),
            ("row_to_batch_index", "row_to_batch_index"),
        ):
            value = metadata.get(src)
            try:
                if isinstance(value, torch.Tensor) and value.numel() > int(row) >= 0:
                    ctx[dst] = int(value.detach().reshape(-1)[int(row)].cpu().item())
                elif isinstance(value, (list, tuple)) and 0 <= int(row) < len(value):
                    ctx[dst] = int(value[int(row)])
            except Exception:
                pass
    return _json_dict(ctx)


def _operator_record_row_context(entry: dict[str, Any] | None, row: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    for record in entry.get("row_records", []):
        if int(record.get("row", -1)) == int(row):
            context = record.get("context", {})
            return dict(context) if isinstance(context, dict) else {}
    context = entry.get("batch_context", {})
    return dict(context) if isinstance(context, dict) else {}


def _tensor_metadata(tensor: torch.Tensor, *, row0: bool = False) -> dict[str, Any]:
    shape = [int(x) for x in tensor.shape]
    if row0 and shape:
        row_shape = [int(x) for x in tensor.detach()[0].shape]
    else:
        row_shape = shape
    return {
        "shape": shape,
        "row0_shape": row_shape,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": [int(x) for x in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
    }


def _json_dict(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() <= 8:
                out[str(key)] = _jsonable(value.detach().cpu().reshape(-1).tolist())
            else:
                out[str(key)] = {
                    "shape": [int(x) for x in value.shape],
                    "dtype": str(value.dtype),
                }
        elif isinstance(value, dict):
            out[str(key)] = _json_dict(value)
        elif isinstance(value, (list, tuple)):
            out[str(key)] = [
                _json_dict(item) if isinstance(item, dict) else _json_scalar(item)
                for item in value
            ]
        else:
            out[str(key)] = _json_scalar(value)
    return out


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if value is None:
        return None
    return str(value)


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


def _selected_trace_rows(tensor: torch.Tensor) -> list[int]:
    if tensor.ndim == 0 or tensor.shape[0] <= 0:
        return []
    row_count = int(tensor.shape[0])
    raw = os.environ.get(ROW_TRACE_ROWS_ENV, "").strip()
    if not raw:
        return [0]
    selected: set[int] = set()
    if raw.lower() in {"all", "*"}:
        selected.update(range(min(row_count, 16)))
    else:
        for part in raw.split(","):
            token = part.strip()
            if not token:
                continue
            try:
                if "-" in token:
                    start_s, end_s = token.split("-", 1)
                    start, end = int(start_s), int(end_s)
                    if end < start:
                        start, end = end, start
                    selected.update(range(start, end + 1))
                else:
                    selected.add(int(token))
            except ValueError:
                continue
    selected.add(0)
    return [idx for idx in sorted(selected) if 0 <= idx < row_count]


def _trace_row_tensor(entry: dict[str, Any], row: int) -> torch.Tensor | None:
    row_tensors = entry.get("_row_tensors")
    if isinstance(row_tensors, dict):
        tensor = row_tensors.get(int(row))
        if isinstance(tensor, torch.Tensor):
            return tensor
    if int(row) == 0:
        tensor = entry.get("_row0_tensor")
        if isinstance(tensor, torch.Tensor):
            return tensor
    return None


def _operator_row_tensor(
    entry: dict[str, Any],
    row: int,
    kind: str,
) -> torch.Tensor | None:
    key = "_input_row_tensors" if kind == "input" else "_output_row_tensors"
    row_tensors = entry.get(key)
    if isinstance(row_tensors, dict):
        tensor = row_tensors.get(int(row))
        if isinstance(tensor, torch.Tensor):
            return tensor
    if int(row) == 0:
        fallback = "_input_row0_tensor" if kind == "input" else "_output_row0_tensor"
        tensor = entry.get(fallback)
        if isinstance(tensor, torch.Tensor):
            return tensor
    return None


def _row_tensor_record(row_tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "available": True,
        "shape": [int(x) for x in row_tensor.shape],
        "dtype": str(row_tensor.dtype),
        "raw_sha256": _raw_checksum(row_tensor),
        "summary": _tensor_summary(row_tensor.float()),
    }


def _row_record(tensor: torch.Tensor, row: int) -> dict[str, Any]:
    row_tensor = tensor.detach()[int(row)].contiguous().cpu()
    return {
        "shape": [int(x) for x in row_tensor.shape],
        "dtype": str(row_tensor.dtype),
        "raw_sha256": _raw_checksum(row_tensor),
        "summary": _tensor_summary(row_tensor.float()),
    }


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
    summary["raw_sha256"] = _raw_checksum(row0)
    return summary


def _checksum(flat: torch.Tensor) -> str:
    contiguous = flat.float().contiguous()
    try:
        payload = contiguous.numpy().tobytes()
    except Exception:
        payload = bytes(str(contiguous.tolist()), encoding="utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _raw_checksum(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous().cpu()
    try:
        payload = contiguous.view(torch.uint8).numpy().tobytes()
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
    "OPERATOR_PARITY_ATOL_ENV",
    "OPERATOR_PARITY_ENV",
    "OPERATOR_PARITY_LAYERS_ENV",
    "OPERATOR_PARITY_OPERATORS_ENV",
    "OPERATOR_PARITY_RTOL_ENV",
    "ROW0_LAYER_PARITY_ENV",
    "ROW0_LAYER_PARITY_ATOL_ENV",
    "LAYER2_SWA_LIFECYCLE_TRACE_ENV",
    "SWA_LIFECYCLE_TRACE_LAYERS_ENV",
    "MOE_CONTRACT_ORACLE_ENV",
    "MOE_CONTRACT_ORACLE_LAYERS_ENV",
    "WO_A_PROJECTION_ORACLE_ENV",
    "WO_A_PROJECTION_ORACLE_LAYERS_ENV",
    "WO_B_PROJECTION_ORACLE_ENV",
    "WO_B_PROJECTION_ORACLE_LAYERS_ENV",
    "capture_cache_rows",
    "clone_operator_row0_input",
    "compare_operator_trace_rows",
    "compare_operator_traces",
    "compare_row0_layer_traces",
    "env_flag",
    "export_attention_backend_trace",
    "export_layer2_swa_store_trace",
    "export_moe_contract_oracle_trace",
    "export_moe_microbatch_runtime_trace",
    "export_operator_trace",
    "export_row0_layer_trace",
    "export_wo_a_projection_oracle_trace",
    "export_wo_b_projection_oracle_trace",
    "get_moe_contract_oracle_trace",
    "get_moe_microbatch_runtime_trace",
    "get_operator_trace",
    "get_row0_layer_trace",
    "get_wo_a_projection_oracle_trace",
    "get_wo_b_projection_oracle_trace",
    "operator_parity_atol",
    "operator_parity_enabled",
    "operator_parity_layer_enabled",
    "operator_parity_rtol",
    "record_attention_backend",
    "record_layer2_swa_store",
    "record_operator_capture",
    "record_row0_tensor",
    "record_wo_a_projection_oracle",
    "record_wo_b_projection_oracle",
    "reset_row0_layer_trace",
    "row0_layer_parity_atol",
    "row0_layer_parity_enabled",
    "row_tensor_trace_enabled",
    "layer2_swa_lifecycle_trace_enabled",
    "swa_lifecycle_trace_enabled",
    "swa_lifecycle_trace_layers",
    "moe_contract_oracle_enabled",
    "record_moe_contract_oracle",
    "record_moe_microbatch_runtime",
    "tensor_compare_stats",
    "wo_a_projection_oracle_enabled",
    "wo_b_projection_oracle_enabled",
]

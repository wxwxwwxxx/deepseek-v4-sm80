from __future__ import annotations

import hashlib
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from minisgl.attention import BaseAttnMetadata
from minisgl.attention.deepseek_v4 import DSV4AttentionMetadata
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.layers import BaseOP, OPList
from minisgl.utils import (
    div_ceil,
    div_even,
    dsv4_direct_copy_nvtx,
    dsv4_memory_debug,
    dsv4_mtp_debug,
    dsv4_owner_timing,
    dsv4_prefix_debug,
)

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


_MARLIN_WNA16_KEEP_HIDDEN_REF_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_KEEP_HIDDEN_REF"
_MARLIN_WNA16_FORCE_PREPACKED_RAW_PRESENT_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_FORCE_PREPACKED_WITH_RAW_PRESENT"
)
_MARLIN_WNA16_RELEASE_LAYER_FILTER_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_LAYER_FILTER"
_MARLIN_WNA16_RELEASE_WEIGHTS_ONLY_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_WEIGHTS_ONLY"
_MARLIN_WNA16_RELEASE_SCALES_ONLY_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_SCALES_ONLY"
_MARLIN_WNA16_RELEASE_AFTER_GRAPH_CAPTURE_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_AFTER_GRAPH_CAPTURE"
)
_MARLIN_WNA16_RELEASE_TIMING_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING"
_MARLIN_WNA16_POISON_HIDDEN_REF_PATTERN_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_POISON_HIDDEN_REF_PATTERN"
)
_MARLIN_WNA16_QUARANTINE_BLOCKS_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS"
_MARLIN_WNA16_QUARANTINE_BYTES_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_BYTES"
_MARLIN_WNA16_QUARANTINE_PATTERN_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_PATTERN"
_MARLIN_WNA16_POISON_THEN_FREE_ENV = dsv4_memory_debug.DSV4_MARLIN_WNA16_POISON_THEN_FREE_ENV
_MARLIN_WNA16_POISON_THEN_FREE_BYTES_ENV = (
    dsv4_memory_debug.DSV4_MARLIN_WNA16_POISON_THEN_FREE_BYTES_ENV
)
_MARLIN_WNA16_POISON_THEN_FREE_PATTERN_ENV = (
    dsv4_memory_debug.DSV4_MARLIN_WNA16_POISON_THEN_FREE_PATTERN_ENV
)
_MARLIN_WNA16_CACHE_INTEGRITY_LAYERS_ENV = "MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_LAYERS"
_MARLIN_WNA16_CACHE_INTEGRITY_MAX_FORWARD_LOGS_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_MAX_FORWARD_LOGS"
)
DSV4_EXPERIMENTAL_MTP_ENV = "MINISGL_DSV4_EXPERIMENTAL_MTP"
DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH_ENV = (
    "MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH"
)
DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH_TIMING_ENV = (
    "MINISGL_DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH_TIMING"
)


def dsv4_experimental_mtp_enabled() -> bool:
    return dsv4_kernel.dsv4_env_flag(DSV4_EXPERIMENTAL_MTP_ENV)


def _dsv4_capture_nvtx(name: str):
    if not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_GRAPH_CAPTURE_NVTX"):
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.nvtx.range(f"dsv4.{name}")


def _record_warmup_memory(
    owner: str,
    stage: str,
    *,
    layer_id: int | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    dsv4_memory_debug.record_warmup_memory(
        owner=owner,
        stage=stage,
        layer_id=layer_id,
        extra=extra,
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bytes(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    token = raw.strip().lower()
    multipliers = {
        "kib": 1 << 10,
        "kb": 1 << 10,
        "mib": 1 << 20,
        "mb": 1 << 20,
        "gib": 1 << 30,
        "gb": 1 << 30,
    }
    for suffix, multiplier in multipliers.items():
        if token.endswith(suffix):
            try:
                return int(float(token[: -len(suffix)]) * multiplier)
            except ValueError:
                return default
    try:
        return int(token)
    except ValueError:
        return default


def _marlin_wna16_release_timing() -> str:
    if dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_RELEASE_AFTER_GRAPH_CAPTURE_ENV):
        return "after_graph_capture"
    raw = os.environ.get(_MARLIN_WNA16_RELEASE_TIMING_ENV, "model_prepare").strip().lower()
    aliases = {
        "": "model_prepare",
        "immediate": "model_prepare",
        "after_prebuild": "model_prepare",
        "after_full_model_prebuild": "model_prepare",
        "model_prepare": "model_prepare",
        "before_kv": "before_kv_alloc",
        "before_kv_alloc": "before_kv_alloc",
        "before_kv_allocation": "before_kv_alloc",
        "after_kv": "after_kv_alloc",
        "after_kv_alloc": "after_kv_alloc",
        "after_kv_allocation": "after_kv_alloc",
        "before_warmup": "before_warmup_forward",
        "before_warmup_forward": "before_warmup_forward",
        "after_warmup": "after_warmup_forward",
        "after_warmup_forward": "after_warmup_forward",
        "after_graph": "after_graph_capture",
        "after_graph_capture": "after_graph_capture",
        "after_first_decode": "after_first_decode",
        "after_decode_step1": "after_first_decode",
    }
    return aliases.get(raw, raw)


def _marlin_wna16_release_deferred_from_model_prepare() -> bool:
    return _marlin_wna16_release_timing() != "model_prepare"


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


def _layer_selected_by_env(layer_id: int | None, env_name: str) -> bool:
    selected = _parse_int_filter(os.environ.get(env_name))
    return selected is None or (layer_id is not None and int(layer_id) in selected)


def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _debug_activations_enabled() -> bool:
    recorder = dsv4_prefix_debug.get_dsv4_prefix_debug_recorder()
    return recorder is not None and bool(getattr(recorder, "capture_activations", False))


def _capture_debug_activation(
    name: str,
    tensor: torch.Tensor,
    row_indices: torch.Tensor | None = None,
) -> None:
    try:
        batch = get_global_ctx().batch
    except Exception:
        batch = None
    if (
        dsv4_memory_debug.marlin_wna16_layer2_owner_probe_enabled()
        and (name.startswith("layer2.") or name in {"embedding", "final_norm", "lm_head_logits"})
        and not _cuda_graph_capture_active()
    ):
        stage = "unknown"
        if batch is not None:
            stage = (
                f"{getattr(batch, 'phase', 'unknown')}"
                f"_bs{int(getattr(batch, 'size', 0))}"
                f"_padded{int(getattr(batch, 'padded_size', getattr(batch, 'size', 0)))}"
            )
        dsv4_memory_debug.record_owner_tensor(
            owner_label=f"dsv4.layer2_owner_probe.{name}",
            stage=stage,
            tensor=tensor,
            include_integrity=True,
            extra={"activation_name": name},
        )
    dsv4_mtp_debug.record_row0_tensor(batch, name, tensor)
    dsv4_prefix_debug.capture_dsv4_activation(name, tensor, batch, row_indices=row_indices)


def _dsv4_debug_batch() -> Batch | None:
    try:
        return get_global_ctx().batch
    except Exception:
        return None


def _dsv4_debug_positions(batch: Batch | None, *, device: torch.device) -> torch.Tensor | None:
    positions = getattr(batch, "positions", None)
    if not isinstance(positions, torch.Tensor) or positions.numel() == 0:
        return None
    try:
        return positions.to(device=device, dtype=torch.long)
    except Exception:
        return positions


def _dsv4_is_target_verify_batch(batch: Batch | None) -> bool:
    return bool(getattr(batch, "dsv4_target_verify_metadata", None) is not None)


def _dsv4_target_verify_moe_microbatch_enabled() -> bool:
    return dsv4_kernel.dsv4_env_flag(DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH_ENV)


def _dsv4_target_verify_moe_microbatch_timing_enabled() -> bool:
    return dsv4_kernel.dsv4_env_flag(DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH_TIMING_ENV)


@dataclass(frozen=True)
class _DSV4TargetVerifyMoEMicrobatchContract:
    batch_size: int
    verify_width: int
    rows: int
    row_order: str
    chunk_rows: int
    chunk_count: int
    active_rows: int
    padded_rows: int

    def as_record(self) -> dict[str, object]:
        return {
            "batch_size": int(self.batch_size),
            "verify_width": int(self.verify_width),
            "rows": int(self.rows),
            "row_order": self.row_order,
            "chunk_rows": int(self.chunk_rows),
            "chunk_count": int(self.chunk_count),
            "active_rows": int(self.active_rows),
            "padded_rows": int(self.padded_rows),
            "chunking": "contiguous_flattened_chunks",
        }


def _dsv4_target_verify_moe_microbatch_contract(
    batch: Batch | None,
    rows: int,
) -> _DSV4TargetVerifyMoEMicrobatchContract | None:
    if not _dsv4_target_verify_moe_microbatch_enabled():
        return None
    if not _dsv4_is_target_verify_batch(batch):
        return None
    metadata = getattr(batch, "dsv4_target_verify_metadata", None)
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch requires "
            "dsv4_target_verify_metadata."
        )
    rows = int(rows)
    if rows <= 0:
        raise RuntimeError("DeepSeek V4 target-verify MoE microbatch got no rows.")

    extend_lens = metadata.get("extend_lens")
    if not isinstance(extend_lens, (list, tuple)) or not extend_lens:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch cannot derive B: "
            "metadata.extend_lens is missing."
        )
    batch_size = int(len(extend_lens))
    if batch_size <= 0:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch cannot derive a positive B."
        )
    live_batch_size = int(getattr(batch, "size", 0) or len(getattr(batch, "reqs", [])) or 0)
    if live_batch_size != batch_size:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch B mismatch: "
            f"metadata B={batch_size}, batch.size={live_batch_size}."
        )

    raw_width = metadata.get("speculative_num_draft_tokens")
    try:
        verify_width = int(raw_width)
    except (TypeError, ValueError):
        verify_width = 0
    if verify_width <= 0:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch cannot derive W: "
            "metadata.speculative_num_draft_tokens is missing."
        )
    if any(int(length) != verify_width for length in extend_lens):
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch requires fixed-width rows: "
            f"extend_lens={list(extend_lens)}, W={verify_width}."
        )
    if rows != batch_size * verify_width:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch shape mismatch: "
            f"rows={rows}, B={batch_size}, W={verify_width}."
        )

    row_depths = metadata.get("row_depths")
    if not isinstance(row_depths, (list, tuple)) or len(row_depths) != rows:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch cannot derive row order: "
            "metadata.row_depths is missing or has the wrong length."
        )
    depth_values = [int(value) for value in row_depths]
    request_major = [
        depth for _request in range(batch_size) for depth in range(verify_width)
    ]
    depth_major = [
        depth for depth in range(verify_width) for _request in range(batch_size)
    ]
    if depth_values == request_major:
        row_order = "request_major"
    elif depth_values == depth_major:
        row_order = "depth_major"
    else:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch cannot generalize row order: "
            f"row_depths_head={depth_values[: min(len(depth_values), 16)]}, "
            f"B={batch_size}, W={verify_width}."
        )

    active_mask = metadata.get("active_row_mask")
    padded_mask = metadata.get("padded_row_mask")
    if not isinstance(active_mask, (list, tuple)) or len(active_mask) != rows:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch requires active_row_mask "
            "with one entry per row."
        )
    if not isinstance(padded_mask, (list, tuple)) or len(padded_mask) != rows:
        raise RuntimeError(
            "DeepSeek V4 target-verify MoE microbatch requires padded_row_mask "
            "with one entry per row."
        )
    active_rows = sum(1 for value in active_mask if bool(value))
    padded_rows = sum(1 for value in padded_mask if bool(value))

    return _DSV4TargetVerifyMoEMicrobatchContract(
        batch_size=batch_size,
        verify_width=verify_width,
        rows=rows,
        row_order=row_order,
        chunk_rows=batch_size,
        chunk_count=verify_width,
        active_rows=active_rows,
        padded_rows=padded_rows,
    )


def _dsv4_moe_needs_router_logits() -> bool:
    return any(
        dsv4_mtp_debug.operator_parity_enabled(name)
        for name in ("router_logits", "topk_ids", "topk_weights")
    )


def _dsv4_moe_tensor_head(tensor: torch.Tensor | None, *, limit: int = 16) -> list[object]:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return []
    try:
        flat = tensor.detach().reshape(-1)[: int(limit)].cpu()
        values = flat.tolist()
        if not isinstance(values, list):
            values = [values]
        out: list[object] = []
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
    except Exception:
        return []


def _dsv4_moe_tensor_brief(tensor: torch.Tensor | None) -> dict[str, object]:
    if not isinstance(tensor, torch.Tensor):
        return {"available": False}
    info: dict[str, object] = {
        "available": True,
        "shape": [int(x) for x in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": [int(x) for x in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "head": _dsv4_moe_tensor_head(tensor, limit=16),
    }
    if tensor.ndim > 1 and tensor.shape[0] > 0:
        info["row0_head"] = _dsv4_moe_tensor_head(tensor[0], limit=16)
    return info


def _dsv4_moe_row0_checksum(tensor: torch.Tensor | None) -> str | None:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0 or tensor.ndim == 0:
        return None
    try:
        row = tensor.detach()[0].float().contiguous().cpu()
        return hashlib.sha256(row.numpy().tobytes()).hexdigest()[:16]
    except Exception:
        return None


def _dsv4_moe_reduce_tensor_census(tensor: torch.Tensor | None) -> dict[str, object]:
    info = _dsv4_moe_tensor_brief(tensor)
    checksum = _dsv4_moe_row0_checksum(tensor)
    if checksum is not None:
        info["row0_checksum"] = checksum
    return info


def _dsv4_moe_reduce_census_enabled() -> bool:
    return dsv4_mtp_debug.operator_parity_enabled(
        "expert_reduce_output"
    ) or dsv4_mtp_debug.operator_parity_enabled("moe_output")


def _dsv4_moe_reduce_tensor_census_if_enabled(
    tensor: torch.Tensor | None,
) -> dict[str, object]:
    if not _dsv4_moe_reduce_census_enabled():
        return {"available": False, "disabled": True}
    return _dsv4_moe_reduce_tensor_census(tensor)


def _dsv4_moe_comm_backend_name(comm: DistributedCommunicator) -> str:
    try:
        return type(comm.plugins[-1]).__name__
    except Exception:
        return "unknown"


def _dsv4_moe_reduce_boundary_base_extra(
    *,
    stage: str,
    tp_size: int,
    comm: DistributedCommunicator,
    reduce_label: str,
    pre_reduce: torch.Tensor,
) -> dict[str, object]:
    try:
        tp_rank = int(get_tp_info().rank)
    except Exception:
        tp_rank = -1
    return {
        "stage": stage,
        "tp_size": int(tp_size),
        "tp_rank": tp_rank,
        "pre_reduce": _dsv4_moe_reduce_tensor_census_if_enabled(pre_reduce),
        "local_rank_contribution": _dsv4_moe_reduce_tensor_census_if_enabled(pre_reduce),
        "communication_backend": _dsv4_moe_comm_backend_name(comm),
        "communication_label": str(reduce_label),
        "communication_op": "all_reduce" if tp_size > 1 else "none",
        "all_reduce": bool(tp_size > 1),
        "reduce_scatter": False,
        "skip_post_experts_all_reduce": False,
        "final_cast_site": "moe_output",
    }


def _dsv4_moe_expert_histogram(indices: torch.Tensor | None) -> list[dict[str, int]]:
    if not isinstance(indices, torch.Tensor) or indices.numel() == 0:
        return []
    try:
        flat = indices.detach().reshape(-1).cpu().long()
        valid = flat[flat >= 0]
        if valid.numel() == 0:
            return []
        experts, counts = torch.unique(valid, sorted=True, return_counts=True)
        return [
            {"expert_id": int(expert), "count": int(count)}
            for expert, count in zip(experts.tolist()[:32], counts.tolist()[:32])
        ]
    except Exception:
        return []


def _dsv4_moe_route_extra(
    *,
    weights: torch.Tensor | None,
    indices: torch.Tensor | None,
    input_ids: torch.Tensor | None = None,
    moe_plan: dsv4_kernel.DSV4MoEExecutionPlan | None = None,
    reduce_once: bool | None = None,
    hash_topk: bool | None = None,
    stage: str | None = None,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if stage is not None:
        extra["stage"] = stage
    if reduce_once is not None:
        extra["reduce_once"] = bool(reduce_once)
    if hash_topk is not None:
        extra["hash_topk"] = bool(hash_topk)
    if isinstance(input_ids, torch.Tensor):
        extra["input_ids"] = _dsv4_moe_tensor_brief(input_ids)
    if isinstance(indices, torch.Tensor):
        extra.update(
            {
                "topk_ids": _dsv4_moe_tensor_brief(indices),
                "expert_histogram": _dsv4_moe_expert_histogram(indices),
            }
        )
        if indices.ndim >= 2:
            extra["tokens"] = int(indices.shape[0])
            extra["topk"] = int(indices.shape[1])
    if isinstance(weights, torch.Tensor):
        extra["topk_weights"] = _dsv4_moe_tensor_brief(weights)
    if moe_plan is not None:
        route_plan = moe_plan.route_plan
        extra["moe_plan"] = {
            "tokens": int(moe_plan.tokens),
            "hidden": int(moe_plan.hidden),
            "num_experts": int(moe_plan.num_experts),
            "reduce_once": bool(moe_plan.reduce_once),
            "final_reduce_label": str(moe_plan.final_reduce_label),
            "route_count": int(route_plan.route_count),
            "topk": int(route_plan.topk),
            "block_size_m": int(route_plan.block_size_m),
            "sorted_route_ids": _dsv4_moe_tensor_brief(route_plan.sorted_route_ids),
            "expert_ids": _dsv4_moe_tensor_brief(route_plan.expert_ids),
            "num_tokens_post_padded": _dsv4_moe_tensor_brief(
                route_plan.num_tokens_post_padded
            ),
        }
    return extra


def _dsv4_moe_record_operator(
    batch: Batch | None,
    *,
    operator_name: str,
    layer_id: int,
    input_tensor: torch.Tensor | None,
    output_tensor: torch.Tensor | None,
    positions: torch.Tensor | None,
    path: str,
    params: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
    input_row0: torch.Tensor | None = None,
) -> None:
    if not dsv4_mtp_debug.operator_parity_enabled(operator_name):
        return
    if input_row0 is None:
        input_row0 = dsv4_mtp_debug.clone_operator_row0_input(operator_name, input_tensor)
    dsv4_mtp_debug.record_operator_capture(
        batch,
        operator_name=operator_name,
        layer_id=int(layer_id),
        input_row0=input_row0,
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        positions=positions,
        path=path,
        params=params,
        extra=extra,
    )


def _dsv4_moe_router_logits_debug(
    gate: "DSV4MoEGate",
    flat: torch.Tensor,
) -> torch.Tensor | None:
    if not _dsv4_moe_needs_router_logits():
        return None
    try:
        weight = _cached_gate_fp32_weight(gate, "_cached_gate_weight_fp32", gate.weight)
        return F.linear(flat.float(), weight.float())
    except Exception:
        return None


def _debug_can_materialize_tensor(tensor: torch.Tensor) -> bool:
    if not _debug_activations_enabled():
        return False
    if tensor.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return False
        except Exception:
            return False
    return True


def _compressed_debug_end_indices(
    positions: torch.Tensor,
    ratio: int,
) -> torch.Tensor | None:
    if positions.numel() == 0 or ratio <= 0:
        return None
    if positions.is_cuda:
        try:
            if torch.cuda.is_current_stream_capturing():
                return None
        except Exception:
            return None
    pos = positions.to(dtype=torch.long)
    end_indices = torch.nonzero((pos + 1) % ratio == 0, as_tuple=False).flatten()
    if end_indices.numel() == 0:
        return None
    offsets = torch.arange(ratio, dtype=torch.long, device=pos.device)
    gather = end_indices[:, None] - (ratio - 1) + offsets[None, :]
    valid = gather[:, 0] >= 0
    if bool(torch.any(valid)):
        gather_valid = gather[valid]
        expected = pos[end_indices[valid]][:, None] - (ratio - 1) + offsets[None, :]
        contiguous = torch.all(pos[gather_valid] == expected, dim=1)
        valid_rows = torch.nonzero(valid, as_tuple=False).flatten()[contiguous]
        return end_indices[valid_rows]
    return None


def _compressed_debug_row_indices(
    positions: torch.Tensor,
    ratio: int,
    batch: Batch,
) -> torch.Tensor | None:
    if not _debug_activations_enabled() or positions.numel() == 0 or ratio <= 0:
        return None
    end_indices = _compressed_debug_end_indices(positions, ratio)
    if end_indices is None:
        return None
    if end_indices.numel() == 0:
        return None

    output_rows = torch.arange(end_indices.numel(), dtype=torch.long, device=end_indices.device)
    selected = []
    start = 0
    for req in getattr(batch, "reqs", []):
        length = int(req.extend_len)
        end = start + length
        mask = (end_indices >= start) & (end_indices < end)
        if bool(torch.any(mask)):
            selected.append(output_rows[mask][-1])
        start = end
    if not selected:
        return None
    return torch.stack(selected)


def _capture_compressed_debug_window(
    name: str,
    tensor: torch.Tensor,
    positions: torch.Tensor,
    ratio: int,
    batch: Batch,
) -> None:
    if not _debug_can_materialize_tensor(tensor):
        return
    end_indices = _compressed_debug_end_indices(positions, ratio)
    if end_indices is None or end_indices.numel() == 0:
        return
    offsets = torch.arange(ratio, dtype=torch.long, device=end_indices.device)
    selected = []
    start = 0
    for req in getattr(batch, "reqs", []):
        length = int(req.extend_len)
        end = start + length
        mask = (end_indices >= start) & (end_indices < end)
        if bool(torch.any(mask)):
            window_end = end_indices[mask][-1]
            selected.append(window_end - (ratio - 1) + offsets)
        start = end
    if not selected:
        return
    gather = torch.stack(selected)
    gather_flat = gather.reshape(-1).to(device=tensor.device, dtype=torch.long)
    if gather_flat.numel() == 0 or int(gather_flat.max().item()) >= tensor.shape[0]:
        return
    window = tensor.index_select(0, gather_flat).reshape(
        gather.shape[0],
        gather.shape[1],
        *tensor.shape[1:],
    )
    dsv4_prefix_debug.capture_dsv4_activation(name, window, batch, row_indices=None)


def _owner_timing_prefix(owner_label: str) -> str:
    if owner_label.endswith(".attn.q_wqb") or ".attn.q_wqb" in owner_label:
        return "dsv4.owner.attn.q_wqb"
    if owner_label.endswith(".attn.wo_b") or ".attn.wo_b" in owner_label:
        return "dsv4.owner.attn.wo_b"
    if (
        owner_label.endswith(".shared_experts.down_proj")
        or ".shared_experts.down_proj" in owner_label
    ):
        return "dsv4.owner.shared_down"
    return f"dsv4.owner.{owner_label}"


def _marlin_wna16_released_items(
    release_reports: list[dict[str, object]],
) -> list[dict[str, object]]:
    released_items: list[dict[str, object]] = []
    for report in release_reports:
        for item in report.get("released", []):
            if isinstance(item, dict) and int(item.get("bytes", 0) or 0) > 0:
                released_items.append(item)
    return released_items


def _cached_hc_bf16_weight(owner: object, cache_name: str, weight: torch.Tensor) -> torch.Tensor:
    if not (dsv4_kernel.linear_bf16_fp32_upstream_enabled() and weight.is_cuda):
        return weight
    meta_name = f"{cache_name}_meta"
    meta = (
        weight.data_ptr(),
        int(getattr(weight, "_version", 0)),
        weight.device.type,
        weight.device.index,
        tuple(weight.shape),
        tuple(weight.stride()),
    )
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        cached = weight.to(torch.bfloat16).contiguous()
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)
    return cached


def _cached_fp32_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    *,
    toggle: str,
) -> torch.Tensor:
    if not (
        dsv4_kernel.dsv4_env_flag(toggle) and weight.is_cuda and weight.dtype == torch.bfloat16
    ):
        return weight
    meta_name = f"{cache_name}_meta"
    meta = (
        weight.data_ptr(),
        int(getattr(weight, "_version", 0)),
        weight.device.type,
        weight.device.index,
        tuple(weight.shape),
        tuple(weight.stride()),
    )
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        cached = weight.float().contiguous()
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)
    return cached


def _cached_projection_scale(
    owner: object,
    cache_name: str,
    scale: torch.Tensor | None,
) -> torch.Tensor | None:
    if scale is None:
        return None
    if not (
        dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_STATIC_SCALE_CACHE_TOGGLE) and scale.is_cuda
    ):
        return scale
    if scale.dtype is torch.float32 and scale.is_contiguous():
        return scale

    meta_name = f"{cache_name}_meta"
    meta = (
        scale.data_ptr(),
        int(getattr(scale, "_version", 0)),
        scale.device.type,
        scale.device.index,
        scale.dtype,
        tuple(scale.shape),
        tuple(scale.stride()),
        int(scale.storage_offset()),
    )
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        cached = scale.float().contiguous()
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)
    return cached


def _tensor_cache_meta(tensor: torch.Tensor | None) -> tuple | None:
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        int(getattr(tensor, "_version", 0)),
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
    )


def _fp8_bf16_weight_cache_meta(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> tuple:
    return (_tensor_cache_meta(weight), _tensor_cache_meta(scale), out_dtype)


def _cached_fp8_bf16_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    allow_build: bool,
    owner_label: str,
) -> torch.Tensor:
    if weight.dtype != dsv4_kernel.fp8_dtype():
        raise RuntimeError(
            f"{owner_label} cached BF16 weight path requires FP8 weights, got {weight.dtype}."
        )
    if scale is not None and scale.device != weight.device:
        raise RuntimeError(
            f"{owner_label} cached BF16 weight path requires scale on the same device "
            f"as weight, got weight={weight.device} scale={scale.device}."
        )
    if out_dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} cached BF16 weight path requires out_dtype=torch.bfloat16, got {out_dtype}."
        )

    meta_name = f"{cache_name}_meta"
    meta = _fp8_bf16_weight_cache_meta(weight, scale, out_dtype)
    cached = getattr(owner, cache_name, None)
    if cached is not None and getattr(owner, meta_name, None) == meta:
        return cached

    if not allow_build:
        raise RuntimeError(
            f"{owner_label} cached BF16 weight is missing or stale. "
            "Call prepare_for_cuda_graph_capture() after weights are loaded and before "
            "decode CUDA graph capture/replay; rebuilding inside forward is disabled."
        )

    cached = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=out_dtype).contiguous()
    setattr(owner, cache_name, cached)
    setattr(owner, meta_name, meta)
    return cached


def _prepare_fp8_marlin_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    owner_label: str,
    release_original: bool,
) -> dict[str, object]:
    from minisgl.kernel import dense_fp8_marlin

    meta_name = f"{cache_name}_meta"
    meta = _fp8_bf16_weight_cache_meta(weight, scale, torch.bfloat16)
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        cached = dense_fp8_marlin.prepare_dense_fp8_marlin_weight(
            weight,
            scale,
            owner_label=owner_label,
        )
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)

    released: list[dict[str, object]] = []
    if release_original:
        if hasattr(owner, "weight"):
            released.append(
                {
                    "attribute": "weight",
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype),
                    "bytes": int(dense_fp8_marlin.tensor_bytes(weight)),
                }
            )
            delattr(owner, "weight")
        if scale is not None and hasattr(owner, "weight_scale_inv"):
            released.append(
                {
                    "attribute": "weight_scale_inv",
                    "shape": list(scale.shape),
                    "dtype": str(scale.dtype),
                    "bytes": int(dense_fp8_marlin.tensor_bytes(scale)),
                }
            )
            delattr(owner, "weight_scale_inv")

    report = dense_fp8_marlin.prepare_dense_fp8_marlin_report(cached, owner_label=owner_label)
    report["released_original"] = bool(released)
    report["released"] = released
    return report


def _forward_fp8_marlin_weight(
    owner: object,
    cache_name: str,
    x: torch.Tensor,
    *,
    owner_label: str,
) -> torch.Tensor:
    from minisgl.kernel import dense_fp8_marlin

    cached = getattr(owner, cache_name, None)
    if cached is None:
        raise RuntimeError(
            f"{owner_label} dense FP8 Marlin weight is missing. Call "
            "prepare_for_cuda_graph_capture() after weights are loaded and before "
            "decode CUDA graph capture/replay; rebuilding inside forward is disabled."
        )
    if not dsv4_owner_timing.enabled():
        return dense_fp8_marlin.apply_dense_fp8_marlin_linear(
            x,
            cached,
            owner_label=owner_label,
        )
    with dsv4_owner_timing.maybe_cuda_range(
        f"{_owner_timing_prefix(owner_label)}.dense_fp8_marlin_local_total",
        {
            "owner_label": owner_label,
            "input": dsv4_owner_timing.tensor_metadata(x),
        },
    ):
        return dense_fp8_marlin.apply_dense_fp8_marlin_linear(
            x,
            cached,
            owner_label=owner_label,
        )


def _cached_bf16_pretransposed_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    *,
    allow_build: bool,
    owner_label: str,
) -> torch.Tensor:
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise RuntimeError(
            f"{owner_label} pretransposed BF16 cache requires a 2D BF16 weight, "
            f"got shape={tuple(weight.shape)} dtype={weight.dtype}."
        )
    meta_name = f"{cache_name}_meta"
    meta = _tensor_cache_meta(weight)
    cached = getattr(owner, cache_name, None)
    if cached is not None and getattr(owner, meta_name, None) == meta:
        return cached
    if not allow_build:
        raise RuntimeError(
            f"{owner_label} pretransposed BF16 weight is missing or stale. "
            "Call prepare_for_cuda_graph_capture() after weights are loaded and before "
            "decode CUDA graph capture/replay; rebuilding inside forward is disabled."
        )
    cached = weight.t().contiguous()
    setattr(owner, cache_name, cached)
    setattr(owner, meta_name, meta)
    return cached


def _linear_cached_bf16_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    owner: object,
    cache_name: str,
    owner_label: str,
) -> torch.Tensor:
    if not dsv4_owner_timing.enabled():
        if not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE):
            return F.linear(x, weight)
        rows = x.numel() // x.shape[-1]
        if rows > 16:
            return F.linear(x, weight)
        weight_t = _cached_bf16_pretransposed_weight(
            owner,
            f"{cache_name}_pretransposed",
            weight,
            allow_build=False,
            owner_label=owner_label,
        )
        x_2d = x.reshape(rows, x.shape[-1])
        y = torch.mm(x_2d, weight_t)
        return y.reshape(*x.shape[:-1], weight_t.shape[-1])

    prefix = _owner_timing_prefix(owner_label)
    metadata = {
        "owner_label": owner_label,
        "input": dsv4_owner_timing.tensor_metadata(x),
        "weight": dsv4_owner_timing.tensor_metadata(weight),
    }
    if not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE):
        with dsv4_owner_timing.maybe_cuda_range(
            f"{prefix}.bf16_cache_linear",
            {**metadata, "pretransposed": False},
        ):
            return F.linear(x, weight)
    rows = x.numel() // x.shape[-1]
    if rows > 16:
        with dsv4_owner_timing.maybe_cuda_range(
            f"{prefix}.bf16_cache_linear",
            {**metadata, "pretransposed": False, "rows": int(rows)},
        ):
            return F.linear(x, weight)
    weight_t = _cached_bf16_pretransposed_weight(
        owner,
        f"{cache_name}_pretransposed",
        weight,
        allow_build=False,
        owner_label=owner_label,
    )
    with dsv4_owner_timing.maybe_cuda_range(
        f"{prefix}.bf16_cache_input_reshape",
        {**metadata, "rows": int(rows)},
    ):
        x_2d = x.reshape(rows, x.shape[-1])
    with dsv4_owner_timing.maybe_cuda_range(
        f"{prefix}.bf16_cache_linear",
        {
            **metadata,
            "pretransposed": True,
            "rows": int(rows),
            "reshaped": dsv4_owner_timing.tensor_metadata(x_2d),
            "weight_t": dsv4_owner_timing.tensor_metadata(weight_t),
        },
    ):
        y = torch.mm(x_2d, weight_t)
    with dsv4_owner_timing.maybe_cuda_range(
        f"{prefix}.bf16_cache_output_reshape",
        {**metadata, "output": dsv4_owner_timing.tensor_metadata(y)},
    ):
        return y.reshape(*x.shape[:-1], weight_t.shape[-1])


def _fp8_cached_bf16_weight_local_projection(
    x: torch.Tensor,
    cached_weight: torch.Tensor,
    *,
    owner: object,
    cache_name: str,
    owner_label: str,
    accum_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if accum_dtype is not None:
        x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
        return F.linear(
            x_quant.to(accum_dtype),
            cached_weight.to(accum_dtype),
        ).to(x.dtype)
    if not dsv4_owner_timing.enabled():
        x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
        return _linear_cached_bf16_weight(
            x_quant,
            cached_weight,
            owner=owner,
            cache_name=cache_name,
            owner_label=owner_label,
        )

    prefix = _owner_timing_prefix(owner_label)
    metadata = {
        "owner_label": owner_label,
        "input": dsv4_owner_timing.tensor_metadata(x),
        "weight": dsv4_owner_timing.tensor_metadata(cached_weight),
    }
    with dsv4_owner_timing.maybe_cuda_range(f"{prefix}.bf16_cache_local_total", metadata):
        with dsv4_owner_timing.maybe_cuda_range(
            f"{prefix}.bf16_cache_activation_quantize",
            metadata,
        ):
            x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
        return _linear_cached_bf16_weight(
            x_quant,
            cached_weight,
            owner=owner,
            cache_name=cache_name,
            owner_label=owner_label,
        )


def _fp8_cached_bf16_weight_local_projection_row_invariant(
    x: torch.Tensor,
    cached_weight: torch.Tensor,
    *,
    owner: object,
    cache_name: str,
    owner_label: str,
    accum_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    if rows <= 1:
        return _fp8_cached_bf16_weight_local_projection(
            x,
            cached_weight,
            owner=owner,
            cache_name=cache_name,
            owner_label=owner_label,
            accum_dtype=accum_dtype,
        )
    x_2d = x.reshape(rows, x.shape[-1])
    chunks = [
        _fp8_cached_bf16_weight_local_projection(
            x_2d[row : row + 1],
            cached_weight,
            owner=owner,
            cache_name=cache_name,
            owner_label=owner_label,
            accum_dtype=accum_dtype,
        )
        for row in range(rows)
    ]
    y = torch.cat(chunks, dim=0)
    return y.reshape(*x.shape[:-1], y.shape[-1])


def _row_invariant_all_reduce(
    comm: DistributedCommunicator,
    y: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    rows = y.numel() // y.shape[-1]
    if rows <= 1:
        return comm.all_reduce(y, label=label)
    y_2d = y.reshape(rows, y.shape[-1])
    chunks = [
        comm.all_reduce(y_2d[row : row + 1].contiguous(), label=label)
        for row in range(rows)
    ]
    return torch.cat(chunks, dim=0).reshape_as(y)


def _projection_all_reduce(
    comm: DistributedCommunicator,
    y: torch.Tensor,
    *,
    label: str,
    row_invariant_reduce: bool = False,
    reduce_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    original_dtype = y.dtype
    reduce_input = y if reduce_dtype is None or y.dtype == reduce_dtype else y.to(reduce_dtype)
    if row_invariant_reduce:
        reduced = _row_invariant_all_reduce(comm, reduce_input, label=label)
    else:
        reduced = comm.all_reduce(reduce_input, label=label)
    if reduced.dtype != original_dtype:
        reduced = reduced.to(original_dtype)
    return reduced


def _prepare_bf16_pretransposed_report(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    *,
    owner_label: str,
) -> dict[str, object]:
    cached = _cached_bf16_pretransposed_weight(
        owner,
        f"{cache_name}_pretransposed",
        weight,
        allow_build=True,
        owner_label=owner_label,
    )
    return {
        "shape": list(cached.shape),
        "dtype": str(cached.dtype),
        "device": str(cached.device),
        "bytes": int(cached.numel() * cached.element_size()),
    }


def _wo_a_bf16_bmm_weight_cache_meta(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    num_local_groups: int,
    o_lora_rank: int,
    d_per_group: int,
) -> tuple:
    return (
        _fp8_bf16_weight_cache_meta(weight, scale, out_dtype),
        int(num_local_groups),
        int(o_lora_rank),
        int(d_per_group),
    )


def _cached_wo_a_bf16_bmm_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    num_local_groups: int,
    o_lora_rank: int,
    d_per_group: int,
    allow_build: bool,
    owner_label: str,
) -> torch.Tensor:
    if num_local_groups <= 0:
        raise RuntimeError(f"{owner_label} BF16 BMM cache requires num_local_groups > 0.")
    if o_lora_rank <= 0:
        raise RuntimeError(f"{owner_label} BF16 BMM cache requires o_lora_rank > 0.")
    if d_per_group <= 0:
        raise RuntimeError(f"{owner_label} BF16 BMM cache requires d_per_group > 0.")
    expected_shape = (num_local_groups * o_lora_rank, d_per_group)
    if tuple(weight.shape) != expected_shape:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache expected FP8 weight shape {expected_shape}, "
            f"got {tuple(weight.shape)}."
        )
    if weight.dtype != dsv4_kernel.fp8_dtype():
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache requires FP8 weights, got {weight.dtype}."
        )
    if scale is not None and scale.device != weight.device:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache requires scale on the same device as weight, "
            f"got weight={weight.device} scale={scale.device}."
        )
    if out_dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache requires out_dtype=torch.bfloat16, got {out_dtype}."
        )

    meta_name = f"{cache_name}_meta"
    meta = _wo_a_bf16_bmm_weight_cache_meta(
        weight,
        scale,
        out_dtype=out_dtype,
        num_local_groups=num_local_groups,
        o_lora_rank=o_lora_rank,
        d_per_group=d_per_group,
    )
    cached = getattr(owner, cache_name, None)
    if cached is not None and getattr(owner, meta_name, None) == meta:
        return cached

    if not allow_build:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache is missing or stale. "
            "Call prepare_for_cuda_graph_capture() after weights are loaded and before "
            "decode CUDA graph capture/replay; rebuilding inside forward is disabled."
        )

    dequant = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=out_dtype)
    cached = dequant.view(num_local_groups, o_lora_rank, d_per_group).transpose(1, 2).contiguous()
    setattr(owner, cache_name, cached)
    setattr(owner, meta_name, meta)
    return cached


def _wo_a_bf16_bmm_projection(
    o: torch.Tensor,
    cached_weight: torch.Tensor,
    *,
    owner_label: str,
) -> torch.Tensor:
    if o.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection requires bf16 activations, got {o.dtype}."
        )
    if cached_weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection requires bf16 cached weight, "
            f"got {cached_weight.dtype}."
        )
    if o.ndim != 3 or cached_weight.ndim != 3:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection expects o=[tokens, groups, d] and "
            f"weight=[groups, d, rank], got o={tuple(o.shape)} weight={tuple(cached_weight.shape)}."
        )
    tokens, num_local_groups, d_per_group = o.shape
    if cached_weight.shape[0] != num_local_groups or cached_weight.shape[1] != d_per_group:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection shape mismatch: "
            f"o={tuple(o.shape)} weight={tuple(cached_weight.shape)}."
        )
    x = o.transpose(0, 1).contiguous()
    y = torch.bmm(x, cached_weight)
    return y.transpose(0, 1).reshape(tokens, num_local_groups * cached_weight.shape[2])


def _wo_a_bf16_bmm_projection_row_invariant(
    o: torch.Tensor,
    cached_weight: torch.Tensor,
    *,
    owner_label: str,
) -> torch.Tensor:
    if o.ndim != 3 or o.shape[0] <= 1:
        return _wo_a_bf16_bmm_projection(
            o,
            cached_weight,
            owner_label=owner_label,
        )
    return torch.cat(
        [
            _wo_a_bf16_bmm_projection(
                o[row : row + 1],
                cached_weight,
                owner_label=owner_label,
            )
            for row in range(o.shape[0])
        ],
        dim=0,
    )


def _wo_a_bf16_sglang_style_einsum_projection(
    o: torch.Tensor,
    cached_weight: torch.Tensor,
    *,
    owner_label: str,
) -> torch.Tensor:
    if o.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} SGLang-style wo_a projection requires bf16 activations, "
            f"got {o.dtype}."
        )
    if cached_weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} SGLang-style wo_a projection requires bf16 cached weight, "
            f"got {cached_weight.dtype}."
        )
    if o.ndim != 3 or cached_weight.ndim != 3:
        raise RuntimeError(
            f"{owner_label} SGLang-style wo_a projection expects o=[tokens, groups, d] "
            f"and weight=[groups, d, rank], got o={tuple(o.shape)} "
            f"weight={tuple(cached_weight.shape)}."
        )
    weight = cached_weight.transpose(1, 2).contiguous()
    return torch.einsum("tgd,grd->tgr", o, weight).reshape(o.shape[0], -1)


def _cached_gate_fp32_weight(owner: object, cache_name: str, weight: torch.Tensor) -> torch.Tensor:
    return _cached_fp32_weight(
        owner,
        cache_name,
        weight,
        toggle="MINISGL_DSV4_SM80_GATE_FP32_WEIGHT_CACHE",
    )


def _cached_indexer_store_norm_fp32_weight(
    owner: object, cache_name: str, weight: torch.Tensor
) -> torch.Tensor:
    return _cached_fp32_weight(
        owner,
        cache_name,
        weight,
        toggle="MINISGL_DSV4_SM80_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE",
    )


def _cached_fused_wqa_wkv_fp8_weight(
    owner: object,
    cache_name: str,
    weight_q: torch.Tensor,
    scale_q: torch.Tensor | None,
    weight_kv: torch.Tensor,
    scale_kv: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    if not (
        dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE")
        and weight_q.is_cuda
        and weight_kv.is_cuda
        and weight_q.dtype is dsv4_kernel.fp8_dtype()
        and weight_kv.dtype is dsv4_kernel.fp8_dtype()
        and out_dtype is torch.bfloat16
        and weight_q.ndim == 2
        and weight_kv.ndim == 2
        and weight_q.shape[-1] == weight_kv.shape[-1]
    ):
        return None
    if scale_q is not None and not scale_q.is_cuda:
        return None
    if scale_kv is not None and not scale_kv.is_cuda:
        return None

    def _tensor_meta(tensor: torch.Tensor | None):
        if tensor is None:
            return None
        return (
            tensor.data_ptr(),
            int(getattr(tensor, "_version", 0)),
            tensor.device.type,
            tensor.device.index,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    meta_name = f"{cache_name}_meta"
    meta = (
        _tensor_meta(weight_q),
        _tensor_meta(scale_q),
        _tensor_meta(weight_kv),
        _tensor_meta(scale_kv),
        out_dtype,
    )
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        q = dsv4_kernel.dequant_fp8_weight(weight_q, scale_q, out_dtype=out_dtype)
        kv = dsv4_kernel.dequant_fp8_weight(weight_kv, scale_kv, out_dtype=out_dtype)
        cached = torch.cat((q, kv), dim=0).contiguous()
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)
    return cached


@dataclass
class DSV4FallbackAttentionMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class DSV4RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.empty(size, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dsv4_kernel.rms_norm_fallback(x, self.weight, eps=self.eps)


class DSV4VocabParallelEmbedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        tp = get_tp_info()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, tp.size)
        start_idx = self.num_embeddings_tp * tp.rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim, dtype=torch.bfloat16)
        self._comm = DistributedCommunicator()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_ids.long(), self.weight)
        start, end = self.vocab_range
        local_ids = input_ids.long() - start
        mask = (local_ids < 0) | (local_ids >= end - start)
        local_ids = local_ids.masked_fill(mask, 0)
        y = F.embedding(local_ids, self.weight)
        y = y.masked_fill(mask.unsqueeze(-1), 0)
        return self._comm.all_reduce(y, label="dsv4.embedding_all_reduce")

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.weight.float())
        if self.tp_size == 1:
            return logits[:, : self.num_embeddings]
        gathered = self._comm.all_gather(logits, label="dsv4.lm_head_all_gather")
        if x.shape[0] == 1:
            return gathered.view(1, -1)[:, : self.num_embeddings]
        output = gathered.view((self.tp_size,) + tuple(logits.shape))
        output = output.permute(1, 0, 2).contiguous()
        return output.reshape(x.shape[0], self.tp_size * logits.shape[1])[:, : self.num_embeddings]


class DSV4Linear(BaseOP):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        weight_dtype: torch.dtype = torch.bfloat16,
        scale_dtype: torch.dtype | None = None,
        col_parallel: bool = False,
        row_parallel: bool = False,
    ):
        tp = get_tp_info()
        assert not (col_parallel and row_parallel)
        self.row_parallel = row_parallel
        self.col_parallel = col_parallel
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        local_input_size = div_even(input_size, tp.size) if row_parallel else input_size
        local_output_size = div_even(output_size, tp.size) if col_parallel else output_size
        self.weight = torch.empty(local_output_size, local_input_size, dtype=weight_dtype)
        if scale_dtype is not None:
            self.weight_scale_inv = torch.empty(
                dsv4_kernel.scale_dim(local_output_size),
                dsv4_kernel.scale_dim(local_input_size),
                dtype=scale_dtype,
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        reduce: bool = True,
        reduce_label: str | None = None,
        fp8_gemm: bool | None = None,
        debug_pre_reduce_name: str | None = None,
        debug_post_reduce_name: str | None = None,
        row_invariant_reduce: bool = False,
        reduce_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        scale = getattr(self, "weight_scale_inv", None)
        scale = _cached_projection_scale(self, "_dsv4_weight_scale_fp32_contiguous", scale)
        if self.weight.dtype is torch.int8:
            y = dsv4_kernel.quantized_linear_ref(x, self.weight, scale, weight_kind="fp4")
        elif self.weight.dtype is dsv4_kernel.fp8_dtype():
            y = dsv4_kernel.quantized_linear_ref(
                x,
                self.weight,
                scale,
                weight_kind="fp8",
                fp8_gemm=fp8_gemm,
            )
        else:
            y = F.linear(x, self.weight.to(x.dtype))
        if debug_pre_reduce_name is not None:
            _capture_debug_activation(debug_pre_reduce_name, y)
        if reduce and self.row_parallel and self._tp_size > 1:
            label = reduce_label or "dsv4.row_parallel_projection_all_reduce"
            y = _projection_all_reduce(
                self._comm,
                y,
                label=label,
                row_invariant_reduce=row_invariant_reduce,
                reduce_dtype=reduce_dtype,
            )
        if debug_post_reduce_name is not None:
            _capture_debug_activation(debug_post_reduce_name, y)
        return y

    def prepare_fp8_bf16_weight_cache(
        self,
        cache_name: str,
        *,
        owner_label: str,
    ) -> dict[str, object]:
        scale = getattr(self, "weight_scale_inv", None)
        cached = _cached_fp8_bf16_weight(
            self,
            cache_name,
            self.weight,
            scale,
            out_dtype=torch.bfloat16,
            allow_build=True,
            owner_label=owner_label,
        )
        pretransposed_report = None
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE):
            pretransposed_report = _prepare_bf16_pretransposed_report(
                self,
                cache_name,
                cached,
                owner_label=owner_label,
            )
        return {
            "owner": owner_label,
            "shape": list(cached.shape),
            "dtype": str(cached.dtype),
            "device": str(cached.device),
            "bytes": int(cached.numel() * cached.element_size()),
            "pretransposed": pretransposed_report,
            "pretransposed_bytes": (
                0 if pretransposed_report is None else int(pretransposed_report["bytes"])
            ),
        }

    def prepare_fp8_marlin_weight_cache(
        self,
        cache_name: str,
        *,
        owner_label: str,
        release_original: bool = True,
    ) -> dict[str, object]:
        scale = getattr(self, "weight_scale_inv", None)
        return _prepare_fp8_marlin_weight(
            self,
            cache_name,
            self.weight,
            scale,
            owner_label=owner_label,
            release_original=release_original,
        )

    def forward_fp8_marlin_weight(
        self,
        x: torch.Tensor,
        *,
        cache_name: str,
        owner_label: str,
        reduce: bool = False,
        reduce_label: str | None = None,
        debug_pre_reduce_name: str | None = None,
        debug_post_reduce_name: str | None = None,
        row_invariant_reduce: bool = False,
        reduce_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        y = _forward_fp8_marlin_weight(
            self,
            cache_name,
            x,
            owner_label=owner_label,
        )
        if debug_pre_reduce_name is not None:
            _capture_debug_activation(debug_pre_reduce_name, y)
        if reduce and self.row_parallel and self._tp_size > 1:
            label = reduce_label or "dsv4.row_parallel_projection_all_reduce"
            y = _projection_all_reduce(
                self._comm,
                y,
                label=label,
                row_invariant_reduce=row_invariant_reduce,
                reduce_dtype=reduce_dtype,
            )
        if debug_post_reduce_name is not None:
            _capture_debug_activation(debug_post_reduce_name, y)
        return y

    def forward_fp8_cached_bf16_weight(
        self,
        x: torch.Tensor,
        *,
        cache_name: str,
        owner_label: str,
        reduce: bool = False,
        reduce_label: str | None = None,
        debug_pre_reduce_name: str | None = None,
        debug_post_reduce_name: str | None = None,
        row_invariant_local: bool = False,
        row_invariant_reduce: bool = False,
        local_accum_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        scale = getattr(self, "weight_scale_inv", None)
        cached_weight = _cached_fp8_bf16_weight(
            self,
            cache_name,
            self.weight,
            scale,
            out_dtype=x.dtype,
            allow_build=False,
            owner_label=owner_label,
        )
        local_projection = (
            _fp8_cached_bf16_weight_local_projection_row_invariant
            if row_invariant_local
            else _fp8_cached_bf16_weight_local_projection
        )
        y = local_projection(
            x,
            cached_weight,
            owner=self,
            cache_name=cache_name,
            owner_label=owner_label,
            accum_dtype=local_accum_dtype,
        )
        if debug_pre_reduce_name is not None:
            _capture_debug_activation(debug_pre_reduce_name, y)
        if reduce and self.row_parallel and self._tp_size > 1:
            label = reduce_label or "dsv4.row_parallel_projection_all_reduce"
            y = _projection_all_reduce(
                self._comm,
                y,
                label=label,
                row_invariant_reduce=row_invariant_reduce,
                reduce_dtype=reduce_dtype,
            )
        if debug_post_reduce_name is not None:
            _capture_debug_activation(debug_post_reduce_name, y)
        return y

    def forward_fp8_cached_bf16_weight_row_invariant(
        self,
        x: torch.Tensor,
        *,
        cache_name: str,
        owner_label: str,
        reduce: bool = False,
        reduce_label: str | None = None,
        debug_pre_reduce_name: str | None = None,
        debug_post_reduce_name: str | None = None,
        row_invariant_reduce: bool = False,
        reduce_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.forward_fp8_cached_bf16_weight(
            x,
            cache_name=cache_name,
            owner_label=owner_label,
            reduce=reduce,
            reduce_label=reduce_label,
            debug_pre_reduce_name=debug_pre_reduce_name,
            debug_post_reduce_name=debug_post_reduce_name,
            row_invariant_local=True,
            row_invariant_reduce=row_invariant_reduce,
            local_accum_dtype=None,
            reduce_dtype=reduce_dtype,
        )


class DSV4Compressor(BaseOP):
    def __init__(self, config: ModelConfig, ratio: int, head_dim: int):
        self.ratio = ratio
        self.head_dim = head_dim
        self.overlap = ratio == 4
        coff = 2 if ratio == 4 else 1
        self.ape = torch.empty(ratio, coff * head_dim, dtype=torch.float32)
        self.wkv_gate = DSV4Linear(
            config.hidden_size,
            2 * coff * head_dim,
            weight_dtype=torch.bfloat16,
        )
        self.norm = DSV4RMSNorm(head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        apply_norm: bool = True,
    ) -> torch.Tensor:
        return dsv4_kernel.compress_forward_fallback(
            x,
            positions,
            ratio=self.ratio,
            head_dim=self.head_dim,
            overlap=self.overlap,
            ape=self.ape,
            wkv_gate=self.wkv_gate,
            norm=self.norm,
            apply_norm=apply_norm,
        )


class DSV4Indexer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = layer_id
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.weight_scale = (self.head_dim**-0.5) * (self.n_heads**-0.5)
        self.wq_b = DSV4Linear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.weights_proj = DSV4Linear(
            config.hidden_size,
            self.n_heads,
            weight_dtype=torch.bfloat16,
        )
        self.compressor = DSV4Compressor(config, ratio=4, head_dim=self.head_dim)

    @property
    def _wq_b_bf16_weight_cache_name(self) -> str:
        return "_dsv4_indexer_wq_b_bf16_weight_cache"

    @property
    def _wq_b_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.indexer.wq_b"

    def prepare_wq_b_bf16_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE
        ):
            return None
        return self.wq_b.prepare_fp8_bf16_weight_cache(
            self._wq_b_bf16_weight_cache_name,
            owner_label=self._wq_b_owner_label,
        )

    def _wq_b_forward(self, q_lora: torch.Tensor) -> torch.Tensor:
        with _dsv4_capture_nvtx("indexer.wq_b"):
            if dsv4_kernel.dsv4_env_flag(
                dsv4_kernel.DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE
            ):
                return self.wq_b.forward_fp8_cached_bf16_weight(
                    q_lora,
                    cache_name=self._wq_b_bf16_weight_cache_name,
                    owner_label=self._wq_b_owner_label,
                )
            fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_INDEXER_WQB_FP8_GEMM"
            )
            return self.wq_b.forward(q_lora, fp8_gemm=fp8_gemm if fp8_gemm else None)

    def prepare_bf16_query(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        *,
        rotary_dim: int,
        base: float,
        original_seq_len: int,
        factor: float,
        beta_fast: int,
        beta_slow: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._wq_b_forward(q_lora).view(-1, self.n_heads, self.head_dim)
        q = dsv4_kernel.indexer_q_rope_hadamard_bf16_fallback(
            q,
            positions,
            rotary_dim=rotary_dim,
            base=base,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        with _dsv4_capture_nvtx("indexer.weights_proj"):
            weights = self.weights_proj.forward(x) * self.weight_scale
        return q, weights

    def prepare_fp8_query(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        *,
        rotary_dim: int,
        base: float,
        original_seq_len: int,
        factor: float,
        beta_fast: int,
        beta_slow: int,
    ) -> dsv4_kernel.DSV4IndexerFP8Query:
        q = self._wq_b_forward(q_lora).view(-1, self.n_heads, self.head_dim)
        with _dsv4_capture_nvtx("indexer.weights_proj"):
            weights = self.weights_proj.forward(x)
        return dsv4_kernel.indexer_q_rope_fp8_fallback(
            q,
            weights,
            positions,
            rotary_dim=rotary_dim,
            base=base,
            softmax_scale=self.head_dim**-0.5,
            head_scale=self.n_heads**-0.5,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        *,
        apply_norm: bool = True,
        touch_projections: bool = True,
    ) -> torch.Tensor:
        with _dsv4_capture_nvtx("indexer.compressor"):
            compressed_kv = self.compressor.forward(x, positions, apply_norm=apply_norm)
        if touch_projections:
            self._wq_b_forward(q_lora)
            with _dsv4_capture_nvtx("indexer.weights_proj"):
                self.weights_proj.forward(x)
        return compressed_kv


class DSV4Attention(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        compress_ratio_override: int | None = None,
    ):
        tp = get_tp_info()
        self.layer_id = layer_id
        self.num_heads = config.num_qo_heads
        self.num_local_heads = div_even(config.num_qo_heads, tp.size)
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.window_size = config.window_size
        self.softmax_scale = config.head_dim**-0.5
        self.o_groups = config.o_groups
        self.num_local_groups = div_even(config.o_groups, tp.size)
        self.o_lora_rank = config.o_lora_rank
        self.rms_norm_eps = config.rms_norm_eps
        ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else (config.compress_ratios[layer_id] if layer_id < len(config.compress_ratios) else 0)
        )
        self.compress_ratio = ratio
        self.rope_base = (
            config.compress_rope_theta
            if ratio and config.compress_rope_theta is not None
            else config.rotary_config.base
        )
        self.original_seq_len = config.original_seq_len if ratio else 0
        self.rope_factor = config.rope_factor
        self.beta_fast = config.beta_fast
        self.beta_slow = config.beta_slow
        self.attn_sink = torch.empty(self.num_local_heads, dtype=torch.float32)
        self.wq_a = DSV4Linear(
            config.hidden_size,
            config.q_lora_rank,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.wq_b = DSV4Linear(
            config.q_lora_rank,
            config.num_qo_heads * config.head_dim,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            col_parallel=True,
        )
        self.q_norm = DSV4RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wkv = DSV4Linear(
            config.hidden_size,
            config.head_dim,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.kv_norm = DSV4RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wo_a = DSV4Linear(
            config.num_qo_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            col_parallel=True,
        )
        self.wo_b = DSV4Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            row_parallel=True,
        )

        if ratio in (4, 128):
            self.compressor = DSV4Compressor(config, ratio=ratio, head_dim=config.head_dim)
        if ratio == 4:
            self.indexer = DSV4Indexer(config, layer_id)

    @staticmethod
    def _swa_store_out_loc(attn_backend, batch: Batch | torch.Tensor) -> torch.Tensor:
        out_loc = batch.out_loc if isinstance(batch, Batch) else batch
        metadata = getattr(batch, "attn_metadata", None) if isinstance(batch, Batch) else None
        if isinstance(metadata, DSV4AttentionMetadata):
            cached = getattr(metadata.core_metadata, "swa_out_loc", None)
            rows = int(out_loc.shape[0])
            if cached is not None and int(cached.shape[0]) >= rows:
                return cached[:rows]
        kvcache = getattr(attn_backend, "kvcache", None)
        translate = getattr(kvcache, "translate_full_locs_to_swa_locs", None)
        if callable(translate) and bool(
            getattr(kvcache, "swa_independent_lifecycle_enabled", False)
        ):
            return translate(out_loc).to(device=out_loc.device, dtype=out_loc.dtype)
        return out_loc

    @property
    def _q_wqb_bf16_weight_cache_name(self) -> str:
        return "_dsv4_q_wqb_bf16_weight_cache"

    @property
    def _q_wqb_marlin_weight_cache_name(self) -> str:
        return "_dsv4_q_wqb_dense_fp8_marlin_weight_cache"

    @property
    def _q_wqb_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.q_wqb"

    @property
    def _wo_b_bf16_weight_cache_name(self) -> str:
        return "_dsv4_wo_b_bf16_weight_cache"

    @property
    def _wo_b_marlin_weight_cache_name(self) -> str:
        return "_dsv4_wo_b_dense_fp8_marlin_weight_cache"

    @property
    def _wo_b_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.wo_b"

    @property
    def _wo_a_bf16_bmm_cache_name(self) -> str:
        return "_dsv4_wo_a_bf16_bmm_weight_cache"

    @property
    def _wo_a_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.wo_a"

    def _wo_a_d_per_group(self) -> int:
        return self.num_local_heads * self.head_dim // self.num_local_groups

    def prepare_q_wqb_bf16_weight_cache(self) -> dict[str, object] | None:
        if dsv4_kernel.dense_fp8_marlin_projection_enabled():
            return None
        if not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE):
            return None
        return self.wq_b.prepare_fp8_bf16_weight_cache(
            self._q_wqb_bf16_weight_cache_name,
            owner_label=self._q_wqb_owner_label,
        )

    def prepare_q_wqb_marlin_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dense_fp8_marlin_projection_enabled():
            return None
        return self.wq_b.prepare_fp8_marlin_weight_cache(
            self._q_wqb_marlin_weight_cache_name,
            owner_label=self._q_wqb_owner_label,
        )

    def prepare_wo_b_bf16_weight_cache(self) -> dict[str, object] | None:
        if dsv4_kernel.dense_fp8_marlin_projection_enabled():
            return None
        if not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_B_BF16_WEIGHT_CACHE_TOGGLE):
            return None
        return self.wo_b.prepare_fp8_bf16_weight_cache(
            self._wo_b_bf16_weight_cache_name,
            owner_label=self._wo_b_owner_label,
        )

    def prepare_wo_b_marlin_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dense_fp8_marlin_projection_enabled():
            return None
        return self.wo_b.prepare_fp8_marlin_weight_cache(
            self._wo_b_marlin_weight_cache_name,
            owner_label=self._wo_b_owner_label,
        )

    def prepare_wo_a_bf16_bmm_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE):
            return None
        scale = getattr(self.wo_a, "weight_scale_inv", None)
        d_per_group = self._wo_a_d_per_group()
        cached = _cached_wo_a_bf16_bmm_weight(
            self.wo_a,
            self._wo_a_bf16_bmm_cache_name,
            self.wo_a.weight,
            scale,
            out_dtype=torch.bfloat16,
            num_local_groups=self.num_local_groups,
            o_lora_rank=self.o_lora_rank,
            d_per_group=d_per_group,
            allow_build=True,
            owner_label=self._wo_a_owner_label,
        )
        return {
            "owner": self._wo_a_owner_label,
            "shape": list(cached.shape),
            "source_weight_shape": list(self.wo_a.weight.shape),
            "scale_shape": list(scale.shape) if scale is not None else None,
            "dtype": str(cached.dtype),
            "device": str(cached.device),
            "bytes": int(cached.numel() * cached.element_size()),
            "num_local_groups": int(self.num_local_groups),
            "d_per_group": int(d_per_group),
            "o_lora_rank": int(self.o_lora_rank),
        }

    def prepare_indexer_wq_b_bf16_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE
        ):
            return None
        if not hasattr(self, "indexer"):
            return None
        return self.indexer.prepare_wq_b_bf16_weight_cache()

    def prepare_fused_wqa_wkv_pretranspose_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE):
            return None
        cached = _cached_fused_wqa_wkv_fp8_weight(
            self,
            "_cached_fused_wqa_wkv_bf16_weight",
            self.wq_a.weight,
            getattr(self.wq_a, "weight_scale_inv", None),
            self.wkv.weight,
            getattr(self.wkv, "weight_scale_inv", None),
            out_dtype=torch.bfloat16,
        )
        if cached is None:
            return None
        owner_label = f"layer{self.layer_id}.attn.q_proj"
        pretransposed_report = _prepare_bf16_pretransposed_report(
            self,
            "_cached_fused_wqa_wkv_bf16_weight",
            cached,
            owner_label=owner_label,
        )
        return {
            "owner": owner_label,
            "shape": list(cached.shape),
            "dtype": str(cached.dtype),
            "device": str(cached.device),
            "bytes": int(cached.numel() * cached.element_size()),
            "pretransposed": pretransposed_report,
            "pretransposed_bytes": int(pretransposed_report["bytes"]),
        }

    def _sequence_spans(self, batch: Batch, total_tokens: int) -> list[tuple[int, int]]:
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        spans = []
        offset = 0
        for req in reqs:
            length = req.extend_len if batch.is_prefill else 1
            spans.append((offset, offset + length))
            offset += length
        if offset != total_tokens:
            return [(0, total_tokens)]
        return spans

    def _fallback_attention(self, q: torch.Tensor, kv: torch.Tensor, batch: Batch) -> torch.Tensor:
        spans = self._sequence_spans(batch, q.shape[0])
        return dsv4_kernel.sequence_mqa_attention_fallback(
            q,
            kv,
            spans,
            window_size=self.window_size,
            softmax_scale=self.softmax_scale,
            attn_sink=self.attn_sink,
        )

    def _q_wqb_per_row_probe(
        self,
        q_lora: torch.Tensor,
        q_batched: torch.Tensor,
        *,
        path: str,
    ) -> dict[str, object] | None:
        if not dsv4_mtp_debug.operator_parity_enabled("wq_b"):
            return None
        rows = q_lora.numel() // q_lora.shape[-1]
        if rows <= 1:
            return {
                "available": False,
                "reason": "single row",
                "rows": int(rows),
            }
        try:
            q_lora_2d = q_lora.reshape(rows, q_lora.shape[-1])
            if dsv4_kernel.dense_fp8_marlin_projection_enabled():
                chunks = [
                    self.wq_b.forward_fp8_marlin_weight(
                        q_lora_2d[row : row + 1],
                        cache_name=self._q_wqb_marlin_weight_cache_name,
                        owner_label=self._q_wqb_owner_label,
                    )
                    for row in range(rows)
                ]
            elif dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE):
                chunks = [
                    self.wq_b.forward_fp8_cached_bf16_weight(
                        q_lora_2d[row : row + 1],
                        cache_name=self._q_wqb_bf16_weight_cache_name,
                        owner_label=self._q_wqb_owner_label,
                    )
                    for row in range(rows)
                ]
            else:
                q_wqb_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                    "MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM"
                )
                chunks = [
                    self.wq_b.forward(
                        q_lora_2d[row : row + 1],
                        fp8_gemm=q_wqb_fp8_gemm if q_wqb_fp8_gemm else None,
                    )
                    for row in range(rows)
                ]
            q_per_row = torch.cat(chunks, dim=0).view_as(q_batched)
            if q_per_row.is_cuda:
                torch.cuda.synchronize(q_per_row.device)
            return {
                "available": True,
                "source": "same q_wqb path executed one row at a time",
                "path": path,
                "rows": int(rows),
                "per_row_vs_batched": dsv4_mtp_debug.tensor_compare_stats(
                    q_per_row,
                    q_batched,
                ),
            }
        except Exception as exc:
            return {
                "available": False,
                "path": path,
                "rows": int(rows),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def _q_wqb_weight_cache_probe(self, *, path: str) -> dict[str, object]:
        probe: dict[str, object] = {
            "available": True,
            "owner_label": self._q_wqb_owner_label,
            "path": str(path),
            "dense_fp8_marlin_projection": bool(
                dsv4_kernel.dense_fp8_marlin_projection_enabled()
            ),
            "q_wqb_bf16_weight_cache": bool(
                dsv4_kernel.dsv4_env_flag(
                    dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE
                )
            ),
            "q_wqb_fp8_gemm": bool(
                dsv4_kernel.dsv4_sm80_triton_enabled(
                    "MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM"
                )
            ),
            "source_weight": dsv4_mtp_debug.tensor_probe_record(
                getattr(self.wq_b, "weight", None)
            ),
            "source_scale": dsv4_mtp_debug.tensor_probe_record(
                getattr(self.wq_b, "weight_scale_inv", None)
            ),
        }
        if dsv4_kernel.dense_fp8_marlin_projection_enabled():
            cache_name = self._q_wqb_marlin_weight_cache_name
            cached = getattr(self.wq_b, cache_name, None)
            probe["cache_name"] = cache_name
            probe["cache_kind"] = "dense_fp8_marlin"
            if cached is None:
                probe["cache"] = {"available": False}
            else:
                probe["cache"] = {
                    "available": True,
                    "shape": [int(cached.size_n), int(cached.size_k)],
                    "source_signature": [
                        {
                            "data_ptr": int(sig[0]),
                            "shape": [int(x) for x in sig[1]],
                            "dtype": str(sig[2]),
                        }
                        for sig in getattr(cached, "source_signature", ())
                    ],
                    "prepared_weight": dsv4_mtp_debug.tensor_probe_record(
                        getattr(cached, "weight", None)
                    ),
                    "prepared_scale": dsv4_mtp_debug.tensor_probe_record(
                        getattr(cached, "weight_scale", None)
                    ),
                    "workspace": dsv4_mtp_debug.tensor_probe_record(
                        getattr(cached, "workspace", None)
                    ),
                }
            return probe

        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE):
            cache_name = self._q_wqb_bf16_weight_cache_name
            probe["cache_name"] = cache_name
            probe["cache_kind"] = "fp8_cached_bf16_weight"
            probe["cache"] = dsv4_mtp_debug.tensor_probe_record(
                getattr(self.wq_b, cache_name, None)
            )
            return probe

        probe["cache_name"] = None
        probe["cache_kind"] = "quantized_linear_ref"
        probe["cache"] = {"available": False, "reason": "no prepared q_wqb cache path"}
        return probe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        with dsv4_direct_copy_nvtx(
            f"attention_boundary.positions_to_i64.layer{self.layer_id}",
            positions=batch.positions,
        ):
            positions = batch.positions.to(device=x.device, dtype=torch.long)
        attn_backend = getattr(get_global_ctx(), "attn_backend", None)
        attn_metadata = getattr(batch, "attn_metadata", None)
        use_dsv4_backend = isinstance(attn_metadata, DSV4AttentionMetadata)
        read_only_frozen_kv = bool(getattr(batch, "frozen_kv_read_only", False))
        force_verify_exact_kv_store = bool(
            getattr(batch, "dsv4_force_exact_kv_store", False)
        )
        is_target_verify = getattr(batch, "dsv4_target_verify_metadata", None) is not None
        kv_norm_rope_store_enabled = (
            use_dsv4_backend
            and attn_backend is not None
            and not read_only_frozen_kv
            and not force_verify_exact_kv_store
            and x.is_cuda
            and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_KV_BF16")
        )
        fused_q_kv_rmsnorm = (
            dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_FUSED_Q_KV_RMSNORM")
            and not kv_norm_rope_store_enabled
        )
        fused_q_kv_norm_rope_store = (
            kv_norm_rope_store_enabled
            and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE")
        )
        q_hidden_debug_input = dsv4_mtp_debug.clone_operator_row0_input(
            "q_path_hidden_input",
            x,
        )
        dsv4_mtp_debug.record_operator_capture(
            batch,
            operator_name="q_path_hidden_input",
            layer_id=int(self.layer_id),
            input_row0=q_hidden_debug_input,
            output_tensor=x,
            positions=positions,
            path="mini.attention.q_path_hidden_input",
            params={"hidden_size": int(x.shape[-1])},
            extra={"is_target_verify": bool(is_target_verify)},
        )
        kv_from_shared_wqa_wkv = None
        kv = None
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_proj"):
            q_wqa_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_Q_WQA_FP8_GEMM"
            )
            fused_wqa_wkv_shared_act = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT"
            )
            wq_a_debug_input = dsv4_mtp_debug.clone_operator_row0_input("wq_a", x)
            wq_a_path = "mini.wq_a.quantized_linear_ref"
            if fused_wqa_wkv_shared_act:
                cached_fused_weight = _cached_fused_wqa_wkv_fp8_weight(
                    self,
                    "_cached_fused_wqa_wkv_bf16_weight",
                    self.wq_a.weight,
                    getattr(self.wq_a, "weight_scale_inv", None),
                    self.wkv.weight,
                    getattr(self.wkv, "weight_scale_inv", None),
                    out_dtype=x.dtype,
                )
                if cached_fused_weight is not None:
                    wq_a_path = "mini.fused_wqa_wkv.cached_bf16_weight.shared_fp8_activation"
                    x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
                    qkv = _linear_cached_bf16_weight(
                        x_quant,
                        cached_fused_weight,
                        owner=self,
                        cache_name="_cached_fused_wqa_wkv_bf16_weight",
                        owner_label=f"layer{self.layer_id}.attn.q_proj",
                    )
                    q_lora_raw, kv_from_shared_wqa_wkv = qkv.split(
                        [self.q_norm.weight.shape[0], self.head_dim],
                        dim=-1,
                    )
                else:
                    wq_a_path = "mini.fused_wqa_wkv.quantized_linear_fp8_pair_shared_activation_ref"
                    q_lora_raw, kv_from_shared_wqa_wkv = (
                        dsv4_kernel.quantized_linear_fp8_pair_shared_activation_ref(
                            x,
                            self.wq_a.weight,
                            getattr(self.wq_a, "weight_scale_inv", None),
                            self.wkv.weight,
                            getattr(self.wkv, "weight_scale_inv", None),
                        )
                    )
            else:
                if q_wqa_fp8_gemm:
                    wq_a_path = "mini.wq_a.quantized_linear_ref.fp8_gemm"
                q_lora_raw = self.wq_a.forward(
                    x,
                    fp8_gemm=q_wqa_fp8_gemm if q_wqa_fp8_gemm else None,
                )
            _capture_debug_activation(f"layer{self.layer_id}.wqa_output", q_lora_raw)
            dsv4_mtp_debug.record_operator_capture(
                batch,
                operator_name="wq_a",
                layer_id=int(self.layer_id),
                input_row0=wq_a_debug_input,
                output_tensor=q_lora_raw,
                positions=positions,
                path=wq_a_path,
                params={
                    "input_size": int(x.shape[-1]),
                    "q_lora_rank": int(q_lora_raw.shape[-1]),
                    "fused_wqa_wkv_shared_act": bool(fused_wqa_wkv_shared_act),
                    "q_wqa_fp8_gemm": bool(q_wqa_fp8_gemm),
                },
                extra={"output_boundary": "q_lora_raw"},
            )
            if kv_from_shared_wqa_wkv is not None:
                _capture_debug_activation(
                    f"layer{self.layer_id}.wkv_shared_activation_output",
                    kv_from_shared_wqa_wkv,
                )
            if not fused_q_kv_rmsnorm:
                q_norm_debug_input = dsv4_mtp_debug.clone_operator_row0_input(
                    "q_norm",
                    q_lora_raw,
                )
                q_lora = self.q_norm.forward(q_lora_raw)
                _capture_debug_activation(f"layer{self.layer_id}.q_lora_after_norm", q_lora)
                dsv4_mtp_debug.record_operator_capture(
                    batch,
                    operator_name="q_norm",
                    layer_id=int(self.layer_id),
                    input_row0=q_norm_debug_input,
                    output_tensor=q_lora,
                    positions=positions,
                    path="mini.DSV4RMSNorm.forward",
                    params={
                        "rms_norm_eps": float(self.rms_norm_eps),
                        "q_lora_rank": int(q_lora.shape[-1]),
                    },
                    extra={"output_boundary": "q_norm_output_and_wq_b_input"},
                    private={"q_norm_weight": self.q_norm.weight},
                )
        if fused_q_kv_rmsnorm:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
                _capture_debug_activation(f"layer{self.layer_id}.wkv_output", kv)
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_kv_rmsnorm"):
                q_norm_debug_input = dsv4_mtp_debug.clone_operator_row0_input(
                    "q_norm",
                    q_lora_raw,
                )
                q_lora, kv = dsv4_kernel.rms_norm_pair_fallback(
                    q_lora_raw,
                    kv,
                    self.q_norm.weight,
                    self.kv_norm.weight,
                    eps=self.rms_norm_eps,
                )
                _capture_debug_activation(f"layer{self.layer_id}.q_lora_after_norm", q_lora)
                _capture_debug_activation(f"layer{self.layer_id}.kv_after_kv_norm", kv)
                dsv4_mtp_debug.record_operator_capture(
                    batch,
                    operator_name="q_norm",
                    layer_id=int(self.layer_id),
                    input_row0=q_norm_debug_input,
                    output_tensor=q_lora,
                    positions=positions,
                    path="mini.rms_norm_pair_fallback.q_branch",
                    params={
                        "rms_norm_eps": float(self.rms_norm_eps),
                        "q_lora_rank": int(q_lora.shape[-1]),
                        "fused_q_kv_rmsnorm": bool(fused_q_kv_rmsnorm),
                    },
                    extra={"output_boundary": "q_norm_output_and_wq_b_input"},
                    private={"q_norm_weight": self.q_norm.weight},
                )
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_wqb"):
            wq_b_debug_input = dsv4_mtp_debug.clone_operator_row0_input("wq_b", q_lora)
            q_wqb_path = "mini.wq_b.quantized_linear_ref"
            if dsv4_kernel.dense_fp8_marlin_projection_enabled():
                q_wqb_path = "mini.wq_b.forward_fp8_marlin_weight"
                q = self.wq_b.forward_fp8_marlin_weight(
                    q_lora,
                    cache_name=self._q_wqb_marlin_weight_cache_name,
                    owner_label=self._q_wqb_owner_label,
                ).view(-1, self.num_local_heads, self.head_dim)
            elif dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE):
                q_wqb_path = "mini.wq_b.forward_fp8_cached_bf16_weight"
                if is_target_verify:
                    q_wqb_path = "mini.wq_b.forward_fp8_cached_bf16_weight.row_invariant_local"
                q = self.wq_b.forward_fp8_cached_bf16_weight(
                    q_lora,
                    cache_name=self._q_wqb_bf16_weight_cache_name,
                    owner_label=self._q_wqb_owner_label,
                    row_invariant_local=is_target_verify,
                ).view(-1, self.num_local_heads, self.head_dim)
            else:
                q_wqb_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                    "MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM"
                )
                if q_wqb_fp8_gemm:
                    q_wqb_path = "mini.wq_b.quantized_linear_ref.fp8_gemm"
                q = self.wq_b.forward(
                    q_lora,
                    fp8_gemm=q_wqb_fp8_gemm if q_wqb_fp8_gemm else None,
                ).view(-1, self.num_local_heads, self.head_dim)
            _capture_debug_activation(f"layer{self.layer_id}.q_wqb_output", q)
            q_wqb_per_row_probe = self._q_wqb_per_row_probe(
                q_lora,
                q,
                path=q_wqb_path,
            )
            q_wqb_weight_cache_probe = None
            if dsv4_mtp_debug.operator_parity_enabled(
                "wq_b"
            ) and dsv4_mtp_debug.operator_parity_layer_enabled(int(self.layer_id)):
                q_wqb_weight_cache_probe = self._q_wqb_weight_cache_probe(
                    path=q_wqb_path
                )
            dsv4_mtp_debug.record_operator_capture(
                batch,
                operator_name="wq_b",
                layer_id=int(self.layer_id),
                input_row0=wq_b_debug_input,
                input_tensor=q_lora,
                output_tensor=q,
                positions=positions,
                path=q_wqb_path,
                params={
                    "q_lora_rank": int(q_lora.shape[-1]),
                    "num_local_heads": int(self.num_local_heads),
                    "head_dim": int(self.head_dim),
                    "q_wqb_bf16_weight_cache": bool(
                        dsv4_kernel.dsv4_env_flag(
                            dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE
                        )
                    ),
                    "dense_fp8_marlin_projection": bool(
                        dsv4_kernel.dense_fp8_marlin_projection_enabled()
                    ),
                    "row_invariant_local": bool(
                        is_target_verify
                        and dsv4_kernel.dsv4_env_flag(
                            dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE
                        )
                        and not dsv4_kernel.dense_fp8_marlin_projection_enabled()
                    ),
                },
                extra={
                    "output_boundary": "wq_b_local_output_q_wqb_output",
                    "per_row_probe": q_wqb_per_row_probe,
                    "weight_cache_probe": q_wqb_weight_cache_probe,
                },
            )
            q_wqb_output_debug_input = dsv4_mtp_debug.clone_operator_row0_input(
                "q_wqb_output",
                q,
            )
            dsv4_mtp_debug.record_operator_capture(
                batch,
                operator_name="q_wqb_output",
                layer_id=int(self.layer_id),
                input_row0=q_wqb_output_debug_input,
                output_tensor=q,
                positions=positions,
                path=q_wqb_path,
                params={
                    "num_local_heads": int(self.num_local_heads),
                    "head_dim": int(self.head_dim),
                },
                extra={"boundary": "q_wqb_output"},
            )
        if fused_q_kv_norm_rope_store and kv is None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
                _capture_debug_activation(f"layer{self.layer_id}.wkv_output", kv)
        q_norm_rope_debug_input = dsv4_mtp_debug.clone_operator_row0_input(
            "q_norm_rope",
            q,
        )
        q_norm_rope_common_params = {
            "rms_norm_eps": float(self.rms_norm_eps),
            "rotary_dim": int(self.rope_head_dim),
            "base": float(self.rope_base),
            "original_seq_len": int(self.original_seq_len),
            "factor": float(self.rope_factor),
            "beta_fast": int(self.beta_fast),
            "beta_slow": int(self.beta_slow),
            "num_local_heads": int(self.num_local_heads),
            "head_dim": int(self.head_dim),
        }
        q_norm_rope_path = "mini.q_norm_rope_fallback"
        q_kv_norm_rope_cache_written = False
        layer2_swa_lifecycle_trace = (
            dsv4_mtp_debug.swa_lifecycle_trace_enabled(int(self.layer_id))
            and use_dsv4_backend
            and attn_backend is not None
            and not read_only_frozen_kv
            and isinstance(batch, Batch)
        )
        if fused_q_kv_norm_rope_store and kv is not None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_kv_norm_rope_store"):
                swa_store_out_loc = self._swa_store_out_loc(attn_backend, batch)
                swa_cache = attn_backend.kvcache.swa_cache(self.layer_id)
                swa_cache_before = (
                    dsv4_mtp_debug.capture_cache_rows(swa_cache, swa_store_out_loc)
                    if layer2_swa_lifecycle_trace
                    else None
                )
                q_kv_norm_rope_cache_written = dsv4_kernel.q_kv_norm_rope_cache_fallback(
                    q,
                    kv,
                    positions,
                    norm_weight=self.kv_norm.weight,
                    rms_norm_eps=self.rms_norm_eps,
                    cache=swa_cache,
                    out_loc=swa_store_out_loc,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )
                if q_kv_norm_rope_cache_written:
                    dsv4_mtp_debug.record_layer2_swa_store(
                        batch,
                        layer_id=int(self.layer_id),
                        path="q_kv_norm_rope_cache_fallback",
                        kv=kv,
                        full_out_loc=getattr(batch, "out_loc", None),
                        swa_out_loc=swa_store_out_loc,
                        positions=positions,
                        cache_before=swa_cache_before,
                        cache_after=(
                            dsv4_mtp_debug.capture_cache_rows(
                                swa_cache,
                                swa_store_out_loc,
                            )
                            if layer2_swa_lifecycle_trace
                            else None
                        ),
                        extra={
                            "fused_q_kv_norm_rope_store": True,
                            "is_target_verify": bool(is_target_verify),
                        },
                    )
                    q_norm_rope_path = "mini.q_kv_norm_rope_cache_fallback"
                    _capture_debug_activation(f"layer{self.layer_id}.q_after_q_norm_rope", q)
                    _capture_debug_activation(f"layer{self.layer_id}.kv_after_kv_norm_rope", kv)
        if not q_kv_norm_rope_cache_written:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_norm_rope"):
                dsv4_kernel.q_norm_rope_fallback(
                    q,
                    positions,
                    rms_norm_eps=self.rms_norm_eps,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )
                _capture_debug_activation(f"layer{self.layer_id}.q_after_q_norm_rope", q)
        dsv4_mtp_debug.record_operator_capture(
            batch,
            operator_name="q_norm_rope",
            layer_id=int(self.layer_id),
            input_row0=q_norm_rope_debug_input,
            output_tensor=q,
            positions=positions,
            path=q_norm_rope_path,
            params=q_norm_rope_common_params,
            extra={
                "fused_q_kv_norm_rope_store_requested": bool(
                    fused_q_kv_norm_rope_store
                ),
                "q_kv_norm_rope_cache_written": bool(
                    q_kv_norm_rope_cache_written
                ),
                "kv_norm_rope_store_enabled": bool(kv_norm_rope_store_enabled),
                "fused_q_kv_rmsnorm": bool(fused_q_kv_rmsnorm),
                "is_target_verify": bool(is_target_verify),
            },
        )

        if kv is None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
                _capture_debug_activation(f"layer{self.layer_id}.wkv_output", kv)
        kv_cache_written = False
        if kv_norm_rope_store_enabled:
            if not q_kv_norm_rope_cache_written:
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_norm_rope_store"):
                    swa_store_out_loc = self._swa_store_out_loc(attn_backend, batch)
                    swa_cache = attn_backend.kvcache.swa_cache(self.layer_id)
                    swa_cache_before = (
                        dsv4_mtp_debug.capture_cache_rows(swa_cache, swa_store_out_loc)
                        if layer2_swa_lifecycle_trace
                        else None
                    )
                    dsv4_kernel.k_norm_rope_cache_fallback(
                        kv,
                        positions,
                        norm_weight=self.kv_norm.weight,
                        rms_norm_eps=self.rms_norm_eps,
                        cache=swa_cache,
                        out_loc=swa_store_out_loc,
                        rotary_dim=self.rope_head_dim,
                        base=float(self.rope_base),
                        original_seq_len=self.original_seq_len,
                        factor=self.rope_factor,
                        beta_fast=self.beta_fast,
                        beta_slow=self.beta_slow,
                    )
                    dsv4_mtp_debug.record_layer2_swa_store(
                        batch,
                        layer_id=int(self.layer_id),
                        path="kv_norm_rope_cache_fallback",
                        kv=kv,
                        full_out_loc=getattr(batch, "out_loc", None),
                        swa_out_loc=swa_store_out_loc,
                        positions=positions,
                        cache_before=swa_cache_before,
                        cache_after=(
                            dsv4_mtp_debug.capture_cache_rows(
                                swa_cache,
                                swa_store_out_loc,
                            )
                            if layer2_swa_lifecycle_trace
                            else None
                        ),
                        extra={
                            "fused_q_kv_norm_rope_store": False,
                            "is_target_verify": bool(is_target_verify),
                        },
                    )
                    _capture_debug_activation(
                        f"layer{self.layer_id}.kv_after_kv_norm_rope",
                        kv,
                    )
            kv_cache_written = True
        else:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_norm_rope"):
                if not fused_q_kv_rmsnorm:
                    kv = self.kv_norm.forward(kv)
                dsv4_kernel.k_norm_rope_cache_fallback(
                    kv,
                    positions,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )
                _capture_debug_activation(f"layer{self.layer_id}.kv_after_kv_norm_rope", kv)
        if self.rope_head_dim < kv.shape[-1]:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_quant"):
                kv[..., : -self.rope_head_dim] = dsv4_kernel.quantize_fp8_activation_ref(
                    kv[..., : -self.rope_head_dim], block_size=64
                )

        compress_store_fuses_norm = (
            use_dsv4_backend
            and attn_backend is not None
            and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_COMPRESS_STORE")
        )

        if hasattr(self, "indexer"):
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.indexer"):
                indexer_select_fp8 = (
                    use_dsv4_backend
                    and attn_backend is not None
                    and dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_INDEXER_FP8_CACHE_TOGGLE)
                )
                indexer_select_bf16 = (
                    not indexer_select_fp8
                    and use_dsv4_backend
                    and attn_backend is not None
                    and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_INDEXER_BF16")
                )
                indexer_q = None
                indexer_weights = None
                indexer_fp8_query = None
                if indexer_select_fp8:
                    indexer_fp8_query = self.indexer.prepare_fp8_query(
                        x,
                        q_lora,
                        positions,
                        rotary_dim=self.rope_head_dim,
                        base=float(self.rope_base),
                        original_seq_len=self.original_seq_len,
                        factor=self.rope_factor,
                        beta_fast=self.beta_fast,
                        beta_slow=self.beta_slow,
                    )
                if indexer_select_bf16:
                    indexer_q, indexer_weights = self.indexer.prepare_bf16_query(
                        x,
                        q_lora,
                        positions,
                        rotary_dim=self.rope_head_dim,
                        base=float(self.rope_base),
                        original_seq_len=self.original_seq_len,
                        factor=self.rope_factor,
                        beta_fast=self.beta_fast,
                        beta_slow=self.beta_slow,
                    )
                    _capture_debug_activation(f"layer{self.layer_id}.indexer_query_bf16", indexer_q)
                    _capture_debug_activation(
                        f"layer{self.layer_id}.indexer_query_weights",
                        indexer_weights,
                    )
                _capture_compressed_debug_window(
                    f"layer{self.layer_id}.indexer_compressor_input_window",
                    x,
                    positions,
                    4,
                    batch,
                )
                indexer_kv = self.indexer.forward(
                    x,
                    q_lora,
                    positions,
                    apply_norm=not compress_store_fuses_norm,
                    touch_projections=not (indexer_select_bf16 or indexer_select_fp8),
                )
                if indexer_fp8_query is not None:
                    _capture_debug_activation(
                        f"layer{self.layer_id}.indexer_query_fp8_values",
                        indexer_fp8_query.q_values,
                    )
                    _capture_debug_activation(
                        f"layer{self.layer_id}.indexer_query_fp8_weights",
                        indexer_fp8_query.weights,
                    )
                compressed_debug_rows = _compressed_debug_row_indices(
                    positions,
                    4,
                    batch,
                )
                _capture_debug_activation(
                    f"layer{self.layer_id}.indexer_output",
                    indexer_kv,
                    row_indices=compressed_debug_rows,
                )
            if (
                use_dsv4_backend
                and not read_only_frozen_kv
                and hasattr(attn_backend, "store_indexer")
            ):
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.indexer_store"):
                    indexer_store_norm_weight = None
                    if compress_store_fuses_norm:
                        indexer_store_norm_weight = _cached_indexer_store_norm_fp32_weight(
                            self.indexer.compressor.norm,
                            "_dsv4_indexer_store_norm_fp32_weight",
                            self.indexer.compressor.norm.weight,
                        )
                    attn_backend.store_indexer(
                        self.layer_id,
                        indexer_kv,
                        batch,
                        norm_weight=indexer_store_norm_weight,
                        rms_norm_eps=self.rms_norm_eps if compress_store_fuses_norm else None,
                        rotary_dim=self.rope_head_dim,
                        base=float(self.rope_base),
                        original_seq_len=self.original_seq_len,
                        factor=self.rope_factor,
                        beta_fast=self.beta_fast,
                        beta_slow=self.beta_slow,
                        apply_hadamard=indexer_select_bf16,
                    )
            if indexer_select_fp8 and indexer_fp8_query is not None:
                if not hasattr(attn_backend, "select_indexer_fp8"):
                    raise RuntimeError(
                        f"{dsv4_kernel.DSV4_SM80_INDEXER_FP8_CACHE_TOGGLE}=1 requires "
                        "a DSV4 attention backend with select_indexer_fp8."
                    )
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.indexer_select_fp8"):
                    attn_backend.select_indexer_fp8(
                        self.layer_id,
                        indexer_fp8_query.q_values,
                        indexer_fp8_query.weights,
                        batch,
                    )
            if (
                indexer_select_bf16
                and indexer_q is not None
                and indexer_weights is not None
                and hasattr(attn_backend, "select_indexer")
            ):
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.indexer_select"):
                    attn_backend.select_indexer(
                        self.layer_id,
                        indexer_q,
                        indexer_weights,
                        batch,
                    )
        if hasattr(self, "compressor"):
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.compress"):
                _capture_compressed_debug_window(
                    f"layer{self.layer_id}.compressor_input_r{self.compress_ratio}_window",
                    x,
                    positions,
                    self.compress_ratio,
                    batch,
                )
                compressed_kv = self.compressor.forward(
                    x,
                    positions,
                    apply_norm=not compress_store_fuses_norm,
                )
                compressed_debug_rows = _compressed_debug_row_indices(
                    positions,
                    self.compress_ratio,
                    batch,
                )
                _capture_debug_activation(
                    f"layer{self.layer_id}.compressor_output",
                    compressed_kv,
                    row_indices=compressed_debug_rows,
                )
            if (
                use_dsv4_backend
                and not read_only_frozen_kv
                and self.compress_ratio == 128
                and hasattr(attn_backend, "write_c128_mtp_prefix_states")
            ):
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.c128_mtp_prefix"):
                    online_compressed_kv = attn_backend.write_c128_mtp_prefix_states(
                        self.layer_id,
                        self.compressor,
                        x,
                        batch,
                        apply_norm=not compress_store_fuses_norm,
                    )
                    if online_compressed_kv is not None:
                        compressed_kv = online_compressed_kv
                        _capture_debug_activation(
                            f"layer{self.layer_id}.compressor_output_online_c128",
                            compressed_kv,
                            row_indices=_compressed_debug_row_indices(
                                positions,
                                self.compress_ratio,
                                batch,
                            ),
                        )
            if (
                use_dsv4_backend
                and not read_only_frozen_kv
                and hasattr(attn_backend, "store_compressed")
            ):
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.compress_store"):
                    attn_backend.store_compressed(
                        self.layer_id,
                        compressed_kv,
                        batch,
                        self.compress_ratio,
                        norm_weight=(
                            self.compressor.norm.weight if compress_store_fuses_norm else None
                        ),
                        rms_norm_eps=self.rms_norm_eps if compress_store_fuses_norm else None,
                        rotary_dim=self.rope_head_dim,
                        base=float(self.rope_base),
                        original_seq_len=self.original_seq_len,
                        factor=self.rope_factor,
                        beta_fast=self.beta_fast,
                        beta_slow=self.beta_slow,
                    )

        if use_dsv4_backend and attn_backend is not None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.backend"):
                o = attn_backend.forward(
                    q,
                    kv,
                    kv,
                    self.layer_id,
                    batch,
                    compress_ratio=self.compress_ratio,
                    attn_sink=self.attn_sink,
                    swa_cache_written=kv_cache_written,
                )
        else:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.fallback_backend"):
                o = self._fallback_attention(q, kv, batch)
        _capture_debug_activation(f"layer{self.layer_id}.merged_attention_output_before_wo", o)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.o_rope"):
            dsv4_kernel.apply_rotary_tail(
                o,
                positions,
                rotary_dim=self.rope_head_dim,
                base=float(self.rope_base),
                inverse=True,
                original_seq_len=self.original_seq_len,
                factor=self.rope_factor,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
            )
            _capture_debug_activation(
                f"layer{self.layer_id}.merged_attention_output_after_inverse_rope",
                o,
            )
        d_per_group = self._wo_a_d_per_group()
        o = o.reshape(x.shape[0], self.num_local_groups, d_per_group)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.wo_a"):
            if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE):
                wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
                cached_wo_a = _cached_wo_a_bf16_bmm_weight(
                    self.wo_a,
                    self._wo_a_bf16_bmm_cache_name,
                    self.wo_a.weight,
                    wo_a_scale,
                    out_dtype=o.dtype,
                    num_local_groups=self.num_local_groups,
                    o_lora_rank=self.o_lora_rank,
                    d_per_group=d_per_group,
                    allow_build=False,
                    owner_label=self._wo_a_owner_label,
                )
                if dsv4_mtp_debug.wo_a_projection_oracle_enabled(self.layer_id):
                    actual_row_o = _wo_a_bf16_bmm_projection(
                        o,
                        cached_wo_a,
                        owner_label=self._wo_a_owner_label,
                    )
                    row_invariant_o = _wo_a_bf16_bmm_projection_row_invariant(
                        o,
                        cached_wo_a,
                        owner_label=self._wo_a_owner_label,
                    )
                    sglang_einsum_o = _wo_a_bf16_sglang_style_einsum_projection(
                        o,
                        cached_wo_a,
                        owner_label=self._wo_a_owner_label,
                    )
                    dsv4_mtp_debug.record_wo_a_projection_oracle(
                        batch,
                        layer_id=self.layer_id,
                        input_tensor=o,
                        cached_weight=cached_wo_a,
                        outputs={
                            "actual_row_bf16_bmm": actual_row_o,
                            "row_invariant_bf16_bmm": row_invariant_o,
                            "sglang_style_bf16_einsum": sglang_einsum_o,
                        },
                        selected_output="actual_row_bf16_bmm",
                        backend_path="mini.cached_bf16_bmm.actual_row",
                        extra={
                            "is_target_verify": bool(is_target_verify),
                            "num_local_groups": int(self.num_local_groups),
                            "d_per_group": int(d_per_group),
                            "o_lora_rank": int(self.o_lora_rank),
                        },
                    )
                    o = actual_row_o
                else:
                    o = _wo_a_bf16_bmm_projection(
                        o,
                        cached_wo_a,
                        owner_label=self._wo_a_owner_label,
                    )
            else:
                wo_a_scale = _cached_projection_scale(
                    self.wo_a,
                    "_dsv4_weight_scale_fp32_contiguous",
                    getattr(self.wo_a, "weight_scale_inv", None),
                )
                o = dsv4_kernel.wo_a_grouped_projection_fallback(
                    o,
                    self.wo_a.weight,
                    wo_a_scale,
                    num_local_groups=self.num_local_groups,
                    o_lora_rank=self.o_lora_rank,
                )
            _capture_debug_activation(f"layer{self.layer_id}.attention_wo_a_output", o)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.wo_b"):
            wo_b_local_name = f"layer{self.layer_id}.attention_wo_b_local_output_before_reduce"
            wo_b_reduce_name = f"layer{self.layer_id}.attention_wo_b_post_all_reduce_output"
            if dsv4_kernel.dense_fp8_marlin_projection_enabled():
                out = self.wo_b.forward_fp8_marlin_weight(
                    o,
                    cache_name=self._wo_b_marlin_weight_cache_name,
                    owner_label=self._wo_b_owner_label,
                    reduce=True,
                    reduce_label="dsv4.attn.wo_b.row_parallel_projection_all_reduce",
                    debug_pre_reduce_name=wo_b_local_name,
                    debug_post_reduce_name=wo_b_reduce_name,
                    row_invariant_reduce=False,
                    reduce_dtype=torch.float32,
                )
                _capture_debug_activation(f"layer{self.layer_id}.final_attention_output", out)
                return out
            if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_B_BF16_WEIGHT_CACHE_TOGGLE):
                if dsv4_mtp_debug.wo_b_projection_oracle_enabled(self.layer_id):
                    wo_b_scale = getattr(self.wo_b, "weight_scale_inv", None)
                    cached_wo_b = _cached_fp8_bf16_weight(
                        self.wo_b,
                        self._wo_b_bf16_weight_cache_name,
                        self.wo_b.weight,
                        wo_b_scale,
                        out_dtype=o.dtype,
                        allow_build=False,
                        owner_label=self._wo_b_owner_label,
                    )
                    actual_row_local = _fp8_cached_bf16_weight_local_projection(
                        o,
                        cached_wo_b,
                        owner=self.wo_b,
                        cache_name=self._wo_b_bf16_weight_cache_name,
                        owner_label=self._wo_b_owner_label,
                    )
                    actual_row_fp32_accum_local = (
                        _fp8_cached_bf16_weight_local_projection(
                            o,
                            cached_wo_b,
                            owner=self.wo_b,
                            cache_name=self._wo_b_bf16_weight_cache_name,
                            owner_label=self._wo_b_owner_label,
                            accum_dtype=torch.float32,
                        )
                    )
                    row_invariant_local = (
                        _fp8_cached_bf16_weight_local_projection_row_invariant(
                            o,
                            cached_wo_b,
                            owner=self.wo_b,
                            cache_name=self._wo_b_bf16_weight_cache_name,
                            owner_label=self._wo_b_owner_label,
                        )
                    )
                    dsv4_mtp_debug.record_wo_b_projection_oracle(
                        batch,
                        layer_id=self.layer_id,
                        input_tensor=o,
                        cached_weight=cached_wo_b,
                        outputs={
                            "actual_row_cached_bf16_linear": actual_row_local,
                            "actual_row_cached_fp32_accum_linear": (
                                actual_row_fp32_accum_local
                            ),
                            "row_invariant_cached_bf16_linear": row_invariant_local,
                        },
                        selected_output=(
                            "actual_row_cached_fp32_accum_linear"
                        ),
                        backend_path="mini.fp8_cached_bf16_weight.actual_row_fp32_accum",
                        extra={
                            "is_target_verify": bool(is_target_verify),
                            "local_accum_dtype": "torch.float32",
                            "reduce_label": "dsv4.attn.wo_b.row_parallel_projection_all_reduce",
                            "row_invariant_local": False,
                            "row_invariant_reduce": False,
                            "reduce_dtype": "torch.float32",
                        },
                    )
                out = self.wo_b.forward_fp8_cached_bf16_weight(
                    o,
                    cache_name=self._wo_b_bf16_weight_cache_name,
                    owner_label=self._wo_b_owner_label,
                    reduce=True,
                    reduce_label="dsv4.attn.wo_b.row_parallel_projection_all_reduce",
                    debug_pre_reduce_name=wo_b_local_name,
                    debug_post_reduce_name=wo_b_reduce_name,
                    row_invariant_local=False,
                    row_invariant_reduce=False,
                    local_accum_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                _capture_debug_activation(f"layer{self.layer_id}.final_attention_output", out)
                return out
            wo_b_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_WO_B_FP8_GEMM")
            out = self.wo_b.forward(
                o,
                fp8_gemm=wo_b_fp8_gemm if wo_b_fp8_gemm else None,
                debug_pre_reduce_name=wo_b_local_name,
                debug_post_reduce_name=wo_b_reduce_name,
                row_invariant_reduce=False,
                reduce_dtype=torch.float32,
            )
            _capture_debug_activation(f"layer{self.layer_id}.final_attention_output", out)
            return out


class DSV4TopK(BaseOP):
    def __init__(self, config: ModelConfig):
        self.tid2eid = torch.empty(
            config.vocab_size,
            config.num_experts_per_tok,
            dtype=torch.int64,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tid2eid[input_ids.long()]


class DSV4MoEGate(BaseOP):
    def __init__(self, config: ModelConfig, *, has_correction_bias: bool):
        self.weight = torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.bfloat16)
        if has_correction_bias:
            self.e_score_correction_bias = torch.empty(config.n_routed_experts, dtype=torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        topk: int,
        scoring_func: str,
        routed_scaling_factor: float,
        hash_topk: DSV4TopK | None = None,
        row_invariant_local: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = _cached_gate_fp32_weight(self, "_cached_gate_weight_fp32", self.weight)
        rows = hidden_states.numel() // hidden_states.shape[-1]
        if row_invariant_local and rows > 1:
            hidden_2d = hidden_states.reshape(rows, hidden_states.shape[-1])
            input_ids_1d = None
            if isinstance(input_ids, torch.Tensor):
                input_ids_1d = input_ids.reshape(-1)
            weight_chunks: list[torch.Tensor] = []
            index_chunks: list[torch.Tensor] = []
            for row in range(rows):
                row_input_ids = None
                if input_ids_1d is not None and row < int(input_ids_1d.numel()):
                    row_input_ids = input_ids_1d[row : row + 1]
                row_weights, row_indices = dsv4_kernel.moe_gate_fallback(
                    hidden_2d[row : row + 1],
                    weight,
                    input_ids=row_input_ids,
                    topk=topk,
                    scoring_func=scoring_func,
                    routed_scaling_factor=routed_scaling_factor,
                    correction_bias=getattr(self, "e_score_correction_bias", None),
                    hash_topk=hash_topk,
                )
                weight_chunks.append(row_weights)
                index_chunks.append(row_indices)
            return torch.cat(weight_chunks, dim=0), torch.cat(index_chunks, dim=0)
        return dsv4_kernel.moe_gate_fallback(
            hidden_states,
            weight,
            input_ids=input_ids,
            topk=topk,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=getattr(self, "e_score_correction_bias", None),
            hash_topk=hash_topk,
        )


class DSV4FusedRoutedExperts(BaseOP):
    def __init__(self, config: ModelConfig, *, layer_id: int | None = None):
        tp = get_tp_info()
        self.layer_id = layer_id
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        local_intermediate = div_even(config.moe_intermediate_size, tp.size)
        self.swiglu_limit = config.swiglu_limit or 0.0
        self.w13_weight = torch.empty(
            config.n_routed_experts,
            2,
            local_intermediate,
            config.hidden_size // 2,
            dtype=torch.int8,
        )
        self.w13_weight_scale_inv = torch.empty(
            config.n_routed_experts,
            2,
            local_intermediate,
            div_ceil(config.hidden_size, 32),
            dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.w2_weight = torch.empty(
            config.n_routed_experts,
            config.hidden_size,
            local_intermediate // 2,
            dtype=torch.int8,
        )
        self.w2_weight_scale_inv = torch.empty(
            config.n_routed_experts,
            config.hidden_size,
            div_ceil(local_intermediate, 32),
            dtype=dsv4_kernel.e8m0_dtype(),
        )
        self._moe_v2_workspace = dsv4_kernel.DSV4MoEWorkspace()
        self._marlin_wna16_weights = None
        self._marlin_wna16_released_original_expert_weights = False
        self._marlin_wna16_source_bytes = 0
        self._marlin_wna16_released_original_expert_bytes = 0
        self._marlin_wna16_hidden_original_expert_refs: list[torch.Tensor] = []
        self._marlin_wna16_integrity_forward_logs = 0

    @property
    def _marlin_owner_label(self) -> str:
        if self.layer_id is None:
            return "moe.routed_experts.marlin_wna16"
        return f"layer{self.layer_id}.moe.routed_experts.marlin_wna16"

    def _released_raw_weight_error(self, *, missing: list[str] | None = None) -> str:
        suffix = f" owner={self._marlin_owner_label}."
        if missing:
            suffix = f" owner={self._marlin_owner_label}; missing={missing}."
        return f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR}{suffix}"

    def _raw_expert_weight_names(self) -> tuple[str, str, str, str]:
        return (
            "w13_weight",
            "w13_weight_scale_inv",
            "w2_weight",
            "w2_weight_scale_inv",
        )

    def _missing_raw_expert_weights(self) -> list[str]:
        return [name for name in self._raw_expert_weight_names() if not hasattr(self, name)]

    def _raw_expert_weight_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = self._missing_raw_expert_weights()
        if missing:
            raise RuntimeError(self._released_raw_weight_error(missing=missing))
        return (
            self.w13_weight,
            self.w13_weight_scale_inv,
            self.w2_weight,
            self.w2_weight_scale_inv,
        )

    def _marlin_cache_tensors(self) -> dict[str, torch.Tensor | None]:
        cache = self._marlin_wna16_weights
        return {
            "w13": getattr(cache, "w13", None),
            "w13_scale": getattr(cache, "w13_scale", None),
            "w2": getattr(cache, "w2", None),
            "w2_scale": getattr(cache, "w2_scale", None),
        }

    def _cache_integrity_enabled(self) -> bool:
        return dsv4_memory_debug.env_flag(
            dsv4_memory_debug.DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG_ENV
        ) and _layer_selected_by_env(
            self.layer_id,
            _MARLIN_WNA16_CACHE_INTEGRITY_LAYERS_ENV,
        )

    def _audit_marlin_wna16_cache_integrity(self, stage: str) -> None:
        if not self._cache_integrity_enabled() or _cuda_graph_capture_active():
            return
        try:
            batch = get_global_ctx().batch
            batch_context = {
                "phase": batch.phase,
                "batch_size": int(batch.size),
                "padded_size": int(getattr(batch, "padded_size", batch.size)),
                "reqs": [
                    {
                        "uid": int(req.uid),
                        "cached_len": int(req.cached_len),
                        "device_len": int(req.device_len),
                        "extend_len": int(req.extend_len),
                    }
                    for req in batch.reqs
                ],
            }
        except Exception:
            batch_context = None
        dsv4_memory_debug.append_jsonl(
            "marlin_wna16_cache_integrity",
            {
                "event": "dsv4_marlin_wna16_cache_integrity",
                "stage": stage,
                "owner": self._marlin_owner_label,
                "layer_id": self.layer_id,
                "raw_missing": self._missing_raw_expert_weights(),
                "released_original": bool(self._marlin_wna16_released_original_expert_weights),
                "released_original_bytes": int(self._marlin_wna16_released_original_expert_bytes),
                "hidden_ref_count": len(self._marlin_wna16_hidden_original_expert_refs),
                "hidden_ref_bytes": int(
                    sum(
                        dsv4_memory_debug.tensor_nbytes(tensor)
                        for tensor in self._marlin_wna16_hidden_original_expert_refs
                    )
                ),
                "force_prepacked_raw_present": dsv4_kernel.dsv4_env_flag(
                    _MARLIN_WNA16_FORCE_PREPACKED_RAW_PRESENT_ENV
                ),
                "batch": batch_context,
                "cache_tensors": {
                    name: dsv4_memory_debug.tensor_integrity_summary(tensor)
                    for name, tensor in self._marlin_cache_tensors().items()
                },
            },
        )

    def _marlin_cache_report(
        self,
        *,
        source_bytes: int,
        released: list[dict[str, object]],
        already_present: bool,
        signature_match_before: bool | None,
        elapsed_ms: float,
    ) -> dict[str, object]:
        cache_tensors = self._marlin_cache_tensors()
        persistent_bytes = int(
            sum(dsv4_memory_debug.tensor_nbytes(tensor) for tensor in cache_tensors.values())
        )
        source_bytes = int(source_bytes)
        if source_bytes:
            self._marlin_wna16_source_bytes = source_bytes
        else:
            source_bytes = int(self._marlin_wna16_source_bytes)
        released_this_call_bytes = int(sum(int(item["bytes"]) for item in released))
        if released_this_call_bytes:
            self._marlin_wna16_released_original_expert_bytes = released_this_call_bytes
        return {
            "owner": self._marlin_owner_label,
            "layer_id": self.layer_id,
            "already_present": bool(already_present),
            "signature_match_before": signature_match_before,
            "elapsed_ms": elapsed_ms,
            "persistent_bytes": persistent_bytes,
            "source_bytes": source_bytes,
            "released_original_bytes": int(self._marlin_wna16_released_original_expert_bytes),
            "released_original_this_call_bytes": released_this_call_bytes,
            "released_original": bool(self._marlin_wna16_released_original_expert_weights),
            "raw_weights_available_after": not self._missing_raw_expert_weights(),
            "hidden_ref_count": len(self._marlin_wna16_hidden_original_expert_refs),
            "hidden_ref_bytes": int(
                sum(
                    dsv4_memory_debug.tensor_nbytes(tensor)
                    for tensor in self._marlin_wna16_hidden_original_expert_refs
                )
            ),
            "runtime_policy": (
                "marlin_wna16_prepacked_only"
                if self._marlin_wna16_released_original_expert_weights
                else "raw_weights_available"
            ),
            "fallback_error": (
                dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR
                if self._marlin_wna16_released_original_expert_weights
                else None
            ),
            "cache_tensors": {
                name: dsv4_memory_debug.tensor_summary(tensor)
                for name, tensor in cache_tensors.items()
            },
            "released": released,
        }

    def prepare_marlin_wna16_weight_cache(
        self,
        *,
        release_original: bool = False,
    ) -> dict[str, object]:
        from minisgl.kernel import marlin_wna16

        existing_cache = self._marlin_wna16_weights
        raw_available = not self._missing_raw_expert_weights()
        if not raw_available:
            if existing_cache is None:
                raise RuntimeError(self._released_raw_weight_error())
            return self._marlin_cache_report(
                source_bytes=0,
                released=[],
                already_present=True,
                signature_match_before=None,
                elapsed_ms=0.0,
            )

        w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
        source_tensors = {
            "w13_weight": w13_weight,
            "w13_weight_scale_inv": w13_scale,
            "w2_weight": w2_weight,
            "w2_weight_scale_inv": w2_scale,
        }
        source_bytes = int(
            sum(dsv4_memory_debug.tensor_nbytes(tensor) for tensor in source_tensors.values())
        )
        self._marlin_wna16_source_bytes = source_bytes
        signature_match_before = (
            existing_cache.matches(w13_weight, w13_scale, w2_weight, w2_scale)
            if existing_cache is not None
            else False
        )
        start_s = time.perf_counter()
        if existing_cache is None or not signature_match_before:
            self._marlin_wna16_weights = marlin_wna16.prepare_moe_mxfp4_weights(
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                params_dtype=torch.bfloat16,
                owner_label=self._marlin_owner_label,
                cache_was_present=existing_cache is not None,
                cache_signature_match=signature_match_before,
            )
            self._audit_marlin_wna16_cache_integrity("after_prepare_cache_build")
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0

        released: list[dict[str, object]] = []
        if release_original:
            return self.release_marlin_wna16_original_expert_weights(
                already_present=existing_cache is not None,
                signature_match_before=signature_match_before,
                elapsed_ms=elapsed_ms,
            )

        return self._marlin_cache_report(
            source_bytes=source_bytes,
            released=released,
            already_present=existing_cache is not None,
            signature_match_before=signature_match_before,
            elapsed_ms=elapsed_ms,
        )

    def release_marlin_wna16_original_expert_weights(
        self,
        *,
        already_present: bool = True,
        signature_match_before: bool | None = True,
        elapsed_ms: float = 0.0,
    ) -> dict[str, object]:
        if self._marlin_wna16_weights is None:
            raise RuntimeError(
                f"{self._marlin_owner_label} cannot release original expert weights "
                "before Marlin WNA16 cache is built."
            )
        if self._missing_raw_expert_weights():
            self._marlin_wna16_released_original_expert_weights = True
            return self._marlin_cache_report(
                source_bytes=0,
                released=[],
                already_present=already_present,
                signature_match_before=signature_match_before,
                elapsed_ms=elapsed_ms,
            )

        w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
        if not self._marlin_wna16_weights.matches(w13_weight, w13_scale, w2_weight, w2_scale):
            raise RuntimeError(
                f"{self._marlin_owner_label} cannot release original expert weights because "
                "the prebuilt Marlin WNA16 cache signature does not match the live source tensors."
            )
        source_tensors = {
            "w13_weight": w13_weight,
            "w13_weight_scale_inv": w13_scale,
            "w2_weight": w2_weight,
            "w2_weight_scale_inv": w2_scale,
        }
        source_bytes = int(
            sum(dsv4_memory_debug.tensor_nbytes(tensor) for tensor in source_tensors.values())
        )
        self._marlin_wna16_source_bytes = source_bytes
        if any(tensor.is_cuda for tensor in source_tensors.values()):
            # The release preset immediately makes source storage reusable for KV/cache
            # allocation, so make the post-load repack boundary explicit.
            torch.cuda.synchronize(w13_weight.device)
        self._audit_marlin_wna16_cache_integrity("before_release_original")
        released: list[dict[str, object]] = []
        release_names = self._marlin_wna16_release_attribute_names()
        if dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_KEEP_HIDDEN_REF_ENV):
            self._marlin_wna16_hidden_original_expert_refs.extend(
                tensor for name, tensor in source_tensors.items() if name in release_names
            )
            self._maybe_poison_hidden_original_expert_refs(source_tensors, release_names)
        for name, tensor in list(source_tensors.items()):
            freed_record = dsv4_memory_debug.register_marlin_wna16_freed_tensor(
                tensor=tensor,
                layer_id=self.layer_id,
                component=name,
                owner=self._marlin_owner_label,
                released=name in release_names,
                stage="before_release_original",
                extra={
                    "weights_only": dsv4_kernel.dsv4_env_flag(
                        _MARLIN_WNA16_RELEASE_WEIGHTS_ONLY_ENV
                    ),
                    "scales_only": dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_RELEASE_SCALES_ONLY_ENV),
                    "keep_hidden_ref": dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_KEEP_HIDDEN_REF_ENV),
                },
            )
            if name not in release_names:
                continue
            released.append(
                {
                    "attribute": name,
                    "component": name,
                    "layer_id": self.layer_id,
                    "data_ptr": int(freed_record.get("data_ptr", 0) or 0),
                    "start": int(freed_record.get("start", 0) or 0),
                    "end": int(freed_record.get("end", 0) or 0),
                    "shape": list(freed_record.get("shape", [])),
                    "stride": list(freed_record.get("stride", [])),
                    "dtype": str(freed_record.get("dtype")),
                    "bytes": int(freed_record.get("bytes", 0) or 0),
                    "released": True,
                }
            )
            delattr(self, name)
        self._marlin_wna16_released_original_expert_weights = True
        self._marlin_wna16_released_original_expert_bytes = int(
            sum(int(item["bytes"]) for item in released)
        )
        self._audit_marlin_wna16_cache_integrity("after_release_original")
        return self._marlin_cache_report(
            source_bytes=source_bytes,
            released=released,
            already_present=already_present,
            signature_match_before=signature_match_before,
            elapsed_ms=elapsed_ms,
        )

    def _marlin_wna16_release_attribute_names(self) -> set[str]:
        weights_only = dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_RELEASE_WEIGHTS_ONLY_ENV)
        scales_only = dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_RELEASE_SCALES_ONLY_ENV)
        if weights_only and not scales_only:
            return {"w13_weight", "w2_weight"}
        if scales_only and not weights_only:
            return {"w13_weight_scale_inv", "w2_weight_scale_inv"}
        return set(self._raw_expert_weight_names())

    def _maybe_poison_hidden_original_expert_refs(
        self,
        source_tensors: dict[str, torch.Tensor],
        release_names: set[str],
    ) -> None:
        pattern = os.environ.get(_MARLIN_WNA16_POISON_HIDDEN_REF_PATTERN_ENV, "").strip().lower()
        if not pattern or pattern in {"none", "off", "0", "false", "no"}:
            return
        for name, tensor in source_tensors.items():
            if name not in release_names:
                continue
            with torch.no_grad():
                if pattern in {"zero", "zeros"}:
                    tensor.fill_(0)
                elif pattern in {"one", "ones"}:
                    tensor.fill_(1)
                elif pattern in {"neg1", "negative_one", "-1"}:
                    tensor.fill_(-1)
                elif pattern in {"nan", "nans"} and torch.is_floating_point(tensor):
                    tensor.fill_(float("nan"))
                elif pattern in {"nan", "nans"}:
                    tensor.fill_(127)
                elif pattern.startswith("value:"):
                    value = int(pattern.split(":", 1)[1], 0)
                    tensor.fill_(value)
                else:
                    tensor.fill_(127)
            dsv4_memory_debug.append_jsonl(
                "marlin_wna16_poison",
                {
                    "event": "dsv4_marlin_wna16_hidden_ref_poison",
                    "owner": self._marlin_owner_label,
                    "layer_id": self.layer_id,
                    "component": name,
                    "pattern": pattern,
                    "tensor": dsv4_memory_debug.tensor_summary(tensor),
                },
            )

    def _record_moe_owner_tensors(
        self,
        *,
        stage: str,
        hidden_states: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        grouped: torch.Tensor | None = None,
        moe_plan: dsv4_kernel.DSV4MoEExecutionPlan | None = None,
    ) -> None:
        if not dsv4_memory_debug.marlin_wna16_release_ledger_enabled():
            return
        tensors: dict[str, torch.Tensor | None] = {
            "hidden_states": hidden_states,
            "route_weights": weights,
            "route_indices": indices,
            "grouped_output": grouped,
        }
        if moe_plan is not None:
            route_plan = moe_plan.route_plan
            tensors.update(
                {
                    "moe_plan.route_weights": moe_plan.route_weights,
                    "moe_plan.sorted_route_ids": route_plan.sorted_route_ids,
                    "moe_plan.expert_ids": route_plan.expert_ids,
                    "moe_plan.num_tokens_post_padded": route_plan.num_tokens_post_padded,
                }
            )
        workspace_buffers = getattr(self._moe_v2_workspace, "_buffers", {})
        for name, tensor in workspace_buffers.items():
            tensors[f"workspace.{name}"] = tensor
        dsv4_memory_debug.record_owner_tensors(
            owner_prefix=f"{self._marlin_owner_label}.moe_forward",
            stage=stage,
            tensors=tensors,
            extra={"layer_id": self.layer_id},
        )

    def _expert_forward(
        self, local_idx: int, x: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        w1 = dsv4_kernel.quantized_linear_ref(
            x,
            self.w13_weight[local_idx, 0],
            self.w13_weight_scale_inv[local_idx, 0],
            weight_kind="fp4",
        ).float()
        w3 = dsv4_kernel.quantized_linear_ref(
            x,
            self.w13_weight[local_idx, 1],
            self.w13_weight_scale_inv[local_idx, 1],
            weight_kind="fp4",
        ).float()
        hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            w1,
            w3,
            swiglu_limit=self.swiglu_limit,
            weights=weights,
        )
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.expert_hidden_to_input_dtype.expert{local_idx}",
            hidden=hidden,
        ):
            hidden_for_w2 = hidden.to(x.dtype)
        return dsv4_kernel.quantized_linear_ref(
            hidden_for_w2,
            self.w2_weight[local_idx],
            self.w2_weight_scale_inv[local_idx],
            weight_kind="fp4",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *,
        reduce: bool = True,
        moe_plan: dsv4_kernel.DSV4MoEExecutionPlan | None = None,
    ) -> torch.Tensor:
        backend = dsv4_kernel.require_supported_moe_expert_backend()
        self._record_moe_owner_tensors(
            stage=f"layer{self.layer_id}.moe_forward.input",
            hidden_states=hidden_states,
            weights=weights,
            indices=indices,
            moe_plan=moe_plan,
        )
        missing_raw_weights = self._missing_raw_expert_weights()
        if missing_raw_weights and backend != dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16:
            raise RuntimeError(
                f"{self._released_raw_weight_error(missing=missing_raw_weights)} "
                f"requested_backend={backend!r}."
            )
        if backend == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16:
            force_prepacked = dsv4_kernel.dsv4_env_flag(
                _MARLIN_WNA16_FORCE_PREPACKED_RAW_PRESENT_ENV
            )
            if missing_raw_weights or force_prepacked:
                if self._marlin_wna16_weights is None:
                    raise RuntimeError(
                        f"{self._released_raw_weight_error(missing=missing_raw_weights)} "
                        "Marlin WNA16 prebuilt cache is missing."
                    )
                max_forward_logs = _env_int(
                    _MARLIN_WNA16_CACHE_INTEGRITY_MAX_FORWARD_LOGS_ENV,
                    2,
                )
                if self._marlin_wna16_integrity_forward_logs < max_forward_logs:
                    self._audit_marlin_wna16_cache_integrity("before_forward_prepacked")
                grouped = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16_prepacked(
                    hidden_states,
                    weights,
                    indices,
                    self._marlin_wna16_weights,
                    swiglu_limit=self.swiglu_limit,
                )
                if self._marlin_wna16_integrity_forward_logs < max_forward_logs:
                    self._audit_marlin_wna16_cache_integrity("after_forward_prepacked")
                    self._marlin_wna16_integrity_forward_logs += 1
            else:
                w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
                grouped, self._marlin_wna16_weights = (
                    dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16(
                        hidden_states,
                        weights,
                        indices,
                        w13_weight,
                        w13_scale,
                        w2_weight,
                        w2_scale,
                        swiglu_limit=self.swiglu_limit,
                        cache=self._marlin_wna16_weights,
                        owner_label=self._marlin_owner_label,
                    )
                )
            self._record_moe_owner_tensors(
                stage=f"layer{self.layer_id}.moe_forward.marlin_output",
                grouped=grouped,
                moe_plan=moe_plan,
            )
            if reduce and self._tp_size > 1:
                with dsv4_direct_copy_nvtx(
                    "moe_shared_expert_staging.routed_grouped_to_fp32_for_reduce",
                    grouped=grouped,
                ):
                    grouped_for_reduce = grouped.float()
                grouped_reduced = self._comm.all_reduce(
                    grouped_for_reduce,
                    label="dsv4.routed_expert_all_reduce",
                )
                with dsv4_direct_copy_nvtx(
                    "moe_shared_expert_staging.routed_reduce_to_grouped_dtype",
                    grouped=grouped_reduced,
                ):
                    grouped = grouped_reduced.to(grouped.dtype)
            return grouped

        workspace = None
        if (
            moe_plan is not None
            and moe_plan.route_plan.route_count <= dsv4_kernel.DSV4_SM80_MOE_V2_WORKSPACE_MAX_ROUTES
        ):
            workspace = self._moe_v2_workspace
        w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
        grouped = dsv4_kernel.moe_route_dispatch_bf16_grouped(
            hidden_states,
            weights,
            indices,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            swiglu_limit=self.swiglu_limit,
            moe_plan=moe_plan,
            workspace=workspace,
        )
        self._record_moe_owner_tensors(
            stage=f"layer{self.layer_id}.moe_forward.grouped_output",
            grouped=grouped,
            moe_plan=moe_plan,
        )
        if grouped is not None:
            if reduce and self._tp_size > 1:
                with dsv4_direct_copy_nvtx(
                    "moe_shared_expert_staging.routed_grouped_to_fp32_for_reduce",
                    grouped=grouped,
                ):
                    grouped_for_reduce = grouped.float()
                grouped_reduced = self._comm.all_reduce(
                    grouped_for_reduce,
                    label="dsv4.routed_expert_all_reduce",
                )
                with dsv4_direct_copy_nvtx(
                    "moe_shared_expert_staging.routed_reduce_to_grouped_dtype",
                    grouped=grouped_reduced,
                ):
                    grouped = grouped_reduced.to(grouped.dtype)
            return grouped

        y = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx in range(self.w13_weight.shape[0]):
            token_idx, top_idx = torch.where(indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            y[token_idx] += self._expert_forward(
                int(expert_idx),
                hidden_states[token_idx],
                weights[token_idx, top_idx, None],
            ).float()
        if reduce and self._tp_size > 1:
            y = self._comm.all_reduce(y, label="dsv4.routed_expert_all_reduce")
        with dsv4_direct_copy_nvtx(
            "moe_shared_expert_staging.routed_fallback_to_hidden_dtype",
            y=y,
        ):
            return y.to(hidden_states.dtype)


class DSV4SharedExperts(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        self.layer_id = layer_id
        intermediate = config.moe_intermediate_size * max(config.n_shared_experts, 1)
        self.swiglu_limit = config.swiglu_limit or 0.0
        self.gate_up_proj = DSV4Linear(
            config.hidden_size,
            2 * intermediate,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            col_parallel=True,
        )
        self.down_proj = DSV4Linear(
            intermediate,
            config.hidden_size,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            row_parallel=True,
        )

    @property
    def _gate_up_bf16_weight_cache_name(self) -> str:
        return "_dsv4_shared_gate_up_bf16_weight_cache"

    @property
    def _down_bf16_weight_cache_name(self) -> str:
        return "_dsv4_shared_down_bf16_weight_cache"

    @property
    def _down_marlin_weight_cache_name(self) -> str:
        return "_dsv4_shared_down_dense_fp8_marlin_weight_cache"

    @property
    def _gate_up_owner_label(self) -> str:
        if self.layer_id is None:
            return "shared_experts.gate_up_proj"
        return f"layer{self.layer_id}.shared_experts.gate_up_proj"

    @property
    def _down_owner_label(self) -> str:
        if self.layer_id is None:
            return "shared_experts.down_proj"
        return f"layer{self.layer_id}.shared_experts.down_proj"

    def prepare_bf16_weight_cache(self) -> list[dict[str, object]]:
        if not dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
        ):
            return []
        reports = [
            self.gate_up_proj.prepare_fp8_bf16_weight_cache(
                self._gate_up_bf16_weight_cache_name,
                owner_label=self._gate_up_owner_label,
            ),
        ]
        if not dsv4_kernel.dense_fp8_marlin_projection_enabled():
            reports.append(
                self.down_proj.prepare_fp8_bf16_weight_cache(
                    self._down_bf16_weight_cache_name,
                    owner_label=self._down_owner_label,
                )
            )
        return reports

    def prepare_down_marlin_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dense_fp8_marlin_projection_enabled():
            return None
        return self.down_proj.prepare_fp8_marlin_weight_cache(
            self._down_marlin_weight_cache_name,
            owner_label=self._down_owner_label,
        )

    def prepare_down_bf16_weight_cache(self) -> dict[str, object]:
        return self.down_proj.prepare_fp8_bf16_weight_cache(
            self._down_bf16_weight_cache_name,
            owner_label=self._down_owner_label,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        reduce: bool = True,
        row_invariant_local: bool = False,
    ) -> torch.Tensor:
        rows = hidden_states.numel() // hidden_states.shape[-1]
        if row_invariant_local and rows > 1:
            hidden_2d = hidden_states.reshape(rows, hidden_states.shape[-1])
            chunks = [
                self.forward(
                    hidden_2d[row : row + 1],
                    reduce=reduce,
                    row_invariant_local=False,
                )
                for row in range(rows)
            ]
            return torch.cat(chunks, dim=0).reshape(*hidden_states.shape[:-1], -1)
        fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_SHARED_FP8_GEMM")
        use_bf16_weight_cache = dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
        )
        with _dsv4_capture_nvtx("shared_experts.gate_up_proj"):
            if use_bf16_weight_cache:
                gate_up = self.gate_up_proj.forward_fp8_cached_bf16_weight(
                    hidden_states,
                    cache_name=self._gate_up_bf16_weight_cache_name,
                    owner_label=self._gate_up_owner_label,
                )
            else:
                gate_up = self.gate_up_proj.forward(
                    hidden_states,
                    fp8_gemm=fp8_gemm if fp8_gemm else None,
                )
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            gate,
            up,
            swiglu_limit=self.swiglu_limit,
        )
        with _dsv4_capture_nvtx("shared_experts.down_proj"):
            with dsv4_direct_copy_nvtx(
                "moe_shared_expert_staging.shared_hidden_to_up_dtype",
                hidden=hidden,
            ):
                hidden_for_down = hidden.to(up.dtype)
            if dsv4_kernel.dense_fp8_marlin_projection_enabled():
                return self.down_proj.forward_fp8_marlin_weight(
                    hidden_for_down,
                    cache_name=self._down_marlin_weight_cache_name,
                    owner_label=self._down_owner_label,
                    reduce=reduce,
                    reduce_label="dsv4.shared_expert_all_reduce",
                )
            if use_bf16_weight_cache:
                return self.down_proj.forward_fp8_cached_bf16_weight(
                    hidden_for_down,
                    cache_name=self._down_bf16_weight_cache_name,
                    owner_label=self._down_owner_label,
                    reduce=reduce,
                    reduce_label="dsv4.shared_expert_all_reduce",
                )
            return self.down_proj.forward(
                hidden_for_down,
                reduce=reduce,
                reduce_label="dsv4.shared_expert_all_reduce",
                fp8_gemm=fp8_gemm if fp8_gemm else None,
            )


def _dsv4_moe_reduce_once_input(
    output: torch.Tensor,
    *,
    hidden_dtype: torch.dtype,
    layer_id: int,
    path: str,
) -> torch.Tensor:
    if (
        hidden_dtype == torch.bfloat16
        and output.dtype != torch.bfloat16
        and dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE)
    ):
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.{path}_to_bf16_for_reduce.layer{layer_id}",
            output=output,
        ):
            return output.to(torch.bfloat16)
    return output


@dataclass(frozen=True)
class DSV4FusedMoERunnerPrepareResult:
    weights: torch.Tensor
    indices: torch.Tensor
    moe_plan: dsv4_kernel.DSV4MoEExecutionPlan


class DSV4FusedMoERunner:
    """Mini-owned exact-path runner shaped after vLLM's standard FusedMoE runner."""

    def __init__(
        self,
        *,
        layer_id: int,
        gate: DSV4MoEGate,
        experts: DSV4FusedRoutedExperts,
        shared_experts: DSV4SharedExperts | None,
        topk_count: int,
        scoring_func: str,
        routed_scaling_factor: float,
        tp_size: int,
    ) -> None:
        self.layer_id = layer_id
        self.gate = gate
        self.experts = experts
        self.shared_experts = shared_experts
        self.topk_count = topk_count
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self._tp_size = tp_size

    def route(
        self,
        flat: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        hash_topk: DSV4TopK | None,
        row_invariant_local: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate.forward(
            flat,
            input_ids=input_ids,
            topk=self.topk_count,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            hash_topk=hash_topk,
            row_invariant_local=row_invariant_local,
        )

    def prepare(
        self,
        flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> DSV4FusedMoERunnerPrepareResult:
        if hasattr(self.experts, "w13_weight"):
            num_experts = self.experts.w13_weight.shape[0]
        elif self.experts._marlin_wna16_weights is not None:
            num_experts = self.experts._marlin_wna16_weights.w13.shape[0]
        else:
            raise RuntimeError(
                f"layer{self.layer_id}.moe runner cannot build a route plan because "
                f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR} "
                "Marlin WNA16 cache is missing."
            )
        moe_plan = dsv4_kernel.build_moe_v2_execution_plan(
            flat,
            weights,
            indices,
            num_experts=num_experts,
            block_size_m=16,
            reduce_once=True,
        )
        return DSV4FusedMoERunnerPrepareResult(
            weights=weights,
            indices=indices,
            moe_plan=moe_plan,
        )

    def apply_experts(
        self,
        flat: torch.Tensor,
        prepared: DSV4FusedMoERunnerPrepareResult,
    ) -> torch.Tensor:
        return self.experts.forward(
            flat,
            prepared.weights,
            prepared.indices,
            reduce=False,
            moe_plan=prepared.moe_plan,
        )

    def apply_experts_row_invariant(
        self,
        flat: torch.Tensor,
        prepared: DSV4FusedMoERunnerPrepareResult,
    ) -> torch.Tensor:
        rows = flat.numel() // flat.shape[-1]
        if rows <= 1:
            return self.apply_experts(flat, prepared)
        flat_2d = flat.reshape(rows, flat.shape[-1])
        weights_2d = prepared.weights.reshape(rows, prepared.weights.shape[-1])
        indices_2d = prepared.indices.reshape(rows, prepared.indices.shape[-1])
        chunks: list[torch.Tensor] = []
        for row in range(rows):
            row_flat = flat_2d[row : row + 1].contiguous()
            row_weights = weights_2d[row : row + 1].contiguous()
            row_indices = indices_2d[row : row + 1].contiguous()
            row_prepared = self.prepare(row_flat, row_weights, row_indices)
            chunks.append(self.apply_experts(row_flat, row_prepared))
        return torch.cat(chunks, dim=0).reshape_as(flat)

    def finalize_routed(self, routed_output: torch.Tensor) -> torch.Tensor:
        # The current grouped FP4 backend already applies top-k weights and
        # sums routes to [tokens, hidden]. Keep the boundary explicit so a
        # future exact backend can return per-route output here.
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.runner_finalize_to_fp32.layer{self.layer_id}",
            routed_output=routed_output,
        ):
            return routed_output.float()

    def apply_shared_raw(
        self,
        flat: torch.Tensor,
        *,
        row_invariant_local: bool = False,
    ) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        return self.shared_experts.forward(
            flat,
            reduce=False,
            row_invariant_local=row_invariant_local,
        )

    def finalize_shared(self, shared_output: torch.Tensor) -> torch.Tensor:
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.runner_shared_to_fp32.layer{self.layer_id}",
            shared=shared_output,
        ):
            return shared_output.float()

    def apply_shared(self, flat: torch.Tensor) -> torch.Tensor | None:
        shared = self.apply_shared_raw(flat)
        if shared is None:
            return None
        return self.finalize_shared(shared)

    def maybe_reduce_final(
        self,
        output: torch.Tensor,
        *,
        comm: DistributedCommunicator,
        hidden_dtype: torch.dtype,
        reduce_label: str,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        boundary_extra = _dsv4_moe_reduce_boundary_base_extra(
            stage="runner.post_all_reduce",
            tp_size=self._tp_size,
            comm=comm,
            reduce_label=reduce_label,
            pre_reduce=output,
        )
        if self._tp_size > 1:
            output = _dsv4_moe_reduce_once_input(
                output,
                hidden_dtype=hidden_dtype,
                layer_id=self.layer_id,
                path="runner_output",
            )
            boundary_extra["communication_input"] = _dsv4_moe_reduce_tensor_census_if_enabled(output)
            boundary_extra["communication_input_dtype"] = str(output.dtype)
            output = comm.all_reduce(output, label=reduce_label)
            boundary_extra["post_reduce"] = _dsv4_moe_reduce_tensor_census_if_enabled(output)
            boundary_extra["communication_output_dtype"] = str(output.dtype)
            return output, boundary_extra
        boundary_extra["communication_input"] = _dsv4_moe_reduce_tensor_census_if_enabled(output)
        boundary_extra["communication_input_dtype"] = str(output.dtype)
        boundary_extra["post_reduce"] = _dsv4_moe_reduce_tensor_census_if_enabled(output)
        boundary_extra["communication_output_dtype"] = str(output.dtype)
        return output, boundary_extra

    def _contract_active_source_rows(self, batch: Batch | None, rows: int) -> list[int]:
        metadata = getattr(batch, "dsv4_target_verify_metadata", None)
        if not isinstance(metadata, dict):
            return list(range(int(rows)))
        mask = metadata.get("active_row_mask")
        if isinstance(mask, torch.Tensor):
            values = [bool(x) for x in mask.detach().reshape(-1).cpu().tolist()]
        elif isinstance(mask, (list, tuple)):
            values = [bool(x) for x in mask]
        else:
            values = []
        if len(values) < int(rows):
            return list(range(int(rows)))
        return [idx for idx, active in enumerate(values[: int(rows)]) if active]

    def _contract_chunk_size(self, batch: Batch | None, rows: int) -> int:
        raw = os.environ.get("MINISGL_DSV4_MTP_MOE_CONTRACT_ORACLE_MICROBATCH", "").strip()
        if raw:
            try:
                value = int(raw)
                if value > 0:
                    return min(value, int(rows))
            except ValueError:
                pass
        size = int(getattr(batch, "size", 0) or len(getattr(batch, "reqs", [])) or 0)
        if size <= 0:
            size = 1
        return min(size, int(rows))

    @staticmethod
    def _contract_cat_optional(chunks: list[torch.Tensor | None]) -> torch.Tensor | None:
        tensors = [chunk for chunk in chunks if isinstance(chunk, torch.Tensor)]
        if not tensors:
            return None
        if len(tensors) != len(chunks):
            return None
        return torch.cat(tensors, dim=0)

    def _contract_run_once(
        self,
        flat: torch.Tensor,
        flat_input_ids: torch.Tensor,
        *,
        comm: DistributedCommunicator,
        hash_topk: DSV4TopK | None,
        row_invariant_local: bool,
    ) -> dict[str, torch.Tensor | None]:
        weights, indices = self.route(
            flat,
            flat_input_ids,
            hash_topk=hash_topk,
            row_invariant_local=row_invariant_local,
        )
        prepared = self.prepare(flat, weights, indices)
        if row_invariant_local:
            routed_raw = self.apply_experts_row_invariant(flat, prepared)
        else:
            routed_raw = self.apply_experts(flat, prepared)
        routed = self.finalize_routed(routed_raw)
        shared_raw = self.apply_shared_raw(
            flat,
            row_invariant_local=row_invariant_local,
        )
        shared = self.finalize_shared(shared_raw) if shared_raw is not None else None
        aggregate = routed + shared if shared is not None else routed
        reduced, _reduce_extra = self.maybe_reduce_final(
            aggregate,
            comm=comm,
            hidden_dtype=flat.dtype,
            reduce_label=prepared.moe_plan.final_reduce_label,
        )
        output = reduced.to(flat.dtype)
        return {
            "topk_ids": indices,
            "topk_weights": weights,
            "routed_expert_output_raw": routed_raw,
            "routed_expert_output": routed,
            "shared_expert_output_raw": shared_raw,
            "shared_expert_output": shared,
            "expert_aggregate_before_reduce": aggregate,
            "expert_reduce_output": reduced,
            "moe_output": output,
        }

    def _contract_run_variant(
        self,
        flat: torch.Tensor,
        flat_input_ids: torch.Tensor,
        *,
        comm: DistributedCommunicator,
        hash_topk: DSV4TopK | None,
        source_rows: list[int],
        row_invariant_local: bool,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        rows = int(flat.shape[0])
        if chunk_size is None or int(chunk_size) <= 0 or int(chunk_size) >= rows:
            return self._contract_run_once(
                flat,
                flat_input_ids,
                comm=comm,
                hash_topk=hash_topk,
                row_invariant_local=row_invariant_local,
            )
        by_name: dict[str, list[torch.Tensor | None]] = {}
        for start in range(0, rows, int(chunk_size)):
            end = min(start + int(chunk_size), rows)
            chunk = self._contract_run_once(
                flat[start:end].contiguous(),
                flat_input_ids[start:end].contiguous(),
                comm=comm,
                hash_topk=hash_topk,
                row_invariant_local=row_invariant_local,
            )
            for name, tensor in chunk.items():
                by_name.setdefault(name, []).append(tensor)
        return {
            name: self._contract_cat_optional(chunks)
            for name, chunks in by_name.items()
        }

    def _record_contract_variant(
        self,
        batch: Batch | None,
        *,
        variant: str,
        source_rows: list[int],
        tensors: dict[str, torch.Tensor | None],
        positions: torch.Tensor | None,
        reference_tensor: torch.Tensor,
        params: dict[str, object],
        extra: dict[str, object],
    ) -> None:
        dsv4_mtp_debug.record_moe_contract_oracle(
            batch,
            layer_id=self.layer_id,
            variant=variant,
            source_rows=source_rows,
            tensors=tensors,
            positions=positions,
            params=params,
            extra=extra,
            reference_tensor=reference_tensor,
        )

    def _maybe_record_contract_oracle(
        self,
        *,
        batch: Batch | None,
        hidden_states: torch.Tensor,
        flat: torch.Tensor,
        flat_input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        comm: DistributedCommunicator,
        hash_topk: DSV4TopK | None,
        common_params: dict[str, object],
    ) -> None:
        if not dsv4_mtp_debug.moe_contract_oracle_enabled(self.layer_id):
            return
        if not _dsv4_is_target_verify_batch(batch):
            return
        if flat.ndim != 2 or flat.shape[0] <= 0:
            return
        try:
            if flat.is_cuda and torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            return
        rows = int(flat.shape[0])
        source_rows = list(range(rows))
        active_rows = self._contract_active_source_rows(batch, rows)
        if not active_rows:
            active_rows = source_rows
        microbatch = self._contract_chunk_size(batch, rows)
        variants: list[tuple[str, list[int], torch.Tensor, torch.Tensor, bool, int | None, dict[str, object]]] = [
            (
                "target_full_batch",
                source_rows,
                flat,
                flat_input_ids,
                False,
                None,
                {
                    "contract": "full target verify batch",
                    "runtime_warning": "debug-only oracle; replays MoE and all-reduce",
                },
            ),
            (
                "target_active_only",
                active_rows,
                flat[active_rows].contiguous(),
                flat_input_ids[active_rows].contiguous(),
                False,
                None,
                {
                    "contract": "active-only target rows",
                    "active_rows": active_rows,
                    "runtime_warning": "debug-only oracle; replays MoE and all-reduce",
                },
            ),
            (
                "target_row_by_row_reference",
                source_rows,
                flat,
                flat_input_ids,
                False,
                1,
                {
                    "contract": "row-by-row reference",
                    "runtime_warning": "slow correctness oracle; one MoE/reduce per row",
                },
            ),
            (
                "target_normal_shape_microbatch",
                source_rows,
                flat,
                flat_input_ids,
                False,
                microbatch,
                {
                    "contract": "normal-shape-compatible microbatch",
                    "microbatch_rows": int(microbatch),
                    "runtime_warning": "debug-only oracle; one MoE/reduce per microbatch",
                },
            ),
            (
                "target_current_row_invariant_replay",
                source_rows,
                flat,
                flat_input_ids,
                True,
                None,
                {
                    "contract": "Mini current target row-invariant local replay",
                    "runtime_warning": "debug-only replay of current local contract",
                },
            ),
        ]
        with torch.no_grad():
            for (
                variant,
                variant_source_rows,
                variant_flat,
                variant_input_ids,
                row_invariant_local,
                chunk_size,
                extra,
            ) in variants:
                tensors = self._contract_run_variant(
                    variant_flat,
                    variant_input_ids,
                    comm=comm,
                    hash_topk=hash_topk,
                    source_rows=variant_source_rows,
                    row_invariant_local=row_invariant_local,
                    chunk_size=chunk_size,
                )
                self._record_contract_variant(
                    batch,
                    variant=variant,
                    source_rows=variant_source_rows,
                    tensors=tensors,
                    positions=positions,
                    reference_tensor=hidden_states,
                    params={
                        **common_params,
                        "contract_oracle": True,
                        "row_invariant_local": bool(row_invariant_local),
                        "chunk_size": None if chunk_size is None else int(chunk_size),
                    },
                    extra=extra,
                )

    def _forward_target_verify_microbatch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        flat: torch.Tensor,
        flat_input_ids: torch.Tensor,
        batch: Batch,
        positions: torch.Tensor | None,
        comm: DistributedCommunicator,
        hash_topk: DSV4TopK | None,
        common_params: dict[str, object],
        contract: _DSV4TargetVerifyMoEMicrobatchContract,
    ) -> torch.Tensor:
        runtime_params = {
            **common_params,
            "target_verify_row_invariant_local": False,
            "target_verify_moe_microbatch_runtime": True,
            "target_verify_moe_microbatch_contract": contract.as_record(),
        }
        timing_ms: float | None = None
        timing_enabled = _dsv4_target_verify_moe_microbatch_timing_enabled()
        start_event = None
        end_event = None
        start_s = 0.0
        if timing_enabled and flat.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        elif timing_enabled:
            start_s = time.perf_counter()

        with _dsv4_capture_nvtx(
            f"layer{self.layer_id}.mlp.runner.target_verify_moe_microbatch"
        ):
            tensors = self._contract_run_variant(
                flat,
                flat_input_ids,
                comm=comm,
                hash_topk=hash_topk,
                source_rows=list(range(int(flat.shape[0]))),
                row_invariant_local=False,
                chunk_size=contract.chunk_rows,
            )

        if timing_enabled and start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            timing_ms = float(start_event.elapsed_time(end_event))
        elif timing_enabled:
            timing_ms = float((time.perf_counter() - start_s) * 1000.0)

        output = tensors.get("moe_output")
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "DeepSeek V4 target-verify MoE microbatch did not produce moe_output."
            )
        output = output.view_as(hidden_states)
        weights = tensors.get("topk_weights")
        indices = tensors.get("topk_ids")
        routed_raw = tensors.get("routed_expert_output_raw")
        routed = tensors.get("routed_expert_output")
        shared_raw = tensors.get("shared_expert_output_raw")
        shared = tensors.get("shared_expert_output")
        aggregate = tensors.get("expert_aggregate_before_reduce")
        reduced = tensors.get("expert_reduce_output")

        route_extra = {
            "stage": "runner.target_verify_moe_microbatch",
            "microbatch_contract": contract.as_record(),
            "runtime_warning": "target-verify MoE executes one route/expert/reduce path per microbatch",
        }
        if isinstance(weights, torch.Tensor) and isinstance(indices, torch.Tensor):
            route_extra.update(
                _dsv4_moe_route_extra(
                    weights=weights,
                    indices=indices,
                    input_ids=flat_input_ids,
                    reduce_once=True,
                    hash_topk=hash_topk is not None,
                    stage="runner.target_verify_moe_microbatch",
                )
            )
            _dsv4_moe_record_operator(
                batch,
                operator_name="topk_ids",
                layer_id=self.layer_id,
                input_tensor=flat,
                output_tensor=indices,
                positions=positions,
                path="mini.moe.runner.target_verify_microbatch.topk_ids",
                params=runtime_params,
                extra=route_extra,
            )
            _dsv4_moe_record_operator(
                batch,
                operator_name="topk_weights",
                layer_id=self.layer_id,
                input_tensor=flat,
                output_tensor=weights,
                positions=positions,
                path="mini.moe.runner.target_verify_microbatch.topk_weights",
                params=runtime_params,
                extra=route_extra,
            )
        router_logits = _dsv4_moe_router_logits_debug(self.gate, flat)
        _dsv4_moe_record_operator(
            batch,
            operator_name="router_logits",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=router_logits,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.router_logits.linear_bf16_fp32_debug",
            params=runtime_params,
            extra=route_extra,
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_input",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=flat,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.routed_expert_input",
            params=runtime_params,
            extra=route_extra,
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_output_raw",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=routed_raw,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.routed_experts.forward_raw",
            params=runtime_params,
            extra={
                **route_extra,
                "stage": "runner.target_verify_microbatch.routed_output_raw",
            },
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_output",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=routed,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.routed_experts.forward_fp32_finalize",
            params=runtime_params,
            extra={
                **route_extra,
                "stage": "runner.target_verify_microbatch.routed_output_fp32_finalize",
                "raw_output": _dsv4_moe_reduce_tensor_census(routed_raw),
                "finalized_output": _dsv4_moe_reduce_tensor_census(routed),
            },
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="shared_expert_input",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=flat,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.shared_expert_input",
            params=runtime_params,
            extra={
                "stage": "runner.target_verify_microbatch.shared_input",
                "shared_experts_present": self.shared_experts is not None,
                "microbatch_contract": contract.as_record(),
            },
        )
        if isinstance(shared_raw, torch.Tensor):
            _dsv4_moe_record_operator(
                batch,
                operator_name="shared_expert_output_raw",
                layer_id=self.layer_id,
                input_tensor=flat,
                output_tensor=shared_raw,
                positions=positions,
                path="mini.moe.runner.target_verify_microbatch.shared_experts.forward_raw",
                params=runtime_params,
                extra={"stage": "runner.target_verify_microbatch.shared_output_raw"},
            )
        if isinstance(shared, torch.Tensor):
            _dsv4_moe_record_operator(
                batch,
                operator_name="shared_expert_output",
                layer_id=self.layer_id,
                input_tensor=flat,
                output_tensor=shared,
                positions=positions,
                path="mini.moe.runner.target_verify_microbatch.shared_experts.forward_fp32",
                params=runtime_params,
                extra={
                    "stage": "runner.target_verify_microbatch.shared_output_fp32_finalize",
                    "raw_output": _dsv4_moe_reduce_tensor_census(shared_raw),
                    "finalized_output": _dsv4_moe_reduce_tensor_census(shared),
                },
            )
        _dsv4_moe_record_operator(
            batch,
            operator_name="expert_aggregate_before_reduce",
            layer_id=self.layer_id,
            input_tensor=routed,
            output_tensor=aggregate,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.routed_plus_shared_fp32",
            params=runtime_params,
            extra={
                "stage": "runner.target_verify_microbatch.aggregate_before_reduce",
                "shared_experts_present": shared is not None,
                "routed_input": _dsv4_moe_reduce_tensor_census(routed),
                "shared_input": _dsv4_moe_reduce_tensor_census(shared),
                "aggregate_output": _dsv4_moe_reduce_tensor_census(aggregate),
                "aggregate_order": "routed_fp32_plus_shared_fp32",
                "microbatch_contract": contract.as_record(),
            },
        )
        reduce_label = "per_microbatch_moe_plan_final_reduce"
        reduce_boundary_extra = _dsv4_moe_reduce_boundary_base_extra(
            stage="runner.target_verify_microbatch.post_all_reduce",
            tp_size=self._tp_size,
            comm=comm,
            reduce_label=reduce_label,
            pre_reduce=aggregate if isinstance(aggregate, torch.Tensor) else flat,
        )
        reduce_boundary_extra.update(
            {
                "communication_input": _dsv4_moe_reduce_tensor_census_if_enabled(aggregate),
                "communication_input_dtype": str(getattr(aggregate, "dtype", "")),
                "post_reduce": _dsv4_moe_reduce_tensor_census_if_enabled(reduced),
                "communication_output_dtype": str(getattr(reduced, "dtype", "")),
                "microbatch_contract": contract.as_record(),
            }
        )
        reduce_input_row0 = dsv4_mtp_debug.clone_operator_row0_input(
            "expert_reduce_output",
            aggregate,
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="expert_reduce_output",
            layer_id=self.layer_id,
            input_tensor=aggregate,
            output_tensor=reduced,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.post_all_reduce",
            params={**runtime_params, "reduce_label": reduce_label},
            extra={**reduce_boundary_extra, "post_all_reduce": bool(self._tp_size > 1)},
            input_row0=reduce_input_row0,
        )
        moe_output_extra = {
            "stage": "runner.target_verify_microbatch.moe_output",
            "final_cast_input": _dsv4_moe_reduce_tensor_census_if_enabled(reduced),
            "final_cast_output": _dsv4_moe_reduce_tensor_census_if_enabled(output),
            "final_cast_input_dtype": str(getattr(reduced, "dtype", "")),
            "final_cast_output_dtype": str(output.dtype),
            "final_cast_order": "post_reduce_to_hidden_dtype",
            "microbatch_contract": contract.as_record(),
        }
        _dsv4_moe_record_operator(
            batch,
            operator_name="moe_output",
            layer_id=self.layer_id,
            input_tensor=reduced,
            output_tensor=output,
            positions=positions,
            path="mini.moe.runner.target_verify_microbatch.output_to_hidden_dtype",
            params=runtime_params,
            extra=moe_output_extra,
        )
        dsv4_mtp_debug.record_moe_microbatch_runtime(
            batch,
            {
                "layer_id": int(self.layer_id),
                "enabled": True,
                "expert_backend": dsv4_kernel.dsv4_moe_expert_backend(),
                "original_moe_calls": 1,
                "runtime_moe_calls": int(contract.chunk_count),
                "elapsed_ms": timing_ms,
                **contract.as_record(),
            },
        )
        self._maybe_record_contract_oracle(
            batch=batch,
            hidden_states=hidden_states,
            flat=flat,
            flat_input_ids=flat_input_ids,
            positions=positions,
            comm=comm,
            hash_topk=hash_topk,
            common_params=runtime_params,
        )
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        comm: DistributedCommunicator,
        hash_topk: DSV4TopK | None,
    ) -> torch.Tensor:
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        flat_input_ids = input_ids.view(-1)
        batch = _dsv4_debug_batch()
        positions = _dsv4_debug_positions(batch, device=hidden_states.device)
        is_target_verify = _dsv4_is_target_verify_batch(batch)
        common_params = {
            "runner": True,
            "topk_count": int(self.topk_count),
            "scoring_func": str(self.scoring_func),
            "routed_scaling_factor": float(self.routed_scaling_factor),
            "expert_backend": dsv4_kernel.dsv4_moe_expert_backend(),
            "tp_size": int(self._tp_size),
            "target_verify_row_invariant_local": bool(is_target_verify),
        }
        microbatch_contract = _dsv4_target_verify_moe_microbatch_contract(
            batch,
            int(flat.shape[0]),
        )
        if microbatch_contract is not None:
            return self._forward_target_verify_microbatch(
                hidden_states,
                input_ids,
                flat=flat,
                flat_input_ids=flat_input_ids,
                batch=batch,
                positions=positions,
                comm=comm,
                hash_topk=hash_topk,
                common_params=common_params,
                contract=microbatch_contract,
            )
        _record_warmup_memory("moe.gate", "before", layer_id=self.layer_id)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.route"):
            weights, indices = self.route(
                flat,
                flat_input_ids,
                hash_topk=hash_topk,
                row_invariant_local=is_target_verify,
            )
        _record_warmup_memory("moe.gate", "after", layer_id=self.layer_id)
        router_logits = _dsv4_moe_router_logits_debug(self.gate, flat)
        route_extra = _dsv4_moe_route_extra(
            weights=weights,
            indices=indices,
            input_ids=flat_input_ids,
            reduce_once=True,
            hash_topk=hash_topk is not None,
            stage="runner.route",
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="router_logits",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=router_logits,
            positions=positions,
            path="mini.moe.runner.router_logits.linear_bf16_fp32_debug",
            params=common_params,
            extra=route_extra,
        )
        topk_input = router_logits if router_logits is not None else flat
        _dsv4_moe_record_operator(
            batch,
            operator_name="topk_ids",
            layer_id=self.layer_id,
            input_tensor=topk_input,
            output_tensor=indices,
            positions=positions,
            path="mini.moe.runner.topk_ids",
            params=common_params,
            extra=route_extra,
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="topk_weights",
            layer_id=self.layer_id,
            input_tensor=topk_input,
            output_tensor=weights,
            positions=positions,
            path="mini.moe.runner.topk_weights",
            params=common_params,
            extra=route_extra,
        )
        _record_warmup_memory("moe.route_plan", "before", layer_id=self.layer_id)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.prepare"):
            prepared = self.prepare(flat, weights, indices)
        _record_warmup_memory("moe.route_plan", "after", layer_id=self.layer_id)
        route_extra = _dsv4_moe_route_extra(
            weights=weights,
            indices=indices,
            input_ids=flat_input_ids,
            moe_plan=prepared.moe_plan,
            reduce_once=True,
            hash_topk=hash_topk is not None,
            stage="runner.prepare",
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_input",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=flat,
            positions=positions,
            path="mini.moe.runner.routed_expert_input",
            params=common_params,
            extra=route_extra,
        )
        _record_warmup_memory("moe.routed_experts", "before", layer_id=self.layer_id)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.experts"):
            if is_target_verify:
                routed = self.apply_experts_row_invariant(flat, prepared)
            else:
                routed = self.apply_experts(flat, prepared)
        _record_warmup_memory("moe.routed_experts", "after", layer_id=self.layer_id)
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_output_raw",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=routed,
            positions=positions,
            path="mini.moe.runner.routed_experts.forward_raw",
            params=common_params,
            extra={**route_extra, "stage": "runner.routed_output_raw"},
        )
        _record_warmup_memory("moe.finalize_routed", "before", layer_id=self.layer_id)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.finalize"):
            y = self.finalize_routed(routed)
        _record_warmup_memory("moe.finalize_routed", "after", layer_id=self.layer_id)
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_output",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=y,
            positions=positions,
            path="mini.moe.runner.routed_experts.forward_fp32_finalize",
            params=common_params,
            extra={
                **route_extra,
                "stage": "runner.routed_output_fp32_finalize",
                "raw_output": _dsv4_moe_reduce_tensor_census(routed),
                "finalized_output": _dsv4_moe_reduce_tensor_census(y),
            },
        )
        _record_warmup_memory("moe.shared_experts", "before", layer_id=self.layer_id)
        _dsv4_moe_record_operator(
            batch,
            operator_name="shared_expert_input",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=flat,
            positions=positions,
            path="mini.moe.runner.shared_expert_input",
            params=common_params,
            extra={"stage": "runner.shared_input", "shared_experts_present": self.shared_experts is not None},
        )
        routed_for_aggregate = y
        shared = None
        shared_raw = None
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.shared"):
            shared_raw = self.apply_shared_raw(
                flat,
                row_invariant_local=is_target_verify,
            )
            if shared_raw is not None:
                _dsv4_moe_record_operator(
                    batch,
                    operator_name="shared_expert_output_raw",
                    layer_id=self.layer_id,
                    input_tensor=flat,
                    output_tensor=shared_raw,
                    positions=positions,
                    path="mini.moe.runner.shared_experts.forward_raw",
                    params=common_params,
                    extra={"stage": "runner.shared_output_raw"},
                )
                shared = self.finalize_shared(shared_raw)
                y = y + shared
        _record_warmup_memory("moe.shared_experts", "after", layer_id=self.layer_id)
        if shared is not None:
            _dsv4_moe_record_operator(
                batch,
                operator_name="shared_expert_output",
                layer_id=self.layer_id,
                input_tensor=flat,
                output_tensor=shared,
                positions=positions,
                path="mini.moe.runner.shared_experts.forward_fp32",
                params=common_params,
                extra={
                    "stage": "runner.shared_output_fp32_finalize",
                    "raw_output": _dsv4_moe_reduce_tensor_census(shared_raw),
                    "finalized_output": _dsv4_moe_reduce_tensor_census(shared),
                },
            )
        if dsv4_mtp_debug.operator_parity_enabled("expert_aggregate_fp32_add_probe"):
            aggregate_fp32_probe = (
                routed_for_aggregate + shared
                if shared is not None
                else routed_for_aggregate
            )
            _dsv4_moe_record_operator(
                batch,
                operator_name="expert_aggregate_fp32_add_probe",
                layer_id=self.layer_id,
                input_tensor=routed_for_aggregate,
                output_tensor=aggregate_fp32_probe,
                positions=positions,
                path="mini.moe.runner.routed_plus_shared_fp32_probe",
                params=common_params,
                extra={
                    "stage": "runner.aggregate_fp32_add_probe",
                    "shared_experts_present": shared is not None,
                    "routed_input": _dsv4_moe_reduce_tensor_census(routed_for_aggregate),
                    "shared_input": _dsv4_moe_reduce_tensor_census(shared),
                },
            )
        if dsv4_mtp_debug.operator_parity_enabled("expert_aggregate_bf16_add_probe"):
            routed_bf16 = routed_for_aggregate.to(flat.dtype)
            if shared is None:
                aggregate_bf16_probe = routed_bf16.float()
            else:
                aggregate_bf16_probe = (routed_bf16 + shared.to(flat.dtype)).float()
            _dsv4_moe_record_operator(
                batch,
                operator_name="expert_aggregate_bf16_add_probe",
                layer_id=self.layer_id,
                input_tensor=routed_bf16,
                output_tensor=aggregate_bf16_probe,
                positions=positions,
                path="mini.moe.runner.routed_plus_shared_hidden_dtype_probe",
                params=common_params,
                extra={
                    "stage": "runner.aggregate_bf16_add_probe",
                    "hidden_dtype": str(flat.dtype),
                    "shared_experts_present": shared is not None,
                    "routed_input": _dsv4_moe_reduce_tensor_census(routed_bf16),
                    "shared_input": _dsv4_moe_reduce_tensor_census(
                        shared.to(flat.dtype) if shared is not None else None
                    ),
                },
            )
        _dsv4_moe_record_operator(
            batch,
            operator_name="expert_aggregate_before_reduce",
            layer_id=self.layer_id,
            input_tensor=routed_for_aggregate,
            output_tensor=y,
            positions=positions,
            path="mini.moe.runner.routed_plus_shared_fp32",
            params=common_params,
            extra={
                "stage": "runner.aggregate_before_reduce",
                "shared_experts_present": shared is not None,
                "routed_input": _dsv4_moe_reduce_tensor_census(routed_for_aggregate),
                "shared_input": _dsv4_moe_reduce_tensor_census(shared),
                "aggregate_output": _dsv4_moe_reduce_tensor_census(y),
                "aggregate_order": "routed_fp32_plus_shared_fp32",
            },
        )
        y_before_reduce = y
        reduce_input_row0 = dsv4_mtp_debug.clone_operator_row0_input(
            "expert_reduce_output",
            y_before_reduce,
        )
        _record_warmup_memory("moe.reduce_once", "before", layer_id=self.layer_id)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.reduce_once"):
            y, reduce_boundary_extra = self.maybe_reduce_final(
                y,
                comm=comm,
                hidden_dtype=flat.dtype,
                reduce_label=prepared.moe_plan.final_reduce_label,
            )
        _record_warmup_memory("moe.reduce_once", "after", layer_id=self.layer_id)
        _dsv4_moe_record_operator(
            batch,
            operator_name="expert_reduce_output",
            layer_id=self.layer_id,
            input_tensor=y_before_reduce,
            output_tensor=y,
            positions=positions,
            path="mini.moe.runner.post_all_reduce",
            params={**common_params, "reduce_label": str(prepared.moe_plan.final_reduce_label)},
            extra={**reduce_boundary_extra, "post_all_reduce": bool(self._tp_size > 1)},
            input_row0=reduce_input_row0,
        )
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.runner_output_to_flat_dtype.layer{self.layer_id}",
            y=y,
        ):
            output = y.to(flat.dtype).view_as(hidden_states)
        moe_output_extra = {
            "stage": "runner.moe_output",
            "final_cast_input": _dsv4_moe_reduce_tensor_census_if_enabled(y),
            "final_cast_output": _dsv4_moe_reduce_tensor_census_if_enabled(output),
            "final_cast_input_dtype": str(y.dtype),
            "final_cast_output_dtype": str(output.dtype),
            "final_cast_order": "post_reduce_to_hidden_dtype",
        }
        _dsv4_moe_record_operator(
            batch,
            operator_name="moe_output",
            layer_id=self.layer_id,
            input_tensor=y,
            output_tensor=output,
            positions=positions,
            path="mini.moe.runner.output_to_hidden_dtype",
            params=common_params,
            extra=moe_output_extra,
        )
        self._maybe_record_contract_oracle(
            batch=batch,
            hidden_states=hidden_states,
            flat=flat,
            flat_input_ids=flat_input_ids,
            positions=positions,
            comm=comm,
            hash_topk=hash_topk,
            common_params=common_params,
        )
        return output


class DSV4MoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, is_nextn: bool = False):
        tp = get_tp_info()
        self.layer_id = layer_id
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        is_hash_layer = layer_id < config.n_hash_layers and not is_nextn
        self.topk_count = config.num_experts_per_tok
        self.scoring_func = config.scoring_func or "sqrtsoftplus"
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = DSV4MoEGate(config, has_correction_bias=not is_hash_layer)
        if is_hash_layer:
            self.topk = DSV4TopK(config)
        self.experts = DSV4FusedRoutedExperts(config, layer_id=layer_id)
        if config.n_shared_experts > 0:
            self.shared_experts = DSV4SharedExperts(config, layer_id=layer_id)
        self._runner = DSV4FusedMoERunner(
            layer_id=layer_id,
            gate=self.gate,
            experts=self.experts,
            shared_experts=getattr(self, "shared_experts", None),
            topk_count=self.topk_count,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            tp_size=self._tp_size,
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch = _dsv4_debug_batch()
        positions = _dsv4_debug_positions(batch, device=hidden_states.device)
        is_target_verify = _dsv4_is_target_verify_batch(batch)
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        common_params = {
            "runner": False,
            "topk_count": int(self.topk_count),
            "scoring_func": str(self.scoring_func),
            "routed_scaling_factor": float(self.routed_scaling_factor),
            "expert_backend": dsv4_kernel.dsv4_moe_expert_backend(),
            "tp_size": int(self._tp_size),
            "target_verify_row_invariant_local": bool(is_target_verify),
        }
        _dsv4_moe_record_operator(
            batch,
            operator_name="moe_input",
            layer_id=self.layer_id,
            input_tensor=hidden_states,
            output_tensor=hidden_states,
            positions=positions,
            path="mini.moe.input",
            params=common_params,
            extra={"stage": "moe_input"},
        )
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE):
            output = self._runner.forward(
                hidden_states,
                input_ids,
                comm=self._comm,
                hash_topk=getattr(self, "topk", None),
            )
            return output
        if is_target_verify and _dsv4_target_verify_moe_microbatch_enabled():
            raise RuntimeError(
                "DeepSeek V4 target-verify MoE microbatch runtime requires the "
                "fused MoE runner path. Enable the A100 victory/fused runner "
                "MoE backend or disable "
                f"{DSV4_MTP_TARGET_VERIFY_MOE_MICROBATCH_ENV}."
            )

        _record_warmup_memory("moe.gate", "before", layer_id=self.layer_id)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.gate"):
            weights, indices = self.gate.forward(
                flat,
                input_ids=input_ids.view(-1),
                topk=self.topk_count,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                hash_topk=getattr(self, "topk", None),
                row_invariant_local=is_target_verify,
            )
        _record_warmup_memory("moe.gate", "after", layer_id=self.layer_id)
        flat_input_ids = input_ids.view(-1)
        router_logits = _dsv4_moe_router_logits_debug(self.gate, flat)
        reduce_once = dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE
        ) or dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE)
        route_extra = _dsv4_moe_route_extra(
            weights=weights,
            indices=indices,
            input_ids=flat_input_ids,
            reduce_once=reduce_once,
            hash_topk=hasattr(self, "topk"),
            stage="non_runner.route",
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="router_logits",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=router_logits,
            positions=positions,
            path="mini.moe.router_logits.linear_bf16_fp32_debug",
            params=common_params,
            extra=route_extra,
        )
        topk_input = router_logits if router_logits is not None else flat
        _dsv4_moe_record_operator(
            batch,
            operator_name="topk_ids",
            layer_id=self.layer_id,
            input_tensor=topk_input,
            output_tensor=indices,
            positions=positions,
            path="mini.moe.topk_ids",
            params=common_params,
            extra=route_extra,
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="topk_weights",
            layer_id=self.layer_id,
            input_tensor=topk_input,
            output_tensor=weights,
            positions=positions,
            path="mini.moe.topk_weights",
            params=common_params,
            extra=route_extra,
        )
        moe_v2 = dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE)
        reduce_once = moe_v2 or dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE)
        moe_plan = None
        if moe_v2:
            if hasattr(self.experts, "w13_weight"):
                num_experts = self.experts.w13_weight.shape[0]
            elif self.experts._marlin_wna16_weights is not None:
                num_experts = self.experts._marlin_wna16_weights.w13.shape[0]
            else:
                raise RuntimeError(
                    f"layer{self.layer_id}.moe cannot build a route plan because "
                    f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR} "
                    "Marlin WNA16 cache is missing."
                )
            _record_warmup_memory("moe.route_plan", "before", layer_id=self.layer_id)
            moe_plan = dsv4_kernel.build_moe_v2_execution_plan(
                flat,
                weights,
                indices,
                num_experts=num_experts,
                block_size_m=16,
                reduce_once=reduce_once,
            )
            _record_warmup_memory("moe.route_plan", "after", layer_id=self.layer_id)
            route_extra = _dsv4_moe_route_extra(
                weights=weights,
                indices=indices,
                input_ids=flat_input_ids,
                moe_plan=moe_plan,
                reduce_once=reduce_once,
                hash_topk=hasattr(self, "topk"),
                stage="non_runner.route_plan",
            )
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_input",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=flat,
            positions=positions,
            path="mini.moe.routed_expert_input",
            params=common_params,
            extra=route_extra,
        )
        _record_warmup_memory("moe.routed_experts", "before", layer_id=self.layer_id)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.routed"):
            if moe_plan is None:
                routed_raw = self.experts.forward(
                    flat,
                    weights,
                    indices,
                    reduce=not reduce_once,
                )
            else:
                routed_raw = self.experts.forward(
                    flat,
                    weights,
                    indices,
                    reduce=not reduce_once,
                    moe_plan=moe_plan,
                )
            y = routed_raw.float()
        _record_warmup_memory("moe.routed_experts", "after", layer_id=self.layer_id)
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_output_raw",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=routed_raw,
            positions=positions,
            path="mini.moe.routed_experts.forward_raw",
            params=common_params,
            extra={**route_extra, "stage": "non_runner.routed_output_raw"},
        )
        _dsv4_moe_record_operator(
            batch,
            operator_name="routed_expert_output",
            layer_id=self.layer_id,
            input_tensor=flat,
            output_tensor=y,
            positions=positions,
            path="mini.moe.routed_experts.forward_fp32",
            params=common_params,
            extra={
                **route_extra,
                "stage": "non_runner.routed_output_fp32_finalize",
                "raw_output": _dsv4_moe_reduce_tensor_census(routed_raw),
                "finalized_output": _dsv4_moe_reduce_tensor_census(y),
            },
        )
        routed_for_aggregate = y
        shared = None
        shared_raw = None
        if hasattr(self, "shared_experts"):
            _record_warmup_memory("moe.shared_experts", "before", layer_id=self.layer_id)
            _dsv4_moe_record_operator(
                batch,
                operator_name="shared_expert_input",
                layer_id=self.layer_id,
                input_tensor=flat,
                output_tensor=flat,
                positions=positions,
                path="mini.moe.shared_expert_input",
                params=common_params,
                extra={"stage": "non_runner.shared_input", "shared_experts_present": True},
            )
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.shared"):
                shared_raw = self.shared_experts.forward(
                    flat,
                    reduce=not reduce_once,
                    row_invariant_local=is_target_verify,
                )
                _dsv4_moe_record_operator(
                    batch,
                    operator_name="shared_expert_output_raw",
                    layer_id=self.layer_id,
                    input_tensor=flat,
                    output_tensor=shared_raw,
                    positions=positions,
                    path="mini.moe.shared_experts.forward_raw",
                    params=common_params,
                    extra={"stage": "non_runner.shared_output_raw"},
                )
                shared = shared_raw.float()
                y = y + shared
            _record_warmup_memory("moe.shared_experts", "after", layer_id=self.layer_id)
            _dsv4_moe_record_operator(
                batch,
                operator_name="shared_expert_output",
                layer_id=self.layer_id,
                input_tensor=flat,
                output_tensor=shared,
                positions=positions,
                path="mini.moe.shared_experts.forward_fp32",
                params=common_params,
                extra={
                    "stage": "non_runner.shared_output_fp32_finalize",
                    "raw_output": _dsv4_moe_reduce_tensor_census(shared_raw),
                    "finalized_output": _dsv4_moe_reduce_tensor_census(shared),
                },
            )
        if dsv4_mtp_debug.operator_parity_enabled("expert_aggregate_fp32_add_probe"):
            aggregate_fp32_probe = (
                routed_for_aggregate + shared
                if shared is not None
                else routed_for_aggregate
            )
            _dsv4_moe_record_operator(
                batch,
                operator_name="expert_aggregate_fp32_add_probe",
                layer_id=self.layer_id,
                input_tensor=routed_for_aggregate,
                output_tensor=aggregate_fp32_probe,
                positions=positions,
                path="mini.moe.routed_plus_shared_fp32_probe",
                params=common_params,
                extra={
                    "stage": "non_runner.aggregate_fp32_add_probe",
                    "shared_experts_present": shared is not None,
                    "routed_input": _dsv4_moe_reduce_tensor_census(routed_for_aggregate),
                    "shared_input": _dsv4_moe_reduce_tensor_census(shared),
                },
            )
        if dsv4_mtp_debug.operator_parity_enabled("expert_aggregate_bf16_add_probe"):
            routed_bf16 = routed_for_aggregate.to(flat.dtype)
            if shared is None:
                aggregate_bf16_probe = routed_bf16.float()
            else:
                aggregate_bf16_probe = (routed_bf16 + shared.to(flat.dtype)).float()
            _dsv4_moe_record_operator(
                batch,
                operator_name="expert_aggregate_bf16_add_probe",
                layer_id=self.layer_id,
                input_tensor=routed_bf16,
                output_tensor=aggregate_bf16_probe,
                positions=positions,
                path="mini.moe.routed_plus_shared_hidden_dtype_probe",
                params=common_params,
                extra={
                    "stage": "non_runner.aggregate_bf16_add_probe",
                    "hidden_dtype": str(flat.dtype),
                    "shared_experts_present": shared is not None,
                    "routed_input": _dsv4_moe_reduce_tensor_census(routed_bf16),
                    "shared_input": _dsv4_moe_reduce_tensor_census(
                        shared.to(flat.dtype) if shared is not None else None
                    ),
                },
            )
        _dsv4_moe_record_operator(
            batch,
            operator_name="expert_aggregate_before_reduce",
            layer_id=self.layer_id,
            input_tensor=routed_for_aggregate,
            output_tensor=y,
            positions=positions,
            path="mini.moe.routed_plus_shared_fp32",
            params=common_params,
            extra={
                "stage": "non_runner.aggregate_before_reduce",
                "shared_experts_present": shared is not None,
                "routed_input": _dsv4_moe_reduce_tensor_census(routed_for_aggregate),
                "shared_input": _dsv4_moe_reduce_tensor_census(shared),
                "aggregate_output": _dsv4_moe_reduce_tensor_census(y),
                "aggregate_order": "routed_fp32_plus_shared_fp32",
            },
        )
        y_before_reduce = y
        reduce_input_row0 = dsv4_mtp_debug.clone_operator_row0_input(
            "expert_reduce_output",
            y_before_reduce,
        )
        reduce_boundary_extra = _dsv4_moe_reduce_boundary_base_extra(
            stage="non_runner.post_all_reduce",
            tp_size=self._tp_size if reduce_once else 1,
            comm=self._comm,
            reduce_label="dsv4.v1_moe_reduce_once_all_reduce",
            pre_reduce=y_before_reduce,
        )
        if reduce_once and self._tp_size > 1:
            _record_warmup_memory("moe.reduce_once", "before", layer_id=self.layer_id)
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.reduce_once"):
                y = _dsv4_moe_reduce_once_input(
                    y,
                    hidden_dtype=flat.dtype,
                    layer_id=self.layer_id,
                    path="non_runner_output",
                )
                reduce_boundary_extra["communication_input"] = (
                    _dsv4_moe_reduce_tensor_census_if_enabled(y)
                )
                reduce_boundary_extra["communication_input_dtype"] = str(y.dtype)
                y = self._comm.all_reduce(y, label="dsv4.v1_moe_reduce_once_all_reduce")
                reduce_boundary_extra["post_reduce"] = (
                    _dsv4_moe_reduce_tensor_census_if_enabled(y)
                )
                reduce_boundary_extra["communication_output_dtype"] = str(y.dtype)
            _record_warmup_memory("moe.reduce_once", "after", layer_id=self.layer_id)
        else:
            reduce_boundary_extra["communication_input"] = (
                _dsv4_moe_reduce_tensor_census_if_enabled(y)
            )
            reduce_boundary_extra["communication_input_dtype"] = str(y.dtype)
            reduce_boundary_extra["post_reduce"] = _dsv4_moe_reduce_tensor_census_if_enabled(y)
            reduce_boundary_extra["communication_output_dtype"] = str(y.dtype)
        _dsv4_moe_record_operator(
            batch,
            operator_name="expert_reduce_output",
            layer_id=self.layer_id,
            input_tensor=y_before_reduce,
            output_tensor=y,
            positions=positions,
            path=(
                "mini.moe.post_all_reduce" if reduce_once else "mini.moe.no_final_reduce"
            ),
            params={**common_params, "reduce_label": "dsv4.v1_moe_reduce_once_all_reduce"},
            extra={
                **reduce_boundary_extra,
                "post_all_reduce": bool(reduce_once and self._tp_size > 1),
            },
            input_row0=reduce_input_row0,
        )
        output = y.to(flat.dtype).view_as(hidden_states)
        moe_output_extra = {
            "stage": "moe_output",
            "final_cast_input": _dsv4_moe_reduce_tensor_census_if_enabled(y),
            "final_cast_output": _dsv4_moe_reduce_tensor_census_if_enabled(output),
            "final_cast_input_dtype": str(y.dtype),
            "final_cast_output_dtype": str(output.dtype),
            "final_cast_order": "post_reduce_to_hidden_dtype",
        }
        _dsv4_moe_record_operator(
            batch,
            operator_name="moe_output",
            layer_id=self.layer_id,
            input_tensor=y,
            output_tensor=output,
            positions=positions,
            path="mini.moe.output_to_hidden_dtype",
            params=common_params,
            extra=moe_output_extra,
        )
        return output


class DeepseekV4DecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        is_nextn: bool = False,
        compress_ratio_override: int | None = None,
    ):
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.self_attn = DSV4Attention(
            config,
            layer_id,
            compress_ratio_override=compress_ratio_override,
        )
        self.mlp = DSV4MoE(config, layer_id, is_nextn=is_nextn)
        self.input_layernorm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)

        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)
        self._hc_attn_fn_bf16: torch.Tensor | None = None
        self._hc_attn_fn_bf16_meta: tuple | None = None
        self._hc_ffn_fn_bf16: torch.Tensor | None = None
        self._hc_ffn_fn_bf16_meta: tuple | None = None

    def _hc_pre(
        self,
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return dsv4_kernel.hc_pre_fallback(
            x,
            fn,
            scale,
            base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
            norm_eps=self.norm_eps,
        )

    def _hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return dsv4_kernel.hc_post_fallback(x, residual, post, comb)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        layer_id = self.self_attn.layer_id
        _capture_debug_activation(f"layer{layer_id}.input", x)
        residual = x
        attn_fn = _cached_hc_bf16_weight(self, "_hc_attn_fn_bf16", self.hc_attn_fn)
        _record_warmup_memory("layer.hc_attn_pre", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_attn_pre"):
            y, post, comb = self._hc_pre(x, attn_fn, self.hc_attn_scale, self.hc_attn_base)
        _record_warmup_memory("layer.hc_attn_pre", "after", layer_id=layer_id)
        _record_warmup_memory("layer.attn_input_norm", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.attn_input_norm"):
            y = self.input_layernorm.forward(y)
            _capture_debug_activation(f"layer{layer_id}.attention_input", y)
        _record_warmup_memory("layer.attn_input_norm", "after", layer_id=layer_id)
        _record_warmup_memory("layer.attention", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.attn"):
            y = self.self_attn.forward(y)
            _capture_debug_activation(f"layer{layer_id}.attention_output", y)
        _record_warmup_memory("layer.attention", "after", layer_id=layer_id)
        _record_warmup_memory("layer.hc_attn_post", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_attn_post"):
            x = self._hc_post(y, residual, post, comb)
            _capture_debug_activation(f"layer{layer_id}.post_attention_residual", x)
        _record_warmup_memory("layer.hc_attn_post", "after", layer_id=layer_id)

        residual = x
        ffn_fn = _cached_hc_bf16_weight(self, "_hc_ffn_fn_bf16", self.hc_ffn_fn)
        _record_warmup_memory("layer.hc_ffn_pre", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_ffn_pre"):
            y, post, comb = self._hc_pre(x, ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        _record_warmup_memory("layer.hc_ffn_pre", "after", layer_id=layer_id)
        _record_warmup_memory("layer.mlp_input_norm", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.mlp_input_norm"):
            y = self.post_attention_layernorm.forward(y)
            _capture_debug_activation(f"layer{layer_id}.moe_input", y)
        _record_warmup_memory("layer.mlp_input_norm", "after", layer_id=layer_id)
        _record_warmup_memory("layer.moe", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.mlp"):
            y = self.mlp.forward(y, input_ids)
            _capture_debug_activation(f"layer{layer_id}.moe_output", y)
        _record_warmup_memory("layer.moe", "after", layer_id=layer_id)
        _record_warmup_memory("layer.hc_ffn_post", "before", layer_id=layer_id)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_ffn_post"):
            output = self._hc_post(y, residual, post, comb)
            _capture_debug_activation(f"layer{layer_id}.post_moe_residual", output)
        _record_warmup_memory("layer.hc_ffn_post", "after", layer_id=layer_id)
        _record_warmup_memory("layer.output", "after", layer_id=layer_id)
        return output


@dataclass(frozen=True)
class DSV4HiddenForwardOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    hidden_states_before_norm: torch.Tensor


@dataclass(frozen=True)
class DSV4MTPForwardOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    hidden_states_before_norm: torch.Tensor


class DSV4MTPSharedHead(BaseOP):
    def __init__(self, config: ModelConfig):
        self.norm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)


class DeepseekV4MTPModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.backbone_hidden_size = config.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.enorm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.e_proj = DSV4Linear(
            config.hidden_size,
            config.hidden_size,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.h_proj = DSV4Linear(
            config.hidden_size,
            config.hidden_size,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )

        decoder_config = config
        if not config.compress_ratios or config.compress_ratios[0] != 0:
            ratios = [0] + list(config.compress_ratios[1:])
            decoder_config = replace(config, compress_ratios=ratios)
        self.decoder = DeepseekV4DecoderLayer(
            decoder_config,
            layer_id=0,
            is_nextn=True,
            compress_ratio_override=0,
        )
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = torch.empty(config.hc_mult, hc_dim, dtype=torch.float32)
        self.hc_head_base = torch.empty(config.hc_mult, dtype=torch.float32)
        self.hc_head_scale = torch.empty(1, dtype=torch.float32)
        self.shared_head = DSV4MTPSharedHead(config)
        self._hc_head_fn_bf16: torch.Tensor | None = None
        self._hc_head_fn_bf16_meta: tuple | None = None

    def prepare_for_cuda_graph_capture(self) -> dict[str, object]:
        q_wqb_reports: list[dict[str, object]] = []
        report = self.decoder.self_attn.prepare_q_wqb_bf16_weight_cache()
        if report is not None:
            q_wqb_reports.append(report)

        q_wqb_marlin_reports: list[dict[str, object]] = []
        report = self.decoder.self_attn.prepare_q_wqb_marlin_weight_cache()
        if report is not None:
            q_wqb_marlin_reports.append(report)

        wo_b_reports: list[dict[str, object]] = []
        report = self.decoder.self_attn.prepare_wo_b_bf16_weight_cache()
        if report is not None:
            wo_b_reports.append(report)

        wo_b_marlin_reports: list[dict[str, object]] = []
        report = self.decoder.self_attn.prepare_wo_b_marlin_weight_cache()
        if report is not None:
            wo_b_marlin_reports.append(report)

        wo_a_reports: list[dict[str, object]] = []
        report = self.decoder.self_attn.prepare_wo_a_bf16_bmm_cache()
        if report is not None:
            wo_a_reports.append(report)

        indexer_wq_b_reports: list[dict[str, object]] = []
        report = self.decoder.self_attn.prepare_indexer_wq_b_bf16_weight_cache()
        if report is not None:
            indexer_wq_b_reports.append(report)

        fused_wqa_wkv_reports: list[dict[str, object]] = []
        report = self.decoder.self_attn.prepare_fused_wqa_wkv_pretranspose_cache()
        if report is not None:
            fused_wqa_wkv_reports.append(report)

        shared_expert_reports: list[dict[str, object]] = []
        shared_experts = getattr(self.decoder.mlp, "shared_experts", None)
        if shared_experts is not None:
            shared_expert_reports.extend(shared_experts.prepare_bf16_weight_cache())

        shared_down_marlin_reports: list[dict[str, object]] = []
        if shared_experts is not None:
            report = shared_experts.prepare_down_marlin_weight_cache()
            if report is not None:
                shared_down_marlin_reports.append(report)

        moe_marlin_wna16_reports: list[dict[str, object]] = []
        moe_marlin_backend = dsv4_kernel.dsv4_moe_expert_backend()
        moe_marlin_prebuild_enabled = dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_MARLIN_WNA16_PREBUILD_ENV
        )
        if (
            moe_marlin_prebuild_enabled
            and moe_marlin_backend == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16
        ):
            moe_marlin_wna16_reports.append(
                self.decoder.mlp.experts.prepare_marlin_wna16_weight_cache(
                    release_original=False,
                )
            )

        total_q_wqb_bytes = int(sum(int(report["bytes"]) for report in q_wqb_reports))
        total_wo_b_bytes = int(sum(int(report["bytes"]) for report in wo_b_reports))
        total_wo_a_bytes = int(sum(int(report["bytes"]) for report in wo_a_reports))
        total_indexer_wq_b_bytes = int(sum(int(report["bytes"]) for report in indexer_wq_b_reports))
        total_shared_expert_bytes = int(
            sum(int(report["bytes"]) for report in shared_expert_reports)
        )
        marlin_reports = q_wqb_marlin_reports + wo_b_marlin_reports + shared_down_marlin_reports
        total_marlin_persistent_bytes = int(
            sum(int(report["persistent_bytes"]) for report in marlin_reports)
        )
        total_marlin_workspace_bytes = int(
            sum(int(report["workspace_bytes"]) for report in marlin_reports)
        )
        total_moe_marlin_wna16_persistent_bytes = int(
            sum(int(report["persistent_bytes"]) for report in moe_marlin_wna16_reports)
        )
        total_moe_marlin_wna16_source_bytes = int(
            sum(int(report["source_bytes"]) for report in moe_marlin_wna16_reports)
        )
        total_pretransposed_bytes = int(
            sum(int(report.get("pretransposed_bytes", 0)) for report in fused_wqa_wkv_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in q_wqb_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in wo_b_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in indexer_wq_b_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in shared_expert_reports)
        )
        return {
            "q_wqb_bf16_weight_cache": {
                "enabled": bool(q_wqb_reports),
                "layers_cached": len(q_wqb_reports),
                "total_bytes": total_q_wqb_bytes,
                "entries": q_wqb_reports,
            },
            "wo_b_bf16_weight_cache": {
                "enabled": bool(wo_b_reports),
                "layers_cached": len(wo_b_reports),
                "total_bytes": total_wo_b_bytes,
                "entries": wo_b_reports,
            },
            "wo_a_bf16_bmm_cache": {
                "enabled": bool(wo_a_reports),
                "layers_cached": len(wo_a_reports),
                "total_bytes": total_wo_a_bytes,
                "entries": wo_a_reports,
            },
            "indexer_wq_b_bf16_weight_cache": {
                "enabled": bool(indexer_wq_b_reports),
                "layers_cached": len(indexer_wq_b_reports),
                "total_bytes": total_indexer_wq_b_bytes,
                "entries": indexer_wq_b_reports,
            },
            "fused_wqa_wkv_bf16_pretranspose_cache": {
                "enabled": bool(fused_wqa_wkv_reports),
                "layers_cached": len(fused_wqa_wkv_reports),
                "total_bytes": int(sum(int(report["bytes"]) for report in fused_wqa_wkv_reports)),
                "total_pretransposed_bytes": int(
                    sum(
                        int(report.get("pretransposed_bytes", 0))
                        for report in fused_wqa_wkv_reports
                    )
                ),
                "entries": fused_wqa_wkv_reports,
            },
            "shared_expert_bf16_weight_cache": {
                "enabled": bool(shared_expert_reports),
                "layers_cached": max(
                    sum(
                        1
                        for report in shared_expert_reports
                        if str(report["owner"]).endswith("gate_up_proj")
                    ),
                    sum(
                        1
                        for report in shared_expert_reports
                        if str(report["owner"]).endswith("down_proj")
                    ),
                ),
                "total_bytes": total_shared_expert_bytes,
                "entries": shared_expert_reports,
            },
            "projection_bf16_weight_cache_total": {
                "total_bytes": (
                    total_q_wqb_bytes
                    + total_wo_b_bytes
                    + total_wo_a_bytes
                    + total_indexer_wq_b_bytes
                    + total_shared_expert_bytes
                ),
            },
            "dense_fp8_marlin_projection_cache": {
                "enabled": bool(marlin_reports),
                "layers_cached": max(
                    len(q_wqb_marlin_reports),
                    len(wo_b_marlin_reports),
                    len(shared_down_marlin_reports),
                ),
                "total_persistent_bytes": total_marlin_persistent_bytes,
                "total_workspace_bytes": total_marlin_workspace_bytes,
                "q_wqb": q_wqb_marlin_reports,
                "wo_b": wo_b_marlin_reports,
                "shared_down": shared_down_marlin_reports,
            },
            "moe_marlin_wna16_cache": {
                "enabled": bool(moe_marlin_wna16_reports),
                "backend": moe_marlin_backend,
                "prebuild_requested": bool(moe_marlin_prebuild_enabled),
                "layers_cached": len(moe_marlin_wna16_reports),
                "total_persistent_bytes": total_moe_marlin_wna16_persistent_bytes,
                "total_source_bytes": total_moe_marlin_wna16_source_bytes,
                "entries": moe_marlin_wna16_reports,
            },
            "bf16_small_gemm_pretranspose_cache_total": {
                "enabled": total_pretransposed_bytes > 0,
                "total_pretransposed_bytes": total_pretransposed_bytes,
            },
        }

    def _reshape_target_hidden(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        if target_hidden_states.ndim == 3:
            expected = (self.hc_mult, self.hidden_size)
            if tuple(target_hidden_states.shape[1:]) != expected:
                raise ValueError(
                    "DeepSeek V4 MTP target hidden must have trailing shape "
                    f"{expected}, got {tuple(target_hidden_states.shape[1:])}"
                )
            return target_hidden_states
        if target_hidden_states.ndim != 2:
            raise ValueError(
                "DeepSeek V4 MTP target hidden must be [tokens, hc_mult * hidden] "
                f"or [tokens, hc_mult, hidden], got rank {target_hidden_states.ndim}"
            )
        expected_last_dim = self.hc_mult * self.hidden_size
        if target_hidden_states.shape[-1] != expected_last_dim:
            raise ValueError(
                "DeepSeek V4 MTP target hidden last dimension must be "
                f"{expected_last_dim}, got {target_hidden_states.shape[-1]}"
            )
        return target_hidden_states.view(-1, self.hc_mult, self.hidden_size)

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        hc_head_fn = _cached_hc_bf16_weight(self, "_hc_head_fn_bf16", self.hc_head_fn)
        return dsv4_kernel.hc_head_fallback(
            x,
            hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            eps=self.hc_eps,
            norm_eps=self.rms_norm_eps,
        )

    def forward_one_step(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        *,
        embed_tokens: DSV4VocabParallelEmbedding,
        lm_head: DSV4VocabParallelEmbedding,
    ) -> DSV4MTPForwardOutput:
        token_hidden = embed_tokens.forward(input_ids)
        if token_hidden.shape[0] > 0:
            target_hidden = self._reshape_target_hidden(target_hidden_states).to(
                device=token_hidden.device,
                dtype=token_hidden.dtype,
            )
            if target_hidden.shape[0] != token_hidden.shape[0]:
                raise ValueError(
                    "DeepSeek V4 MTP input_ids and target hidden must have the same "
                    f"token count, got {token_hidden.shape[0]} and {target_hidden.shape[0]}"
                )
            flat_hidden = target_hidden.reshape(-1, self.hidden_size)
            h_proj = self.h_proj.forward(self.hnorm.forward(flat_hidden))
            h_proj = h_proj.view(token_hidden.shape[0], self.hc_mult, self.hidden_size)
            e_proj = self.e_proj.forward(self.enorm.forward(token_hidden))
            hidden_states = e_proj[:, None, :] + h_proj
        else:
            hidden_states = token_hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)

        hidden_states = self.decoder.forward(hidden_states, input_ids)
        pre_hc_head = hidden_states.flatten(1)
        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.shared_head.norm.forward(hidden_states)
        logits = lm_head.linear(hidden_states)
        return DSV4MTPForwardOutput(
            logits=logits,
            hidden_states=hidden_states,
            hidden_states_before_norm=pre_hc_head,
        )


class DeepseekV4Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.embed_tokens = DSV4VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [DeepseekV4DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = torch.empty(config.hc_mult, hc_dim, dtype=torch.float32)
        self.hc_head_base = torch.empty(config.hc_mult, dtype=torch.float32)
        self.hc_head_scale = torch.empty(1, dtype=torch.float32)
        self._hc_head_fn_bf16: torch.Tensor | None = None
        self._hc_head_fn_bf16_meta: tuple | None = None
        self._marlin_wna16_release_quarantine_tensors: list[torch.Tensor] = []
        self._marlin_wna16_release_quarantine_records: list[dict[str, object]] = []

    def prepare_for_cuda_graph_capture(self) -> dict[str, object]:
        fused_wqa_wkv_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE):
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_fused_wqa_wkv_pretranspose_cache()
                if report is not None:
                    fused_wqa_wkv_reports.append(report)
        q_wqb_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE):
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_q_wqb_bf16_weight_cache()
                if report is not None:
                    q_wqb_reports.append(report)
        q_wqb_marlin_reports: list[dict[str, object]] = []
        if dsv4_kernel.dense_fp8_marlin_projection_enabled():
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_q_wqb_marlin_weight_cache()
                if report is not None:
                    q_wqb_marlin_reports.append(report)
        wo_b_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_B_BF16_WEIGHT_CACHE_TOGGLE):
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_wo_b_bf16_weight_cache()
                if report is not None:
                    wo_b_reports.append(report)
        wo_b_marlin_reports: list[dict[str, object]] = []
        if dsv4_kernel.dense_fp8_marlin_projection_enabled():
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_wo_b_marlin_weight_cache()
                if report is not None:
                    wo_b_marlin_reports.append(report)
        wo_a_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE):
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_wo_a_bf16_bmm_cache()
                if report is not None:
                    wo_a_reports.append(report)
        indexer_wq_b_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE):
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_indexer_wq_b_bf16_weight_cache()
                if report is not None:
                    indexer_wq_b_reports.append(report)
        shared_expert_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE):
            for layer in self.layers.op_list:
                shared_experts = getattr(layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    shared_expert_reports.extend(shared_experts.prepare_bf16_weight_cache())
        shared_down_marlin_reports: list[dict[str, object]] = []
        if dsv4_kernel.dense_fp8_marlin_projection_enabled():
            for layer in self.layers.op_list:
                shared_experts = getattr(layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    report = shared_experts.prepare_down_marlin_weight_cache()
                    if report is not None:
                        shared_down_marlin_reports.append(report)
        moe_marlin_wna16_reports: list[dict[str, object]] = []
        moe_marlin_wna16_prebuild_reports: list[dict[str, object]] = []
        moe_marlin_wna16_release_reports: list[dict[str, object]] = []
        moe_marlin_backend = dsv4_kernel.dsv4_moe_expert_backend()
        moe_marlin_prebuild_enabled = dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_MARLIN_WNA16_PREBUILD_ENV
        )
        moe_marlin_release_original = dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS_ENV
        )
        moe_marlin_release_after_graph_capture = dsv4_kernel.dsv4_env_flag(
            _MARLIN_WNA16_RELEASE_AFTER_GRAPH_CAPTURE_ENV
        )
        moe_marlin_release_timing = _marlin_wna16_release_timing()
        moe_marlin_release_deferred = _marlin_wna16_release_deferred_from_model_prepare()
        if moe_marlin_release_original and not moe_marlin_prebuild_enabled:
            raise RuntimeError(
                f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS_ENV}=1 "
                f"requires {dsv4_kernel.DSV4_MARLIN_WNA16_PREBUILD_ENV}=1 so the "
                "Marlin WNA16 cache exists before original expert weights are released."
            )
        if (
            moe_marlin_release_original
            and moe_marlin_backend != dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16
        ):
            raise RuntimeError(
                f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS_ENV}=1 "
                f"requires backend={dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16!r}, "
                f"got {moe_marlin_backend!r}."
            )
        if (
            moe_marlin_prebuild_enabled
            and moe_marlin_backend == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16
        ):
            for layer in self.layers.op_list:
                moe_marlin_wna16_prebuild_reports.append(
                    layer.mlp.experts.prepare_marlin_wna16_weight_cache(
                        release_original=False,
                    )
                )
            moe_marlin_wna16_reports = moe_marlin_wna16_prebuild_reports
            self.audit_marlin_wna16_cache_integrity("after_full_model_prebuild")
            if moe_marlin_release_original and not moe_marlin_release_deferred:
                moe_marlin_wna16_release_reports = (
                    self.release_marlin_wna16_original_expert_weights(
                        stage_label="model_prepare_release",
                    )["entries"]
                )
                moe_marlin_wna16_reports = moe_marlin_wna16_release_reports
        total_q_wqb_bytes = int(sum(int(report["bytes"]) for report in q_wqb_reports))
        total_wo_b_bytes = int(sum(int(report["bytes"]) for report in wo_b_reports))
        total_indexer_wq_b_bytes = int(sum(int(report["bytes"]) for report in indexer_wq_b_reports))
        total_wo_a_bytes = int(sum(int(report["bytes"]) for report in wo_a_reports))
        total_shared_expert_bytes = int(
            sum(int(report["bytes"]) for report in shared_expert_reports)
        )
        marlin_reports = q_wqb_marlin_reports + wo_b_marlin_reports + shared_down_marlin_reports
        total_q_wqb_marlin_bytes = int(
            sum(int(report["persistent_bytes"]) for report in q_wqb_marlin_reports)
        )
        total_wo_b_marlin_bytes = int(
            sum(int(report["persistent_bytes"]) for report in wo_b_marlin_reports)
        )
        total_shared_down_marlin_bytes = int(
            sum(int(report["persistent_bytes"]) for report in shared_down_marlin_reports)
        )
        total_marlin_workspace_bytes = int(
            sum(int(report["workspace_bytes"]) for report in marlin_reports)
        )
        total_marlin_original_released_bytes = int(
            sum(
                int(released["bytes"])
                for report in marlin_reports
                for released in report.get("released", [])
            )
        )
        total_moe_marlin_wna16_persistent_bytes = int(
            sum(int(report["persistent_bytes"]) for report in moe_marlin_wna16_reports)
        )
        total_moe_marlin_wna16_source_bytes = int(
            sum(int(report["source_bytes"]) for report in moe_marlin_wna16_reports)
        )
        total_moe_marlin_wna16_released_bytes = int(
            sum(int(report["released_original_bytes"]) for report in moe_marlin_wna16_reports)
        )
        total_pretransposed_bytes = int(
            sum(int(report.get("pretransposed_bytes", 0)) for report in fused_wqa_wkv_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in q_wqb_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in wo_b_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in indexer_wq_b_reports)
            + sum(int(report.get("pretransposed_bytes", 0)) for report in shared_expert_reports)
        )
        projection_cache_owners = []
        if q_wqb_reports:
            projection_cache_owners.append("attn.q_wqb")
        if wo_b_reports:
            projection_cache_owners.append("attn.wo_b")
        if indexer_wq_b_reports:
            projection_cache_owners.append("indexer.wq_b")
        if wo_a_reports:
            projection_cache_owners.append("attn.wo_a")
        if fused_wqa_wkv_reports:
            projection_cache_owners.append("attention WQA/WKV/compress")
        if shared_expert_reports:
            if any(
                str(report["owner"]).endswith("gate_up_proj") for report in shared_expert_reports
            ):
                projection_cache_owners.append("shared_experts.gate_up_proj")
            if any(str(report["owner"]).endswith("down_proj") for report in shared_expert_reports):
                projection_cache_owners.append("shared_experts.down_proj")
        return {
            "attribution_disable_toggles": {
                "env": dsv4_kernel.DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV,
                "raw": os.environ.get(dsv4_kernel.DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV, ""),
                "disabled_toggles": list(dsv4_kernel.dsv4_env_disabled_toggles()),
            },
            "fused_wqa_wkv_bf16_pretranspose_cache": {
                "enabled": bool(fused_wqa_wkv_reports),
                "toggle": dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE,
                "layers_cached": len(fused_wqa_wkv_reports),
                "total_bytes": int(sum(int(report["bytes"]) for report in fused_wqa_wkv_reports)),
                "total_pretransposed_bytes": int(
                    sum(
                        int(report.get("pretransposed_bytes", 0))
                        for report in fused_wqa_wkv_reports
                    )
                ),
                "entries": fused_wqa_wkv_reports,
            },
            "q_wqb_bf16_weight_cache": {
                "enabled": bool(q_wqb_reports),
                "toggle": dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE,
                "layers_cached": len(q_wqb_reports),
                "total_bytes": total_q_wqb_bytes,
                "total_pretransposed_bytes": int(
                    sum(int(report.get("pretransposed_bytes", 0)) for report in q_wqb_reports)
                ),
                "entries": q_wqb_reports,
            },
            "wo_b_bf16_weight_cache": {
                "enabled": bool(wo_b_reports),
                "toggle": dsv4_kernel.DSV4_SM80_WO_B_BF16_WEIGHT_CACHE_TOGGLE,
                "layers_cached": len(wo_b_reports),
                "total_bytes": total_wo_b_bytes,
                "total_pretransposed_bytes": int(
                    sum(int(report.get("pretransposed_bytes", 0)) for report in wo_b_reports)
                ),
                "entries": wo_b_reports,
            },
            "wo_a_bf16_bmm_cache": {
                "enabled": bool(wo_a_reports),
                "toggle": dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE,
                "layers_cached": len(wo_a_reports),
                "total_bytes": total_wo_a_bytes,
                "entries": wo_a_reports,
            },
            "indexer_wq_b_bf16_weight_cache": {
                "enabled": bool(indexer_wq_b_reports),
                "toggle": dsv4_kernel.DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE,
                "layers_cached": len(indexer_wq_b_reports),
                "total_bytes": total_indexer_wq_b_bytes,
                "total_pretransposed_bytes": int(
                    sum(
                        int(report.get("pretransposed_bytes", 0)) for report in indexer_wq_b_reports
                    )
                ),
                "entries": indexer_wq_b_reports,
            },
            "shared_expert_bf16_weight_cache": {
                "enabled": bool(shared_expert_reports),
                "toggle": dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE,
                "layers_cached": max(
                    sum(
                        1
                        for report in shared_expert_reports
                        if str(report["owner"]).endswith("gate_up_proj")
                    ),
                    sum(
                        1
                        for report in shared_expert_reports
                        if str(report["owner"]).endswith("down_proj")
                    ),
                ),
                "total_bytes": total_shared_expert_bytes,
                "total_pretransposed_bytes": int(
                    sum(
                        int(report.get("pretransposed_bytes", 0))
                        for report in shared_expert_reports
                    )
                ),
                "entries": shared_expert_reports,
            },
            "projection_bf16_weight_cache_total": {
                "total_bytes": (
                    total_q_wqb_bytes
                    + total_wo_b_bytes
                    + total_indexer_wq_b_bytes
                    + total_wo_a_bytes
                    + total_shared_expert_bytes
                ),
                "owners": projection_cache_owners,
            },
            "dense_fp8_marlin_projection_cache": {
                "enabled": bool(marlin_reports),
                "backend": "mini_dense_fp8_marlin_w8a16_block",
                "toggle": dsv4_kernel.DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION_TOGGLE,
                "legacy_alias_toggle": dsv4_kernel.DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION_TOGGLE,
                "owners": ["attn.q_wqb", "attn.wo_b", "shared_experts.down_proj"],
                "layers_cached": max(
                    len(q_wqb_marlin_reports),
                    len(wo_b_marlin_reports),
                    len(shared_down_marlin_reports),
                ),
                "total_persistent_bytes": (
                    total_q_wqb_marlin_bytes
                    + total_wo_b_marlin_bytes
                    + total_shared_down_marlin_bytes
                ),
                "total_workspace_bytes": total_marlin_workspace_bytes,
                "total_original_released_bytes": total_marlin_original_released_bytes,
                "duplicate_bf16_cache_for_switched_owners": bool(
                    q_wqb_reports
                    or wo_b_reports
                    or any(
                        str(report["owner"]).endswith("down_proj")
                        for report in shared_expert_reports
                    )
                ),
                "q_wqb": {
                    "layers_cached": len(q_wqb_marlin_reports),
                    "total_persistent_bytes": total_q_wqb_marlin_bytes,
                    "entries": q_wqb_marlin_reports,
                },
                "wo_b": {
                    "layers_cached": len(wo_b_marlin_reports),
                    "total_persistent_bytes": total_wo_b_marlin_bytes,
                    "entries": wo_b_marlin_reports,
                },
                "shared_down": {
                    "layers_cached": len(shared_down_marlin_reports),
                    "total_persistent_bytes": total_shared_down_marlin_bytes,
                    "entries": shared_down_marlin_reports,
                },
            },
            "moe_marlin_wna16_cache": {
                "enabled": bool(moe_marlin_wna16_reports),
                "backend": moe_marlin_backend,
                "prebuild_toggle": dsv4_kernel.DSV4_MARLIN_WNA16_PREBUILD_ENV,
                "prebuild_requested": bool(moe_marlin_prebuild_enabled),
                "release_original_toggle": (
                    dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS_ENV
                ),
                "release_original_requested": bool(moe_marlin_release_original),
                "release_after_graph_capture_requested": bool(
                    moe_marlin_release_after_graph_capture
                ),
                "debug_release_timing": moe_marlin_release_timing,
                "debug_release_deferred_from_model_prepare": bool(moe_marlin_release_deferred),
                "release_layer_filter": os.environ.get(
                    _MARLIN_WNA16_RELEASE_LAYER_FILTER_ENV,
                    "all",
                ),
                "release_weights_only": dsv4_kernel.dsv4_env_flag(
                    _MARLIN_WNA16_RELEASE_WEIGHTS_ONLY_ENV
                ),
                "release_scales_only": dsv4_kernel.dsv4_env_flag(
                    _MARLIN_WNA16_RELEASE_SCALES_ONLY_ENV
                ),
                "keep_hidden_ref": dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_KEEP_HIDDEN_REF_ENV),
                "layers_cached": len(moe_marlin_wna16_reports),
                "total_persistent_bytes": total_moe_marlin_wna16_persistent_bytes,
                "total_source_bytes": total_moe_marlin_wna16_source_bytes,
                "total_released_original_bytes": total_moe_marlin_wna16_released_bytes,
                "release_runtime_policy": (
                    "marlin_wna16_prepacked_only"
                    if (
                        moe_marlin_release_original
                        and not moe_marlin_release_deferred
                        and bool(moe_marlin_wna16_reports)
                    )
                    else None
                ),
                "fail_closed_error": (
                    dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR
                    if moe_marlin_release_original
                    else None
                ),
                "prebuild_entries": moe_marlin_wna16_prebuild_reports,
                "release_entries": moe_marlin_wna16_release_reports,
                "entries": moe_marlin_wna16_reports,
            },
            "bf16_small_gemm_pretranspose_cache_total": {
                "enabled": total_pretransposed_bytes > 0,
                "toggle": dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE,
                "total_pretransposed_bytes": total_pretransposed_bytes,
                "owners": [
                    "attention WQA/WKV/compress",
                    "attn.q_wqb",
                    "attn.wo_b",
                    "indexer.wq_b",
                    "shared_experts.gate_up_proj",
                    "shared_experts.down_proj",
                ],
            },
        }

    def release_marlin_wna16_original_expert_weights(
        self,
        *,
        stage_label: str,
    ) -> dict[str, object]:
        release_reports: list[dict[str, object]] = []
        skipped_layers: list[int] = []
        self.audit_marlin_wna16_cache_integrity(f"{stage_label}:before")
        for layer in self.layers.op_list:
            experts = getattr(layer.mlp, "experts", None)
            layer_id = getattr(experts, "layer_id", None)
            if not _layer_selected_by_env(layer_id, _MARLIN_WNA16_RELEASE_LAYER_FILTER_ENV):
                if layer_id is not None:
                    skipped_layers.append(int(layer_id))
                continue
            release_reports.append(experts.release_marlin_wna16_original_expert_weights())
        poison_then_free_report = self._maybe_poison_then_free_marlin_wna16_released_blocks(
            release_reports,
            stage_label=stage_label,
        )
        quarantine_report = self._maybe_quarantine_marlin_wna16_released_blocks(
            release_reports,
            stage_label=stage_label,
        )
        self.audit_marlin_wna16_cache_integrity(f"{stage_label}:after")

        def _released_this_call(report: dict[str, object]) -> int:
            return int(
                report.get(
                    "released_original_this_call_bytes",
                    report.get("released_original_bytes", 0),
                )
            )

        return {
            "stage_label": stage_label,
            "release_layer_filter": os.environ.get(
                _MARLIN_WNA16_RELEASE_LAYER_FILTER_ENV,
                "all",
            ),
            "entries": release_reports,
            "skipped_layers": skipped_layers,
            "layers_released": len(release_reports),
            "total_released_original_bytes": int(
                sum(int(report["released_original_bytes"]) for report in release_reports)
            ),
            "total_released_original_this_call_bytes": int(
                sum(_released_this_call(report) for report in release_reports)
            ),
            "total_hidden_ref_bytes": int(
                sum(int(report.get("hidden_ref_bytes", 0)) for report in release_reports)
            ),
            "poison_then_free": poison_then_free_report,
            "quarantine": quarantine_report,
        }

    def _maybe_poison_then_free_marlin_wna16_released_blocks(
        self,
        release_reports: list[dict[str, object]],
        *,
        stage_label: str,
    ) -> dict[str, object]:
        if not dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_POISON_THEN_FREE_ENV):
            return {"enabled": False}
        released_items = _marlin_wna16_released_items(release_reports)
        total_released = int(sum(int(item.get("bytes", 0) or 0) for item in released_items))
        target_bytes = _env_bytes(_MARLIN_WNA16_POISON_THEN_FREE_BYTES_ENV, total_released)
        if target_bytes is None:
            target_bytes = total_released
        target_bytes = max(0, min(int(target_bytes), total_released))
        pattern = (
            os.environ.get(
                _MARLIN_WNA16_POISON_THEN_FREE_PATTERN_ENV,
                "nan_byte",
            )
            .strip()
            .lower()
        )
        device = self._marlin_wna16_release_device()
        if device is None or device.type != "cuda":
            return {
                "enabled": True,
                "allocated_bytes": 0,
                "target_bytes": target_bytes,
                "pattern": pattern,
                "error": "no_cuda_device_found",
            }

        poison_tensors: list[torch.Tensor] = []
        poison_tensor: torch.Tensor | None = None
        records: list[dict[str, object]] = []
        allocated = 0
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        try:
            for idx, item in enumerate(released_items):
                if allocated >= target_bytes:
                    break
                requested = min(int(item.get("bytes", 0) or 0), target_bytes - allocated)
                if requested <= 0:
                    continue
                poison_tensor = torch.empty((requested,), dtype=torch.uint8, device=device)
                self._fill_poison_then_free_tensor(poison_tensor, pattern, idx)
                poison_tensors.append(poison_tensor)
                allocated += requested
                owner = (
                    f"marlin_wna16.poison_then_free.layer{item.get('layer_id')}"
                    f".{item.get('component', item.get('attribute', idx))}"
                )
                record = dsv4_memory_debug.record_owner_tensor(
                    owner_label=owner,
                    stage=f"{stage_label}:poison_then_free_allocated",
                    tensor=poison_tensor,
                    include_integrity=True,
                    extra={
                        "pattern": pattern,
                        "source_released_item": item,
                        "poison_index": idx,
                    },
                )
                if record is not None:
                    records.append(record)
            torch.cuda.synchronize(device)
        finally:
            poison_count = len(poison_tensors)
            poison_tensor = None
            poison_tensors.clear()
            torch.cuda.synchronize(device)
        dsv4_memory_debug.append_jsonl(
            "marlin_wna16_poison_then_free",
            {
                "event": "dsv4_marlin_wna16_poison_then_free_released",
                "stage": f"{stage_label}:poison_then_free_released",
                "pattern": pattern,
                "target_bytes": int(target_bytes),
                "allocated_bytes": int(allocated),
                "tensor_count": int(poison_count),
                "source_item_count": len(released_items),
            },
        )
        return {
            "enabled": True,
            "stage_label": stage_label,
            "pattern": pattern,
            "target_bytes": int(target_bytes),
            "allocated_bytes": int(allocated),
            "allocated_gib": allocated / float(1 << 30),
            "tensor_count": int(poison_count),
            "records": records,
        }

    def _maybe_quarantine_marlin_wna16_released_blocks(
        self,
        release_reports: list[dict[str, object]],
        *,
        stage_label: str,
    ) -> dict[str, object]:
        if not dsv4_kernel.dsv4_env_flag(_MARLIN_WNA16_QUARANTINE_BLOCKS_ENV):
            return {"enabled": False}
        released_items = _marlin_wna16_released_items(release_reports)
        total_released = int(sum(int(item.get("bytes", 0) or 0) for item in released_items))
        target_bytes = _env_bytes(_MARLIN_WNA16_QUARANTINE_BYTES_ENV, total_released)
        if target_bytes is None:
            target_bytes = total_released
        target_bytes = max(0, min(int(target_bytes), total_released))
        pattern = os.environ.get(_MARLIN_WNA16_QUARANTINE_PATTERN_ENV, "zero").strip().lower()
        records: list[dict[str, object]] = []
        allocated = 0
        device = self._marlin_wna16_release_device()
        if device is None or device.type != "cuda":
            return {
                "enabled": True,
                "allocated_bytes": 0,
                "target_bytes": target_bytes,
                "error": "no_cuda_device_found",
            }
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        for idx, item in enumerate(released_items):
            if allocated >= target_bytes:
                break
            requested = min(int(item.get("bytes", 0) or 0), target_bytes - allocated)
            if requested <= 0:
                continue
            tensor = torch.empty((requested,), dtype=torch.uint8, device=device)
            self._fill_quarantine_tensor(tensor, pattern, idx)
            self._marlin_wna16_release_quarantine_tensors.append(tensor)
            allocated += requested
            owner = (
                f"marlin_wna16.release_quarantine.layer{item.get('layer_id')}"
                f".{item.get('component', item.get('attribute', idx))}"
            )
            record = dsv4_memory_debug.record_owner_tensor(
                owner_label=owner,
                stage=f"{stage_label}:quarantine",
                tensor=tensor,
                include_integrity=dsv4_memory_debug.env_flag(
                    dsv4_memory_debug.DSV4_MARLIN_WNA16_GUARD_INTEGRITY_ENV
                ),
                extra={
                    "pattern": pattern,
                    "source_released_item": item,
                    "quarantine_index": idx,
                },
            )
            guard_record: dict[str, object] = {
                "owner": owner,
                "stage": f"{stage_label}:quarantine",
                "pattern": pattern,
                "quarantine_index": idx,
                "tensor_index": len(self._marlin_wna16_release_quarantine_tensors) - 1,
                "source_released_item": item,
                "tensor": dsv4_memory_debug.tensor_summary(tensor),
            }
            if dsv4_memory_debug.env_flag(dsv4_memory_debug.DSV4_MARLIN_WNA16_GUARD_INTEGRITY_ENV):
                guard_record["initial_integrity"] = dsv4_memory_debug.tensor_integrity_summary(
                    tensor
                )
            self._marlin_wna16_release_quarantine_records.append(guard_record)
            if record is not None:
                if "initial_integrity" in guard_record:
                    record["initial_integrity"] = guard_record["initial_integrity"]
                records.append(record)
        return {
            "enabled": True,
            "stage_label": stage_label,
            "pattern": pattern,
            "target_bytes": target_bytes,
            "allocated_bytes": allocated,
            "allocated_gib": allocated / float(1 << 30),
            "tensor_count": len(records),
            "records": records,
        }

    def _marlin_wna16_release_device(self) -> torch.device | None:
        for layer in self.layers.op_list:
            experts = getattr(layer.mlp, "experts", None)
            cache = getattr(experts, "_marlin_wna16_weights", None)
            for name in ("w13", "w2", "w13_scale", "w2_scale"):
                tensor = getattr(cache, name, None)
                if isinstance(tensor, torch.Tensor):
                    return tensor.device
        return None

    def _fill_quarantine_tensor(self, tensor: torch.Tensor, pattern: str, index: int) -> None:
        with torch.no_grad():
            if pattern in {"zero", "zeros"}:
                tensor.fill_(0)
            elif pattern in {"one", "ones"}:
                tensor.fill_(1)
            elif pattern in {"index", "deterministic"}:
                tensor.fill_(int(index) % 251)
            elif pattern.startswith("value:"):
                tensor.fill_(int(pattern.split(":", 1)[1], 0) % 256)
            else:
                tensor.fill_(127)

    def _fill_poison_then_free_tensor(
        self,
        tensor: torch.Tensor,
        pattern: str,
        index: int,
    ) -> None:
        with torch.no_grad():
            if pattern in {"zero", "zeros"}:
                tensor.fill_(0)
            elif pattern in {"one", "ones"}:
                tensor.fill_(1)
            elif pattern in {"index", "deterministic"}:
                tensor.fill_((int(index) * 37 + 17) % 251)
            elif pattern in {"nan", "nan_byte", "ff", "0xff"}:
                tensor.fill_(0xFF)
            elif pattern in {"7f", "0x7f"}:
                tensor.fill_(0x7F)
            elif pattern in {"a5", "0xa5"}:
                tensor.fill_(0xA5)
            elif pattern.startswith("value:"):
                tensor.fill_(int(pattern.split(":", 1)[1], 0) % 256)
            else:
                tensor.fill_(0xFF)

    def check_marlin_wna16_release_guards(self, stage: str) -> dict[str, object]:
        enabled = dsv4_memory_debug.env_flag(
            dsv4_memory_debug.DSV4_MARLIN_WNA16_GUARD_INTEGRITY_ENV
        )
        if not enabled or not self._marlin_wna16_release_quarantine_records:
            return {
                "enabled": enabled,
                "stage": stage,
                "guard_count": len(self._marlin_wna16_release_quarantine_records),
                "mutated_count": 0,
            }
        records: list[dict[str, object]] = []
        mutated_count = 0
        for guard in self._marlin_wna16_release_quarantine_records:
            tensor_index = int(guard.get("tensor_index", -1))
            tensor = (
                self._marlin_wna16_release_quarantine_tensors[tensor_index]
                if 0 <= tensor_index < len(self._marlin_wna16_release_quarantine_tensors)
                else None
            )
            current = dsv4_memory_debug.tensor_integrity_summary(tensor)
            initial = guard.get("initial_integrity")
            mutated = False
            if isinstance(initial, dict):
                for key in ("sample_checksum", "finite_ratio", "sample_abs_max"):
                    if current.get(key) != initial.get(key):
                        mutated = True
                        break
            if mutated:
                mutated_count += 1
            record = {
                "event": "dsv4_marlin_wna16_release_guard_check",
                "stage": stage,
                "owner": guard.get("owner"),
                "quarantine_index": guard.get("quarantine_index"),
                "tensor_index": tensor_index,
                "pattern": guard.get("pattern"),
                "mutated": bool(mutated),
                "initial_integrity": initial,
                "current_integrity": current,
                "source_released_item": guard.get("source_released_item"),
            }
            dsv4_memory_debug.append_jsonl("marlin_wna16_release_guards", record)
            records.append(record)
        return {
            "enabled": True,
            "stage": stage,
            "guard_count": len(records),
            "mutated_count": mutated_count,
            "records": records,
        }

    def audit_marlin_wna16_cache_integrity(self, stage: str) -> None:
        if not dsv4_memory_debug.env_flag(
            dsv4_memory_debug.DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG_ENV
        ):
            return
        for layer in self.layers.op_list:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None:
                experts._audit_marlin_wna16_cache_integrity(stage)

    def record_marlin_wna16_owner_allocations(self, stage: str) -> None:
        if not dsv4_memory_debug.marlin_wna16_release_ledger_enabled():
            return
        for layer in self.layers.op_list:
            experts = getattr(layer.mlp, "experts", None)
            if experts is None:
                continue
            dsv4_memory_debug.record_owner_tensors(
                owner_prefix=f"{experts._marlin_owner_label}.packed_cache",
                stage=stage,
                tensors=experts._marlin_cache_tensors(),
                extra={
                    "layer_id": experts.layer_id,
                    "released_original": bool(
                        experts._marlin_wna16_released_original_expert_weights
                    ),
                },
            )
        for idx, tensor in enumerate(self._marlin_wna16_release_quarantine_tensors):
            dsv4_memory_debug.record_owner_tensor(
                owner_label=f"marlin_wna16.release_quarantine.live.{idx}",
                stage=stage,
                tensor=tensor,
                extra={"quarantine_index": idx},
            )

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        hc_head_fn = _cached_hc_bf16_weight(self, "_hc_head_fn_bf16", self.hc_head_fn)
        return dsv4_kernel.hc_head_fallback(
            x,
            hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            eps=self.hc_eps,
            norm_eps=self.norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_hidden_states_before_norm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _record_warmup_memory("model.embed", "before")
        with _dsv4_capture_nvtx("model.embed"):
            x = self.embed_tokens.forward(input_ids)
            _capture_debug_activation("embedding", x)
        _record_warmup_memory("model.embed", "after")
        _record_warmup_memory("model.hc_expand", "before")
        with _dsv4_capture_nvtx("model.hc_expand"):
            x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        _record_warmup_memory("model.hc_expand", "after")
        for layer in self.layers.op_list:
            layer_id = int(getattr(layer.self_attn, "layer_id", -1))
            _record_warmup_memory("decoder_layer", "before", layer_id=layer_id)
            x = layer.forward(x, input_ids)
            _record_warmup_memory("decoder_layer", "after", layer_id=layer_id)
        _record_warmup_memory("model.hc_head", "before")
        hidden_states_before_norm = x.flatten(1)
        _capture_debug_activation("hidden_before_final_norm", hidden_states_before_norm)
        with _dsv4_capture_nvtx("model.hc_head"):
            x = self._hc_head(x)
        _record_warmup_memory("model.hc_head", "after")
        _record_warmup_memory("model.final_norm", "before")
        with _dsv4_capture_nvtx("model.final_norm"):
            x = self.norm.forward(x)
            _capture_debug_activation("final_norm", x)
            _record_warmup_memory("model.final_norm", "after")
            if return_hidden_states_before_norm:
                return x, hidden_states_before_norm
            return x


class DeepseekV4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = DeepseekV4Model(config)
        self.lm_head = DSV4VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        if dsv4_experimental_mtp_enabled():
            self.mtp = DeepseekV4MTPModel(config)
        super().__init__()

    def prepare_for_cuda_graph_capture(self) -> dict[str, object]:
        report = self.model.prepare_for_cuda_graph_capture()
        mtp = getattr(self, "mtp", None)
        if mtp is not None:
            report = dict(report)
            report["mtp"] = mtp.prepare_for_cuda_graph_capture()
        return report

    def release_marlin_wna16_original_expert_weights(
        self,
        *,
        stage_label: str,
    ) -> dict[str, object]:
        return self.model.release_marlin_wna16_original_expert_weights(stage_label=stage_label)

    def check_marlin_wna16_release_guards(self, stage: str) -> dict[str, object]:
        return self.model.check_marlin_wna16_release_guards(stage)

    def audit_marlin_wna16_cache_integrity(self, stage: str) -> None:
        return self.model.audit_marlin_wna16_cache_integrity(stage)

    def record_marlin_wna16_owner_allocations(self, stage: str) -> None:
        return self.model.record_marlin_wna16_owner_allocations(stage)

    def forward(self):
        batch = get_global_ctx().batch
        _record_warmup_memory("model", "before")
        output = self.model.forward(batch.input_ids)
        _record_warmup_memory("model", "after")
        if batch.is_prefill:
            output = output[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        _record_warmup_memory("lm_head", "before")
        with _dsv4_capture_nvtx("lm_head"):
            logits = self.lm_head.linear(output)
            _capture_debug_activation("lm_head_logits", logits)
            _record_warmup_memory("lm_head", "after")
            return logits

    def forward_with_hidden(self) -> DSV4HiddenForwardOutput:
        batch = get_global_ctx().batch
        output, hidden_states_before_norm = self.model.forward(
            batch.input_ids,
            return_hidden_states_before_norm=True,
        )
        if batch.is_prefill:
            last_indices = batch.attn_metadata.get_last_indices(batch.size)
            output = output[last_indices].contiguous()
            hidden_states_before_norm = hidden_states_before_norm[last_indices].contiguous()
        logits = self.lm_head.linear(output)
        _capture_debug_activation("lm_head_logits", logits)
        return DSV4HiddenForwardOutput(
            logits=logits,
            hidden_states=output,
            hidden_states_before_norm=hidden_states_before_norm,
        )

    def mtp_forward_one_step(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> DSV4MTPForwardOutput:
        mtp = getattr(self, "mtp", None)
        if mtp is None:
            raise RuntimeError(
                f"DeepSeek V4 MTP is not enabled. Set {DSV4_EXPERIMENTAL_MTP_ENV}=1 "
                "or pass --enable-dsv4-mtp before constructing the model."
            )
        return mtp.forward_one_step(
            input_ids,
            target_hidden_states,
            embed_tokens=self.model.embed_tokens,
            lm_head=self.lm_head,
        )


__all__ = [
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4MTPModel",
    "DSV4FallbackAttentionMetadata",
    "DSV4AttentionMetadata",
    "DSV4_EXPERIMENTAL_MTP_ENV",
    "dsv4_experimental_mtp_enabled",
]

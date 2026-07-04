from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    dsv4_owner_timing,
    dsv4_prefix_debug,
)

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


def _dsv4_capture_nvtx(name: str):
    if not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_GRAPH_CAPTURE_NVTX"):
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.nvtx.range(f"dsv4.{name}")


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
    dsv4_prefix_debug.capture_dsv4_activation(name, tensor, batch, row_indices=row_indices)


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
    if owner_label.endswith(".shared_experts.down_proj") or ".shared_experts.down_proj" in owner_label:
        return "dsv4.owner.shared_down"
    return f"dsv4.owner.{owner_label}"


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
        if not dsv4_kernel.dsv4_env_flag(
            dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE
        ):
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
        if reduce and self.row_parallel and self._tp_size > 1:
            y = self._comm.all_reduce(
                y,
                label=reduce_label or "dsv4.row_parallel_projection_all_reduce",
            )
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
    ) -> torch.Tensor:
        y = _forward_fp8_marlin_weight(
            self,
            cache_name,
            x,
            owner_label=owner_label,
        )
        if reduce and self.row_parallel and self._tp_size > 1:
            y = self._comm.all_reduce(
                y,
                label=reduce_label or "dsv4.row_parallel_projection_all_reduce",
            )
        return y

    def forward_fp8_cached_bf16_weight(
        self,
        x: torch.Tensor,
        *,
        cache_name: str,
        owner_label: str,
        reduce: bool = False,
        reduce_label: str | None = None,
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
        if not dsv4_owner_timing.enabled():
            x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
            y = _linear_cached_bf16_weight(
                x_quant,
                cached_weight,
                owner=self,
                cache_name=cache_name,
                owner_label=owner_label,
            )
            if reduce and self.row_parallel and self._tp_size > 1:
                y = self._comm.all_reduce(
                    y,
                    label=reduce_label or "dsv4.row_parallel_projection_all_reduce",
                )
            return y
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
            y = _linear_cached_bf16_weight(
                x_quant,
                cached_weight,
                owner=self,
                cache_name=cache_name,
                owner_label=owner_label,
            )
        if reduce and self.row_parallel and self._tp_size > 1:
            y = self._comm.all_reduce(
                y,
                label=reduce_label or "dsv4.row_parallel_projection_all_reduce",
            )
        return y


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
    def __init__(self, config: ModelConfig, layer_id: int):
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
        ratio = config.compress_ratios[layer_id] if layer_id < len(config.compress_ratios) else 0
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
        kv_norm_rope_store_enabled = (
            use_dsv4_backend
            and attn_backend is not None
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
        kv_from_shared_wqa_wkv = None
        kv = None
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_proj"):
            q_wqa_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_Q_WQA_FP8_GEMM"
            )
            fused_wqa_wkv_shared_act = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT"
            )
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
                q_lora_raw = self.wq_a.forward(
                    x,
                    fp8_gemm=q_wqa_fp8_gemm if q_wqa_fp8_gemm else None,
                )
            _capture_debug_activation(f"layer{self.layer_id}.wqa_output", q_lora_raw)
            if kv_from_shared_wqa_wkv is not None:
                _capture_debug_activation(
                    f"layer{self.layer_id}.wkv_shared_activation_output",
                    kv_from_shared_wqa_wkv,
                )
            if not fused_q_kv_rmsnorm:
                q_lora = self.q_norm.forward(q_lora_raw)
                _capture_debug_activation(f"layer{self.layer_id}.q_lora_after_norm", q_lora)
        if fused_q_kv_rmsnorm:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
                _capture_debug_activation(f"layer{self.layer_id}.wkv_output", kv)
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_kv_rmsnorm"):
                q_lora, kv = dsv4_kernel.rms_norm_pair_fallback(
                    q_lora_raw,
                    kv,
                    self.q_norm.weight,
                    self.kv_norm.weight,
                    eps=self.rms_norm_eps,
                )
                _capture_debug_activation(f"layer{self.layer_id}.q_lora_after_norm", q_lora)
                _capture_debug_activation(f"layer{self.layer_id}.kv_after_kv_norm", kv)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_wqb"):
            if dsv4_kernel.dense_fp8_marlin_projection_enabled():
                q = self.wq_b.forward_fp8_marlin_weight(
                    q_lora,
                    cache_name=self._q_wqb_marlin_weight_cache_name,
                    owner_label=self._q_wqb_owner_label,
                ).view(-1, self.num_local_heads, self.head_dim)
            elif dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE):
                q = self.wq_b.forward_fp8_cached_bf16_weight(
                    q_lora,
                    cache_name=self._q_wqb_bf16_weight_cache_name,
                    owner_label=self._q_wqb_owner_label,
                ).view(-1, self.num_local_heads, self.head_dim)
            else:
                q_wqb_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                    "MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM"
                )
                q = self.wq_b.forward(
                    q_lora,
                    fp8_gemm=q_wqb_fp8_gemm if q_wqb_fp8_gemm else None,
                ).view(-1, self.num_local_heads, self.head_dim)
            _capture_debug_activation(f"layer{self.layer_id}.q_wqb_output", q)
        if fused_q_kv_norm_rope_store and kv is None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
                _capture_debug_activation(f"layer{self.layer_id}.wkv_output", kv)
        q_kv_norm_rope_cache_written = False
        if fused_q_kv_norm_rope_store and kv is not None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_kv_norm_rope_store"):
                q_kv_norm_rope_cache_written = dsv4_kernel.q_kv_norm_rope_cache_fallback(
                    q,
                    kv,
                    positions,
                    norm_weight=self.kv_norm.weight,
                    rms_norm_eps=self.rms_norm_eps,
                    cache=attn_backend.kvcache.swa_cache(self.layer_id),
                    out_loc=batch.out_loc,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )
                if q_kv_norm_rope_cache_written:
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
                    dsv4_kernel.k_norm_rope_cache_fallback(
                        kv,
                        positions,
                        norm_weight=self.kv_norm.weight,
                        rms_norm_eps=self.rms_norm_eps,
                        cache=attn_backend.kvcache.swa_cache(self.layer_id),
                        out_loc=batch.out_loc,
                        rotary_dim=self.rope_head_dim,
                        base=float(self.rope_base),
                        original_seq_len=self.original_seq_len,
                        factor=self.rope_factor,
                        beta_fast=self.beta_fast,
                        beta_slow=self.beta_slow,
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
            if use_dsv4_backend and hasattr(attn_backend, "store_indexer"):
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
            if use_dsv4_backend and hasattr(attn_backend, "store_compressed"):
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
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.wo_b"):
            if dsv4_kernel.dense_fp8_marlin_projection_enabled():
                out = self.wo_b.forward_fp8_marlin_weight(
                    o,
                    cache_name=self._wo_b_marlin_weight_cache_name,
                    owner_label=self._wo_b_owner_label,
                    reduce=True,
                    reduce_label="dsv4.attn.wo_b.row_parallel_projection_all_reduce",
                )
                _capture_debug_activation(f"layer{self.layer_id}.final_attention_output", out)
                return out
            if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_B_BF16_WEIGHT_CACHE_TOGGLE):
                out = self.wo_b.forward_fp8_cached_bf16_weight(
                    o,
                    cache_name=self._wo_b_bf16_weight_cache_name,
                    owner_label=self._wo_b_owner_label,
                    reduce=True,
                    reduce_label="dsv4.attn.wo_b.row_parallel_projection_all_reduce",
                )
                _capture_debug_activation(f"layer{self.layer_id}.final_attention_output", out)
                return out
            wo_b_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_WO_B_FP8_GEMM")
            out = self.wo_b.forward(
                o,
                fp8_gemm=wo_b_fp8_gemm if wo_b_fp8_gemm else None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = _cached_gate_fp32_weight(self, "_cached_gate_weight_fp32", self.weight)
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
    def __init__(self, config: ModelConfig):
        tp = get_tp_info()
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
        if backend == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16:
            grouped, self._marlin_wna16_weights = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16(
                hidden_states,
                weights,
                indices,
                self.w13_weight,
                self.w13_weight_scale_inv,
                self.w2_weight,
                self.w2_weight_scale_inv,
                swiglu_limit=self.swiglu_limit,
                cache=self._marlin_wna16_weights,
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
        grouped = dsv4_kernel.moe_route_dispatch_bf16_grouped(
            hidden_states,
            weights,
            indices,
            self.w13_weight,
            self.w13_weight_scale_inv,
            self.w2_weight,
            self.w2_weight_scale_inv,
            swiglu_limit=self.swiglu_limit,
            moe_plan=moe_plan,
            workspace=workspace,
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

    def forward(self, hidden_states: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate.forward(
            flat,
            input_ids=input_ids,
            topk=self.topk_count,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            hash_topk=hash_topk,
        )

    def prepare(
        self,
        flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> DSV4FusedMoERunnerPrepareResult:
        moe_plan = dsv4_kernel.build_moe_v2_execution_plan(
            flat,
            weights,
            indices,
            num_experts=self.experts.w13_weight.shape[0],
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

    def finalize_routed(self, routed_output: torch.Tensor) -> torch.Tensor:
        # The current grouped FP4 backend already applies top-k weights and
        # sums routes to [tokens, hidden]. Keep the boundary explicit so a
        # future exact backend can return per-route output here.
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.runner_finalize_to_fp32.layer{self.layer_id}",
            routed_output=routed_output,
        ):
            return routed_output.float()

    def apply_shared(self, flat: torch.Tensor) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        shared = self.shared_experts.forward(flat, reduce=False)
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.runner_shared_to_fp32.layer{self.layer_id}",
            shared=shared,
        ):
            return shared.float()

    def maybe_reduce_final(
        self,
        output: torch.Tensor,
        *,
        comm: DistributedCommunicator,
        reduce_label: str,
    ) -> torch.Tensor:
        if self._tp_size > 1:
            return comm.all_reduce(output, label=reduce_label)
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
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.route"):
            weights, indices = self.route(flat, flat_input_ids, hash_topk=hash_topk)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.prepare"):
            prepared = self.prepare(flat, weights, indices)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.experts"):
            routed = self.apply_experts(flat, prepared)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.finalize"):
            y = self.finalize_routed(routed)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.shared"):
            shared = self.apply_shared(flat)
            if shared is not None:
                y = y + shared
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.runner.reduce_once"):
            y = self.maybe_reduce_final(
                y,
                comm=comm,
                reduce_label=prepared.moe_plan.final_reduce_label,
            )
        with dsv4_direct_copy_nvtx(
            f"moe_shared_expert_staging.runner_output_to_flat_dtype.layer{self.layer_id}",
            y=y,
        ):
            return y.to(flat.dtype).view_as(hidden_states)


class DSV4MoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp = get_tp_info()
        self.layer_id = layer_id
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        is_hash_layer = layer_id < config.n_hash_layers
        self.topk_count = config.num_experts_per_tok
        self.scoring_func = config.scoring_func or "sqrtsoftplus"
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = DSV4MoEGate(config, has_correction_bias=not is_hash_layer)
        if is_hash_layer:
            self.topk = DSV4TopK(config)
        self.experts = DSV4FusedRoutedExperts(config)
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
        if dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE):
            return self._runner.forward(
                hidden_states,
                input_ids,
                comm=self._comm,
                hash_topk=getattr(self, "topk", None),
            )

        flat = hidden_states.view(-1, hidden_states.shape[-1])
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.gate"):
            weights, indices = self.gate.forward(
                flat,
                input_ids=input_ids.view(-1),
                topk=self.topk_count,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                hash_topk=getattr(self, "topk", None),
            )
        moe_v2 = dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE)
        reduce_once = moe_v2 or dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE)
        moe_plan = None
        if moe_v2:
            moe_plan = dsv4_kernel.build_moe_v2_execution_plan(
                flat,
                weights,
                indices,
                num_experts=self.experts.w13_weight.shape[0],
                block_size_m=16,
                reduce_once=reduce_once,
            )
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.routed"):
            if moe_plan is None:
                y = self.experts.forward(flat, weights, indices, reduce=not reduce_once).float()
            else:
                y = self.experts.forward(
                    flat,
                    weights,
                    indices,
                    reduce=not reduce_once,
                    moe_plan=moe_plan,
                ).float()
        if hasattr(self, "shared_experts"):
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.shared"):
                y = y + self.shared_experts.forward(flat, reduce=not reduce_once).float()
        if reduce_once and self._tp_size > 1:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.reduce_once"):
                y = self._comm.all_reduce(y, label="dsv4.v1_moe_reduce_once_all_reduce")
        return y.to(flat.dtype).view_as(hidden_states)


class DeepseekV4DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.self_attn = DSV4Attention(config, layer_id)
        self.mlp = DSV4MoE(config, layer_id)
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
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_attn_pre"):
            y, post, comb = self._hc_pre(x, attn_fn, self.hc_attn_scale, self.hc_attn_base)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.attn_input_norm"):
            y = self.input_layernorm.forward(y)
            _capture_debug_activation(f"layer{layer_id}.attention_input", y)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.attn"):
            y = self.self_attn.forward(y)
            _capture_debug_activation(f"layer{layer_id}.attention_output", y)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_attn_post"):
            x = self._hc_post(y, residual, post, comb)

        residual = x
        ffn_fn = _cached_hc_bf16_weight(self, "_hc_ffn_fn_bf16", self.hc_ffn_fn)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_ffn_pre"):
            y, post, comb = self._hc_pre(x, ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.mlp_input_norm"):
            y = self.post_attention_layernorm.forward(y)
            _capture_debug_activation(f"layer{layer_id}.moe_input", y)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.mlp"):
            y = self.mlp.forward(y, input_ids)
            _capture_debug_activation(f"layer{layer_id}.moe_output", y)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_ffn_post"):
            return self._hc_post(y, residual, post, comb)


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
                "raw": os.environ.get(
                    dsv4_kernel.DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV, ""
                ),
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with _dsv4_capture_nvtx("model.embed"):
            x = self.embed_tokens.forward(input_ids)
            _capture_debug_activation("embedding", x)
        with _dsv4_capture_nvtx("model.hc_expand"):
            x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers.op_list:
            x = layer.forward(x, input_ids)
        with _dsv4_capture_nvtx("model.hc_head"):
            x = self._hc_head(x)
        with _dsv4_capture_nvtx("model.final_norm"):
            x = self.norm.forward(x)
            _capture_debug_activation("final_norm", x)
            return x


class DeepseekV4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = DeepseekV4Model(config)
        self.lm_head = DSV4VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        super().__init__()

    def prepare_for_cuda_graph_capture(self) -> dict[str, object]:
        return self.model.prepare_for_cuda_graph_capture()

    def forward(self):
        batch = get_global_ctx().batch
        output = self.model.forward(batch.input_ids)
        if batch.is_prefill:
            output = output[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        with _dsv4_capture_nvtx("lm_head"):
            logits = self.lm_head.linear(output)
            _capture_debug_activation("lm_head_logits", logits)
            return logits


__all__ = ["DeepseekV4ForCausalLM", "DSV4FallbackAttentionMetadata", "DSV4AttentionMetadata"]

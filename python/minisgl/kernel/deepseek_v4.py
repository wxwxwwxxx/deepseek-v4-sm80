"""DeepSeek V4 fused-kernel wrapper boundary.

The functions in this module are correctness-first fallbacks with the same
semantic boundaries as SGLang's DSV4 fused kernels.  High-performance sm80
ports should replace internals here instead of leaking optional kernel imports
into model or attention code.
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from minisgl.dsv4_runtime import get_dsv4_runtime_config
from minisgl.kernel.utils import load_jit
from minisgl.utils import div_ceil

WeightKind = Literal["bf16", "fp8", "fp4"]
KernelStatus = Literal["native", "fallback", "unsupported", "todo"]
DSV4KernelMode = Literal["fallback", "bf16_direct", "fp8_act", "fp4_act"]


DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4 = "grouped_fp4"
DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16 = "marlin_wna16"
DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT = 512





DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR = (
    "Marlin WNA16 release preset has released raw routed expert weights; "
    "fallback/grouped_fp4 backend is unavailable in this Engine. Use the "
    "non-release preset or recreate the Engine with release disabled."
)
DSV4_INDEXER_CAPTURE_WIDTH_MODES = ("current", "table_width", "seq_len_aligned")
DSV4_SM80_MOE_EXPERT_BACKENDS: tuple[str, ...] = (
    DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4,
    DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16,
)
DSV4_SM80_MOE_V2_WORKSPACE_MAX_ROUTES = 512


@dataclass(frozen=True)
class DSV4PagedMQAMetadata:
    indptr: torch.Tensor
    indices: torch.Tensor
    lengths: torch.Tensor
    max_length: int

    @property
    def row_count(self) -> int:
        return int(self.lengths.numel())


@dataclass(frozen=True)
class DSV4TwoSourceAttentionMetadata:
    compressed_indices: torch.Tensor
    compressed_lengths: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.swa_lengths.numel())


@dataclass(frozen=True)
class DSV4TopKTransformOutput:
    raw_indices: torch.Tensor
    page_indices: torch.Tensor
    full_indices: torch.Tensor
    backend: str
    topk_lens: torch.Tensor | None = None


@dataclass(frozen=True)
class DSV4IndexerSelectOutput:
    logits: torch.Tensor
    topk: DSV4TopKTransformOutput
    backend: str


@dataclass(frozen=True)
class DSV4IndexerFP8Query:
    q_values: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class DSV4KernelCapability:
    cuda_available: bool
    cuda_capability: tuple[int, int] | None
    is_sm80: bool
    sgl_kernel_available: bool
    sgl_kernel_error: str | None
    sgl_kernel_sm80_common_ops: bool
    sgl_kernel_dsv4_ops: dict[str, bool]
    flash_mla_available: bool
    flash_mla_error: str | None
    flashinfer_available: bool
    flashinfer_error: str | None
    deep_gemm_available: bool
    deep_gemm_error: str | None
    deep_gemm_usable: bool
    tilelang_available: bool
    tilelang_error: str | None
    triton_available: bool
    triton_error: str | None
    marlin_available: bool
    marlin_error: str | None


@dataclass(frozen=True)
class DSV4MoERoutePlan:
    sorted_route_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    route_count: int
    topk: int
    block_size_m: int


@dataclass(frozen=True)
class DSV4MoEExecutionPlan:
    route_plan: DSV4MoERoutePlan
    route_weights: torch.Tensor
    tokens: int
    hidden: int
    num_experts: int
    reduce_once: bool
    final_reduce_label: str


class DSV4MoEWorkspace:
    """Reusable per-layer temporary buffers for the exact grouped MoE path."""

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        *,
        zero: bool = False,
    ) -> torch.Tensor:
        numel = math.prod(shape)
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.dtype != dtype
            or buffer.device != device
            or buffer.numel() < numel
        ):
            buffer = torch.empty((numel,), dtype=dtype, device=device)
            self._buffers[name] = buffer
        out = buffer[:numel].view(shape)
        if zero:
            out.zero_()
        return out


def _module_available(name: str) -> tuple[bool, str | None]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:  # pragma: no cover - optional packages can fail in __init__.
        return False, f"{type(exc).__name__}: {exc}"
    if spec is None:
        return False, "module not installed"
    try:
        __import__(name)
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _cuda_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - defensive for unusual CUDA setups.
        return None


def _has_sgl_common_ops_for_sm80() -> bool:
    try:
        import sgl_kernel  # type: ignore

        package_paths = list(getattr(sgl_kernel, "__path__", []))
    except Exception:
        return False
    return any(
        os.path.exists(os.path.join(package_path, "sm80", "common_ops.abi3.so"))
        for package_path in package_paths
    )


@lru_cache(maxsize=1)
def detect_dsv4_kernel_capabilities() -> DSV4KernelCapability:
    cap = _cuda_capability()
    is_sm80 = cap == (8, 0)

    sgl_ok, sgl_err = _module_available("sgl_kernel")
    dsv4_ops = {
        name: bool(sgl_ok and hasattr(torch.ops.sgl_kernel, name))
        for name in (
            "deepseek_v4_topk_transform_512",
            "dsv4_fused_q_indexer_rope_hadamard_quant",
            "dsv4_fused_q_indexer_rope_hadamard_fp4_quant",
        )
    }

    flash_mla_ok, flash_mla_err = _module_available("sgl_kernel.flash_mla")
    flashinfer_ok, flashinfer_err = _module_available("flashinfer")
    deep_gemm_ok, deep_gemm_err = _module_available("deep_gemm")
    tilelang_ok, tilelang_err = _module_available("tilelang")
    triton_ok, triton_err = _module_available("triton")
    marlin_ok, marlin_err = _module_available("marlin")

    deep_gemm_usable = bool(deep_gemm_ok and cap is not None and cap[0] >= 9)
    return DSV4KernelCapability(
        cuda_available=torch.cuda.is_available(),
        cuda_capability=cap,
        is_sm80=is_sm80,
        sgl_kernel_available=sgl_ok,
        sgl_kernel_error=sgl_err,
        sgl_kernel_sm80_common_ops=_has_sgl_common_ops_for_sm80(),
        sgl_kernel_dsv4_ops=dsv4_ops,
        flash_mla_available=flash_mla_ok,
        flash_mla_error=flash_mla_err,
        flashinfer_available=flashinfer_ok,
        flashinfer_error=flashinfer_err,
        deep_gemm_available=deep_gemm_ok,
        deep_gemm_error=deep_gemm_err,
        deep_gemm_usable=deep_gemm_usable,
        tilelang_available=tilelang_ok,
        tilelang_error=tilelang_err,
        triton_available=triton_ok,
        triton_error=triton_err,
        marlin_available=marlin_ok,
        marlin_error=marlin_err,
    )






def dsv4_optimized_enabled() -> bool:
    return get_dsv4_runtime_config().optimized


def dsv4_optimized_triton_enabled() -> bool:
    if not dsv4_optimized_enabled():
        return False
    cap = detect_dsv4_kernel_capabilities()
    return bool(cap.is_sm80 and cap.triton_available)




def warmup_indexer_fp8_backend(device: torch.device) -> None:
    if not dsv4_optimized_triton_enabled():
        return
    if torch.device(device).type != "cuda":
        return
    warmup = getattr(_triton_dsv4_ops(), "warmup_indexer_fp8_lut", None)
    if callable(warmup):
        warmup(device)


def dsv4_moe_expert_backend() -> str:
    return get_dsv4_runtime_config().moe_expert_backend


def require_supported_moe_expert_backend() -> str:
    backend = dsv4_moe_expert_backend()
    if backend not in DSV4_SM80_MOE_EXPERT_BACKENDS:
        raise RuntimeError(f"Unsupported typed DSV4 MoE backend: {backend!r}")
    return backend


def moe_route_dispatch_bf16_marlin_wna16(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    cache=None,
    owner_label: str | None = None,
    moe_plan: DSV4MoEExecutionPlan | None = None,
) -> tuple[torch.Tensor, object]:
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Marlin WNA16 expects fp16/bf16 hidden states, got {hidden_states.dtype}")
    if not hidden_states.is_cuda:
        raise ValueError("Marlin WNA16 requires CUDA hidden states")
    if hidden_states.ndim != 2 or weights.shape != indices.shape or indices.ndim != 2:
        raise ValueError(
            "Marlin WNA16 expects hidden [tokens, hidden] and matching weights/indices [tokens, topk]"
        )
    if w13_weight.ndim != 4 or w13_weight.shape[1] != 2 or w2_weight.ndim != 3:
        raise ValueError("Marlin WNA16 expects mini raw w13 [E,2,N,K/2] and w2 [E,K,N/2]")
    if w13_weight.shape[0] != w2_weight.shape[0]:
        raise ValueError("Marlin WNA16 w13/w2 expert counts differ")
    if hidden_states.shape[1] != w13_weight.shape[-1] * 2:
        raise ValueError("Marlin WNA16 hidden size does not match packed W13")

    from minisgl.kernel import marlin_wna16

    cache_was_present = cache is not None
    cache_signature_match = bool(
        cache is not None and cache.matches(w13_weight, w13_scale, w2_weight, w2_scale)
    )
    if not cache_signature_match:
        cache = marlin_wna16.prepare_moe_mxfp4_weights(
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            params_dtype=hidden_states.dtype,
            owner_label=owner_label,
            cache_was_present=cache_was_present,
            cache_signature_match=cache_signature_match,
        )
    output = _run_moe_bf16_marlin_wna16_prepacked(
        hidden_states,
        weights,
        indices,
        cache,
        swiglu_limit=swiglu_limit,
        moe_plan=moe_plan,
    )
    return output, cache


def moe_route_dispatch_bf16_marlin_wna16_prepacked(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    cache,
    *,
    swiglu_limit: float = 0.0,
    moe_plan: DSV4MoEExecutionPlan | None = None,
) -> torch.Tensor:
    return _run_moe_bf16_marlin_wna16_prepacked(
        hidden_states,
        weights,
        indices,
        cache,
        swiglu_limit=swiglu_limit,
        moe_plan=moe_plan,
    )


def _run_moe_bf16_marlin_wna16_prepacked(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    cache,
    *,
    swiglu_limit: float,
    moe_plan: DSV4MoEExecutionPlan | None,
) -> torch.Tensor:
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Marlin WNA16 expects fp16/bf16 hidden states, got {hidden_states.dtype}")
    if not hidden_states.is_cuda:
        raise ValueError("Marlin WNA16 requires CUDA hidden states")
    if hidden_states.ndim != 2 or weights.shape != indices.shape or indices.ndim != 2:
        raise ValueError(
            "Marlin WNA16 expects hidden [tokens, hidden] and matching weights/indices [tokens, topk]"
        )
    if cache is None:
        raise RuntimeError("Marlin WNA16 prepacked dispatch requires a prepared weight cache.")
    if not all(hasattr(cache, name) for name in ("w13", "w2", "w13_scale", "w2_scale")):
        raise RuntimeError("Marlin WNA16 prepacked dispatch received an invalid cache object.")

    from minisgl.kernel import marlin_wna16

    experts = cache.w13.shape[0]
    if moe_plan is None:
        block_size_m = marlin_wna16.choose_block_size(
            tokens=hidden_states.shape[0],
            topk=indices.shape[1],
            experts=experts,
            input_dtype=None,
        )
        route_plan = build_moe_route_plan(
            indices,
            num_experts=experts,
            block_size_m=block_size_m,
        )
        topk_weights = weights
    else:
        if (
            moe_plan.tokens != hidden_states.shape[0]
            or moe_plan.hidden != hidden_states.shape[1]
            or moe_plan.num_experts != experts
            or moe_plan.route_plan.route_count != indices.numel()
            or moe_plan.route_plan.topk != indices.shape[1]
        ):
            raise ValueError("Marlin WNA16 received an incompatible DSV4 MoE execution plan")
        route_plan = moe_plan.route_plan
        topk_weights = moe_plan.route_weights.view(hidden_states.shape[0], indices.shape[1])
    output = marlin_wna16.run_moe(
        hidden_states,
        topk_weights,
        cache,
        sorted_token_ids=route_plan.sorted_route_ids,
        expert_ids=route_plan.expert_ids,
        num_tokens_post_padded=route_plan.num_tokens_post_padded,
        block_size_m=route_plan.block_size_m,
        swiglu_limit=swiglu_limit,
    )
    return output


def dsv4_optimized_cuda_enabled() -> bool:
    if not dsv4_optimized_enabled():
        return False
    cap = detect_dsv4_kernel_capabilities()
    return bool(cap.is_sm80 and cap.cuda_available)


def linear_bf16_fp32_upstream_enabled() -> bool:
    return dsv4_optimized_cuda_enabled()




def _triton_dsv4_ops():
    from minisgl.kernel.triton import deepseek_v4 as triton_dsv4

    return triton_dsv4


def fp8_dtype() -> torch.dtype:
    return getattr(torch, "float8_e4m3fn", torch.uint8)


def e8m0_dtype() -> torch.dtype:
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def scale_dim(size: int, block_size: int = 128) -> int:
    return div_ceil(size, block_size)


_FP4_TABLE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _fp4_table(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    table = _FP4_TABLE_CACHE.get(key)
    if table is None:
        table = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        _FP4_TABLE_CACHE[key] = table
    return table


def dequant_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    w = weight.float()
    if scale is None:
        return w.to(out_dtype)
    out_features, in_features = w.shape
    expanded = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expanded = expanded[:out_features, :in_features]
    return (w * expanded).to(out_dtype)


def dequant_fp4_weight(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    packed = weight.contiguous().view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    table = _fp4_table(weight.device)
    unpacked = torch.stack((table[low.long()], table[high.long()]), dim=-1).flatten(-2)
    if scale is None:
        return unpacked.to(out_dtype)
    expanded = scale.float().repeat_interleave(32, dim=-1)
    expanded = expanded[..., : unpacked.shape[-1]]
    return (unpacked * expanded).to(out_dtype)


def quantize_fp8_activation_ref(x: torch.Tensor, *, block_size: int = 128) -> torch.Tensor:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or x.numel() == 0 or x.shape[-1] % block_size != 0:
        return x
    if dsv4_optimized_triton_enabled():
        try:
            y = _triton_dsv4_ops().fp8_activation_quantize(x, block_size=block_size)
            if y is not None:
                return y
        except Exception as exc:
            if _cuda_graph_capture_active(x.device):
                raise RuntimeError(
                    "DSV4 CUDA graph capture failed in Triton FP8 activation quant."
                ) from exc
    dtype = x.dtype
    flat = x.contiguous().view(-1, x.shape[-1]).float()
    groups = flat.view(flat.shape[0], flat.shape[1] // block_size, block_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    y = (groups / scale).clamp(-448.0, 448.0).to(fp8).float() * scale
    return y.reshape_as(flat).reshape_as(x).to(dtype)


def quantize_indexer_fp8_cache_ref(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        raise RuntimeError("torch.float8_e4m3fn is required for FP8 indexer cache")
    if kv.ndim < 2:
        raise ValueError(f"DSV4 FP8 indexer quant expects [..., dim], got {kv.shape}")
    flat = kv.contiguous().view(-1, kv.shape[-1]).to(torch.bfloat16).to(torch.float32)
    amax = flat.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0))).to(torch.float32)
    values = (flat / scale).clamp(-448.0, 448.0).to(fp8).view(torch.uint8)
    scale_bytes = scale.contiguous().view(torch.uint8).view(flat.shape[0], 4)
    return (
        values.view(*kv.shape[:-1], kv.shape[-1]).contiguous(),
        scale_bytes.view(*kv.shape[:-1], 4).contiguous(),
    )


def dequantize_indexer_fp8_cache_ref(
    values: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        raise RuntimeError("torch.float8_e4m3fn is required for FP8 indexer cache")
    if values.dtype is not torch.uint8 or scales.dtype is not torch.uint8:
        raise ValueError("DSV4 FP8 indexer cache values/scales must be uint8 tensors")
    if scales.shape[:-1] != values.shape[:-1] or scales.shape[-1] != 4:
        raise ValueError(
            "DSV4 FP8 indexer scales must have shape values.shape[:-1] + (4,), "
            f"got values={tuple(values.shape)} scales={tuple(scales.shape)}"
        )
    scale = scales.contiguous().view(torch.float32).view(*values.shape[:-1], 1)
    return (values.contiguous().view(fp8).to(torch.float32) * scale).to(out_dtype)


def pack_indexer_fp8_paged_cache_ref(
    values: torch.Tensor,
    scales: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    if values.ndim != 2 or scales.ndim != 2:
        raise ValueError(
            "DSV4 paged FP8 indexer pack expects values [slots, dim] and scales [slots, 4], "
            f"got values={tuple(values.shape)} scales={tuple(scales.shape)}"
        )
    if values.dtype is not torch.uint8 or scales.dtype is not torch.uint8:
        raise ValueError("DSV4 paged FP8 indexer values/scales must be uint8 tensors")
    if scales.shape != (values.shape[0], 4):
        raise ValueError(
            "DSV4 paged FP8 indexer scales must be [slots, 4], "
            f"got values={tuple(values.shape)} scales={tuple(scales.shape)}"
        )
    if page_size <= 0:
        raise ValueError(f"DSV4 paged FP8 indexer page_size must be positive, got {page_size}")

    slots, dim = values.shape
    pages = div_ceil(slots, page_size)
    packed = torch.zeros(
        (pages, page_size * (dim + 4)),
        dtype=torch.uint8,
        device=values.device,
    )
    page_bytes = page_size * (dim + 4)
    data = packed.as_strided((pages, page_size, dim), (page_bytes, dim, 1))
    scale_region = packed.as_strided(
        (pages, page_size, 4),
        (page_bytes, 4, 1),
        storage_offset=page_size * dim,
    )
    padded_values = torch.zeros((pages * page_size, dim), dtype=torch.uint8, device=values.device)
    padded_scales = torch.zeros((pages * page_size, 4), dtype=torch.uint8, device=values.device)
    padded_values[:slots] = values.contiguous()
    padded_scales[:slots] = scales.contiguous()
    data.copy_(padded_values.view(pages, page_size, dim))
    scale_region.copy_(padded_scales.view(pages, page_size, 4))
    return packed


def quantize_indexer_fp8_paged_cache_ref(
    kv: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    values, scales = quantize_indexer_fp8_cache_ref(kv)
    return pack_indexer_fp8_paged_cache_ref(
        values.view(-1, values.shape[-1]),
        scales.view(-1, 4),
        page_size=page_size,
    )


def dequantize_indexer_fp8_paged_cache_ref(
    packed_cache: torch.Tensor,
    *,
    page_size: int,
    dim: int,
    slots: int | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if packed_cache.ndim != 2 or packed_cache.dtype is not torch.uint8:
        raise ValueError(
            "DSV4 paged FP8 indexer cache must be a uint8 [pages, page_bytes] tensor, "
            f"got {tuple(packed_cache.shape)} {packed_cache.dtype}"
        )
    if page_size <= 0 or dim <= 0:
        raise ValueError(
            f"DSV4 paged FP8 dequant expects positive page_size/dim, got {page_size}/{dim}"
        )
    if packed_cache.shape[-1] != page_size * (dim + 4):
        raise ValueError(
            "DSV4 paged FP8 indexer cache page byte mismatch: "
            f"got {packed_cache.shape[-1]}, expected {page_size * (dim + 4)}"
        )
    pages = packed_cache.shape[0]
    page_bytes = page_size * (dim + 4)
    values = packed_cache.as_strided((pages, page_size, dim), (page_bytes, dim, 1)).reshape(
        pages * page_size, dim
    )
    scales = packed_cache.as_strided(
        (pages, page_size, 4),
        (page_bytes, 4, 1),
        storage_offset=page_size * dim,
    ).reshape(pages * page_size, 4)
    if slots is not None:
        values = values[:slots]
        scales = scales[:slots]
    return dequantize_indexer_fp8_cache_ref(values, scales, out_dtype=out_dtype)


def indexer_q_rope_fp8_fallback(
    q: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    softmax_scale: float,
    head_scale: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> DSV4IndexerFP8Query:
    if q.ndim != 3:
        raise ValueError(f"DSV4 FP8 indexer q expects [tokens, heads, dim], got {q.shape}")
    if weights.shape[:2] != q.shape[:2]:
        raise ValueError(
            "DSV4 FP8 indexer weights must match q [tokens, heads], "
            f"got weights={tuple(weights.shape)} q={tuple(q.shape)}"
        )
    q_work = q.contiguous()
    apply_rotary_tail(
        q_work,
        positions,
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    q_values = None
    weights_out = None
    if dsv4_optimized_triton_enabled():
        try:
            triton_quant = _triton_dsv4_ops().indexer_fp8_quantize_fold(
                q_work,
                weights,
                softmax_scale=float(softmax_scale),
                head_scale=float(head_scale),
            )
            if triton_quant is not None:
                q_values, weights_out = triton_quant
        except Exception as exc:
            if _cuda_graph_capture_active(q_work.device):
                raise RuntimeError(
                    "DSV4 CUDA graph capture failed in FP8 indexer Q quantize."
                ) from exc
    if q_values is None or weights_out is None:
        if _cuda_graph_capture_active(q_work.device):
            raise RuntimeError(
                "DSV4 CUDA graph capture requires the Triton FP8 indexer Q quantize path."
            )
        q_values, q_scale_bytes = quantize_indexer_fp8_cache_ref(q_work)
        q_scale = q_scale_bytes.contiguous().view(torch.float32).view(*q.shape[:2])
        weights_out = (
            weights.squeeze(-1).to(device=q.device, dtype=torch.float32)
            * q_scale
            * float(softmax_scale)
            * float(head_scale)
        )
    return DSV4IndexerFP8Query(q_values=q_values.contiguous(), weights=weights_out.contiguous())


def quantized_linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    weight_kind: WeightKind,
) -> torch.Tensor:
    if weight_kind == "fp4":
        x = quantize_fp8_activation_ref(x)
        w = dequant_fp4_weight(weight, scale, out_dtype=x.dtype)
    elif weight_kind == "fp8":
        x = quantize_fp8_activation_ref(x)
        w = dequant_fp8_weight(weight, scale, out_dtype=x.dtype)
    else:
        w = weight.to(x.dtype)
    return F.linear(x, w)


def quantized_linear_fp8_pair_shared_activation_ref(
    x: torch.Tensor,
    weight_a: torch.Tensor,
    scale_a: torch.Tensor | None,
    weight_b: torch.Tensor,
    scale_b: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_quant = quantize_fp8_activation_ref(x)
    w_a = dequant_fp8_weight(weight_a, scale_a, out_dtype=x_quant.dtype)
    w_b = dequant_fp8_weight(weight_b, scale_b, out_dtype=x_quant.dtype)
    return F.linear(x_quant, w_a), F.linear(x_quant, w_b)


def _linear_bf16_fp32_upstream(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out_shape = tuple(x.shape[:-1]) + (weight.shape[0],)
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    weight_t = weight.contiguous().t()
    return torch.mm(x_2d, weight_t, out_dtype=torch.float32).reshape(out_shape)


def linear_bf16_fp32_fallback(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        linear_bf16_fp32_upstream_enabled()
        and x.is_cuda
        and weight.is_cuda
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and x.shape[-1] == weight.shape[-1]
    ):
        return _linear_bf16_fp32_upstream(x, weight)
    return F.linear(x.float(), weight.float())


def rms_norm_fallback(x: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    if dsv4_optimized_triton_enabled():
        y = _triton_dsv4_ops().rms_norm_bf16(x, weight, eps=eps)
        if y is not None:
            return y
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(dtype)




def _compress_forward_vectorized(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    ratio: int,
    head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    wkv_gate,
    norm,
    apply_norm: bool,
) -> torch.Tensor | None:
    if ratio <= 0:
        return None
    positions = positions.to(device=x.device, dtype=torch.long)
    end_indices = torch.nonzero((positions + 1) % ratio == 0, as_tuple=False).flatten()
    if end_indices.numel() == 0:
        return x.new_empty((0, head_dim))
    offsets = torch.arange(ratio, dtype=torch.long, device=x.device)
    gather = end_indices[:, None] - (ratio - 1) + offsets[None, :]
    valid = gather[:, 0] >= 0
    if bool(torch.any(valid)):
        gather_valid = gather[valid]
        expected = positions[end_indices[valid]][:, None] - (ratio - 1) + offsets[None, :]
        contiguous = torch.all(positions[gather_valid] == expected, dim=1)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()[contiguous]
        gather = gather[valid_indices]
    else:
        gather = gather[:0]
    if gather.numel() == 0:
        return x.new_empty((0, head_dim))

    flat_indices = gather.reshape(-1)
    projected = wkv_gate.forward(x.index_select(0, flat_indices)).float()
    kv, score = projected.chunk(2, dim=-1)
    slot = (positions.index_select(0, flat_indices) % ratio).to(torch.long)
    score = score + ape[slot].float()
    kv = kv.view(-1, ratio, kv.shape[-1])
    score = score.view(-1, ratio, score.shape[-1])
    if overlap:
        if kv.shape[-1] != 2 * head_dim or score.shape[-1] != 2 * head_dim:
            return None
        kv = torch.cat([kv[..., :head_dim], kv[..., head_dim:]], dim=1)
        score = torch.cat([score[..., :head_dim], score[..., head_dim:]], dim=1)
    else:
        if kv.shape[-1] != head_dim or score.shape[-1] != head_dim:
            return None
    pooled = (kv * score.softmax(dim=1)).sum(dim=1).to(x.dtype)
    return norm.forward(pooled) if apply_norm else pooled


def apply_rotary_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    inverse: bool = False,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    if rotary_dim <= 0:
        return x
    if rotary_dim % 2 != 0:
        raise ValueError(f"DeepSeek V4 rotary_dim must be even, got {rotary_dim}")
    if dsv4_optimized_triton_enabled():
        try:
            if _triton_dsv4_ops().apply_rotary_tail(
                x,
                positions,
                rotary_dim=rotary_dim,
                base=base,
                inverse=inverse,
                original_seq_len=original_seq_len,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            ):
                return x
        except Exception:
            pass

    pos = positions.to(device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device) / rotary_dim)
    )
    if original_seq_len > 0:

        def correction_dim(num_rotations: float) -> float:
            return (
                rotary_dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rotary_dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(rotary_dim // 2, dtype=torch.float32, device=x.device) - low)
            / max(high - low, 1),
            0,
            1,
        )
        smooth = 1 - ramp
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

    freqs = torch.outer(pos, inv_freq)
    if inverse:
        freqs = -freqs
    cos = freqs.cos()
    sin = freqs.sin()
    while cos.ndim < x[..., -rotary_dim:].ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    rope = x[..., -rotary_dim:].float().unflatten(-1, (-1, 2))
    a, b = rope[..., 0], rope[..., 1]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    x[..., -rotary_dim:] = rotated.to(x.dtype)
    return x


def q_norm_rope_fallback(
    q: torch.Tensor,
    positions: torch.Tensor,
    *,
    rms_norm_eps: float,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    if dsv4_optimized_triton_enabled():
        try:
            if _triton_dsv4_ops().q_norm_rope(
                q,
                positions,
                rms_norm_eps=rms_norm_eps,
                rotary_dim=rotary_dim,
                base=base,
                original_seq_len=original_seq_len,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            ):
                return q
        except Exception:
            pass
    q_fp32 = q.float()
    scale = torch.rsqrt(q_fp32.square().mean(-1, keepdim=True) + rms_norm_eps)
    q.copy_((q_fp32 * scale).to(q.dtype))
    return apply_rotary_tail(
        q,
        positions,
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )


def norm_rope_inplace_fallback(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    weight: torch.Tensor,
    eps: float,
    rotary_dim: int,
    base: float,
) -> torch.Tensor:
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    x.copy_((y * weight.float()).to(dtype))
    return apply_rotary_tail(x, positions, rotary_dim=rotary_dim, base=base)


def k_norm_rope_cache_fallback(
    kv: torch.Tensor,
    positions: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    rms_norm_eps: float | None = None,
    cache: torch.Tensor | None = None,
    out_loc: torch.Tensor | None = None,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    if (norm_weight is None) != (rms_norm_eps is None):
        raise ValueError(
            "k_norm_rope_cache_fallback requires norm_weight and rms_norm_eps together"
        )
    if (cache is None) != (out_loc is None):
        raise ValueError("k_norm_rope_cache_fallback requires cache and out_loc together")

    has_cache = cache is not None and out_loc is not None
    if norm_weight is not None:
        if kv.ndim != 2:
            raise ValueError(
                f"DSV4 K norm/cache path expects kv shape [tokens, dim], got {kv.shape}"
            )
        if norm_weight.numel() != kv.shape[-1]:
            raise ValueError(
                "DSV4 K norm weight must match kv dim, "
                f"got weight={norm_weight.numel()} dim={kv.shape[-1]}"
            )
        if has_cache and dsv4_optimized_triton_enabled():
            try:
                if _triton_dsv4_ops().k_norm_rope_cache_bf16(
                    kv,
                    positions,
                    norm_weight,
                    cache,
                    out_loc,
                    rms_norm_eps=float(rms_norm_eps),
                    rotary_dim=rotary_dim,
                    base=base,
                    original_seq_len=original_seq_len,
                    factor=factor,
                    beta_fast=beta_fast,
                    beta_slow=beta_slow,
                ):
                    return kv
            except Exception:
                pass
        y = kv.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + float(rms_norm_eps))
        kv.copy_((y * norm_weight.float()).to(kv.dtype))

    out = apply_rotary_tail(
        kv,
        positions,
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    if has_cache:
        dim = out.shape[-1]
        flat = out.reshape(-1, dim)
        if cache.shape[-1] != dim:
            raise ValueError(f"DSV4 K cache dim mismatch: cache dim={cache.shape[-1]} kv dim={dim}")
        loc = out_loc.to(device=cache.device, dtype=torch.long).reshape(-1)
        if loc.numel() != flat.shape[0]:
            raise ValueError(
                "DSV4 K cache loc count must match kv rows, "
                f"got loc={loc.numel()} rows={flat.shape[0]}"
            )
        valid = loc >= 0
        if bool(torch.any(valid)):
            cache[loc[valid]] = flat[valid].to(cache.dtype)
    return out


def q_kv_norm_rope_cache_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    *,
    norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cache: torch.Tensor,
    out_loc: torch.Tensor,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> bool:
    if not dsv4_optimized_triton_enabled():
        return False
    try:
        return bool(
            _triton_dsv4_ops().q_kv_norm_rope_cache_bf16(
                q,
                kv,
                positions,
                norm_weight,
                cache,
                out_loc,
                rms_norm_eps=float(rms_norm_eps),
                rotary_dim=rotary_dim,
                base=base,
                original_seq_len=original_seq_len,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
        )
    except Exception:
        return False








def compress_forward_fallback(
    x: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    ratio: int,
    head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    wkv_gate,
    norm,
    apply_norm: bool = True,
) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_empty((0, head_dim))
    if _cuda_graph_capture_active(x.device):
        return x.new_empty((0, head_dim))
    if positions is None:
        positions = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
    else:
        positions = positions.to(device=x.device, dtype=torch.long)
    if dsv4_optimized_triton_enabled():
        fast = _compress_forward_vectorized(
            x,
            positions,
            ratio=ratio,
            head_dim=head_dim,
            overlap=overlap,
            ape=ape,
            wkv_gate=wkv_gate,
            norm=norm,
            apply_norm=apply_norm,
        )
        if fast is not None:
            return fast
    projected = wkv_gate.forward(x).float()
    kv, score = projected.chunk(2, dim=-1)

    rows = []
    for end_index in torch.nonzero((positions + 1) % ratio == 0, as_tuple=False).flatten().tolist():
        start = int(end_index) - ratio + 1
        if start < 0:
            continue
        end = int(end_index) + 1
        expected = torch.arange(
            int(positions[end_index].item()) - ratio + 1,
            int(positions[end_index].item()) + 1,
            dtype=positions.dtype,
            device=positions.device,
        )
        if not bool(torch.equal(positions[start:end], expected)):
            continue
        slot = (positions[start:end] % ratio).to(torch.long)
        local_score = score[start:end] + ape[slot].float()
        local_kv = kv[start:end]
        if overlap:
            left = local_kv[:, :head_dim]
            right = local_kv[:, head_dim:]
            local_score = torch.cat(
                [local_score[:, :head_dim], local_score[:, head_dim:]],
                dim=0,
            )
            local_kv = torch.cat([left, right], dim=0)
        pooled = (local_kv * local_score.softmax(dim=0)).sum(dim=0, keepdim=True)
        pooled = pooled.to(x.dtype)
        rows.append(norm.forward(pooled) if apply_norm else pooled)
    if not rows:
        return x.new_empty((0, head_dim))
    return torch.cat(rows, dim=0)






def get_paged_mqa_logits_metadata_fallback(
    context_indices: list[torch.Tensor] | DSV4PagedMQAMetadata,
    *,
    device: torch.device | None = None,
) -> DSV4PagedMQAMetadata:
    if isinstance(context_indices, DSV4PagedMQAMetadata):
        if device is None or context_indices.indices.device == device:
            return context_indices
        return DSV4PagedMQAMetadata(
            indptr=context_indices.indptr.to(device=device),
            indices=context_indices.indices.to(device=device),
            lengths=context_indices.lengths.to(device=device),
            max_length=context_indices.max_length,
        )

    if not context_indices:
        out_device = device if device is not None else torch.device("cpu")
        indptr = torch.zeros(1, dtype=torch.int32, device=out_device)
        empty = torch.empty(0, dtype=torch.int32, device=out_device)
        return DSV4PagedMQAMetadata(indptr=indptr, indices=empty, lengths=empty, max_length=0)

    out_device = device if device is not None else context_indices[0].device
    lengths_list: list[int] = []
    rows: list[torch.Tensor] = []
    for row in context_indices:
        row_indices = row.reshape(-1).to(device=out_device, dtype=torch.int32)
        lengths_list.append(int(row_indices.numel()))
        if row_indices.numel() > 0:
            rows.append(row_indices)

    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=out_device)
    indptr = F.pad(lengths.cumsum(dim=0), (1, 0))
    indices = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int32, device=out_device)
    max_length = max(lengths_list) if lengths_list else 0
    return DSV4PagedMQAMetadata(
        indptr=indptr,
        indices=indices,
        lengths=lengths,
        max_length=max_length,
    )


def _paged_mqa_row_indices(
    metadata: DSV4PagedMQAMetadata,
    row: int,
) -> torch.Tensor:
    start = int(metadata.indptr[row].item())
    end = int(metadata.indptr[row + 1].item())
    return metadata.indices[start:end]


@lru_cache(maxsize=1)
def _local_dsv4_sparse_attention_module():
    return load_jit(
        "dsv4_sparse_attention_two_source_bf16",
        cuda_files=["dsv4_sparse_attention_two_source_bf16.cu"],
        cuda_wrappers=[
            (
                "sparse_attention_with_compressed",
                "DSV4SparseAttentionTwoSourceBF16Kernel<true>::run",
            ),
            (
                "sparse_attention_swa_only",
                "DSV4SparseAttentionTwoSourceBF16Kernel<false>::run",
            ),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def dsv4_sparse_attention_two_source_bf16(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    compressed_cache: torch.Tensor | None = None,
    compressed_indices: torch.Tensor | None = None,
    compressed_lengths: torch.Tensor | None = None,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor | None:
    if not (
        dsv4_optimized_enabled()
        and detect_dsv4_kernel_capabilities().is_sm80
    ):
        return None
    if (
        q.ndim != 3
        or swa_cache.ndim != 2
        or swa_indices.ndim != 2
        or swa_lengths.ndim != 1
        or q.shape[-1] != 512
        or swa_cache.shape[-1] != q.shape[-1]
        or q.shape[0] != swa_indices.shape[0]
        or q.shape[0] != swa_lengths.numel()
        or not q.is_cuda
        or not swa_cache.is_cuda
        or not swa_indices.is_cuda
        or not swa_lengths.is_cuda
        or q.dtype is not torch.bfloat16
        or swa_cache.dtype is not torch.bfloat16
        or swa_indices.dtype is not torch.int32
        or swa_lengths.dtype is not torch.int32
        or not q.is_contiguous()
        or not swa_cache.is_contiguous()
        or swa_indices.stride(-1) != 1
    ):
        return None

    has_compressed = (
        compressed_cache is not None
        and compressed_indices is not None
        and compressed_lengths is not None
        and compressed_cache.numel() > 0
    )
    if has_compressed:
        if (
            compressed_cache.ndim != 2
            or compressed_indices.ndim != 2
            or compressed_lengths.ndim != 1
            or compressed_cache.shape[-1] != q.shape[-1]
            or compressed_indices.shape[0] != q.shape[0]
            or compressed_lengths.numel() != q.shape[0]
            or not compressed_cache.is_cuda
            or not compressed_indices.is_cuda
            or not compressed_lengths.is_cuda
            or compressed_cache.dtype is not torch.bfloat16
            or compressed_indices.dtype is not torch.int32
            or compressed_lengths.dtype is not torch.int32
            or not compressed_cache.is_contiguous()
            or compressed_indices.stride(-1) != 1
        ):
            return None
        compressed_cache_arg = compressed_cache
        compressed_indices_arg = compressed_indices
        compressed_lengths_arg = compressed_lengths
    else:
        compressed_cache_arg = swa_cache[:0]
        compressed_indices_arg = swa_indices[:, :1]
        compressed_lengths_arg = torch.zeros_like(swa_lengths)

    if attn_sink is None:
        sink = q.new_empty((1,), dtype=torch.float32)
    else:
        if (
            not attn_sink.is_cuda
            or attn_sink.dtype is not torch.float32
            or attn_sink.numel() < q.shape[1]
        ):
            return None
        sink = attn_sink[: q.shape[1]].contiguous()

    out = torch.empty_like(q)
    try:
        module = _local_dsv4_sparse_attention_module()
        run = (
            module.sparse_attention_with_compressed
            if has_compressed
            else module.sparse_attention_swa_only
        )
        run(
            q,
            compressed_cache_arg,
            compressed_indices_arg,
            compressed_lengths_arg,
            swa_cache,
            swa_indices,
            swa_lengths,
            sink,
            out,
            float(softmax_scale),
            attn_sink is not None,
        )
        return out
    except Exception:
        return None


def dsv4_sparse_attention_two_source_splitk_bf16(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    compressed_cache: torch.Tensor | None = None,
    compressed_indices: torch.Tensor | None = None,
    compressed_lengths: torch.Tensor | None = None,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor | None:
    if not dsv4_optimized_enabled():
        return None
    cap = detect_dsv4_kernel_capabilities()
    try:
        out = _triton_dsv4_ops().sparse_attention_splitk_bf16(
            q,
            swa_cache,
            swa_indices,
            swa_lengths,
            compressed_cache=compressed_cache,
            compressed_indices=compressed_indices,
            compressed_lengths=compressed_lengths,
            softmax_scale=softmax_scale,
            attn_sink=attn_sink,
        )
    except Exception as exc:
        raise RuntimeError(
            "Optimized DSV4 exact bf16 sparse split-K decode failed."
        ) from exc
    if out is None:
        raise RuntimeError(
            "DSV4 exact bf16 sparse split-K decode does not support this "
            "tensor contract; optimized mode does not silently change backends."
        )
    return out


def hadamard_transform_ref(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] <= 0 or x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError(
            "DSV4 Hadamard transform requires a positive power-of-two last dim, "
            f"got {x.shape[-1]}"
        )
    dtype = x.dtype
    y = x.float()
    dim = y.shape[-1]
    step = 1
    while step < dim:
        y = y.reshape(*y.shape[:-1], -1, step * 2)
        left = y[..., :step].clone()
        right = y[..., step : step * 2].clone()
        y[..., :step] = left + right
        y[..., step : step * 2] = left - right
        y = y.reshape(*y.shape[:-2], -1)
        step *= 2
    return (y * (dim**-0.5)).to(dtype)


def indexer_q_rope_hadamard_bf16_fallback(
    q: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"DSV4 indexer q expects shape [tokens, heads, dim], got {q.shape}")
    out = q.contiguous()
    if rotary_dim > 0:
        apply_rotary_tail(
            out,
            positions,
            rotary_dim=rotary_dim,
            base=base,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
    return hadamard_transform_ref(out)


def indexer_kv_hadamard_fallback(kv: torch.Tensor) -> torch.Tensor:
    if kv.numel() == 0:
        return kv
    kv.copy_(hadamard_transform_ref(kv).to(kv.dtype))
    return kv


def _cuda_graph_capture_active(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _indexer_capture_width_mode() -> str:
    return "current"


def _indexer_capture_seq_len_override() -> int | None:
    return None


def _seq_len_aligned_width_for_capture(page_size: int) -> int | None:
    override = _indexer_capture_seq_len_override()
    if override is None:
        return None
    if page_size <= 0:
        return override
    return div_ceil(max(override, 0), page_size) * page_size


def _indexer_capture_static_max_seq_len(
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    capture_active: bool,
) -> tuple[int | None, str, str | None]:
    del seq_lens
    if not capture_active:
        return None, "eager_dynamic_seq_lens", None
    table_width = int(page_table.shape[1])
    current = table_width * int(page_size)
    mode = _indexer_capture_width_mode()
    if mode == "current":
        return current, mode, None
    if mode == "table_width":
        return table_width, mode, "diagnostic_only_may_truncate_page_based_tables"
    seq_len_aligned = _seq_len_aligned_width_for_capture(int(page_size))
    if seq_len_aligned is None:
        raise RuntimeError(
            f"{DSV4_INDEXER_CAPTURE_WIDTH_MODE_ENV}=seq_len_aligned requires "
            f"{DSV4_INDEXER_CAPTURE_SEQ_LEN_OVERRIDE_ENV}; reading CUDA seq_lens "
            "with .item() during graph capture would break capture."
        )
    return seq_len_aligned, mode, "uses_env_seq_len_override"


def _indexer_width_candidates(
    rows: int,
    page_table: torch.Tensor,
    page_size: int,
) -> tuple[dict[str, int | None], dict[str, int | None], str]:
    table_width = int(page_table.shape[1])
    table_times_page = table_width * int(page_size)
    seq_len_aligned = _seq_len_aligned_width_for_capture(int(page_size))
    widths: dict[str, int | None] = {
        "page_table_width": table_width,
        "seq_lens_max": None,
        "page_table_width_times_page_size": table_times_page,
        "seq_len_aligned": seq_len_aligned,
    }
    bytes_by_candidate = {
        name: None if width is None else int(rows) * int(width) * 4
        for name, width in widths.items()
    }
    seq_lens_status = (
        "not_read_during_cuda_graph_capture"
        if seq_len_aligned is None
        else "seq_len_override_env"
    )
    return widths, bytes_by_candidate, seq_lens_status




def indexer_bf16_logits_fallback(
    q: torch.Tensor,
    cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    weights: torch.Tensor | None = None,
    _backend: list[str] | None = None,
    layer_id: int | None = None,
) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"DSV4 indexer q expects shape [rows, heads, dim], got {q.shape}")
    if cache.ndim != 2 or cache.shape[-1] != q.shape[-1]:
        raise ValueError(
            "DSV4 indexer cache must be [slots, dim] with dim matching q, "
            f"got cache={tuple(cache.shape)} q={tuple(q.shape)}"
        )
    if seq_lens.ndim != 1 or seq_lens.shape[0] != q.shape[0]:
        raise ValueError("DSV4 indexer seq_lens must have shape [rows]")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("DSV4 indexer page_table must have shape [rows, pages]")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"DSV4 indexer page_size must be a positive power of two, got {page_size}")
    if weights is not None and weights.shape[:2] != q.shape[:2]:
        raise ValueError(
            "DSV4 indexer weights must have shape [rows, heads] or [rows, heads, 1], "
            f"got weights={tuple(weights.shape)} q={tuple(q.shape)}"
        )

    capture_active = _cuda_graph_capture_active(q.device)
    static_max_seq_len, width_mode, width_mode_note = _indexer_capture_static_max_seq_len(
        seq_lens,
        page_table,
        page_size,
        capture_active,
    )
    if dsv4_optimized_triton_enabled() and weights is not None:
        try:
            logits = _triton_dsv4_ops().indexer_bf16_logits(
                q,
                cache,
                weights,
                seq_lens,
                page_table,
                page_size=page_size,
                max_seq_len=static_max_seq_len,
            )
            if logits is not None:
                if _backend is not None:
                    _backend.append("triton")
                return logits
            if capture_active:
                raise RuntimeError(
                    "DSV4 CUDA graph capture requires the Triton indexer bf16 logits path; "
                    "the current tensor layout/dtype was unsupported."
                )
        except Exception as exc:
            if capture_active:
                raise RuntimeError(
                    "DSV4 CUDA graph capture failed in Triton indexer bf16 logits."
                ) from exc

    rows = q.shape[0]
    max_seq_len = (
        int(static_max_seq_len)
        if static_max_seq_len is not None
        else int(seq_lens.clamp_min(0).max().item()) if seq_lens.numel() else 0
    )
    logits = torch.full(
        (rows, max(max_seq_len, 1)), float("-inf"), dtype=torch.float32, device=q.device
    )
    if rows == 0 or max_seq_len <= 0:
        return logits[:, :0]

    page_bits = (page_size - 1).bit_length()
    q_f = q.float()
    cache_f = cache.to(device=q.device, dtype=torch.float32)
    page_table_i = page_table.to(device=q.device, dtype=torch.int32)
    if weights is None:
        weights_f = torch.ones(q.shape[:2], dtype=torch.float32, device=q.device)
    else:
        weights_f = weights.squeeze(-1).to(device=q.device, dtype=torch.float32)

    for row in range(rows):
        length = int(seq_lens[row].item())
        if length <= 0:
            continue
        length = min(length, logits.shape[1])
        raw = torch.arange(length, dtype=torch.long, device=q.device)
        page_idx = raw >> page_bits
        offset = raw & (page_size - 1)
        valid = page_idx < page_table.shape[1]
        physical_page = torch.full_like(raw, -1)
        if bool(torch.any(valid)):
            physical_page[valid] = page_table_i[row, page_idx[valid]].to(torch.long)
        valid = valid & (physical_page >= 0)
        cache_rows = physical_page * page_size + offset
        row_scores = torch.full((length,), float("-inf"), dtype=torch.float32, device=q.device)
        if bool(torch.any(valid)):
            kv = cache_f[cache_rows[valid]]
            scores = torch.einsum("hd,td->th", q_f[row], kv)
            scores = torch.relu(scores) * weights_f[row][None, :]
            row_scores[valid] = scores.sum(dim=-1)
        logits[row, :length] = row_scores
    if _backend is not None:
        _backend.append("torch")
    return logits




def indexer_fp8_paged_logits_fallback(
    q_values: torch.Tensor,
    packed_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    weights: torch.Tensor,
    _backend: list[str] | None = None,
    layer_id: int | None = None,
) -> torch.Tensor:
    if q_values.ndim != 3:
        raise ValueError(
            f"DSV4 paged FP8 indexer q expects shape [rows, heads, dim], got {q_values.shape}"
        )
    if packed_cache.ndim != 2 or packed_cache.shape[-1] != page_size * (q_values.shape[-1] + 4):
        raise ValueError(
            "DSV4 paged FP8 indexer cache must be [pages, page_size * (dim + 4)], "
            f"got cache={tuple(packed_cache.shape)} q={tuple(q_values.shape)} page_size={page_size}"
        )
    if q_values.dtype is not torch.uint8 or packed_cache.dtype is not torch.uint8:
        raise ValueError("DSV4 paged FP8 indexer q/cache values must be uint8 byte tensors")
    if seq_lens.ndim != 1 or seq_lens.shape[0] != q_values.shape[0]:
        raise ValueError("DSV4 paged FP8 indexer seq_lens must have shape [rows]")
    if page_table.ndim != 2 or page_table.shape[0] != q_values.shape[0]:
        raise ValueError("DSV4 paged FP8 indexer page_table must have shape [rows, pages]")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"DSV4 indexer page_size must be a positive power of two, got {page_size}")
    if weights.ndim not in (2, 3) or weights.shape[:2] != q_values.shape[:2]:
        raise ValueError(
            "DSV4 paged FP8 indexer weights must have shape [rows, heads] or [rows, heads, 1], "
            f"got weights={tuple(weights.shape)} q={tuple(q_values.shape)}"
        )
    if weights.ndim == 3 and weights.shape[-1] != 1:
        raise ValueError(
            "DSV4 paged FP8 indexer weights with rank 3 must have a singleton last dimension, "
            f"got {tuple(weights.shape)}"
        )

    capture_active = _cuda_graph_capture_active(q_values.device)
    static_max_seq_len, width_mode, width_mode_note = _indexer_capture_static_max_seq_len(
        seq_lens,
        page_table,
        page_size,
        capture_active,
    )
    if dsv4_optimized_triton_enabled():
        try:
            logits = _triton_dsv4_ops().indexer_fp8_paged_logits(
                q_values,
                packed_cache,
                weights,
                seq_lens,
                page_table,
                page_size=page_size,
                max_seq_len=static_max_seq_len,
            )
            if logits is not None:
                if _backend is not None:
                    _backend.append("triton_fp8_paged_vllm")
                return logits
            if capture_active:
                raise RuntimeError(
                    "DSV4 CUDA graph capture requires the Triton paged FP8 indexer logits path; "
                    "the current tensor layout/dtype was unsupported."
                )
        except torch.OutOfMemoryError:
            # Retrying the full torch oracle with the same output shape only
            # hides the native owner and doubles allocator pressure.
            raise
        except Exception as exc:
            if capture_active:
                raise RuntimeError(
                    "DSV4 CUDA graph capture failed in paged FP8 indexer logits."
                ) from exc

    rows = q_values.shape[0]
    max_seq_len = (
        int(static_max_seq_len)
        if static_max_seq_len is not None
        else int(seq_lens.clamp_min(0).max().item()) if seq_lens.numel() else 0
    )
    logits = torch.full(
        (rows, max(max_seq_len, 1)), float("-inf"), dtype=torch.float32, device=q_values.device
    )
    if rows == 0 or max_seq_len <= 0:
        return logits[:, :0]

    q_f = q_values.contiguous().view(fp8_dtype()).to(torch.float32)
    cache_f = dequantize_indexer_fp8_paged_cache_ref(
        packed_cache,
        page_size=page_size,
        dim=q_values.shape[-1],
        out_dtype=torch.float32,
    )
    page_bits = (page_size - 1).bit_length()
    page_table_i = page_table.to(device=q_values.device, dtype=torch.int32)
    weights_f = weights.squeeze(-1).to(device=q_values.device, dtype=torch.float32)

    for row in range(rows):
        length = int(seq_lens[row].item())
        if length <= 0:
            continue
        length = min(length, logits.shape[1])
        raw = torch.arange(length, dtype=torch.long, device=q_values.device)
        page_idx = raw >> page_bits
        offset = raw & (page_size - 1)
        valid = page_idx < page_table.shape[1]
        physical_page = torch.full_like(raw, -1)
        if bool(torch.any(valid)):
            physical_page[valid] = page_table_i[row, page_idx[valid]].to(torch.long)
        valid = valid & (physical_page >= 0)
        cache_rows = physical_page * page_size + offset
        row_scores = torch.full(
            (length,), float("-inf"), dtype=torch.float32, device=q_values.device
        )
        if bool(torch.any(valid)):
            kv = cache_f[cache_rows[valid]]
            scores = torch.einsum("hd,td->th", q_f[row], kv)
            scores = torch.relu(scores) * weights_f[row][None, :]
            row_scores[valid] = scores.sum(dim=-1)
        logits[row, :length] = row_scores
    if _backend is not None:
        _backend.append("torch_fp8_paged")
    return logits


def indexer_select_fp8_paged_fallback(
    q_values: torch.Tensor,
    weights: torch.Tensor,
    packed_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int = 512,
    ratio: int = 4,
    layer_id: int | None = None,
) -> DSV4IndexerSelectOutput:
    rows = int(q_values.shape[0])
    capture_active = _cuda_graph_capture_active(q_values.device)
    max_seq_len = (
        0
        if capture_active
        else int(seq_lens.clamp_min(0).max().item()) if seq_lens.numel() else 0
    )
    max_logits_mb = DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT
    max_logits_bytes = max_logits_mb * 1024 * 1024
    full_logits_bytes = rows * max(max_seq_len, 1) * torch.float32.itemsize

    # vLLM bounds the sparse-indexer prefill logits workspace along the query
    # dimension.  Preserve the existing mini Triton logits and native top-k
    # kernels, but never require their full [all query rows, max_seq_len]
    # product when it exceeds the configured workspace budget.
    if (
        not capture_active
        and rows > 0
        and max_seq_len > 0
        and full_logits_bytes > max_logits_bytes
    ):
        max_chunk_rows = max(
            1,
            max_logits_bytes // (max_seq_len * torch.float32.itemsize),
        )
        raw_indices = torch.empty((rows, width), dtype=torch.int32, device=q_values.device)
        page_indices = torch.empty_like(raw_indices)
        full_indices = torch.empty_like(raw_indices)
        topk_lens = torch.empty((rows,), dtype=torch.int32, device=q_values.device)
        backend_names: set[str] = set()
        chunk_count = 0
        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            logits_backend: list[str] = []
            {
                "layer_id": int(layer_id) if layer_id is not None else -1,
                "max_c4_seq_len": int(max_seq_len),
                "slice_rows": int(end - start),
                "logits_elements": int((end - start) * max_seq_len),
                "logits_bytes": int((end - start) * max_seq_len * 4),
                "topk_width": int(width),
            }
            chunk_logits = indexer_fp8_paged_logits_fallback(
                q_values[start:end],
                packed_cache,
                seq_lens[start:end],
                page_table[start:end],
                page_size=page_size,
                weights=weights[start:end],
                _backend=logits_backend,
                layer_id=layer_id,
            )
            chunk_topk = topk_transform_512_full_fallback(
                chunk_logits,
                seq_lens[start:end].to(
                    device=chunk_logits.device, dtype=torch.int32
                ),
                page_table[start:end].to(
                    device=chunk_logits.device, dtype=torch.int32
                ),
                page_size=page_size,
                width=width,
                ratio=ratio,
            )
            raw_indices[start:end].copy_(chunk_topk.raw_indices)
            page_indices[start:end].copy_(chunk_topk.page_indices)
            full_indices[start:end].copy_(chunk_topk.full_indices)
            if chunk_topk.topk_lens is None:
                topk_lens[start:end].copy_(
                    seq_lens[start:end].clamp(min=0, max=width).to(torch.int32)
                )
            else:
                topk_lens[start:end].copy_(chunk_topk.topk_lens)
            backend_names.add(logits_backend[0] if logits_backend else "torch_fp8_paged")
            backend_names.add(chunk_topk.backend)
            chunk_count += 1

        # Full logits are an oracle/debug surface, not a release-path output.
        # Returning an explicit empty tensor keeps the existing output contract
        # while making it impossible for downstream release code to retain the
        # bounded per-chunk workspaces.
        logits = torch.empty((0, 0), dtype=torch.float32, device=q_values.device)
        topk = DSV4TopKTransformOutput(
            raw_indices,
            page_indices,
            full_indices,
            "bounded_query_chunks",
            topk_lens,
        )
        backends = ",".join(sorted(backend_names))
        return DSV4IndexerSelectOutput(
            logits=logits,
            topk=topk,
            backend=(
                f"bounded_query_chunks[{chunk_count};{max_logits_mb}MiB]"
                f"+{backends}"
            ),
        )

    logits_backend: list[str] = []
    {
        "layer_id": int(layer_id) if layer_id is not None else -1,
        "max_c4_seq_len": int(max_seq_len),
        "slice_rows": int(rows),
        "logits_elements": int(rows * max_seq_len),
        "logits_bytes": int(rows * max_seq_len * 4),
        "topk_width": int(width),
    }
    logits = indexer_fp8_paged_logits_fallback(
        q_values,
        packed_cache,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=weights,
        _backend=logits_backend,
        layer_id=layer_id,
    )
    topk = topk_transform_512_full_fallback(
        logits,
        seq_lens.to(device=logits.device, dtype=torch.int32),
        page_table.to(device=logits.device, dtype=torch.int32),
        page_size=page_size,
        width=width,
        ratio=ratio,
    )
    backend = logits_backend[0] if logits_backend else "torch_fp8_paged"
    return DSV4IndexerSelectOutput(logits=logits, topk=topk, backend=f"{backend}+{topk.backend}")


def remap_indexer_topk_locs(
    raw_indices: torch.Tensor,
    component_page_table: torch.Tensor,
    full_page_table: torch.Tensor,
    *,
    component_page_size: int,
    full_page_size: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Map compressed raw top-k indices without int64 matrix temporaries."""
    if not dsv4_optimized_triton_enabled():
        return None
    try:
        return _triton_dsv4_ops().remap_indexer_topk_locs(
            raw_indices,
            component_page_table,
            full_page_table,
            component_page_size=int(component_page_size),
            full_page_size=int(full_page_size),
            ratio=int(ratio),
        )
    except Exception:
        if _cuda_graph_capture_active(raw_indices.device):
            raise
        return None


def c128_prefill_page_indices_one_surface(
    component_page_table: torch.Tensor,
    c128_lengths: torch.Tensor,
    *,
    width: int,
    component_page_size: int,
    out: torch.Tensor | None = None,
    _backend: list[str] | None = None,
) -> torch.Tensor | None:
    """Build the release eager-prefill C128 final-location surface.

    This native micro boundary consumes only the Route-B component page table
    and raw C128 lengths, writes int32 component locations, and writes ``-1``
    for invalid tails/pages. It deliberately cannot materialize raw/full
    matrices or full-size int64 intermediates. TARGET 12.595 owns integration
    into attention metadata construction; decode graph contracts are unchanged.
    """
    cap = detect_dsv4_kernel_capabilities()
    if not (cap.is_sm80 and cap.triton_available):
        return None
    try:
        result = _triton_dsv4_ops().c128_prefill_page_indices_one_surface(
            component_page_table,
            c128_lengths,
            width=int(width),
            component_page_size=int(component_page_size),
            out=out,
        )
    except Exception:
        if _cuda_graph_capture_active(component_page_table.device):
            raise
        return None
    if result is not None and _backend is not None:
        _backend.append("triton_c128_prefill_one_surface")
    return result




def indexer_select_bf16_fallback(
    q: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int = 512,
    ratio: int = 4,
    layer_id: int | None = None,
) -> DSV4IndexerSelectOutput:
    logits_backend: list[str] = []
    logits = indexer_bf16_logits_fallback(
        q,
        cache,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=weights,
        _backend=logits_backend,
        layer_id=layer_id,
    )
    topk = topk_transform_512_full_fallback(
        logits,
        seq_lens.to(device=logits.device, dtype=torch.int32),
        page_table.to(device=logits.device, dtype=torch.int32),
        page_size=page_size,
        width=width,
        ratio=ratio,
    )
    backend = logits_backend[0] if logits_backend else "torch"
    return DSV4IndexerSelectOutput(logits=logits, topk=topk, backend=f"{backend}+{topk.backend}")


def hc_split_sinkhorn_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = mixes.view(-1, mix_hc).float()
    hc_scale = hc_scale.float()
    hc_base = hc_base.float()

    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post_start = hc_mult
    post_end = 2 * hc_mult
    post = 2 * torch.sigmoid(
        mixes[:, post_start:post_end] * hc_scale[1] + hc_base[post_start:post_end]
    )
    comb_raw = mixes[:, post_end:].view(-1, hc_mult, hc_mult)
    comb_base = hc_base[post_end:].view(hc_mult, hc_mult)
    comb = torch.softmax(comb_raw * hc_scale[2] + comb_base, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre_fallback(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x.shape
    flat = x.flatten(1)
    if dsv4_optimized_triton_enabled() and dsv4_optimized_triton_enabled():
        mixes = linear_bf16_fp32_fallback(flat, fn)
        fused = _triton_dsv4_ops().hc_prenorm_split_pre(
            mixes.contiguous(),
            x,
            scale,
            base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            norm_eps=norm_eps,
        )
        if fused is not None:
            return fused
    flat_float = flat.float()
    rsqrt = torch.rsqrt(flat_float.square().mean(-1, keepdim=True) + norm_eps)
    mixes = linear_bf16_fp32_fallback(flat, fn) * rsqrt
    if dsv4_optimized_triton_enabled():
        fused = _triton_dsv4_ops().hc_split_pre(
            mixes.contiguous(),
            x,
            scale,
            base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
        )
        if fused is not None:
            return fused
    pre, post, comb = hc_split_sinkhorn_ref(mixes, scale, base, hc_mult, sinkhorn_iters, eps)
    y = torch.sum(pre.to(x.dtype).unsqueeze(-1) * x.view(shape), dim=1)
    return y, post.to(x.dtype), comb.to(x.dtype)


def hc_post_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if dsv4_optimized_triton_enabled():
        fused = _triton_dsv4_ops().hc_post(x, residual, post, comb)
        if fused is not None:
            return fused
    return post.unsqueeze(-1) * x.unsqueeze(1) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(2), dim=1
    )


def hc_head_fallback(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    eps: float,
    norm_eps: float,
) -> torch.Tensor:
    shape = x.shape
    flat = x.flatten(1)
    if dsv4_optimized_triton_enabled() and dsv4_optimized_triton_enabled():
        mixes = linear_bf16_fp32_fallback(flat, fn)
        fused = _triton_dsv4_ops().hc_prenorm_head(
            mixes.contiguous(),
            x,
            scale,
            base,
            hc_mult=shape[1],
            eps=eps,
            norm_eps=norm_eps,
        )
        if fused is not None:
            return fused
    flat_float = flat.float()
    rsqrt = torch.rsqrt(flat_float.square().mean(-1, keepdim=True) + norm_eps)
    mixes = linear_bf16_fp32_fallback(flat, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + eps
    return torch.sum(pre.to(x.dtype).unsqueeze(-1) * x.view(shape), dim=1)


def sequence_mqa_attention_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    spans: list[tuple[int, int]],
    *,
    window_size: int,
    softmax_scale: float,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q)
    sink = attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)
    for start, end in spans:
        q_seq = q[start:end].float()
        kv_seq = kv[start:end].float()
        seq_len = end - start
        for local_idx in range(seq_len):
            ctx_start = max(0, local_idx - window_size + 1) if window_size else 0
            candidates = kv_seq[ctx_start : local_idx + 1]
            scores = torch.einsum("hd,td->ht", q_seq[local_idx], candidates)
            scores = scores * softmax_scale
            max_score = torch.maximum(scores.max(dim=-1).values, sink)
            exp_scores = torch.exp(scores - max_score[:, None])
            denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
            attn = exp_scores / denom[:, None]
            out[start + local_idx] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
    return out


def paged_mqa_attention_fallback(
    q: torch.Tensor,
    cache: torch.Tensor,
    context_indices: list[torch.Tensor] | DSV4PagedMQAMetadata,
    *,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"DSV4 fallback expects q shape [tokens, heads, dim], got {q.shape}")
    metadata = get_paged_mqa_logits_metadata_fallback(context_indices, device=q.device)
    if metadata.row_count != q.shape[0]:
        raise ValueError(
            "DSV4 paged MQA metadata row count must match q tokens, "
            f"got {metadata.row_count} rows for {q.shape[0]} tokens"
        )
    if dsv4_optimized_triton_enabled():
        try:
            out = _triton_dsv4_ops().paged_mqa_attention_bf16(
                q,
                cache,
                metadata.indptr,
                metadata.indices,
                metadata.lengths,
                softmax_scale=softmax_scale,
                attn_sink=attn_sink,
                max_length=metadata.max_length,
            )
            if out is not None:
                return out
        except Exception:
            pass
    out = torch.empty_like(q)
    sink = (
        attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)
        if attn_sink is not None
        else None
    )
    for row in range(metadata.row_count):
        indices = _paged_mqa_row_indices(metadata, row)
        if indices.numel() == 0:
            out[row].zero_()
            continue
        candidates = cache[indices.to(torch.long)].float()
        scores = torch.einsum("hd,td->ht", q[row].float(), candidates) * softmax_scale
        if sink is None:
            attn = torch.softmax(scores, dim=-1)
        else:
            max_score = torch.maximum(scores.max(dim=-1).values, sink)
            exp_scores = torch.exp(scores - max_score[:, None])
            denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
            attn = exp_scores / denom[:, None]
        out[row] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
    return out


def wo_a_grouped_projection_fallback(
    o: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    num_local_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    d_per_group = o.shape[-1]
    wo_a = dequant_fp8_weight(weight, scale, out_dtype=o.dtype)
    wo_a = wo_a.view(num_local_groups, o_lora_rank, d_per_group)
    return torch.einsum("tgd,grd->tgr", o, wo_a).reshape(o.shape[0], -1)


def hash_topk_fallback(hash_topk, input_ids: torch.Tensor) -> torch.Tensor:
    return hash_topk.forward(input_ids.view(-1)).long()


def build_moe_route_plan(
    indices: torch.Tensor,
    *,
    num_experts: int,
    block_size_m: int = 16,
) -> DSV4MoERoutePlan:
    if indices.ndim != 2:
        raise ValueError(
            f"DSV4 MoE route plan expects indices shape [tokens, topk], got {indices.shape}"
        )
    if num_experts <= 0:
        raise ValueError(f"DSV4 MoE route plan requires num_experts > 0, got {num_experts}")
    if block_size_m <= 0:
        raise ValueError(f"DSV4 MoE route plan requires block_size_m > 0, got {block_size_m}")

    route_count = indices.numel()
    topk = indices.shape[1]
    device = indices.device
    if route_count and indices.is_cuda and dsv4_optimized_triton_enabled():
        try:
            route_plan = _triton_dsv4_ops().build_moe_route_plan(
                indices,
                num_experts=num_experts,
                block_size_m=block_size_m,
            )
            if route_plan is not None:
                sorted_route_ids, expert_ids, num_tokens_post_padded = route_plan
                return DSV4MoERoutePlan(
                    sorted_route_ids=sorted_route_ids,
                    expert_ids=expert_ids,
                    num_tokens_post_padded=num_tokens_post_padded,
                    route_count=route_count,
                    topk=topk,
                    block_size_m=block_size_m,
                )
        except Exception:
            pass

    flat_indices = indices.reshape(-1).to(torch.long)
    valid = (flat_indices >= 0) & (flat_indices < num_experts)
    valid_route_ids = torch.arange(route_count, device=device, dtype=torch.long)[valid]
    valid_expert_ids = flat_indices[valid]

    if valid_route_ids.numel() == 0:
        empty_ids = torch.empty((0,), dtype=torch.int32, device=device)
        return DSV4MoERoutePlan(
            sorted_route_ids=empty_ids,
            expert_ids=empty_ids,
            num_tokens_post_padded=torch.zeros((1,), dtype=torch.int32, device=device),
            route_count=route_count,
            topk=topk,
            block_size_m=block_size_m,
        )

    sort_key = valid_expert_ids * max(route_count, 1) + valid_route_ids
    order = torch.argsort(sort_key)
    sorted_valid_routes = valid_route_ids[order]
    sorted_valid_experts = valid_expert_ids[order]

    counts = torch.bincount(valid_expert_ids, minlength=num_experts).to(torch.long)
    padded_counts = ((counts + block_size_m - 1) // block_size_m) * block_size_m
    total_padded = int(padded_counts.sum().item())
    sorted_route_ids = torch.full(
        (total_padded,),
        route_count,
        dtype=torch.int32,
        device=device,
    )

    counts_before = counts.cumsum(0) - counts
    padded_offsets = padded_counts.cumsum(0) - padded_counts
    compact_positions = torch.arange(
        sorted_valid_routes.numel(),
        device=device,
        dtype=torch.long,
    )
    local_ranks = compact_positions - counts_before[sorted_valid_experts]
    padded_positions = padded_offsets[sorted_valid_experts] + local_ranks
    sorted_route_ids[padded_positions] = sorted_valid_routes.to(torch.int32)

    blocks_per_expert = (padded_counts // block_size_m).to(torch.long)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device),
        blocks_per_expert,
    )
    return DSV4MoERoutePlan(
        sorted_route_ids=sorted_route_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor([total_padded], dtype=torch.int32, device=device),
        route_count=route_count,
        topk=topk,
        block_size_m=block_size_m,
    )


def build_moe_v2_execution_plan(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_experts: int,
    block_size_m: int = 16,
    reduce_once: bool = True,
    final_reduce_label: str = "dsv4.v1_moe_reduce_once_all_reduce",
) -> DSV4MoEExecutionPlan:
    if hidden_states.ndim != 2:
        raise ValueError(
            f"DSV4 MoE V2 execution plan expects hidden_states [tokens, hidden], got {hidden_states.shape}"
        )
    if weights.shape != indices.shape or indices.ndim != 2:
        raise ValueError(
            "DSV4 MoE V2 execution plan expects matching weights/indices [tokens, topk], "
            f"got weights={weights.shape}, indices={indices.shape}"
        )
    route_plan = build_moe_route_plan(
        indices,
        num_experts=num_experts,
        block_size_m=block_size_m,
    )
    route_weights = (
        weights.to(device=hidden_states.device, dtype=torch.float32).reshape(-1).contiguous()
    )
    return DSV4MoEExecutionPlan(
        route_plan=route_plan,
        route_weights=route_weights,
        tokens=int(hidden_states.shape[0]),
        hidden=int(hidden_states.shape[1]),
        num_experts=int(num_experts),
        reduce_once=bool(reduce_once),
        final_reduce_label=final_reduce_label,
    )


def moe_execution_block_size(*, tokens: int, topk: int, num_experts: int) -> int:
    """Use the production backend's route blocking for the authoritative plan."""
    if dsv4_moe_expert_backend() == DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16:
        from minisgl.kernel import marlin_wna16

        return marlin_wna16.choose_block_size(
            tokens=tokens,
            topk=topk,
            experts=num_experts,
            input_dtype=None,
        )
    return 16


def mask_moe_routes_live_rows(
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_token_non_padded: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared live-route contract before any route planning."""
    if num_token_non_padded is None:
        return weights, indices
    if weights.ndim != 2 or indices.shape != weights.shape:
        raise ValueError(
            "DSV4 live-route masking expects matching weights/indices [tokens, topk]"
        )
    if num_token_non_padded.numel() != 1:
        raise ValueError("num_token_non_padded must contain exactly one element")
    if num_token_non_padded.dtype != torch.int32:
        raise TypeError("num_token_non_padded must use torch.int32")
    if num_token_non_padded.device != weights.device or indices.device != weights.device:
        raise ValueError("live count, route weights, and route IDs must share a device")
    if weights.is_cuda:
        try:
            if _triton_dsv4_ops().mask_moe_routes_live_rows(
                weights, indices, num_token_non_padded
            ):
                return weights, indices
        except Exception:
            pass
    rows = torch.arange(weights.shape[0], device=weights.device)
    live = (rows < num_token_non_padded).unsqueeze(1)
    return (
        torch.where(live, weights, torch.zeros_like(weights)),
        torch.where(live, indices, torch.full_like(indices, -1)),
    )


def zero_moe_padded_rows(
    output: torch.Tensor,
    num_token_non_padded: torch.Tensor | None,
) -> torch.Tensor:
    """Finalize excluded MoE rows without clearing the maximum workspace."""
    if num_token_non_padded is None:
        return output
    if output.ndim != 2:
        raise ValueError("DSV4 MoE padded-row finalize expects [tokens, hidden]")
    if num_token_non_padded.numel() != 1 or num_token_non_padded.dtype != torch.int32:
        raise ValueError("num_token_non_padded must be a one-element int32 tensor")
    if num_token_non_padded.device != output.device:
        raise ValueError("live count and MoE output must share a device")
    if output.is_cuda:
        try:
            if _triton_dsv4_ops().zero_moe_padded_rows(output, num_token_non_padded):
                return output
        except Exception:
            pass
    rows = torch.arange(output.shape[0], device=output.device)
    return torch.where(
        (rows < num_token_non_padded).unsqueeze(1),
        output,
        torch.zeros_like(output),
    )


def moe_gate_fallback(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_ids: torch.Tensor | None,
    topk: int,
    scoring_func: str,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
    hash_topk=None,
    num_token_non_padded: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(hidden_states.float(), weight.float())
    if scoring_func == "softmax":
        original_scores = scores.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        original_scores = scores.sigmoid()
    else:
        original_scores = F.softplus(scores).sqrt()

    if hash_topk is not None:
        if input_ids is None:
            raise ValueError("DeepSeek V4 hash routing requires input_ids")
        indices = hash_topk_fallback(hash_topk, input_ids)
    else:
        scores_for_topk = original_scores
        if correction_bias is not None:
            scores_for_topk = scores_for_topk + correction_bias.float()
        indices = scores_for_topk.topk(topk, dim=-1).indices

    weights = original_scores.gather(1, indices)
    if scoring_func != "softmax":
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    weights = weights * routed_scaling_factor
    return mask_moe_routes_live_rows(weights, indices, num_token_non_padded)


def silu_and_mul_clamp_fallback(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if dsv4_optimized_triton_enabled():
        try:
            out = _triton_dsv4_ops().silu_and_mul_clamp(
                gate,
                up,
                swiglu_limit=swiglu_limit,
                weights=weights,
            )
            if out is not None:
                return out
        except Exception:
            pass
    gate_f = gate.float()
    up_f = up.float()
    if swiglu_limit > 0:
        up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
        gate_f = torch.clamp(gate_f, max=swiglu_limit)
    out = F.silu(gate_f) * up_f
    if weights is not None:
        out = out * weights
    return out




def moe_route_dispatch_bf16_grouped(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    moe_plan: DSV4MoEExecutionPlan | None = None,
    workspace: DSV4MoEWorkspace | None = None,
) -> torch.Tensor | None:
    if not dsv4_optimized_triton_enabled():
        return None
    if hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_cuda:
        return None
    if hidden_states.ndim != 2 or weights.shape != indices.shape or indices.ndim != 2:
        return None
    if w13_weight.ndim != 4 or w13_weight.shape[1] != 2 or w2_weight.ndim != 3:
        return None
    if w13_weight.shape[0] != w2_weight.shape[0]:
        return None
    if hidden_states.shape[1] != w13_weight.shape[-1] * 2:
        return None

    try:
        if moe_plan is None:
            route_plan = build_moe_route_plan(
                indices,
                num_experts=w13_weight.shape[0],
                block_size_m=16,
            )
            route_weights = weights.to(
                device=hidden_states.device,
                dtype=torch.float32,
            ).contiguous()
        else:
            if (
                moe_plan.tokens != hidden_states.shape[0]
                or moe_plan.hidden != hidden_states.shape[1]
                or moe_plan.num_experts != w13_weight.shape[0]
                or moe_plan.route_plan.route_count != weights.numel()
                or moe_plan.route_plan.topk != indices.shape[1]
            ):
                return None
            route_plan = moe_plan.route_plan
            route_weights = moe_plan.route_weights
        return _triton_dsv4_ops().grouped_fp4_moe(
            hidden_states.contiguous(),
            route_weights,
            w13_weight.contiguous(),
            w13_scale,
            w2_weight.contiguous(),
            w2_scale,
            route_plan.sorted_route_ids,
            route_plan.expert_ids,
            route_plan.num_tokens_post_padded,
            route_count=route_plan.route_count,
            topk=route_plan.topk,
            block_size_m=route_plan.block_size_m,
            swiglu_limit=swiglu_limit,
            workspace=workspace,
        )
    except Exception:
        return None


@lru_cache(maxsize=1)
def _local_dsv4_topk_v1_module(width: int):
    return load_jit(
        "dsv4_topk_v1",
        str(int(width)),
        cuda_files=["dsv4_topk_v1.cu"],
        cuda_wrappers=[
            ("topk_transform", f"DSV4TopKTransformKernel<{int(width)}>::run"),
            (
                "topk_transform_global_lens",
                f"DSV4TopKTransformGlobalLensKernel<{int(width)}>::run",
            ),
        ],
    )


def _run_local_cuda_topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    raw_indices: torch.Tensor,
    width: int,
) -> bool:
    if not (
        dsv4_optimized_enabled()
        and scores.is_cuda
        and detect_dsv4_kernel_capabilities().is_sm80
        and width in (512, 1024)
    ):
        return False
    try:
        module = _local_dsv4_topk_v1_module(int(width))
        module.topk_transform(
            scores.contiguous(),
            seq_lens.contiguous(),
            page_table.contiguous(),
            page_indices,
            page_size,
            raw_indices,
        )
        return True
    except Exception:
        return False


def _run_local_cuda_global_topk_lens_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    raw_indices: torch.Tensor,
    full_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    width: int,
    ratio: int,
) -> bool:
    if not (
        dsv4_optimized_enabled()
        and scores.is_cuda
        and detect_dsv4_kernel_capabilities().is_sm80
        and width in (512, 1024)
        and ratio > 0
    ):
        return False
    try:
        module = _local_dsv4_topk_v1_module(int(width))
        module.topk_transform_global_lens(
            scores.contiguous(),
            seq_lens.contiguous(),
            page_table.contiguous(),
            page_indices,
            page_size,
            raw_indices,
            full_indices,
            topk_lens,
            ratio,
        )
        return True
    except Exception:
        return False


def _validate_full_topk_inputs(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int,
    ratio: int,
) -> None:
    if scores.dim() != 2:
        raise ValueError(f"scores must have shape [B, max_seq_len], got {tuple(scores.shape)}")
    if seq_lens.dim() != 1 or seq_lens.shape[0] != scores.shape[0]:
        raise ValueError("seq_lens must be int32/int64 with shape [B]")
    if page_table.dim() != 2 or page_table.shape[0] != scores.shape[0]:
        raise ValueError("page_table must have shape [B, num_pages]")
    if scores.shape[1] > 0 and page_table.shape[1] == 0:
        raise ValueError("page_table must contain at least one page when scores are non-empty")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page_size must be a positive power of two, got {page_size}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")


def _topk_transform_512_full_torch(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int,
    ratio: int,
) -> DSV4TopKTransformOutput:
    _validate_full_topk_inputs(
        scores,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=ratio,
    )
    batch, max_seq_len = scores.shape
    device = scores.device
    raw_indices = torch.full((batch, width), -1, dtype=torch.int32, device=device)
    page_indices = torch.full_like(raw_indices, -1)
    full_indices = torch.full_like(raw_indices, -1)
    if batch == 0 or max_seq_len == 0:
        topk_lens = torch.zeros(batch, dtype=torch.int32, device=device)
        return DSV4TopKTransformOutput(raw_indices, page_indices, full_indices, "torch", topk_lens)

    lens = seq_lens.to(device=device, dtype=torch.long).clamp(min=0, max=max_seq_len)
    topk_lens = lens.clamp(max=width).to(torch.int32)
    actual_k = min(width, max_seq_len)
    if actual_k > 0:
        positions = torch.arange(max_seq_len, dtype=torch.long, device=device)
        masked_scores = scores.float().clone()
        masked_scores.masked_fill_(positions[None, :] >= lens[:, None], float("-inf"))
        _values, topk_raw = torch.topk(masked_scores, k=actual_k, dim=1, largest=True, sorted=False)
        raw_indices[:, :actual_k] = topk_raw.to(torch.int32)

    sequential = torch.arange(width, dtype=torch.int32, device=device).expand(batch, -1)
    sequential_valid = sequential.to(torch.long) < lens[:, None]
    raw_indices = torch.where(
        (lens <= width)[:, None],
        torch.where(sequential_valid, sequential, torch.full_like(sequential, -1)),
        raw_indices,
    )

    page_bits = (page_size - 1).bit_length()
    page_mask = page_size - 1
    page_idx = raw_indices.to(torch.long) >> page_bits
    offset = raw_indices.to(torch.long) & page_mask
    valid = raw_indices >= 0
    valid = valid & (page_idx >= 0) & (page_idx < page_table.shape[1])
    clamped_page_idx = page_idx.clamp(min=0, max=max(page_table.shape[1] - 1, 0))
    physical_pages = torch.gather(
        page_table.to(device=device, dtype=torch.int32),
        dim=1,
        index=clamped_page_idx,
    ).to(torch.long)
    valid = valid & (physical_pages >= 0)
    page_values = (physical_pages << page_bits) | offset
    page_indices = torch.where(valid, page_values.to(torch.int32), page_indices)
    full_values = page_indices.to(torch.long) * int(ratio) + (int(ratio) - 1)
    full_indices = torch.where(valid, full_values.to(torch.int32), full_indices)
    return DSV4TopKTransformOutput(raw_indices, page_indices, full_indices, "torch", topk_lens)


def topk_transform_512_full_fallback(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int = 512,
    ratio: int = 4,
) -> DSV4TopKTransformOutput:
    _validate_full_topk_inputs(
        scores,
        seq_lens,
        page_table,
        page_size=int(page_size),
        width=int(width),
        ratio=int(ratio),
    )
    force_torch = False
    raw_indices = torch.empty(
        (scores.shape[0], int(width)), dtype=torch.int32, device=scores.device
    )
    page_indices = torch.empty_like(raw_indices)
    full_indices = torch.empty_like(raw_indices)
    topk_lens = torch.empty((scores.shape[0],), dtype=torch.int32, device=scores.device)
    clamped_lens = seq_lens.to(device=scores.device, dtype=torch.int32).clamp(
        min=0, max=scores.shape[1]
    )
    if not force_torch and _run_local_cuda_global_topk_lens_512(
        scores.to(torch.float32),
        clamped_lens,
        page_table.to(device=scores.device, dtype=torch.int32),
        page_indices,
        int(page_size),
        raw_indices,
        full_indices,
        topk_lens,
        int(width),
        int(ratio),
    ):
        return DSV4TopKTransformOutput(
            raw_indices,
            page_indices,
            full_indices,
            "local_cuda_global_topk_lens",
            topk_lens,
        )
    if dsv4_optimized_enabled() and _cuda_graph_capture_active(
        scores.device
    ):
        raise RuntimeError(
            "Optimized DSV4 CUDA graph capture requires the global topk/lens JIT path."
        )
    if not force_torch and _run_local_cuda_topk_transform_512(
        scores.to(torch.float32),
        clamped_lens,
        page_table.to(device=scores.device, dtype=torch.int32),
        page_indices,
        int(page_size),
        raw_indices,
        int(width),
    ):
        valid = page_indices >= 0
        full_values = page_indices.to(torch.long) * int(ratio) + (int(ratio) - 1)
        full_indices = torch.where(
            valid,
            full_values.to(torch.int32),
            torch.full_like(page_indices, -1),
        )
        return DSV4TopKTransformOutput(raw_indices, page_indices, full_indices, "local_cuda_v1")
    return _topk_transform_512_full_torch(
        scores,
        seq_lens,
        page_table,
        page_size=int(page_size),
        width=int(width),
        ratio=int(ratio),
    )










def store_swa_fallback(
    kvcache,
    layer_id: int,
    kv: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    out_loc_is_swa: bool = False,
) -> None:
    store_loc = out_loc
    translate = getattr(kvcache, "translate_full_locs_to_swa_locs", None)
    if (
        not out_loc_is_swa
        and callable(translate)
        and bool(getattr(kvcache, "swa_independent_lifecycle_enabled", False))
    ):
        store_loc = translate(out_loc)
    if out_loc_is_swa:
        if not bool(torch.all(store_loc >= 0).item()):
            raise RuntimeError("DSV4 SWA write requested for full loc without live SWA mapping")
        kvcache.swa_cache(layer_id)[store_loc.long()] = kv.reshape(
            -1, kvcache.swa_cache(layer_id).shape[-1]
        ).to(kvcache.swa_cache(layer_id).dtype)
        return
    kvcache.store_swa(layer_id, kv, out_loc)


def store_compressed_fallback(
    kvcache,
    layer_id: int,
    kv: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    loc_flat = loc.reshape(-1)
    if dsv4_optimized_triton_enabled():
        try:
            if _triton_dsv4_ops().store_cache(kvcache.component_cache(layer_id), kv, loc):
                return
        except Exception:
            pass
    if loc.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSV4 masked compressed cache store requires Triton during CUDA graph capture"
        )
    valid = loc_flat >= 0
    if bool(torch.any(valid)):
        kvcache.store_compressed(layer_id, kv.reshape(-1, kv.shape[-1])[valid], loc_flat[valid])


def store_indexer_fallback(kvcache, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
    loc_flat = loc.reshape(-1)
    if dsv4_optimized_triton_enabled():
        try:
            if _triton_dsv4_ops().store_cache(kvcache.indexer_cache(layer_id), kv, loc):
                return
        except Exception:
            pass
    if loc.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSV4 masked indexer cache store requires Triton during CUDA graph capture"
        )
    valid = loc_flat >= 0
    if bool(torch.any(valid)):
        kvcache.store_indexer(layer_id, kv.reshape(-1, kv.shape[-1])[valid], loc_flat[valid])


def store_indexer_fp8_cache_fallback(
    kvcache,
    layer_id: int,
    kv: torch.Tensor,
    loc: torch.Tensor,
) -> bool:
    if not dsv4_optimized_enabled():
        return False
    if not hasattr(kvcache, "has_indexer_fp8_cache") or not kvcache.has_indexer_fp8_cache():
        raise RuntimeError(
            "Optimized DSV4 requires DeepSeekV4KVCache to be allocated with an "
            "FP8 indexer side cache."
        )
    flat = kv.reshape(-1, kv.shape[-1]).contiguous()
    loc_flat = loc.to(device=flat.device, dtype=torch.long).reshape(-1)
    if loc_flat.numel() != flat.shape[0]:
        raise ValueError(
            "DSV4 FP8 indexer cache loc count must match kv rows, "
            f"got loc={loc_flat.numel()} rows={flat.shape[0]}"
        )
    if flat.numel() == 0:
        return True

    if hasattr(kvcache, "has_indexer_fp8_paged_cache") and kvcache.has_indexer_fp8_paged_cache():
        packed_cache = kvcache.indexer_fp8_paged_cache(layer_id)
        page_size = int(kvcache.indexer_fp8_page_size)
        if packed_cache.shape[-1] != page_size * (flat.shape[-1] + 4):
            raise ValueError(
                "DSV4 paged FP8 indexer cache dim mismatch: "
                f"cache page bytes={packed_cache.shape[-1]} kv dim={flat.shape[-1]} "
                f"page_size={page_size}"
            )
        if dsv4_optimized_triton_enabled():
            try:
                if _triton_dsv4_ops().indexer_fp8_paged_quant_store(
                    flat,
                    loc_flat,
                    packed_cache,
                    page_size=page_size,
                ):
                    return True
            except Exception as exc:
                if _cuda_graph_capture_active(flat.device):
                    raise RuntimeError(
                        "DSV4 CUDA graph capture failed in paged FP8 indexer cache store."
                    ) from exc
        if _cuda_graph_capture_active(flat.device):
            raise RuntimeError(
                "DSV4 CUDA graph capture requires the Triton paged FP8 indexer cache store path."
            )

        valid = loc_flat >= 0
        if bool(torch.any(valid)):
            q_values, q_scales = quantize_indexer_fp8_cache_ref(flat[valid])
            loc_valid = loc_flat[valid].to(device=packed_cache.device, dtype=torch.long)
            pages = loc_valid // page_size
            offsets = loc_valid - pages * page_size
            page_bytes = page_size * (flat.shape[-1] + 4)
            data = packed_cache.as_strided(
                (packed_cache.shape[0], page_size, flat.shape[-1]),
                (page_bytes, flat.shape[-1], 1),
            )
            scales = packed_cache.as_strided(
                (packed_cache.shape[0], page_size, 4),
                (page_bytes, 4, 1),
                storage_offset=page_size * flat.shape[-1],
            )
            data[pages, offsets] = q_values.to(device=packed_cache.device)
            scales[pages, offsets] = q_scales.to(device=packed_cache.device)
        return True

    values, scales = kvcache.indexer_fp8_cache(layer_id)
    if flat.shape[-1] != values.shape[-1]:
        raise ValueError(
            f"DSV4 FP8 indexer cache dim mismatch: cache dim={values.shape[-1]} kv dim={flat.shape[-1]}"
        )

    if dsv4_optimized_triton_enabled():
        try:
            if _triton_dsv4_ops().indexer_fp8_quant_store(flat, loc_flat, values, scales):
                return True
        except Exception as exc:
            if _cuda_graph_capture_active(flat.device):
                raise RuntimeError(
                    "DSV4 CUDA graph capture failed in FP8 indexer cache store."
                ) from exc
    if _cuda_graph_capture_active(flat.device):
        raise RuntimeError(
            "DSV4 CUDA graph capture requires the Triton FP8 indexer cache store path."
        )

    valid = loc_flat >= 0
    if bool(torch.any(valid)):
        q_values, q_scales = quantize_indexer_fp8_cache_ref(flat[valid])
        values[loc_flat[valid].to(device=values.device)] = q_values.to(device=values.device)
        scales[loc_flat[valid].to(device=scales.device)] = q_scales.to(device=scales.device)
    return True




def copy_masked_compressed_locs(
    raw_out_loc: torch.Tensor,
    positions: torch.Tensor,
    c4_out_loc: torch.Tensor | None,
    c128_out_loc: torch.Tensor | None,
    rows: int,
) -> None:
    if (
        c4_out_loc is not None
        and c128_out_loc is not None
        and dsv4_optimized_triton_enabled()
    ):
        try:
            if _triton_dsv4_ops().copy_masked_compressed_locs(
                raw_out_loc,
                positions,
                c4_out_loc,
                c128_out_loc,
                rows,
            ):
                return
        except Exception:
            pass
    _copy_masked_compressed_locs_fallback(c4_out_loc, raw_out_loc, positions, rows, ratio=4)
    _copy_masked_compressed_locs_fallback(c128_out_loc, raw_out_loc, positions, rows, ratio=128)




def direct_decode_index_metadata_for_replay(
    *,
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    c4_page_table: torch.Tensor | None,
    c128_page_table: torch.Tensor | None,
    dst_swa_page_indices: torch.Tensor,
    dst_c4_sparse_raw_indices: torch.Tensor,
    dst_c4_sparse_page_indices: torch.Tensor,
    dst_c4_sparse_full_indices: torch.Tensor,
    dst_c128_raw_indices: torch.Tensor,
    dst_c128_page_indices: torch.Tensor,
    dst_c128_full_indices: torch.Tensor,
    rows: int,
    page_size: int,
    window_size: int,
    index_topk: int,
    direct_swa: bool,
    direct_c4: bool,
    direct_c128: bool,
    swa_full_to_swa_page: torch.Tensor | None = None,
    swa_dummy_token_start: int = -1,
    swa_dummy_page: int = -1,
    swa_independent: bool = False,
) -> bool:
    if not dsv4_optimized_enabled():
        return False
    if rows < 0 or page_size <= 0 or window_size <= 0 or index_topk <= 0:
        return False
    if rows == 0:
        return True
    tensors = [
        ctx_page_table,
        table_indices,
        positions,
    ]
    if direct_swa:
        tensors.append(dst_swa_page_indices)
        if swa_independent:
            if swa_full_to_swa_page is None:
                return False
            tensors.append(swa_full_to_swa_page)
    if direct_c4:
        if c4_page_table is None:
            return False
        tensors.extend(
            [
                c4_page_table,
                dst_c4_sparse_raw_indices,
                dst_c4_sparse_page_indices,
                dst_c4_sparse_full_indices,
            ]
        )
    if direct_c128:
        if c128_page_table is None:
            return False
        tensors.extend(
            [
                c128_page_table,
                dst_c128_raw_indices,
                dst_c128_page_indices,
                dst_c128_full_indices,
            ]
        )
    if not (direct_swa or direct_c4 or direct_c128):
        return False
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if (
        ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or positions.ndim != 1
        or table_indices.numel() < rows
        or positions.numel() < rows
        or page_size & (page_size - 1)
    ):
        return False
    try:
        return bool(
            _triton_dsv4_ops().direct_decode_index_metadata_for_replay(
                ctx_page_table,
                table_indices[:rows],
                positions[:rows],
                c4_page_table[:rows] if c4_page_table is not None else None,
                c128_page_table[:rows] if c128_page_table is not None else None,
                dst_swa_page_indices,
                dst_c4_sparse_raw_indices,
                dst_c4_sparse_page_indices,
                dst_c4_sparse_full_indices,
                dst_c128_raw_indices,
                dst_c128_page_indices,
                dst_c128_full_indices,
                page_size=int(page_size),
                window_size=int(window_size),
                index_topk=int(index_topk),
                direct_swa=bool(direct_swa),
                direct_c4=bool(direct_c4),
                direct_c128=bool(direct_c128),
                swa_full_to_swa_page=swa_full_to_swa_page,
                swa_dummy_token_start=int(swa_dummy_token_start),
                swa_dummy_page=int(swa_dummy_page),
                swa_independent=bool(swa_independent),
            )
        )
    except Exception:
        return False


def copy_decode_metadata_for_replay(
    *,
    dst_raw_out_loc: torch.Tensor,
    src_raw_out_loc: torch.Tensor,
    dst_seq_lens: torch.Tensor,
    src_seq_lens: torch.Tensor,
    dst_req_seq_lens: torch.Tensor,
    src_req_seq_lens: torch.Tensor,
    dst_extend_lens: torch.Tensor,
    src_extend_lens: torch.Tensor,
    dst_positions: torch.Tensor,
    src_positions: torch.Tensor,
    dst_req_table_indices: torch.Tensor,
    src_req_table_indices: torch.Tensor,
    dst_swa_topk_lengths: torch.Tensor,
    src_swa_topk_lengths: torch.Tensor,
    dst_c4_topk_lengths_raw: torch.Tensor,
    src_c4_topk_lengths_raw: torch.Tensor,
    dst_c4_topk_lengths_clamp1: torch.Tensor,
    src_c4_topk_lengths_clamp1: torch.Tensor,
    dst_c4_sparse_topk_lengths: torch.Tensor,
    src_c4_sparse_topk_lengths: torch.Tensor,
    dst_c128_topk_lengths_clamp1: torch.Tensor,
    src_c128_topk_lengths_clamp1: torch.Tensor,
    dst_cu_seqlens_q: torch.Tensor,
    src_cu_seqlens_q: torch.Tensor,
    dst_page_table: torch.Tensor,
    src_page_table: torch.Tensor,
    dst_swa_page_indices: torch.Tensor,
    src_swa_page_indices: torch.Tensor,
    dst_c4_sparse_raw_indices: torch.Tensor,
    src_c4_sparse_raw_indices: torch.Tensor,
    dst_c4_sparse_page_indices: torch.Tensor,
    src_c4_sparse_page_indices: torch.Tensor,
    dst_c4_sparse_full_indices: torch.Tensor,
    src_c4_sparse_full_indices: torch.Tensor,
    dst_c128_raw_indices: torch.Tensor,
    src_c128_raw_indices: torch.Tensor,
    dst_c128_page_indices: torch.Tensor,
    src_c128_page_indices: torch.Tensor,
    dst_c128_full_indices: torch.Tensor,
    src_c128_full_indices: torch.Tensor,
    rows: int,
    graph_inputs_bound: bool,
    skip_swa_page_indices: bool = False,
    skip_c4_sparse_indices: bool = False,
    skip_c128_indices: bool = False,
) -> bool:
    if not dsv4_optimized_enabled():
        return False
    tensors = (
        dst_raw_out_loc,
        src_raw_out_loc,
        dst_seq_lens,
        src_seq_lens,
        dst_req_seq_lens,
        src_req_seq_lens,
        dst_extend_lens,
        src_extend_lens,
        dst_positions,
        src_positions,
        dst_req_table_indices,
        src_req_table_indices,
        dst_swa_topk_lengths,
        src_swa_topk_lengths,
        dst_c4_topk_lengths_raw,
        src_c4_topk_lengths_raw,
        dst_c4_topk_lengths_clamp1,
        src_c4_topk_lengths_clamp1,
        dst_c4_sparse_topk_lengths,
        src_c4_sparse_topk_lengths,
        dst_c128_topk_lengths_clamp1,
        src_c128_topk_lengths_clamp1,
        dst_cu_seqlens_q,
        src_cu_seqlens_q,
        dst_page_table,
        src_page_table,
        dst_swa_page_indices,
        src_swa_page_indices,
        dst_c4_sparse_raw_indices,
        src_c4_sparse_raw_indices,
        dst_c4_sparse_page_indices,
        src_c4_sparse_page_indices,
        dst_c4_sparse_full_indices,
        src_c4_sparse_full_indices,
        dst_c128_raw_indices,
        src_c128_raw_indices,
        dst_c128_page_indices,
        src_c128_page_indices,
        dst_c128_full_indices,
        src_c128_full_indices,
    )
    if rows <= 0:
        return True
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if not all(t.dim() == 1 for t in tensors[:24]):
        return False
    if not all(t.dim() == 2 for t in tensors[24:]):
        return False
    if any(t.shape[0] < rows for t in tensors[:22]):
        return False
    if dst_cu_seqlens_q.shape[0] < rows + 1 or src_cu_seqlens_q.shape[0] < rows + 1:
        return False
    for dst, src in (
        (dst_page_table, src_page_table),
        (dst_swa_page_indices, src_swa_page_indices),
        (dst_c4_sparse_raw_indices, src_c4_sparse_raw_indices),
        (dst_c4_sparse_page_indices, src_c4_sparse_page_indices),
        (dst_c4_sparse_full_indices, src_c4_sparse_full_indices),
        (dst_c128_raw_indices, src_c128_raw_indices),
        (dst_c128_page_indices, src_c128_page_indices),
        (dst_c128_full_indices, src_c128_full_indices),
    ):
        if dst.shape[0] < rows or src.shape[0] < rows:
            return False
    try:
        return bool(
            _triton_dsv4_ops().copy_decode_metadata_for_replay(
                dst_raw_out_loc=dst_raw_out_loc,
                src_raw_out_loc=src_raw_out_loc,
                dst_seq_lens=dst_seq_lens,
                src_seq_lens=src_seq_lens,
                dst_req_seq_lens=dst_req_seq_lens,
                src_req_seq_lens=src_req_seq_lens,
                dst_extend_lens=dst_extend_lens,
                src_extend_lens=src_extend_lens,
                dst_positions=dst_positions,
                src_positions=src_positions,
                dst_req_table_indices=dst_req_table_indices,
                src_req_table_indices=src_req_table_indices,
                dst_swa_topk_lengths=dst_swa_topk_lengths,
                src_swa_topk_lengths=src_swa_topk_lengths,
                dst_c4_topk_lengths_raw=dst_c4_topk_lengths_raw,
                src_c4_topk_lengths_raw=src_c4_topk_lengths_raw,
                dst_c4_topk_lengths_clamp1=dst_c4_topk_lengths_clamp1,
                src_c4_topk_lengths_clamp1=src_c4_topk_lengths_clamp1,
                dst_c4_sparse_topk_lengths=dst_c4_sparse_topk_lengths,
                src_c4_sparse_topk_lengths=src_c4_sparse_topk_lengths,
                dst_c128_topk_lengths_clamp1=dst_c128_topk_lengths_clamp1,
                src_c128_topk_lengths_clamp1=src_c128_topk_lengths_clamp1,
                dst_cu_seqlens_q=dst_cu_seqlens_q,
                src_cu_seqlens_q=src_cu_seqlens_q,
                dst_page_table=dst_page_table,
                src_page_table=src_page_table,
                dst_swa_page_indices=dst_swa_page_indices,
                src_swa_page_indices=src_swa_page_indices,
                dst_c4_sparse_raw_indices=dst_c4_sparse_raw_indices,
                src_c4_sparse_raw_indices=src_c4_sparse_raw_indices,
                dst_c4_sparse_page_indices=dst_c4_sparse_page_indices,
                src_c4_sparse_page_indices=src_c4_sparse_page_indices,
                dst_c4_sparse_full_indices=dst_c4_sparse_full_indices,
                src_c4_sparse_full_indices=src_c4_sparse_full_indices,
                dst_c128_raw_indices=dst_c128_raw_indices,
                src_c128_raw_indices=src_c128_raw_indices,
                dst_c128_page_indices=dst_c128_page_indices,
                src_c128_page_indices=src_c128_page_indices,
                dst_c128_full_indices=dst_c128_full_indices,
                src_c128_full_indices=src_c128_full_indices,
                rows=int(rows),
                graph_inputs_bound=bool(graph_inputs_bound),
                skip_swa_page_indices=bool(skip_swa_page_indices),
                skip_c4_sparse_indices=bool(skip_c4_sparse_indices),
                skip_c128_indices=bool(skip_c128_indices),
            )
        )
    except Exception:
        return False


def copy_component_write_locs_for_replay(
    *,
    c4_page_table: torch.Tensor | None,
    c128_page_table: torch.Tensor | None,
    c4_indexer_page_table: torch.Tensor | None,
    positions: torch.Tensor,
    c4_out_loc: torch.Tensor | None,
    c128_out_loc: torch.Tensor | None,
    c4_indexer_out_loc: torch.Tensor | None,
    rows: int,
    page_size: int,
) -> bool:
    if not dsv4_optimized_enabled():
        return False
    tensors = (
        c4_page_table,
        c128_page_table,
        c4_indexer_page_table,
        positions,
        c4_out_loc,
        c128_out_loc,
        c4_indexer_out_loc,
    )
    if rows <= 0:
        return True
    if page_size <= 0:
        return False
    if any(t is None for t in tensors):
        return False
    assert c4_page_table is not None
    assert c128_page_table is not None
    assert c4_indexer_page_table is not None
    assert c4_out_loc is not None
    assert c128_out_loc is not None
    assert c4_indexer_out_loc is not None
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if (
        c4_page_table.ndim != 2
        or c128_page_table.ndim != 2
        or c4_indexer_page_table.ndim != 2
        or positions.ndim != 1
        or c4_out_loc.ndim != 1
        or c128_out_loc.ndim != 1
        or c4_indexer_out_loc.ndim != 1
        or positions.numel() < rows
        or c4_page_table.shape[0] < rows
        or c128_page_table.shape[0] < rows
        or c4_indexer_page_table.shape[0] < rows
        or c4_out_loc.numel() < rows
        or c128_out_loc.numel() < rows
        or c4_indexer_out_loc.numel() < rows
    ):
        return False
    try:
        return bool(
            _triton_dsv4_ops().copy_component_write_locs_for_replay(
                c4_page_table=c4_page_table,
                c128_page_table=c128_page_table,
                c4_indexer_page_table=c4_indexer_page_table,
                positions=positions,
                c4_out_loc=c4_out_loc,
                c128_out_loc=c128_out_loc,
                c4_indexer_out_loc=c4_indexer_out_loc,
                rows=int(rows),
                page_size=int(page_size),
            )
        )
    except Exception:
        return False


def prep_decode_metadata_in_graph(
    *,
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    materialized_seq_lens: torch.Tensor,
    c4_page_table: torch.Tensor | None,
    c128_page_table: torch.Tensor | None,
    c4_indexer_page_table: torch.Tensor | None,
    dst_seq_lens: torch.Tensor,
    dst_swa_topk_lengths: torch.Tensor,
    dst_c4_topk_lengths_raw: torch.Tensor,
    dst_c4_topk_lengths_clamp1: torch.Tensor,
    dst_c4_sparse_topk_lengths: torch.Tensor,
    dst_c128_topk_lengths_clamp1: torch.Tensor,
    dst_swa_page_indices: torch.Tensor,
    dst_c4_sparse_raw_indices: torch.Tensor,
    dst_c4_sparse_page_indices: torch.Tensor,
    dst_c4_sparse_full_indices: torch.Tensor,
    dst_c128_raw_indices: torch.Tensor,
    dst_c128_page_indices: torch.Tensor,
    dst_c128_full_indices: torch.Tensor,
    dst_c4_out_loc: torch.Tensor | None,
    dst_c128_out_loc: torch.Tensor | None,
    dst_c4_indexer_out_loc: torch.Tensor | None,
    dst_swa_out_loc: torch.Tensor | None = None,
    rows: int,
    page_size: int,
    window_size: int,
    index_topk: int,
    swa_full_to_swa_page: torch.Tensor | None = None,
    swa_dummy_token_start: int = -1,
    swa_dummy_page: int = -1,
    swa_independent: bool = False,
) -> bool:
    if not dsv4_optimized_enabled():
        return False
    if rows < 0 or page_size <= 0 or window_size <= 0 or index_topk <= 0:
        return False
    if page_size & (page_size - 1):
        return False
    if rows == 0:
        return True
    if c4_page_table is None or c128_page_table is None or c4_indexer_page_table is None:
        return False
    if dst_c4_out_loc is None or dst_c128_out_loc is None or dst_c4_indexer_out_loc is None:
        return False
    tensors = [
        ctx_page_table,
        table_indices,
        positions,
        raw_out_loc,
        materialized_seq_lens,
        c4_page_table,
        c128_page_table,
        c4_indexer_page_table,
        dst_seq_lens,
        dst_swa_topk_lengths,
        dst_c4_topk_lengths_raw,
        dst_c4_topk_lengths_clamp1,
        dst_c4_sparse_topk_lengths,
        dst_c128_topk_lengths_clamp1,
        dst_swa_page_indices,
        dst_c4_sparse_raw_indices,
        dst_c4_sparse_page_indices,
        dst_c4_sparse_full_indices,
        dst_c128_raw_indices,
        dst_c128_page_indices,
        dst_c128_full_indices,
        dst_c4_out_loc,
        dst_c128_out_loc,
        dst_c4_indexer_out_loc,
    ]
    if swa_independent:
        if (
            swa_full_to_swa_page is None
            or swa_dummy_token_start < 0
            or swa_dummy_page < 0
        ):
            return False
        tensors.append(swa_full_to_swa_page)
    if dst_swa_out_loc is not None:
        tensors.append(dst_swa_out_loc)
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if (
        ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or positions.ndim != 1
        or raw_out_loc.ndim != 1
        or materialized_seq_lens.ndim != 1
        or c4_page_table.ndim != 2
        or c128_page_table.ndim != 2
        or c4_indexer_page_table.ndim != 2
        or any(t.ndim != 1 for t in tensors[8:14])
        or any(t.ndim != 2 for t in tensors[14:21])
        or any(t.ndim != 1 for t in tensors[21:])
    ):
        return False
    if (
        table_indices.numel() < rows
        or positions.numel() < rows
        or raw_out_loc.numel() < rows
        or materialized_seq_lens.numel() < rows
        or c4_page_table.shape[0] < rows
        or c128_page_table.shape[0] < rows
        or c4_indexer_page_table.shape[0] < rows
        or any(t.numel() < rows for t in tensors[8:14])
        or any(t.shape[0] < rows for t in tensors[14:21])
        or any(t.numel() < rows for t in tensors[21:24])
    ):
        return False
    if swa_independent:
        assert swa_full_to_swa_page is not None
        if swa_full_to_swa_page.ndim != 1:
            return False
    if dst_swa_out_loc is not None and (
        dst_swa_out_loc.ndim != 1 or dst_swa_out_loc.numel() < rows
    ):
        return False
    dummy_swa_full_to_swa_page = (
        swa_full_to_swa_page if swa_full_to_swa_page is not None else table_indices
    )
    dummy_swa_out_loc = dst_swa_out_loc if dst_swa_out_loc is not None else raw_out_loc
    try:
        return bool(
            _triton_dsv4_ops().prep_decode_metadata_in_graph(
                ctx_page_table,
                table_indices[:rows],
                positions[:rows],
                raw_out_loc[:rows],
                materialized_seq_lens[:rows],
                c4_page_table[:rows],
                c128_page_table[:rows],
                c4_indexer_page_table[:rows],
                dst_seq_lens,
                dst_swa_topk_lengths,
                dst_c4_topk_lengths_raw,
                dst_c4_topk_lengths_clamp1,
                dst_c4_sparse_topk_lengths,
                dst_c128_topk_lengths_clamp1,
                dst_swa_page_indices,
                dst_c4_sparse_raw_indices,
                dst_c4_sparse_page_indices,
                dst_c4_sparse_full_indices,
                dst_c128_raw_indices,
                dst_c128_page_indices,
                dst_c128_full_indices,
                dst_c4_out_loc,
                dst_c128_out_loc,
                dst_c4_indexer_out_loc,
                dummy_swa_full_to_swa_page,
                dummy_swa_out_loc,
                page_size=int(page_size),
                window_size=int(window_size),
                index_topk=int(index_topk),
                swa_independent=bool(swa_independent),
                swa_dummy_token_start=int(swa_dummy_token_start),
                swa_dummy_page=int(swa_dummy_page),
                write_swa_out_loc=dst_swa_out_loc is not None,
            )
        )
    except Exception:
        return False


def _copy_masked_compressed_locs_fallback(
    dst: torch.Tensor | None,
    raw_out_loc: torch.Tensor,
    positions: torch.Tensor,
    rows: int,
    *,
    ratio: Literal[4, 128],
) -> None:
    if dst is None:
        return
    dst[:rows].copy_(
        torch.where(
            (positions[:rows] + 1) % ratio == 0,
            raw_out_loc[:rows].div(ratio, rounding_mode="floor"),
            torch.full_like(raw_out_loc[:rows], -1),
        )
    )
    if dst.shape[0] > rows:
        dst[rows:].fill_(-1)


def _compressed_store_cache(kvcache, layer_id: int, cache_type: str) -> torch.Tensor:
    if cache_type == "compressed":
        return kvcache.component_cache(layer_id)
    if cache_type == "indexer":
        return kvcache.indexer_cache(layer_id)
    raise ValueError(f"Unsupported DSV4 compressed cache_type: {cache_type}")


def compress_norm_rope_store_fallback(
    kvcache,
    layer_id: int,
    kv: torch.Tensor,
    loc: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    rms_norm_eps: float | None = None,
    rotary_dim: int = 0,
    base: float = 10000.0,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    cache_type: Literal["compressed", "indexer"] = "compressed",
    apply_hadamard: bool = False,
) -> None:
    if (norm_weight is None) != (rms_norm_eps is None):
        raise ValueError(
            "compress_norm_rope_store_fallback requires norm_weight and rms_norm_eps together"
        )

    if positions is None and rotary_dim > 0:
        raise ValueError("compress_norm_rope_store_fallback requires positions when rotary_dim > 0")
    if rotary_dim <= 0 and norm_weight is None and not apply_hadamard:
        if cache_type == "compressed":
            store_compressed_fallback(kvcache, layer_id, kv, loc)
        else:
            store_indexer_fallback(kvcache, layer_id, kv, loc)
        return

    dim = kv.shape[-1]
    flat = kv.reshape(-1, dim)
    loc_flat = loc.to(device=flat.device, dtype=torch.long).reshape(-1)
    positions_flat = (
        positions.to(device=flat.device, dtype=torch.long).reshape(-1)
        if positions is not None
        else None
    )
    if loc_flat.numel() != flat.shape[0]:
        raise ValueError(
            "DSV4 compressed cache loc count must match kv rows, "
            f"got loc={loc_flat.numel()} rows={flat.shape[0]}"
        )
    if positions_flat is not None and positions_flat.numel() != flat.shape[0]:
        raise ValueError(
            "DSV4 compressed cache positions count must match kv rows, "
            f"got positions={positions_flat.numel()} rows={flat.shape[0]}"
        )
    if norm_weight is not None and norm_weight.numel() != dim:
        raise ValueError(
            "DSV4 compressed norm weight must match kv dim, "
            f"got weight={norm_weight.numel()} dim={dim}"
        )

    cache = _compressed_store_cache(kvcache, layer_id, cache_type)
    if cache.shape[-1] != dim:
        raise ValueError(
            f"DSV4 compressed cache dim mismatch: cache dim={cache.shape[-1]} kv dim={dim}"
        )

    if (
        dsv4_optimized_triton_enabled()
        and not apply_hadamard
        and positions_flat is not None
        and not (cache_type == "indexer" and dsv4_optimized_enabled())
    ):
        try:
            if _triton_dsv4_ops().compress_norm_rope_store_bf16(
                flat,
                positions_flat,
                norm_weight,
                cache,
                loc_flat,
                rms_norm_eps=float(rms_norm_eps or 0.0),
                rotary_dim=rotary_dim,
                base=base,
                original_seq_len=original_seq_len,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            ):
                return
        except Exception:
            pass

    if norm_weight is not None:
        y = flat.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + float(rms_norm_eps))
        flat.copy_((y * norm_weight.float()).to(flat.dtype))
    if rotary_dim > 0:
        assert positions_flat is not None
        apply_rotary_tail(
            flat,
            positions_flat,
            rotary_dim=rotary_dim,
            base=base,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
    if apply_hadamard:
        indexer_kv_hadamard_fallback(flat)

    if cache_type == "indexer" and dsv4_optimized_enabled():
        store_indexer_fp8_cache_fallback(kvcache, layer_id, flat, loc_flat)
        return

    if dsv4_optimized_triton_enabled():
        try:
            if _triton_dsv4_ops().store_cache(cache, flat, loc_flat):
                return
        except Exception:
            pass
    if loc_flat.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DSV4 compressed cache fallback store is not CUDA graph safe")

    valid = loc_flat >= 0
    if bool(torch.any(valid)):
        cache[loc_flat[valid].to(device=cache.device)] = flat[valid].to(cache.dtype)












__all__ = [
    "DSV4_KERNEL_INVENTORY",
    "DSV4KernelCapability",
    "DSV4KernelInventoryEntry",
    "DSV4KernelMode",
    "DSV4IndexerFP8Query",
    "DSV4IndexerSelectOutput",
    "DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT",
    "DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4",
    "DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16",
    "DSV4_SM80_MOE_EXPERT_BACKENDS",
    "DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR",
    "DSV4_SM80_MOE_V2_WORKSPACE_MAX_ROUTES",
    "DSV4DecodeMetadataDeforestOutput",
    "DSV4MoEExecutionPlan",
    "DSV4MoERoutePlan",
    "DSV4MoEWorkspace",
    "DSV4PagedMQAMetadata",
    "DSV4TopKTransformOutput",
    "DSV4TwoSourceAttentionMetadata",
    "apply_rotary_tail",
    "build_moe_route_plan",
    "build_moe_v2_execution_plan",
    "copy_masked_compressed_locs",
    "copy_decode_metadata_for_replay",
    "copy_component_write_locs_for_replay",
    "prep_decode_metadata_in_graph",
    "direct_decode_index_metadata_for_replay",
    "direct_c4_sparse_metadata_for_replay",
    "decode_metadata_deforest_fallback",
    "compress_norm_rope_store_fallback",
    "compress_forward_fallback",
    "compressor_plan_fallback",
    "flash_mla_sparse_prefill",
    "flash_mla_with_kvcache",
    "get_paged_mqa_logits_metadata_fallback",
    "make_name",
    "triton_create_paged_compress_data",
    "dequant_fp4_weight",
    "dequant_fp8_weight",
    "dequantize_indexer_fp8_cache_ref",
    "dequantize_indexer_fp8_paged_cache_ref",
    "detect_dsv4_kernel_capabilities",
    "dsv4_moe_expert_backend",
    "dsv4_optimized_cuda_enabled",
    "dsv4_optimized_enabled",
    "dsv4_optimized_triton_enabled",
    "dsv4_kernel_inventory_by_wrapper",
    "dsv4_sparse_attention_two_source_bf16",
    "dsv4_sparse_attention_two_source_splitk_bf16",
    "e8m0_dtype",
    "fp8_dtype",
    "fused_q_indexer_rope_first_quant",
    "fused_q_indexer_rope_hadamard_fp4_quant",
    "fused_q_indexer_rope_hadamard_quant",
    "hash_topk_fallback",
    "hadamard_transform_ref",
    "hc_head_fallback",
    "hc_post_fallback",
    "hc_pre_fallback",
    "hc_split_sinkhorn_ref",
    "indexer_bf16_logits_fallback",
    "indexer_fp8_logits_fallback",
    "indexer_fp8_paged_logits_fallback",
    "indexer_kv_hadamard_fallback",
    "indexer_q_rope_fp8_fallback",
    "indexer_q_rope_hadamard_bf16_fallback",
    "indexer_select_bf16_fallback",
    "indexer_select_fp8_fallback",
    "indexer_select_fp8_paged_fallback",
    "k_norm_rope_cache_fallback",
    "linear_bf16_fp32_fallback",
    "linear_bf16_fp32_upstream_enabled",
    "mega_moe_pre_dispatch_fallback",
    "moe_gate_fallback",
    "moe_route_dispatch_bf16_grouped",
    "moe_route_dispatch_bf16_marlin_wna16",
    "moe_route_dispatch_bf16_marlin_wna16_prepacked",
    "norm_rope_inplace_fallback",
    "paged_mqa_attention_fallback",
    "plan_topk_v2_fallback",
    "q_norm_rope_fallback",
    "q_kv_norm_rope_cache_fallback",
    "pack_indexer_fp8_paged_cache_ref",
    "quantize_indexer_fp8_cache_ref",
    "quantize_indexer_fp8_paged_cache_ref",
    "quantize_fp8_activation_ref",
    "quantized_linear_fp8_pair_shared_activation_ref",
    "quantized_linear_ref",
    "remap_indexer_topk_locs",
    "require_supported_moe_expert_backend",
    "rms_norm_pair_fallback",
    "scale_dim",
    "sequence_mqa_attention_fallback",
    "silu_and_mul_clamp_fallback",
    "silu_and_mul_contig_post_quant",
    "silu_and_mul_masked_post_quant",
    "store_compressed_fallback",
    "store_indexer_fp8_cache_fallback",
    "store_indexer_fallback",
    "store_swa_fallback",
    "topk_transform_512_fallback",
    "topk_transform_512_full_fallback",
    "topk_transform_512_v2_fallback",
    "unsupported_kernel",
    "wo_a_grouped_projection_fallback",
]

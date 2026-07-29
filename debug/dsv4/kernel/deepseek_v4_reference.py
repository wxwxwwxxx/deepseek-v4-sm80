"""Debug-only mathematical references for DeepSeek V4 kernel validation.

Nothing in the release runtime imports this module. Tests and microbenchmarks may
use these deliberately slow implementations as local operator-level references;
model-level oracle data comes from the official DeepSeek runtime captures.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from minisgl.kernel import deepseek_v4 as _release
from minisgl.utils import div_ceil

WeightKind = Literal["bf16", "fp8", "fp4"]

DSV4IndexerFP8Query = _release.DSV4IndexerFP8Query
DSV4IndexerSelectOutput = _release.DSV4IndexerSelectOutput
DSV4PagedMQAMetadata = _release.DSV4PagedMQAMetadata
DSV4TopKTransformOutput = _release.DSV4TopKTransformOutput
DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT = _release.DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT

# Shared release infrastructure. Keeping these aliases explicit makes the debug
# dependency one-way: release never imports debug.
_c4_indexer_rmsnorm_bf16_native = _release._c4_indexer_rmsnorm_bf16_native
_cuda_graph_capture_active = _release._cuda_graph_capture_active
_indexer_capture_static_max_seq_len = _release._indexer_capture_static_max_seq_len
_linear_bf16_fp32_upstream = _release._linear_bf16_fp32_upstream
_local_dsv4_topk_v1_module = _release._local_dsv4_topk_v1_module
_triton_dsv4_ops = _release._triton_dsv4_ops
_validate_full_topk_inputs = _release._validate_full_topk_inputs
dequant_fp8_weight = _release.dequant_fp8_weight
detect_dsv4_kernel_capabilities = _release.detect_dsv4_kernel_capabilities
dsv4_triton_available = _release.dsv4_triton_available
fp8_dtype = _release.fp8_dtype
linear_bf16_fp32_upstream_enabled = _release.linear_bf16_fp32_upstream_enabled

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


def quantize_fp8_activation_ref(
    x: torch.Tensor,
    *,
    block_size: int = 128,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or x.numel() == 0 or x.shape[-1] % block_size != 0:
        if out is not None:
            out.copy_(x)
            return out
        return x
    if dsv4_triton_available():
        try:
            y = _triton_dsv4_ops().fp8_activation_quantize(
                x,
                block_size=block_size,
                out=out,
            )
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
    y = y.reshape_as(flat).reshape_as(x).to(dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y


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
    if dsv4_triton_available():
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
    if dsv4_triton_available():
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
    if dsv4_triton_available():
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
    if dsv4_triton_available():
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
    publish_swa_qat: bool = False,
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
        if has_cache and dsv4_triton_available():
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
                    publish_swa_qat=publish_swa_qat,
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
        non_rope = dim - rotary_dim
        if publish_swa_qat and non_rope == 448:
            flat[:, :non_rope] = quantize_fp8_activation_ref(flat[:, :non_rope], block_size=64)
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
    publish_swa_qat: bool = False,
) -> bool:
    if not dsv4_triton_available():
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
                publish_swa_qat=publish_swa_qat,
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
    if dsv4_triton_available():
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


def hadamard_transform_ref(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] <= 0 or x.shape[-1] & (x.shape[-1] - 1):
        raise ValueError(
            f"DSV4 Hadamard transform requires a positive power-of-two last dim, got {x.shape[-1]}"
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
    if dsv4_triton_available() and weights is not None:
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
        else int(seq_lens.clamp_min(0).max().item())
        if seq_lens.numel()
        else 0
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
    if dsv4_triton_available():
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
        else int(seq_lens.clamp_min(0).max().item())
        if seq_lens.numel()
        else 0
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
        0 if capture_active else int(seq_lens.clamp_min(0).max().item()) if seq_lens.numel() else 0
    )
    max_logits_mb = DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT
    max_logits_bytes = max_logits_mb * 1024 * 1024
    full_logits_bytes = rows * max(max_seq_len, 1) * torch.float32.itemsize

    # vLLM bounds the sparse-indexer prefill logits workspace along the query
    # dimension.  Preserve the existing mini Triton logits and native top-k
    # kernels, but never require their full [all query rows, max_seq_len]
    # product when it exceeds the configured workspace budget.
    if not capture_active and rows > 0 and max_seq_len > 0 and full_logits_bytes > max_logits_bytes:
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
                seq_lens[start:end].to(device=chunk_logits.device, dtype=torch.int32),
                page_table[start:end].to(device=chunk_logits.device, dtype=torch.int32),
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
            backend=(f"bounded_query_chunks[{chunk_count};{max_logits_mb}MiB]+{backends}"),
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
    if dsv4_triton_available() and dsv4_triton_available():
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
    if dsv4_triton_available():
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
    if dsv4_triton_available():
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
    if dsv4_triton_available() and dsv4_triton_available():
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
    if dsv4_triton_available():
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


def silu_and_mul_clamp_fallback(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if dsv4_triton_available():
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
        scores.is_cuda and detect_dsv4_kernel_capabilities().is_ampere and width in (512, 1024)
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
        scores.is_cuda
        and detect_dsv4_kernel_capabilities().is_ampere
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
    positions = torch.arange(max_seq_len, dtype=torch.long, device=device)
    valid_scores = positions[None, :] < lens[:, None]
    row_finite = torch.isfinite(scores.float()) | ~valid_scores
    row_finite = row_finite.all(dim=1)
    topk_lens = torch.where(
        row_finite,
        lens.clamp(max=width).to(torch.int32),
        torch.full((batch,), -1, dtype=torch.int32, device=device),
    )
    actual_k = min(width, max_seq_len)
    if actual_k > 0:
        masked_scores = scores.float().clone()
        # This allocation-tolerant fallback is also the explicit Candidate-B
        # reference: stable descending score order preserves ascending raw
        # logical index for exact ties, including positive/negative zero.
        masked_scores.masked_fill_(~valid_scores, float("-inf"))
        topk_raw = torch.argsort(masked_scores, dim=1, descending=True, stable=True)
        topk_raw = topk_raw[:, :actual_k].to(torch.int32)
        valid_slots = (
            torch.arange(actual_k, device=device)[None, :] < lens.clamp(max=width)[:, None]
        )
        valid_slots = valid_slots & row_finite[:, None]
        raw_indices[:, :actual_k] = torch.where(
            valid_slots, topk_raw, torch.full_like(topk_raw, -1)
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
    if _cuda_graph_capture_active(scores.device):
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
    if dsv4_triton_available():
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
    if dsv4_triton_available():
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
        if dsv4_triton_available():
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

    if dsv4_triton_available():
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


def copy_masked_compressed_locs_fallback(
    raw_out_loc: torch.Tensor,
    positions: torch.Tensor,
    c4_out_loc: torch.Tensor | None,
    c128_out_loc: torch.Tensor | None,
    rows: int,
) -> None:
    _copy_masked_compressed_locs_fallback(
        c4_out_loc,
        raw_out_loc,
        positions,
        rows,
        ratio=4,
    )
    _copy_masked_compressed_locs_fallback(
        c128_out_loc,
        raw_out_loc,
        positions,
        rows,
        ratio=128,
    )


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
        cache_type == "indexer"
        and apply_hadamard
        and norm_weight is not None
        and positions_flat is not None
        and hasattr(kvcache, "has_indexer_fp8_paged_cache")
        and kvcache.has_indexer_fp8_paged_cache()
        and flat.is_cuda
        and detect_dsv4_kernel_capabilities().is_ampere
    ):
        if not dsv4_triton_available():
            raise RuntimeError("The qualified SM80 C4 indexer publication ABI requires Triton.")
        packed_cache = kvcache.indexer_fp8_paged_cache(layer_id)
        page_size = int(kvcache.indexer_fp8_page_size)
        try:
            norm_weight_fp32 = norm_weight.to(
                device=flat.device,
                dtype=torch.float32,
            ).contiguous()
            rms_accepted = _c4_indexer_rmsnorm_bf16_native(
                flat,
                norm_weight_fp32,
                loc_flat,
                rms_norm_eps=float(rms_norm_eps),
            )
            if not rms_accepted:
                raise RuntimeError("The qualified SM80 C4 indexer RMSNorm stage rejected its ABI.")
            rope_accepted = _triton_dsv4_ops().indexer_rotary_tail_valid(
                flat,
                positions_flat,
                loc_flat,
                rotary_dim=rotary_dim,
                base=base,
                original_seq_len=original_seq_len,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
            store_accepted = rope_accepted and _triton_dsv4_ops().indexer_hadamard_fp8_paged_store(
                flat,
                loc_flat,
                packed_cache,
                page_size=page_size,
            )
            if not store_accepted:
                raise RuntimeError(
                    "C4 indexer publication partially launched but the "
                    "native RoPE/Hadamard/store cluster rejected its ABI."
                )
            return
        except Exception as exc:
            raise RuntimeError(
                "DSV4 fused C4 indexer publication cluster failed after ABI selection."
            ) from exc

    if (
        dsv4_triton_available()
        and not apply_hadamard
        and positions_flat is not None
        and cache_type != "indexer"
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

    if cache_type == "indexer":
        store_indexer_fp8_cache_fallback(kvcache, layer_id, flat, loc_flat)
        return

    if dsv4_triton_available():
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
    "apply_rotary_tail",
    "compress_forward_fallback",
    "compress_norm_rope_store_fallback",
    "copy_masked_compressed_locs_fallback",
    "dequant_fp4_weight",
    "dequantize_indexer_fp8_cache_ref",
    "dequantize_indexer_fp8_paged_cache_ref",
    "get_paged_mqa_logits_metadata_fallback",
    "hadamard_transform_ref",
    "hash_topk_fallback",
    "hc_head_fallback",
    "hc_post_fallback",
    "hc_pre_fallback",
    "hc_split_sinkhorn_ref",
    "indexer_bf16_logits_fallback",
    "indexer_fp8_paged_logits_fallback",
    "indexer_kv_hadamard_fallback",
    "indexer_q_rope_fp8_fallback",
    "indexer_q_rope_hadamard_bf16_fallback",
    "indexer_select_bf16_fallback",
    "indexer_select_fp8_paged_fallback",
    "k_norm_rope_cache_fallback",
    "linear_bf16_fp32_fallback",
    "norm_rope_inplace_fallback",
    "pack_indexer_fp8_paged_cache_ref",
    "paged_mqa_attention_fallback",
    "q_kv_norm_rope_cache_fallback",
    "q_norm_rope_fallback",
    "quantize_fp8_activation_ref",
    "quantize_indexer_fp8_cache_ref",
    "quantize_indexer_fp8_paged_cache_ref",
    "quantized_linear_fp8_pair_shared_activation_ref",
    "quantized_linear_ref",
    "rms_norm_fallback",
    "silu_and_mul_clamp_fallback",
    "store_compressed_fallback",
    "store_indexer_fallback",
    "store_indexer_fp8_cache_fallback",
    "store_swa_fallback",
    "topk_transform_512_full_fallback",
    "wo_a_grouped_projection_fallback",
]

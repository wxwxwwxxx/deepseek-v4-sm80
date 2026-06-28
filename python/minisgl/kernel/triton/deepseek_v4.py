from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_clamp_kernel(
    gate_ptr,
    up_ptr,
    weights_ptr,
    out_ptr,
    n_elements,
    hidden_dim: tl.constexpr,
    swiglu_limit: tl.constexpr,
    has_weights: tl.constexpr,
    weights_mode: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if swiglu_limit > 0.0:
        up = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)
        gate = tl.minimum(gate, swiglu_limit)

    out = gate * tl.sigmoid(gate) * up
    if has_weights:
        if weights_mode == 0:
            weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            rows = offsets // hidden_dim
            weights = tl.load(weights_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        out *= weights
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _rotary_tail_kernel(
    x_ptr,
    positions_ptr,
    n_rows: tl.constexpr,
    heads_per_token: tl.constexpr,
    dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    log_base: tl.constexpr,
    inverse: tl.constexpr,
    use_scaling: tl.constexpr,
    factor: tl.constexpr,
    low: tl.constexpr,
    high: tl.constexpr,
    scale_denom: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    pair_offsets = tl.arange(0, BLOCK_HALF)
    pair_mask = pair_offsets < (rotary_dim // 2)
    token = row // heads_per_token
    pos = tl.load(positions_ptr + token).to(tl.float32)

    inv_freq = tl.exp(-((2.0 * pair_offsets.to(tl.float32)) / rotary_dim) * log_base)
    if use_scaling:
        ramp = (pair_offsets.to(tl.float32) - low) / scale_denom
        ramp = tl.minimum(tl.maximum(ramp, 0.0), 1.0)
        smooth = 1.0 - ramp
        inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth

    theta = pos * inv_freq
    if inverse:
        theta = -theta
    cos = tl.cos(theta)
    sin = tl.sin(theta)

    tail = dim - rotary_dim
    a_offsets = row * dim + tail + pair_offsets * 2
    b_offsets = a_offsets + 1
    a = tl.load(x_ptr + a_offsets, mask=pair_mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + b_offsets, mask=pair_mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + a_offsets, a * cos - b * sin, mask=pair_mask)
    tl.store(x_ptr + b_offsets, a * sin + b * cos, mask=pair_mask)


@triton.jit
def _q_norm_rope_kernel(
    q_ptr,
    positions_ptr,
    n_rows: tl.constexpr,
    heads_per_token: tl.constexpr,
    dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    log_base: tl.constexpr,
    use_scaling: tl.constexpr,
    factor: tl.constexpr,
    low: tl.constexpr,
    high: tl.constexpr,
    scale_denom: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim
    row_base = row * dim
    values = tl.load(q_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / dim
    scale = tl.rsqrt(mean_square + eps)
    normed = values * scale

    pair_offsets = tl.arange(0, BLOCK_HALF)
    pair_mask = pair_offsets < (rotary_dim // 2)
    token = row // heads_per_token
    pos = tl.load(positions_ptr + token).to(tl.float32)
    inv_freq = tl.exp(-((2.0 * pair_offsets.to(tl.float32)) / rotary_dim) * log_base)
    if use_scaling:
        ramp = (pair_offsets.to(tl.float32) - low) / scale_denom
        ramp = tl.minimum(tl.maximum(ramp, 0.0), 1.0)
        smooth = 1.0 - ramp
        inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth
    theta = pos * inv_freq
    cos = tl.cos(theta)
    sin = tl.sin(theta)

    tail = dim - rotary_dim
    a_offsets = row_base + tail + pair_offsets * 2
    b_offsets = a_offsets + 1
    a = tl.load(q_ptr + a_offsets, mask=pair_mask, other=0.0).to(tl.float32) * scale
    b = tl.load(q_ptr + b_offsets, mask=pair_mask, other=0.0).to(tl.float32) * scale
    tl.store(q_ptr + row_base + offsets, normed, mask=mask)
    tl.store(q_ptr + a_offsets, a * cos - b * sin, mask=pair_mask)
    tl.store(q_ptr + b_offsets, a * sin + b * cos, mask=pair_mask)


@triton.jit
def _store_cache_kernel(
    kv_ptr,
    loc_ptr,
    cache_ptr,
    n_rows,
    dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = row_offsets < n_rows
    dim_mask = dim_offsets < dim
    locs = tl.load(loc_ptr + row_offsets, mask=row_mask, other=-1).to(tl.int64)
    values = tl.load(
        kv_ptr + row_offsets[:, None] * dim + dim_offsets[None, :],
        mask=row_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    tl.store(
        cache_ptr + locs[:, None] * dim + dim_offsets[None, :],
        values,
        mask=row_mask[:, None] & dim_mask[None, :] & (locs[:, None] >= 0),
    )


@triton.jit
def _pad_indices_kernel(
    indices_ptr,
    out_ptr,
    n_rows,
    in_width: tl.constexpr,
    out_width: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = n_rows * out_width
    mask = offsets < total
    cols = offsets % out_width
    rows = offsets // out_width
    values = tl.load(
        indices_ptr + rows * in_width + cols,
        mask=mask & (cols < in_width),
        other=-1,
    )
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def _fp4_e2m1_value(nibble):
    mag = nibble & 0x07
    value = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )
    return tl.where(nibble >= 8, -value, value)


@triton.jit
def _fp8_e4m3fn_value(bits):
    sign = bits & 0x80
    exp = (bits >> 3) & 0x0F
    mant = bits & 0x07
    abs_bits = bits & 0x7F
    normal = tl.exp2(exp.to(tl.float32) - 7.0) * (1.0 + mant.to(tl.float32) * 0.125)
    subnormal = mant.to(tl.float32) * 0.001953125
    value = tl.where(exp == 0, subnormal, normal)
    value = tl.where(abs_bits == 0, 0.0, value)
    return tl.where(sign != 0, -value, value)


@triton.jit
def _quantized_linear_fp8_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            weight_ptr + offs_n[None, :] * K + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        ).to(tl.int32)
        b = _fp8_e4m3fn_value(b).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(
                scale_ptr + (offs_n[None, :] // 128) * SCALE_K + (offs_k[:, None] // 128),
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                other=0.0,
            ).to(tl.float32)
            b *= scale
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _quantized_linear_fp4_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WEIGHT_K_BYTES: tl.constexpr,
    SCALE_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        packed = tl.load(
            weight_ptr + offs_n[None, :] * WEIGHT_K_BYTES + (offs_k[:, None] // 2),
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0,
        ).to(tl.int32)
        nibble = tl.where((offs_k[:, None] & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
        b = _fp4_e2m1_value(nibble).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(
                scale_ptr + offs_n[None, :] * SCALE_K + (offs_k[:, None] // 32),
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                other=0.0,
            ).to(tl.float32)
            b *= scale
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _rope_scaling(
    *,
    rotary_dim: int,
    base: float,
    original_seq_len: int,
    beta_fast: int,
    beta_slow: int,
) -> tuple[bool, int, int]:
    if original_seq_len <= 0:
        return False, 0, 0

    def correction_dim(num_rotations: float) -> float:
        return rotary_dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), rotary_dim // 2 - 1)
    return True, low, high


def _heads_per_token(x: torch.Tensor, positions: torch.Tensor) -> int | None:
    if positions.numel() == 0:
        return None
    dim = x.shape[-1]
    rows = x.numel() // dim
    if rows % positions.numel() != 0:
        return None
    return rows // positions.numel()


def silu_and_mul_clamp(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if gate.shape != up.shape or gate.numel() == 0 or not gate.is_cuda or not up.is_cuda:
        return None
    if gate.stride(-1) != 1 or up.stride(-1) != 1:
        return None

    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = torch.empty(gate_c.shape, dtype=torch.float32, device=gate_c.device)
    hidden_dim = gate_c.shape[-1]
    has_weights = weights is not None
    weights_mode = 0
    weights_c = gate_c
    if weights is not None:
        if not weights.is_cuda:
            return None
        weights_c = weights.contiguous()
        if weights_c.numel() == gate_c.numel():
            weights_mode = 0
        elif weights_c.numel() == gate_c.numel() // hidden_dim:
            weights_mode = 1
        else:
            return None
    block = 1024
    grid = (triton.cdiv(gate_c.numel(), block),)
    _silu_and_mul_clamp_kernel[grid](
        gate_c,
        up_c,
        weights_c,
        out,
        gate_c.numel(),
        hidden_dim=hidden_dim,
        swiglu_limit=float(swiglu_limit),
        has_weights=has_weights,
        weights_mode=weights_mode,
        BLOCK=block,
    )
    return out


def apply_rotary_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    inverse: bool,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> bool:
    if (
        rotary_dim <= 0
        or x.numel() == 0
        or not x.is_cuda
        or not positions.is_cuda
        or not x.is_contiguous()
    ):
        return False
    heads = _heads_per_token(x, positions)
    if heads is None:
        return False
    dim = x.shape[-1]
    rows = x.numel() // dim
    use_scaling, low, high = _rope_scaling(
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    block_half = triton.next_power_of_2(rotary_dim // 2)
    _rotary_tail_kernel[(rows,)](
        x,
        positions.contiguous(),
        n_rows=rows,
        heads_per_token=heads,
        dim=dim,
        rotary_dim=rotary_dim,
        log_base=math.log(base),
        inverse=bool(inverse),
        use_scaling=use_scaling,
        factor=float(factor),
        low=low,
        high=high,
        scale_denom=max(high - low, 1),
        BLOCK_HALF=block_half,
    )
    return True


def q_norm_rope(
    q: torch.Tensor,
    positions: torch.Tensor,
    *,
    rms_norm_eps: float,
    rotary_dim: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> bool:
    if (
        q.ndim != 3
        or q.numel() == 0
        or rotary_dim <= 0
        or not q.is_cuda
        or not positions.is_cuda
        or not q.is_contiguous()
    ):
        return False
    heads = _heads_per_token(q, positions)
    if heads is None:
        return False
    dim = q.shape[-1]
    rows = q.numel() // dim
    use_scaling, low, high = _rope_scaling(
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    block_d = triton.next_power_of_2(dim)
    block_half = triton.next_power_of_2(rotary_dim // 2)
    _q_norm_rope_kernel[(rows,)](
        q,
        positions.contiguous(),
        n_rows=rows,
        heads_per_token=heads,
        dim=dim,
        rotary_dim=rotary_dim,
        eps=float(rms_norm_eps),
        log_base=math.log(base),
        use_scaling=use_scaling,
        factor=float(factor),
        low=low,
        high=high,
        scale_denom=max(high - low, 1),
        BLOCK_D=block_d,
        BLOCK_HALF=block_half,
    )
    return True


def store_cache(cache: torch.Tensor, kv: torch.Tensor, loc: torch.Tensor) -> bool:
    if kv.numel() == 0:
        return True
    if not cache.is_cuda or not kv.is_cuda or not loc.is_cuda or not cache.is_contiguous():
        return False
    dim = cache.shape[-1]
    kv_2d = kv.reshape(-1, dim).contiguous()
    loc_c = loc.reshape(-1).contiguous()
    if loc_c.numel() != kv_2d.shape[0]:
        return False
    block_m = 16
    block_d = triton.next_power_of_2(min(dim, 1024))
    grid = (triton.cdiv(kv_2d.shape[0], block_m), triton.cdiv(dim, block_d))
    _store_cache_kernel[grid](
        kv_2d,
        loc_c,
        cache,
        kv_2d.shape[0],
        dim=dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    return True


def topk_transform_512(indices: torch.Tensor, *, width: int) -> torch.Tensor | None:
    if indices.numel() == 0 or not indices.is_cuda:
        return None
    if indices.shape[-1] == width:
        return indices
    if indices.ndim < 2:
        return None
    in_width = indices.shape[-1]
    n_rows = indices.numel() // in_width
    out = torch.empty((*indices.shape[:-1], width), dtype=indices.dtype, device=indices.device)
    block = 1024
    _pad_indices_kernel[(triton.cdiv(out.numel(), block),)](
        indices.contiguous(),
        out,
        n_rows,
        in_width=in_width,
        out_width=width,
        BLOCK=block,
    )
    return out


def _flatten_linear_input(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]] | None:
    if x.ndim == 0 or x.shape[-1] == 0 or x.numel() == 0:
        return None
    if x.dtype is not torch.bfloat16 or not x.is_cuda or x.stride(-1) != 1:
        return None
    shape = tuple(x.shape[:-1])
    return x.contiguous().view(-1, x.shape[-1]), shape


def quantized_linear_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor | None:
    flattened = _flatten_linear_input(x)
    if flattened is None:
        return None
    if weight.ndim != 2 or not weight.is_cuda or weight.shape[-1] != x.shape[-1]:
        return None
    x_2d, leading_shape = flattened
    m, k = x_2d.shape
    if m > 16:
        return None
    weight_c = weight.contiguous().view(torch.uint8)
    scale_c = (
        scale.float().contiguous()
        if scale is not None
        else weight_c.new_empty((1, 1), dtype=torch.float32)
    )
    out = torch.empty((*leading_shape, weight.shape[0]), dtype=x.dtype, device=x.device)
    out_2d = out.view(x_2d.shape[0], weight.shape[0])
    n = weight.shape[0]
    block_m = 16 if m <= 16 else 32
    block_n = 64
    block_k = 64
    _quantized_linear_fp8_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x_2d,
        weight_c,
        scale_c,
        out_2d,
        M=m,
        N=n,
        K=k,
        SCALE_K=triton.cdiv(k, 128),
        HAS_SCALE=scale is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


def quantized_linear_fp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor | None:
    flattened = _flatten_linear_input(x)
    if flattened is None:
        return None
    if weight.ndim != 2 or not weight.is_cuda or weight.shape[-1] * 2 < x.shape[-1]:
        return None
    x_2d, leading_shape = flattened
    m, k = x_2d.shape
    if m > 8:
        return None
    weight_c = weight.contiguous()
    scale_c = (
        scale.float().contiguous()
        if scale is not None
        else weight_c.new_empty((1, 1), dtype=torch.float32)
    )
    out = torch.empty((*leading_shape, weight_c.shape[0]), dtype=x.dtype, device=x.device)
    out_2d = out.view(x_2d.shape[0], weight_c.shape[0])
    n = weight_c.shape[0]
    if k % 2 != 0:
        return None
    block_m = 16 if m <= 16 else 32
    block_n = 64
    block_k = 64
    _quantized_linear_fp4_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x_2d,
        weight_c,
        scale_c,
        out_2d,
        M=m,
        N=n,
        K=k,
        WEIGHT_K_BYTES=weight_c.shape[-1],
        SCALE_K=triton.cdiv(k, 32),
        HAS_SCALE=scale is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out


__all__ = [
    "apply_rotary_tail",
    "q_norm_rope",
    "quantized_linear_fp4",
    "quantized_linear_fp8",
    "silu_and_mul_clamp",
    "store_cache",
    "topk_transform_512",
]

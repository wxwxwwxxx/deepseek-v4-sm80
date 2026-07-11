from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_E4M3FN_TO_BF16_LUTS: dict[tuple[str, int | None], torch.Tensor] = {}


def _get_e4m3fn_to_bf16_lut(device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    key = (device.type, device.index)
    lut = _E4M3FN_TO_BF16_LUTS.get(key)
    if lut is not None and lut.device == device:
        return lut

    values = []
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exp_bits = (byte >> 3) & 0x0F
        mant = byte & 0x07
        if exp_bits == 0:
            value = (mant / 8.0) * 2.0**-6
        else:
            value = (1.0 + mant / 8.0) * 2.0 ** (exp_bits - 7)
        values.append(sign * value)

    lut = torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).to(device)
    _E4M3FN_TO_BF16_LUTS[key] = lut
    return lut


def warmup_indexer_fp8_lut(device: torch.device) -> bool:
    _get_e4m3fn_to_bf16_lut(torch.device(device))
    return True


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
def _rms_norm_bf16_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim
    values = tl.load(x_ptr + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / dim
    scale = tl.rsqrt(mean_square + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = values * scale * weight
    tl.store(out_ptr + row * dim + offsets, out, mask=mask)


@triton.jit
def _rms_norm_pair_bf16_kernel(
    q_ptr,
    q_weight_ptr,
    q_out_ptr,
    kv_ptr,
    kv_weight_ptr,
    kv_out_ptr,
    q_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    task = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)

    if task == 0:
        dim = q_dim
        x_ptr = q_ptr + row * q_dim
        weight_ptr = q_weight_ptr
        out_ptr = q_out_ptr + row * q_dim
    else:
        dim = kv_dim
        x_ptr = kv_ptr + row * kv_dim
        weight_ptr = kv_weight_ptr
        out_ptr = kv_out_ptr + row * kv_dim

    mask = offsets < dim
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / dim
    scale = tl.rsqrt(mean_square + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = values * scale * weight
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
    theta = theta - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
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

    tail = dim - rotary_dim
    tail_offsets = offsets - tail
    tail_mask = mask & (tail_offsets >= 0)
    tail_indices = tl.maximum(tail_offsets, 0)
    pair_offsets = tail_indices // 2
    token = row // heads_per_token
    pos = tl.load(positions_ptr + token).to(tl.float32)
    inv_freq = tl.exp(-((2.0 * pair_offsets.to(tl.float32)) / rotary_dim) * log_base)
    if use_scaling:
        ramp = (pair_offsets.to(tl.float32) - low) / scale_denom
        ramp = tl.minimum(tl.maximum(ramp, 0.0), 1.0)
        smooth = 1.0 - ramp
        inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth
    theta = pos * inv_freq
    theta = theta - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
    cos = tl.cos(theta)
    sin = tl.sin(theta)

    a_offsets = row_base + tail + pair_offsets * 2
    b_offsets = a_offsets + 1
    a = tl.load(q_ptr + a_offsets, mask=tail_mask, other=0.0).to(tl.float32) * scale
    b = tl.load(q_ptr + b_offsets, mask=tail_mask, other=0.0).to(tl.float32) * scale
    rotated_a = a * cos - b * sin
    rotated_b = a * sin + b * cos
    rotated = tl.where((tail_indices & 1) == 0, rotated_a, rotated_b)
    out = tl.where(tail_mask, rotated, normed)
    tl.store(q_ptr + row_base + offsets, out, mask=mask)


@triton.jit
def _k_norm_rope_cache_bf16_kernel(
    kv_ptr,
    positions_ptr,
    norm_weight_ptr,
    cache_ptr,
    loc_ptr,
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

    values = tl.load(kv_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / dim
    scale = tl.rsqrt(mean_square + eps)
    normed = values * scale * weights

    loc = tl.load(loc_ptr + row).to(tl.int64)
    cache_base = loc * dim
    valid_loc = loc >= 0

    tail = dim - rotary_dim
    tail_offsets = offsets - tail
    tail_mask = mask & (tail_offsets >= 0)
    tail_indices = tl.maximum(tail_offsets, 0)
    pair_offsets = tail_indices // 2
    pos = tl.load(positions_ptr + row).to(tl.float32)
    inv_freq = tl.exp(-((2.0 * pair_offsets.to(tl.float32)) / rotary_dim) * log_base)
    if use_scaling:
        ramp = (pair_offsets.to(tl.float32) - low) / scale_denom
        ramp = tl.minimum(tl.maximum(ramp, 0.0), 1.0)
        smooth = 1.0 - ramp
        inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth

    theta = pos * inv_freq
    theta = theta - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
    cos = tl.cos(theta)
    sin = tl.sin(theta)

    a_dim_offsets = tail + pair_offsets * 2
    b_dim_offsets = a_dim_offsets + 1
    a_offsets = row_base + a_dim_offsets
    b_offsets = row_base + b_dim_offsets
    a = tl.load(kv_ptr + a_offsets, mask=tail_mask, other=0.0).to(tl.float32)
    b = tl.load(kv_ptr + b_offsets, mask=tail_mask, other=0.0).to(tl.float32)
    a_weight = tl.load(norm_weight_ptr + a_dim_offsets, mask=tail_mask, other=0.0).to(tl.float32)
    b_weight = tl.load(norm_weight_ptr + b_dim_offsets, mask=tail_mask, other=0.0).to(tl.float32)
    a = a * scale * a_weight
    b = b * scale * b_weight
    rotated_a = a * cos - b * sin
    rotated_b = a * sin + b * cos
    rotated = tl.where((tail_indices & 1) == 0, rotated_a, rotated_b)
    out = tl.where(tail_mask, rotated, normed)
    tl.store(kv_ptr + row_base + offsets, out, mask=mask)
    tl.store(cache_ptr + cache_base + offsets, out, mask=mask & valid_loc)


@triton.jit
def _q_kv_norm_rope_cache_bf16_kernel(
    q_ptr,
    kv_ptr,
    positions_ptr,
    norm_weight_ptr,
    cache_ptr,
    loc_ptr,
    q_rows: tl.constexpr,
    kv_rows: tl.constexpr,
    heads_per_token: tl.constexpr,
    q_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    kv_stride0: tl.constexpr,
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
    task = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_D)

    if task == 0:
        mask = (offsets < q_dim) & (row < q_rows)
        row_base = row * q_dim
        values = tl.load(q_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / q_dim
        scale = tl.rsqrt(mean_square + eps)
        normed = values * scale

        tail = q_dim - rotary_dim
        tail_offsets = offsets - tail
        tail_mask = mask & (tail_offsets >= 0)
        tail_indices = tl.maximum(tail_offsets, 0)
        pair_offsets = tail_indices // 2
        token = row // heads_per_token
        pos = tl.load(positions_ptr + token, mask=row < q_rows, other=0).to(tl.float32)
        inv_freq = tl.exp(-((2.0 * pair_offsets.to(tl.float32)) / rotary_dim) * log_base)
        if use_scaling:
            ramp = (pair_offsets.to(tl.float32) - low) / scale_denom
            ramp = tl.minimum(tl.maximum(ramp, 0.0), 1.0)
            smooth = 1.0 - ramp
            inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth
        theta = pos * inv_freq
        theta = (
            theta - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
        )
        cos = tl.cos(theta)
        sin = tl.sin(theta)

        a_offsets = row_base + tail + pair_offsets * 2
        b_offsets = a_offsets + 1
        a = tl.load(q_ptr + a_offsets, mask=tail_mask, other=0.0).to(tl.float32) * scale
        b = tl.load(q_ptr + b_offsets, mask=tail_mask, other=0.0).to(tl.float32) * scale
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos
        rotated = tl.where((tail_indices & 1) == 0, rotated_a, rotated_b)
        out = tl.where(tail_mask, rotated, normed)
        tl.store(q_ptr + row_base + offsets, out, mask=mask)
    else:
        mask = (offsets < kv_dim) & (row < kv_rows)
        row_base = row * kv_stride0
        values = tl.load(kv_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / kv_dim
        scale = tl.rsqrt(mean_square + eps)
        normed = values * scale * weights

        loc = tl.load(loc_ptr + row, mask=row < kv_rows, other=-1).to(tl.int64)
        cache_base = loc * kv_dim
        valid_loc = loc >= 0

        tail = kv_dim - rotary_dim
        tail_offsets = offsets - tail
        tail_mask = mask & (tail_offsets >= 0)
        tail_indices = tl.maximum(tail_offsets, 0)
        pair_offsets = tail_indices // 2
        pos = tl.load(positions_ptr + row, mask=row < kv_rows, other=0).to(tl.float32)
        inv_freq = tl.exp(-((2.0 * pair_offsets.to(tl.float32)) / rotary_dim) * log_base)
        if use_scaling:
            ramp = (pair_offsets.to(tl.float32) - low) / scale_denom
            ramp = tl.minimum(tl.maximum(ramp, 0.0), 1.0)
            smooth = 1.0 - ramp
            inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth

        theta = pos * inv_freq
        theta = (
            theta - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
        )
        cos = tl.cos(theta)
        sin = tl.sin(theta)

        a_dim_offsets = tail + pair_offsets * 2
        b_dim_offsets = a_dim_offsets + 1
        a_offsets = row_base + a_dim_offsets
        b_offsets = row_base + b_dim_offsets
        a = tl.load(kv_ptr + a_offsets, mask=tail_mask, other=0.0).to(tl.float32)
        b = tl.load(kv_ptr + b_offsets, mask=tail_mask, other=0.0).to(tl.float32)
        a_weight = tl.load(norm_weight_ptr + a_dim_offsets, mask=tail_mask, other=0.0).to(
            tl.float32
        )
        b_weight = tl.load(norm_weight_ptr + b_dim_offsets, mask=tail_mask, other=0.0).to(
            tl.float32
        )
        a = a * scale * a_weight
        b = b * scale * b_weight
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos
        rotated = tl.where((tail_indices & 1) == 0, rotated_a, rotated_b)
        out = tl.where(tail_mask, rotated, normed)
        tl.store(kv_ptr + row_base + offsets, out, mask=mask)
        tl.store(cache_ptr + cache_base + offsets, out, mask=mask & valid_loc)


@triton.jit
def _compress_norm_rope_store_bf16_kernel(
    kv_ptr,
    positions_ptr,
    norm_weight_ptr,
    cache_ptr,
    loc_ptr,
    dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    log_base: tl.constexpr,
    use_scaling: tl.constexpr,
    factor: tl.constexpr,
    low: tl.constexpr,
    high: tl.constexpr,
    scale_denom: tl.constexpr,
    has_norm: tl.constexpr,
    apply_rope: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim
    row_base = row * dim

    values = tl.load(kv_ptr + row_base + offsets, mask=mask, other=0.0).to(tl.float32)
    out = values
    scale = tl.full((), 1.0, dtype=tl.float32)
    if has_norm:
        weights = tl.load(norm_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / dim
        scale = tl.rsqrt(mean_square + eps)
        out = values * scale * weights

    if apply_rope:
        tail = dim - rotary_dim
        tail_offsets = offsets - tail
        tail_mask = mask & (tail_offsets >= 0)
        tail_indices = tl.maximum(tail_offsets, 0)
        pair_offsets = tail_indices // 2
        pos = tl.load(positions_ptr + row).to(tl.float32)
        inv_freq = tl.exp(-((2.0 * pair_offsets.to(tl.float32)) / rotary_dim) * log_base)
        if use_scaling:
            ramp = (pair_offsets.to(tl.float32) - low) / scale_denom
            ramp = tl.minimum(tl.maximum(ramp, 0.0), 1.0)
            smooth = 1.0 - ramp
            inv_freq = inv_freq / factor * (1.0 - smooth) + inv_freq * smooth

        theta = pos * inv_freq
        theta = (
            theta - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
        )
        cos = tl.cos(theta)
        sin = tl.sin(theta)

        a_dim_offsets = tail + pair_offsets * 2
        b_dim_offsets = a_dim_offsets + 1
        a_offsets = row_base + a_dim_offsets
        b_offsets = row_base + b_dim_offsets
        a = tl.load(kv_ptr + a_offsets, mask=tail_mask, other=0.0).to(tl.float32)
        b = tl.load(kv_ptr + b_offsets, mask=tail_mask, other=0.0).to(tl.float32)
        if has_norm:
            a_weight = tl.load(norm_weight_ptr + a_dim_offsets, mask=tail_mask, other=0.0).to(
                tl.float32
            )
            b_weight = tl.load(norm_weight_ptr + b_dim_offsets, mask=tail_mask, other=0.0).to(
                tl.float32
            )
            a = a * scale * a_weight
            b = b * scale * b_weight
        rotated_a = a * cos - b * sin
        rotated_b = a * sin + b * cos
        rotated = tl.where((tail_indices & 1) == 0, rotated_a, rotated_b)
        out = tl.where(tail_mask, rotated, out)

    loc = tl.load(loc_ptr + row).to(tl.int64)
    valid_loc = loc >= 0
    cache_base = loc * dim
    tl.store(kv_ptr + row_base + offsets, out, mask=mask)
    tl.store(cache_ptr + cache_base + offsets, out, mask=mask & valid_loc)


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
def _build_decode_metadata_indices_kernel(
    ctx_page_table_ptr,
    table_indices_ptr,
    positions_ptr,
    page_table_ptr,
    swa_page_indices_ptr,
    swa_topk_lengths_ptr,
    c4_topk_lengths_raw_ptr,
    c4_topk_lengths_clamp1_ptr,
    c4_sparse_topk_lengths_ptr,
    c4_sparse_raw_indices_ptr,
    c4_sparse_page_indices_ptr,
    c4_sparse_full_indices_ptr,
    c128_topk_lengths_clamp1_ptr,
    c128_raw_indices_ptr,
    c128_page_indices_ptr,
    c128_full_indices_ptr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    page_table_width: tl.constexpr,
    swa_width: tl.constexpr,
    c4_width: tl.constexpr,
    c128_width: tl.constexpr,
    page_size: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    window_size: tl.constexpr,
    index_topk: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    table_idx = tl.load(table_indices_ptr + row)
    pos = tl.load(positions_ptr + row)
    seq_len = pos + 1

    page_mask = offsets < page_table_width
    logical_page_pos = offsets * page_size
    page_valid = page_mask & (logical_page_pos < max_seqlen_k)
    page_value = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + logical_page_pos,
        mask=page_valid & (logical_page_pos < ctx_page_table_width),
        other=-1,
    )
    physical_page = tl.where(page_value >= 0, page_value // page_size, page_value)
    tl.store(page_table_ptr + row * page_table_width + offsets, physical_page, mask=page_mask)

    swa_mask = offsets < swa_width
    swa_logical_pos = pos - offsets
    swa_valid = (offsets < window_size) & (swa_logical_pos >= 0)
    swa_value = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + swa_logical_pos,
        mask=swa_valid & (swa_logical_pos < ctx_page_table_width),
        other=-1,
    )
    tl.store(
        swa_page_indices_ptr + row * swa_width + offsets,
        tl.where(swa_valid, swa_value, -1),
        mask=swa_mask,
    )
    tl.store(swa_topk_lengths_ptr + row, tl.minimum(seq_len, window_size))

    c4_len = seq_len // 4
    c4_sparse_len = tl.minimum(c4_len, index_topk)
    c4_start = tl.maximum(c4_len - index_topk, 0)
    tl.store(c4_topk_lengths_raw_ptr + row, c4_len)
    tl.store(c4_topk_lengths_clamp1_ptr + row, tl.maximum(c4_len, 1))
    tl.store(c4_sparse_topk_lengths_ptr + row, c4_sparse_len)

    c4_mask = offsets < c4_width
    c4_valid = offsets < c4_sparse_len
    c4_raw = c4_start + offsets
    c4_full_pos = c4_raw * 4 + 3
    c4_full = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c4_full_pos,
        mask=c4_valid & (c4_full_pos < ctx_page_table_width),
        other=-1,
    )
    c4_full_valid = c4_valid & (c4_full >= 0)
    tl.store(
        c4_sparse_raw_indices_ptr + row * c4_width + offsets,
        tl.where(c4_valid, c4_raw, -1),
        mask=c4_mask,
    )
    tl.store(
        c4_sparse_page_indices_ptr + row * c4_width + offsets,
        tl.where(c4_full_valid, c4_full // 4, -1),
        mask=c4_mask,
    )
    tl.store(
        c4_sparse_full_indices_ptr + row * c4_width + offsets,
        tl.where(c4_full_valid, c4_full, -1),
        mask=c4_mask,
    )

    c128_len = seq_len // 128
    tl.store(c128_topk_lengths_clamp1_ptr + row, tl.maximum(c128_len, 1))
    c128_mask = offsets < c128_width
    c128_valid = offsets < c128_len
    c128_raw = offsets
    c128_full_pos = c128_raw * 128 + 127
    c128_full = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c128_full_pos,
        mask=c128_valid & (c128_full_pos < ctx_page_table_width),
        other=-1,
    )
    c128_full_valid = c128_valid & (c128_full >= 0)
    tl.store(
        c128_raw_indices_ptr + row * c128_width + offsets,
        tl.where(c128_valid, c128_raw, -1),
        mask=c128_mask,
    )
    tl.store(
        c128_page_indices_ptr + row * c128_width + offsets,
        tl.where(c128_full_valid, c128_full // 128, -1),
        mask=c128_mask,
    )
    tl.store(
        c128_full_indices_ptr + row * c128_width + offsets,
        tl.where(c128_full_valid, c128_full, -1),
        mask=c128_mask,
    )


@triton.jit
def _build_decode_metadata_indices_component_kernel(
    ctx_page_table_ptr,
    table_indices_ptr,
    positions_ptr,
    c4_page_table_ptr,
    c128_page_table_ptr,
    page_table_ptr,
    swa_page_indices_ptr,
    swa_topk_lengths_ptr,
    c4_topk_lengths_raw_ptr,
    c4_topk_lengths_clamp1_ptr,
    c4_sparse_topk_lengths_ptr,
    c4_sparse_raw_indices_ptr,
    c4_sparse_page_indices_ptr,
    c4_sparse_full_indices_ptr,
    c128_topk_lengths_clamp1_ptr,
    c128_raw_indices_ptr,
    c128_page_indices_ptr,
    c128_full_indices_ptr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    c4_page_table_width: tl.constexpr,
    c128_page_table_width: tl.constexpr,
    page_table_width: tl.constexpr,
    swa_width: tl.constexpr,
    c4_width: tl.constexpr,
    c128_width: tl.constexpr,
    page_size: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    window_size: tl.constexpr,
    index_topk: tl.constexpr,
    c4_component_page_size: tl.constexpr,
    c128_component_page_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    table_idx = tl.load(table_indices_ptr + row)
    pos = tl.load(positions_ptr + row)
    seq_len = pos + 1

    page_mask = offsets < page_table_width
    logical_page_pos = offsets * page_size
    page_valid = page_mask & (logical_page_pos < max_seqlen_k)
    page_value = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + logical_page_pos,
        mask=page_valid & (logical_page_pos < ctx_page_table_width),
        other=-1,
    )
    physical_page = tl.where(page_value >= 0, page_value // page_size, page_value)
    tl.store(page_table_ptr + row * page_table_width + offsets, physical_page, mask=page_mask)

    swa_mask = offsets < swa_width
    swa_logical_pos = pos - offsets
    swa_valid = (offsets < window_size) & (swa_logical_pos >= 0)
    swa_value = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + swa_logical_pos,
        mask=swa_valid & (swa_logical_pos < ctx_page_table_width),
        other=-1,
    )
    tl.store(
        swa_page_indices_ptr + row * swa_width + offsets,
        tl.where(swa_valid, swa_value, -1),
        mask=swa_mask,
    )
    tl.store(swa_topk_lengths_ptr + row, tl.minimum(seq_len, window_size))

    c4_len = seq_len // 4
    c4_sparse_len = tl.minimum(c4_len, index_topk)
    c4_start = tl.maximum(c4_len - index_topk, 0)
    tl.store(c4_topk_lengths_raw_ptr + row, c4_len)
    tl.store(c4_topk_lengths_clamp1_ptr + row, tl.maximum(c4_len, 1))
    tl.store(c4_sparse_topk_lengths_ptr + row, c4_sparse_len)

    c4_mask = offsets < c4_width
    c4_valid = offsets < c4_sparse_len
    c4_raw = c4_start + offsets
    c4_full_pos = c4_raw * 4 + 3
    c4_full = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c4_full_pos,
        mask=c4_valid & (c4_full_pos < ctx_page_table_width),
        other=-1,
    )
    c4_full_valid = c4_valid & (c4_full >= 0)
    c4_logical_page = c4_raw // c4_component_page_size
    c4_offset = c4_raw - c4_logical_page * c4_component_page_size
    c4_component_page = tl.load(
        c4_page_table_ptr + row * c4_page_table_width + c4_logical_page,
        mask=c4_valid & (c4_logical_page < c4_page_table_width),
        other=-1,
    )
    c4_component_valid = c4_valid & (c4_component_page >= 0)
    c4_component_loc = c4_component_page * c4_component_page_size + c4_offset
    tl.store(
        c4_sparse_raw_indices_ptr + row * c4_width + offsets,
        tl.where(c4_valid, c4_raw, -1),
        mask=c4_mask,
    )
    tl.store(
        c4_sparse_page_indices_ptr + row * c4_width + offsets,
        tl.where(c4_component_valid, c4_component_loc, -1),
        mask=c4_mask,
    )
    tl.store(
        c4_sparse_full_indices_ptr + row * c4_width + offsets,
        tl.where(c4_full_valid, c4_full, -1),
        mask=c4_mask,
    )

    c128_len = seq_len // 128
    tl.store(c128_topk_lengths_clamp1_ptr + row, tl.maximum(c128_len, 1))
    c128_mask = offsets < c128_width
    c128_valid = offsets < c128_len
    c128_raw = offsets
    c128_full_pos = c128_raw * 128 + 127
    c128_full = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c128_full_pos,
        mask=c128_valid & (c128_full_pos < ctx_page_table_width),
        other=-1,
    )
    c128_full_valid = c128_valid & (c128_full >= 0)
    c128_logical_page = c128_raw // c128_component_page_size
    c128_offset = c128_raw - c128_logical_page * c128_component_page_size
    c128_component_page = tl.load(
        c128_page_table_ptr + row * c128_page_table_width + c128_logical_page,
        mask=c128_valid & (c128_logical_page < c128_page_table_width),
        other=-1,
    )
    c128_component_valid = c128_valid & (c128_component_page >= 0)
    c128_component_loc = c128_component_page * c128_component_page_size + c128_offset
    tl.store(
        c128_raw_indices_ptr + row * c128_width + offsets,
        tl.where(c128_valid, c128_raw, -1),
        mask=c128_mask,
    )
    tl.store(
        c128_page_indices_ptr + row * c128_width + offsets,
        tl.where(c128_component_valid, c128_component_loc, -1),
        mask=c128_mask,
    )
    tl.store(
        c128_full_indices_ptr + row * c128_width + offsets,
        tl.where(c128_full_valid, c128_full, -1),
        mask=c128_mask,
    )


@triton.jit
def _direct_c4_sparse_metadata_for_replay_kernel(
    ctx_page_table_ptr,
    table_indices_ptr,
    positions_ptr,
    c4_page_table_ptr,
    dst_c4_sparse_raw_indices_ptr,
    dst_c4_sparse_page_indices_ptr,
    dst_c4_sparse_full_indices_ptr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    c4_page_table_width: tl.constexpr,
    c4_width: tl.constexpr,
    page_size: tl.constexpr,
    index_topk: tl.constexpr,
    component_loc_ownership: tl.constexpr,
    c4_component_page_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < c4_width
    table_idx = tl.load(table_indices_ptr + row)
    pos = tl.load(positions_ptr + row)
    seq_len = pos + 1

    c4_len = seq_len // 4
    c4_sparse_len = tl.minimum(c4_len, index_topk)
    c4_start = tl.maximum(c4_len - index_topk, 0)
    c4_valid = offsets < c4_sparse_len
    c4_raw = c4_start + offsets

    c4_full_pos = c4_raw * 4 + 3
    c4_full = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c4_full_pos,
        mask=c4_valid & (c4_full_pos >= 0) & (c4_full_pos < ctx_page_table_width),
        other=-1,
    )
    c4_full_valid = c4_valid & (c4_full >= 0)

    if component_loc_ownership:
        c4_logical_page = c4_raw // c4_component_page_size
        c4_offset = c4_raw - c4_logical_page * c4_component_page_size
        c4_component_page = tl.load(
            c4_page_table_ptr + row * c4_page_table_width + c4_logical_page,
            mask=c4_valid & (c4_logical_page >= 0) & (c4_logical_page < c4_page_table_width),
            other=-1,
        )
        c4_component_loc = c4_component_page * c4_component_page_size + c4_offset
        c4_page = tl.where(c4_valid & (c4_component_page >= 0), c4_component_loc, -1)
    else:
        c4_page = tl.where(c4_full_valid, c4_full // 4, -1)

    tl.store(
        dst_c4_sparse_raw_indices_ptr + row * c4_width + offsets,
        tl.where(c4_valid, c4_raw, -1),
        mask=mask,
    )
    tl.store(
        dst_c4_sparse_page_indices_ptr + row * c4_width + offsets,
        c4_page,
        mask=mask,
    )
    tl.store(
        dst_c4_sparse_full_indices_ptr + row * c4_width + offsets,
        tl.where(c4_full_valid, c4_full, -1),
        mask=mask,
    )


@triton.jit
def _direct_decode_index_metadata_for_replay_kernel(
    ctx_page_table_ptr,
    table_indices_ptr,
    positions_ptr,
    c4_page_table_ptr,
    c128_page_table_ptr,
    swa_full_to_swa_page_ptr,
    dst_swa_page_indices_ptr,
    dst_c4_sparse_raw_indices_ptr,
    dst_c4_sparse_page_indices_ptr,
    dst_c4_sparse_full_indices_ptr,
    dst_c128_raw_indices_ptr,
    dst_c128_page_indices_ptr,
    dst_c128_full_indices_ptr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    swa_full_to_swa_page_width: tl.constexpr,
    c4_page_table_width: tl.constexpr,
    c128_page_table_width: tl.constexpr,
    swa_width: tl.constexpr,
    c4_width: tl.constexpr,
    c128_width: tl.constexpr,
    page_size: tl.constexpr,
    window_size: tl.constexpr,
    index_topk: tl.constexpr,
    direct_swa: tl.constexpr,
    direct_c4: tl.constexpr,
    direct_c128: tl.constexpr,
    swa_independent: tl.constexpr,
    swa_dummy_token_start: tl.constexpr,
    swa_dummy_page: tl.constexpr,
    c4_component_page_size: tl.constexpr,
    c128_component_page_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    table_idx = tl.load(table_indices_ptr + row)
    pos = tl.load(positions_ptr + row)
    seq_len = pos + 1

    if direct_swa:
        swa_mask = offsets < swa_width
        swa_logical_pos = pos - offsets
        swa_valid = (offsets < window_size) & (swa_logical_pos >= 0)
        swa_value = tl.load(
            ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + swa_logical_pos,
            mask=swa_valid & (swa_logical_pos < ctx_page_table_width),
            other=-1,
        )
        if swa_independent:
            full_page = swa_value // page_size
            page_offset = swa_value - full_page * page_size
            full_valid = swa_valid & (swa_value >= 0)
            mapped_page = tl.load(
                swa_full_to_swa_page_ptr + full_page,
                mask=full_valid
                & (full_page >= 0)
                & (full_page < swa_full_to_swa_page_width),
                other=-1,
            )
            mapped_loc = mapped_page * page_size + page_offset
            dummy_loc = swa_dummy_page * page_size
            swa_value = tl.where(
                swa_value == swa_dummy_token_start,
                dummy_loc,
                tl.where(full_valid & (mapped_page >= 0), mapped_loc, -1),
            )
        tl.store(
            dst_swa_page_indices_ptr + row * swa_width + offsets,
            tl.where(swa_valid, swa_value, -1),
            mask=swa_mask,
        )

    if direct_c4:
        c4_len = seq_len // 4
        c4_sparse_len = tl.minimum(c4_len, index_topk)
        c4_start = tl.maximum(c4_len - index_topk, 0)
        c4_mask = offsets < c4_width
        c4_valid = offsets < c4_sparse_len
        c4_raw = c4_start + offsets
        c4_full_pos = c4_raw * 4 + 3
        c4_full = tl.load(
            ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c4_full_pos,
            mask=c4_valid & (c4_full_pos >= 0) & (c4_full_pos < ctx_page_table_width),
            other=-1,
        )
        c4_full_valid = c4_valid & (c4_full >= 0)
        c4_logical_page = c4_raw // c4_component_page_size
        c4_offset = c4_raw - c4_logical_page * c4_component_page_size
        c4_component_page = tl.load(
            c4_page_table_ptr + row * c4_page_table_width + c4_logical_page,
            mask=c4_valid & (c4_logical_page >= 0) & (c4_logical_page < c4_page_table_width),
            other=-1,
        )
        c4_component_loc = c4_component_page * c4_component_page_size + c4_offset
        tl.store(
            dst_c4_sparse_raw_indices_ptr + row * c4_width + offsets,
            tl.where(c4_valid, c4_raw, -1),
            mask=c4_mask,
        )
        tl.store(
            dst_c4_sparse_page_indices_ptr + row * c4_width + offsets,
            tl.where(c4_valid & (c4_component_page >= 0), c4_component_loc, -1),
            mask=c4_mask,
        )
        tl.store(
            dst_c4_sparse_full_indices_ptr + row * c4_width + offsets,
            tl.where(c4_full_valid, c4_full, -1),
            mask=c4_mask,
        )

    if direct_c128:
        c128_len = seq_len // 128
        c128_mask = offsets < c128_width
        c128_valid = offsets < c128_len
        c128_raw = offsets
        c128_full_pos = c128_raw * 128 + 127
        c128_full = tl.load(
            ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c128_full_pos,
            mask=c128_valid & (c128_full_pos >= 0) & (c128_full_pos < ctx_page_table_width),
            other=-1,
        )
        c128_full_valid = c128_valid & (c128_full >= 0)
        c128_logical_page = c128_raw // c128_component_page_size
        c128_offset = c128_raw - c128_logical_page * c128_component_page_size
        c128_component_page = tl.load(
            c128_page_table_ptr + row * c128_page_table_width + c128_logical_page,
            mask=c128_valid
            & (c128_logical_page >= 0)
            & (c128_logical_page < c128_page_table_width),
            other=-1,
        )
        c128_component_loc = c128_component_page * c128_component_page_size + c128_offset
        tl.store(
            dst_c128_raw_indices_ptr + row * c128_width + offsets,
            tl.where(c128_valid, c128_raw, -1),
            mask=c128_mask,
        )
        tl.store(
            dst_c128_page_indices_ptr + row * c128_width + offsets,
            tl.where(c128_valid & (c128_component_page >= 0), c128_component_loc, -1),
            mask=c128_mask,
        )
        tl.store(
            dst_c128_full_indices_ptr + row * c128_width + offsets,
            tl.where(c128_full_valid, c128_full, -1),
            mask=c128_mask,
        )


@triton.jit
def _copy_masked_compressed_locs_kernel(
    raw_out_loc_ptr,
    positions_ptr,
    c4_out_loc_ptr,
    c128_out_loc_ptr,
    rows,
    n_elements,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    active = offsets < rows
    positions = tl.load(positions_ptr + offsets, mask=active, other=0)
    raw_out_loc = tl.load(raw_out_loc_ptr + offsets, mask=active, other=0)
    seq_lens = positions + 1
    c4 = tl.where(active & ((seq_lens % 4) == 0), raw_out_loc // 4, -1)
    c128 = tl.where(active & ((seq_lens % 128) == 0), raw_out_loc // 128, -1)
    tl.store(c4_out_loc_ptr + offsets, c4, mask=mask)
    tl.store(c128_out_loc_ptr + offsets, c128, mask=mask)


@triton.jit
def _copy_component_write_locs_for_replay_kernel(
    c4_page_table_ptr,
    c128_page_table_ptr,
    c4_indexer_page_table_ptr,
    positions_ptr,
    c4_out_loc_ptr,
    c128_out_loc_ptr,
    c4_indexer_out_loc_ptr,
    rows,
    n_elements,
    c4_page_table_width: tl.constexpr,
    c128_page_table_width: tl.constexpr,
    c4_indexer_page_table_width: tl.constexpr,
    c4_component_page_size: tl.constexpr,
    c128_component_page_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    active = offsets < rows
    positions = tl.load(positions_ptr + offsets, mask=active, other=-1)
    seq_lens = positions + 1

    c4_boundary = active & ((seq_lens % 4) == 0)
    c4_raw = (seq_lens // 4) - 1
    c4_logical_page = c4_raw // c4_component_page_size
    c4_offset = c4_raw - c4_logical_page * c4_component_page_size
    c4_component_page = tl.load(
        c4_page_table_ptr + offsets * c4_page_table_width + c4_logical_page,
        mask=c4_boundary & (c4_logical_page >= 0) & (c4_logical_page < c4_page_table_width),
        other=-1,
    )
    c4_loc = c4_component_page * c4_component_page_size + c4_offset
    c4 = tl.where(c4_boundary & (c4_component_page >= 0), c4_loc, -1)

    c4_indexer_component_page = tl.load(
        c4_indexer_page_table_ptr + offsets * c4_indexer_page_table_width + c4_logical_page,
        mask=c4_boundary & (c4_logical_page >= 0) & (c4_logical_page < c4_indexer_page_table_width),
        other=-1,
    )
    c4_indexer_loc = c4_indexer_component_page * c4_component_page_size + c4_offset
    c4_indexer = tl.where(
        c4_boundary & (c4_indexer_component_page >= 0),
        c4_indexer_loc,
        -1,
    )

    c128_boundary = active & ((seq_lens % 128) == 0)
    c128_raw = (seq_lens // 128) - 1
    c128_logical_page = c128_raw // c128_component_page_size
    c128_offset = c128_raw - c128_logical_page * c128_component_page_size
    c128_component_page = tl.load(
        c128_page_table_ptr + offsets * c128_page_table_width + c128_logical_page,
        mask=c128_boundary & (c128_logical_page >= 0) & (c128_logical_page < c128_page_table_width),
        other=-1,
    )
    c128_loc = c128_component_page * c128_component_page_size + c128_offset
    c128 = tl.where(c128_boundary & (c128_component_page >= 0), c128_loc, -1)

    tl.store(c4_out_loc_ptr + offsets, c4, mask=mask)
    tl.store(c128_out_loc_ptr + offsets, c128, mask=mask)
    tl.store(c4_indexer_out_loc_ptr + offsets, c4_indexer, mask=mask)


@triton.jit
def _prep_decode_metadata_in_graph_kernel(
    ctx_page_table_ptr,
    table_indices_ptr,
    positions_ptr,
    raw_out_loc_ptr,
    materialized_seq_lens_ptr,
    c4_page_table_ptr,
    c128_page_table_ptr,
    c4_indexer_page_table_ptr,
    dst_seq_lens_ptr,
    dst_swa_topk_lengths_ptr,
    dst_c4_topk_lengths_raw_ptr,
    dst_c4_topk_lengths_clamp1_ptr,
    dst_c4_sparse_topk_lengths_ptr,
    dst_c128_topk_lengths_clamp1_ptr,
    dst_swa_page_indices_ptr,
    dst_c4_sparse_raw_indices_ptr,
    dst_c4_sparse_page_indices_ptr,
    dst_c4_sparse_full_indices_ptr,
    dst_c128_raw_indices_ptr,
    dst_c128_page_indices_ptr,
    dst_c128_full_indices_ptr,
    dst_c4_out_loc_ptr,
    dst_c128_out_loc_ptr,
    dst_c4_indexer_out_loc_ptr,
    swa_full_to_swa_page_ptr,
    dst_swa_out_loc_ptr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    swa_full_to_swa_page_width: tl.constexpr,
    c4_page_table_width: tl.constexpr,
    c128_page_table_width: tl.constexpr,
    c4_indexer_page_table_width: tl.constexpr,
    swa_width: tl.constexpr,
    c4_width: tl.constexpr,
    c128_width: tl.constexpr,
    page_size: tl.constexpr,
    window_size: tl.constexpr,
    index_topk: tl.constexpr,
    swa_independent: tl.constexpr,
    swa_dummy_token_start: tl.constexpr,
    swa_dummy_page: tl.constexpr,
    write_swa_out_loc: tl.constexpr,
    c4_component_page_size: tl.constexpr,
    c128_component_page_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    table_idx = tl.load(table_indices_ptr + row)
    pos = tl.load(positions_ptr + row)
    raw_out_loc = tl.load(raw_out_loc_ptr + row)
    seq_len = pos + 1
    materialized_seq_len = tl.maximum(tl.load(materialized_seq_lens_ptr + row), 0)
    capped_seq_len = tl.minimum(seq_len, materialized_seq_len)

    tl.store(dst_seq_lens_ptr + row, seq_len)
    tl.store(dst_swa_topk_lengths_ptr + row, tl.minimum(seq_len, window_size))

    c4_len = capped_seq_len // 4
    c4_len_clamp1 = tl.maximum(c4_len, 1)
    c4_sparse_len = tl.minimum(tl.maximum(c4_len, 0), index_topk)
    tl.store(dst_c4_topk_lengths_raw_ptr + row, c4_len)
    tl.store(dst_c4_topk_lengths_clamp1_ptr + row, c4_len_clamp1)
    tl.store(dst_c4_sparse_topk_lengths_ptr + row, c4_sparse_len)

    c128_len = capped_seq_len // 128
    tl.store(dst_c128_topk_lengths_clamp1_ptr + row, tl.maximum(c128_len, 1))

    swa_mask = offsets < swa_width
    swa_logical_pos = pos - offsets
    swa_valid = (offsets < window_size) & (swa_logical_pos >= 0)
    swa_value = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + swa_logical_pos,
        mask=swa_valid & (swa_logical_pos < ctx_page_table_width),
        other=-1,
    )
    if swa_independent:
        full_page = swa_value // page_size
        page_offset = swa_value - full_page * page_size
        full_valid = swa_valid & (swa_value >= 0)
        mapped_page = tl.load(
            swa_full_to_swa_page_ptr + full_page,
            mask=full_valid
            & (full_page >= 0)
            & (full_page < swa_full_to_swa_page_width),
            other=-1,
        )
        mapped_loc = mapped_page * page_size + page_offset
        dummy_loc = swa_dummy_page * page_size
        swa_value = tl.where(
            swa_value == swa_dummy_token_start,
            dummy_loc,
            tl.where(full_valid & (mapped_page >= 0), mapped_loc, -1),
        )
    tl.store(
        dst_swa_page_indices_ptr + row * swa_width + offsets,
        tl.where(swa_valid, swa_value, -1),
        mask=swa_mask,
    )
    if write_swa_out_loc:
        swa_out_full_page = raw_out_loc // page_size
        swa_out_page_offset = raw_out_loc - swa_out_full_page * page_size
        swa_out_valid = raw_out_loc >= 0
        swa_out_page = tl.load(
            swa_full_to_swa_page_ptr + swa_out_full_page,
            mask=swa_independent
            & swa_out_valid
            & (swa_out_full_page >= 0)
            & (swa_out_full_page < swa_full_to_swa_page_width),
            other=-1,
        )
        swa_out_loc = swa_out_page * page_size + swa_out_page_offset
        if swa_independent:
            swa_out_loc = tl.where(
                raw_out_loc == swa_dummy_token_start,
                swa_dummy_page * page_size,
                tl.where(swa_out_valid & (swa_out_page >= 0), swa_out_loc, -1),
            )
        else:
            swa_out_loc = tl.where(swa_out_valid, raw_out_loc, -1)
        tl.store(dst_swa_out_loc_ptr + row, swa_out_loc)

    c4_mask = offsets < c4_width
    c4_valid = offsets < c4_sparse_len
    c4_start = tl.maximum(c4_len - index_topk, 0)
    c4_raw = c4_start + offsets
    c4_full_pos = c4_raw * 4 + 3
    c4_full = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c4_full_pos,
        mask=c4_valid & (c4_full_pos >= 0) & (c4_full_pos < ctx_page_table_width),
        other=-1,
    )
    c4_full_valid = c4_valid & (c4_full >= 0)
    c4_logical_page = c4_raw // c4_component_page_size
    c4_offset = c4_raw - c4_logical_page * c4_component_page_size
    c4_component_page = tl.load(
        c4_page_table_ptr + row * c4_page_table_width + c4_logical_page,
        mask=c4_valid & (c4_logical_page >= 0) & (c4_logical_page < c4_page_table_width),
        other=-1,
    )
    c4_component_loc = c4_component_page * c4_component_page_size + c4_offset
    tl.store(
        dst_c4_sparse_raw_indices_ptr + row * c4_width + offsets,
        tl.where(c4_valid, c4_raw, -1),
        mask=c4_mask,
    )
    tl.store(
        dst_c4_sparse_page_indices_ptr + row * c4_width + offsets,
        tl.where(c4_valid & (c4_component_page >= 0), c4_component_loc, -1),
        mask=c4_mask,
    )
    tl.store(
        dst_c4_sparse_full_indices_ptr + row * c4_width + offsets,
        tl.where(c4_full_valid, c4_full, -1),
        mask=c4_mask,
    )

    c128_mask = offsets < c128_width
    c128_valid = offsets < c128_len
    c128_raw = offsets
    c128_full_pos = c128_raw * 128 + 127
    c128_full = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + c128_full_pos,
        mask=c128_valid & (c128_full_pos >= 0) & (c128_full_pos < ctx_page_table_width),
        other=-1,
    )
    c128_full_valid = c128_valid & (c128_full >= 0)
    c128_logical_page = c128_raw // c128_component_page_size
    c128_offset = c128_raw - c128_logical_page * c128_component_page_size
    c128_component_page = tl.load(
        c128_page_table_ptr + row * c128_page_table_width + c128_logical_page,
        mask=c128_valid
        & (c128_logical_page >= 0)
        & (c128_logical_page < c128_page_table_width),
        other=-1,
    )
    c128_component_loc = c128_component_page * c128_component_page_size + c128_offset
    tl.store(
        dst_c128_raw_indices_ptr + row * c128_width + offsets,
        tl.where(c128_valid, c128_raw, -1),
        mask=c128_mask,
    )
    tl.store(
        dst_c128_page_indices_ptr + row * c128_width + offsets,
        tl.where(c128_valid & (c128_component_page >= 0), c128_component_loc, -1),
        mask=c128_mask,
    )
    tl.store(
        dst_c128_full_indices_ptr + row * c128_width + offsets,
        tl.where(c128_full_valid, c128_full, -1),
        mask=c128_mask,
    )

    c4_boundary = (seq_len % 4) == 0
    c4_write_raw = (seq_len // 4) - 1
    c4_write_logical_page = c4_write_raw // c4_component_page_size
    c4_write_offset = c4_write_raw - c4_write_logical_page * c4_component_page_size
    c4_write_component_page = tl.load(
        c4_page_table_ptr + row * c4_page_table_width + c4_write_logical_page,
        mask=c4_boundary
        & (c4_write_logical_page >= 0)
        & (c4_write_logical_page < c4_page_table_width),
        other=-1,
    )
    c4_write_loc = c4_write_component_page * c4_component_page_size + c4_write_offset
    c4_write_loc = tl.where(c4_boundary & (c4_write_component_page >= 0), c4_write_loc, -1)

    c4_indexer_component_page = tl.load(
        c4_indexer_page_table_ptr
        + row * c4_indexer_page_table_width
        + c4_write_logical_page,
        mask=c4_boundary
        & (c4_write_logical_page >= 0)
        & (c4_write_logical_page < c4_indexer_page_table_width),
        other=-1,
    )
    c4_indexer_loc = c4_indexer_component_page * c4_component_page_size + c4_write_offset
    c4_indexer_loc = tl.where(
        c4_boundary & (c4_indexer_component_page >= 0),
        c4_indexer_loc,
        -1,
    )

    c128_boundary = (seq_len % 128) == 0
    c128_write_raw = (seq_len // 128) - 1
    c128_write_logical_page = c128_write_raw // c128_component_page_size
    c128_write_offset = c128_write_raw - c128_write_logical_page * c128_component_page_size
    c128_write_component_page = tl.load(
        c128_page_table_ptr + row * c128_page_table_width + c128_write_logical_page,
        mask=c128_boundary
        & (c128_write_logical_page >= 0)
        & (c128_write_logical_page < c128_page_table_width),
        other=-1,
    )
    c128_write_loc = c128_write_component_page * c128_component_page_size + c128_write_offset
    c128_write_loc = tl.where(
        c128_boundary & (c128_write_component_page >= 0),
        c128_write_loc,
        -1,
    )

    # Keep raw_out_loc live in the captured dependency set for fallback parity
    # with SGLang's raw-decode surface; component locs are the active output.
    _ = raw_out_loc
    tl.store(dst_c4_out_loc_ptr + row, c4_write_loc)
    tl.store(dst_c128_out_loc_ptr + row, c128_write_loc)
    tl.store(dst_c4_indexer_out_loc_ptr + row, c4_indexer_loc)


@triton.jit
def _copy_1d_i32(src_ptr, dst_ptr, offsets, n):
    mask = offsets < n
    values = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _copy_2d_i32_fill(
    src_ptr,
    dst_ptr,
    offsets,
    rows: tl.constexpr,
    dst_width: tl.constexpr,
    src_width: tl.constexpr,
    fill_value: tl.constexpr,
):
    total = rows * dst_width
    mask = offsets < total
    row = offsets // dst_width
    col = offsets - row * dst_width
    has_src = col < src_width
    values = tl.load(
        src_ptr + row * src_width + col,
        mask=mask & has_src,
        other=fill_value,
    )
    values = tl.where(has_src, values, fill_value)
    tl.store(dst_ptr + row * dst_width + col, values, mask=mask)


@triton.jit
def _copy_decode_metadata_for_replay_kernel(
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
    rows: tl.constexpr,
    graph_inputs_bound: tl.constexpr,
    dst_page_table_width: tl.constexpr,
    src_page_table_width: tl.constexpr,
    dst_swa_page_indices_width: tl.constexpr,
    src_swa_page_indices_width: tl.constexpr,
    dst_c4_sparse_raw_indices_width: tl.constexpr,
    src_c4_sparse_raw_indices_width: tl.constexpr,
    dst_c4_sparse_page_indices_width: tl.constexpr,
    src_c4_sparse_page_indices_width: tl.constexpr,
    dst_c4_sparse_full_indices_width: tl.constexpr,
    src_c4_sparse_full_indices_width: tl.constexpr,
    dst_c128_raw_indices_width: tl.constexpr,
    src_c128_raw_indices_width: tl.constexpr,
    dst_c128_page_indices_width: tl.constexpr,
    src_c128_page_indices_width: tl.constexpr,
    dst_c128_full_indices_width: tl.constexpr,
    src_c128_full_indices_width: tl.constexpr,
    skip_swa_page_indices: tl.constexpr,
    skip_c4_sparse_indices: tl.constexpr,
    skip_c128_indices: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    field = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    if field == 0:
        if not graph_inputs_bound:
            _copy_1d_i32(src_raw_out_loc, dst_raw_out_loc, offsets, rows)
    elif field == 1:
        _copy_1d_i32(src_seq_lens, dst_seq_lens, offsets, rows)
    elif field == 2:
        _copy_1d_i32(src_req_seq_lens, dst_req_seq_lens, offsets, rows)
    elif field == 3:
        _copy_1d_i32(src_extend_lens, dst_extend_lens, offsets, rows)
    elif field == 4:
        if not graph_inputs_bound:
            _copy_1d_i32(src_positions, dst_positions, offsets, rows)
    elif field == 5:
        _copy_1d_i32(src_req_table_indices, dst_req_table_indices, offsets, rows)
    elif field == 6:
        _copy_1d_i32(src_swa_topk_lengths, dst_swa_topk_lengths, offsets, rows)
    elif field == 7:
        _copy_1d_i32(src_c4_topk_lengths_raw, dst_c4_topk_lengths_raw, offsets, rows)
    elif field == 8:
        _copy_1d_i32(src_c4_topk_lengths_clamp1, dst_c4_topk_lengths_clamp1, offsets, rows)
    elif field == 9:
        _copy_1d_i32(src_c4_sparse_topk_lengths, dst_c4_sparse_topk_lengths, offsets, rows)
    elif field == 10:
        _copy_1d_i32(
            src_c128_topk_lengths_clamp1,
            dst_c128_topk_lengths_clamp1,
            offsets,
            rows,
        )
    elif field == 11:
        _copy_1d_i32(src_cu_seqlens_q, dst_cu_seqlens_q, offsets, rows + 1)
    elif field == 12:
        _copy_2d_i32_fill(
            src_page_table,
            dst_page_table,
            offsets,
            rows,
            dst_page_table_width,
            src_page_table_width,
            0,
        )
    elif field == 13:
        if not skip_swa_page_indices:
            _copy_2d_i32_fill(
                src_swa_page_indices,
                dst_swa_page_indices,
                offsets,
                rows,
                dst_swa_page_indices_width,
                src_swa_page_indices_width,
                -1,
            )
    elif field == 14:
        if not skip_c4_sparse_indices:
            _copy_2d_i32_fill(
                src_c4_sparse_raw_indices,
                dst_c4_sparse_raw_indices,
                offsets,
                rows,
                dst_c4_sparse_raw_indices_width,
                src_c4_sparse_raw_indices_width,
                -1,
            )
    elif field == 15:
        if not skip_c4_sparse_indices:
            _copy_2d_i32_fill(
                src_c4_sparse_page_indices,
                dst_c4_sparse_page_indices,
                offsets,
                rows,
                dst_c4_sparse_page_indices_width,
                src_c4_sparse_page_indices_width,
                -1,
            )
    elif field == 16:
        if not skip_c4_sparse_indices:
            _copy_2d_i32_fill(
                src_c4_sparse_full_indices,
                dst_c4_sparse_full_indices,
                offsets,
                rows,
                dst_c4_sparse_full_indices_width,
                src_c4_sparse_full_indices_width,
                -1,
            )
    elif field == 17:
        if not skip_c128_indices:
            _copy_2d_i32_fill(
                src_c128_raw_indices,
                dst_c128_raw_indices,
                offsets,
                rows,
                dst_c128_raw_indices_width,
                src_c128_raw_indices_width,
                -1,
            )
    elif field == 18:
        if not skip_c128_indices:
            _copy_2d_i32_fill(
                src_c128_page_indices,
                dst_c128_page_indices,
                offsets,
                rows,
                dst_c128_page_indices_width,
                src_c128_page_indices_width,
                -1,
            )
    else:
        if not skip_c128_indices:
            _copy_2d_i32_fill(
                src_c128_full_indices,
                dst_c128_full_indices,
                offsets,
                rows,
                dst_c128_full_indices_width,
                src_c128_full_indices_width,
                -1,
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
def _hc_split_pre_kernel(
    mixes_ptr,
    x_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    comb_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    hc_mult: tl.constexpr,
    mix_hc: tl.constexpr,
    eps: tl.constexpr,
    sinkhorn_steps: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    d_block = tl.program_id(1)
    hc_offsets = tl.arange(0, BLOCK_HC)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hc_mask = hc_offsets < hc_mult
    d_mask = d_offsets < hidden

    mix_base = token * mix_hc
    scale0 = tl.load(scale_ptr + 0).to(tl.float32)
    scale1 = tl.load(scale_ptr + 1).to(tl.float32)
    scale2 = tl.load(scale_ptr + 2).to(tl.float32)

    pre_logits = tl.load(mixes_ptr + mix_base + hc_offsets, mask=hc_mask, other=0.0).to(
        tl.float32
    ) * scale0 + tl.load(base_ptr + hc_offsets, mask=hc_mask, other=0.0).to(tl.float32)
    pre = tl.sigmoid(pre_logits) + eps
    pre = tl.where(hc_mask, pre, 0.0)

    post_start = hc_mult
    post_logits = tl.load(
        mixes_ptr + mix_base + post_start + hc_offsets, mask=hc_mask, other=0.0
    ).to(tl.float32) * scale1 + tl.load(
        base_ptr + post_start + hc_offsets, mask=hc_mask, other=0.0
    ).to(
        tl.float32
    )
    post = 2.0 * tl.sigmoid(post_logits)
    post = tl.where(hc_mask, post, 0.0)

    rows = tl.arange(0, BLOCK_HC)[:, None]
    cols = tl.arange(0, BLOCK_HC)[None, :]
    matrix_mask = (rows < hc_mult) & (cols < hc_mult)
    comb_start = 2 * hc_mult
    comb_offsets = comb_start + rows * hc_mult + cols
    comb_logits = tl.load(mixes_ptr + mix_base + comb_offsets, mask=matrix_mask, other=0.0).to(
        tl.float32
    ) * scale2 + tl.load(base_ptr + comb_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
    comb_logits = tl.where(matrix_mask, comb_logits, -3.4028234663852886e38)
    row_max = tl.max(comb_logits, axis=1)
    exp_logits = tl.where(matrix_mask, tl.exp(comb_logits - row_max[:, None]), 0.0)
    row_sum = tl.sum(exp_logits, axis=1)
    comb = exp_logits / tl.maximum(row_sum[:, None], eps)
    comb = tl.where(matrix_mask, comb + eps, 0.0)

    col_sum = tl.sum(comb, axis=0)
    comb = tl.where(matrix_mask, comb / (col_sum[None, :] + eps), 0.0)
    for _ in range(0, sinkhorn_steps):
        row_sum_iter = tl.sum(comb, axis=1)
        comb = tl.where(matrix_mask, comb / (row_sum_iter[:, None] + eps), 0.0)
        col_sum_iter = tl.sum(comb, axis=0)
        comb = tl.where(matrix_mask, comb / (col_sum_iter[None, :] + eps), 0.0)

    if d_block == 0:
        tl.store(post_ptr + token * hc_mult + hc_offsets, post, mask=hc_mask)
        tl.store(
            comb_ptr + token * hc_mult * hc_mult + rows * hc_mult + cols,
            comb,
            mask=matrix_mask,
        )

    x_offsets = token * hc_mult * hidden + hc_offsets[:, None] * hidden + d_offsets[None, :]
    x_values = tl.load(x_ptr + x_offsets, mask=hc_mask[:, None] & d_mask[None, :], other=0.0)
    y = tl.sum(pre.to(tl.bfloat16)[:, None].to(tl.float32) * x_values.to(tl.float32), axis=0)
    tl.store(y_ptr + token * hidden + d_offsets, y, mask=d_mask)


@triton.jit
def _hc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    d_block = tl.program_id(1)
    i_offsets = tl.arange(0, BLOCK_HC)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    i_mask = i_offsets < hc_mult
    d_mask = d_offsets < hidden

    x_values = tl.load(
        x_ptr + token * hidden + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    post = tl.load(
        post_ptr + token * hc_mult + i_offsets,
        mask=i_mask,
        other=0.0,
    ).to(tl.float32)
    out = post[:, None] * x_values[None, :]

    for j in range(0, BLOCK_HC):
        valid_j = j < hc_mult
        residual = tl.load(
            residual_ptr + token * hc_mult * hidden + j * hidden + d_offsets,
            mask=valid_j & d_mask,
            other=0.0,
        ).to(tl.float32)
        comb = tl.load(
            comb_ptr + token * hc_mult * hc_mult + j * hc_mult + i_offsets,
            mask=valid_j & i_mask,
            other=0.0,
        ).to(tl.float32)
        out += comb[:, None] * residual[None, :]

    tl.store(
        out_ptr + token * hc_mult * hidden + i_offsets[:, None] * hidden + d_offsets[None, :],
        out,
        mask=i_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _hc_prenorm_split_pre_kernel(
    mixes_ptr,
    x_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    hc_mult: tl.constexpr,
    mix_hc: tl.constexpr,
    eps: tl.constexpr,
    norm_eps: tl.constexpr,
    sinkhorn_steps: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    total = hidden * hc_mult
    token_base = token * total

    sq_sum = tl.zeros((), dtype=tl.float32)
    for start in range(0, total, BLOCK_N):
        offsets = start + tl.arange(0, BLOCK_N)
        mask = offsets < total
        values = tl.load(x_ptr + token_base + offsets, mask=mask, other=0.0).to(tl.float32)
        sq_sum += tl.sum(values * values)
    rsqrt = 1.0 / tl.sqrt(sq_sum / total + norm_eps)

    hc_offsets = tl.arange(0, BLOCK_HC)
    hc_mask = hc_offsets < hc_mult
    mix_base = token * mix_hc
    scale0 = tl.load(scale_ptr + 0).to(tl.float32)
    scale1 = tl.load(scale_ptr + 1).to(tl.float32)
    scale2 = tl.load(scale_ptr + 2).to(tl.float32)

    pre_logits = tl.load(mixes_ptr + mix_base + hc_offsets, mask=hc_mask, other=0.0).to(
        tl.float32
    ) * rsqrt * scale0 + tl.load(base_ptr + hc_offsets, mask=hc_mask, other=0.0).to(tl.float32)
    pre = tl.sigmoid(pre_logits) + eps
    pre = tl.where(hc_mask, pre, 0.0)
    tl.store(pre_ptr + token * hc_mult + hc_offsets, pre, mask=hc_mask)

    post_start = hc_mult
    post_logits = tl.load(
        mixes_ptr + mix_base + post_start + hc_offsets, mask=hc_mask, other=0.0
    ).to(tl.float32) * rsqrt * scale1 + tl.load(
        base_ptr + post_start + hc_offsets, mask=hc_mask, other=0.0
    ).to(
        tl.float32
    )
    post = 2.0 * tl.sigmoid(post_logits)
    post = tl.where(hc_mask, post, 0.0)
    tl.store(post_ptr + token * hc_mult + hc_offsets, post, mask=hc_mask)

    rows = tl.arange(0, BLOCK_HC)[:, None]
    cols = tl.arange(0, BLOCK_HC)[None, :]
    matrix_mask = (rows < hc_mult) & (cols < hc_mult)
    comb_start = 2 * hc_mult
    comb_offsets = comb_start + rows * hc_mult + cols
    comb_logits = tl.load(mixes_ptr + mix_base + comb_offsets, mask=matrix_mask, other=0.0).to(
        tl.float32
    ) * rsqrt * scale2 + tl.load(base_ptr + comb_offsets, mask=matrix_mask, other=0.0).to(
        tl.float32
    )
    comb_logits = tl.where(matrix_mask, comb_logits, -3.4028234663852886e38)
    row_max = tl.max(comb_logits, axis=1)
    exp_logits = tl.where(matrix_mask, tl.exp(comb_logits - row_max[:, None]), 0.0)
    row_sum = tl.sum(exp_logits, axis=1)
    comb = exp_logits / tl.maximum(row_sum[:, None], eps)
    comb = tl.where(matrix_mask, comb + eps, 0.0)

    col_sum = tl.sum(comb, axis=0)
    comb = tl.where(matrix_mask, comb / (col_sum[None, :] + eps), 0.0)
    for _ in range(0, sinkhorn_steps):
        row_sum_iter = tl.sum(comb, axis=1)
        comb = tl.where(matrix_mask, comb / (row_sum_iter[:, None] + eps), 0.0)
        col_sum_iter = tl.sum(comb, axis=0)
        comb = tl.where(matrix_mask, comb / (col_sum_iter[None, :] + eps), 0.0)

    tl.store(
        comb_ptr + token * hc_mult * hc_mult + rows * hc_mult + cols,
        comb,
        mask=matrix_mask,
    )


@triton.jit
def _hc_prenorm_head_pre_kernel(
    mixes_ptr,
    x_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    hc_mult: tl.constexpr,
    eps: tl.constexpr,
    norm_eps: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    total = hidden * hc_mult
    token_base = token * total

    sq_sum = tl.zeros((), dtype=tl.float32)
    for start in range(0, total, BLOCK_N):
        offsets = start + tl.arange(0, BLOCK_N)
        mask = offsets < total
        values = tl.load(x_ptr + token_base + offsets, mask=mask, other=0.0).to(tl.float32)
        sq_sum += tl.sum(values * values)
    rsqrt = 1.0 / tl.sqrt(sq_sum / total + norm_eps)

    hc_offsets = tl.arange(0, BLOCK_HC)
    hc_mask = hc_offsets < hc_mult
    scale = tl.load(scale_ptr + 0).to(tl.float32)
    logits = tl.load(mixes_ptr + token * hc_mult + hc_offsets, mask=hc_mask, other=0.0).to(
        tl.float32
    ) * rsqrt * scale + tl.load(base_ptr + hc_offsets, mask=hc_mask, other=0.0).to(tl.float32)
    pre = tl.sigmoid(logits) + eps
    pre = tl.where(hc_mask, pre, 0.0)
    tl.store(pre_ptr + token * hc_mult + hc_offsets, pre, mask=hc_mask)


@triton.jit
def _hc_layer_input_kernel(
    x_ptr,
    pre_ptr,
    y_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK_HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    d_block = tl.program_id(1)
    hc_offsets = tl.arange(0, BLOCK_HC)
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hc_mask = hc_offsets < hc_mult
    d_mask = d_offsets < hidden

    pre = tl.load(
        pre_ptr + token * hc_mult + hc_offsets,
        mask=hc_mask,
        other=0.0,
    ).to(tl.float32)
    x_offsets = token * hc_mult * hidden + hc_offsets[:, None] * hidden + d_offsets[None, :]
    x_values = tl.load(x_ptr + x_offsets, mask=hc_mask[:, None] & d_mask[None, :], other=0.0)
    y = tl.sum(pre[:, None] * x_values.to(tl.float32), axis=0)
    tl.store(y_ptr + token * hidden + d_offsets, y, mask=d_mask)


@triton.jit
def _paged_mqa_attention_bf16_kernel(
    q_ptr,
    cache_ptr,
    indptr_ptr,
    indices_ptr,
    lengths_ptr,
    sink_ptr,
    out_ptr,
    num_heads: tl.constexpr,
    dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    max_length: tl.constexpr,
    has_sink: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < dim

    row_start = tl.load(indptr_ptr + row).to(tl.int64)
    row_len = tl.load(lengths_ptr + row).to(tl.int32)
    q_base = (row * num_heads + head) * dim
    q_vec = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m_i = tl.full((), -3.4028234663852886e38, dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    if has_sink:
        m_i = tl.load(sink_ptr + head).to(tl.float32)
        l_i = 1.0

    n_offsets = tl.arange(0, BLOCK_N)
    for n_start in range(0, max_length, BLOCK_N):
        local_offsets = n_start + n_offsets
        n_mask = local_offsets < row_len
        cache_rows = tl.load(
            indices_ptr + row_start + local_offsets,
            mask=n_mask,
            other=0,
        ).to(tl.int64)
        kv = tl.load(
            cache_ptr + cache_rows[:, None] * dim + d_offsets[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(kv * q_vec[None, :], axis=1) * softmax_scale
        scores = tl.where(n_mask, scores, -3.4028234663852886e38)

        block_m = tl.max(scores, axis=0)
        block_has_values = row_len > n_start
        new_m = tl.where(block_has_values, tl.maximum(m_i, block_m), m_i)
        alpha = tl.where(l_i == 0.0, 0.0, tl.exp(m_i - new_m))
        probs = tl.where(n_mask, tl.exp(scores - new_m), 0.0)
        p_sum = tl.sum(probs, axis=0)

        acc = acc * alpha + tl.sum(probs[:, None] * kv, axis=0)
        l_i = l_i * alpha + p_sum
        m_i = new_m

    out = acc / l_i
    out = tl.where((row_len > 0) & d_mask, out, 0.0)
    tl.store(out_ptr + q_base + d_offsets, out, mask=d_mask)


@triton.jit
def _indexer_bf16_logits_kernel(
    q_ptr,
    cache_ptr,
    weights_ptr,
    seq_lens_ptr,
    page_table_ptr,
    logits_ptr,
    num_pages: tl.constexpr,
    num_heads: tl.constexpr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    page_bits: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    n_offsets = block * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < dim

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    valid_n = n_offsets < seq_len
    page_idx = n_offsets >> page_bits
    offset = n_offsets & (page_size - 1)
    physical_page = tl.load(
        page_table_ptr + row * num_pages + page_idx,
        mask=valid_n & (page_idx < num_pages),
        other=-1,
    ).to(tl.int64)
    valid = valid_n & (physical_page >= 0)
    cache_rows = physical_page * page_size + offset
    kv = tl.load(
        cache_ptr + cache_rows[:, None] * dim + d_offsets[None, :],
        mask=valid[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in range(0, num_heads):
        q = tl.load(
            q_ptr + (row * num_heads + head) * dim + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(kv * q[None, :], axis=1)
        score = tl.maximum(score, 0.0)
        weight = tl.load(weights_ptr + row * num_heads + head).to(tl.float32)
        acc += score * weight

    out = tl.where(valid, acc, -float("inf"))
    tl.store(logits_ptr + row * max_seq_len + n_offsets, out, mask=n_offsets < max_seq_len)


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
def _decode_e4m3fn_to_bf16_lut(u, lut_ptr):
    return tl.load(lut_ptr + u.to(tl.uint32))


@triton.jit
def _encode_e4m3fn_sw(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = (bits >> 31) & 1
    abs_bits = bits & 0x7FFFFFFF
    exp_fp32 = (abs_bits >> 23).to(tl.int32)
    mant_fp32 = abs_bits & 0x7FFFFF

    is_zero = abs_bits == 0
    is_inf_or_nan = exp_fp32 == 0xFF
    is_nan = is_inf_or_nan & (mant_fp32 != 0)

    exp_fp8 = exp_fp32 - 120
    mant_extracted = mant_fp32 >> 20
    round_bit = (mant_fp32 >> 19) & 1
    sticky = (mant_fp32 & 0x7FFFF) != 0
    odd = (mant_extracted & 1) == 1
    round_up = round_bit & (sticky.to(tl.uint32) | odd.to(tl.uint32))
    mant_rounded = mant_extracted + round_up
    carry = mant_rounded == 8
    exp_after = exp_fp8 + carry.to(tl.int32)
    mant_after = tl.where(carry, 0, mant_rounded)
    packed_normal = ((exp_after.to(tl.uint32) & 0xF) << 3) | (mant_after & 0x7)

    impl_mant = (tl.full((), 1, tl.uint32) << 23) | mant_fp32
    sub_shift = (141 - exp_fp32).to(tl.uint32)
    safe_shift = tl.minimum(sub_shift, 31)
    sub_m_int = impl_mant >> safe_shift
    sub_round_bit = tl.where(
        safe_shift >= 1,
        (impl_mant >> (safe_shift - 1)) & 1,
        tl.zeros_like(impl_mant),
    )
    sticky_mask = tl.where(
        safe_shift >= 2,
        (tl.full((), 1, tl.uint32) << (safe_shift - 1)) - 1,
        tl.zeros_like(impl_mant),
    )
    sub_sticky = (impl_mant & sticky_mask) != 0
    sub_odd = (sub_m_int & 1) == 1
    sub_round_up = sub_round_bit & (sub_sticky.to(tl.uint32) | sub_odd.to(tl.uint32))
    sub_m_rounded = sub_m_int + sub_round_up
    sub_promotes = sub_m_rounded == 8
    sub_packed = tl.where(
        sub_promotes,
        tl.full((), 0x08, tl.uint32),
        sub_m_rounded & 0x7,
    )

    over_max_finite = (exp_after >= 16) | ((exp_after == 15) & (mant_after == 7))
    packed_normal = tl.where(over_max_finite, 0x7E, packed_normal)

    is_subnormal = exp_fp8 <= 0
    encoded = tl.where(is_subnormal, sub_packed, packed_normal)
    encoded = tl.where(is_zero, tl.zeros_like(encoded), encoded)
    encoded = tl.where(is_nan, tl.full((), 0x7F, tl.uint32), encoded)
    encoded = encoded | (sign << 7)
    return encoded.to(tl.uint8)


@triton.jit
def _indexer_fp8_quant_store_kernel(
    kv_ptr,
    loc_ptr,
    values_ptr,
    scales_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim
    loc = tl.load(loc_ptr + row).to(tl.int64)
    valid_loc = loc >= 0

    values = tl.load(kv_ptr + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
    values = values.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    inv_scale = tl.exp2(-exponent)
    scaled = tl.clamp(values * inv_scale, -448.0, 448.0)
    encoded = _encode_e4m3fn_sw(scaled)

    tl.store(values_ptr + loc * dim + offsets, encoded, mask=mask & valid_loc)
    scale = tl.exp2(exponent)
    scale_ptr = (scales_ptr + loc * 4).to(tl.pointer_type(tl.float32))
    tl.store(scale_ptr, scale, mask=valid_loc)


@triton.jit
def _indexer_fp8_quantize_kernel(
    kv_ptr,
    values_ptr,
    scales_ptr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim

    values = tl.load(kv_ptr + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
    values = values.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    inv_scale = tl.exp2(-exponent)
    scaled = tl.clamp(values * inv_scale, -448.0, 448.0)
    encoded = _encode_e4m3fn_sw(scaled)

    tl.store(values_ptr + row * dim + offsets, encoded, mask=mask)
    scale = tl.exp2(exponent)
    scale_ptr = (scales_ptr + row * 4).to(tl.pointer_type(tl.float32))
    tl.store(scale_ptr, scale)


@triton.jit
def _indexer_fp8_quantize_fold_kernel(
    q_ptr,
    weights_ptr,
    values_ptr,
    weights_out_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_w_t,
    stride_w_h,
    stride_wo_t,
    stride_wo_h,
    num_heads: tl.constexpr,
    dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    head_scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    token = row // num_heads
    head = row - token * num_heads
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim

    q = tl.load(
        q_ptr + token * stride_q_t + head * stride_q_h + offsets * stride_q_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q = q.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(q), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    inv_scale = tl.exp2(-exponent)
    scaled = tl.clamp(q * inv_scale, -448.0, 448.0)
    encoded = _encode_e4m3fn_sw(scaled)

    tl.store(values_ptr + row * dim + offsets, encoded, mask=mask)
    weight = tl.load(weights_ptr + token * stride_w_t + head * stride_w_h).to(tl.float32)
    folded = weight * tl.exp2(exponent) * softmax_scale * head_scale
    tl.store(weights_out_ptr + token * stride_wo_t + head * stride_wo_h, folded)


@triton.jit
def _indexer_fp8_paged_quant_store_kernel(
    kv_ptr,
    loc_ptr,
    cache_ptr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    page_bytes: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim
    loc = tl.load(loc_ptr + row).to(tl.int64)
    valid_loc = loc >= 0

    values = tl.load(kv_ptr + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
    values = values.to(tl.bfloat16).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    inv_scale = tl.exp2(-exponent)
    scaled = tl.clamp(values * inv_scale, -448.0, 448.0)
    encoded = _encode_e4m3fn_sw(scaled)

    page = loc // page_size
    page_offset = loc - page * page_size
    page_base = cache_ptr + page * page_bytes
    value_ptr = page_base + page_offset * dim
    scale_ptr = (page_base + page_size * dim + page_offset * 4).to(tl.pointer_type(tl.float32))

    tl.store(value_ptr + offsets, encoded, mask=mask & valid_loc)
    tl.store(scale_ptr, tl.exp2(exponent), mask=valid_loc)


@triton.jit
def _fp8_activation_quantize_kernel(
    x_ptr,
    out_ptr,
    cols: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < block_size
    base = row * cols + group * block_size

    values = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    exponent = tl.ceil(tl.log2(absmax / 448.0))
    scale = tl.exp2(exponent)
    scaled = tl.clamp(values / scale, -448.0, 448.0)
    encoded = _encode_e4m3fn_sw(scaled)
    dequant = _fp8_e4m3fn_value(encoded.to(tl.uint32)) * scale
    tl.store(out_ptr + base + offsets, dequant, mask=mask)


@triton.jit
def _indexer_fp8_logits_kernel(
    q_ptr,
    cache_values_ptr,
    cache_scales_ptr,
    weights_ptr,
    seq_lens_ptr,
    page_table_ptr,
    logits_ptr,
    num_pages: tl.constexpr,
    num_heads: tl.constexpr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    page_bits: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    n_offsets = block * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < dim

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    valid_n = n_offsets < seq_len
    page_idx = n_offsets >> page_bits
    offset = n_offsets & (page_size - 1)
    physical_page = tl.load(
        page_table_ptr + row * num_pages + page_idx,
        mask=valid_n & (page_idx < num_pages),
        other=-1,
    ).to(tl.int64)
    valid = valid_n & (physical_page >= 0)
    cache_rows = physical_page * page_size + offset

    cache_bits = tl.load(
        cache_values_ptr + cache_rows[:, None] * dim + d_offsets[None, :],
        mask=valid[:, None] & d_mask[None, :],
        other=0,
    ).to(tl.int32)
    cache_scale = tl.load(
        cache_scales_ptr.to(tl.pointer_type(tl.float32)) + cache_rows,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    kv = _fp8_e4m3fn_value(cache_bits).to(tl.float32) * cache_scale[:, None]

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in range(0, num_heads):
        q_bits = tl.load(
            q_ptr + (row * num_heads + head) * dim + d_offsets,
            mask=d_mask,
            other=0,
        ).to(tl.int32)
        q = _fp8_e4m3fn_value(q_bits).to(tl.float32)
        score = tl.sum(kv * q[None, :], axis=1)
        score = tl.maximum(score, 0.0)
        weight = tl.load(weights_ptr + row * num_heads + head).to(tl.float32)
        acc += score * weight

    out = tl.where(valid, acc, -float("inf"))
    tl.store(logits_ptr + row * max_seq_len + n_offsets, out, mask=n_offsets < max_seq_len)


@triton.jit
def _indexer_fp8_paged_logits_kernel(
    q_ptr,
    cache_ptr,
    e4m3fn_to_bf16_ptr,
    weights_ptr,
    seq_lens_ptr,
    page_table_ptr,
    logits_ptr,
    stride_q_r,
    stride_q_h,
    stride_q_d,
    stride_w_r,
    stride_w_h,
    stride_pt_r,
    stride_pt_p,
    stride_l_r,
    stride_l_n,
    num_pages: tl.constexpr,
    num_heads: tl.constexpr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    page_bytes: tl.constexpr,
    max_seq_len: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    page_rank = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    if page_rank * page_size >= seq_len:
        return

    physical_page = tl.load(
        page_table_ptr + row * stride_pt_r + page_rank * stride_pt_p,
        mask=page_rank < num_pages,
        other=-1,
    ).to(tl.int64)
    if physical_page < 0:
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_h = offs_h < num_heads
    mask_d = offs_d < dim
    mask_n = offs_n < page_size

    q_byte = tl.load(
        q_ptr + row * stride_q_r + offs_h[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0,
    )
    q = _decode_e4m3fn_to_bf16_lut(q_byte, e4m3fn_to_bf16_ptr)

    page_base = cache_ptr + physical_page * page_bytes
    k_byte = tl.load(
        page_base + offs_n[:, None] * dim + offs_d[None, :],
        mask=mask_n[:, None] & mask_d[None, :],
        other=0,
    )
    k_scale = tl.load(
        (page_base + page_size * dim + offs_n * 4).to(tl.pointer_type(tl.float32)),
        mask=mask_n,
        other=0.0,
    )
    k = (
        _decode_e4m3fn_to_bf16_lut(k_byte, e4m3fn_to_bf16_ptr).to(tl.float32) * k_scale[:, None]
    ).to(tl.bfloat16)

    scores = tl.dot(q, tl.trans(k))
    weights = tl.load(
        weights_ptr + row * stride_w_r + offs_h * stride_w_h,
        mask=mask_h,
        other=0.0,
    )
    scores = tl.where(scores > 0, scores, 0.0) * weights[:, None]
    out = tl.sum(scores, axis=0)

    k_offsets = page_rank * page_size + offs_n
    valid = mask_n & (k_offsets < seq_len)
    out = tl.where(valid, out, -float("inf"))
    tl.store(
        logits_ptr + row * stride_l_r + k_offsets * stride_l_n,
        out,
        mask=mask_n & (k_offsets < max_seq_len),
    )


@triton.jit
def _remap_indexer_topk_locs_kernel(
    raw_indices_ptr,
    component_page_table_ptr,
    full_page_table_ptr,
    component_locs_ptr,
    full_locs_ptr,
    numel,
    width: tl.constexpr,
    component_table_width: tl.constexpr,
    full_table_width: tl.constexpr,
    component_page_size: tl.constexpr,
    full_page_size: tl.constexpr,
    ratio: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < numel
    raw = tl.load(raw_indices_ptr + offsets, mask=active, other=-1).to(tl.int64)
    rows = offsets // width

    component_logical_page = raw // component_page_size
    component_offset = raw - component_logical_page * component_page_size
    component_valid = (
        active
        & (raw >= 0)
        & (component_logical_page >= 0)
        & (component_logical_page < component_table_width)
    )
    component_page = tl.load(
        component_page_table_ptr
        + rows * component_table_width
        + component_logical_page,
        mask=component_valid,
        other=-1,
    ).to(tl.int64)
    component_loc = component_page * component_page_size + component_offset
    tl.store(
        component_locs_ptr + offsets,
        tl.where(component_valid & (component_page >= 0), component_loc, -1),
        mask=active,
    )

    full_position = raw * ratio + (ratio - 1)
    full_logical_page = full_position // full_page_size
    full_offset = full_position - full_logical_page * full_page_size
    full_valid = (
        active
        & (raw >= 0)
        & (full_logical_page >= 0)
        & (full_logical_page < full_table_width)
    )
    full_page = tl.load(
        full_page_table_ptr + rows * full_table_width + full_logical_page,
        mask=full_valid,
        other=-1,
    ).to(tl.int64)
    full_loc = full_page * full_page_size + full_offset
    tl.store(
        full_locs_ptr + offsets,
        tl.where(full_valid & (full_page >= 0), full_loc, -1),
        mask=active,
    )


@triton.jit
def _c128_prefill_page_indices_kernel(
    component_page_table_ptr,
    c128_lengths_ptr,
    output_ptr,
    width: tl.constexpr,
    component_table_width: tl.constexpr,
    component_page_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Write the final C128 component-location surface directly.

    The eager-prefill contract intentionally has no raw/full matrix input or
    output. All index arithmetic stays in int32 at the tensor boundary.
    """
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_width = cols < width
    length = tl.load(c128_lengths_ptr + row).to(tl.int32)
    logical_page = cols // component_page_size
    page_in_table = logical_page < component_table_width
    page = tl.load(
        component_page_table_ptr + row * component_table_width + logical_page,
        mask=in_width & page_in_table,
        other=-1,
    ).to(tl.int32)
    offset = cols - logical_page * component_page_size
    valid = in_width & (cols < length) & page_in_table & (page >= 0)
    loc = page * component_page_size + offset
    tl.store(output_ptr + row * width + cols, tl.where(valid, loc, -1), mask=in_width)


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
def _wo_a_grouped_projection_fp8_kernel(
    o_ptr,
    weight_ptr,
    scale_ptr,
    out_ptr,
    T: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    SCALE_D: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    pid_t = tl.program_id(0)
    group = tl.program_id(1)
    pid_r = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d_base = tl.arange(0, BLOCK_D)
    global_r = group * R + offs_r

    acc = tl.zeros((BLOCK_T, BLOCK_R), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + offs_d_base
        a = tl.load(
            o_ptr + (offs_t[:, None] * G + group) * D + offs_d[None, :],
            mask=(offs_t[:, None] < T) & (offs_d[None, :] < D),
            other=0.0,
        )
        b = tl.load(
            weight_ptr + global_r[None, :] * D + offs_d[:, None],
            mask=(offs_r[None, :] < R) & (offs_d[:, None] < D),
            other=0,
        ).to(tl.int32)
        b = _fp8_e4m3fn_value(b).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(
                scale_ptr + (global_r[None, :] // 128) * SCALE_D + (offs_d[:, None] // 128),
                mask=(offs_r[None, :] < R) & (offs_d[:, None] < D),
                other=0.0,
            ).to(tl.float32)
            b *= scale
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_t[:, None] * (G * R) + global_r[None, :],
        acc,
        mask=(offs_t[:, None] < T) & (offs_r[None, :] < R),
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


@triton.jit
def _grouped_fp4_linear_kernel(
    a_ptr,
    weight_ptr,
    scale_ptr,
    sorted_route_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    out_ptr,
    route_count: tl.constexpr,
    topk: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WEIGHT_K_BYTES: tl.constexpr,
    SCALE_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_SLOT: tl.constexpr,
    SLOT: tl.constexpr,
    A_ROWS_ARE_ROUTES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    padded_tokens = tl.load(num_tokens_post_padded_ptr)
    route_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    route_ids = tl.load(
        sorted_route_ids_ptr + route_offsets,
        mask=route_offsets < padded_tokens,
        other=route_count,
    ).to(tl.int64)
    valid_routes = route_ids < route_count
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_base = tl.arange(0, BLOCK_SIZE_K)
    a_rows = route_ids
    if not A_ROWS_ARE_ROUTES:
        a_rows = route_ids // topk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            a_ptr + a_rows[:, None] * K + offs_k[None, :],
            mask=valid_routes[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        if HAS_SLOT:
            weight_offsets = ((expert * 2 + SLOT) * N + offs_n[None, :]) * WEIGHT_K_BYTES + (
                offs_k[:, None] // 2
            )
            scale_offsets = ((expert * 2 + SLOT) * N + offs_n[None, :]) * SCALE_K + (
                offs_k[:, None] // 32
            )
        else:
            weight_offsets = (expert * N + offs_n[None, :]) * WEIGHT_K_BYTES + (
                offs_k[:, None] // 2
            )
            scale_offsets = (expert * N + offs_n[None, :]) * SCALE_K + (offs_k[:, None] // 32)
        weight_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        packed = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0).to(tl.int32)
        nibble = tl.where((offs_k[:, None] & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
        b = _fp4_e2m1_value(nibble).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(scale_ptr + scale_offsets, mask=weight_mask, other=0.0).to(tl.float32)
            b *= scale
        acc += tl.dot(a, b.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        out_ptr + route_ids[:, None] * N + offs_n[None, :],
        acc,
        mask=valid_routes[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _moe_route_count_kernel(
    indices_ptr,
    counts_ptr,
    route_count,
    num_experts: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
    mask = offsets < route_count
    expert = tl.load(indices_ptr + offsets, mask=mask, other=-1).to(tl.int64)
    valid = mask & (expert >= 0) & (expert < num_experts)
    tl.atomic_add(counts_ptr + expert, 1, sem="relaxed", mask=valid)


@triton.jit
def _moe_route_offsets_kernel(
    counts_ptr,
    padded_offsets_ptr,
    blocks_per_expert_ptr,
    num_tokens_post_padded_ptr,
    block_size_m: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
) -> None:
    offsets = tl.arange(0, BLOCK_EXPERTS)
    mask = offsets < num_experts
    counts = tl.load(counts_ptr + offsets, mask=mask, other=0)
    padded = ((counts + block_size_m - 1) // block_size_m) * block_size_m
    prefix = tl.cumsum(padded, 0)
    padded_offsets = prefix - padded
    tl.store(padded_offsets_ptr + offsets, padded_offsets, mask=mask)
    tl.store(blocks_per_expert_ptr + offsets, padded // block_size_m, mask=mask)
    tl.store(num_tokens_post_padded_ptr, tl.sum(padded, axis=0))


@triton.jit
def _moe_route_fill_kernel(
    indices_ptr,
    counts_ptr,
    padded_offsets_ptr,
    blocks_per_expert_ptr,
    sorted_route_ids_ptr,
    expert_ids_ptr,
    route_count,
    block_size_m: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
) -> None:
    expert = tl.program_id(0)
    count = tl.load(counts_ptr + expert)
    padded_offset = tl.load(padded_offsets_ptr + expert)
    padded_count = ((count + block_size_m - 1) // block_size_m) * block_size_m
    block_offset = padded_offset // block_size_m
    route_offsets_base = tl.arange(0, BLOCK_ROUTES)

    running_count = tl.full((), 0, dtype=tl.int32)
    start = 0
    while start < route_count:
        route_offsets = start + route_offsets_base
        mask = route_offsets < route_count
        route_expert = tl.load(indices_ptr + route_offsets, mask=mask, other=-1).to(tl.int64)
        matches = (route_expert == expert) & mask
        match_i32 = matches.to(tl.int32)
        local_rank = tl.cumsum(match_i32, 0) - 1
        tl.store(
            sorted_route_ids_ptr + padded_offset + running_count + local_rank,
            route_offsets.to(tl.int32),
            mask=matches,
        )
        running_count += tl.sum(match_i32, axis=0)
        start += BLOCK_ROUTES

    pad_start = count
    while pad_start < padded_count:
        pad_offsets = pad_start + route_offsets_base
        pad_mask = pad_offsets < padded_count
        tl.store(
            sorted_route_ids_ptr + padded_offset + pad_offsets,
            route_count,
            mask=pad_mask,
        )
        pad_start += BLOCK_ROUTES

    blocks = tl.load(blocks_per_expert_ptr + expert)
    block_start = 0
    while block_start < blocks:
        block_offsets = block_start + route_offsets_base
        block_mask = block_offsets < blocks
        tl.store(expert_ids_ptr + block_offset + block_offsets, expert, mask=block_mask)
        block_start += BLOCK_ROUTES


@triton.jit
def _grouped_fp4_w13_kernel(
    a_ptr,
    weight_ptr,
    scale_ptr,
    sorted_route_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    gate_ptr,
    up_ptr,
    route_count: tl.constexpr,
    topk: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WEIGHT_K_BYTES: tl.constexpr,
    SCALE_K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    padded_tokens = tl.load(num_tokens_post_padded_ptr)
    route_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    route_ids = tl.load(
        sorted_route_ids_ptr + route_offsets,
        mask=route_offsets < padded_tokens,
        other=route_count,
    ).to(tl.int64)
    valid_routes = route_ids < route_count
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_base = tl.arange(0, BLOCK_SIZE_K)
    a_rows = route_ids // topk

    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + offs_k_base
        a = tl.load(
            a_ptr + a_rows[:, None] * K + offs_k[None, :],
            mask=valid_routes[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        weight_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        gate_weight_offsets = ((expert * 2) * N + offs_n[None, :]) * WEIGHT_K_BYTES + (
            offs_k[:, None] // 2
        )
        up_weight_offsets = ((expert * 2 + 1) * N + offs_n[None, :]) * WEIGHT_K_BYTES + (
            offs_k[:, None] // 2
        )
        gate_packed = tl.load(weight_ptr + gate_weight_offsets, mask=weight_mask, other=0).to(
            tl.int32
        )
        up_packed = tl.load(weight_ptr + up_weight_offsets, mask=weight_mask, other=0).to(tl.int32)
        gate_nibble = tl.where(
            (offs_k[:, None] & 1) == 0,
            gate_packed & 0x0F,
            (gate_packed >> 4) & 0x0F,
        )
        up_nibble = tl.where(
            (offs_k[:, None] & 1) == 0,
            up_packed & 0x0F,
            (up_packed >> 4) & 0x0F,
        )
        gate_b = _fp4_e2m1_value(gate_nibble).to(tl.float32)
        up_b = _fp4_e2m1_value(up_nibble).to(tl.float32)
        if HAS_SCALE:
            gate_scale_offsets = ((expert * 2) * N + offs_n[None, :]) * SCALE_K + (
                offs_k[:, None] // 32
            )
            up_scale_offsets = ((expert * 2 + 1) * N + offs_n[None, :]) * SCALE_K + (
                offs_k[:, None] // 32
            )
            gate_scale = tl.load(
                scale_ptr + gate_scale_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                scale_ptr + up_scale_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)
            gate_b *= gate_scale
            up_b *= up_scale
        gate_acc += tl.dot(a, gate_b.to(tl.bfloat16), out_dtype=tl.float32)
        up_acc += tl.dot(a, up_b.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        gate_ptr + route_ids[:, None] * N + offs_n[None, :],
        gate_acc,
        mask=valid_routes[:, None] & (offs_n[None, :] < N),
    )
    tl.store(
        up_ptr + route_ids[:, None] * N + offs_n[None, :],
        up_acc,
        mask=valid_routes[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _moe_route_sum_kernel(
    routed_ptr,
    out_ptr,
    tokens: tl.constexpr,
    hidden: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    block_n = tl.program_id(1)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < hidden
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for top_idx in range(0, topk):
        route = token * topk + top_idx
        values = tl.load(routed_ptr + route * hidden + offs_n, mask=mask, other=0.0).to(tl.float32)
        acc += values
    tl.store(out_ptr + token * hidden + offs_n, acc, mask=mask)


@triton.jit
def _grouped_fp4_moe_fused_compute_kernel(
    a_ptr,
    weights_ptr,
    w13_weight_ptr,
    w13_scale_ptr,
    w2_weight_ptr,
    w2_scale_ptr,
    sorted_route_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    out_ptr,
    route_count: tl.constexpr,
    topk: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    W13_K_BYTES: tl.constexpr,
    W13_SCALE_K: tl.constexpr,
    W2_K_BYTES: tl.constexpr,
    W2_SCALE_K: tl.constexpr,
    HAS_W13_SCALE: tl.constexpr,
    HAS_W2_SCALE: tl.constexpr,
    swiglu_limit: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    padded_tokens = tl.load(num_tokens_post_padded_ptr)
    route_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    route_ids = tl.load(
        sorted_route_ids_ptr + route_offsets,
        mask=route_offsets < padded_tokens,
        other=route_count,
    ).to(tl.int64)
    valid_routes = route_ids < route_count
    token_rows = route_ids // topk
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    route_weights = tl.load(weights_ptr + route_ids, mask=valid_routes, other=0.0).to(tl.float32)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_i_base = tl.arange(0, BLOCK_SIZE_I)
    offs_k_base = tl.arange(0, BLOCK_SIZE_K)
    out_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i_start in range(0, I, BLOCK_SIZE_I):
        offs_i = i_start + offs_i_base
        gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_I), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_I), dtype=tl.float32)

        for k_start in range(0, H, BLOCK_SIZE_K):
            offs_k = k_start + offs_k_base
            a = tl.load(
                a_ptr + token_rows[:, None] * H + offs_k[None, :],
                mask=valid_routes[:, None] & (offs_k[None, :] < H),
                other=0.0,
            )
            weight_mask = (offs_i[None, :] < I) & (offs_k[:, None] < H)
            gate_weight_offsets = ((expert * 2) * I + offs_i[None, :]) * W13_K_BYTES + (
                offs_k[:, None] // 2
            )
            up_weight_offsets = ((expert * 2 + 1) * I + offs_i[None, :]) * W13_K_BYTES + (
                offs_k[:, None] // 2
            )
            gate_packed = tl.load(
                w13_weight_ptr + gate_weight_offsets,
                mask=weight_mask,
                other=0,
            ).to(tl.int32)
            up_packed = tl.load(
                w13_weight_ptr + up_weight_offsets,
                mask=weight_mask,
                other=0,
            ).to(tl.int32)
            gate_nibble = tl.where(
                (offs_k[:, None] & 1) == 0,
                gate_packed & 0x0F,
                (gate_packed >> 4) & 0x0F,
            )
            up_nibble = tl.where(
                (offs_k[:, None] & 1) == 0,
                up_packed & 0x0F,
                (up_packed >> 4) & 0x0F,
            )
            gate_b = _fp4_e2m1_value(gate_nibble).to(tl.float32)
            up_b = _fp4_e2m1_value(up_nibble).to(tl.float32)
            if HAS_W13_SCALE:
                gate_scale_offsets = ((expert * 2) * I + offs_i[None, :]) * W13_SCALE_K + (
                    offs_k[:, None] // 32
                )
                up_scale_offsets = ((expert * 2 + 1) * I + offs_i[None, :]) * W13_SCALE_K + (
                    offs_k[:, None] // 32
                )
                gate_scale = tl.load(
                    w13_scale_ptr + gate_scale_offsets,
                    mask=weight_mask,
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    w13_scale_ptr + up_scale_offsets,
                    mask=weight_mask,
                    other=0.0,
                ).to(tl.float32)
                gate_b *= gate_scale
                up_b *= up_scale
            gate_acc += tl.dot(a, gate_b.to(tl.bfloat16), out_dtype=tl.float32)
            up_acc += tl.dot(a, up_b.to(tl.bfloat16), out_dtype=tl.float32)

        gate_act = gate_acc.to(tl.bfloat16).to(tl.float32)
        up_act = up_acc.to(tl.bfloat16).to(tl.float32)
        if swiglu_limit > 0.0:
            up_act = tl.minimum(tl.maximum(up_act, -swiglu_limit), swiglu_limit)
            gate_act = tl.minimum(gate_act, swiglu_limit)
        activated = gate_act * tl.sigmoid(gate_act) * up_act
        activated *= route_weights[:, None]

        w2_mask = (offs_i[:, None] < I) & (offs_n[None, :] < H)
        w2_offsets = (expert * H + offs_n[None, :]) * W2_K_BYTES + (offs_i[:, None] // 2)
        w2_packed = tl.load(w2_weight_ptr + w2_offsets, mask=w2_mask, other=0).to(tl.int32)
        w2_nibble = tl.where(
            (offs_i[:, None] & 1) == 0,
            w2_packed & 0x0F,
            (w2_packed >> 4) & 0x0F,
        )
        w2_b = _fp4_e2m1_value(w2_nibble).to(tl.float32)
        if HAS_W2_SCALE:
            w2_scale_offsets = (expert * H + offs_n[None, :]) * W2_SCALE_K + (offs_i[:, None] // 32)
            w2_scale = tl.load(w2_scale_ptr + w2_scale_offsets, mask=w2_mask, other=0.0).to(
                tl.float32
            )
            w2_b *= w2_scale
        out_acc += tl.dot(activated.to(tl.bfloat16), w2_b.to(tl.bfloat16), out_dtype=tl.float32)

    tl.store(
        out_ptr + route_ids[:, None] * H + offs_n[None, :],
        out_acc,
        mask=valid_routes[:, None] & (offs_n[None, :] < H),
    )


@triton.jit
def _sparse_bf16_gather_with_mask_kernel(
    cache_ptr,
    indices_ptr,
    lengths_ptr,
    out_ptr,
    invalid_mask_ptr,
    cache_rows,
    topk_in: tl.constexpr,
    total_topk: tl.constexpr,
    slot_offset: tl.constexpr,
    head_dim: tl.constexpr,
    has_lengths: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    token = pid // topk_in
    slot = pid % topk_in
    dst_slot = slot_offset + slot
    dst_row = token * total_topk + dst_slot

    index = tl.load(indices_ptr + pid).to(tl.int64)
    in_range = (index >= 0) & (index < cache_rows)
    beyond_length = False
    if has_lengths:
        length = tl.load(lengths_ptr + token).to(tl.int64)
        beyond_length = slot.to(tl.int64) >= length
    invalid = (~in_range) | beyond_length
    tl.store(invalid_mask_ptr + dst_row, invalid)

    safe_index = tl.where(in_range, index, 0)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim
    values = tl.load(
        cache_ptr + safe_index * head_dim + offsets,
        mask=dim_mask & in_range,
        other=0.0,
    )
    tl.store(out_ptr + dst_row * head_dim + offsets, values, mask=dim_mask)


@triton.jit
def _sparse_splitk_bf16_split_kernel(
    q_ptr,
    kv_ptr,
    invalid_mask_ptr,
    acc_split_ptr,
    max_split_ptr,
    sum_split_ptr,
    sm_scale_log2,
    H: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    TOTAL_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    split = tl.program_id(1)
    head_block = tl.program_id(2)

    chunk_size: tl.constexpr = (TOTAL_TOPK + SPLIT_T - 1) // SPLIT_T
    n_start_chunk = split * chunk_size
    n_end_chunk = tl.minimum(n_start_chunk + chunk_size, TOTAL_TOPK)

    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < H
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < D

    q = tl.load(
        q_ptr + token * H * D + head_offsets[:, None] * D + dim_offsets[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    e_max = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)

    n_iter: tl.constexpr = (chunk_size + BLOCK_N - 1) // BLOCK_N
    for n_block in tl.static_range(0, n_iter):
        n_start = n_start_chunk + n_block * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < n_end_chunk

        invalid = tl.load(
            invalid_mask_ptr + token * TOTAL_TOPK + n_offsets,
            mask=n_mask,
            other=1,
        )
        valid = (invalid == 0) & n_mask

        kv = tl.load(
            kv_ptr + token * TOTAL_TOPK * D + n_offsets[:, None] * D + dim_offsets[None, :],
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(kv), out_dtype=tl.float32)
        qk *= sm_scale_log2
        qk = tl.where(head_mask[:, None] & valid[None, :], qk, -1.0e30)

        n_e_max = tl.maximum(tl.max(qk, axis=1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(tl.bfloat16), kv, out_dtype=tl.float32)
        e_sum = e_sum * re_scale + tl.sum(p, axis=1)
        e_max = n_e_max

    dv_offsets = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offsets < D_V
    acc_base = (
        token * SPLIT_T * H * D_V
        + split * H * D_V
        + head_offsets[:, None] * D_V
        + dv_offsets[None, :]
    )
    tl.store(acc_split_ptr + acc_base, acc, mask=head_mask[:, None] & dv_mask[None, :])
    stat_base = token * SPLIT_T * H + split * H + head_offsets
    tl.store(max_split_ptr + stat_base, e_max, mask=head_mask)
    tl.store(sum_split_ptr + stat_base, e_sum, mask=head_mask)


@triton.jit
def _sparse_splitk_bf16_combine_kernel(
    acc_split_ptr,
    max_split_ptr,
    sum_split_ptr,
    attn_sink_ptr,
    out_ptr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    SPLIT_T: tl.constexpr,
    HAS_SINK: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    head_block = tl.program_id(1)

    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < H

    e_max_global = tl.zeros((BLOCK_H,), dtype=tl.float32) - 1.0e30
    for split in tl.static_range(0, SPLIT_T):
        m = tl.load(
            max_split_ptr + token * SPLIT_T * H + split * H + head_offsets,
            mask=head_mask,
            other=-1.0e30,
        )
        e_max_global = tl.maximum(e_max_global, m)

    sink_log2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if HAS_SINK:
        sink = tl.load(attn_sink_ptr + head_offsets, mask=head_mask, other=0.0)
        sink_log2 = sink * 1.4426950408889634
        e_max_global = tl.maximum(e_max_global, sink_log2)

    dv_offsets = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offsets < D_V
    acc_global = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    sum_global = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for split in tl.static_range(0, SPLIT_T):
        m = tl.load(
            max_split_ptr + token * SPLIT_T * H + split * H + head_offsets,
            mask=head_mask,
            other=-1.0e30,
        )
        split_sum = tl.load(
            sum_split_ptr + token * SPLIT_T * H + split * H + head_offsets,
            mask=head_mask,
            other=0.0,
        )
        scale = tl.exp2(m - e_max_global)
        sum_global += scale * split_sum

        acc_base = (
            token * SPLIT_T * H * D_V
            + split * H * D_V
            + head_offsets[:, None] * D_V
            + dv_offsets[None, :]
        )
        acc = tl.load(
            acc_split_ptr + acc_base,
            mask=head_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        acc_global += scale[:, None] * acc

    if HAS_SINK:
        sum_global += tl.exp2(sink_log2 - e_max_global)
    sum_safe = tl.where(sum_global > 0.0, sum_global, 1.0)
    out = (acc_global / sum_safe[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + token * H * D_V + head_offsets[:, None] * D_V + dv_offsets[None, :],
        out,
        mask=head_mask[:, None] & dv_mask[None, :],
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
        return (
            rotary_dim
            * math.log(original_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
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


def _gather_scope_bf16(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    gathered: torch.Tensor,
    invalid_mask: torch.Tensor,
    *,
    slot_offset: int,
    total_topk: int,
    head_dim: int,
) -> None:
    n_tokens, topk_in = indices.shape
    if n_tokens == 0 or topk_in == 0:
        return
    if lengths is None:
        length_arg = torch.empty(0, dtype=torch.int32, device=indices.device)
        has_lengths = False
    else:
        length_arg = lengths.reshape(n_tokens)
        has_lengths = True
    _sparse_bf16_gather_with_mask_kernel[(n_tokens * topk_in,)](
        cache,
        indices.reshape(-1),
        length_arg,
        gathered,
        invalid_mask,
        cache.shape[0],
        topk_in=topk_in,
        total_topk=total_topk,
        slot_offset=slot_offset,
        head_dim=head_dim,
        has_lengths=has_lengths,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=8,
    )


def sparse_attention_splitk_bf16(
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
    if (
        q.ndim != 3
        or swa_cache.ndim != 2
        or swa_indices.ndim != 2
        or swa_lengths.ndim != 1
        or q.dtype is not torch.bfloat16
        or swa_cache.dtype is not torch.bfloat16
        or swa_indices.dtype is not torch.int32
        or swa_lengths.dtype is not torch.int32
        or not q.is_cuda
        or not swa_cache.is_cuda
        or not swa_indices.is_cuda
        or not swa_lengths.is_cuda
        or q.shape[-1] != 512
        or swa_cache.shape[-1] != q.shape[-1]
        or q.shape[0] != swa_indices.shape[0]
        or q.shape[0] != swa_lengths.numel()
        or q.stride(-1) != 1
        or swa_cache.stride(-1) != 1
        or swa_indices.stride(-1) != 1
    ):
        return None

    q = q.contiguous()
    swa_cache = swa_cache.contiguous()
    swa_indices = swa_indices.contiguous()
    swa_lengths = swa_lengths.contiguous()

    has_compressed = (
        compressed_cache is not None
        and compressed_indices is not None
        and compressed_lengths is not None
        and compressed_cache.numel() > 0
        and compressed_indices.shape[-1] > 0
    )
    if has_compressed:
        assert compressed_cache is not None
        assert compressed_indices is not None
        assert compressed_lengths is not None
        if (
            compressed_cache.ndim != 2
            or compressed_indices.ndim != 2
            or compressed_lengths.ndim != 1
            or compressed_cache.dtype is not torch.bfloat16
            or compressed_indices.dtype is not torch.int32
            or compressed_lengths.dtype is not torch.int32
            or not compressed_cache.is_cuda
            or not compressed_indices.is_cuda
            or not compressed_lengths.is_cuda
            or compressed_cache.shape[-1] != q.shape[-1]
            or compressed_indices.shape[0] != q.shape[0]
            or compressed_lengths.numel() != q.shape[0]
            or compressed_cache.stride(-1) != 1
            or compressed_indices.stride(-1) != 1
        ):
            return None
        compressed_cache = compressed_cache.contiguous()
        compressed_indices = compressed_indices.contiguous()
        compressed_lengths = compressed_lengths.contiguous()
        compressed_topk = compressed_indices.shape[1]
    else:
        compressed_topk = 0

    if attn_sink is not None and (
        not attn_sink.is_cuda
        or attn_sink.dtype is not torch.float32
        or attn_sink.numel() < q.shape[1]
    ):
        return None

    n_tokens, heads, head_dim = q.shape
    swa_topk = swa_indices.shape[1]
    total_topk = compressed_topk + swa_topk
    if n_tokens == 0:
        return torch.empty_like(q)
    if total_topk <= 0:
        return torch.empty_like(q)

    gathered = torch.empty((n_tokens, total_topk, head_dim), dtype=torch.bfloat16, device=q.device)
    invalid_mask = torch.empty((n_tokens, total_topk), dtype=torch.bool, device=q.device)

    slot_offset = 0
    if has_compressed:
        assert compressed_cache is not None
        assert compressed_indices is not None
        assert compressed_lengths is not None
        _gather_scope_bf16(
            compressed_cache,
            compressed_indices,
            compressed_lengths,
            gathered,
            invalid_mask,
            slot_offset=slot_offset,
            total_topk=total_topk,
            head_dim=head_dim,
        )
        slot_offset += compressed_topk

    _gather_scope_bf16(
        swa_cache,
        swa_indices,
        swa_lengths,
        gathered,
        invalid_mask,
        slot_offset=slot_offset,
        total_topk=total_topk,
        head_dim=head_dim,
    )

    block_h = 16
    block_n = 32
    block_d = triton.next_power_of_2(head_dim)
    block_dv = block_d
    split_t = max(1, min(16, triton.cdiv(total_topk, block_n)))

    out = torch.empty_like(q)
    acc_split = torch.empty(
        (n_tokens, split_t, heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    max_split = torch.empty((n_tokens, split_t, heads), dtype=torch.float32, device=q.device)
    sum_split = torch.empty_like(max_split)

    grid_split = (n_tokens, split_t, triton.cdiv(heads, block_h))
    _sparse_splitk_bf16_split_kernel[grid_split](
        q,
        gathered,
        invalid_mask,
        acc_split,
        max_split,
        sum_split,
        float(softmax_scale) * 1.4426950408889634,
        H=heads,
        D=head_dim,
        D_V=head_dim,
        TOTAL_TOPK=total_topk,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        num_warps=4,
    )

    sink = (
        attn_sink[:heads].contiguous()
        if attn_sink is not None
        else q.new_empty((1,), dtype=torch.float32)
    )
    grid_combine = (n_tokens, triton.cdiv(heads, block_h))
    _sparse_splitk_bf16_combine_kernel[grid_combine](
        acc_split,
        max_split,
        sum_split,
        sink,
        out,
        H=heads,
        D_V=head_dim,
        BLOCK_H=block_h,
        BLOCK_DV=block_dv,
        SPLIT_T=split_t,
        HAS_SINK=attn_sink is not None,
        num_warps=4,
    )
    return out


def rms_norm_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor | None:
    if (
        x.ndim == 0
        or x.numel() == 0
        or x.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
        or not x.is_cuda
        or not weight.is_cuda
        or x.stride(-1) != 1
        or weight.numel() != x.shape[-1]
    ):
        return None
    dim = x.shape[-1]
    if dim <= 0:
        return None
    block_d = triton.next_power_of_2(dim)
    if block_d > 8192:
        return None
    x_c = x.contiguous()
    rows = x_c.numel() // dim
    out = torch.empty_like(x_c)
    _rms_norm_bf16_kernel[(rows,)](
        x_c,
        weight.contiguous(),
        out,
        rows=rows,
        dim=dim,
        eps=float(eps),
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out.reshape_as(x)


def rms_norm_pair_bf16(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        q.ndim == 0
        or kv.ndim == 0
        or q.numel() == 0
        or kv.numel() == 0
        or q.dtype is not torch.bfloat16
        or kv.dtype is not torch.bfloat16
        or q_weight.dtype is not torch.bfloat16
        or kv_weight.dtype is not torch.bfloat16
        or not q.is_cuda
        or not kv.is_cuda
        or not q_weight.is_cuda
        or not kv_weight.is_cuda
        or q.stride(-1) != 1
        or kv.stride(-1) != 1
        or q_weight.numel() != q.shape[-1]
        or kv_weight.numel() != kv.shape[-1]
    ):
        return None
    q_dim = q.shape[-1]
    kv_dim = kv.shape[-1]
    if q_dim <= 0 or kv_dim <= 0:
        return None
    block_d = triton.next_power_of_2(max(q_dim, kv_dim))
    if block_d > 8192:
        return None
    q_c = q.contiguous()
    kv_c = kv.contiguous()
    q_rows = q_c.numel() // q_dim
    kv_rows = kv_c.numel() // kv_dim
    if q_rows != kv_rows:
        return None
    q_out = torch.empty_like(q_c)
    kv_out = torch.empty_like(kv_c)
    _rms_norm_pair_bf16_kernel[(q_rows, 2)](
        q_c,
        q_weight.contiguous(),
        q_out,
        kv_c,
        kv_weight.contiguous(),
        kv_out,
        q_dim=q_dim,
        kv_dim=kv_dim,
        eps=float(eps),
        BLOCK_D=block_d,
        num_warps=8,
    )
    return q_out.reshape_as(q), kv_out.reshape_as(kv)


def _workspace_tensor(
    workspace,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    zero: bool = False,
) -> torch.Tensor:
    if workspace is not None:
        tensor_fn = getattr(workspace, "tensor", None)
        if tensor_fn is not None:
            return tensor_fn(name, shape, dtype, device, zero=zero)
    if zero:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.empty(shape, dtype=dtype, device=device)


def _silu_and_mul_clamp_out(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float,
    weights: torch.Tensor | None = None,
    out_dtype: torch.dtype,
    workspace=None,
    workspace_name: str,
) -> torch.Tensor | None:
    if gate.shape != up.shape or gate.numel() == 0 or not gate.is_cuda or not up.is_cuda:
        return None
    if gate.stride(-1) != 1 or up.stride(-1) != 1:
        return None

    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = _workspace_tensor(
        workspace,
        workspace_name,
        tuple(gate_c.shape),
        out_dtype,
        gate_c.device,
    )
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


def silu_and_mul_clamp(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor | None:
    return _silu_and_mul_clamp_out(
        gate,
        up,
        swiglu_limit=swiglu_limit,
        weights=weights,
        out_dtype=torch.float32,
        workspace=None,
        workspace_name="swiglu_fp32",
    )


def silu_and_mul_clamp_bf16(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float,
    weights: torch.Tensor | None = None,
    workspace=None,
) -> torch.Tensor | None:
    return _silu_and_mul_clamp_out(
        gate,
        up,
        swiglu_limit=swiglu_limit,
        weights=weights,
        out_dtype=torch.bfloat16,
        workspace=workspace,
        workspace_name="swiglu_bf16",
    )


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


def k_norm_rope_cache_bf16(
    kv: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    cache: torch.Tensor,
    out_loc: torch.Tensor,
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
        kv.ndim != 2
        or kv.numel() == 0
        or rotary_dim <= 0
        or not kv.is_cuda
        or not positions.is_cuda
        or not norm_weight.is_cuda
        or not cache.is_cuda
        or not out_loc.is_cuda
        or not kv.is_contiguous()
        or not cache.is_contiguous()
    ):
        return False
    if positions.numel() != kv.shape[0] or out_loc.numel() != kv.shape[0]:
        return False
    dim = kv.shape[-1]
    if cache.shape[-1] != dim or norm_weight.numel() != dim or rotary_dim > dim:
        return False
    use_scaling, low, high = _rope_scaling(
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    block_d = triton.next_power_of_2(dim)
    block_half = triton.next_power_of_2(rotary_dim // 2)
    _k_norm_rope_cache_bf16_kernel[(kv.shape[0],)](
        kv,
        positions.contiguous(),
        norm_weight.contiguous(),
        cache,
        out_loc.contiguous(),
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


def q_kv_norm_rope_cache_bf16(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    cache: torch.Tensor,
    out_loc: torch.Tensor,
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
        or kv.ndim != 2
        or q.numel() == 0
        or kv.numel() == 0
        or rotary_dim <= 0
        or not q.is_cuda
        or not kv.is_cuda
        or not positions.is_cuda
        or not norm_weight.is_cuda
        or not cache.is_cuda
        or not out_loc.is_cuda
        or not q.is_contiguous()
        or kv.stride(-1) != 1
        or not cache.is_contiguous()
    ):
        return False
    heads = _heads_per_token(q, positions)
    if heads is None:
        return False
    if positions.numel() != kv.shape[0] or out_loc.numel() != kv.shape[0]:
        return False
    q_dim = q.shape[-1]
    kv_dim = kv.shape[-1]
    q_rows = q.numel() // q_dim
    kv_rows = kv.shape[0]
    if (
        q_dim <= 0
        or kv_dim <= 0
        or rotary_dim > q_dim
        or rotary_dim > kv_dim
        or cache.shape[-1] != kv_dim
        or norm_weight.numel() != kv_dim
    ):
        return False
    use_scaling, low, high = _rope_scaling(
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    block_d = triton.next_power_of_2(max(q_dim, kv_dim))
    block_half = triton.next_power_of_2(rotary_dim // 2)
    _q_kv_norm_rope_cache_bf16_kernel[(max(q_rows, kv_rows), 2)](
        q,
        kv,
        positions.contiguous(),
        norm_weight.contiguous(),
        cache,
        out_loc.contiguous(),
        q_rows=q_rows,
        kv_rows=kv_rows,
        heads_per_token=heads,
        q_dim=q_dim,
        kv_dim=kv_dim,
        kv_stride0=kv.stride(0),
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


def compress_norm_rope_store_bf16(
    kv: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor | None,
    cache: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    rms_norm_eps: float,
    rotary_dim: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> bool:
    has_norm = norm_weight is not None
    apply_rope = rotary_dim > 0
    if (
        kv.ndim != 2
        or kv.numel() == 0
        or not kv.is_cuda
        or not positions.is_cuda
        or not cache.is_cuda
        or not out_loc.is_cuda
        or not kv.is_contiguous()
        or not cache.is_contiguous()
        or (has_norm and not norm_weight.is_cuda)
    ):
        return False
    if positions.numel() != kv.shape[0] or out_loc.numel() != kv.shape[0]:
        return False
    dim = kv.shape[-1]
    if cache.shape[-1] != dim or rotary_dim > dim:
        return False
    if has_norm and norm_weight.numel() != dim:
        return False
    if not has_norm and not apply_rope:
        return False
    use_scaling, low, high = _rope_scaling(
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    block_d = triton.next_power_of_2(dim)
    block_half = triton.next_power_of_2(max(rotary_dim // 2, 1))
    norm_weight_c = norm_weight.contiguous() if norm_weight is not None else kv
    _compress_norm_rope_store_bf16_kernel[(kv.shape[0],)](
        kv,
        positions.contiguous(),
        norm_weight_c,
        cache,
        out_loc.contiguous(),
        dim=dim,
        rotary_dim=rotary_dim,
        eps=float(rms_norm_eps),
        log_base=math.log(base),
        use_scaling=use_scaling and apply_rope,
        factor=float(factor),
        low=low,
        high=high,
        scale_denom=max(high - low, 1),
        has_norm=has_norm,
        apply_rope=apply_rope,
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


def build_decode_metadata_indices(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    page_table: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    c4_topk_lengths_raw: torch.Tensor,
    c4_topk_lengths_clamp1: torch.Tensor,
    c4_sparse_topk_lengths: torch.Tensor,
    c4_sparse_raw_indices: torch.Tensor,
    c4_sparse_page_indices: torch.Tensor,
    c4_sparse_full_indices: torch.Tensor,
    c128_topk_lengths_clamp1: torch.Tensor,
    c128_raw_indices: torch.Tensor,
    c128_page_indices: torch.Tensor,
    c128_full_indices: torch.Tensor,
    *,
    page_size: int,
    max_seqlen_k: int,
    window_size: int,
    index_topk: int,
) -> bool:
    tensors = (
        ctx_page_table,
        table_indices,
        positions,
        page_table,
        swa_page_indices,
        swa_topk_lengths,
        c4_topk_lengths_raw,
        c4_topk_lengths_clamp1,
        c4_sparse_topk_lengths,
        c4_sparse_raw_indices,
        c4_sparse_page_indices,
        c4_sparse_full_indices,
        c128_topk_lengths_clamp1,
        c128_raw_indices,
        c128_page_indices,
        c128_full_indices,
    )
    rows = int(positions.numel())
    if (
        ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or positions.ndim != 1
        or page_table.ndim != 2
        or swa_page_indices.ndim != 2
        or c4_sparse_raw_indices.ndim != 2
        or c128_raw_indices.ndim != 2
        or table_indices.shape != positions.shape
        or page_table.shape[0] != rows
        or swa_page_indices.shape[0] != rows
        or c4_sparse_raw_indices.shape[0] != rows
        or c4_sparse_page_indices.shape != c4_sparse_raw_indices.shape
        or c4_sparse_full_indices.shape != c4_sparse_raw_indices.shape
        or c128_raw_indices.shape[0] != rows
        or c128_page_indices.shape != c128_raw_indices.shape
        or c128_full_indices.shape != c128_raw_indices.shape
        or page_size <= 0
        or page_size & (page_size - 1)
        or window_size <= 0
        or index_topk <= 0
    ):
        return False
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if rows <= 0:
        return True
    max_width = max(
        int(page_table.shape[1]),
        int(swa_page_indices.shape[1]),
        int(c4_sparse_raw_indices.shape[1]),
        int(c128_raw_indices.shape[1]),
    )
    block = triton.next_power_of_2(max(max_width, 1))
    _build_decode_metadata_indices_kernel[(rows,)](
        ctx_page_table,
        table_indices,
        positions,
        page_table,
        swa_page_indices,
        swa_topk_lengths,
        c4_topk_lengths_raw,
        c4_topk_lengths_clamp1,
        c4_sparse_topk_lengths,
        c4_sparse_raw_indices,
        c4_sparse_page_indices,
        c4_sparse_full_indices,
        c128_topk_lengths_clamp1,
        c128_raw_indices,
        c128_page_indices,
        c128_full_indices,
        ctx_page_table_stride0=ctx_page_table.stride(0),
        ctx_page_table_width=ctx_page_table.shape[1],
        page_table_width=page_table.shape[1],
        swa_width=swa_page_indices.shape[1],
        c4_width=c4_sparse_raw_indices.shape[1],
        c128_width=c128_raw_indices.shape[1],
        page_size=int(page_size),
        max_seqlen_k=int(max_seqlen_k),
        window_size=int(window_size),
        index_topk=int(index_topk),
        BLOCK=block,
    )
    return True


def build_decode_metadata_indices_component(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    c4_page_table: torch.Tensor,
    c128_page_table: torch.Tensor,
    page_table: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    c4_topk_lengths_raw: torch.Tensor,
    c4_topk_lengths_clamp1: torch.Tensor,
    c4_sparse_topk_lengths: torch.Tensor,
    c4_sparse_raw_indices: torch.Tensor,
    c4_sparse_page_indices: torch.Tensor,
    c4_sparse_full_indices: torch.Tensor,
    c128_topk_lengths_clamp1: torch.Tensor,
    c128_raw_indices: torch.Tensor,
    c128_page_indices: torch.Tensor,
    c128_full_indices: torch.Tensor,
    *,
    page_size: int,
    max_seqlen_k: int,
    window_size: int,
    index_topk: int,
) -> bool:
    tensors = (
        ctx_page_table,
        table_indices,
        positions,
        c4_page_table,
        c128_page_table,
        page_table,
        swa_page_indices,
        swa_topk_lengths,
        c4_topk_lengths_raw,
        c4_topk_lengths_clamp1,
        c4_sparse_topk_lengths,
        c4_sparse_raw_indices,
        c4_sparse_page_indices,
        c4_sparse_full_indices,
        c128_topk_lengths_clamp1,
        c128_raw_indices,
        c128_page_indices,
        c128_full_indices,
    )
    rows = int(positions.numel())
    if (
        ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or positions.ndim != 1
        or c4_page_table.ndim != 2
        or c128_page_table.ndim != 2
        or page_table.ndim != 2
        or swa_page_indices.ndim != 2
        or c4_sparse_raw_indices.ndim != 2
        or c128_raw_indices.ndim != 2
        or table_indices.shape != positions.shape
        or c4_page_table.shape[0] != rows
        or c128_page_table.shape[0] != rows
        or page_table.shape[0] != rows
        or swa_page_indices.shape[0] != rows
        or c4_sparse_raw_indices.shape[0] != rows
        or c4_sparse_page_indices.shape != c4_sparse_raw_indices.shape
        or c4_sparse_full_indices.shape != c4_sparse_raw_indices.shape
        or c128_raw_indices.shape[0] != rows
        or c128_page_indices.shape != c128_raw_indices.shape
        or c128_full_indices.shape != c128_raw_indices.shape
        or page_size <= 0
        or page_size & (page_size - 1)
        or window_size <= 0
        or index_topk <= 0
    ):
        return False
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if rows <= 0:
        return True
    max_width = max(
        int(page_table.shape[1]),
        int(swa_page_indices.shape[1]),
        int(c4_sparse_raw_indices.shape[1]),
        int(c128_raw_indices.shape[1]),
    )
    c4_component_page_size = max(int(page_size) // 4, 1)
    c128_component_page_size = max(int(page_size) // 128, 1)
    block = triton.next_power_of_2(max(max_width, 1))
    _build_decode_metadata_indices_component_kernel[(rows,)](
        ctx_page_table,
        table_indices,
        positions,
        c4_page_table,
        c128_page_table,
        page_table,
        swa_page_indices,
        swa_topk_lengths,
        c4_topk_lengths_raw,
        c4_topk_lengths_clamp1,
        c4_sparse_topk_lengths,
        c4_sparse_raw_indices,
        c4_sparse_page_indices,
        c4_sparse_full_indices,
        c128_topk_lengths_clamp1,
        c128_raw_indices,
        c128_page_indices,
        c128_full_indices,
        ctx_page_table_stride0=ctx_page_table.stride(0),
        ctx_page_table_width=ctx_page_table.shape[1],
        c4_page_table_width=c4_page_table.shape[1],
        c128_page_table_width=c128_page_table.shape[1],
        page_table_width=page_table.shape[1],
        swa_width=swa_page_indices.shape[1],
        c4_width=c4_sparse_raw_indices.shape[1],
        c128_width=c128_raw_indices.shape[1],
        page_size=int(page_size),
        max_seqlen_k=int(max_seqlen_k),
        window_size=int(window_size),
        index_topk=int(index_topk),
        c4_component_page_size=c4_component_page_size,
        c128_component_page_size=c128_component_page_size,
        BLOCK=block,
    )
    return True


def direct_c4_sparse_metadata_for_replay(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    c4_page_table: torch.Tensor | None,
    dst_c4_sparse_raw_indices: torch.Tensor,
    dst_c4_sparse_page_indices: torch.Tensor,
    dst_c4_sparse_full_indices: torch.Tensor,
    *,
    page_size: int,
    index_topk: int,
    component_loc_ownership: bool,
) -> bool:
    tensors = (
        ctx_page_table,
        table_indices,
        positions,
        dst_c4_sparse_raw_indices,
        dst_c4_sparse_page_indices,
        dst_c4_sparse_full_indices,
    )
    rows = int(positions.numel())
    if (
        ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or positions.ndim != 1
        or dst_c4_sparse_raw_indices.ndim != 2
        or dst_c4_sparse_page_indices.shape != dst_c4_sparse_raw_indices.shape
        or dst_c4_sparse_full_indices.shape != dst_c4_sparse_raw_indices.shape
        or table_indices.shape != positions.shape
        or dst_c4_sparse_raw_indices.shape[0] < rows
        or page_size <= 0
        or page_size & (page_size - 1)
        or index_topk <= 0
    ):
        return False
    if component_loc_ownership:
        if c4_page_table is None or c4_page_table.ndim != 2 or c4_page_table.shape[0] < rows:
            return False
        tensors = (*tensors, c4_page_table)
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if rows <= 0:
        return True
    c4_width = int(dst_c4_sparse_raw_indices.shape[1])
    if c4_width <= 0:
        return False
    c4_component_page_size = max(int(page_size) // 4, 1)
    block = triton.next_power_of_2(max(c4_width, 1))
    dummy_c4_page_table = c4_page_table if c4_page_table is not None else ctx_page_table
    _direct_c4_sparse_metadata_for_replay_kernel[(rows,)](
        ctx_page_table,
        table_indices,
        positions,
        dummy_c4_page_table,
        dst_c4_sparse_raw_indices,
        dst_c4_sparse_page_indices,
        dst_c4_sparse_full_indices,
        ctx_page_table_stride0=ctx_page_table.stride(0),
        ctx_page_table_width=ctx_page_table.shape[1],
        c4_page_table_width=dummy_c4_page_table.shape[1],
        c4_width=c4_width,
        page_size=int(page_size),
        index_topk=int(index_topk),
        component_loc_ownership=bool(component_loc_ownership),
        c4_component_page_size=c4_component_page_size,
        BLOCK=block,
    )
    return True


def direct_decode_index_metadata_for_replay(
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
    *,
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
    rows = int(positions.numel())
    tensors = [ctx_page_table, table_indices, positions]
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
    if (
        not (direct_swa or direct_c4 or direct_c128)
        or ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or positions.ndim != 1
        or table_indices.shape != positions.shape
        or page_size <= 0
        or page_size & (page_size - 1)
        or window_size <= 0
        or index_topk <= 0
    ):
        return False
    if direct_swa and swa_independent and (
        swa_full_to_swa_page is None
        or swa_full_to_swa_page.ndim != 1
        or swa_dummy_token_start < 0
        or swa_dummy_page < 0
    ):
        return False
    if direct_swa and (dst_swa_page_indices.ndim != 2 or dst_swa_page_indices.shape[0] < rows):
        return False
    if direct_c4:
        assert c4_page_table is not None
        if (
            c4_page_table.ndim != 2
            or c4_page_table.shape[0] < rows
            or dst_c4_sparse_raw_indices.ndim != 2
            or dst_c4_sparse_raw_indices.shape[0] < rows
            or dst_c4_sparse_page_indices.shape != dst_c4_sparse_raw_indices.shape
            or dst_c4_sparse_full_indices.shape != dst_c4_sparse_raw_indices.shape
        ):
            return False
    if direct_c128:
        assert c128_page_table is not None
        if (
            c128_page_table.ndim != 2
            or c128_page_table.shape[0] < rows
            or dst_c128_raw_indices.ndim != 2
            or dst_c128_raw_indices.shape[0] < rows
            or dst_c128_page_indices.shape != dst_c128_raw_indices.shape
            or dst_c128_full_indices.shape != dst_c128_raw_indices.shape
        ):
            return False
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if rows <= 0:
        return True

    dummy_c4_page_table = c4_page_table if c4_page_table is not None else ctx_page_table
    dummy_c128_page_table = c128_page_table if c128_page_table is not None else ctx_page_table
    dummy_swa_full_to_swa_page = (
        swa_full_to_swa_page if swa_full_to_swa_page is not None else table_indices
    )
    swa_width = int(dst_swa_page_indices.shape[1]) if direct_swa else 1
    c4_width = int(dst_c4_sparse_raw_indices.shape[1]) if direct_c4 else 1
    c128_width = int(dst_c128_raw_indices.shape[1]) if direct_c128 else 1
    max_width = max(
        swa_width if direct_swa else 0,
        c4_width if direct_c4 else 0,
        c128_width if direct_c128 else 0,
        1,
    )
    block = triton.next_power_of_2(max_width)
    c4_component_page_size = max(int(page_size) // 4, 1)
    c128_component_page_size = max(int(page_size) // 128, 1)
    _direct_decode_index_metadata_for_replay_kernel[(rows,)](
        ctx_page_table,
        table_indices,
        positions,
        dummy_c4_page_table,
        dummy_c128_page_table,
        dummy_swa_full_to_swa_page,
        dst_swa_page_indices,
        dst_c4_sparse_raw_indices,
        dst_c4_sparse_page_indices,
        dst_c4_sparse_full_indices,
        dst_c128_raw_indices,
        dst_c128_page_indices,
        dst_c128_full_indices,
        ctx_page_table_stride0=ctx_page_table.stride(0),
        ctx_page_table_width=ctx_page_table.shape[1],
        swa_full_to_swa_page_width=dummy_swa_full_to_swa_page.shape[0],
        c4_page_table_width=dummy_c4_page_table.shape[1],
        c128_page_table_width=dummy_c128_page_table.shape[1],
        swa_width=swa_width,
        c4_width=c4_width,
        c128_width=c128_width,
        page_size=int(page_size),
        window_size=int(window_size),
        index_topk=int(index_topk),
        direct_swa=bool(direct_swa),
        direct_c4=bool(direct_c4),
        direct_c128=bool(direct_c128),
        swa_independent=bool(swa_independent),
        swa_dummy_token_start=int(swa_dummy_token_start),
        swa_dummy_page=int(swa_dummy_page),
        c4_component_page_size=c4_component_page_size,
        c128_component_page_size=c128_component_page_size,
        BLOCK=block,
    )
    return True


def copy_masked_compressed_locs(
    raw_out_loc: torch.Tensor,
    positions: torch.Tensor,
    c4_out_loc: torch.Tensor | None,
    c128_out_loc: torch.Tensor | None,
    rows: int,
) -> bool:
    if (
        c4_out_loc is None
        or c128_out_loc is None
        or not raw_out_loc.is_cuda
        or not positions.is_cuda
        or not c4_out_loc.is_cuda
        or not c128_out_loc.is_cuda
        or c4_out_loc.numel() != c128_out_loc.numel()
        or rows < 0
    ):
        return False
    n_elements = c4_out_loc.numel()
    if positions.numel() < rows or raw_out_loc.numel() < rows:
        return False
    block = 16
    grid = (triton.cdiv(n_elements, block),)
    _copy_masked_compressed_locs_kernel[grid](
        raw_out_loc,
        positions,
        c4_out_loc,
        c128_out_loc,
        rows,
        n_elements,
        BLOCK=block,
    )
    return True


def copy_component_write_locs_for_replay(
    *,
    c4_page_table: torch.Tensor,
    c128_page_table: torch.Tensor,
    c4_indexer_page_table: torch.Tensor,
    positions: torch.Tensor,
    c4_out_loc: torch.Tensor,
    c128_out_loc: torch.Tensor,
    c4_indexer_out_loc: torch.Tensor,
    rows: int,
    page_size: int,
) -> bool:
    tensors = (
        c4_page_table,
        c128_page_table,
        c4_indexer_page_table,
        positions,
        c4_out_loc,
        c128_out_loc,
        c4_indexer_out_loc,
    )
    if (
        rows < 0
        or page_size <= 0
        or c4_page_table.ndim != 2
        or c128_page_table.ndim != 2
        or c4_indexer_page_table.ndim != 2
        or positions.ndim != 1
        or c4_out_loc.ndim != 1
        or c128_out_loc.ndim != 1
        or c4_indexer_out_loc.ndim != 1
    ):
        return False
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if (
        positions.numel() < rows
        or c4_page_table.shape[0] < rows
        or c128_page_table.shape[0] < rows
        or c4_indexer_page_table.shape[0] < rows
        or c4_out_loc.numel() < rows
        or c128_out_loc.numel() < rows
        or c4_indexer_out_loc.numel() < rows
    ):
        return False
    n_elements = min(c4_out_loc.numel(), c128_out_loc.numel(), c4_indexer_out_loc.numel())
    if n_elements <= 0:
        return True
    c4_component_page_size = max(int(page_size) // 4, 1)
    c128_component_page_size = max(int(page_size) // 128, 1)
    block = 16
    grid = (triton.cdiv(n_elements, block),)
    _copy_component_write_locs_for_replay_kernel[grid](
        c4_page_table,
        c128_page_table,
        c4_indexer_page_table,
        positions,
        c4_out_loc,
        c128_out_loc,
        c4_indexer_out_loc,
        rows,
        n_elements,
        c4_page_table_width=c4_page_table.shape[1],
        c128_page_table_width=c128_page_table.shape[1],
        c4_indexer_page_table_width=c4_indexer_page_table.shape[1],
        c4_component_page_size=c4_component_page_size,
        c128_component_page_size=c128_component_page_size,
        BLOCK=block,
    )
    return True


def prep_decode_metadata_in_graph(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    materialized_seq_lens: torch.Tensor,
    c4_page_table: torch.Tensor,
    c128_page_table: torch.Tensor,
    c4_indexer_page_table: torch.Tensor,
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
    dst_c4_out_loc: torch.Tensor,
    dst_c128_out_loc: torch.Tensor,
    dst_c4_indexer_out_loc: torch.Tensor,
    swa_full_to_swa_page: torch.Tensor | None = None,
    dst_swa_out_loc: torch.Tensor | None = None,
    *,
    page_size: int,
    window_size: int,
    index_topk: int,
    swa_independent: bool = False,
    swa_dummy_token_start: int = -1,
    swa_dummy_page: int = -1,
    write_swa_out_loc: bool = False,
) -> bool:
    rows = int(positions.numel())
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
            or swa_full_to_swa_page.ndim != 1
            or swa_dummy_token_start < 0
            or swa_dummy_page < 0
        ):
            return False
        tensors.append(swa_full_to_swa_page)
    if write_swa_out_loc:
        if dst_swa_out_loc is None or dst_swa_out_loc.ndim != 1:
            return False
        tensors.append(dst_swa_out_loc)
    if (
        rows <= 0
        or page_size <= 0
        or page_size & (page_size - 1)
        or window_size <= 0
        or index_topk <= 0
        or ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or raw_out_loc.ndim != 1
        or materialized_seq_lens.ndim != 1
        or c4_page_table.ndim != 2
        or c128_page_table.ndim != 2
        or c4_indexer_page_table.ndim != 2
    ):
        return False
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if any(t.numel() < rows for t in tensors[1:5]):
        return False
    if (
        c4_page_table.shape[0] < rows
        or c128_page_table.shape[0] < rows
        or c4_indexer_page_table.shape[0] < rows
        or any(t.numel() < rows for t in tensors[8:14])
        or any(t.ndim != 2 or t.shape[0] < rows for t in tensors[14:21])
        or any(t.numel() < rows for t in tensors[21:24])
    ):
        return False
    swa_width = int(dst_swa_page_indices.shape[1])
    c4_width = int(dst_c4_sparse_raw_indices.shape[1])
    c128_width = int(dst_c128_raw_indices.shape[1])
    if (
        dst_c4_sparse_page_indices.shape != dst_c4_sparse_raw_indices.shape
        or dst_c4_sparse_full_indices.shape != dst_c4_sparse_raw_indices.shape
        or dst_c128_page_indices.shape != dst_c128_raw_indices.shape
        or dst_c128_full_indices.shape != dst_c128_raw_indices.shape
        or swa_width <= 0
        or c4_width <= 0
        or c128_width <= 0
    ):
        return False
    max_width = max(swa_width, c4_width, c128_width, 1)
    block = triton.next_power_of_2(max_width)
    c4_component_page_size = max(int(page_size) // 4, 1)
    c128_component_page_size = max(int(page_size) // 128, 1)
    dummy_swa_full_to_swa_page = (
        swa_full_to_swa_page if swa_full_to_swa_page is not None else table_indices
    )
    dummy_swa_out_loc = dst_swa_out_loc if dst_swa_out_loc is not None else raw_out_loc
    _prep_decode_metadata_in_graph_kernel[(rows,)](
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
        dummy_swa_full_to_swa_page,
        dummy_swa_out_loc,
        ctx_page_table_stride0=ctx_page_table.stride(0),
        ctx_page_table_width=ctx_page_table.shape[1],
        swa_full_to_swa_page_width=dummy_swa_full_to_swa_page.shape[0],
        c4_page_table_width=c4_page_table.shape[1],
        c128_page_table_width=c128_page_table.shape[1],
        c4_indexer_page_table_width=c4_indexer_page_table.shape[1],
        swa_width=swa_width,
        c4_width=c4_width,
        c128_width=c128_width,
        page_size=int(page_size),
        window_size=int(window_size),
        index_topk=int(index_topk),
        swa_independent=bool(swa_independent),
        swa_dummy_token_start=int(swa_dummy_token_start),
        swa_dummy_page=int(swa_dummy_page),
        write_swa_out_loc=bool(write_swa_out_loc),
        c4_component_page_size=c4_component_page_size,
        c128_component_page_size=c128_component_page_size,
        BLOCK=block,
    )
    return True


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
    if rows <= 0:
        return True
    max_elements = max(
        rows,
        rows + 1,
        rows * dst_page_table.shape[1],
        0 if skip_swa_page_indices else rows * dst_swa_page_indices.shape[1],
        0 if skip_c4_sparse_indices else rows * dst_c4_sparse_raw_indices.shape[1],
        0 if skip_c4_sparse_indices else rows * dst_c4_sparse_page_indices.shape[1],
        0 if skip_c4_sparse_indices else rows * dst_c4_sparse_full_indices.shape[1],
        0 if skip_c128_indices else rows * dst_c128_raw_indices.shape[1],
        0 if skip_c128_indices else rows * dst_c128_page_indices.shape[1],
        0 if skip_c128_indices else rows * dst_c128_full_indices.shape[1],
    )
    block = 256
    grid = (triton.cdiv(max_elements, block), 20)
    _copy_decode_metadata_for_replay_kernel[grid](
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
        rows=int(rows),
        graph_inputs_bound=bool(graph_inputs_bound),
        dst_page_table_width=dst_page_table.shape[1],
        src_page_table_width=src_page_table.shape[1],
        dst_swa_page_indices_width=dst_swa_page_indices.shape[1],
        src_swa_page_indices_width=src_swa_page_indices.shape[1],
        dst_c4_sparse_raw_indices_width=dst_c4_sparse_raw_indices.shape[1],
        src_c4_sparse_raw_indices_width=src_c4_sparse_raw_indices.shape[1],
        dst_c4_sparse_page_indices_width=dst_c4_sparse_page_indices.shape[1],
        src_c4_sparse_page_indices_width=src_c4_sparse_page_indices.shape[1],
        dst_c4_sparse_full_indices_width=dst_c4_sparse_full_indices.shape[1],
        src_c4_sparse_full_indices_width=src_c4_sparse_full_indices.shape[1],
        dst_c128_raw_indices_width=dst_c128_raw_indices.shape[1],
        src_c128_raw_indices_width=src_c128_raw_indices.shape[1],
        dst_c128_page_indices_width=dst_c128_page_indices.shape[1],
        src_c128_page_indices_width=src_c128_page_indices.shape[1],
        dst_c128_full_indices_width=dst_c128_full_indices.shape[1],
        src_c128_full_indices_width=src_c128_full_indices.shape[1],
        skip_swa_page_indices=bool(skip_swa_page_indices),
        skip_c4_sparse_indices=bool(skip_c4_sparse_indices),
        skip_c128_indices=bool(skip_c128_indices),
        BLOCK=block,
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


def paged_mqa_attention_bf16(
    q: torch.Tensor,
    cache: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
    max_length: int,
) -> torch.Tensor | None:
    if (
        q.ndim != 3
        or cache.ndim != 2
        or q.numel() == 0
        or cache.shape[-1] != q.shape[-1]
        or not q.is_cuda
        or not cache.is_cuda
        or not indptr.is_cuda
        or not indices.is_cuda
        or not lengths.is_cuda
        or not q.is_contiguous()
        or not cache.is_contiguous()
    ):
        return None
    if q.dtype not in (torch.bfloat16, torch.float32) or cache.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        return None
    if q.shape[0] != lengths.numel() or indptr.numel() != q.shape[0] + 1:
        return None
    if max_length == 0:
        return torch.zeros_like(q)
    if max_length > 1024 or q.shape[-1] > 256:
        return None
    if attn_sink is not None and (not attn_sink.is_cuda or attn_sink.numel() < q.shape[1]):
        return None

    tokens, heads, dim = q.shape
    out = torch.empty_like(q)
    sink = (
        attn_sink[:heads].to(device=q.device, dtype=torch.float32).contiguous()
        if attn_sink is not None
        else q.new_empty((1,), dtype=torch.float32)
    )
    block_n = 32
    block_d = triton.next_power_of_2(dim)
    _paged_mqa_attention_bf16_kernel[(tokens, heads)](
        q,
        cache,
        indptr.contiguous(),
        indices.contiguous(),
        lengths.contiguous(),
        sink,
        out,
        num_heads=heads,
        dim=dim,
        softmax_scale=float(softmax_scale),
        max_length=int(max_length),
        has_sink=attn_sink is not None,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def indexer_bf16_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor | None:
    if (
        q.ndim != 3
        or cache.ndim != 2
        or weights.ndim not in (2, 3)
        or seq_lens.ndim != 1
        or page_table.ndim != 2
        or q.numel() == 0
        or cache.shape[-1] != q.shape[-1]
        or weights.shape[0] != q.shape[0]
        or weights.shape[1] != q.shape[1]
        or seq_lens.shape[0] != q.shape[0]
        or page_table.shape[0] != q.shape[0]
        or not q.is_cuda
        or not cache.is_cuda
        or not weights.is_cuda
        or not seq_lens.is_cuda
        or not page_table.is_cuda
        or not q.is_contiguous()
        or not cache.is_contiguous()
    ):
        return None
    if q.dtype not in (torch.bfloat16, torch.float32) or cache.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        return None
    if page_size <= 0 or page_size & (page_size - 1):
        return None
    if q.shape[-1] > 256:
        return None

    if max_seq_len is None:
        max_seq_len = int(seq_lens.clamp_min(0).max().item())
    else:
        max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        return torch.empty((q.shape[0], 0), dtype=torch.float32, device=q.device)

    q_c = q.contiguous()
    cache_c = cache.contiguous()
    weights_c = weights.squeeze(-1).to(device=q.device, dtype=torch.float32).contiguous()
    seq_lens_c = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    page_table_c = page_table.to(device=q.device, dtype=torch.int32).contiguous()
    logits = torch.empty((q.shape[0], max_seq_len), dtype=torch.float32, device=q.device)
    block_n = 16
    block_d = triton.next_power_of_2(q.shape[-1])
    grid = (q.shape[0], triton.cdiv(max_seq_len, block_n))
    _indexer_bf16_logits_kernel[grid](
        q_c,
        cache_c,
        weights_c,
        seq_lens_c,
        page_table_c,
        logits,
        num_pages=page_table_c.shape[1],
        num_heads=q.shape[1],
        dim=q.shape[-1],
        page_size=int(page_size),
        page_bits=(int(page_size) - 1).bit_length(),
        max_seq_len=max_seq_len,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return logits


def indexer_fp8_quant_store(
    kv: torch.Tensor,
    loc: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
) -> bool:
    if (
        kv.ndim != 2
        or loc.ndim != 1
        or values.ndim != 2
        or scales.ndim != 2
        or kv.shape[0] != loc.numel()
        or values.shape[-1] != kv.shape[-1]
        or scales.shape[-1] != 4
        or kv.shape[-1] > 256
        or not kv.is_cuda
        or not loc.is_cuda
        or not values.is_cuda
        or not scales.is_cuda
        or values.dtype is not torch.uint8
        or scales.dtype is not torch.uint8
        or not kv.is_contiguous()
        or not values.is_contiguous()
        or not scales.is_contiguous()
    ):
        return False
    if kv.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if loc.dtype not in (torch.int32, torch.int64):
        return False

    loc_c = loc.to(device=kv.device, dtype=torch.int64).contiguous()
    block_d = triton.next_power_of_2(kv.shape[-1])
    _indexer_fp8_quant_store_kernel[(kv.shape[0],)](
        kv,
        loc_c,
        values,
        scales,
        rows=kv.shape[0],
        dim=kv.shape[-1],
        BLOCK_D=block_d,
        num_warps=4,
    )
    return True


def indexer_fp8_quantize(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        kv.ndim < 2
        or kv.shape[-1] > 256
        or not kv.is_cuda
        or not kv.is_contiguous()
        or kv.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return None
    flat = kv.view(-1, kv.shape[-1])
    values = torch.empty_like(flat, dtype=torch.uint8)
    scales = torch.empty((flat.shape[0], 4), dtype=torch.uint8, device=kv.device)
    block_d = triton.next_power_of_2(kv.shape[-1])
    _indexer_fp8_quantize_kernel[(flat.shape[0],)](
        flat,
        values,
        scales,
        dim=kv.shape[-1],
        BLOCK_D=block_d,
        num_warps=4,
    )
    return (
        values.view(*kv.shape[:-1], kv.shape[-1]).contiguous(),
        scales.view(*kv.shape[:-1], 4).contiguous(),
    )


def indexer_fp8_quantize_fold(
    q: torch.Tensor,
    weights: torch.Tensor,
    *,
    softmax_scale: float,
    head_scale: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        q.ndim != 3
        or weights.ndim not in (2, 3)
        or weights.shape[:2] != q.shape[:2]
        or q.shape[-1] > 256
        or not q.is_cuda
        or not weights.is_cuda
        or not q.is_contiguous()
        or q.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return None
    weights_c = weights.squeeze(-1).to(device=q.device, dtype=torch.float32).contiguous()
    q_values = torch.empty_like(q, dtype=torch.uint8)
    weights_out = torch.empty_like(weights_c, dtype=torch.float32)
    block_d = triton.next_power_of_2(q.shape[-1])
    _indexer_fp8_quantize_fold_kernel[(q.shape[0] * q.shape[1],)](
        q,
        weights_c,
        q_values,
        weights_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights_c.stride(0),
        weights_c.stride(1),
        weights_out.stride(0),
        weights_out.stride(1),
        num_heads=q.shape[1],
        dim=q.shape[-1],
        softmax_scale=float(softmax_scale),
        head_scale=float(head_scale),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return q_values, weights_out


def indexer_fp8_paged_quant_store(
    kv: torch.Tensor,
    loc: torch.Tensor,
    cache: torch.Tensor,
    *,
    page_size: int,
) -> bool:
    if (
        kv.ndim != 2
        or loc.ndim != 1
        or cache.ndim != 2
        or kv.shape[0] != loc.numel()
        or kv.shape[-1] > 256
        or page_size <= 0
        or cache.shape[-1] != page_size * (kv.shape[-1] + 4)
        or not kv.is_cuda
        or not loc.is_cuda
        or not cache.is_cuda
        or cache.dtype is not torch.uint8
        or not kv.is_contiguous()
        or not cache.is_contiguous()
    ):
        return False
    if kv.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if loc.dtype not in (torch.int32, torch.int64):
        return False

    loc_c = loc.to(device=kv.device, dtype=torch.int64).contiguous()
    block_d = triton.next_power_of_2(kv.shape[-1])
    _indexer_fp8_paged_quant_store_kernel[(kv.shape[0],)](
        kv,
        loc_c,
        cache,
        dim=kv.shape[-1],
        page_size=int(page_size),
        page_bytes=int(cache.shape[-1]),
        BLOCK_D=block_d,
        num_warps=1,
    )
    return True


def fp8_activation_quantize(
    x: torch.Tensor,
    *,
    block_size: int = 128,
) -> torch.Tensor | None:
    if (
        x.ndim == 0
        or x.numel() == 0
        or x.shape[-1] % block_size != 0
        or block_size <= 0
        or block_size & (block_size - 1)
        or block_size > 1024
        or not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return None
    x_c = x.contiguous()
    flat = x_c.view(-1, x_c.shape[-1])
    out = torch.empty_like(flat)
    block = triton.next_power_of_2(block_size)
    grid = (flat.shape[0], flat.shape[1] // block_size)
    _fp8_activation_quantize_kernel[grid](
        flat,
        out,
        cols=flat.shape[1],
        block_size=int(block_size),
        BLOCK=block,
        num_warps=4,
    )
    return out.view_as(x_c).reshape_as(x)


def indexer_fp8_logits(
    q_values: torch.Tensor,
    cache_values: torch.Tensor,
    cache_scales: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor | None:
    if (
        q_values.ndim != 3
        or cache_values.ndim != 2
        or cache_scales.ndim != 2
        or weights.ndim not in (2, 3)
        or seq_lens.ndim != 1
        or page_table.ndim != 2
        or q_values.numel() == 0
        or cache_values.shape[-1] != q_values.shape[-1]
        or cache_scales.shape[0] != cache_values.shape[0]
        or cache_scales.shape[-1] != 4
        or weights.shape[0] != q_values.shape[0]
        or weights.shape[1] != q_values.shape[1]
        or seq_lens.shape[0] != q_values.shape[0]
        or page_table.shape[0] != q_values.shape[0]
        or not q_values.is_cuda
        or not cache_values.is_cuda
        or not cache_scales.is_cuda
        or not weights.is_cuda
        or not seq_lens.is_cuda
        or not page_table.is_cuda
        or q_values.dtype is not torch.uint8
        or cache_values.dtype is not torch.uint8
        or cache_scales.dtype is not torch.uint8
        or not q_values.is_contiguous()
        or not cache_values.is_contiguous()
        or not cache_scales.is_contiguous()
    ):
        return None
    if page_size <= 0 or page_size & (page_size - 1):
        return None
    if q_values.shape[-1] > 256:
        return None

    if max_seq_len is None:
        max_seq_len = int(seq_lens.clamp_min(0).max().item())
    else:
        max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        return torch.empty((q_values.shape[0], 0), dtype=torch.float32, device=q_values.device)

    q_c = q_values.contiguous()
    cache_values_c = cache_values.contiguous()
    cache_scales_c = cache_scales.contiguous()
    weights_c = weights.squeeze(-1).to(device=q_values.device, dtype=torch.float32).contiguous()
    seq_lens_c = seq_lens.to(device=q_values.device, dtype=torch.int32).contiguous()
    page_table_c = page_table.to(device=q_values.device, dtype=torch.int32).contiguous()
    logits = torch.empty(
        (q_values.shape[0], max_seq_len), dtype=torch.float32, device=q_values.device
    )
    block_n = 16
    block_d = triton.next_power_of_2(q_values.shape[-1])
    grid = (q_values.shape[0], triton.cdiv(max_seq_len, block_n))
    _indexer_fp8_logits_kernel[grid](
        q_c,
        cache_values_c,
        cache_scales_c,
        weights_c,
        seq_lens_c,
        page_table_c,
        logits,
        num_pages=page_table_c.shape[1],
        num_heads=q_values.shape[1],
        dim=q_values.shape[-1],
        page_size=int(page_size),
        page_bits=(int(page_size) - 1).bit_length(),
        max_seq_len=max_seq_len,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return logits


def indexer_fp8_paged_logits(
    q_values: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    max_seq_len: int | None = None,
) -> torch.Tensor | None:
    if (
        q_values.ndim != 3
        or cache.ndim != 2
        or weights.ndim not in (2, 3)
        or seq_lens.ndim != 1
        or page_table.ndim != 2
        or q_values.numel() == 0
        or cache.shape[-1] != page_size * (q_values.shape[-1] + 4)
        or weights.shape[0] != q_values.shape[0]
        or weights.shape[1] != q_values.shape[1]
        or seq_lens.shape[0] != q_values.shape[0]
        or page_table.shape[0] != q_values.shape[0]
        or not q_values.is_cuda
        or not cache.is_cuda
        or not weights.is_cuda
        or not seq_lens.is_cuda
        or not page_table.is_cuda
        or q_values.dtype is not torch.uint8
        or cache.dtype is not torch.uint8
        or not q_values.is_contiguous()
        or not cache.is_contiguous()
    ):
        return None
    if page_size <= 0 or page_size & (page_size - 1):
        return None
    if q_values.shape[-1] > 256:
        return None

    if max_seq_len is None:
        max_seq_len = int(seq_lens.clamp_min(0).max().item())
    else:
        max_seq_len = int(max_seq_len)
    if max_seq_len <= 0:
        return torch.empty((q_values.shape[0], 0), dtype=torch.float32, device=q_values.device)

    q_c = q_values.contiguous()
    cache_c = cache.contiguous()
    weights_c = weights.squeeze(-1).to(device=q_values.device, dtype=torch.float32).contiguous()
    seq_lens_c = seq_lens.to(device=q_values.device, dtype=torch.int32).contiguous()
    page_table_c = page_table.to(device=q_values.device, dtype=torch.int32).contiguous()
    logits = torch.full(
        (q_values.shape[0], max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=q_values.device,
    )
    block_h = max(16, triton.next_power_of_2(q_values.shape[1]))
    block_d = triton.next_power_of_2(q_values.shape[-1])
    block_n = triton.next_power_of_2(page_size)
    e4m3fn_to_bf16 = _get_e4m3fn_to_bf16_lut(q_values.device)
    grid = (q_values.shape[0], page_table_c.shape[1])
    _indexer_fp8_paged_logits_kernel[grid](
        q_c,
        cache_c,
        e4m3fn_to_bf16,
        weights_c,
        seq_lens_c,
        page_table_c,
        logits,
        q_c.stride(0),
        q_c.stride(1),
        q_c.stride(2),
        weights_c.stride(0),
        weights_c.stride(1),
        page_table_c.stride(0),
        page_table_c.stride(1),
        logits.stride(0),
        logits.stride(1),
        num_pages=page_table_c.shape[1],
        num_heads=q_values.shape[1],
        dim=q_values.shape[-1],
        page_size=int(page_size),
        page_bytes=int(cache_c.shape[-1]),
        max_seq_len=max_seq_len,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=4,
    )
    return logits


def remap_indexer_topk_locs(
    raw_indices: torch.Tensor,
    component_page_table: torch.Tensor,
    full_page_table: torch.Tensor,
    *,
    component_page_size: int,
    full_page_size: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        raw_indices.ndim != 2
        or component_page_table.ndim != 2
        or full_page_table.ndim != 2
        or component_page_table.shape[0] != raw_indices.shape[0]
        or full_page_table.shape[0] != raw_indices.shape[0]
        or not raw_indices.is_cuda
        or not component_page_table.is_cuda
        or not full_page_table.is_cuda
        or raw_indices.dtype not in (torch.int32, torch.int64)
        or component_page_table.dtype not in (torch.int32, torch.int64)
        or full_page_table.dtype not in (torch.int32, torch.int64)
        or not raw_indices.is_contiguous()
        or not component_page_table.is_contiguous()
        or not full_page_table.is_contiguous()
        or component_page_size <= 0
        or full_page_size <= 0
        or ratio <= 0
    ):
        return None
    component_locs = torch.empty_like(raw_indices, dtype=torch.int32)
    full_locs = torch.empty_like(raw_indices, dtype=torch.int32)
    if raw_indices.numel() == 0:
        return component_locs, full_locs
    block = 256
    _remap_indexer_topk_locs_kernel[(triton.cdiv(raw_indices.numel(), block),)](
        raw_indices,
        component_page_table,
        full_page_table,
        component_locs,
        full_locs,
        raw_indices.numel(),
        width=raw_indices.shape[1],
        component_table_width=component_page_table.shape[1],
        full_table_width=full_page_table.shape[1],
        component_page_size=int(component_page_size),
        full_page_size=int(full_page_size),
        ratio=int(ratio),
        BLOCK=block,
        num_warps=4,
    )
    return component_locs, full_locs


def c128_prefill_page_indices_one_surface(
    component_page_table: torch.Tensor,
    c128_lengths: torch.Tensor,
    *,
    width: int,
    component_page_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build final int32 C128 locations without a full-matrix temporary."""
    rows = int(c128_lengths.numel())
    width = int(width)
    if (
        component_page_table.ndim != 2
        or c128_lengths.ndim != 1
        or component_page_table.shape[0] != rows
        or not component_page_table.is_cuda
        or not c128_lengths.is_cuda
        or component_page_table.device != c128_lengths.device
        or component_page_table.dtype != torch.int32
        or c128_lengths.dtype != torch.int32
        or not component_page_table.is_contiguous()
        or not c128_lengths.is_contiguous()
        or width <= 0
        or component_page_size <= 0
    ):
        return None
    if out is None:
        output = torch.empty((rows, width), dtype=torch.int32, device=c128_lengths.device)
    else:
        if (
            out.shape != (rows, width)
            or out.dtype != torch.int32
            or out.device != c128_lengths.device
            or not out.is_contiguous()
        ):
            return None
        output = out
    if output.numel() == 0:
        return output
    block = 256
    _c128_prefill_page_indices_kernel[(rows, triton.cdiv(width, block))](
        component_page_table,
        c128_lengths,
        output,
        width=width,
        component_table_width=component_page_table.shape[1],
        component_page_size=int(component_page_size),
        BLOCK=block,
        num_warps=4,
    )
    return output


def hc_split_pre(
    mixes: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        x.ndim != 3
        or mixes.ndim != 2
        or x.numel() == 0
        or not x.is_cuda
        or not mixes.is_cuda
        or not scale.is_cuda
        or not base.is_cuda
        or not x.is_contiguous()
        or not mixes.is_contiguous()
        or x.shape[0] != mixes.shape[0]
        or x.shape[1] != hc_mult
        or hc_mult < 1
        or hc_mult > 8
    ):
        return None
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.shape[1] != mix_hc or scale.numel() < 3 or base.numel() < mix_hc:
        return None
    if x.dtype is not torch.bfloat16:
        return None
    tokens, _, hidden = x.shape
    y = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    post = torch.empty((tokens, hc_mult), dtype=x.dtype, device=x.device)
    comb = torch.empty((tokens, hc_mult, hc_mult), dtype=x.dtype, device=x.device)
    block_hc = triton.next_power_of_2(hc_mult)
    block_d = 128
    grid = (tokens, triton.cdiv(hidden, block_d))
    _hc_split_pre_kernel[grid](
        mixes,
        x,
        scale.contiguous(),
        base.contiguous(),
        y,
        post,
        comb,
        tokens=tokens,
        hidden=hidden,
        hc_mult=int(hc_mult),
        mix_hc=int(mix_hc),
        eps=float(eps),
        sinkhorn_steps=max(int(sinkhorn_iters) - 1, 0),
        BLOCK_HC=block_hc,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return y, post, comb


def hc_prenorm_split_pre(
    mixes: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        x.ndim != 3
        or mixes.ndim != 2
        or x.numel() == 0
        or not x.is_cuda
        or not mixes.is_cuda
        or not scale.is_cuda
        or not base.is_cuda
        or not x.is_contiguous()
        or not mixes.is_contiguous()
        or x.shape[0] != mixes.shape[0]
        or x.shape[1] != hc_mult
        or hc_mult < 1
        or hc_mult > 8
    ):
        return None
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.shape[1] != mix_hc or scale.numel() < 3 or base.numel() < mix_hc:
        return None
    if x.dtype is not torch.bfloat16:
        return None
    tokens, _, hidden = x.shape
    pre = torch.empty((tokens, hc_mult), dtype=x.dtype, device=x.device)
    y = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    post = torch.empty((tokens, hc_mult), dtype=x.dtype, device=x.device)
    comb = torch.empty((tokens, hc_mult, hc_mult), dtype=x.dtype, device=x.device)
    block_hc = triton.next_power_of_2(hc_mult)
    _hc_prenorm_split_pre_kernel[(tokens,)](
        mixes,
        x,
        scale.contiguous(),
        base.contiguous(),
        pre,
        post,
        comb,
        tokens=tokens,
        hidden=hidden,
        hc_mult=int(hc_mult),
        mix_hc=int(mix_hc),
        eps=float(eps),
        norm_eps=float(norm_eps),
        sinkhorn_steps=max(int(sinkhorn_iters) - 1, 0),
        BLOCK_HC=block_hc,
        BLOCK_N=1024,
        num_warps=4,
    )
    block_d = 256 if hidden >= 256 else 128
    grid = (tokens, triton.cdiv(hidden, block_d))
    _hc_layer_input_kernel[grid](
        x,
        pre,
        y,
        tokens=tokens,
        hidden=hidden,
        hc_mult=int(hc_mult),
        BLOCK_HC=block_hc,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return y, post, comb


def hc_prenorm_head(
    mixes: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int,
    eps: float,
    norm_eps: float,
) -> torch.Tensor | None:
    if (
        x.ndim != 3
        or mixes.ndim != 2
        or x.numel() == 0
        or not x.is_cuda
        or not mixes.is_cuda
        or not scale.is_cuda
        or not base.is_cuda
        or not x.is_contiguous()
        or not mixes.is_contiguous()
        or x.shape[0] != mixes.shape[0]
        or x.shape[1] != hc_mult
        or hc_mult < 1
        or hc_mult > 8
    ):
        return None
    if mixes.shape[1] != hc_mult or scale.numel() < 1 or base.numel() < hc_mult:
        return None
    if x.dtype is not torch.bfloat16:
        return None
    tokens, _, hidden = x.shape
    pre = torch.empty((tokens, hc_mult), dtype=x.dtype, device=x.device)
    y = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_hc = triton.next_power_of_2(hc_mult)
    _hc_prenorm_head_pre_kernel[(tokens,)](
        mixes,
        x,
        scale.contiguous(),
        base.contiguous(),
        pre,
        tokens=tokens,
        hidden=hidden,
        hc_mult=int(hc_mult),
        eps=float(eps),
        norm_eps=float(norm_eps),
        BLOCK_HC=block_hc,
        BLOCK_N=1024,
        num_warps=4,
    )
    block_d = 256 if hidden >= 256 else 128
    grid = (tokens, triton.cdiv(hidden, block_d))
    _hc_layer_input_kernel[grid](
        x,
        pre,
        y,
        tokens=tokens,
        hidden=hidden,
        hc_mult=int(hc_mult),
        BLOCK_HC=block_hc,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return y


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor | None:
    if (
        x.ndim != 2
        or residual.ndim != 3
        or post.ndim != 2
        or comb.ndim != 3
        or x.numel() == 0
        or not x.is_cuda
        or not residual.is_cuda
        or not post.is_cuda
        or not comb.is_cuda
        or not x.is_contiguous()
        or not residual.is_contiguous()
        or not post.is_contiguous()
        or not comb.is_contiguous()
        or residual.shape[0] != x.shape[0]
        or post.shape[0] != x.shape[0]
        or comb.shape[0] != x.shape[0]
        or residual.shape[2] != x.shape[1]
        or post.shape[1] != residual.shape[1]
        or comb.shape[1] != residual.shape[1]
        or comb.shape[2] != residual.shape[1]
    ):
        return None
    if (
        x.dtype is not torch.bfloat16
        or residual.dtype is not torch.bfloat16
        or post.dtype is not torch.bfloat16
        or comb.dtype is not torch.bfloat16
    ):
        return None
    hc_mult = residual.shape[1]
    if hc_mult < 1 or hc_mult > 8:
        return None
    tokens, hidden = x.shape
    out = torch.empty_like(residual)
    block_hc = triton.next_power_of_2(hc_mult)
    block_d = 128
    grid = (tokens, triton.cdiv(hidden, block_d))
    _hc_post_kernel[grid](
        x,
        residual,
        post,
        comb,
        out,
        tokens=tokens,
        hidden=hidden,
        hc_mult=int(hc_mult),
        BLOCK_HC=block_hc,
        BLOCK_D=block_d,
        num_warps=4,
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


def wo_a_grouped_projection_fp8(
    o: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    num_local_groups: int,
    o_lora_rank: int,
) -> torch.Tensor | None:
    if (
        o.ndim != 3
        or o.dtype is not torch.bfloat16
        or not o.is_cuda
        or not weight.is_cuda
        or o.shape[1] != num_local_groups
        or weight.ndim != 2
        or weight.shape[0] != num_local_groups * o_lora_rank
        or weight.shape[1] != o.shape[-1]
        or o.stride(-1) != 1
    ):
        return None
    if o.shape[0] > 16:
        return None
    if getattr(torch, "float8_e4m3fn", None) is None or weight.dtype is not torch.float8_e4m3fn:
        return None
    if scale is not None and (scale.ndim != 2 or not scale.is_cuda):
        return None

    o_c = o.contiguous()
    weight_c = weight.contiguous().view(torch.uint8)
    scale_c = (
        scale.float().contiguous()
        if scale is not None
        else weight_c.new_empty((1, 1), dtype=torch.float32)
    )
    tokens, groups, d_per_group = o_c.shape
    out = torch.empty(
        (tokens, num_local_groups * o_lora_rank),
        dtype=o.dtype,
        device=o.device,
    )
    block_t = 16
    block_r = 64
    block_d = 64
    grid = (
        triton.cdiv(tokens, block_t),
        num_local_groups,
        triton.cdiv(o_lora_rank, block_r),
    )
    _wo_a_grouped_projection_fp8_kernel[grid](
        o_c,
        weight_c,
        scale_c,
        out,
        T=tokens,
        G=groups,
        D=d_per_group,
        R=o_lora_rank,
        SCALE_D=triton.cdiv(d_per_group, 128),
        HAS_SCALE=scale is not None,
        BLOCK_T=block_t,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
    )
    return out


def build_moe_route_plan(
    indices: torch.Tensor,
    *,
    num_experts: int,
    block_size_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        indices.ndim != 2
        or indices.numel() == 0
        or not indices.is_cuda
        or num_experts <= 0
        or block_size_m <= 0
        or num_experts > 1024
    ):
        return None

    route_count = indices.numel()
    max_padded = route_count + min(num_experts, route_count) * (block_size_m - 1)
    max_blocks = triton.cdiv(max_padded, block_size_m)
    indices_c = indices.contiguous()
    counts = torch.zeros((num_experts,), dtype=torch.int32, device=indices.device)
    padded_offsets = torch.empty_like(counts)
    blocks_per_expert = torch.empty_like(counts)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=indices.device)
    sorted_route_ids = torch.empty((max_padded,), dtype=torch.int32, device=indices.device)
    expert_ids = torch.zeros((max_blocks,), dtype=torch.int32, device=indices.device)

    block_routes = 256
    _moe_route_count_kernel[(triton.cdiv(route_count, block_routes),)](
        indices_c,
        counts,
        route_count,
        num_experts=int(num_experts),
        BLOCK_ROUTES=block_routes,
    )
    block_experts = triton.next_power_of_2(num_experts)
    _moe_route_offsets_kernel[(1,)](
        counts,
        padded_offsets,
        blocks_per_expert,
        num_tokens_post_padded,
        block_size_m=int(block_size_m),
        num_experts=int(num_experts),
        BLOCK_EXPERTS=block_experts,
    )
    _moe_route_fill_kernel[(num_experts,)](
        indices_c,
        counts,
        padded_offsets,
        blocks_per_expert,
        sorted_route_ids,
        expert_ids,
        route_count,
        block_size_m=int(block_size_m),
        BLOCK_ROUTES=block_routes,
    )
    return sorted_route_ids, expert_ids, num_tokens_post_padded


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


def _grouped_fp4_w13(
    a: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    route_count: int,
    topk: int,
    block_size_m: int,
    workspace=None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        a.ndim != 2
        or a.dtype is not torch.bfloat16
        or not a.is_cuda
        or not weight.is_cuda
        or not sorted_route_ids.is_cuda
        or not expert_ids.is_cuda
        or not num_tokens_post_padded.is_cuda
        or a.stride(-1) != 1
        or weight.ndim != 4
        or weight.shape[1] != 2
        or weight.dtype is not torch.int8
        or a.shape[-1] % 2 != 0
        or weight.shape[-1] * 2 < a.shape[-1]
    ):
        return None
    if route_count == 0:
        n = weight.shape[2]
        empty = torch.empty((0, n), dtype=a.dtype, device=a.device)
        return empty, empty

    weight_c = weight.contiguous()
    scale_c = (
        scale.float().contiguous()
        if scale is not None
        else weight_c.new_empty((1,), dtype=torch.float32)
    )
    n = weight_c.shape[2]
    gate = _workspace_tensor(
        workspace,
        "w13_gate",
        (route_count, n),
        a.dtype,
        a.device,
        zero=True,
    )
    up = _workspace_tensor(
        workspace,
        "w13_up",
        (route_count, n),
        a.dtype,
        a.device,
        zero=True,
    )
    block_n = 64
    block_k = 64
    grid = (
        triton.cdiv(sorted_route_ids.numel(), block_size_m),
        triton.cdiv(n, block_n),
    )
    _grouped_fp4_w13_kernel[grid](
        a.contiguous(),
        weight_c,
        scale_c,
        sorted_route_ids.contiguous(),
        expert_ids.contiguous(),
        num_tokens_post_padded.contiguous(),
        gate,
        up,
        route_count=route_count,
        topk=topk,
        N=n,
        K=a.shape[-1],
        WEIGHT_K_BYTES=weight_c.shape[-1],
        SCALE_K=triton.cdiv(a.shape[-1], 32),
        HAS_SCALE=scale is not None,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )
    return gate, up


def _grouped_fp4_linear(
    a: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    route_count: int,
    topk: int,
    block_size_m: int,
    slot: int | None,
    a_rows_are_routes: bool,
    workspace=None,
) -> torch.Tensor | None:
    if (
        a.ndim != 2
        or a.dtype is not torch.bfloat16
        or not a.is_cuda
        or not weight.is_cuda
        or not sorted_route_ids.is_cuda
        or not expert_ids.is_cuda
        or not num_tokens_post_padded.is_cuda
        or a.stride(-1) != 1
    ):
        return None
    if route_count == 0:
        n = weight.shape[-2]
        return torch.empty((0, n), dtype=a.dtype, device=a.device)
    if weight.dtype is not torch.int8 or weight.shape[-1] * 2 < a.shape[-1]:
        return None
    if slot is None:
        if weight.ndim != 3:
            return None
        n = weight.shape[1]
        has_slot = False
        slot_value = 0
    else:
        if weight.ndim != 4 or weight.shape[1] <= slot:
            return None
        n = weight.shape[2]
        has_slot = True
        slot_value = int(slot)
    if a.shape[-1] % 2 != 0:
        return None

    weight_c = weight.contiguous()
    scale_c = (
        scale.float().contiguous()
        if scale is not None
        else weight_c.new_empty((1,), dtype=torch.float32)
    )
    out = _workspace_tensor(
        workspace,
        "w2_routed",
        (route_count, n),
        a.dtype,
        a.device,
        zero=True,
    )
    block_n = 64
    block_k = 64
    grid = (
        triton.cdiv(sorted_route_ids.numel(), block_size_m),
        triton.cdiv(n, block_n),
    )
    _grouped_fp4_linear_kernel[grid](
        a.contiguous(),
        weight_c,
        scale_c,
        sorted_route_ids.contiguous(),
        expert_ids.contiguous(),
        num_tokens_post_padded.contiguous(),
        out,
        route_count=route_count,
        topk=topk,
        N=n,
        K=a.shape[-1],
        WEIGHT_K_BYTES=weight_c.shape[-1],
        SCALE_K=triton.cdiv(a.shape[-1], 32),
        HAS_SCALE=scale is not None,
        HAS_SLOT=has_slot,
        SLOT=slot_value,
        A_ROWS_ARE_ROUTES=bool(a_rows_are_routes),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
    )
    return out


def _sum_grouped_routes(
    routed: torch.Tensor,
    *,
    tokens: int,
    hidden: int,
    topk: int,
    workspace=None,
) -> torch.Tensor | None:
    if routed.ndim != 2 or not routed.is_cuda or routed.shape != (tokens * topk, hidden):
        return None
    out = _workspace_tensor(
        workspace,
        "route_sum_out",
        (tokens, hidden),
        routed.dtype,
        routed.device,
    )
    block_n = 64
    _moe_route_sum_kernel[(tokens, triton.cdiv(hidden, block_n))](
        routed.contiguous(),
        out,
        tokens=int(tokens),
        hidden=int(hidden),
        topk=int(topk),
        BLOCK_N=block_n,
    )
    return out


def grouped_fp4_moe_fused_compute(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor | None,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor | None,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    route_count: int,
    topk: int,
    block_size_m: int,
    swiglu_limit: float,
    workspace=None,
) -> torch.Tensor | None:
    if (
        hidden_states.ndim != 2
        or weights.numel() != route_count
        or hidden_states.dtype is not torch.bfloat16
        or not hidden_states.is_cuda
        or not weights.is_cuda
        or not sorted_route_ids.is_cuda
        or not expert_ids.is_cuda
        or not num_tokens_post_padded.is_cuda
        or w13_weight.ndim != 4
        or w13_weight.shape[1] != 2
        or w13_weight.dtype is not torch.int8
        or w2_weight.ndim != 3
        or w2_weight.dtype is not torch.int8
        or w13_weight.shape[0] != w2_weight.shape[0]
        or hidden_states.stride(-1) != 1
        or hidden_states.shape[-1] % 2 != 0
    ):
        return None
    if route_count == 0:
        return torch.empty(
            (0, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device
        )
    hidden = hidden_states.shape[-1]
    intermediate = w13_weight.shape[2]
    if w13_weight.shape[-1] * 2 < hidden or w2_weight.shape[-1] * 2 < intermediate:
        return None

    block_n = 64
    block_i = 32
    block_k = 64
    output_tiles = triton.cdiv(hidden, block_n)
    # This fused layout recomputes w1/w3 for each output-hidden tile.  Keep it
    # on shapes where that tradeoff is still plausible; larger DSV4 shapes use
    # the materialized V1 pipeline below.
    if output_tiles > 8:
        return None

    w13_c = w13_weight.contiguous()
    w2_c = w2_weight.contiguous()
    w13_scale_c = (
        w13_scale.float().contiguous()
        if w13_scale is not None
        else w13_c.new_empty((1,), dtype=torch.float32)
    )
    w2_scale_c = (
        w2_scale.float().contiguous()
        if w2_scale is not None
        else w2_c.new_empty((1,), dtype=torch.float32)
    )
    routed = _workspace_tensor(
        workspace,
        "fused_routed",
        (route_count, hidden),
        hidden_states.dtype,
        hidden_states.device,
        zero=True,
    )
    grid = (
        triton.cdiv(sorted_route_ids.numel(), block_size_m),
        output_tiles,
    )
    _grouped_fp4_moe_fused_compute_kernel[grid](
        hidden_states.contiguous(),
        weights.contiguous(),
        w13_c,
        w13_scale_c,
        w2_c,
        w2_scale_c,
        sorted_route_ids.contiguous(),
        expert_ids.contiguous(),
        num_tokens_post_padded.contiguous(),
        routed,
        route_count=route_count,
        topk=topk,
        H=hidden,
        I=intermediate,
        W13_K_BYTES=w13_c.shape[-1],
        W13_SCALE_K=triton.cdiv(hidden, 32),
        W2_K_BYTES=w2_c.shape[-1],
        W2_SCALE_K=triton.cdiv(intermediate, 32),
        HAS_W13_SCALE=w13_scale is not None,
        HAS_W2_SCALE=w2_scale is not None,
        swiglu_limit=float(swiglu_limit),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_I=block_i,
        BLOCK_SIZE_K=block_k,
    )
    return routed


def grouped_fp4_moe(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor | None,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor | None,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    route_count: int,
    topk: int,
    block_size_m: int,
    swiglu_limit: float,
    workspace=None,
) -> torch.Tensor | None:
    if hidden_states.ndim != 2 or weights.numel() != route_count:
        return None
    if hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_cuda:
        return None
    if w13_weight.ndim != 4 or w13_weight.shape[1] != 2 or w2_weight.ndim != 3:
        return None

    if route_count == 0:
        return torch.zeros_like(hidden_states)

    fused_routed = grouped_fp4_moe_fused_compute(
        hidden_states,
        weights,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        sorted_route_ids,
        expert_ids,
        num_tokens_post_padded,
        route_count=route_count,
        topk=topk,
        block_size_m=block_size_m,
        swiglu_limit=swiglu_limit,
        workspace=workspace,
    )
    if fused_routed is not None:
        return _sum_grouped_routes(
            fused_routed,
            tokens=hidden_states.shape[0],
            hidden=hidden_states.shape[1],
            topk=topk,
            workspace=workspace,
        )

    w13 = _grouped_fp4_w13(
        hidden_states,
        w13_weight,
        w13_scale,
        sorted_route_ids,
        expert_ids,
        num_tokens_post_padded,
        route_count=route_count,
        topk=topk,
        block_size_m=block_size_m,
        workspace=workspace,
    )
    if w13 is None:
        return None
    gate, up = w13

    hidden = silu_and_mul_clamp_bf16(
        gate,
        up,
        swiglu_limit=swiglu_limit,
        weights=weights.reshape(-1, 1),
        workspace=workspace,
    )
    if hidden is None:
        activated = silu_and_mul_clamp(
            gate,
            up,
            swiglu_limit=swiglu_limit,
            weights=weights.reshape(-1, 1),
        )
        if activated is None:
            return None
        hidden = activated.to(hidden_states.dtype)

    routed = _grouped_fp4_linear(
        hidden,
        w2_weight,
        w2_scale,
        sorted_route_ids,
        expert_ids,
        num_tokens_post_padded,
        route_count=route_count,
        topk=topk,
        block_size_m=block_size_m,
        slot=None,
        a_rows_are_routes=True,
        workspace=workspace,
    )
    if routed is None:
        return None
    return _sum_grouped_routes(
        routed,
        tokens=hidden_states.shape[0],
        hidden=hidden_states.shape[1],
        topk=topk,
        workspace=workspace,
    )


__all__ = [
    "apply_rotary_tail",
    "build_decode_metadata_indices",
    "build_decode_metadata_indices_component",
    "direct_decode_index_metadata_for_replay",
    "direct_c4_sparse_metadata_for_replay",
    "compress_norm_rope_store_bf16",
    "build_moe_route_plan",
    "grouped_fp4_moe",
    "grouped_fp4_moe_fused_compute",
    "hc_prenorm_head",
    "hc_prenorm_split_pre",
    "hc_post",
    "hc_split_pre",
    "indexer_bf16_logits",
    "indexer_fp8_logits",
    "indexer_fp8_paged_logits",
    "indexer_fp8_paged_quant_store",
    "indexer_fp8_quantize",
    "indexer_fp8_quantize_fold",
    "indexer_fp8_quant_store",
    "remap_indexer_topk_locs",
    "k_norm_rope_cache_bf16",
    "copy_masked_compressed_locs",
    "copy_component_write_locs_for_replay",
    "copy_decode_metadata_for_replay",
    "prep_decode_metadata_in_graph",
    "paged_mqa_attention_bf16",
    "q_norm_rope",
    "fp8_activation_quantize",
    "quantized_linear_fp4",
    "quantized_linear_fp8",
    "q_kv_norm_rope_cache_bf16",
    "rms_norm_bf16",
    "rms_norm_pair_bf16",
    "silu_and_mul_clamp",
    "silu_and_mul_clamp_bf16",
    "store_cache",
    "topk_transform_512",
    "wo_a_grouped_projection_fp8",
]

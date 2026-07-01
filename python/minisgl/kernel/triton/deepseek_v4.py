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
        theta = theta - tl.floor((theta + 3.141592653589793) / 6.283185307179586) * 6.283185307179586
        cos = tl.cos(theta)
        sin = tl.sin(theta)

        a_dim_offsets = tail + pair_offsets * 2
        b_dim_offsets = a_dim_offsets + 1
        a_offsets = row_base + a_dim_offsets
        b_offsets = row_base + b_dim_offsets
        a = tl.load(kv_ptr + a_offsets, mask=tail_mask, other=0.0).to(tl.float32)
        b = tl.load(kv_ptr + b_offsets, mask=tail_mask, other=0.0).to(tl.float32)
        if has_norm:
            a_weight = tl.load(norm_weight_ptr + a_dim_offsets, mask=tail_mask, other=0.0).to(tl.float32)
            b_weight = tl.load(norm_weight_ptr + b_dim_offsets, mask=tail_mask, other=0.0).to(tl.float32)
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

    pre_logits = (
        tl.load(mixes_ptr + mix_base + hc_offsets, mask=hc_mask, other=0.0).to(tl.float32)
        * scale0
        + tl.load(base_ptr + hc_offsets, mask=hc_mask, other=0.0).to(tl.float32)
    )
    pre = tl.sigmoid(pre_logits) + eps
    pre = tl.where(hc_mask, pre, 0.0)

    post_start = hc_mult
    post_logits = (
        tl.load(mixes_ptr + mix_base + post_start + hc_offsets, mask=hc_mask, other=0.0).to(
            tl.float32
        )
        * scale1
        + tl.load(base_ptr + post_start + hc_offsets, mask=hc_mask, other=0.0).to(tl.float32)
    )
    post = 2.0 * tl.sigmoid(post_logits)
    post = tl.where(hc_mask, post, 0.0)

    rows = tl.arange(0, BLOCK_HC)[:, None]
    cols = tl.arange(0, BLOCK_HC)[None, :]
    matrix_mask = (rows < hc_mult) & (cols < hc_mult)
    comb_start = 2 * hc_mult
    comb_offsets = comb_start + rows * hc_mult + cols
    comb_logits = (
        tl.load(mixes_ptr + mix_base + comb_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
        * scale2
        + tl.load(base_ptr + comb_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
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
            weight_offsets = (
                ((expert * 2 + SLOT) * N + offs_n[None, :]) * WEIGHT_K_BYTES
                + (offs_k[:, None] // 2)
            )
            scale_offsets = (
                ((expert * 2 + SLOT) * N + offs_n[None, :]) * SCALE_K
                + (offs_k[:, None] // 32)
            )
        else:
            weight_offsets = (
                (expert * N + offs_n[None, :]) * WEIGHT_K_BYTES
                + (offs_k[:, None] // 2)
            )
            scale_offsets = (
                (expert * N + offs_n[None, :]) * SCALE_K
                + (offs_k[:, None] // 32)
            )
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
        gate_weight_offsets = (
            ((expert * 2) * N + offs_n[None, :]) * WEIGHT_K_BYTES
            + (offs_k[:, None] // 2)
        )
        up_weight_offsets = (
            ((expert * 2 + 1) * N + offs_n[None, :]) * WEIGHT_K_BYTES
            + (offs_k[:, None] // 2)
        )
        gate_packed = tl.load(weight_ptr + gate_weight_offsets, mask=weight_mask, other=0).to(
            tl.int32
        )
        up_packed = tl.load(weight_ptr + up_weight_offsets, mask=weight_mask, other=0).to(
            tl.int32
        )
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
            gate_scale_offsets = (
                ((expert * 2) * N + offs_n[None, :]) * SCALE_K
                + (offs_k[:, None] // 32)
            )
            up_scale_offsets = (
                ((expert * 2 + 1) * N + offs_n[None, :]) * SCALE_K
                + (offs_k[:, None] // 32)
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
        values = tl.load(routed_ptr + route * hidden + offs_n, mask=mask, other=0.0).to(
            tl.float32
        )
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
            gate_weight_offsets = (
                ((expert * 2) * I + offs_i[None, :]) * W13_K_BYTES
                + (offs_k[:, None] // 2)
            )
            up_weight_offsets = (
                ((expert * 2 + 1) * I + offs_i[None, :]) * W13_K_BYTES
                + (offs_k[:, None] // 2)
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
                gate_scale_offsets = (
                    ((expert * 2) * I + offs_i[None, :]) * W13_SCALE_K
                    + (offs_k[:, None] // 32)
                )
                up_scale_offsets = (
                    ((expert * 2 + 1) * I + offs_i[None, :]) * W13_SCALE_K
                    + (offs_k[:, None] // 32)
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
        w2_offsets = (
            (expert * H + offs_n[None, :]) * W2_K_BYTES
            + (offs_i[:, None] // 2)
        )
        w2_packed = tl.load(w2_weight_ptr + w2_offsets, mask=w2_mask, other=0).to(tl.int32)
        w2_nibble = tl.where(
            (offs_i[:, None] & 1) == 0,
            w2_packed & 0x0F,
            (w2_packed >> 4) & 0x0F,
        )
        w2_b = _fp4_e2m1_value(w2_nibble).to(tl.float32)
        if HAS_W2_SCALE:
            w2_scale_offsets = (
                (expert * H + offs_n[None, :]) * W2_SCALE_K
                + (offs_i[:, None] // 32)
            )
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
    gate = torch.zeros((route_count, n), dtype=a.dtype, device=a.device)
    up = torch.zeros_like(gate)
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
    out = torch.zeros((route_count, n), dtype=a.dtype, device=a.device)
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
) -> torch.Tensor | None:
    if routed.ndim != 2 or not routed.is_cuda or routed.shape != (tokens * topk, hidden):
        return None
    out = torch.empty((tokens, hidden), dtype=routed.dtype, device=routed.device)
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
        return torch.empty((0, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device)
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
    routed = torch.empty((route_count, hidden), dtype=hidden_states.dtype, device=hidden_states.device)
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
    )
    if fused_routed is not None:
        return _sum_grouped_routes(
            fused_routed,
            tokens=hidden_states.shape[0],
            hidden=hidden_states.shape[1],
            topk=topk,
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
    )
    if w13 is None:
        return None
    gate, up = w13

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
    )
    if routed is None:
        return None
    return _sum_grouped_routes(
        routed,
        tokens=hidden_states.shape[0],
        hidden=hidden_states.shape[1],
        topk=topk,
    )


__all__ = [
    "apply_rotary_tail",
    "compress_norm_rope_store_bf16",
    "build_moe_route_plan",
    "grouped_fp4_moe",
    "grouped_fp4_moe_fused_compute",
    "hc_post",
    "hc_split_pre",
    "indexer_bf16_logits",
    "k_norm_rope_cache_bf16",
    "copy_masked_compressed_locs",
    "paged_mqa_attention_bf16",
    "q_norm_rope",
    "quantized_linear_fp4",
    "quantized_linear_fp8",
    "q_kv_norm_rope_cache_bf16",
    "rms_norm_bf16",
    "rms_norm_pair_bf16",
    "silu_and_mul_clamp",
    "store_cache",
    "topk_transform_512",
    "wo_a_grouped_projection_fp8",
]

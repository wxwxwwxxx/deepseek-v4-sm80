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
    publish_swa_qat: tl.constexpr,
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
        published = out
        # The model publishes the 448 non-RoPE SWA dimensions only after its
        # block-64 UE8M0/E4M3FN QAT boundary. Quantize the BF16 producer value
        # once and publish the identical final value to local `kv` and the
        # currently owned SWA row. The caller skips its former QAT pass when
        # this producer reports success.
        if publish_swa_qat and kv_dim - rotary_dim == 448:
            qat_source = out.to(tl.bfloat16).to(tl.float32)
            for group in tl.static_range(0, 7):
                group_mask = mask & (offsets >= group * 64) & (offsets < (group + 1) * 64)
                group_values = tl.where(group_mask, qat_source, 0.0)
                absmax = tl.maximum(tl.max(tl.abs(group_values), axis=0), 1e-4)
                qat_scale = tl.exp2(tl.ceil(tl.log2(absmax / 448.0)))
                scaled = tl.clamp(qat_source / qat_scale, -448.0, 448.0)
                encoded = _encode_e4m3fn_sw(scaled)
                dequantized = _fp8_e4m3fn_value(encoded.to(tl.uint32)) * qat_scale
                published = tl.where(group_mask, dequantized, published)
        tl.store(kv_ptr + row_base + offsets, published, mask=mask)
        tl.store(cache_ptr + cache_base + offsets, published, mask=mask & valid_loc)


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
                mask=full_valid & (full_page >= 0) & (full_page < swa_full_to_swa_page_width),
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
    # Keep the replay ABI surface for scheduler diagnostics, but both online
    # compressor producers publish before their attention consumers execute.
    tl.load(materialized_seq_lens_ptr + row)

    tl.store(dst_seq_lens_ptr + row, seq_len)
    tl.store(dst_swa_topk_lengths_ptr + row, tl.minimum(seq_len, window_size))

    # Both graph-internal producers execute earlier in the captured graph and
    # publish the current boundary before their consumers run.
    c4_len = seq_len // 4
    c4_len_clamp1 = tl.maximum(c4_len, 1)
    c4_sparse_len = tl.minimum(tl.maximum(c4_len, 0), index_topk)
    tl.store(dst_c4_topk_lengths_raw_ptr + row, c4_len)
    tl.store(dst_c4_topk_lengths_clamp1_ptr + row, c4_len_clamp1)
    tl.store(dst_c4_sparse_topk_lengths_ptr + row, c4_sparse_len)

    c128_len = seq_len // 128
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
            mask=full_valid & (full_page >= 0) & (full_page < swa_full_to_swa_page_width),
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
        mask=c128_valid & (c128_logical_page >= 0) & (c128_logical_page < c128_page_table_width),
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
        c4_indexer_page_table_ptr + row * c4_indexer_page_table_width + c4_write_logical_page,
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

    # Keep raw_out_loc live in the captured dependency set for reference parity.
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
    ).to(tl.float32)
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
    ).to(tl.float32)
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
        component_page_table_ptr + rows * component_table_width + component_logical_page,
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
        active & (raw >= 0) & (full_logical_page >= 0) & (full_logical_page < full_table_width)
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
def _mask_moe_routes_live_rows_kernel(
    weights_ptr,
    indices_ptr,
    num_token_non_padded_ptr,
    route_count,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < route_count
    live_rows = tl.load(num_token_non_padded_ptr)
    padded = (offsets // topk) >= live_rows
    write_mask = mask & padded
    tl.store(indices_ptr + offsets, -1, mask=write_mask)
    tl.store(weights_ptr + offsets, 0.0, mask=write_mask)


@triton.jit
def _zero_moe_padded_rows_kernel(
    output_ptr,
    num_token_non_padded_ptr,
    n_elements,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    live_rows = tl.load(num_token_non_padded_ptr)
    padded = (offsets // hidden) >= live_rows
    tl.store(output_ptr + offsets, 0.0, mask=mask & padded)


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
    publish_swa_qat: bool = False,
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
        publish_swa_qat=bool(publish_swa_qat),
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
    if (
        direct_swa
        and swa_independent
        and (
            swa_full_to_swa_page is None
            or swa_full_to_swa_page.ndim != 1
            or swa_dummy_token_start < 0
            or swa_dummy_page < 0
        )
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


@triton.jit
def _c4_online_pool_kernel(
    projected_ptr,
    state_ptr,
    ape_ptr,
    positions_ptr,
    table_indices_ptr,
    ctx_page_table_ptr,
    state_page_mapping_ptr,
    output_ptr,
    rows: tl.constexpr,
    head_dim: tl.constexpr,
    projected_stride0: tl.constexpr,
    state_stride0: tl.constexpr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    state_page_mapping_width: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Official DSV4 C4 overlap reduction over current rows plus paged carry state.

    Each program owns one output row and a head-dimension tile.  Current-call
    values are read directly from ``projected_ptr`` so long prefills cannot
    overwrite their eight-slot carry ring before all completed groups reduce.
    Values before the current chunk are resolved through the scheduler-owned
    full-token table and the existing full-page -> state-page mapping.
    """

    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    pos = tl.load(positions_ptr + row)
    table_idx = tl.load(table_indices_ptr + row)
    boundary = ((pos + 1) % 4) == 0

    score_0 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    score_1 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    score_2 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    score_3 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    score_4 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    score_5 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    score_6 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    score_7 = tl.full((BLOCK_D,), float("-inf"), tl.float32)
    kv_0 = tl.zeros((BLOCK_D,), tl.float32)
    kv_1 = tl.zeros((BLOCK_D,), tl.float32)
    kv_2 = tl.zeros((BLOCK_D,), tl.float32)
    kv_3 = tl.zeros((BLOCK_D,), tl.float32)
    kv_4 = tl.zeros((BLOCK_D,), tl.float32)
    kv_5 = tl.zeros((BLOCK_D,), tl.float32)
    kv_6 = tl.zeros((BLOCK_D,), tl.float32)
    kv_7 = tl.zeros((BLOCK_D,), tl.float32)

    # The eight source positions are p-7..p.  The older four use the left
    # overlap half; the newer four use the right/current half.
    for source_slot in tl.static_range(0, 8):
        delta = 7 - source_slot
        logical_pos = pos - delta
        candidate_row = row - delta
        candidate_in_range = candidate_row >= 0
        candidate_table = tl.load(
            table_indices_ptr + candidate_row,
            mask=candidate_in_range,
            other=-1,
        )
        candidate_pos = tl.load(
            positions_ptr + candidate_row,
            mask=candidate_in_range,
            other=-1,
        )
        current = (
            candidate_in_range
            & (logical_pos >= 0)
            & (candidate_table == table_idx)
            & (candidate_pos == logical_pos)
        )

        full_loc = tl.load(
            ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + logical_pos,
            mask=(logical_pos >= 0) & (logical_pos < ctx_page_table_width),
            other=-1,
        )
        full_page = full_loc // page_size
        state_page = tl.load(
            state_page_mapping_ptr + full_page,
            mask=(full_loc >= 0)
            & (full_page >= 0)
            & (full_page < state_page_mapping_width),
            other=-1,
        )
        state_loc = state_page * 8 + (full_loc % 8)
        persistent = (logical_pos >= 0) & (full_loc >= 0) & (state_page >= 0)

        use_right = source_slot >= 4
        kv_offset = head_dim if use_right else 0
        score_offset = 3 * head_dim if use_right else 2 * head_dim
        current_kv = tl.load(
            projected_ptr + candidate_row * projected_stride0 + kv_offset + d,
            mask=current & d_mask,
            other=0.0,
        ).to(tl.float32)
        current_score = tl.load(
            projected_ptr + candidate_row * projected_stride0 + score_offset + d,
            mask=current & d_mask,
            other=float("-inf"),
        ).to(tl.float32)
        state_kv = tl.load(
            state_ptr + state_loc * state_stride0 + kv_offset + d,
            mask=(~current) & persistent & d_mask,
            other=0.0,
        ).to(tl.float32)
        state_score = tl.load(
            state_ptr + state_loc * state_stride0 + score_offset + d,
            mask=(~current) & persistent & d_mask,
            other=float("-inf"),
        ).to(tl.float32)
        source_kv = tl.where(current, current_kv, state_kv)
        source_score = tl.where(current, current_score, state_score)
        ape_row = source_slot if source_slot < 4 else source_slot - 4
        ape_col = d if source_slot < 4 else head_dim + d
        source_score += tl.load(
            ape_ptr + ape_row * (2 * head_dim) + ape_col,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        source_score = tl.where(current | persistent, source_score, float("-inf"))

        if source_slot == 0:
            kv_0, score_0 = source_kv, source_score
        elif source_slot == 1:
            kv_1, score_1 = source_kv, source_score
        elif source_slot == 2:
            kv_2, score_2 = source_kv, source_score
        elif source_slot == 3:
            kv_3, score_3 = source_kv, source_score
        elif source_slot == 4:
            kv_4, score_4 = source_kv, source_score
        elif source_slot == 5:
            kv_5, score_5 = source_kv, source_score
        elif source_slot == 6:
            kv_6, score_6 = source_kv, source_score
        else:
            kv_7, score_7 = source_kv, source_score

    score_max = tl.maximum(
        tl.maximum(tl.maximum(score_0, score_1), tl.maximum(score_2, score_3)),
        tl.maximum(tl.maximum(score_4, score_5), tl.maximum(score_6, score_7)),
    )
    w0 = tl.exp(score_0 - score_max)
    w1 = tl.exp(score_1 - score_max)
    w2 = tl.exp(score_2 - score_max)
    w3 = tl.exp(score_3 - score_max)
    w4 = tl.exp(score_4 - score_max)
    w5 = tl.exp(score_5 - score_max)
    w6 = tl.exp(score_6 - score_max)
    w7 = tl.exp(score_7 - score_max)
    denom = w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7
    pooled = (
        kv_0 * w0
        + kv_1 * w1
        + kv_2 * w2
        + kv_3 * w3
        + kv_4 * w4
        + kv_5 * w5
        + kv_6 * w6
        + kv_7 * w7
    ) / denom
    tl.store(
        output_ptr + row * head_dim + d,
        tl.where(boundary, pooled, 0.0),
        mask=d_mask,
    )


@triton.jit
def _c4_online_state_store_kernel(
    projected_ptr,
    raw_out_loc_ptr,
    positions_ptr,
    table_indices_ptr,
    state_page_mapping_ptr,
    state_ptr,
    rows: tl.constexpr,
    projected_stride0: tl.constexpr,
    state_stride0: tl.constexpr,
    state_page_mapping_width: tl.constexpr,
    page_size: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    full_loc = tl.load(raw_out_loc_ptr + row)
    full_page = full_loc // page_size
    state_page = tl.load(
        state_page_mapping_ptr + full_page,
        mask=(full_loc >= 0)
        & (full_page >= 0)
        & (full_page < state_page_mapping_width),
        other=-1,
    )
    state_loc = state_page * 8 + (full_loc % 8)

    # Parallel prefill rows separated by eight tokens can alias the ring.  Only
    # the last current-call writer for a state location is allowed to persist.
    later_row = row + 8
    later_in_range = later_row < rows
    later_full_loc = tl.load(
        raw_out_loc_ptr + later_row,
        mask=later_in_range,
        other=-1,
    )
    later_full_page = later_full_loc // page_size
    later_state_page = tl.load(
        state_page_mapping_ptr + later_full_page,
        mask=later_in_range
        & (later_full_loc >= 0)
        & (later_full_page >= 0)
        & (later_full_page < state_page_mapping_width),
        other=-1,
    )
    later_state_loc = later_state_page * 8 + (later_full_loc % 8)
    pos = tl.load(positions_ptr + row)
    table_idx = tl.load(table_indices_ptr + row)
    later_pos = tl.load(positions_ptr + later_row, mask=later_in_range, other=-1)
    later_table = tl.load(table_indices_ptr + later_row, mask=later_in_range, other=-1)
    shadowed = (
        later_in_range
        & (later_table == table_idx)
        & (later_pos == pos + 8)
        & (later_state_loc == state_loc)
    )
    valid = (state_page >= 0) & (~shadowed) & (offsets < width)
    value = tl.load(
        projected_ptr + row * projected_stride0 + offsets,
        mask=valid,
        other=0.0,
    )
    tl.store(state_ptr + state_loc * state_stride0 + offsets, value, mask=valid)


def c4_online_pool_and_update(
    projected: torch.Tensor,
    state: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    table_indices: torch.Tensor,
    raw_out_loc: torch.Tensor,
    ctx_page_table: torch.Tensor,
    state_page_mapping: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor | None:
    """SM80 bridge for SGLang-style fixed-row C4 production.

    ``projected`` and ``state`` contain the official overlap layout
    ``[kv_left, kv_right, score_left, score_right]`` in FP32.  The returned
    tensor has one row per input token; non-boundary rows are zero and are
    suppressed by the fixed publication-location mask.
    """

    tensors = (
        projected,
        state,
        ape,
        positions,
        table_indices,
        raw_out_loc,
        ctx_page_table,
        state_page_mapping,
    )
    if (
        projected.ndim != 2
        or state.ndim != 2
        or projected.shape[1] % 4
        or state.shape[1] != projected.shape[1]
        or ape.shape != (4, projected.shape[1] // 2)
        or projected.dtype is not torch.float32
        or state.dtype is not torch.float32
        or ape.dtype is not torch.float32
        or page_size <= 0
        or page_size & (page_size - 1)
        or not all(t.is_cuda and t.is_contiguous() for t in tensors)
        or any(t.dtype not in (torch.int32, torch.int64) for t in tensors[3:])
    ):
        return None
    rows = int(projected.shape[0])
    if rows == 0:
        return projected.new_empty((0, projected.shape[1] // 4))
    if (
        positions.numel() != rows
        or table_indices.numel() != rows
        or raw_out_loc.numel() != rows
        or ctx_page_table.ndim != 2
        or state_page_mapping.ndim != 1
    ):
        return None

    head_dim = int(projected.shape[1] // 4)
    # Official Compressor.forward casts the pooled FP32 accumulator to the
    # activation dtype before RMSNorm/publication.
    output = torch.empty((rows, head_dim), dtype=torch.bfloat16, device=projected.device)
    block_d = min(triton.next_power_of_2(head_dim), 256)
    _c4_online_pool_kernel[(rows, triton.cdiv(head_dim, block_d))](
        projected,
        state,
        ape,
        positions,
        table_indices,
        ctx_page_table,
        state_page_mapping,
        output,
        rows=rows,
        head_dim=head_dim,
        projected_stride0=projected.stride(0),
        state_stride0=state.stride(0),
        ctx_page_table_stride0=ctx_page_table.stride(0),
        ctx_page_table_width=ctx_page_table.shape[1],
        state_page_mapping_width=state_page_mapping.numel(),
        page_size=int(page_size),
        BLOCK_D=block_d,
        num_warps=4,
    )
    width = int(projected.shape[1])
    block = 256
    _c4_online_state_store_kernel[(rows, triton.cdiv(width, block))](
        projected,
        raw_out_loc,
        positions,
        table_indices,
        state_page_mapping,
        state,
        rows=rows,
        projected_stride0=projected.stride(0),
        state_stride0=state.stride(0),
        state_page_mapping_width=state_page_mapping.numel(),
        page_size=int(page_size),
        width=width,
        BLOCK=block,
        num_warps=4,
    )
    return output


@triton.jit
def _c128_online_pool_kernel(
    projected_ptr,
    state_ptr,
    ape_ptr,
    positions_ptr,
    table_indices_ptr,
    ctx_page_table_ptr,
    state_page_mapping_ptr,
    output_ptr,
    rows: tl.constexpr,
    head_dim: tl.constexpr,
    projected_stride0: tl.constexpr,
    state_stride0: tl.constexpr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    state_page_mapping_width: tl.constexpr,
    page_size: tl.constexpr,
    RATIO: tl.constexpr,
) -> None:
    """Reduce one non-overlap C128 group from current rows plus paged carry."""

    row = tl.program_id(0)
    d = tl.program_id(1)
    pos = tl.load(positions_ptr + row)
    boundary = ((pos + 1) % RATIO) == 0
    if not boundary:
        tl.store(output_ptr + row * head_dim + d, 0.0, mask=d < head_dim)
        return

    source_slot = tl.arange(0, RATIO)
    delta = (RATIO - 1) - source_slot
    logical_pos = pos - delta
    candidate_row = row - delta
    candidate_in_range = (candidate_row >= 0) & (candidate_row < rows)
    table_idx = tl.load(table_indices_ptr + row)
    candidate_table = tl.load(
        table_indices_ptr + candidate_row,
        mask=candidate_in_range,
        other=-1,
    )
    candidate_pos = tl.load(
        positions_ptr + candidate_row,
        mask=candidate_in_range,
        other=-1,
    )
    current = (
        candidate_in_range
        & (logical_pos >= 0)
        & (candidate_table == table_idx)
        & (candidate_pos == logical_pos)
    )

    full_loc = tl.load(
        ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + logical_pos,
        mask=(logical_pos >= 0) & (logical_pos < ctx_page_table_width),
        other=-1,
    )
    full_page = full_loc // page_size
    state_page = tl.load(
        state_page_mapping_ptr + full_page,
        mask=(full_loc >= 0)
        & (full_page >= 0)
        & (full_page < state_page_mapping_width),
        other=-1,
    )
    state_loc = state_page * RATIO + (full_loc % RATIO)
    persistent = (logical_pos >= 0) & (full_loc >= 0) & (state_page >= 0)
    d_mask = d < head_dim

    current_kv = tl.load(
        projected_ptr + candidate_row * projected_stride0 + d,
        mask=current & d_mask,
        other=0.0,
    ).to(tl.float32)
    current_score = tl.load(
        projected_ptr + candidate_row * projected_stride0 + head_dim + d,
        mask=current & d_mask,
        other=float("-inf"),
    ).to(tl.float32)
    current_score += tl.load(
        ape_ptr + source_slot * head_dim + d,
        mask=current & d_mask,
        other=0.0,
    ).to(tl.float32)
    state_kv = tl.load(
        state_ptr + state_loc * state_stride0 + d,
        mask=(~current) & persistent & d_mask,
        other=0.0,
    ).to(tl.float32)
    state_score = tl.load(
        state_ptr + state_loc * state_stride0 + head_dim + d,
        mask=(~current) & persistent & d_mask,
        other=float("-inf"),
    ).to(tl.float32)
    valid = current | persistent
    kv = tl.where(current, current_kv, state_kv)
    score = tl.where(current, current_score, state_score)
    score = tl.where(valid, score, float("-inf"))
    score_max = tl.max(score, axis=0)
    weight = tl.exp(score - score_max)
    denom = tl.sum(weight, axis=0)
    pooled = tl.sum(kv * weight, axis=0) / denom
    tl.store(output_ptr + row * head_dim + d, pooled, mask=d_mask)


@triton.jit
def _c128_online_state_store_kernel(
    projected_ptr,
    ape_ptr,
    raw_out_loc_ptr,
    positions_ptr,
    table_indices_ptr,
    state_page_mapping_ptr,
    state_ptr,
    rows: tl.constexpr,
    head_dim: tl.constexpr,
    projected_stride0: tl.constexpr,
    state_stride0: tl.constexpr,
    state_page_mapping_width: tl.constexpr,
    page_size: tl.constexpr,
    RATIO: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    full_loc = tl.load(raw_out_loc_ptr + row)
    full_page = full_loc // page_size
    state_page = tl.load(
        state_page_mapping_ptr + full_page,
        mask=(full_loc >= 0)
        & (full_page >= 0)
        & (full_page < state_page_mapping_width),
        other=-1,
    )
    state_loc = state_page * RATIO + (full_loc % RATIO)

    # A long prefill may contain more than one group.  Preserve only the last
    # current-call writer to each physical ring slot after every boundary row
    # has consumed current-call projections directly.
    later_row = row + RATIO
    later_in_range = later_row < rows
    later_full_loc = tl.load(
        raw_out_loc_ptr + later_row,
        mask=later_in_range,
        other=-1,
    )
    later_full_page = later_full_loc // page_size
    later_state_page = tl.load(
        state_page_mapping_ptr + later_full_page,
        mask=later_in_range
        & (later_full_loc >= 0)
        & (later_full_page >= 0)
        & (later_full_page < state_page_mapping_width),
        other=-1,
    )
    later_state_loc = later_state_page * RATIO + (later_full_loc % RATIO)
    pos = tl.load(positions_ptr + row)
    table_idx = tl.load(table_indices_ptr + row)
    later_pos = tl.load(positions_ptr + later_row, mask=later_in_range, other=-1)
    later_table = tl.load(
        table_indices_ptr + later_row,
        mask=later_in_range,
        other=-1,
    )
    shadowed = (
        later_in_range
        & (later_table == table_idx)
        & (later_pos == pos + RATIO)
        & (later_state_loc == state_loc)
    )
    valid = (state_page >= 0) & (~shadowed) & (offsets < width)
    value = tl.load(
        projected_ptr + row * projected_stride0 + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    score_half = offsets >= head_dim
    score_offset = tl.maximum(offsets - head_dim, 0)
    ape_value = tl.load(
        ape_ptr + (pos % RATIO) * head_dim + score_offset,
        mask=valid & score_half,
        other=0.0,
    ).to(tl.float32)
    value += tl.where(score_half, ape_value, 0.0)
    tl.store(state_ptr + state_loc * state_stride0 + offsets, value, mask=valid)


def c128_online_pool_and_update(
    projected: torch.Tensor,
    state: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    table_indices: torch.Tensor,
    raw_out_loc: torch.Tensor,
    ctx_page_table: torch.Tensor,
    state_page_mapping: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor | None:
    """SM80 fixed-row producer for the official C128 non-overlap contract."""

    tensors = (
        projected,
        state,
        ape,
        positions,
        table_indices,
        raw_out_loc,
        ctx_page_table,
        state_page_mapping,
    )
    if (
        projected.ndim != 2
        or state.ndim != 2
        or projected.shape[1] % 2
        or state.shape[1] != projected.shape[1]
        or ape.shape != (128, projected.shape[1] // 2)
        or projected.dtype is not torch.float32
        or state.dtype is not torch.float32
        or ape.dtype is not torch.float32
        or page_size <= 0
        or page_size & (page_size - 1)
        or page_size % 128
        or not all(t.is_cuda and t.is_contiguous() for t in tensors)
        or any(t.dtype not in (torch.int32, torch.int64) for t in tensors[3:])
    ):
        return None
    rows = int(projected.shape[0])
    if rows == 0:
        return projected.new_empty((0, projected.shape[1] // 2), dtype=torch.bfloat16)
    if (
        positions.numel() != rows
        or table_indices.numel() != rows
        or raw_out_loc.numel() != rows
        or ctx_page_table.ndim != 2
        or state_page_mapping.ndim != 1
    ):
        return None

    head_dim = int(projected.shape[1] // 2)
    output = torch.empty((rows, head_dim), dtype=torch.bfloat16, device=projected.device)
    _c128_online_pool_kernel[(rows, head_dim)](
        projected,
        state,
        ape,
        positions,
        table_indices,
        ctx_page_table,
        state_page_mapping,
        output,
        rows=rows,
        head_dim=head_dim,
        projected_stride0=projected.stride(0),
        state_stride0=state.stride(0),
        ctx_page_table_stride0=ctx_page_table.stride(0),
        ctx_page_table_width=ctx_page_table.shape[1],
        state_page_mapping_width=state_page_mapping.numel(),
        page_size=int(page_size),
        RATIO=128,
        num_warps=4,
    )
    width = int(projected.shape[1])
    block = 256
    _c128_online_state_store_kernel[(rows, triton.cdiv(width, block))](
        projected,
        ape,
        raw_out_loc,
        positions,
        table_indices,
        state_page_mapping,
        state,
        rows=rows,
        head_dim=head_dim,
        projected_stride0=projected.stride(0),
        state_stride0=state.stride(0),
        state_page_mapping_width=state_page_mapping.numel(),
        page_size=int(page_size),
        RATIO=128,
        width=width,
        BLOCK=block,
        num_warps=4,
    )
    return output


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
    out: torch.Tensor | None = None,
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
    if out is None:
        out_flat = torch.empty_like(flat)
    else:
        if (
            out.device != x.device
            or out.dtype != x.dtype
            or out.shape != x.shape
            or not out.is_contiguous()
        ):
            raise ValueError(
                "fp8_activation_quantize out must be contiguous with the same "
                f"shape/device/dtype as x; x={tuple(x.shape)}/{x.device}/{x.dtype}, "
                f"out={tuple(out.shape)}/{out.device}/{out.dtype}."
            )
        out_flat = out.view_as(flat)
    block = triton.next_power_of_2(block_size)
    grid = (flat.shape[0], flat.shape[1] // block_size)
    _fp8_activation_quantize_kernel[grid](
        flat,
        out_flat,
        cols=flat.shape[1],
        block_size=int(block_size),
        BLOCK=block,
        num_warps=4,
    )
    return out_flat.view_as(x_c).reshape_as(x)


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


def mask_moe_routes_live_rows(
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_token_non_padded: torch.Tensor,
) -> bool:
    """In-place graph-safe route masking before histogram/sort/align."""
    if (
        weights.ndim != 2
        or indices.shape != weights.shape
        or not weights.is_cuda
        or not indices.is_cuda
        or not num_token_non_padded.is_cuda
        or num_token_non_padded.numel() != 1
        or num_token_non_padded.dtype != torch.int32
        or weights.device != indices.device
        or weights.device != num_token_non_padded.device
        or not weights.is_contiguous()
        or not indices.is_contiguous()
    ):
        return False
    route_count = weights.numel()
    if route_count == 0:
        return True
    block = 256
    _mask_moe_routes_live_rows_kernel[(triton.cdiv(route_count, block),)](
        weights,
        indices,
        num_token_non_padded,
        route_count,
        topk=int(weights.shape[1]),
        BLOCK=block,
    )
    return True


def zero_moe_padded_rows(
    output: torch.Tensor,
    num_token_non_padded: torch.Tensor,
) -> bool:
    """Zero only excluded MoE output rows, leaving live rows untouched."""
    if (
        output.ndim != 2
        or not output.is_cuda
        or not output.is_contiguous()
        or not num_token_non_padded.is_cuda
        or num_token_non_padded.numel() != 1
        or num_token_non_padded.dtype != torch.int32
        or output.device != num_token_non_padded.device
    ):
        return False
    n_elements = output.numel()
    if n_elements == 0:
        return True
    block = 256
    _zero_moe_padded_rows_kernel[(triton.cdiv(n_elements, block),)](
        output,
        num_token_non_padded,
        n_elements,
        hidden=int(output.shape[1]),
        BLOCK=block,
    )
    return True


__all__ = [
    "apply_rotary_tail",
    "c4_online_pool_and_update",
    "c128_online_pool_and_update",
    "direct_decode_index_metadata_for_replay",
    "compress_norm_rope_store_bf16",
    "build_moe_route_plan",
    "mask_moe_routes_live_rows",
    "zero_moe_padded_rows",
    "hc_prenorm_head",
    "hc_prenorm_split_pre",
    "hc_post",
    "hc_split_pre",
    "indexer_fp8_paged_logits",
    "indexer_fp8_paged_quant_store",
    "indexer_fp8_quantize_fold",
    "remap_indexer_topk_locs",
    "copy_masked_compressed_locs",
    "copy_component_write_locs_for_replay",
    "copy_decode_metadata_for_replay",
    "prep_decode_metadata_in_graph",
    "paged_mqa_attention_bf16",
    "fp8_activation_quantize",
    "q_kv_norm_rope_cache_bf16",
    "rms_norm_bf16",
    "silu_and_mul_clamp",
    "silu_and_mul_clamp_bf16",
]

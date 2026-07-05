from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

try:
    from minisgl.kernel import deepseek_v4 as dsv4_kernel
except Exception:  # pragma: no cover - this harness can still run standalone.
    dsv4_kernel = None


HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_BLOCK = 64
NOPE_BLOCKS = NOPE_DIM // QUANT_BLOCK
SCALE_DIM = NOPE_BLOCKS + 1
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2
TOKEN_BYTES_UNPADDED = TOKEN_DATA_BYTES + SCALE_DIM
PAGE_STRIDE_ALIGNMENT = TOKEN_DATA_BYTES
FP8_MAX = 448.0
UE8M0_BIAS = 127


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def packed_page_bytes(page_size: int) -> int:
    return align_up(page_size * TOKEN_BYTES_UNPADDED, PAGE_STRIDE_ALIGNMENT)


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
def _decode_e4m3fn_sw(u):
    u32 = u.to(tl.uint32)
    sign = tl.where((u32 & 0x80) != 0, -1.0, 1.0)
    exp_bits = ((u32 >> 3) & 0x0F).to(tl.int32)
    mant = (u32 & 0x07).to(tl.float32)
    subnormal = (mant / 8.0) * 0.015625
    normal = (1.0 + mant / 8.0) * tl.exp2(exp_bits.to(tl.float32) - 7.0)
    return sign * tl.where(exp_bits == 0, subnormal, normal)


@triton.jit
def _bf16_store_kernel(
    rows_ptr,
    loc_ptr,
    cache_ptr,
    n_rows,
    capacity: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim
    loc = tl.load(loc_ptr + row).to(tl.int64)
    valid = (row < n_rows) & (loc >= 0) & (loc < capacity)
    values = tl.load(rows_ptr + row * dim + offsets, mask=mask, other=0.0)
    tl.store(cache_ptr + loc * dim + offsets, values, mask=mask & valid)


@triton.jit
def _bf16_gather_kernel(
    cache_ptr,
    indices_ptr,
    out_ptr,
    n_indices,
    capacity: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < dim
    idx = tl.load(indices_ptr + row).to(tl.int64)
    valid = (row < n_indices) & (idx >= 0) & (idx < capacity)
    safe_idx = tl.where(valid, idx, 0)
    values = tl.load(cache_ptr + safe_idx * dim + offsets, mask=mask, other=0.0)
    values = tl.where(valid, values, tl.zeros_like(values))
    tl.store(out_ptr + row * dim + offsets, values, mask=mask)


@triton.jit
def _packed_mla_store_kernel(
    rows_ptr,
    loc_ptr,
    cache_u8_ptr,
    cache_bf16_ptr,
    n_rows,
    capacity: tl.constexpr,
    page_size: tl.constexpr,
    page_bytes: tl.constexpr,
    page_bytes_bf16: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    nope_blocks: tl.constexpr,
    quant_block: tl.constexpr,
    scale_dim: tl.constexpr,
    token_data_bytes: tl.constexpr,
    fp8_max: tl.constexpr,
    ue8m0_bias: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    loc = tl.load(loc_ptr + row).to(tl.int64)
    valid = (row < n_rows) & (loc >= 0) & (loc < capacity)
    page = loc // page_size
    slot = loc - page * page_size

    if tile == nope_blocks:
        rope_vals = tl.load(
            rows_ptr + row * head_dim + nope_dim + offsets,
            mask=offsets < rope_dim,
            other=0.0,
        )
        rope_offset = (
            page * page_bytes_bf16
            + (slot * token_data_bytes + nope_dim) // 2
            + offsets
        )
        tl.store(cache_bf16_ptr + rope_offset, rope_vals, mask=(offsets < rope_dim) & valid)
        pad_offset = page * page_bytes + page_size * token_data_bytes + slot * scale_dim + tile
        tl.store(cache_u8_ptr + pad_offset, tl.zeros((), dtype=tl.uint8), mask=valid)
    else:
        x = tl.load(
            rows_ptr + row * head_dim + tile * quant_block + offsets,
            mask=offsets < quant_block,
            other=0.0,
        ).to(tl.float32)
        x = x.to(tl.bfloat16).to(tl.float32)
        absmax = tl.max(tl.abs(x), axis=0)
        scale = tl.maximum(absmax, 1.0e-8) / fp8_max
        exponent = tl.ceil(tl.log2(scale))
        inv_scale = tl.exp2(-exponent)
        encoded = _encode_e4m3fn_sw(tl.clamp(x * inv_scale, -fp8_max, fp8_max))
        value_offset = page * page_bytes + slot * token_data_bytes + tile * quant_block + offsets
        tl.store(cache_u8_ptr + value_offset, encoded, mask=(offsets < quant_block) & valid)
        scale_u8 = (exponent.to(tl.int32) + ue8m0_bias).to(tl.uint8)
        scale_offset = page * page_bytes + page_size * token_data_bytes + slot * scale_dim + tile
        tl.store(cache_u8_ptr + scale_offset, scale_u8, mask=valid)


@triton.jit
def _packed_mla_gather_kernel(
    cache_u8_ptr,
    cache_bf16_ptr,
    indices_ptr,
    out_ptr,
    n_indices,
    capacity: tl.constexpr,
    page_size: tl.constexpr,
    page_bytes: tl.constexpr,
    page_bytes_bf16: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    nope_blocks: tl.constexpr,
    quant_block: tl.constexpr,
    scale_dim: tl.constexpr,
    token_data_bytes: tl.constexpr,
    ue8m0_bias: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    idx = tl.load(indices_ptr + row).to(tl.int64)
    valid = (row < n_indices) & (idx >= 0) & (idx < capacity)
    safe_idx = tl.where(valid, idx, 0)
    page = safe_idx // page_size
    slot = safe_idx - page * page_size

    for qb in tl.static_range(nope_blocks):
        offsets = qb * quant_block + tl.arange(0, BLOCK)
        value_offset = page * page_bytes + slot * token_data_bytes + offsets
        x_u8 = tl.load(cache_u8_ptr + value_offset, mask=offsets < nope_dim, other=0)
        x = _decode_e4m3fn_sw(x_u8)
        scale_offset = page * page_bytes + page_size * token_data_bytes + slot * scale_dim + qb
        scale_u8 = tl.load(cache_u8_ptr + scale_offset)
        scale = tl.exp2(scale_u8.to(tl.float32) - ue8m0_bias)
        out_vals = tl.where(valid, x * scale, tl.zeros_like(x))
        tl.store(out_ptr + row * head_dim + offsets, out_vals.to(tl.bfloat16), mask=offsets < nope_dim)

    rope_offsets = tl.arange(0, BLOCK)
    rope_src = (
        page * page_bytes_bf16
        + (slot * token_data_bytes + nope_dim) // 2
        + rope_offsets
    )
    rope = tl.load(cache_bf16_ptr + rope_src, mask=rope_offsets < rope_dim, other=0.0)
    rope = tl.where(valid, rope, tl.zeros_like(rope))
    tl.store(out_ptr + row * head_dim + nope_dim + rope_offsets, rope, mask=rope_offsets < rope_dim)


def make_packed_cache(num_pages: int, page_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((num_pages, packed_page_bytes(page_size)), dtype=torch.uint8, device=device)


def bf16_store(rows: torch.Tensor, locs: torch.Tensor, cache: torch.Tensor) -> None:
    if rows.numel() == 0:
        return
    _bf16_store_kernel[(rows.shape[0],)](
        rows,
        locs,
        cache,
        rows.shape[0],
        capacity=cache.shape[0],
        dim=HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        num_warps=8,
    )


def bf16_gather(cache: torch.Tensor, indices: torch.Tensor, out: torch.Tensor) -> None:
    if indices.numel() == 0:
        return
    _bf16_gather_kernel[(indices.numel(),)](
        cache,
        indices,
        out,
        indices.numel(),
        capacity=cache.shape[0],
        dim=HEAD_DIM,
        BLOCK_D=triton.next_power_of_2(HEAD_DIM),
        num_warps=8,
    )


def packed_mla_store(rows: torch.Tensor, locs: torch.Tensor, cache: torch.Tensor, page_size: int) -> None:
    if rows.numel() == 0:
        return
    page_bytes = int(cache.shape[1])
    _packed_mla_store_kernel[(rows.shape[0], NOPE_BLOCKS + 1)](
        rows,
        locs,
        cache,
        cache.view(torch.bfloat16),
        rows.shape[0],
        capacity=cache.shape[0] * page_size,
        page_size=page_size,
        page_bytes=page_bytes,
        page_bytes_bf16=page_bytes // 2,
        head_dim=HEAD_DIM,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        nope_blocks=NOPE_BLOCKS,
        quant_block=QUANT_BLOCK,
        scale_dim=SCALE_DIM,
        token_data_bytes=TOKEN_DATA_BYTES,
        fp8_max=FP8_MAX,
        ue8m0_bias=UE8M0_BIAS,
        BLOCK=QUANT_BLOCK,
        num_warps=2,
    )


def packed_mla_gather(cache: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, page_size: int) -> None:
    if indices.numel() == 0:
        return
    page_bytes = int(cache.shape[1])
    _packed_mla_gather_kernel[(indices.numel(),)](
        cache,
        cache.view(torch.bfloat16),
        indices,
        out,
        indices.numel(),
        capacity=cache.shape[0] * page_size,
        page_size=page_size,
        page_bytes=page_bytes,
        page_bytes_bf16=page_bytes // 2,
        head_dim=HEAD_DIM,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        nope_blocks=NOPE_BLOCKS,
        quant_block=QUANT_BLOCK,
        scale_dim=SCALE_DIM,
        token_data_bytes=TOKEN_DATA_BYTES,
        ue8m0_bias=UE8M0_BIAS,
        BLOCK=QUANT_BLOCK,
        num_warps=2,
    )


def pack_reference(rows: torch.Tensor, locs: torch.Tensor, cache: torch.Tensor, page_size: int) -> None:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        raise RuntimeError("torch.float8_e4m3fn is required for byte parity reference")
    if rows.numel() == 0:
        return
    page_bytes = cache.shape[1]
    valid = (locs >= 0) & (locs < cache.shape[0] * page_size)
    if not bool(torch.any(valid)):
        return
    rows_v = rows[valid].contiguous().to(torch.bfloat16)
    locs_v = locs[valid].to(torch.long)
    pages = locs_v // page_size
    slots = locs_v - pages * page_size
    flat = cache.view(cache.shape[0], page_bytes)

    nope = rows_v[:, :NOPE_DIM].to(torch.float32).view(-1, NOPE_BLOCKS, QUANT_BLOCK)
    absmax = nope.abs().amax(dim=-1).clamp_min(1.0e-8)
    exponents = torch.ceil(torch.log2(absmax / FP8_MAX))
    scale = torch.pow(2.0, exponents).unsqueeze(-1)
    encoded = (nope / scale).clamp(-FP8_MAX, FP8_MAX).to(fp8).view(torch.uint8)
    encoded = encoded.view(rows_v.shape[0], NOPE_DIM)
    scales = (exponents.to(torch.int32) + UE8M0_BIAS).clamp(0, 255).to(torch.uint8)
    scales = torch.cat(
        [scales, torch.zeros((scales.shape[0], 1), dtype=torch.uint8, device=scales.device)],
        dim=-1,
    )
    rope_bytes = rows_v[:, NOPE_DIM:].contiguous().view(torch.uint8).view(rows_v.shape[0], ROPE_DIM * 2)

    token_offsets = slots * TOKEN_DATA_BYTES
    scale_offsets = page_size * TOKEN_DATA_BYTES + slots * SCALE_DIM
    fp8_idx = token_offsets[:, None] + torch.arange(NOPE_DIM, device=rows.device)
    rope_idx = token_offsets[:, None] + NOPE_DIM + torch.arange(ROPE_DIM * 2, device=rows.device)
    scale_idx = scale_offsets[:, None] + torch.arange(SCALE_DIM, device=rows.device)
    flat[pages[:, None], fp8_idx] = encoded
    flat[pages[:, None], rope_idx] = rope_bytes
    flat[pages[:, None], scale_idx] = scales


@dataclass(frozen=True)
class MetadataCase:
    name: str
    bs: int
    positions: torch.Tensor
    ctx_page_table: torch.Tensor
    out_locs: torch.Tensor
    page_table: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor
    touched_locs: torch.Tensor
    description: str


def _token_table_from_pages(pages: list[list[int]], page_size: int, max_seq_len: int, device: torch.device) -> torch.Tensor:
    table = torch.full((len(pages), max_seq_len), -1, dtype=torch.int32, device=device)
    for row, row_pages in enumerate(pages):
        for logical_page, physical_page in enumerate(row_pages):
            start = logical_page * page_size
            end = min(start + page_size, max_seq_len)
            if start >= max_seq_len:
                break
            offsets = torch.arange(end - start, dtype=torch.int32, device=device)
            table[row, start:end] = int(physical_page) * page_size + offsets
    return table


def _derive_metadata(
    *,
    name: str,
    ctx_page_table: torch.Tensor,
    positions: torch.Tensor,
    page_size: int,
    window_size: int,
    description: str,
) -> MetadataCase:
    device = ctx_page_table.device
    bs = positions.numel()
    rows = torch.arange(bs, dtype=torch.long, device=device)
    out_locs = ctx_page_table[rows, positions.to(torch.long)].to(torch.int32)
    max_seq_len = int(positions.max().item()) + 1 if positions.numel() else 1
    page_width = max(1, math.ceil(max_seq_len / page_size))
    page_table = torch.full((bs, page_width), -1, dtype=torch.int32, device=device)
    for logical_page in range(page_width):
        logical_pos = logical_page * page_size
        vals = ctx_page_table[rows, torch.full((bs,), logical_pos, dtype=torch.long, device=device)]
        page_table[:, logical_page] = torch.where(vals >= 0, vals // page_size, vals).to(torch.int32)

    offsets = torch.arange(window_size, dtype=torch.int32, device=device)
    logical_positions = positions[:, None].to(torch.int32) - offsets[None, :]
    valid = logical_positions >= 0
    clamped = logical_positions.clamp_min(0).to(torch.long)
    row_idx = rows[:, None].expand_as(clamped)
    swa = ctx_page_table[row_idx, clamped].to(torch.int32)
    swa = torch.where(valid, swa, torch.full_like(swa, -1))
    lengths = torch.clamp(positions.to(torch.int32) + 1, max=window_size)
    touched = torch.unique(torch.cat([out_locs.reshape(-1), swa.reshape(-1)]))
    touched = touched[touched >= 0]
    return MetadataCase(
        name=name,
        bs=bs,
        positions=positions.to(torch.int32),
        ctx_page_table=ctx_page_table,
        out_locs=out_locs,
        page_table=page_table,
        swa_indices=swa.contiguous(),
        swa_lengths=lengths.contiguous(),
        touched_locs=touched.to(torch.int32).contiguous(),
        description=description,
    )


def make_real_decode_case(
    *,
    bs: int,
    page_size: int,
    num_pages: int,
    window_size: int,
    device: torch.device,
) -> MetadataCase:
    max_seq_len = 4096
    pages_per_req = math.ceil(max_seq_len / page_size)
    pages: list[list[int]] = []
    cursor = 0
    for row in range(bs):
        row_pages = [int((cursor + row * 7 + i * 3) % num_pages) for i in range(pages_per_req)]
        pages.append(row_pages)
        cursor += pages_per_req
    positions = torch.tensor(
        [max_seq_len - 1 - ((row * 17) % 97) for row in range(bs)],
        dtype=torch.long,
        device=device,
    )
    return _derive_metadata(
        name=f"real_mini_decode_bs{bs}",
        ctx_page_table=_token_table_from_pages(pages, page_size, max_seq_len, device),
        positions=positions,
        page_size=page_size,
        window_size=window_size,
        description="mini-style token page table, out_loc, and SWA tail indices at DSV4 decode shape",
    )


def make_tail_heavy_case(
    *,
    bs: int,
    page_size: int,
    num_pages: int,
    window_size: int,
    device: torch.device,
) -> MetadataCase:
    max_seq_len = 1024
    pages_per_req = math.ceil(max_seq_len / page_size)
    pages = [
        [int((row * pages_per_req + i * 5 + 11) % num_pages) for i in range(pages_per_req)]
        for row in range(bs)
    ]
    base_positions = [page_size - 1, page_size, page_size + 1, 2 * page_size - 1, 2 * page_size]
    positions = torch.tensor(
        [base_positions[row % len(base_positions)] + (row // len(base_positions)) for row in range(bs)],
        dtype=torch.long,
        device=device,
    ).clamp(max=max_seq_len - 1)
    return _derive_metadata(
        name=f"tail_heavy_decode_bs{bs}",
        ctx_page_table=_token_table_from_pages(pages, page_size, max_seq_len, device),
        positions=positions,
        page_size=page_size,
        window_size=window_size,
        description="decode rows concentrated around page tails and page-boundary crossings",
    )


def make_prefix_remap_case(
    *,
    bs: int,
    page_size: int,
    num_pages: int,
    window_size: int,
    device: torch.device,
) -> MetadataCase:
    max_seq_len = 768
    pages_per_req = math.ceil(max_seq_len / page_size)
    shared_prefix_pages = [3, 17]
    pages: list[list[int]] = []
    for row in range(bs):
        suffix = [int((41 + row * 13 + i * 7) % num_pages) for i in range(max(0, pages_per_req - 2))]
        row_pages = (shared_prefix_pages + suffix)[:pages_per_req]
        pages.append(row_pages)
    positions = torch.tensor(
        [511 + (row % 4) * 8 for row in range(bs)],
        dtype=torch.long,
        device=device,
    ).clamp(max=max_seq_len - 1)
    return _derive_metadata(
        name=f"prefix_hit_remap_touched_rows_bs{bs}",
        ctx_page_table=_token_table_from_pages(pages, page_size, max_seq_len, device),
        positions=positions,
        page_size=page_size,
        window_size=window_size,
        description="prefix-hit/remap simulation: shared retained prefix pages plus per-request suffix pages; checks only SWA touched rows",
    )


def make_random_rows_case(
    *,
    bs: int,
    page_size: int,
    num_pages: int,
    window_size: int,
    device: torch.device,
) -> MetadataCase:
    del bs
    max_seq_len = 512
    pages = [[int((i * 19 + 5) % num_pages) for i in range(math.ceil(max_seq_len / page_size))]]
    positions = torch.tensor([383], dtype=torch.long, device=device)
    case = _derive_metadata(
        name="random_bf16_rows",
        ctx_page_table=_token_table_from_pages(pages, page_size, max_seq_len, device),
        positions=positions,
        page_size=page_size,
        window_size=window_size,
        description="random BF16 rows with selected-row gather/dequant over a mini-style flat loc table",
    )
    extra = torch.arange(29, 29 + 97, dtype=torch.int32, device=device) * 3
    extra = extra % (num_pages * page_size)
    touched = torch.unique(torch.cat([case.touched_locs, extra.to(torch.int32)]))
    return MetadataCase(
        name=case.name,
        bs=case.bs,
        positions=case.positions,
        ctx_page_table=case.ctx_page_table,
        out_locs=case.out_locs,
        page_table=case.page_table,
        swa_indices=case.swa_indices,
        swa_lengths=case.swa_lengths,
        touched_locs=touched.contiguous(),
        description=case.description,
    )


def make_rows(count: int, device: torch.device, seed: int, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    rows = torch.randn((count, HEAD_DIM), dtype=torch.float32, device=device, generator=gen) * scale
    return rows.to(torch.bfloat16).contiguous()


def _metric_abs(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[float, float]:
    if mask is not None:
        a = a[mask]
        b = b[mask]
    if a.numel() == 0:
        return 0.0, 0.0
    err = (a.to(torch.float32) - b.to(torch.float32)).abs()
    return float(err.max().item()), float(err.mean().item())


def _valid_flat_mask(indices: torch.Tensor, capacity: int) -> torch.Tensor:
    return (indices >= 0) & (indices < capacity)


def correctness_for_case(
    case: MetadataCase,
    *,
    page_size: int,
    num_pages: int,
    seed: int,
    tolerance_max: float,
    tolerance_mean: float,
) -> dict[str, Any]:
    device = case.touched_locs.device
    rows = make_rows(case.touched_locs.numel(), device, seed)
    bf16_cache = torch.zeros((num_pages * page_size, HEAD_DIM), dtype=torch.bfloat16, device=device)
    packed_cache = make_packed_cache(num_pages, page_size, device)
    bf16_store(rows, case.touched_locs, bf16_cache)
    packed_mla_store(rows, case.touched_locs, packed_cache, page_size)
    torch.cuda.synchronize()

    ref_cache = torch.zeros_like(packed_cache)
    byte_mismatch: int | None
    byte_mismatch_error: str | None = None
    try:
        pack_reference(rows, case.touched_locs, ref_cache, page_size)
        torch.cuda.synchronize()
        byte_mismatch = int(torch.count_nonzero(packed_cache != ref_cache).item())
    except Exception as exc:  # pragma: no cover - depends on torch fp8 support.
        byte_mismatch = None
        byte_mismatch_error = repr(exc)

    store_dequant = torch.empty_like(rows)
    packed_mla_gather(packed_cache, case.touched_locs, store_dequant, page_size)

    selected = case.swa_indices.reshape(-1).contiguous()
    bf16_selected = torch.empty((selected.numel(), HEAD_DIM), dtype=torch.bfloat16, device=device)
    fp8_selected = torch.empty_like(bf16_selected)
    bf16_gather(bf16_cache, selected, bf16_selected)
    packed_mla_gather(packed_cache, selected, fp8_selected, page_size)
    torch.cuda.synchronize()

    valid = _valid_flat_mask(selected, num_pages * page_size)
    store_nope_max, store_nope_mean = _metric_abs(store_dequant[:, :NOPE_DIM], rows[:, :NOPE_DIM])
    store_rope_max, store_rope_mean = _metric_abs(store_dequant[:, NOPE_DIM:], rows[:, NOPE_DIM:])
    gather_nope_max, gather_nope_mean = _metric_abs(
        fp8_selected[:, :NOPE_DIM],
        bf16_selected[:, :NOPE_DIM],
        valid,
    )
    gather_rope_max, gather_rope_mean = _metric_abs(
        fp8_selected[:, NOPE_DIM:],
        bf16_selected[:, NOPE_DIM:],
        valid,
    )
    passed = (
        store_nope_max <= tolerance_max
        and gather_nope_max <= tolerance_max
        and store_nope_mean <= tolerance_mean
        and gather_nope_mean <= tolerance_mean
        and store_rope_max == 0.0
        and gather_rope_max == 0.0
        and (byte_mismatch in (0, None))
    )
    return {
        "name": case.name,
        "description": case.description,
        "bs": case.bs,
        "positions": [int(x) for x in case.positions.detach().cpu().tolist()],
        "page_table_shape": list(case.page_table.shape),
        "swa_indices_shape": list(case.swa_indices.shape),
        "out_locs": [int(x) for x in case.out_locs.detach().cpu().tolist()],
        "stored_rows": int(case.touched_locs.numel()),
        "selected_slots": int(selected.numel()),
        "valid_selected_slots": int(valid.sum().item()),
        "store_nope_max_abs_error": store_nope_max,
        "store_nope_mean_abs_error": store_nope_mean,
        "store_rope_max_abs_error": store_rope_max,
        "store_rope_mean_abs_error": store_rope_mean,
        "gather_nope_max_abs_error": gather_nope_max,
        "gather_nope_mean_abs_error": gather_nope_mean,
        "gather_rope_max_abs_error": gather_rope_max,
        "gather_rope_mean_abs_error": gather_rope_mean,
        "packed_byte_mismatch_vs_torch_ref": byte_mismatch,
        "packed_byte_mismatch_error": byte_mismatch_error,
        "tolerance": {
            "nope_max_abs": tolerance_max,
            "nope_mean_abs": tolerance_mean,
            "rope_max_abs": 0.0,
        },
        "passed": bool(passed),
    }


def time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def traffic_bytes(kind: str, rows: int, selected: int) -> int:
    bf16_row = HEAD_DIM * 2
    fp8_store_row = HEAD_DIM * 2 + TOKEN_BYTES_UNPADDED
    fp8_gather_row = (NOPE_DIM + ROPE_DIM * 2 + NOPE_BLOCKS) + bf16_row
    if kind == "bf16_store":
        return rows * (bf16_row + bf16_row)
    if kind == "bf16_gather":
        return selected * (bf16_row + bf16_row)
    if kind == "bf16_combined":
        return traffic_bytes("bf16_store", rows, selected) + traffic_bytes("bf16_gather", rows, selected)
    if kind == "fp8_store":
        return rows * fp8_store_row
    if kind == "fp8_gather":
        return selected * fp8_gather_row
    if kind == "fp8_combined":
        return traffic_bytes("fp8_store", rows, selected) + traffic_bytes("fp8_gather", rows, selected)
    raise ValueError(kind)


def bench_bucket(
    *,
    bs: int,
    page_size: int,
    num_pages: int,
    window_size: int,
    seed: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    case = make_real_decode_case(
        bs=bs,
        page_size=page_size,
        num_pages=num_pages,
        window_size=window_size,
        device=device,
    )
    rows_all = make_rows(case.touched_locs.numel(), device, seed)
    current_rows = make_rows(case.out_locs.numel(), device, seed + 1000)
    selected = case.swa_indices.reshape(-1).contiguous()
    bf16_cache = torch.zeros((num_pages * page_size, HEAD_DIM), dtype=torch.bfloat16, device=device)
    packed_cache = make_packed_cache(num_pages, page_size, device)
    bf16_store(rows_all, case.touched_locs, bf16_cache)
    packed_mla_store(rows_all, case.touched_locs, packed_cache, page_size)
    bf16_out = torch.empty((selected.numel(), HEAD_DIM), dtype=torch.bfloat16, device=device)
    fp8_out = torch.empty_like(bf16_out)
    torch.cuda.synchronize()

    timings = {
        "bf16_store_ms": time_cuda(
            lambda: bf16_store(current_rows, case.out_locs, bf16_cache),
            warmup=warmup,
            iters=iters,
        ),
        "bf16_selected_gather_ms": time_cuda(
            lambda: bf16_gather(bf16_cache, selected, bf16_out),
            warmup=warmup,
            iters=iters,
        ),
        "bf16_combined_store_gather_ms": time_cuda(
            lambda: (bf16_store(current_rows, case.out_locs, bf16_cache), bf16_gather(bf16_cache, selected, bf16_out)),
            warmup=warmup,
            iters=iters,
        ),
        "fp8_packed_store_quant_ms": time_cuda(
            lambda: packed_mla_store(current_rows, case.out_locs, packed_cache, page_size),
            warmup=warmup,
            iters=iters,
        ),
        "fp8_selected_gather_dequant_ms": time_cuda(
            lambda: packed_mla_gather(packed_cache, selected, fp8_out, page_size),
            warmup=warmup,
            iters=iters,
        ),
        "fp8_combined_store_gather_dequant_ms": time_cuda(
            lambda: (
                packed_mla_store(current_rows, case.out_locs, packed_cache, page_size),
                packed_mla_gather(packed_cache, selected, fp8_out, page_size),
            ),
            warmup=warmup,
            iters=iters,
        ),
    }
    rows = int(case.out_locs.numel())
    selected_count = int(selected.numel())
    traffic = {
        "bf16_store_bytes": traffic_bytes("bf16_store", rows, selected_count),
        "bf16_selected_gather_bytes": traffic_bytes("bf16_gather", rows, selected_count),
        "bf16_combined_store_gather_bytes": traffic_bytes("bf16_combined", rows, selected_count),
        "fp8_packed_store_quant_bytes": traffic_bytes("fp8_store", rows, selected_count),
        "fp8_selected_gather_dequant_bytes": traffic_bytes("fp8_gather", rows, selected_count),
        "fp8_combined_store_gather_dequant_bytes": traffic_bytes("fp8_combined", rows, selected_count),
    }
    effective_gbps = {}
    for key, value in timings.items():
        bytes_key = key.replace("_ms", "_bytes")
        if bytes_key in traffic and value > 0:
            effective_gbps[key.replace("_ms", "_effective_gbps")] = traffic[bytes_key] / value / 1.0e6
    return {
        "bucket_bs": bs,
        "page_size": page_size,
        "num_pages": num_pages,
        "head_dim": HEAD_DIM,
        "window_size": window_size,
        "store_rows": rows,
        "selected_rows": selected_count,
        "workspace_bytes": selected_count * HEAD_DIM * 2,
        **timings,
        **traffic,
        **effective_gbps,
    }


def graph_check_bucket(
    *,
    bs: int,
    page_size: int,
    num_pages: int,
    window_size: int,
    seed: int,
    replays: int,
    device: torch.device,
) -> dict[str, Any]:
    case = make_real_decode_case(
        bs=bs,
        page_size=page_size,
        num_pages=num_pages,
        window_size=window_size,
        device=device,
    )
    rows_all = make_rows(case.touched_locs.numel(), device, seed)
    current_rows = make_rows(case.out_locs.numel(), device, seed + 2000)
    selected = case.swa_indices.reshape(-1).contiguous()
    packed_cache = make_packed_cache(num_pages, page_size, device)
    packed_mla_store(rows_all, case.touched_locs, packed_cache, page_size)
    out = torch.empty((selected.numel(), HEAD_DIM), dtype=torch.bfloat16, device=device)
    for _ in range(3):
        packed_mla_store(current_rows, case.out_locs, packed_cache, page_size)
        packed_mla_gather(packed_cache, selected, out, page_size)
    torch.cuda.synchronize()

    before_capture = int(torch.cuda.memory_allocated(device))
    graph = torch.cuda.CUDAGraph()
    capture_error: str | None = None
    replay_deltas: list[int] = []
    try:
        with torch.cuda.graph(graph):
            packed_mla_store(current_rows, case.out_locs, packed_cache, page_size)
            packed_mla_gather(packed_cache, selected, out, page_size)
        torch.cuda.synchronize()
        after_capture = int(torch.cuda.memory_allocated(device))
        for _ in range(replays):
            before = int(torch.cuda.memory_allocated(device))
            graph.replay()
            torch.cuda.synchronize()
            after = int(torch.cuda.memory_allocated(device))
            replay_deltas.append(after - before)
    except Exception as exc:
        after_capture = int(torch.cuda.memory_allocated(device))
        capture_error = repr(exc)
    return {
        "bucket_bs": bs,
        "captured": capture_error is None,
        "capture_error": capture_error,
        "replay_count": 0 if capture_error else replays,
        "eager_fallback_count": 0 if capture_error is None else 1,
        "allocated_before_capture_bytes": before_capture,
        "allocated_after_capture_bytes": after_capture,
        "capture_alloc_delta_bytes": after_capture - before_capture,
        "max_replay_alloc_delta_bytes": max(replay_deltas) if replay_deltas else None,
        "workspace_preallocated": True,
        "workspace_bytes": int(out.numel() * out.element_size()),
        "dynamic_allocation_during_replay": bool(any(delta != 0 for delta in replay_deltas)),
    }


def capacity_ledger(
    *,
    num_layers: int,
    page_size: int,
    num_pages: int,
    device: torch.device,
    measure: bool,
) -> dict[str, Any]:
    page_bytes = packed_page_bytes(page_size)
    bf16_swa = num_layers * num_pages * page_size * HEAD_DIM * 2
    fp8_swa = num_layers * num_pages * page_bytes
    scale_bytes = num_layers * num_pages * page_size * SCALE_DIM
    padding_bytes = num_layers * num_pages * (page_bytes - page_size * TOKEN_BYTES_UNPADDED)
    data_bytes = num_layers * num_pages * page_size * TOKEN_DATA_BYTES
    measured: dict[str, Any] = {"enabled": False}
    if measure:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = int(torch.cuda.memory_allocated(device))
        bf16_tensor = torch.empty(
            (num_layers, num_pages, page_size, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        torch.cuda.synchronize()
        bf16_alloc = int(torch.cuda.memory_allocated(device)) - base
        del bf16_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base2 = int(torch.cuda.memory_allocated(device))
        fp8_tensor = torch.empty(
            (num_layers, num_pages, page_bytes),
            dtype=torch.uint8,
            device=device,
        )
        torch.cuda.synchronize()
        fp8_alloc = int(torch.cuda.memory_allocated(device)) - base2
        del fp8_tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        measured = {
            "enabled": True,
            "bf16_swa_alloc_delta_bytes": bf16_alloc,
            "fp8_swa_alloc_delta_bytes": fp8_alloc,
            "saved_alloc_delta_bytes": bf16_alloc - fp8_alloc,
            "saved_alloc_delta_gib": (bf16_alloc - fp8_alloc) / (1024**3),
        }
    return {
        "num_layers": num_layers,
        "page_size": page_size,
        "num_pages": num_pages,
        "packed_page_bytes": page_bytes,
        "bf16_swa_bytes": bf16_swa,
        "fp8_swa_packed_bytes": fp8_swa,
        "saved_bytes": bf16_swa - fp8_swa,
        "bf16_swa_gib": bf16_swa / (1024**3),
        "fp8_swa_packed_gib": fp8_swa / (1024**3),
        "saved_gib": (bf16_swa - fp8_swa) / (1024**3),
        "token_data_bytes": data_bytes,
        "scale_bytes": scale_bytes,
        "page_padding_bytes": padding_bytes,
        "scale_gib": scale_bytes / (1024**3),
        "page_padding_gib": padding_bytes / (1024**3),
        "measured_allocation": measured,
    }


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="TARGET 09.4 SWA packed MLA FP8 slice harness.")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=43)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--bench-buckets", default="1,2,4,8,16")
    parser.add_argument("--graph-buckets", default="1,2,4,8,16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--graph-replays", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--tolerance-max", type=float, default=0.25)
    parser.add_argument("--tolerance-mean", type=float, default=0.02)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-capacity-alloc-probe", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones"
        / "target09_minimal_fp8_kv_cache_slice"
        / "summaries"
        / "swa_packed_mla_slice_harness.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This harness requires CUDA.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.time()

    correctness_cases = [
        make_random_rows_case(
            bs=1,
            page_size=args.page_size,
            num_pages=args.num_pages,
            window_size=args.window_size,
            device=device,
        ),
        make_real_decode_case(
            bs=16,
            page_size=args.page_size,
            num_pages=args.num_pages,
            window_size=args.window_size,
            device=device,
        ),
        make_tail_heavy_case(
            bs=16,
            page_size=args.page_size,
            num_pages=args.num_pages,
            window_size=args.window_size,
            device=device,
        ),
        make_prefix_remap_case(
            bs=16,
            page_size=args.page_size,
            num_pages=args.num_pages,
            window_size=args.window_size,
            device=device,
        ),
    ]
    correctness = [
        correctness_for_case(
            case,
            page_size=args.page_size,
            num_pages=args.num_pages,
            seed=args.seed + idx,
            tolerance_max=args.tolerance_max,
            tolerance_mean=args.tolerance_mean,
        )
        for idx, case in enumerate(correctness_cases)
    ]

    microbench = [
        bench_bucket(
            bs=bs,
            page_size=args.page_size,
            num_pages=args.num_pages,
            window_size=args.window_size,
            seed=args.seed + 100 * bs,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        for bs in parse_ints(args.bench_buckets)
    ]

    graph = []
    if not args.skip_graph:
        graph = [
            graph_check_bucket(
                bs=bs,
                page_size=args.page_size,
                num_pages=args.num_pages,
                window_size=args.window_size,
                seed=args.seed + 1000 * bs,
                replays=args.graph_replays,
                device=device,
            )
            for bs in parse_ints(args.graph_buckets)
        ]

    ledger = capacity_ledger(
        num_layers=args.num_layers,
        page_size=args.page_size,
        num_pages=args.num_pages,
        device=device,
        measure=not args.skip_capacity_alloc_probe,
    )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - started,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": torch.cuda.get_device_capability(device),
        "source_aligned_layout": {
            "head_dim": HEAD_DIM,
            "nope_fp8_dim": NOPE_DIM,
            "rope_bf16_dim": ROPE_DIM,
            "quant_block": QUANT_BLOCK,
            "ue8m0_scale_bytes_per_token": SCALE_DIM,
            "token_data_bytes": TOKEN_DATA_BYTES,
            "token_bytes_unpadded": TOKEN_BYTES_UNPADDED,
            "page_stride_alignment": PAGE_STRIDE_ALIGNMENT,
            "page_bytes": packed_page_bytes(args.page_size),
            "page_padding_bytes": packed_page_bytes(args.page_size)
            - args.page_size * TOKEN_BYTES_UNPADDED,
        },
        "mini_shapes": {
            "num_layers": args.num_layers,
            "page_size": args.page_size,
            "num_pages": args.num_pages,
            "window_size": args.window_size,
            "graph_buckets": parse_ints(args.graph_buckets),
            "bench_buckets": parse_ints(args.bench_buckets),
        },
        "correctness": correctness,
        "microbench": microbench,
        "graph_safety": graph,
        "capacity_ledger": ledger,
        "capabilities": (
            asdict(dsv4_kernel.detect_dsv4_kernel_capabilities())
            if dsv4_kernel is not None
            else None
        ),
        "overall": {
            "correctness_passed": all(item["passed"] for item in correctness),
            "graph_passed": (not graph)
            or all(
                item["captured"]
                and item["replay_count"] > 0
                and item["eager_fallback_count"] == 0
                and not item["dynamic_allocation_during_replay"]
                for item in graph
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["overall"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

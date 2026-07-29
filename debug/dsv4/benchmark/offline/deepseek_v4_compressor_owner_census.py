#!/usr/bin/env python3
"""Saved-input Target 17.1 compressor owner census and C4 feasibility probes.

This file is deliberately outside the production import path.  It calls the
current production Triton kernels directly to time individual launches, and
contains only two debug counterfactuals:

* an explicit compact C4 boundary-row list;
* a fixed-capacity C4 plan whose invalid entries return before source loads.

Neither counterfactual is reachable from the engine/model runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kernel.triton import deepseek_v4 as dsv4_triton
from torch.profiler import ProfilerActivity, profile

from debug.dsv4.kernel import deepseek_v4_reference as dsv4_reference  # noqa: E402

PAGE_SIZE = 256
HIDDEN_DIM = 16384
ROTARY_DIM = 64
PRODUCTION_COMPRESS_BASE = 160000.0
PRODUCTION_ORIGINAL_SEQ_LEN = 65536
PRODUCTION_ROPE_FACTOR = 16.0
PRODUCTION_BETA_FAST = 32
PRODUCTION_BETA_SLOW = 1
PRODUCTION_MAX_POSITION = 1048576
HIGH_PUBLICATION_POSITIONS = [
    0,
    4,
    65532,
    65536,
    65540,
    131068,
    131072,
    524284,
    524288,
    1048568,
    1048572,
]
Component = Literal["c4_attention", "c4_indexer", "c128_attention"]
Pattern = Literal["no_boundary", "natural_mixed", "all_boundary"]

COMPONENTS: dict[Component, dict[str, int]] = {
    "c4_attention": {"ratio": 4, "head_dim": 512, "projected_dim": 2048},
    "c4_indexer": {"ratio": 4, "head_dim": 128, "projected_dim": 512},
    "c128_attention": {"ratio": 128, "head_dim": 512, "projected_dim": 1024},
}


@triton.jit
def _c4_debug_plan_pool_kernel(
    projected_ptr,
    sequence_state_ptr,
    checkpoint_ptr,
    ape_ptr,
    positions_ptr,
    table_indices_ptr,
    ctx_page_table_ptr,
    checkpoint_page_mapping_ptr,
    plan_rows_ptr,
    output_ptr,
    plan_entries: tl.constexpr,
    physical_rows: tl.constexpr,
    head_dim: tl.constexpr,
    projected_stride0: tl.constexpr,
    sequence_state_stride0: tl.constexpr,
    checkpoint_stride0: tl.constexpr,
    sequence_state_slots: tl.constexpr,
    ctx_page_table_stride0: tl.constexpr,
    ctx_page_table_width: tl.constexpr,
    checkpoint_page_mapping_width: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Exact copy of Mini's C4 reduction, indexed through a debug plan."""

    plan_slot = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < head_dim
    row = tl.load(plan_rows_ptr + plan_slot)
    valid_plan = (row >= 0) & (row < physical_rows)
    if not valid_plan:
        tl.store(output_ptr + plan_slot * head_dim + d, 0.0, mask=d_mask)
        return

    pos = tl.load(positions_ptr + row)
    table_idx = tl.load(table_indices_ptr + row)

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

        use_checkpoint = (source_slot < 4) & (logical_pos // page_size != pos // page_size)
        full_loc = tl.load(
            ctx_page_table_ptr + table_idx * ctx_page_table_stride0 + logical_pos,
            mask=use_checkpoint
            & (table_idx >= 0)
            & (logical_pos >= 0)
            & (logical_pos < ctx_page_table_width),
            other=-1,
        )
        full_page = full_loc // page_size
        checkpoint_page = tl.load(
            checkpoint_page_mapping_ptr + full_page,
            mask=use_checkpoint
            & (full_loc >= 0)
            & (full_page >= 0)
            & (full_page < checkpoint_page_mapping_width),
            other=-1,
        )
        checkpoint_loc = checkpoint_page * 4 + (logical_pos % 4)
        checkpoint_persistent = (
            use_checkpoint & (logical_pos >= 0) & (full_loc >= 0) & (checkpoint_page >= 0)
        )
        sequence_loc = table_idx * 8 + (logical_pos % 8)
        sequence_persistent = (
            (~use_checkpoint)
            & (logical_pos >= 0)
            & (table_idx >= 0)
            & (table_idx < sequence_state_slots)
        )

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
        sequence_kv = tl.load(
            sequence_state_ptr + sequence_loc * sequence_state_stride0 + kv_offset + d,
            mask=(~current) & sequence_persistent & d_mask,
            other=0.0,
        ).to(tl.float32)
        sequence_score = tl.load(
            sequence_state_ptr + sequence_loc * sequence_state_stride0 + score_offset + d,
            mask=(~current) & sequence_persistent & d_mask,
            other=float("-inf"),
        ).to(tl.float32)
        checkpoint_kv = tl.load(
            checkpoint_ptr + checkpoint_loc * checkpoint_stride0 + d,
            mask=(~current) & checkpoint_persistent & d_mask,
            other=0.0,
        ).to(tl.float32)
        checkpoint_score = tl.load(
            checkpoint_ptr + checkpoint_loc * checkpoint_stride0 + head_dim + d,
            mask=(~current) & checkpoint_persistent & d_mask,
            other=float("-inf"),
        ).to(tl.float32)
        historical_kv = tl.where(use_checkpoint, checkpoint_kv, sequence_kv)
        historical_score = tl.where(
            use_checkpoint,
            checkpoint_score,
            sequence_score,
        )
        source_kv = tl.where(current, current_kv, historical_kv)
        source_score = tl.where(current, current_score, historical_score)
        ape_row = source_slot if source_slot < 4 else source_slot - 4
        ape_col = d if source_slot < 4 else head_dim + d
        source_score += tl.load(
            ape_ptr + ape_row * (2 * head_dim) + ape_col,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        source_score = tl.where(
            current | checkpoint_persistent | sequence_persistent,
            source_score,
            float("-inf"),
        )

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
    tl.store(output_ptr + plan_slot * head_dim + d, pooled, mask=d_mask)


@triton.jit
def _build_c4_fixed_plan_kernel(
    positions_ptr,
    plan_rows_ptr,
    rows: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    positions = tl.load(positions_ptr + offsets, mask=mask, other=0)
    entries = tl.where(((positions + 1) % 4) == 0, offsets, -1)
    tl.store(plan_rows_ptr + offsets, entries, mask=mask)


@dataclass
class Fixture:
    component: Component
    pattern: Pattern
    rows: int
    projected: torch.Tensor
    sequence_state: torch.Tensor
    checkpoint: torch.Tensor | None
    ape: torch.Tensor
    positions: torch.Tensor
    table_indices: torch.Tensor
    raw_out_loc: torch.Tensor
    ctx_page_table: torch.Tensor | None
    checkpoint_page_mapping: torch.Tensor | None

    @property
    def ratio(self) -> int:
        return COMPONENTS[self.component]["ratio"]

    @property
    def head_dim(self) -> int:
        return COMPONENTS[self.component]["head_dim"]

    @property
    def boundary_rows(self) -> torch.Tensor:
        return (
            torch.nonzero(
                (self.positions + 1) % self.ratio == 0,
                as_tuple=False,
            )
            .flatten()
            .to(torch.int32)
            .contiguous()
        )


class FakeCompressedCache:
    def __init__(self, rows: int, dim: int, device: torch.device) -> None:
        self.cache = torch.empty(
            max(rows, 1),
            dim,
            dtype=torch.bfloat16,
            device=device,
        )

    def component_cache(self, layer_id: int) -> torch.Tensor:
        assert layer_id == 0
        return self.cache


class FakeIndexerCache:
    def __init__(self, rows: int, dim: int, device: torch.device) -> None:
        self._bf16 = torch.empty(
            max(rows, 1),
            dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self._page_size = 64
        pages = max(math.ceil(max(rows, 1) / self._page_size), 1)
        self._packed = torch.empty(
            pages,
            self._page_size * (dim + 4),
            dtype=torch.uint8,
            device=device,
        )

    def indexer_cache(self, layer_id: int) -> torch.Tensor:
        assert layer_id == 0
        return self._bf16

    def has_indexer_fp8_cache(self) -> bool:
        return True

    def has_indexer_fp8_paged_cache(self) -> bool:
        return True

    def indexer_fp8_paged_cache(self, layer_id: int) -> torch.Tensor:
        assert layer_id == 0
        return self._packed

    @property
    def indexer_fp8_page_size(self) -> int:
        return self._page_size


def _positions(
    rows: int,
    ratio: int,
    pattern: Pattern,
    scenario: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scenario == "ragged":
        lengths = [rows // 3, rows // 3, rows - 2 * (rows // 3)]
        starts = [0, 253, 510]
        positions = []
        tables = []
        for table, (length, start) in enumerate(zip(lengths, starts)):
            positions.extend(range(start, start + length))
            tables.extend([table] * length)
        return (
            torch.tensor(positions, dtype=torch.int32, device=device),
            torch.tensor(tables, dtype=torch.int32, device=device),
        )
    if scenario == "continuation":
        return (
            torch.arange(257, 257 + rows, dtype=torch.int32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
        )
    if scenario == "page_boundary":
        return (
            torch.arange(252, 252 + rows, dtype=torch.int32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
        )
    if scenario == "ratio_boundary":
        start = ratio - min(rows, ratio)
        return (
            torch.arange(start, start + rows, dtype=torch.int32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
        )
    if pattern == "no_boundary":
        positions = torch.arange(rows, dtype=torch.int32, device=device) * ratio
        tables = torch.arange(rows, dtype=torch.int32, device=device) % min(rows, 128)
        return positions, tables
    if pattern == "all_boundary":
        positions = torch.arange(rows, dtype=torch.int32, device=device) * ratio + (ratio - 1)
        tables = torch.arange(rows, dtype=torch.int32, device=device) % min(rows, 128)
        return positions, tables
    positions = torch.arange(rows, dtype=torch.int32, device=device)
    tables = torch.zeros(rows, dtype=torch.int32, device=device)
    return positions, tables


def make_fixture(
    component: Component,
    rows: int,
    pattern: Pattern,
    *,
    scenario: str = "matrix",
    parity: bool = False,
    seed: int = 1701,
    device: torch.device,
) -> Fixture:
    config = COMPONENTS[component]
    ratio = config["ratio"]
    head_dim = config["head_dim"]
    projected_dim = config["projected_dim"]
    positions, table_indices = _positions(
        rows,
        ratio,
        pattern,
        scenario,
        device,
    )
    if parity and pattern in ("no_boundary", "all_boundary") and scenario == "matrix":
        table_indices = torch.arange(rows, dtype=torch.int32, device=device)
    slots = max(int(table_indices.max().item()) + 1 if rows else 1, 1)
    generator = torch.Generator(device=device).manual_seed(seed + rows + projected_dim)
    projected = torch.randn(
        rows,
        projected_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    state = torch.randn(
        slots * ratio,
        projected_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    ape = torch.randn(
        ratio,
        (2 if ratio == 4 else 1) * head_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    raw_out_loc = torch.arange(rows, dtype=torch.int32, device=device)
    if ratio == 128:
        return Fixture(
            component,
            pattern,
            rows,
            projected,
            state,
            None,
            ape,
            positions.contiguous(),
            table_indices.contiguous(),
            raw_out_loc,
            None,
            None,
        )

    raw_pages = max(math.ceil(max(rows, 1) / PAGE_SIZE), 1)
    checkpoint = torch.randn(
        raw_pages * 4,
        2 * head_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    checkpoint_mapping = torch.arange(
        raw_pages,
        dtype=torch.int32,
        device=device,
    )
    max_position = int(positions.max().item()) + 1 if rows else 1
    table_width = max_position if parity else min(max(max_position, 1), 1024)
    ctx = torch.full(
        (slots, table_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    if table_width:
        ctx[:, : min(table_width, PAGE_SIZE)] = torch.arange(
            min(table_width, PAGE_SIZE),
            dtype=torch.int32,
            device=device,
        )
    return Fixture(
        component,
        pattern,
        rows,
        projected,
        state,
        checkpoint,
        ape,
        positions.contiguous(),
        table_indices.contiguous(),
        raw_out_loc,
        ctx.contiguous(),
        checkpoint_mapping,
    )


def _launch_c4_pool(fixture: Fixture, output: torch.Tensor) -> None:
    assert fixture.checkpoint is not None
    assert fixture.ctx_page_table is not None
    assert fixture.checkpoint_page_mapping is not None
    head_dim = fixture.head_dim
    block_d = min(triton.next_power_of_2(head_dim), 256)
    dsv4_triton._c4_online_pool_kernel[(fixture.rows, triton.cdiv(head_dim, block_d))](
        fixture.projected,
        fixture.sequence_state,
        fixture.checkpoint,
        fixture.ape,
        fixture.positions,
        fixture.table_indices,
        fixture.ctx_page_table,
        fixture.checkpoint_page_mapping,
        output,
        rows=fixture.rows,
        head_dim=head_dim,
        projected_stride0=fixture.projected.stride(0),
        sequence_state_stride0=fixture.sequence_state.stride(0),
        checkpoint_stride0=fixture.checkpoint.stride(0),
        sequence_state_slots=fixture.sequence_state.shape[0] // 8,
        ctx_page_table_stride0=fixture.ctx_page_table.stride(0),
        ctx_page_table_width=fixture.ctx_page_table.shape[1],
        checkpoint_page_mapping_width=fixture.checkpoint_page_mapping.numel(),
        page_size=PAGE_SIZE,
        BLOCK_D=block_d,
        num_warps=4,
    )


def _launch_c4_state(fixture: Fixture) -> None:
    assert fixture.checkpoint is not None
    assert fixture.checkpoint_page_mapping is not None
    width = fixture.projected.shape[1]
    block = 256
    dsv4_triton._c4_online_state_store_kernel[(fixture.rows, triton.cdiv(width, block))](
        fixture.projected,
        fixture.raw_out_loc,
        fixture.positions,
        fixture.table_indices,
        fixture.checkpoint_page_mapping,
        fixture.sequence_state,
        fixture.checkpoint,
        rows=fixture.rows,
        projected_stride0=fixture.projected.stride(0),
        sequence_state_stride0=fixture.sequence_state.stride(0),
        checkpoint_stride0=fixture.checkpoint.stride(0),
        sequence_state_slots=fixture.sequence_state.shape[0] // 8,
        checkpoint_page_mapping_width=fixture.checkpoint_page_mapping.numel(),
        page_size=PAGE_SIZE,
        head_dim=fixture.head_dim,
        width=width,
        BLOCK=block,
        num_warps=4,
    )


def _launch_c128_pool(fixture: Fixture, output: torch.Tensor) -> None:
    dsv4_triton._c128_online_pool_kernel[(fixture.rows, fixture.head_dim)](
        fixture.projected,
        fixture.sequence_state,
        fixture.ape,
        fixture.positions,
        fixture.table_indices,
        output,
        rows=fixture.rows,
        head_dim=fixture.head_dim,
        projected_stride0=fixture.projected.stride(0),
        state_stride0=fixture.sequence_state.stride(0),
        state_sequence_slots=fixture.sequence_state.shape[0] // 128,
        RATIO=128,
        num_warps=4,
    )


def _launch_c128_state(fixture: Fixture) -> None:
    width = fixture.projected.shape[1]
    block = 256
    dsv4_triton._c128_online_state_store_kernel[(fixture.rows, triton.cdiv(width, block))](
        fixture.projected,
        fixture.ape,
        fixture.positions,
        fixture.table_indices,
        fixture.sequence_state,
        rows=fixture.rows,
        head_dim=fixture.head_dim,
        projected_stride0=fixture.projected.stride(0),
        state_stride0=fixture.sequence_state.stride(0),
        state_sequence_slots=fixture.sequence_state.shape[0] // 128,
        RATIO=128,
        width=width,
        BLOCK=block,
        num_warps=4,
    )


def _launch_plan_pool(
    fixture: Fixture,
    plan: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if plan.numel() == 0:
        return
    assert fixture.checkpoint is not None
    assert fixture.ctx_page_table is not None
    assert fixture.checkpoint_page_mapping is not None
    head_dim = fixture.head_dim
    block_d = min(triton.next_power_of_2(head_dim), 256)
    _c4_debug_plan_pool_kernel[(plan.numel(), triton.cdiv(head_dim, block_d))](
        fixture.projected,
        fixture.sequence_state,
        fixture.checkpoint,
        fixture.ape,
        fixture.positions,
        fixture.table_indices,
        fixture.ctx_page_table,
        fixture.checkpoint_page_mapping,
        plan,
        output,
        plan_entries=plan.numel(),
        physical_rows=fixture.rows,
        head_dim=head_dim,
        projected_stride0=fixture.projected.stride(0),
        sequence_state_stride0=fixture.sequence_state.stride(0),
        checkpoint_stride0=fixture.checkpoint.stride(0),
        sequence_state_slots=fixture.sequence_state.shape[0] // 8,
        ctx_page_table_stride0=fixture.ctx_page_table.stride(0),
        ctx_page_table_width=fixture.ctx_page_table.shape[1],
        checkpoint_page_mapping_width=fixture.checkpoint_page_mapping.numel(),
        page_size=PAGE_SIZE,
        BLOCK_D=block_d,
        num_warps=4,
    )


def _build_fixed_plan(fixture: Fixture, plan: torch.Tensor) -> None:
    block = 256
    _build_c4_fixed_plan_kernel[(triton.cdiv(fixture.rows, block),)](
        fixture.positions,
        plan,
        rows=fixture.rows,
        BLOCK=block,
        num_warps=1,
    )


def _timing_parameters(rows: int) -> tuple[int, int, int]:
    if rows <= 128:
        return 20, 100, 9
    if rows <= 1024:
        return 10, 30, 7
    if rows <= 4096:
        return 5, 10, 7
    return 3, 5, 7


def time_cuda(fn: Callable[[], None], rows: int) -> dict[str, object]:
    warmup, iterations, repeats = _timing_parameters(rows)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {
        "median_us": median,
        "mad_us": mad,
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
        "warmup": warmup,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
    }


def time_cuda_graph_replay(
    fn: Callable[[], None],
    rows: int,
) -> dict[str, object]:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    _, iterations, repeats = _timing_parameters(rows)
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / iterations)
    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {
        "median_us": median,
        "mad_us": mad,
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "replay_memory_allocated_delta_bytes": after_allocated - before_allocated,
        "replay_memory_reserved_delta_bytes": after_reserved - before_reserved,
    }


def profile_cuda_launches(fn: Callable[[], None]) -> dict[str, object]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        fn()
        torch.cuda.synchronize()
    aggregated: dict[str, dict[str, object]] = {}
    for event in trace.events():
        if str(event.device_type) != "DeviceType.CUDA":
            continue
        item = aggregated.setdefault(
            event.name,
            {
                "name": event.name,
                "launches": 0,
                "self_cuda_time_us": 0.0,
            },
        )
        item["launches"] = int(item["launches"]) + 1
        item["self_cuda_time_us"] = float(item["self_cuda_time_us"]) + float(
            event.self_device_time_total
        )
    kernels = list(aggregated.values())
    return {
        "cuda_kernel_launches": sum(int(item["launches"]) for item in kernels),
        "kernels": kernels,
    }


def _projection_case(
    component: Component,
    rows: int,
    device: torch.device,
) -> dict[str, object]:
    projected_dim = COMPONENTS[component]["projected_dim"]
    generator = torch.Generator(device=device).manual_seed(17100 + rows + projected_dim)
    x = torch.randn(
        rows,
        HIDDEN_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    weight = torch.randn(
        projected_dim,
        HIDDEN_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    def invoke() -> None:
        out = dsv4_reference.linear_bf16_fp32_fallback(x, weight)
        if out.dtype is not torch.float32:
            raise AssertionError(out.dtype)

    timing = time_cuda(invoke, rows)
    return {
        "operation": "fp32_projection",
        "component": component,
        "rows": rows,
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": [rows, projected_dim],
        "input_dtype": str(x.dtype),
        "weight_dtype": str(weight.dtype),
        "output_dtype": "torch.float32",
        "estimated_hbm_bytes": (
            x.numel() * x.element_size()
            + weight.numel() * weight.element_size()
            + rows * projected_dim * torch.float32.itemsize
        ),
        "flops": 2 * rows * HIDDEN_DIM * projected_dim,
        "launches": ["cublas/torch.mm"],
        **timing,
    }


def _attention_publication(
    component: Component,
    fixture: Fixture,
    *,
    compact: bool,
) -> tuple[Callable[[], None], dict[str, object]]:
    boundary = (fixture.positions + 1) % fixture.ratio == 0
    if compact:
        source = torch.randn(
            int(boundary.sum().item()),
            fixture.head_dim,
            dtype=torch.bfloat16,
            device=fixture.projected.device,
        )
        positions = fixture.positions[boundary] + 1 - fixture.ratio
        loc = torch.arange(
            source.shape[0],
            dtype=torch.int64,
            device=source.device,
        )
    else:
        source = torch.randn(
            fixture.rows,
            fixture.head_dim,
            dtype=torch.bfloat16,
            device=fixture.projected.device,
        )
        positions = fixture.positions + 1 - fixture.ratio
        loc = torch.where(
            boundary,
            torch.arange(fixture.rows, device=source.device),
            torch.full((fixture.rows,), -1, device=source.device),
        ).to(torch.int64)
    cache = FakeCompressedCache(max(source.shape[0], 1), fixture.head_dim, source.device)
    weight = torch.randn(
        fixture.head_dim,
        dtype=torch.float32,
        device=source.device,
    )

    def invoke() -> None:
        accepted = dsv4_triton.compress_norm_rope_store_bf16(
            source,
            positions,
            weight,
            cache.cache,
            loc,
            rms_norm_eps=1e-6,
            rotary_dim=ROTARY_DIM,
            base=PRODUCTION_COMPRESS_BASE,
            original_seq_len=PRODUCTION_ORIGINAL_SEQ_LEN,
            factor=PRODUCTION_ROPE_FACTOR,
            beta_fast=PRODUCTION_BETA_FAST,
            beta_slow=PRODUCTION_BETA_SLOW,
        )
        if not accepted:
            raise AssertionError("attention publication bridge rejected fixture")

    return invoke, {
        "operation": (
            "attention_publication_compact" if compact else "attention_publication_fixed"
        ),
        "component": component,
        "physical_rows": fixture.rows,
        "processed_rows": source.shape[0],
        "valid_publication_rows": int(boundary.sum().item()),
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "estimated_hbm_bytes_floor": source.numel() * 10,
        "launches": ["_compress_norm_rope_store_bf16_kernel"],
    }


def _indexer_publication(
    fixture: Fixture,
    *,
    compact: bool,
) -> tuple[Callable[[], None], dict[str, object]]:
    boundary = (fixture.positions + 1) % 4 == 0
    rows = int(boundary.sum().item()) if compact else fixture.rows
    source = torch.randn(
        rows,
        fixture.head_dim,
        dtype=torch.bfloat16,
        device=fixture.projected.device,
    )
    positions = fixture.positions[boundary] - 3 if compact else fixture.positions - 3
    if compact:
        loc = torch.arange(rows, dtype=torch.int64, device=source.device)
    else:
        loc = torch.where(
            boundary,
            torch.arange(fixture.rows, device=source.device),
            torch.full((fixture.rows,), -1, device=source.device),
        ).to(torch.int64)
    cache = FakeIndexerCache(max(rows, 1), fixture.head_dim, source.device)
    weight = torch.randn(
        fixture.head_dim,
        dtype=torch.float32,
        device=source.device,
    )

    def invoke() -> None:
        dsv4_reference.compress_norm_rope_store_fallback(
            cache,
            0,
            source,
            loc,
            positions=positions,
            norm_weight=weight,
            rms_norm_eps=1e-6,
            rotary_dim=ROTARY_DIM,
            base=10000.0,
            original_seq_len=4096,
            factor=2.0,
            beta_fast=32,
            beta_slow=1,
            cache_type="indexer",
            apply_hadamard=True,
        )

    return invoke, {
        "operation": (
            "indexer_norm_rope_hadamard_qat_publication_compact"
            if compact
            else "indexer_norm_rope_hadamard_qat_publication_fixed"
        ),
        "component": "c4_indexer",
        "physical_rows": fixture.rows,
        "processed_rows": rows,
        "valid_publication_rows": int(boundary.sum().item()),
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "estimated_hbm_bytes_floor": source.numel() * 12,
        "launches": [
            "PyTorch RMSNorm pointwise/reduction",
            "_apply_rotary_tail_kernel",
            "PyTorch seven-stage Hadamard",
            "_indexer_fp8_paged_quant_store_kernel",
        ],
    }


def _publication_inputs(
    rows: int,
    pattern: Pattern,
    scenario: str,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, _ = _positions(rows, 4, pattern, scenario, device)
    compressed_positions = (positions - 3).to(torch.int64).contiguous()
    valid = (positions + 1) % 4 == 0
    loc = torch.where(
        valid,
        torch.arange(rows, dtype=torch.int64, device=device),
        torch.full((rows,), -1, dtype=torch.int64, device=device),
    )
    if scenario == "page_boundary":
        valid_rows = torch.nonzero(valid, as_tuple=False).flatten()
        loc.fill_(-1)
        loc[valid_rows] = torch.arange(
            62,
            62 + valid_rows.numel(),
            dtype=torch.int64,
            device=device,
        )
    generator = torch.Generator(device=device).manual_seed(seed)
    source = torch.randn(
        rows,
        128,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    norm_weight = torch.randn(
        128,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    return source, compressed_positions, loc.contiguous(), norm_weight, valid


def _publication_cache(rows: int, loc: torch.Tensor) -> FakeIndexerCache:
    valid = loc >= 0
    slots = max(
        rows,
        int(loc[valid].max().item()) + 1 if bool(torch.any(valid)) else 1,
    )
    cache = FakeIndexerCache(slots, 128, loc.device)
    cache._bf16.fill_(-3)
    cache._packed.fill_(0xA5)
    return cache


def _launch_indexer_publication_reference(
    source: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    loc: torch.Tensor,
    cache: FakeIndexerCache,
) -> None:
    values = source.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    source.copy_((values * norm_weight.float()).to(source.dtype))
    dsv4_reference.apply_rotary_tail(
        source,
        positions,
        rotary_dim=64,
        base=PRODUCTION_COMPRESS_BASE,
        original_seq_len=PRODUCTION_ORIGINAL_SEQ_LEN,
        factor=PRODUCTION_ROPE_FACTOR,
        beta_fast=PRODUCTION_BETA_FAST,
        beta_slow=PRODUCTION_BETA_SLOW,
    )
    dsv4_reference.indexer_kv_hadamard_fallback(source)
    dsv4_reference.store_indexer_fp8_cache_fallback(cache, 0, source, loc)


def _launch_indexer_publication_candidate(
    source: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    loc: torch.Tensor,
    cache: FakeIndexerCache,
) -> None:
    dsv4_reference.compress_norm_rope_store_fallback(
        cache,
        0,
        source,
        loc,
        positions=positions,
        norm_weight=norm_weight,
        rms_norm_eps=1e-6,
        rotary_dim=64,
        base=PRODUCTION_COMPRESS_BASE,
        original_seq_len=PRODUCTION_ORIGINAL_SEQ_LEN,
        factor=PRODUCTION_ROPE_FACTOR,
        beta_fast=PRODUCTION_BETA_FAST,
        beta_slow=PRODUCTION_BETA_SLOW,
        cache_type="indexer",
        apply_hadamard=True,
    )


def _publication_exactness_case(
    rows: int,
    pattern: Pattern,
    scenario: str,
    *,
    seed: int,
    device: torch.device,
    publication_positions: list[int] | None = None,
    valid_mask: list[bool] | None = None,
    valid_locs: list[int] | None = None,
) -> dict[str, object]:
    if publication_positions is None:
        source, positions, loc, norm_weight, valid = _publication_inputs(
            rows,
            pattern,
            scenario,
            seed=seed,
            device=device,
        )
    else:
        if len(publication_positions) != rows or valid_mask is None:
            raise ValueError("Explicit publication positions require one validity per row.")
        if len(valid_mask) != rows:
            raise ValueError("Explicit validity mask must match publication positions.")
        generator = torch.Generator(device=device).manual_seed(seed)
        source = torch.randn(
            rows,
            128,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        norm_weight = torch.randn(
            128,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        positions = torch.tensor(
            publication_positions,
            dtype=torch.int64,
            device=device,
        )
        valid = torch.tensor(valid_mask, dtype=torch.bool, device=device)
        loc = torch.full((rows,), -1, dtype=torch.int64, device=device)
        if bool(torch.any(valid)):
            selected_locs = (
                valid_locs if valid_locs is not None else list(range(int(valid.sum().item())))
            )
            if len(selected_locs) != int(valid.sum().item()):
                raise ValueError("Explicit valid locations must match valid row count.")
            loc[valid] = torch.tensor(
                selected_locs,
                dtype=torch.int64,
                device=device,
            )
    reference_source = source.clone()
    candidate_source = source.clone()
    reference_cache = _publication_cache(rows, loc)
    candidate_cache = _publication_cache(rows, loc)
    _launch_indexer_publication_reference(
        reference_source,
        positions,
        norm_weight,
        loc,
        reference_cache,
    )
    _launch_indexer_publication_candidate(
        candidate_source,
        positions,
        norm_weight,
        loc,
        candidate_cache,
    )
    torch.cuda.synchronize()

    page_size = candidate_cache.indexer_fp8_page_size
    page_bytes = candidate_cache._packed.shape[-1]
    reference_values = reference_cache._packed.as_strided(
        (reference_cache._packed.shape[0], page_size, 128),
        (page_bytes, 128, 1),
    ).reshape(-1, 128)
    candidate_values = candidate_cache._packed.as_strided(
        (candidate_cache._packed.shape[0], page_size, 128),
        (page_bytes, 128, 1),
    ).reshape(-1, 128)
    reference_scales = reference_cache._packed.as_strided(
        (reference_cache._packed.shape[0], page_size, 4),
        (page_bytes, 4, 1),
        storage_offset=page_size * 128,
    ).reshape(-1, 4)
    candidate_scales = candidate_cache._packed.as_strided(
        (candidate_cache._packed.shape[0], page_size, 4),
        (page_bytes, 4, 1),
        storage_offset=page_size * 128,
    ).reshape(-1, 4)
    valid_locs = loc[valid]
    invalid_cache_mask = torch.ones(
        reference_values.shape[0],
        dtype=torch.bool,
        device=device,
    )
    invalid_cache_mask[valid_locs] = False
    invalid_locs = torch.nonzero(invalid_cache_mask, as_tuple=False).flatten()
    dequant_reference = dsv4_reference.dequantize_indexer_fp8_paged_cache_ref(
        reference_cache._packed,
        page_size=page_size,
        dim=128,
    )
    dequant_candidate = dsv4_reference.dequantize_indexer_fp8_paged_cache_ref(
        candidate_cache._packed,
        page_size=page_size,
        dim=128,
    )
    values_exact = torch.equal(
        reference_values.index_select(0, valid_locs),
        candidate_values.index_select(0, valid_locs),
    )
    scales_exact = torch.equal(
        reference_scales.index_select(0, valid_locs),
        candidate_scales.index_select(0, valid_locs),
    )
    dequant_exact = torch.equal(
        dequant_reference.index_select(0, valid_locs),
        dequant_candidate.index_select(0, valid_locs),
    )
    invalid_cache_canary = (
        bool(torch.all(candidate_values.index_select(0, invalid_locs) == 0xA5))
        and bool(torch.all(candidate_scales.index_select(0, invalid_locs) == 0xA5))
        if invalid_locs.numel()
        else True
    )
    valid_source_exact = torch.equal(
        reference_source[valid],
        candidate_source[valid],
    )
    invalid_source_untouched = torch.equal(source[~valid], candidate_source[~valid])
    packed_exact = torch.equal(reference_cache._packed, candidate_cache._packed)
    passed = all(
        (
            values_exact,
            scales_exact,
            dequant_exact,
            invalid_cache_canary,
            valid_source_exact,
            invalid_source_untouched,
            packed_exact,
        )
    )
    return {
        "rows": rows,
        "pattern": pattern,
        "scenario": scenario,
        "publication_positions": [int(value) for value in positions.tolist()],
        "publication_position_min": int(positions.min().item()),
        "publication_position_max": int(positions.max().item()),
        "valid_locations": [int(value) for value in loc[valid].tolist()],
        "valid_rows": int(valid.sum().item()),
        "fp8_values_bitwise_exact": values_exact,
        "fp8_scales_bitwise_exact": scales_exact,
        "packed_cache_bitwise_exact": packed_exact,
        "dequantized_values_bitwise_exact": dequant_exact,
        "invalid_cache_canary_untouched": invalid_cache_canary,
        "valid_source_mutation_bitwise_exact": valid_source_exact,
        "invalid_source_untouched": invalid_source_untouched,
        "source_alias_preserved": True,
        "passed": passed,
    }


def _publication_graph_probe(device: torch.device) -> dict[str, object]:
    rows = 128
    source, positions, loc, norm_weight, _ = _publication_inputs(
        rows,
        "natural_mixed",
        "matrix",
        seed=17600,
        device=device,
    )
    candidate_source = torch.empty_like(source)
    candidate_cache = _publication_cache(rows, loc)
    eager_source = source.clone()
    eager_cache = _publication_cache(rows, loc)
    high_positions = torch.tensor(
        HIGH_PUBLICATION_POSITIONS,
        dtype=torch.int64,
        device=device,
    )
    positions.copy_(
        high_positions[
            torch.arange(rows, dtype=torch.int64, device=device) % high_positions.numel()
        ]
    )
    dsv4_kernel.warmup_indexer_fp8_backend(
        device,
        base=PRODUCTION_COMPRESS_BASE,
        original_seq_len=PRODUCTION_ORIGINAL_SEQ_LEN,
        factor=PRODUCTION_ROPE_FACTOR,
        beta_fast=PRODUCTION_BETA_FAST,
        beta_slow=PRODUCTION_BETA_SLOW,
        page_size=64,
    )
    _launch_indexer_publication_candidate(
        eager_source,
        positions,
        norm_weight,
        loc,
        eager_cache,
    )
    torch.cuda.synchronize()

    def invoke() -> None:
        candidate_source.copy_(source)
        _launch_indexer_publication_candidate(
            candidate_source,
            positions,
            norm_weight,
            loc,
            candidate_cache,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    pointers = {
        "source": source.data_ptr(),
        "candidate_source": candidate_source.data_ptr(),
        "positions": positions.data_ptr(),
        "loc": loc.data_ptr(),
        "norm_weight": norm_weight.data_ptr(),
        "cache": candidate_cache._packed.data_ptr(),
    }
    primary = torch.empty_like(candidate_cache._packed)
    repeat = torch.empty_like(candidate_cache._packed)
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    graph.replay()
    primary.copy_(candidate_cache._packed)
    graph.replay()
    repeat.copy_(candidate_cache._packed)
    torch.cuda.synchronize()
    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()
    stable = pointers == {
        "source": source.data_ptr(),
        "candidate_source": candidate_source.data_ptr(),
        "positions": positions.data_ptr(),
        "loc": loc.data_ptr(),
        "norm_weight": norm_weight.data_ptr(),
        "cache": candidate_cache._packed.data_ptr(),
    }
    graph_eager_cache_exact = torch.equal(primary, eager_cache._packed)
    graph_eager_source_exact = torch.equal(candidate_source, eager_source)
    passed = (
        stable
        and torch.equal(primary, repeat)
        and graph_eager_cache_exact
        and graph_eager_source_exact
        and after_allocated == before_allocated
        and after_reserved == before_reserved
    )
    return {
        "status": "pass" if passed else "fail",
        "stable_pointers": stable,
        "primary_repeat_cache_bitwise": torch.equal(primary, repeat),
        "graph_eager_cache_bitwise": graph_eager_cache_exact,
        "graph_eager_source_bitwise": graph_eager_source_exact,
        "memory_allocated_delta_bytes": after_allocated - before_allocated,
        "memory_reserved_delta_bytes": after_reserved - before_reserved,
        "host_sync_in_replay": False,
        "pointers": pointers,
        "publication_position_min": int(positions.min().item()),
        "publication_position_max": int(positions.max().item()),
        "rope_parameters": {
            "rotary_dim": ROTARY_DIM,
            "base": PRODUCTION_COMPRESS_BASE,
            "original_seq_len": PRODUCTION_ORIGINAL_SEQ_LEN,
            "factor": PRODUCTION_ROPE_FACTOR,
            "beta_fast": PRODUCTION_BETA_FAST,
            "beta_slow": PRODUCTION_BETA_SLOW,
            "model_max_position": PRODUCTION_MAX_POSITION,
        },
    }


def _publication_backend_report(device: torch.device) -> dict[str, object]:
    decode_rows = [1, 4, 16, 64, 128]
    prefill_rows = [1024, 4096, 8192]
    patterns: list[Pattern] = ["no_boundary", "natural_mixed", "all_boundary"]
    cases = [
        _publication_exactness_case(
            rows,
            pattern,
            "matrix",
            seed=17700 + index,
            device=device,
        )
        for index, (rows, pattern) in enumerate(
            (rows, pattern) for rows in decode_rows + prefill_rows for pattern in patterns
        )
    ]
    for index, (scenario, rows) in enumerate(
        (
            ("ratio_boundary", 16),
            ("page_boundary", 16),
            ("continuation", 64),
            ("ragged", 128),
        )
    ):
        cases.append(
            _publication_exactness_case(
                rows,
                "natural_mixed",
                scenario,
                seed=17800 + index,
                device=device,
            )
        )

    high_position_cases = [
        (
            "high_no_valid_rows",
            [0, 4, 65532, 65536],
            [False, False, False, False],
            [],
        ),
        (
            "high_one_valid_row",
            [65540, 131068, 131072, 524284],
            [False, True, False, False],
            [7],
        ),
        (
            "high_natural_mixed_rows",
            [0, 4, 65532, 65536, 65540, 131068, 131072, 524284],
            [False, True, False, True, False, True, False, True],
            [0, 1, 63, 64],
        ),
        (
            "high_all_valid_rows",
            [524288, 1048568, 1048572],
            [True, True, True],
            [61, 62, 63],
        ),
        (
            "high_fp8_page_boundary_locations",
            [65532, 131068, 524284, 1048568],
            [True, True, True, True],
            [62, 63, 64, 65],
        ),
        (
            "high_ragged_independent_starts",
            [65536, 131072, 524288, 1048572, 65540, 131068, 524284, 1048568],
            [True, False, True, False, False, True, False, True],
            [3, 11, 64, 79],
        ),
    ]
    for index, (scenario, positions, valid_mask, valid_locs) in enumerate(high_position_cases):
        cases.append(
            _publication_exactness_case(
                len(positions),
                "natural_mixed",
                scenario,
                seed=17850 + index,
                device=device,
                publication_positions=positions,
                valid_mask=valid_mask,
                valid_locs=valid_locs,
            )
        )

    timings = []
    for index, (rows, pattern) in enumerate(
        (rows, pattern) for rows in decode_rows + prefill_rows for pattern in patterns
    ):
        source, positions, loc, norm_weight, _ = _publication_inputs(
            rows,
            pattern,
            "matrix",
            seed=17900 + index,
            device=device,
        )
        reference_source = source.clone()
        candidate_source = source.clone()
        reference_cache = _publication_cache(rows, loc)
        candidate_cache = _publication_cache(rows, loc)

        def reference() -> None:
            _launch_indexer_publication_reference(
                reference_source,
                positions,
                norm_weight,
                loc,
                reference_cache,
            )

        def candidate() -> None:
            _launch_indexer_publication_candidate(
                candidate_source,
                positions,
                norm_weight,
                loc,
                candidate_cache,
            )

        timing_fn = time_cuda_graph_replay if rows <= 128 else time_cuda
        reference_timing = timing_fn(reference, rows)
        candidate_timing = timing_fn(candidate, rows)
        reference_us = float(reference_timing["median_us"])
        candidate_us = float(candidate_timing["median_us"])
        timings.append(
            {
                "rows": rows,
                "pattern": pattern,
                "mode": "graph" if rows <= 128 else "eager",
                "reference": reference_timing,
                "candidate": candidate_timing,
                "gain_pct": 100.0 * (reference_us - candidate_us) / reference_us,
            }
        )

    source, positions, loc, norm_weight, _ = _publication_inputs(
        128,
        "natural_mixed",
        "matrix",
        seed=18000,
        device=device,
    )
    reference_cache = _publication_cache(128, loc)
    candidate_cache = _publication_cache(128, loc)
    reference_launches = profile_cuda_launches(
        lambda: _launch_indexer_publication_reference(
            source,
            positions,
            norm_weight,
            loc,
            reference_cache,
        )
    )
    candidate_launches = profile_cuda_launches(
        lambda: _launch_indexer_publication_candidate(
            source,
            positions,
            norm_weight,
            loc,
            candidate_cache,
        )
    )
    failures = [index for index, case in enumerate(cases) if not case["passed"]]
    graph_probe = _publication_graph_probe(device)
    return {
        "schema_version": 1,
        "scope": "Target 17.3 saved-input/no-weight publication backend",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "candidate": "three-kernel native cluster: CUDA RMSNorm, Triton RoPE, Triton Hadamard/QAT/store",
        "pdl": False,
        "production_contract": {
            "rotary_dim": ROTARY_DIM,
            "compress_rope_theta": PRODUCTION_COMPRESS_BASE,
            "original_max_position_embeddings": PRODUCTION_ORIGINAL_SEQ_LEN,
            "factor": PRODUCTION_ROPE_FACTOR,
            "beta_fast": PRODUCTION_BETA_FAST,
            "beta_slow": PRODUCTION_BETA_SLOW,
            "model_max_position_embeddings": PRODUCTION_MAX_POSITION,
            "publication_position_min": min(HIGH_PUBLICATION_POSITIONS),
            "publication_position_max": max(HIGH_PUBLICATION_POSITIONS),
        },
        "matrix": {
            "decode_rows": decode_rows,
            "prefill_rows": prefill_rows,
            "patterns": patterns,
            "high_position_scenarios": [scenario for scenario, *_ in high_position_cases],
            "cases_total": len(cases),
            "cases_passed": len(cases) - len(failures),
            "failure_indices": failures,
            "cases": cases,
        },
        "graph_probe": graph_probe,
        "timings": timings,
        "launch_census": {
            "reference": reference_launches,
            "candidate": candidate_launches,
        },
        "status": ("pass" if not failures and graph_probe["status"] == "pass" else "fail"),
    }


def _cache_inventory(root: str | None) -> dict[str, dict[str, int]]:
    if not root:
        return {}
    path = Path(root)
    if not path.exists():
        return {}
    return {
        str(item.relative_to(path)): {
            "size": item.stat().st_size,
            "mtime_ns": item.stat().st_mtime_ns,
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _production_warmup_probe(device: torch.device) -> dict[str, object]:
    triton_ops = dsv4_kernel._triton_dsv4_ops()
    stage_ms: dict[str, float] = {}
    original_native = dsv4_kernel._c4_indexer_rmsnorm_bf16_native
    original_lut = getattr(triton_ops, "warmup_indexer_fp8_lut", None)
    original_rope = triton_ops.indexer_rotary_tail_valid
    original_store = triton_ops.indexer_hadamard_fp8_paged_store

    def timed(name: str, function: Callable[..., object]) -> Callable[..., object]:
        def invoke(*args: object, **kwargs: object) -> object:
            started = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize(device)
            stage_ms[name] = (time.perf_counter() - started) * 1000.0
            return result

        return invoke

    dsv4_kernel._c4_indexer_rmsnorm_bf16_native = timed(
        "native_rmsnorm_compile_load_launch",
        original_native,
    )
    if callable(original_lut):
        triton_ops.warmup_indexer_fp8_lut = timed(
            "existing_indexer_lut_warmup",
            original_lut,
        )
    triton_ops.indexer_rotary_tail_valid = timed(
        "triton_rope_compile_load_launch",
        original_rope,
    )
    triton_ops.indexer_hadamard_fp8_paged_store = timed(
        "triton_hadamard_store_compile_load_launch",
        original_store,
    )
    warmup_started = time.perf_counter()
    try:
        dsv4_kernel.warmup_indexer_fp8_backend(
            device,
            base=PRODUCTION_COMPRESS_BASE,
            original_seq_len=PRODUCTION_ORIGINAL_SEQ_LEN,
            factor=PRODUCTION_ROPE_FACTOR,
            beta_fast=PRODUCTION_BETA_FAST,
            beta_slow=PRODUCTION_BETA_SLOW,
            page_size=64,
        )
        torch.cuda.synchronize(device)
    finally:
        dsv4_kernel._c4_indexer_rmsnorm_bf16_native = original_native
        if callable(original_lut):
            triton_ops.warmup_indexer_fp8_lut = original_lut
        triton_ops.indexer_rotary_tail_valid = original_rope
        triton_ops.indexer_hadamard_fp8_paged_store = original_store
    warmup_total_ms = (time.perf_counter() - warmup_started) * 1000.0

    rows = 128
    generator = torch.Generator(device=device).manual_seed(17310)
    original = torch.randn(
        rows,
        128,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    source = torch.empty_like(original)
    norm_weight = torch.randn(
        128,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    high_positions = torch.tensor(
        HIGH_PUBLICATION_POSITIONS,
        dtype=torch.int64,
        device=device,
    )
    positions = high_positions[
        torch.arange(rows, dtype=torch.int64, device=device) % high_positions.numel()
    ].contiguous()
    valid = torch.arange(rows, device=device) % 4 == 3
    loc = torch.where(
        valid,
        torch.arange(rows, dtype=torch.int64, device=device),
        torch.full((rows,), -1, dtype=torch.int64, device=device),
    )
    cache = _publication_cache(rows, loc)

    def invoke() -> None:
        source.copy_(original)
        _launch_indexer_publication_candidate(
            source,
            positions,
            norm_weight,
            loc,
            cache,
        )

    cache_roots = {
        "triton": os.environ.get("TRITON_CACHE_DIR"),
        "tvm_ffi": os.environ.get("TVM_FFI_CACHE_DIR"),
    }
    before_capture = {name: _cache_inventory(path) for name, path in cache_roots.items()}
    graph = torch.cuda.CUDAGraph()
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph):
        invoke()
    torch.cuda.synchronize(device)
    graph_capture_ms = (time.perf_counter() - capture_started) * 1000.0
    after_capture = {name: _cache_inventory(path) for name, path in cache_roots.items()}
    cache_changes: dict[str, dict[str, list[str]]] = {}
    for name in cache_roots:
        before = before_capture[name]
        after = after_capture[name]
        cache_changes[name] = {
            "created": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "modified": sorted(
                path for path in set(before) & set(after) if before[path] != after[path]
            ),
        }
    compilation_after_warmup = any(
        changes["created"] or changes["removed"] or changes["modified"]
        for changes in cache_changes.values()
    )

    primary = torch.empty_like(cache._packed)
    repeat = torch.empty_like(cache._packed)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    graph.replay()
    primary.copy_(cache._packed)
    graph.replay()
    repeat.copy_(cache._packed)
    torch.cuda.synchronize(device)
    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)
    replay_exact = torch.equal(primary, repeat)
    replay_allocated_delta = after_allocated - before_allocated
    replay_reserved_delta = after_reserved - before_reserved
    passed = (
        not compilation_after_warmup
        and replay_exact
        and replay_allocated_delta == 0
        and replay_reserved_delta == 0
    )
    return {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "scope": "single-GPU no-weight isolated-cache production warmup probe",
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "cache_roots": cache_roots,
        "production_specialization": {
            "rows": 1,
            "valid_rows": 1,
            "rotary_dim": ROTARY_DIM,
            "base": PRODUCTION_COMPRESS_BASE,
            "original_seq_len": PRODUCTION_ORIGINAL_SEQ_LEN,
            "factor": PRODUCTION_ROPE_FACTOR,
            "beta_fast": PRODUCTION_BETA_FAST,
            "beta_slow": PRODUCTION_BETA_SLOW,
            "page_size": 64,
        },
        "timing_ms": {
            "native_compile_load_launch": stage_ms.get("native_rmsnorm_compile_load_launch"),
            "triton_rope_compile_load_launch": stage_ms.get("triton_rope_compile_load_launch"),
            "triton_hadamard_store_compile_load_launch": stage_ms.get(
                "triton_hadamard_store_compile_load_launch"
            ),
            "triton_compile_load_launch_total": (
                stage_ms.get("triton_rope_compile_load_launch", 0.0)
                + stage_ms.get("triton_hadamard_store_compile_load_launch", 0.0)
            ),
            "existing_indexer_lut_warmup": stage_ms.get("existing_indexer_lut_warmup"),
            "warmup_total": warmup_total_ms,
            "subsequent_graph_capture": graph_capture_ms,
        },
        "compilation_occurred_after_warmup": compilation_after_warmup,
        "cache_changes_during_capture": cache_changes,
        "graph_replay": {
            "primary_repeat_bitwise": replay_exact,
            "memory_allocated_delta_bytes": replay_allocated_delta,
            "memory_reserved_delta_bytes": replay_reserved_delta,
            "host_sync_in_serving_hot_path": False,
        },
    }


def _producer_records(fixture: Fixture) -> list[dict[str, object]]:
    output = torch.empty(
        fixture.rows,
        fixture.head_dim,
        dtype=torch.bfloat16,
        device=fixture.projected.device,
    )
    records: list[dict[str, object]] = []
    boundaries = int(((fixture.positions + 1) % fixture.ratio == 0).sum().item())
    common = {
        "component": fixture.component,
        "rows": fixture.rows,
        "pattern": fixture.pattern,
        "valid_publication_rows": boundaries,
        "positions_min": int(fixture.positions.min().item()),
        "positions_max": int(fixture.positions.max().item()),
        "table_count": int(torch.unique(fixture.table_indices).numel()),
    }
    if fixture.ratio == 4:
        pool_timing = time_cuda(lambda: _launch_c4_pool(fixture, output), fixture.rows)
        records.append(
            {
                **common,
                "operation": "c4_pool_current",
                "grid_programs": fixture.rows
                * math.ceil(fixture.head_dim / min(triton.next_power_of_2(fixture.head_dim), 256)),
                "estimated_hbm_bytes_upper": fixture.rows * fixture.head_dim * 96,
                "launches": ["_c4_online_pool_kernel"],
                **pool_timing,
            }
        )
        state_timing = time_cuda(lambda: _launch_c4_state(fixture), fixture.rows)
        records.append(
            {
                **common,
                "operation": "c4_sequence_state_checkpoint_store",
                "grid_programs": fixture.rows * math.ceil(fixture.projected.shape[1] / 256),
                "estimated_hbm_bytes_floor": fixture.rows * fixture.head_dim * 32,
                "launches": ["_c4_online_state_store_kernel"],
                "separate_checkpoint_kernel": False,
                **state_timing,
            }
        )
        compact_plan = fixture.boundary_rows
        compact_output = torch.empty(
            max(compact_plan.numel(), 1),
            fixture.head_dim,
            dtype=torch.bfloat16,
            device=fixture.projected.device,
        )
        compact_timing = time_cuda(
            lambda: _launch_plan_pool(
                fixture,
                compact_plan,
                compact_output[: compact_plan.numel()],
            ),
            fixture.rows,
        )
        records.append(
            {
                **common,
                "operation": "c4_pool_debug_compact_plan",
                "plan_capacity": compact_plan.numel(),
                "grid_programs": compact_plan.numel()
                * math.ceil(fixture.head_dim / min(triton.next_power_of_2(fixture.head_dim), 256)),
                "launches": ["_c4_debug_plan_pool_kernel"],
                **compact_timing,
            }
        )
        fixed_plan = torch.empty(
            fixture.rows,
            dtype=torch.int32,
            device=fixture.projected.device,
        )
        fixed_output = torch.empty_like(output)

        def fixed_invoke() -> None:
            _build_fixed_plan(fixture, fixed_plan)
            _launch_plan_pool(fixture, fixed_plan, fixed_output)

        fixed_timing = time_cuda(fixed_invoke, fixture.rows)
        records.append(
            {
                **common,
                "operation": "c4_pool_debug_fixed_capacity_plan",
                "plan_capacity": fixture.rows,
                "grid_programs": fixture.rows
                * math.ceil(fixture.head_dim / min(triton.next_power_of_2(fixture.head_dim), 256)),
                "launches": [
                    "_build_c4_fixed_plan_kernel",
                    "_c4_debug_plan_pool_kernel",
                ],
                **fixed_timing,
            }
        )
    else:
        pool_timing = time_cuda(
            lambda: _launch_c128_pool(fixture, output),
            fixture.rows,
        )
        records.append(
            {
                **common,
                "operation": "c128_pool_current",
                "grid_programs": fixture.rows * fixture.head_dim,
                "non_boundary_program_behavior": "return before 128-source loads/reduction",
                "launches": ["_c128_online_pool_kernel"],
                **pool_timing,
            }
        )
        state_timing = time_cuda(lambda: _launch_c128_state(fixture), fixture.rows)
        records.append(
            {
                **common,
                "operation": "c128_state_store",
                "grid_programs": fixture.rows * math.ceil(fixture.projected.shape[1] / 256),
                "estimated_hbm_bytes_floor": fixture.rows * fixture.head_dim * 20,
                "launches": ["_c128_online_state_store_kernel"],
                **state_timing,
            }
        )

    if fixture.component == "c4_indexer":
        for compact in (False, True):
            invoke, metadata = _indexer_publication(fixture, compact=compact)
            if metadata["processed_rows"] == 0:
                timing = {
                    "median_us": 0.0,
                    "mad_us": 0.0,
                    "min_us": 0.0,
                    "max_us": 0.0,
                    "samples_us": [],
                    "warmup": 0,
                    "iterations_per_repeat": 0,
                    "repeats": 0,
                }
            else:
                timing = time_cuda(invoke, fixture.rows)
            records.append(
                {
                    **common,
                    **metadata,
                    **timing,
                }
            )
    else:
        for compact in (False, True):
            invoke, metadata = _attention_publication(
                fixture.component,
                fixture,
                compact=compact,
            )
            if metadata["processed_rows"] == 0:
                timing = {
                    "median_us": 0.0,
                    "mad_us": 0.0,
                    "min_us": 0.0,
                    "max_us": 0.0,
                    "samples_us": [],
                    "warmup": 0,
                    "iterations_per_repeat": 0,
                    "repeats": 0,
                }
            else:
                timing = time_cuda(invoke, fixture.rows)
            records.append({**common, **metadata, **timing})
    return records


def _parity_case(
    component: Literal["c4_attention", "c4_indexer"],
    rows: int,
    pattern: Pattern,
    scenario: str,
    device: torch.device,
) -> dict[str, object]:
    fixture = make_fixture(
        component,
        rows,
        pattern,
        scenario=scenario,
        parity=True,
        seed=17200,
        device=device,
    )
    assert fixture.checkpoint is not None
    initial_state = fixture.sequence_state.clone()
    initial_checkpoint = fixture.checkpoint.clone()

    baseline_output = torch.empty(
        rows,
        fixture.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    _launch_c4_pool(fixture, baseline_output)
    _launch_c4_state(fixture)
    baseline_state = fixture.sequence_state.clone()
    baseline_checkpoint = fixture.checkpoint.clone()

    compact_plan = fixture.boundary_rows
    compact_output = torch.empty(
        compact_plan.numel(),
        fixture.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    fixture.sequence_state.copy_(initial_state)
    fixture.checkpoint.copy_(initial_checkpoint)
    _launch_plan_pool(fixture, compact_plan, compact_output)
    _launch_c4_state(fixture)
    compact_state_exact = torch.equal(fixture.sequence_state, baseline_state)
    compact_checkpoint_exact = torch.equal(
        fixture.checkpoint,
        baseline_checkpoint,
    )
    compact_output_exact = torch.equal(
        compact_output,
        baseline_output.index_select(0, compact_plan.to(torch.int64)),
    )

    fixed_plan = torch.empty(rows, dtype=torch.int32, device=device)
    fixed_output = torch.empty_like(baseline_output)
    fixture.sequence_state.copy_(initial_state)
    fixture.checkpoint.copy_(initial_checkpoint)
    _build_fixed_plan(fixture, fixed_plan)
    _launch_plan_pool(fixture, fixed_plan, fixed_output)
    _launch_c4_state(fixture)
    fixed_state_exact = torch.equal(fixture.sequence_state, baseline_state)
    fixed_checkpoint_exact = torch.equal(
        fixture.checkpoint,
        baseline_checkpoint,
    )
    fixed_output_exact = torch.equal(fixed_output, baseline_output)
    torch.cuda.synchronize()
    passed = all(
        (
            compact_output_exact,
            compact_state_exact,
            compact_checkpoint_exact,
            fixed_output_exact,
            fixed_state_exact,
            fixed_checkpoint_exact,
        )
    )
    return {
        "component": component,
        "rows": rows,
        "pattern": pattern,
        "scenario": scenario,
        "boundary_rows": compact_plan.cpu().tolist(),
        "valid_publication_rows": compact_plan.numel(),
        "compact": {
            "output_bitwise_exact": compact_output_exact,
            "mutable_sequence_state_bitwise_exact": compact_state_exact,
            "checkpoint_bitwise_exact": compact_checkpoint_exact,
        },
        "fixed_capacity": {
            "output_bitwise_exact": fixed_output_exact,
            "mutable_sequence_state_bitwise_exact": fixed_state_exact,
            "checkpoint_bitwise_exact": fixed_checkpoint_exact,
            "capacity": rows,
            "invalid_entries": rows - compact_plan.numel(),
        },
        "passed": passed,
    }


def _graph_replay_probe(device: torch.device) -> dict[str, object]:
    fixture = make_fixture(
        "c4_attention",
        128,
        "natural_mixed",
        parity=True,
        seed=17300,
        device=device,
    )
    assert fixture.checkpoint is not None
    plan = torch.empty(128, dtype=torch.int32, device=device)
    output = torch.empty(128, fixture.head_dim, dtype=torch.bfloat16, device=device)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _build_fixed_plan(fixture, plan)
            _launch_plan_pool(fixture, plan, output)
            _launch_c4_state(fixture)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _build_fixed_plan(fixture, plan)
        _launch_plan_pool(fixture, plan, output)
        _launch_c4_state(fixture)
    pointers = {
        "plan": plan.data_ptr(),
        "output": output.data_ptr(),
        "state": fixture.sequence_state.data_ptr(),
        "checkpoint": fixture.checkpoint.data_ptr(),
    }
    primary = torch.empty_like(output)
    repeat = torch.empty_like(output)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    graph.replay()
    primary.copy_(output)
    graph.replay()
    repeat.copy_(output)
    torch.cuda.synchronize()
    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)
    stable = pointers == {
        "plan": plan.data_ptr(),
        "output": output.data_ptr(),
        "state": fixture.sequence_state.data_ptr(),
        "checkpoint": fixture.checkpoint.data_ptr(),
    }
    return {
        "status": "pass"
        if (
            stable
            and torch.equal(primary, repeat)
            and after_allocated == before_allocated
            and after_reserved == before_reserved
        )
        else "fail",
        "stable_pointers": stable,
        "primary_repeat_output_bitwise": torch.equal(primary, repeat),
        "memory_allocated_delta_bytes": after_allocated - before_allocated,
        "memory_reserved_delta_bytes": after_reserved - before_reserved,
        "replay_time_allocation_count": 0
        if after_allocated == before_allocated and after_reserved == before_reserved
        else None,
        "pointers": pointers,
    }


def _representative_launch_census(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {}
    for component in COMPONENTS:
        fixture = make_fixture(
            component,
            128,
            "natural_mixed",
            device=device,
        )
        output = torch.empty(
            128,
            fixture.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        if fixture.ratio == 4:
            result[f"{component}:pool"] = profile_cuda_launches(
                lambda fixture=fixture, output=output: _launch_c4_pool(
                    fixture,
                    output,
                )
            )
            result[f"{component}:state_checkpoint_store"] = profile_cuda_launches(
                lambda fixture=fixture: _launch_c4_state(fixture)
            )
        else:
            result[f"{component}:pool"] = profile_cuda_launches(
                lambda fixture=fixture, output=output: _launch_c128_pool(
                    fixture,
                    output,
                )
            )
            result[f"{component}:state_store"] = profile_cuda_launches(
                lambda fixture=fixture: _launch_c128_state(fixture)
            )
        if component == "c4_indexer":
            invoke, _ = _indexer_publication(fixture, compact=False)
            result[f"{component}:publication"] = profile_cuda_launches(invoke)
        else:
            invoke, _ = _attention_publication(
                component,
                fixture,
                compact=False,
            )
            result[f"{component}:publication"] = profile_cuda_launches(invoke)
    return result


def _graph_decode_owner_census(device: torch.device) -> dict[str, object]:
    records: list[dict[str, object]] = []
    rows_values = [1, 4, 16, 64, 128]
    patterns: list[Pattern] = [
        "no_boundary",
        "natural_mixed",
        "all_boundary",
    ]
    for component in COMPONENTS:
        for rows in rows_values:
            projected_dim = COMPONENTS[component]["projected_dim"]
            x = torch.randn(
                rows,
                HIDDEN_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            weight = torch.randn(
                projected_dim,
                HIDDEN_DIM,
                dtype=torch.bfloat16,
                device=device,
            )

            def projection() -> None:
                out = dsv4_reference.linear_bf16_fp32_fallback(x, weight)
                if out.dtype is not torch.float32:
                    raise AssertionError(out.dtype)

            records.append(
                {
                    "component": component,
                    "rows": rows,
                    "operation": "fp32_projection",
                    **time_cuda_graph_replay(projection, rows),
                }
            )
            for pattern in patterns:
                fixture = make_fixture(
                    component,
                    rows,
                    pattern,
                    device=device,
                )
                output = torch.empty(
                    rows,
                    fixture.head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                boundaries = int(((fixture.positions + 1) % fixture.ratio == 0).sum().item())
                common = {
                    "component": component,
                    "rows": rows,
                    "pattern": pattern,
                    "valid_publication_rows": boundaries,
                }
                if fixture.ratio == 4:
                    records.append(
                        {
                            **common,
                            "operation": "c4_pool_current",
                            **time_cuda_graph_replay(
                                lambda fixture=fixture, output=output: _launch_c4_pool(
                                    fixture,
                                    output,
                                ),
                                rows,
                            ),
                        }
                    )
                    records.append(
                        {
                            **common,
                            "operation": "c4_sequence_state_checkpoint_store",
                            **time_cuda_graph_replay(
                                lambda fixture=fixture: _launch_c4_state(fixture),
                                rows,
                            ),
                        }
                    )
                    fixed_plan = torch.empty(
                        rows,
                        dtype=torch.int32,
                        device=device,
                    )
                    fixed_output = torch.empty_like(output)

                    def fixed_pool() -> None:
                        _build_fixed_plan(fixture, fixed_plan)
                        _launch_plan_pool(fixture, fixed_plan, fixed_output)

                    records.append(
                        {
                            **common,
                            "operation": "c4_pool_debug_fixed_capacity_plan",
                            **time_cuda_graph_replay(fixed_pool, rows),
                        }
                    )
                else:
                    records.append(
                        {
                            **common,
                            "operation": "c128_pool_current",
                            **time_cuda_graph_replay(
                                lambda fixture=fixture, output=output: _launch_c128_pool(
                                    fixture,
                                    output,
                                ),
                                rows,
                            ),
                        }
                    )
                    records.append(
                        {
                            **common,
                            "operation": "c128_state_store",
                            **time_cuda_graph_replay(
                                lambda fixture=fixture: _launch_c128_state(fixture),
                                rows,
                            ),
                        }
                    )
                if component == "c4_indexer":
                    publication, _ = _indexer_publication(
                        fixture,
                        compact=False,
                    )
                    operation = "indexer_norm_rope_hadamard_qat_publication_fixed"
                else:
                    publication, _ = _attention_publication(
                        component,
                        fixture,
                        compact=False,
                    )
                    operation = "attention_publication_fixed"
                records.append(
                    {
                        **common,
                        "operation": operation,
                        **time_cuda_graph_replay(publication, rows),
                    }
                )
    replay_deltas = [
        (
            int(record["replay_memory_allocated_delta_bytes"]),
            int(record["replay_memory_reserved_delta_bytes"]),
        )
        for record in records
    ]
    return {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "rows": rows_values,
        "patterns": patterns,
        "records": records,
        "all_replays_allocation_free": all(
            allocated == 0 and reserved == 0 for allocated, reserved in replay_deltas
        ),
    }


def _phase_average(records: list[dict[str, object]], rows: int) -> dict[str, object]:
    by_key: dict[tuple[str, str, str], float] = {}
    for record in records:
        if int(record.get("rows", record.get("physical_rows", -1))) != rows:
            continue
        component = str(record["component"])
        operation = str(record["operation"])
        pattern = str(record.get("pattern", ""))
        by_key[(component, operation, pattern)] = float(record["median_us"])

    def value(component: str, operation: str, pattern: str) -> float:
        return by_key.get((component, operation, pattern), 0.0)

    c4_patterns = {"no_boundary": 0.75, "all_boundary": 0.25}
    c128_patterns = {"no_boundary": 127 / 128, "all_boundary": 1 / 128}
    layer_counts = {
        "c4_attention": 21,
        "c4_indexer": 21,
        "c128_attention": 20,
    }
    operations: dict[str, float] = {}
    for component in ("c4_attention", "c4_indexer"):
        for operation in (
            "c4_pool_current",
            "c4_sequence_state_checkpoint_store",
            "c4_pool_debug_fixed_capacity_plan",
            "attention_publication_fixed",
            "indexer_norm_rope_hadamard_qat_publication_fixed",
        ):
            weighted = sum(
                weight * value(component, operation, pattern)
                for pattern, weight in c4_patterns.items()
            )
            if weighted:
                operations[f"{component}:{operation}"] = weighted * layer_counts[component]
    for operation in (
        "c128_pool_current",
        "c128_state_store",
        "attention_publication_fixed",
    ):
        weighted = sum(
            weight * value("c128_attention", operation, pattern)
            for pattern, weight in c128_patterns.items()
        )
        if weighted:
            operations[f"c128_attention:{operation}"] = weighted * layer_counts["c128_attention"]
    projection: dict[str, float] = {}
    for component, count in layer_counts.items():
        candidates = [
            record
            for record in records
            if record.get("operation") == "fp32_projection"
            and record.get("component") == component
            and record.get("rows") == rows
        ]
        if candidates:
            projection[component] = float(candidates[0]["median_us"]) * count
    projection_total = sum(projection.values())
    current_total = projection_total + sum(
        value_us for key, value_us in operations.items() if "debug_fixed" not in key
    )
    by_owner = {
        "projection": projection_total,
        "c4_attention_non_projection": sum(
            value_us
            for key, value_us in operations.items()
            if key.startswith("c4_attention:") and "debug_fixed" not in key
        ),
        "c4_indexer_non_projection": sum(
            value_us
            for key, value_us in operations.items()
            if key.startswith("c4_indexer:") and "debug_fixed" not in key
        ),
        "c128_non_projection": sum(
            value_us for key, value_us in operations.items() if key.startswith("c128_attention:")
        ),
    }
    return {
        "rows": rows,
        "closed_batch_phase_weights": {
            "c4": c4_patterns,
            "c128": c128_patterns,
        },
        "layer_counts": layer_counts,
        "projection_by_component_us": projection,
        "operation_layer_aggregate_us": operations,
        "owner_totals_us": by_owner,
        "compressor_cluster_total_us": current_total,
        "owner_percent_of_cluster": {
            owner: 100.0 * value_us / current_total for owner, value_us in by_owner.items()
        },
        "largest_owner": max(by_owner, key=by_owner.get),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise RuntimeError("Target 17.1 census requires SM80")

    decode_rows = [1, 4, 16, 64, 128]
    prefill_rows = [1024, 4096, 8192]
    matrix_rows = decode_rows + prefill_rows
    patterns: list[Pattern] = [
        "no_boundary",
        "natural_mixed",
        "all_boundary",
    ]
    records: list[dict[str, object]] = []
    for component in COMPONENTS:
        for rows in matrix_rows:
            records.append(_projection_case(component, rows, device))
            for pattern in patterns:
                fixture = make_fixture(
                    component,
                    rows,
                    pattern,
                    device=device,
                )
                records.extend(_producer_records(fixture))

    parity_cases: list[dict[str, object]] = []
    for component in ("c4_attention", "c4_indexer"):
        for rows in decode_rows:
            for pattern in patterns:
                parity_cases.append(_parity_case(component, rows, pattern, "matrix", device))
        for rows in prefill_rows:
            parity_cases.append(
                _parity_case(
                    component,
                    rows,
                    "natural_mixed",
                    "matrix",
                    device,
                )
            )
        for scenario, rows in (
            ("ratio_boundary", 16),
            ("page_boundary", 16),
            ("continuation", 64),
            ("ragged", 128),
        ):
            parity_cases.append(
                _parity_case(
                    component,
                    rows,
                    "natural_mixed",
                    scenario,
                    device,
                )
            )
    graph_probe = _graph_replay_probe(device)
    parity_failures = [index for index, case in enumerate(parity_cases) if not case["passed"]]
    report = {
        "schema_version": 1,
        "created_unix": time.time(),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "scope": "saved-input/no-weight; debug counterfactuals are not production reachable",
        "matrix_contract": {
            "decode_rows": decode_rows,
            "prefill_rows": prefill_rows,
            "patterns": patterns,
            "special_scenarios": [
                "ratio_boundary",
                "page_boundary",
                "continuation",
                "ragged",
            ],
        },
        "records": records,
        "closed_batch_m128_phase_average": _phase_average(records, 128),
        "counterfactual_parity": {
            "status": "pass" if not parity_failures else "fail",
            "cases_total": len(parity_cases),
            "cases_passed": len(parity_cases) - len(parity_failures),
            "failure_indices": parity_failures,
            "cases": parity_cases,
        },
        "graph_replay_probe": graph_probe,
        "representative_m128_launch_census": _representative_launch_census(device),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-census-only", action="store_true")
    parser.add_argument("--graph-decode-census-only", action="store_true")
    parser.add_argument("--c4-publication-backend-only", action="store_true")
    parser.add_argument("--production-warmup-probe-only", action="store_true")
    args = parser.parse_args()
    if args.production_warmup_probe_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda")
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise RuntimeError("Target 17.31 warmup probe requires SM80")
        report = _production_warmup_probe(device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": report["status"],
                    "warmup_total_ms": report["timing_ms"]["warmup_total"],
                    "capture_ms": report["timing_ms"]["subsequent_graph_capture"],
                    "compilation_after_warmup": report["compilation_occurred_after_warmup"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "pass" else 1
    if args.c4_publication_backend_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda")
        if torch.cuda.get_device_capability(device) != (8, 0):
            raise RuntimeError("Target 17.3 publication backend requires SM80")
        report = _publication_backend_report(device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": report["status"],
                    "cases_passed": report["matrix"]["cases_passed"],
                    "cases_total": report["matrix"]["cases_total"],
                    "graph": report["graph_probe"]["status"],
                    "candidate_launches": report["launch_census"]["candidate"][
                        "cuda_kernel_launches"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["status"] == "pass" else 1
    if args.graph_decode_census_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        report = _graph_decode_owner_census(torch.device("cuda"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "records": len(report["records"]),
                    "all_replays_allocation_free": report["all_replays_allocation_free"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["all_replays_allocation_free"] else 1
    if args.launch_census_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        report = {
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "representative_m128_launch_census": _representative_launch_census(
                torch.device("cuda")
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(report["records"]),
                "parity": report["counterfactual_parity"]["status"],
                "graph": report["graph_replay_probe"]["status"],
                "m128": report["closed_batch_m128_phase_average"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if (
            report["counterfactual_parity"]["status"] == "pass"
            and report["graph_replay_probe"]["status"] == "pass"
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

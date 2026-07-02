#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402
from minisgl.utils import div_ceil  # noqa: E402


ALIGNMENT = 64


def _pad_last_dim(x: torch.Tensor, multiple: int = ALIGNMENT, value: int = -1) -> torch.Tensor:
    size = x.shape[-1]
    target_size = div_ceil(size, multiple) * multiple
    if target_size == size:
        return x
    out = torch.full((*x.shape[:-1], target_size), value, dtype=x.dtype, device=x.device)
    out[..., :size] = x
    return out


def _gather_full_locs(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    valid = logical_positions >= 0
    clamped = logical_positions.clamp_min(0).to(torch.long)
    rows = table_indices.to(torch.long)[:, None].expand_as(clamped)
    out = ctx_page_table[rows, clamped].to(torch.int32)
    return torch.where(valid, out, torch.full_like(out, -1))


def _compressed_raw_to_full_locs(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    raw_positions = raw_indices * ratio + (ratio - 1)
    return _gather_full_locs(ctx_page_table, table_indices, raw_positions)


def _make_sparse_compressed_indices(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    lengths: torch.Tensor,
    ratio: int,
    index_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(index_topk, 1)
    raw = torch.full((lengths.numel(), width), -1, dtype=torch.int32, device=lengths.device)
    for row, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        start = max(0, int(length) - index_topk)
        values = torch.arange(start, int(length), dtype=torch.int32, device=lengths.device)
        raw[row, : values.numel()] = values
    full = _compressed_raw_to_full_locs(ctx_page_table, table_indices, raw, ratio)
    page = torch.where(full >= 0, full.div(ratio, rounding_mode="floor"), full)
    return (
        _pad_last_dim(raw, value=-1),
        _pad_last_dim(page.to(torch.int32), value=-1),
        _pad_last_dim(full.to(torch.int32), value=-1),
    )


def _make_all_compressed_indices(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    lengths: torch.Tensor,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(int(lengths.max().item()) if lengths.numel() else 0, 1)
    raw = torch.full((lengths.numel(), width), -1, dtype=torch.int32, device=lengths.device)
    for row, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        values = torch.arange(int(length), dtype=torch.int32, device=lengths.device)
        raw[row, : values.numel()] = values
    full = _compressed_raw_to_full_locs(ctx_page_table, table_indices, raw, ratio)
    page = torch.where(full >= 0, full.div(ratio, rounding_mode="floor"), full)
    return (
        _pad_last_dim(raw, value=-1),
        _pad_last_dim(page.to(torch.int32), value=-1),
        _pad_last_dim(full.to(torch.int32), value=-1),
    )


def oracle(
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    *,
    page_size: int,
    max_seqlen_k: int,
    window_size: int,
    index_topk: int,
) -> dict[str, torch.Tensor]:
    seq_lens = positions + 1
    offsets = torch.arange(
        0,
        max(max_seqlen_k, 1),
        page_size,
        dtype=torch.long,
        device=positions.device,
    )
    rows = table_indices.to(torch.long)
    page_table = ctx_page_table[rows[:, None], offsets[None, :]]
    page_table = torch.where(
        page_table >= 0,
        page_table.div(page_size, rounding_mode="floor"),
        page_table,
    ).to(torch.int32)
    swa_offsets = (
        positions[:, None]
        - torch.arange(window_size, dtype=torch.int32, device=positions.device)[None, :]
    )
    swa_page_indices = _pad_last_dim(
        _gather_full_locs(ctx_page_table, table_indices, swa_offsets),
        value=-1,
    )
    swa_topk_lengths = torch.clamp(seq_lens, max=window_size)

    c4_topk_lengths_raw = torch.div(seq_lens, 4, rounding_mode="floor")
    c4_topk_lengths_clamp1 = c4_topk_lengths_raw.clamp_min(1)
    c4_sparse_topk_lengths = c4_topk_lengths_raw.clamp(min=0, max=index_topk)
    c4_sparse_raw_indices, c4_sparse_page_indices, c4_sparse_full_indices = (
        _make_sparse_compressed_indices(
            ctx_page_table,
            table_indices,
            c4_topk_lengths_raw,
            4,
            index_topk,
        )
    )

    c128_lengths_raw = torch.div(seq_lens, 128, rounding_mode="floor")
    c128_topk_lengths_clamp1 = c128_lengths_raw.clamp_min(1)
    c128_raw_indices, c128_page_indices, c128_full_indices = _make_all_compressed_indices(
        ctx_page_table,
        table_indices,
        c128_lengths_raw,
        128,
    )
    return {
        "page_table": page_table,
        "swa_page_indices": swa_page_indices,
        "swa_topk_lengths": swa_topk_lengths,
        "c4_topk_lengths_raw": c4_topk_lengths_raw,
        "c4_topk_lengths_clamp1": c4_topk_lengths_clamp1,
        "c4_sparse_topk_lengths": c4_sparse_topk_lengths,
        "c4_sparse_raw_indices": c4_sparse_raw_indices,
        "c4_sparse_page_indices": c4_sparse_page_indices,
        "c4_sparse_full_indices": c4_sparse_full_indices,
        "c128_topk_lengths_clamp1": c128_topk_lengths_clamp1,
        "c128_raw_indices": c128_raw_indices,
        "c128_page_indices": c128_page_indices,
        "c128_full_indices": c128_full_indices,
    }


def make_inputs(
    *,
    batch_size: int,
    max_seq_len: int,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    max_seq_len = div_ceil(max_seq_len, page_size) * page_size
    num_reqs = max(batch_size + 2, 8)
    ctx_page_table = torch.empty((num_reqs, max_seq_len), dtype=torch.int32, device=device)
    base = torch.arange(max_seq_len, dtype=torch.int32, device=device)
    for row in range(num_reqs):
        ctx_page_table[row] = row * max_seq_len + base
    positions = torch.arange(
        max_seq_len - batch_size,
        max_seq_len,
        dtype=torch.int32,
        device=device,
    )
    table_indices = torch.arange(batch_size, dtype=torch.int32, device=device) % num_reqs
    return ctx_page_table, table_indices, positions, int(positions.max().item()) + 1


def time_cuda(fn, repeats: int) -> tuple[float, list[float]]:
    times = []
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return statistics.mean(times), times


def run_case(
    *,
    batch_size: int,
    max_seq_len: int,
    page_size: int,
    window_size: int,
    index_topk: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    ctx_page_table, table_indices, positions, max_seqlen_k = make_inputs(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        device=device,
    )
    ref = oracle(
        ctx_page_table,
        table_indices,
        positions,
        page_size=page_size,
        max_seqlen_k=max_seqlen_k,
        window_size=window_size,
        index_topk=index_topk,
    )
    out = dsv4_kernel.decode_metadata_deforest_fallback(
        ctx_page_table,
        table_indices,
        positions,
        page_size=page_size,
        max_seqlen_k=max_seqlen_k,
        window_size=window_size,
        index_topk=index_topk,
        alignment=ALIGNMENT,
    )
    if out is None:
        raise RuntimeError("decode_metadata_deforest_fallback returned None")
    actual = out.__dict__
    mismatches = []
    for name, expected in ref.items():
        got = actual[name]
        if expected.shape != got.shape or not torch.equal(expected, got):
            mismatches.append(
                {
                    "name": name,
                    "expected_shape": list(expected.shape),
                    "actual_shape": list(got.shape),
                    "max_abs_diff": None
                    if expected.shape != got.shape
                    else int((expected - got).abs().max().item()),
                }
            )
    old_mean, old_times = time_cuda(
        lambda: oracle(
            ctx_page_table,
            table_indices,
            positions,
            page_size=page_size,
            max_seqlen_k=max_seqlen_k,
            window_size=window_size,
            index_topk=index_topk,
        ),
        repeats,
    )
    new_mean, new_times = time_cuda(
        lambda: dsv4_kernel.decode_metadata_deforest_fallback(
            ctx_page_table,
            table_indices,
            positions,
            page_size=page_size,
            max_seqlen_k=max_seqlen_k,
            window_size=window_size,
            index_topk=index_topk,
            alignment=ALIGNMENT,
        ),
        repeats,
    )
    return {
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "max_seqlen_k": max_seqlen_k,
        "page_size": page_size,
        "window_size": window_size,
        "index_topk": index_topk,
        "equal": not mismatches,
        "mismatches": mismatches,
        "old_mean_s": old_mean,
        "new_mean_s": new_mean,
        "speedup": old_mean / new_mean if new_mean > 0 else None,
        "old_times_s": old_times,
        "new_times_s": new_times,
        "sentinel_policy": "exact -1 padding equality required; no alternate sentinel accepted",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Decode Metadata Deforest Microbench",
        "",
        f"- device: `{payload['device']}`",
        f"- repeats: `{payload['repeats']}`",
        f"- all_equal: `{payload['all_equal']}`",
        "- sentinel policy: exact `-1` padding equality; no tolerated differences observed.",
        "",
        "| BS | Max Seq | Old us | New us | Speedup | Equal |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in payload["cases"]:
        lines.append(
            f"| {case['batch_size']} | {case['max_seq_len']} | "
            f"{case['old_mean_s'] * 1e6:.2f} | {case['new_mean_s'] * 1e6:.2f} | "
            f"{case['speedup']:.2f}x | `{case['equal']}` |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--index-topk", type=int, default=512)
    args = parser.parse_args()

    os.environ[dsv4_kernel.DSV4_SM80_DECODE_METADATA_DEFOREST_TOGGLE] = "1"
    if not torch.cuda.is_available():
        payload = {
            "status": "skipped",
            "reason": "CUDA is not available",
            "cases": [],
            "all_equal": False,
            "device": "cpu",
            "repeats": args.repeats,
        }
    else:
        device = torch.device("cuda")
        cases = []
        for batch_size in (1, 2, 4):
            for max_seq_len in (128, 4096, 5120):
                cases.append(
                    run_case(
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                        page_size=args.page_size,
                        window_size=args.window_size,
                        index_topk=args.index_topk,
                        repeats=args.repeats,
                        device=device,
                    )
                )
        payload = {
            "status": "pass" if all(case["equal"] for case in cases) else "fail",
            "device": str(device),
            "repeats": args.repeats,
            "cases": cases,
            "all_equal": all(case["equal"] for case in cases),
        }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(payload) + "\n")
    print(render_markdown(payload))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

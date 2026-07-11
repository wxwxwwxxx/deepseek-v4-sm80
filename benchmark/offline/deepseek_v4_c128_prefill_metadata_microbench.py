from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import triton

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402

DEFAULT_ROWS = (1, 16, 1024, 8192)
DEFAULT_WIDTHS = (512, 2048, 4096, 5760, 8192)
PATTERNS = ("uniform_valid", "ragged_invalid")


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _make_inputs(
    rows: int,
    width: int,
    *,
    pattern: str,
    component_page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    table_width = max((width + component_page_size - 1) // component_page_size, 1)
    page_table = torch.arange(
        rows * table_width,
        dtype=torch.int32,
        device=device,
    ).reshape(rows, table_width)
    if pattern == "uniform_valid":
        lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    elif pattern == "ragged_invalid":
        lengths = (torch.arange(rows, dtype=torch.int32, device=device) * 509 + 129) % (width + 1)
        boundary = (0, 1, 2, 3, 127, 128, 129, max(width - 1, 0), width)
        count = min(rows, len(boundary))
        lengths[:count].copy_(
            torch.tensor(boundary[:count], dtype=torch.int32, device=device).clamp(max=width)
        )
        # Invalid physical component pages exercise both active holes and tail pages.
        page_table[:, 7::29] = -1
        if rows > 1:
            page_table[1::11, 0] = -1
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return page_table, lengths


def _oracle(
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    width: int,
    component_page_size: int,
) -> torch.Tensor:
    rows = lengths.numel()
    cols = torch.arange(width, dtype=torch.long, device=page_table.device)
    logical_pages = cols.div(component_page_size, rounding_mode="floor")
    page_in_table = logical_pages < page_table.shape[1]
    safe_pages = logical_pages.clamp(max=max(page_table.shape[1] - 1, 0))
    pages = torch.gather(page_table, 1, safe_pages[None, :].expand(rows, -1))
    locs = pages * component_page_size + (cols % component_page_size).to(torch.int32)
    valid = (cols[None, :] < lengths[:, None]) & page_in_table[None, :] & (pages >= 0)
    return torch.where(valid, locs, torch.full_like(locs, -1))


def _exact_mismatches(
    output: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    width: int,
    component_page_size: int,
    full_oracle_max_rows: int,
    sampled_oracle_rows: int,
) -> tuple[int, int, str]:
    rows = output.shape[0]
    if rows <= full_oracle_max_rows:
        selected_output = output
        selected_table = page_table
        selected_lengths = lengths
        mode = "full"
    else:
        compare_rows = min(rows, sampled_oracle_rows)
        row_ids = torch.linspace(
            0,
            rows - 1,
            compare_rows,
            dtype=torch.float64,
            device=output.device,
        ).to(torch.long)
        selected_output = output.index_select(0, row_ids)
        selected_table = page_table.index_select(0, row_ids)
        selected_lengths = lengths.index_select(0, row_ids)
        mode = f"sampled_rows_{compare_rows}"
    expected = _oracle(
        selected_table,
        selected_lengths,
        width=width,
        component_page_size=component_page_size,
    )
    mismatches = int(torch.count_nonzero(selected_output != expected).item())
    return mismatches, int(selected_output.shape[0]), mode


def _time_kernel(
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    *,
    width: int,
    component_page_size: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        result = dsv4_kernel.c128_prefill_page_indices_one_surface(
            page_table,
            lengths,
            width=width,
            component_page_size=component_page_size,
            out=output,
        )
        if result is None:
            raise RuntimeError("native C128 prefill helper was unavailable during warmup")
    torch.cuda.synchronize(output.device)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = dsv4_kernel.c128_prefill_page_indices_one_surface(
            page_table,
            lengths,
            width=width,
            component_page_size=component_page_size,
            out=output,
        )
        if result is None:
            raise RuntimeError("native C128 prefill helper was unavailable during timing")
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def run_case(
    rows: int,
    width: int,
    *,
    pattern: str,
    full_page_size: int,
    compress_ratio: int,
    warmup: int,
    repeats: int,
    full_oracle_max_rows: int,
    sampled_oracle_rows: int,
    device: torch.device,
) -> dict[str, Any]:
    component_page_size = max((full_page_size + compress_ratio - 1) // compress_ratio, 1)
    page_table, lengths = _make_inputs(
        rows,
        width,
        pattern=pattern,
        component_page_size=component_page_size,
        device=device,
    )

    # Compile this width/table-width specialization before memory accounting.
    scratch = torch.empty((1, width), dtype=torch.int32, device=device)
    compiled = dsv4_kernel.c128_prefill_page_indices_one_surface(
        page_table[:1],
        lengths[:1],
        width=width,
        component_page_size=component_page_size,
        out=scratch,
    )
    if compiled is None:
        raise RuntimeError("native C128 prefill helper is unavailable")
    torch.cuda.synchronize(device)
    del scratch, compiled
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))
    free_before, _ = torch.cuda.mem_get_info(device)
    backend: list[str] = []
    output = dsv4_kernel.c128_prefill_page_indices_one_surface(
        page_table,
        lengths,
        width=width,
        component_page_size=component_page_size,
        _backend=backend,
    )
    if output is None:
        raise RuntimeError("native C128 prefill helper is unavailable")
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    output_bytes = int(output.numel() * output.element_size())

    samples = _time_kernel(
        page_table,
        lengths,
        output,
        width=width,
        component_page_size=component_page_size,
        warmup=warmup,
        repeats=repeats,
    )
    mismatch_count, oracle_rows, oracle_mode = _exact_mismatches(
        output,
        page_table,
        lengths,
        width=width,
        component_page_size=component_page_size,
        full_oracle_max_rows=full_oracle_max_rows,
        sampled_oracle_rows=sampled_oracle_rows,
    )
    result = {
        "rows": rows,
        "width": width,
        "pattern": pattern,
        "full_page_size": full_page_size,
        "compress_ratio": compress_ratio,
        "component_page_size": component_page_size,
        "component_table_shape": list(page_table.shape),
        "component_table_bytes": int(page_table.numel() * page_table.element_size()),
        "lengths_bytes": int(lengths.numel() * lengths.element_size()),
        "backend": backend[0] if backend else "unknown",
        "kernel_launch_count": 1,
        "runtime_ms_median": statistics.median(samples),
        "runtime_ms_min": min(samples),
        "runtime_ms_max": max(samples),
        "runtime_ms_samples": samples,
        "output_bytes": output_bytes,
        "peak_allocated_delta_bytes": peak_allocated - allocated_before,
        "peak_reserved_delta_bytes": peak_reserved - reserved_before,
        "driver_free_delta_bytes": int(free_before - free_after),
        "temporary_bytes_beyond_output": max(
            0,
            peak_allocated - allocated_before - output_bytes,
        ),
        "full_matrix_int64_temporary_bytes": 0,
        "oracle_mode": oracle_mode,
        "oracle_rows": oracle_rows,
        "mismatch_count": mismatch_count,
    }
    del output, page_table, lengths
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-weight C128 eager-prefill one-surface metadata microbench"
    )
    parser.add_argument("--rows", default=",".join(map(str, DEFAULT_ROWS)))
    parser.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    parser.add_argument("--patterns", default=",".join(PATTERNS))
    parser.add_argument("--full-page-size", type=int, default=256)
    parser.add_argument("--compress-ratio", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--full-oracle-max-rows", type=int, default=1024)
    parser.add_argument("--sampled-oracle-rows", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (8, 0):
        raise SystemExit("This TARGET 12.59 microbench requires an sm80 CUDA device")

    rows_values = _csv_ints(args.rows)
    widths = _csv_ints(args.widths)
    patterns = tuple(item.strip() for item in args.patterns.split(",") if item.strip())
    results = []
    for rows in rows_values:
        for width in widths:
            for pattern in patterns:
                row = run_case(
                    rows,
                    width,
                    pattern=pattern,
                    full_page_size=args.full_page_size,
                    compress_ratio=args.compress_ratio,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    full_oracle_max_rows=args.full_oracle_max_rows,
                    sampled_oracle_rows=args.sampled_oracle_rows,
                    device=device,
                )
                results.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "config": {
            "rows": list(rows_values),
            "widths": list(widths),
            "patterns": list(patterns),
            "full_page_size": args.full_page_size,
            "compress_ratio": args.compress_ratio,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "full_oracle_max_rows": args.full_oracle_max_rows,
            "sampled_oracle_rows": args.sampled_oracle_rows,
        },
        "results": results,
        "total_mismatch_count": sum(row["mismatch_count"] for row in results),
        "max_temporary_bytes_beyond_output": max(
            row["temporary_bytes_beyond_output"] for row in results
        ),
        "max_full_matrix_int64_temporary_bytes": max(
            row["full_matrix_int64_temporary_bytes"] for row in results
        ),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": payload}, sort_keys=True))


if __name__ == "__main__":
    main()

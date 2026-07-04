#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"


def _first_row(run: str) -> tuple[dict, dict]:
    matrix = RAW / run / "matrix.jsonl"
    row = json.loads(next(line for line in matrix.read_text().splitlines() if line.strip()))
    report = json.loads(Path(row["report_path"]).read_text())
    return row, report


def _perf_line(run: str) -> str:
    row, report = _first_row(run)
    metrics = report["metrics"]
    phase = metrics["phase_totals"]
    buckets = row.get("bucket_coverage") or []
    replay = sum(item.get("replay_count", 0) for item in buckets)
    eager = sum(item.get("eager_count", 0) for item in buckets)
    return (
        f"{run}: {row['status']} variant={row['variant']} "
        f"out={metrics['end_to_end_output_tokens_per_s']:.4f} "
        f"decode={metrics['decode_tokens_per_s']:.4f} "
        f"prepare={phase['decode_prepare_s']:.4f} "
        f"forward={phase['decode_forward_s']:.4f} "
        f"ttft={metrics['ttft_s_mean']:.4f} "
        f"replay/eager={replay}/{eager} "
        f"saved={row.get('prefix_saved_prefill_tokens')}"
    )


def _counter_summary(run: str) -> list[str]:
    _, report = _first_row(run)
    counters = report.get("owner_timing", {}).get("rank0", {}).get("counters", []) or []
    wanted = {
        "swa_page_indices",
        "c4_sparse_raw_indices",
        "c4_sparse_page_indices",
        "c4_sparse_full_indices",
        "c128_raw_indices",
        "c128_page_indices",
        "c128_full_indices",
    }
    labels = [
        "dsv4.metadata_build.bytes",
        "dsv4.replay_metadata_copy.bytes",
        "dsv4.direct_graph_metadata.bytes",
    ]
    totals: dict[tuple[str, str], int] = {}
    for counter in counters:
        metadata = counter.get("metadata") or {}
        if metadata.get("phase") != "decode" or metadata.get("field") not in wanted:
            continue
        key = (counter.get("label"), metadata.get("field"))
        totals[key] = totals.get(key, 0) + int(counter.get("count", 0) or 0)
    lines = []
    for field in sorted(wanted):
        values = [totals.get((label, field), 0) for label in labels]
        if any(values):
            lines.append(f"{field}: build={values[0]} copy={values[1]} direct={values[2]}")
    return lines


def main() -> None:
    for run in [
        "large_phase1_prefix_on",
        "large_route_b_graph_baseline",
        "large_route_b_direct_graph_metadata_v2",
        "prefix_hit_route_b_direct_graph_metadata_direct_only",
        "eviction_pressure_route_b_direct_graph_metadata",
    ]:
        if (RAW / run / "matrix.jsonl").exists():
            print(_perf_line(run))
    if (RAW / "profile_large_route_b_direct_graph_metadata" / "matrix.jsonl").exists():
        print("\nprofile counters:")
        print("\n".join(_counter_summary("profile_large_route_b_direct_graph_metadata")))


if __name__ == "__main__":
    main()

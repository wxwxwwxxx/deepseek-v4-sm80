#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
SUMMARIES = ROOT / "summaries"
SUMMARIES.mkdir(parents=True, exist_ok=True)

MODE_ORDER = [
    "phase1_prefix_on",
    "route_b_graph_baseline",
    "route_b_direct_c4",
    "route_b_direct_full",
]
MODE_LABELS = {
    "phase1_prefix_on": "phase1 prefix on",
    "route_b_graph_baseline": "Route B graph baseline",
    "route_b_direct_c4": "Route B direct C4",
    "route_b_direct_full": "Route B direct SWA+C4+C128",
}


def _load_report(run_dir: Path) -> dict[str, Any] | None:
    matrix = run_dir / "matrix.jsonl"
    if not matrix.exists():
        return None
    for line in matrix.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(row["report_path"])
        if not path.is_absolute():
            path = ROOT.parents[1] / path
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _mode_from_run_name(name: str) -> str | None:
    for mode in MODE_ORDER:
        if name.endswith(mode):
            return mode
    return None


def _reports(prefix: str) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    out = {mode: [] for mode in MODE_ORDER}
    for run_dir in sorted(RAW.glob(f"{prefix}_*")):
        if not run_dir.is_dir():
            continue
        mode = _mode_from_run_name(run_dir.name)
        report = _load_report(run_dir)
        if mode and report:
            out[mode].append((run_dir.name, report))
    return out


def _metrics(report: dict[str, Any]) -> dict[str, float | int | str]:
    metrics = report.get("metrics", {})
    phase = metrics.get("phase_totals", {})
    graph = report.get("config", {}).get("graph_runner_case", {})
    return {
        "status": report.get("status", ""),
        "output_tok_s": float(metrics.get("end_to_end_output_tokens_per_s") or 0.0),
        "decode_tok_s": float(metrics.get("decode_tokens_per_s") or 0.0),
        "ttft_s": float(metrics.get("ttft_s_mean") or 0.0),
        "elapsed_s": float(metrics.get("elapsed_s") or 0.0),
        "prefill_prepare_s": float(phase.get("prefill_prepare_s") or 0.0),
        "decode_prepare_s": float(phase.get("decode_prepare_s") or 0.0),
        "prefill_forward_s": float(phase.get("prefill_forward_s") or 0.0),
        "decode_forward_s": float(phase.get("decode_forward_s") or 0.0),
        "decode_tokens": int(phase.get("decode_tokens") or 0),
        "replay": int(graph.get("replay_count") or 0),
        "eager": int(graph.get("eager_decode_count") or 0),
        "saved_prefill": int(
            metrics.get("prefix_cache", {})
            .get("rank0_final", {})
            .get("saved_prefill_tokens", 0)
            or 0
        ),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(out) + "\n"


def _throughput_tables() -> dict[str, dict[str, float]]:
    reports = _reports("throughput")
    detail_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, float]] = {}
    for mode in MODE_ORDER:
        per_run = []
        for run_name, report in reports[mode]:
            row = {"run": run_name, "mode": MODE_LABELS[mode], **_metrics(report)}
            detail_rows.append(row)
            per_run.append(row)
        if not per_run:
            continue
        values = {key: [float(row[key]) for row in per_run] for key in [
            "output_tok_s",
            "decode_tok_s",
            "ttft_s",
            "decode_prepare_s",
            "decode_forward_s",
            "elapsed_s",
        ]}
        agg = {
            "mode": MODE_LABELS[mode],
            "runs": len(per_run),
            "output_tok_s_mean": _mean(values["output_tok_s"]),
            "output_tok_s_stdev": _stdev(values["output_tok_s"]),
            "decode_tok_s_mean": _mean(values["decode_tok_s"]),
            "decode_prepare_s_mean": _mean(values["decode_prepare_s"]),
            "decode_forward_s_mean": _mean(values["decode_forward_s"]),
            "ttft_s_mean": _mean(values["ttft_s"]),
            "replay": max(int(row["replay"]) for row in per_run),
            "eager": max(int(row["eager"]) for row in per_run),
            "saved_prefill": max(int(row["saved_prefill"]) for row in per_run),
        }
        aggregates[mode] = {k: float(v) for k, v in agg.items() if isinstance(v, (int, float))}
        aggregate_rows.append(agg)

    _write_csv(
        SUMMARIES / "throughput_repeat_detail.csv",
        detail_rows,
        [
            "run",
            "mode",
            "status",
            "output_tok_s",
            "decode_tok_s",
            "ttft_s",
            "decode_prepare_s",
            "decode_forward_s",
            "elapsed_s",
            "replay",
            "eager",
            "saved_prefill",
        ],
    )
    _write_csv(
        SUMMARIES / "throughput_repeat.csv",
        aggregate_rows,
        [
            "mode",
            "runs",
            "output_tok_s_mean",
            "output_tok_s_stdev",
            "decode_tok_s_mean",
            "decode_prepare_s_mean",
            "decode_forward_s_mean",
            "ttft_s_mean",
            "replay",
            "eager",
            "saved_prefill",
        ],
    )
    md_rows = [
        [
            row["mode"],
            row["runs"],
            row["output_tok_s_mean"],
            row["output_tok_s_stdev"],
            row["decode_tok_s_mean"],
            row["decode_prepare_s_mean"],
            row["decode_forward_s_mean"],
            f"{row['replay']}/{row['eager']}",
        ]
        for row in aggregate_rows
    ]
    (SUMMARIES / "throughput_repeat.md").write_text(
        "# Throughput Repeat\n\n"
        "Unprofiled `serving_mixed_112req_wave16` runs. Owner timing is disabled here.\n\n"
        + _md_table(
            [
                "mode",
                "runs",
                "output tok/s mean",
                "output tok/s stdev",
                "decode tok/s mean",
                "decode prepare s mean",
                "decode forward s mean",
                "graph replay/eager",
            ],
            md_rows,
        ),
        encoding="utf-8",
    )
    return aggregates


def _timing(report: dict[str, Any], section: str, label: str) -> float:
    stats = report.get("owner_timing", {}).get(section, {}).get("by_label", {}).get(label, {})
    return float(stats.get("max_rank_total_ms") or 0.0)


def _timing_prefix(report: dict[str, Any], section: str, prefix: str) -> float:
    labels = report.get("owner_timing", {}).get(section, {}).get("by_label", {})
    return sum(
        float(stats.get("max_rank_total_ms") or 0.0)
        for label, stats in labels.items()
        if label.startswith(prefix)
    )


def _counter_rows(report: dict[str, Any], label_filter: str) -> list[dict[str, Any]]:
    rows = []
    counters = report.get("owner_timing", {}).get("rank0", {}).get("counters", []) or []
    for counter in counters:
        if counter.get("label") != label_filter:
            continue
        metadata = counter.get("metadata") or {}
        if metadata.get("phase") != "decode":
            continue
        rows.append(
            {
                "field": metadata.get("field", ""),
                "stable": metadata.get("stable", ""),
                "rows": int(metadata.get("rows") or 0),
                "value": int(counter.get("count") or 0),
            }
        )
    return rows


def _counter_sum(report: dict[str, Any], label_filter: str, *, fields: set[str] | None = None) -> int:
    total = 0
    for row in _counter_rows(report, label_filter):
        if fields is None or row["field"] in fields:
            total += int(row["value"])
    return total


def _owner_tables() -> None:
    reports = {mode: runs[0][1] for mode, runs in _reports("profile").items() if runs}
    prep_fields = [
        "page_table",
        "c4_page_table",
        "c128_page_table",
        "c4_indexer_page_table",
        "swa_page_indices",
        "c4_sparse_raw_indices",
        "c4_sparse_page_indices",
        "c4_sparse_full_indices",
        "c128_raw_indices",
        "c128_page_indices",
        "c128_full_indices",
    ]
    prep_rows = []
    forward_rows = []
    update_rows = []
    for mode in MODE_ORDER:
        report = reports.get(mode)
        if not report:
            continue
        metrics = _metrics(report)
        prep_rows.append(
            {
                "mode": MODE_LABELS[mode],
                "profile_output_tok_s": metrics["output_tok_s"],
                "decode_prepare_s": metrics["decode_prepare_s"],
                "decode_forward_s": metrics["decode_forward_s"],
                "host_attention_metadata_ms": _timing(
                    report, "host", "dsv4.prepare.decode.attention_metadata"
                ),
                "make_component_page_tables_ms": _timing(
                    report, "cuda", "dsv4.metadata.decode.make_component_page_tables"
                ),
                "make_full_page_table_ms": _timing(
                    report, "cuda", "dsv4.metadata.decode.make_page_table"
                ),
                "make_swa_indices_ms": _timing(
                    report, "cuda", "dsv4.metadata.decode.make_swa_indices"
                ),
                "make_c4_sparse_indices_ms": _timing(
                    report, "cuda", "dsv4.metadata.decode.make_c4_sparse_indices"
                ),
                "make_c128_indices_ms": _timing(
                    report, "cuda", "dsv4.metadata.decode.make_c128_indices"
                ),
                "make_write_locs_ms": _timing(
                    report, "cuda", "dsv4.metadata.decode.make_write_locs"
                ),
                "replay_fused_copy_ms": _timing(
                    report, "cuda", "dsv4.replay_metadata.decode.fused_copy"
                ),
                "replay_component_page_tables_ms": _timing_prefix(
                    report, "cuda", "dsv4.replay_metadata.decode.component_page_table."
                ),
                "replay_write_locs_ms": _timing(
                    report, "cuda", "dsv4.replay_metadata.decode.component_write_locs"
                ),
                "direct_index_buffers_ms": _timing(
                    report, "cuda", "dsv4.direct_graph_metadata.decode.index_buffers"
                ),
                "metadata_build_bytes": _counter_sum(
                    report, "dsv4.metadata_build.bytes", fields=set(prep_fields)
                ),
                "replay_copy_bytes": _counter_sum(
                    report, "dsv4.replay_metadata_copy.bytes", fields=set(prep_fields)
                ),
                "direct_graph_bytes": _counter_sum(
                    report, "dsv4.direct_graph_metadata.bytes", fields=set(prep_fields)
                ),
            }
        )

        labels = report.get("owner_timing", {}).get("cuda", {}).get("by_label", {})
        groups = {
            "attention": 0.0,
            "indexer/compressor": 0.0,
            "MoE/shared experts": 0.0,
            "communication": 0.0,
            "other owner": 0.0,
        }
        for label, stats in labels.items():
            ms = float(stats.get("max_rank_total_ms") or 0.0)
            if not ms:
                continue
            low = label.lower()
            if not low.startswith("dsv4.owner."):
                continue
            if "comm" in low or "all_reduce" in low or "all_gather" in low:
                groups["communication"] += ms
            elif "indexer" in low or "compress" in low or "store_cache" in low:
                groups["indexer/compressor"] += ms
            elif "moe" in low or "expert" in low or "gate_up" in low or "shared_down" in low:
                groups["MoE/shared experts"] += ms
            elif "attn" in low or ".attention" in low:
                groups["attention"] += ms
            elif low.startswith("dsv4.owner."):
                groups["other owner"] += ms
        row = {"mode": MODE_LABELS[mode], "decode_forward_s": metrics["decode_forward_s"], **groups}
        forward_rows.append(row)

        for label in [
            "dsv4.metadata_build.calls",
            "dsv4.replay_metadata_copy.calls",
            "dsv4.metadata_build.bytes",
            "dsv4.replay_metadata_copy.bytes",
            "dsv4.direct_graph_metadata.bytes",
        ]:
            by_field: dict[tuple[str, str], int] = {}
            for counter in _counter_rows(report, label):
                key = (counter["field"], counter["stable"])
                by_field[key] = by_field.get(key, 0) + counter["value"]
            for (field, stable), value in sorted(by_field.items()):
                update_rows.append(
                    {
                        "mode": MODE_LABELS[mode],
                        "counter": label,
                        "field": field,
                        "stable": stable,
                        "value": value,
                    }
                )

    _write_csv(SUMMARIES / "prepare_owner_profile.csv", prep_rows, list(prep_rows[0]) if prep_rows else [])
    _write_csv(
        SUMMARIES / "decode_forward_owner_profile.csv",
        forward_rows,
        list(forward_rows[0]) if forward_rows else [],
    )
    _write_csv(
        SUMMARIES / "metadata_update_pressure.csv",
        update_rows,
        ["mode", "counter", "field", "stable", "value"],
    )
    (SUMMARIES / "prepare_owner_profile.md").write_text(
        "# Decode Prepare Owner Profile\n\n"
        "Owner timing profile runs only; do not use these rows as final throughput evidence.\n\n"
        + _md_table(
            [
                "mode",
                "decode prepare s",
                "host attention metadata ms",
                "component tables ms",
                "full page table ms",
                "SWA idx ms",
                "C4 idx ms",
                "C128 idx ms",
                "write locs ms",
                "replay fused copy ms",
                "replay comp tables ms",
                "direct index ms",
                "build bytes",
                "copy bytes",
                "direct bytes",
            ],
            [
                [
                    row["mode"],
                    row["decode_prepare_s"],
                    row["host_attention_metadata_ms"],
                    row["make_component_page_tables_ms"],
                    row["make_full_page_table_ms"],
                    row["make_swa_indices_ms"],
                    row["make_c4_sparse_indices_ms"],
                    row["make_c128_indices_ms"],
                    row["make_write_locs_ms"],
                    row["replay_fused_copy_ms"],
                    row["replay_component_page_tables_ms"],
                    row["direct_index_buffers_ms"],
                    row["metadata_build_bytes"],
                    row["replay_copy_bytes"],
                    row["direct_graph_bytes"],
                ]
                for row in prep_rows
            ],
        ),
        encoding="utf-8",
    )
    (SUMMARIES / "decode_forward_owner_profile.md").write_text(
        "# Decode Forward Owner Profile\n\n"
        "CUDA owner-timing labels grouped by forward compute/communication owner. "
        "Prepare-side metadata/replay labels are excluded here and summarized in `prepare_owner_profile.md`. "
        "Values are max-rank ms.\n\n"
        + _md_table(
            [
                "mode",
                "decode forward s",
                "attention ms",
                "indexer/compressor ms",
                "MoE/shared ms",
                "communication ms",
                "other owner ms",
            ],
            [
                [
                    row["mode"],
                    row["decode_forward_s"],
                    row["attention"],
                    row["indexer/compressor"],
                    row["MoE/shared experts"],
                    row["communication"],
                    row["other owner"],
                ]
                for row in forward_rows
            ],
        ),
        encoding="utf-8",
    )
    focus = [
        row
        for row in update_rows
        if row["counter"] in {"dsv4.metadata_build.calls", "dsv4.replay_metadata_copy.calls"}
        and row["field"] in {
            "page_table",
            "c4_page_table",
            "c128_page_table",
            "c4_indexer_page_table",
            "c128_raw_indices",
            "c128_page_indices",
            "c128_full_indices",
        }
    ]
    (SUMMARIES / "metadata_update_pressure.md").write_text(
        "# Metadata Update Pressure\n\n"
        "Call counts from owner-timing profile runs. Per-request page tables and per-prefix-hit C128 rows are expected to be stable candidates if their calls track decode replay steps.\n\n"
        + _md_table(
            ["mode", "counter", "field", "stable", "value"],
            [[row["mode"], row["counter"], row["field"], row["stable"], row["value"]] for row in focus],
        ),
        encoding="utf-8",
    )


def _prepare_forward_attribution(aggregates: dict[str, dict[str, float]]) -> None:
    rows = []
    phase1 = aggregates.get("phase1_prefix_on", {})
    for mode in MODE_ORDER:
        agg = aggregates.get(mode)
        if not agg:
            continue
        rows.append(
            [
                MODE_LABELS[mode],
                agg.get("output_tok_s_mean", 0.0),
                agg.get("decode_tok_s_mean", 0.0),
                agg.get("decode_prepare_s_mean", 0.0),
                agg.get("decode_forward_s_mean", 0.0),
                (
                    agg.get("decode_prepare_s_mean", 0.0)
                    - phase1.get("decode_prepare_s_mean", 0.0)
                ),
                (
                    agg.get("decode_forward_s_mean", 0.0)
                    - phase1.get("decode_forward_s_mean", 0.0)
                ),
            ]
        )
    direct = aggregates.get("route_b_direct_c4", {})
    baseline = aggregates.get("route_b_graph_baseline", {})
    stable_lines = []
    if baseline and direct:
        gain = direct.get("output_tok_s_mean", 0.0) - baseline.get("output_tok_s_mean", 0.0)
        prep_gain = baseline.get("decode_prepare_s_mean", 0.0) - direct.get(
            "decode_prepare_s_mean", 0.0
        )
        direct_sd = direct.get("output_tok_s_stdev", 0.0)
        stable_lines.append(
            f"Direct C4 mean output gain vs Route B baseline: {gain:.4f} tok/s; "
            f"decode-prepare reduction: {prep_gain:.4f} s; direct-C4 output stdev: {direct_sd:.4f} tok/s."
        )
    (SUMMARIES / "prepare_forward_attribution.md").write_text(
        "# Prepare Versus Forward Attribution\n\n"
        + _md_table(
            [
                "mode",
                "output tok/s",
                "decode tok/s",
                "decode prepare s",
                "decode forward s",
                "prepare delta vs phase1 s",
                "forward delta vs phase1 s",
            ],
            rows,
        )
        + ("\n" + "\n".join(stable_lines) + "\n" if stable_lines else ""),
        encoding="utf-8",
    )


def main() -> None:
    aggregates = _throughput_tables()
    _prepare_forward_attribution(aggregates)
    _owner_tables()


if __name__ == "__main__":
    main()

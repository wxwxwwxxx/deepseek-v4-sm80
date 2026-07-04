#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = ROOT / "raw"
SUMMARIES = ROOT / "summaries"
PREV = REPO / "performance_milestones" / "target08_route_b_remaining_gap_attribution_reset"
SUMMARIES.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
            path = REPO / path
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _metrics(report: dict[str, Any]) -> dict[str, float | int | str]:
    metrics = report.get("metrics", {})
    phase = metrics.get("phase_totals", {})
    graph = report.get("config", {}).get("graph_runner_case", {})
    return {
        "status": report.get("status", ""),
        "output_tok_s": float(metrics.get("end_to_end_output_tokens_per_s") or 0.0),
        "decode_tok_s": float(metrics.get("decode_tokens_per_s") or 0.0),
        "decode_prepare_s": float(phase.get("decode_prepare_s") or 0.0),
        "decode_forward_s": float(phase.get("decode_forward_s") or 0.0),
        "elapsed_s": float(metrics.get("elapsed_s") or 0.0),
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


def _current_throughput() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detail = []
    for run_dir in sorted(RAW.glob("throughput_r*_route_b_lifetime")):
        if not run_dir.is_dir():
            continue
        report = _load_report(run_dir)
        if report is None:
            continue
        detail.append({"run": run_dir.name, **_metrics(report)})

    values = {key: [float(row[key]) for row in detail] for key in [
        "output_tok_s",
        "decode_tok_s",
        "decode_prepare_s",
        "decode_forward_s",
        "elapsed_s",
    ]}
    aggregate = {
        "mode": "Route B direct C4 + lifetime cache",
        "runs": len(detail),
        "output_tok_s_mean": _mean(values["output_tok_s"]),
        "output_tok_s_stdev": _stdev(values["output_tok_s"]),
        "decode_tok_s_mean": _mean(values["decode_tok_s"]),
        "decode_prepare_s_mean": _mean(values["decode_prepare_s"]),
        "decode_forward_s_mean": _mean(values["decode_forward_s"]),
        "elapsed_s_mean": _mean(values["elapsed_s"]),
        "replay": max((int(row["replay"]) for row in detail), default=0),
        "eager": max((int(row["eager"]) for row in detail), default=0),
        "saved_prefill": max((int(row["saved_prefill"]) for row in detail), default=0),
    }
    return detail, aggregate


def _write_throughput() -> dict[str, Any]:
    detail, current = _current_throughput()
    _write_csv(
        SUMMARIES / "throughput_repeat_detail.csv",
        detail,
        [
            "run",
            "status",
            "output_tok_s",
            "decode_tok_s",
            "decode_prepare_s",
            "decode_forward_s",
            "elapsed_s",
            "replay",
            "eager",
            "saved_prefill",
        ],
    )

    prior_rows = _read_csv(PREV / "summaries" / "throughput_repeat.csv")
    wanted = {"phase1 prefix on", "Route B graph baseline", "Route B direct C4"}
    rows: list[dict[str, Any]] = []
    for row in prior_rows:
        if row["mode"] not in wanted:
            continue
        rows.append(
            {
                "mode": row["mode"],
                "runs": int(row["runs"]),
                "output_tok_s_mean": float(row["output_tok_s_mean"]),
                "output_tok_s_stdev": float(row["output_tok_s_stdev"]),
                "decode_tok_s_mean": float(row["decode_tok_s_mean"]),
                "decode_prepare_s_mean": float(row["decode_prepare_s_mean"]),
                "decode_forward_s_mean": float(row["decode_forward_s_mean"]),
                "replay": int(row["replay"]),
                "eager": int(row["eager"]),
                "source": "08.26 frozen baseline",
            }
        )
    current = {**current, "source": "08.27 current opt-in"}
    rows.append(current)
    _write_csv(
        SUMMARIES / "throughput_repeat.csv",
        rows,
        [
            "mode",
            "runs",
            "output_tok_s_mean",
            "output_tok_s_stdev",
            "decode_tok_s_mean",
            "decode_prepare_s_mean",
            "decode_forward_s_mean",
            "replay",
            "eager",
            "source",
        ],
    )
    (SUMMARIES / "throughput_repeat.md").write_text(
        "# Throughput Repeat\n\n"
        "Unprofiled `serving_mixed_112req_wave16`; owner timing disabled. "
        "The first three rows are the TARGET 08.26 frozen comparison set, and "
        "the final row is the TARGET 08.27 opt-in run produced in this milestone.\n\n"
        + _md_table(
            [
                "mode",
                "runs",
                "output tok/s mean",
                "stdev",
                "decode tok/s mean",
                "decode prepare s",
                "decode forward s",
                "graph replay/eager",
                "source",
            ],
            [
                [
                    row["mode"],
                    row["runs"],
                    row["output_tok_s_mean"],
                    row["output_tok_s_stdev"],
                    row["decode_tok_s_mean"],
                    row["decode_prepare_s_mean"],
                    row["decode_forward_s_mean"],
                    f"{row['replay']}/{row['eager']}",
                    row["source"],
                ]
                for row in rows
            ],
        ),
        encoding="utf-8",
    )
    return current


def _load_profile_report() -> dict[str, Any]:
    report = _load_report(RAW / "profile_route_b_lifetime")
    if report is None:
        raise FileNotFoundError("missing profile_route_b_lifetime report")
    return report


def _timing(report: dict[str, Any], section: str, label: str) -> float:
    stats = (
        report.get("owner_timing", {})
        .get(section, {})
        .get("by_label", {})
        .get(label, {})
    )
    return float(stats.get("max_rank_total_ms") or 0.0)


def _timing_prefix(report: dict[str, Any], section: str, prefix: str) -> float:
    labels = report.get("owner_timing", {}).get(section, {}).get("by_label", {})
    return sum(
        float(stats.get("max_rank_total_ms") or 0.0)
        for label, stats in labels.items()
        if label.startswith(prefix)
    )


def _write_profile() -> tuple[dict[str, Any], dict[str, int]]:
    prior = _read_csv(PREV / "summaries" / "prepare_owner_profile.csv")
    prior_direct = next(row for row in prior if row["mode"] == "Route B direct C4")
    report = _load_profile_report()
    metrics = _metrics(report)
    component_replay = _timing_prefix(
        report,
        "cuda",
        "dsv4.replay_metadata.decode.component_page_table.",
    )
    current = {
        "mode": "Route B direct C4 + lifetime cache",
        "profile_output_tok_s": metrics["output_tok_s"],
        "decode_prepare_s": metrics["decode_prepare_s"],
        "decode_forward_s": metrics["decode_forward_s"],
        "host_attention_metadata_ms": _timing(
            report,
            "host",
            "dsv4.prepare.decode.attention_metadata",
        ),
        "make_component_page_tables_ms": _timing(
            report,
            "cuda",
            "dsv4.metadata.decode.make_component_page_tables",
        ),
        "make_full_page_table_ms": _timing(
            report,
            "cuda",
            "dsv4.metadata.decode.make_page_table",
        ),
        "make_swa_indices_ms": _timing(
            report,
            "cuda",
            "dsv4.metadata.decode.make_swa_indices",
        ),
        "make_c4_sparse_indices_ms": _timing(
            report,
            "cuda",
            "dsv4.metadata.decode.make_c4_sparse_indices",
        ),
        "make_c128_indices_ms": _timing(
            report,
            "cuda",
            "dsv4.metadata.decode.make_c128_indices",
        ),
        "make_write_locs_ms": _timing(
            report,
            "cuda",
            "dsv4.metadata.decode.make_write_locs",
        ),
        "replay_fused_copy_ms": _timing(
            report,
            "cuda",
            "dsv4.replay_metadata.decode.fused_copy",
        ),
        "replay_component_page_tables_ms": component_replay,
        "replay_write_locs_ms": _timing(
            report,
            "cuda",
            "dsv4.replay_metadata.decode.component_write_locs",
        ),
        "direct_index_buffers_ms": _timing(
            report,
            "cuda",
            "dsv4.direct_graph_metadata.decode.index_buffers",
        ),
    }
    rows = [
        {**prior_direct, "source": "08.26 Route B direct C4 profile"},
        {**current, "source": "08.27 current opt-in profile"},
    ]
    fields = [
        "mode",
        "profile_output_tok_s",
        "decode_prepare_s",
        "decode_forward_s",
        "host_attention_metadata_ms",
        "make_component_page_tables_ms",
        "make_full_page_table_ms",
        "make_swa_indices_ms",
        "make_c4_sparse_indices_ms",
        "make_c128_indices_ms",
        "make_write_locs_ms",
        "replay_fused_copy_ms",
        "replay_component_page_tables_ms",
        "replay_write_locs_ms",
        "direct_index_buffers_ms",
        "source",
    ]
    _write_csv(SUMMARIES / "prepare_owner_profile.csv", rows, fields)
    (SUMMARIES / "prepare_owner_profile.md").write_text(
        "# Prepare Owner Profile\n\n"
        "Owner-timing profile for `serving_mixed_112req_wave16`. These numbers "
        "include profiling overhead and are for attribution, not throughput.\n\n"
        + _md_table(
            [
                "mode",
                "prepare s",
                "forward s",
                "host attention metadata ms",
                "component tables ms",
                "full page table ms",
                "C4 sparse ms",
                "C128 ms",
                "replay component tables ms",
                "direct index ms",
            ],
            [
                [
                    row["mode"],
                    float(row["decode_prepare_s"]),
                    float(row["decode_forward_s"]),
                    float(row["host_attention_metadata_ms"]),
                    float(row["make_component_page_tables_ms"]),
                    float(row["make_full_page_table_ms"]),
                    float(row["make_c4_sparse_indices_ms"]),
                    float(row["make_c128_indices_ms"]),
                    float(row["replay_component_page_tables_ms"]),
                    float(row["direct_index_buffers_ms"]),
                ]
                for row in rows
            ],
        ),
        encoding="utf-8",
    )

    counters = _cache_counters(report)
    return current, counters


def _cache_counters(report: dict[str, Any]) -> dict[str, int]:
    out = {"dirty_rows": 0, "clean_rows": 0}
    counters = report.get("owner_timing", {}).get("rank0", {}).get("counters", [])
    for counter in counters:
        if counter.get("label") != "dsv4.component_page_table_cache.rows":
            continue
        metadata = counter.get("metadata") or {}
        if metadata.get("phase") != "decode":
            continue
        status = metadata.get("status")
        if status == "dirty":
            out["dirty_rows"] += int(counter.get("count") or 0)
        elif status == "clean":
            out["clean_rows"] += int(counter.get("count") or 0)
    out["total_rows"] = out["dirty_rows"] + out["clean_rows"]
    return out


def _write_metadata_pressure(counters: dict[str, int]) -> None:
    prior = _read_csv(PREV / "summaries" / "metadata_update_pressure.csv")
    wanted_fields = {"c4_page_table", "c128_page_table", "c4_indexer_page_table"}
    prior_rows = [
        row
        for row in prior
        if row["mode"] == "Route B direct C4"
        and row["counter"] == "dsv4.metadata_build.calls"
        and row["field"] in wanted_fields
    ]
    rows: list[dict[str, Any]] = [
        {
            "mode": row["mode"],
            "counter": row["counter"],
            "field": row["field"],
            "stable": row["stable"],
            "value": int(row["value"]),
            "source": "08.26 direct C4",
        }
        for row in prior_rows
    ]
    rows.extend(
        [
            {
                "mode": "Route B direct C4 + lifetime cache",
                "counter": "dsv4.component_page_table_cache.rows",
                "field": "dirty_rows",
                "stable": "request-slot refresh",
                "value": counters["dirty_rows"],
                "source": "08.27 profile rank0",
            },
            {
                "mode": "Route B direct C4 + lifetime cache",
                "counter": "dsv4.component_page_table_cache.rows",
                "field": "clean_rows",
                "stable": "request-slot reuse",
                "value": counters["clean_rows"],
                "source": "08.27 profile rank0",
            },
        ]
    )
    _write_csv(
        SUMMARIES / "metadata_update_pressure.csv",
        rows,
        ["mode", "counter", "field", "stable", "value", "source"],
    )
    (SUMMARIES / "metadata_update_pressure.md").write_text(
        "# Metadata Update Pressure\n\n"
        "The old Route B direct-C4 path rebuilt each component page-table field "
        "once per decode replay step. The new opt-in still selects/copies graph "
        "source rows every step, but actual row rebuilds are request-slot "
        "refreshes.\n\n"
        + _md_table(
            ["mode", "counter", "field", "stable", "value", "source"],
            [
                [
                    row["mode"],
                    row["counter"],
                    row["field"],
                    row["stable"],
                    int(row["value"]),
                    row["source"],
                ]
                for row in rows
            ],
        ),
        encoding="utf-8",
    )


def _write_correctness() -> dict[str, Any]:
    text_path = RAW / (
        "text_smoke_routeb_lifetime_verify."
        "dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime.json"
    )
    text = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else {}
    verify_report = _load_report(RAW / "verify_serving_route_b_lifetime")
    verify_metrics = _metrics(verify_report) if verify_report else {}
    outputs = [
        {
            "prompt": item.get("prompt", ""),
            "text": item.get("text", ""),
        }
        for item in text.get("outputs", [])
    ]
    (SUMMARIES / "correctness.md").write_text(
        "# Correctness\n\n"
        f"- Text smoke status: `{text.get('status', 'missing')}` with "
        "`MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1`.\n"
        f"- Full `serving_mixed_112req_wave16` oracle status: "
        f"`{verify_metrics.get('status', 'missing')}`; graph replay/eager "
        f"`{verify_metrics.get('replay', 0)}/{verify_metrics.get('eager', 0)}`.\n"
        "- Text smoke outputs:\n"
        + "".join(
            f"  - `{item['text']}`\n"
            for item in outputs
        ),
        encoding="utf-8",
    )
    return {"text_status": text.get("status", "missing"), **verify_metrics}


def main() -> None:
    throughput = _write_throughput()
    profile, counters = _write_profile()
    _write_metadata_pressure(counters)
    correctness = _write_correctness()
    summary = {
        "throughput": throughput,
        "profile": profile,
        "component_page_table_cache": counters,
        "correctness": correctness,
    }
    (SUMMARIES / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

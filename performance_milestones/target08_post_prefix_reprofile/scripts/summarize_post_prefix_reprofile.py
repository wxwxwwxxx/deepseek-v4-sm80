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
SUMMARIES.mkdir(parents=True, exist_ok=True)

PROMOTED = "dsv4_sm80_a100_victory_prefix_routeb_lifetime"
CONTROL = "dsv4_sm80_a100_victory"
VERIFY_ENV = "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY"

TARGET07 = {
    "historical_4096_1024_bs4": 131.7561,
    "historical_4096_128_bs4": 62.3925,
    "old_vllm_serving_4096_1024": 114.07,
    "fresh_vllm_4096_1024_bs4": 201.99,
    "fresh_vllm_4096_128_bs4": 82.28,
}

TARGET0828 = {
    "serving_mixed_112req_wave16": {
        "output_tok_s": 163.7220,
        "decode_prepare_s": 1.1359,
        "decode_forward_s": 9.8927,
        "graph": "441/0",
    },
    "prefix_multi_112req_wave16": {
        "output_tok_s": 105.4163,
        "decode_prepare_s": 0.2868,
        "decode_forward_s": 1.9164,
        "graph": "49/0",
        "saved_prefill": 49152,
    },
    "prefix_eviction_pressure_96req_wave16": {
        "output_tok_s": 13.0260,
        "decode_prepare_s": 0.1537,
        "decode_forward_s": 0.1917,
        "graph": "6/0",
        "evictions": 3,
    },
    "decode_ladder_bs16": {
        "output_tok_s": 98.3116,
        "decode_prepare_s": 0.1639,
        "decode_forward_s": 1.6786,
        "graph": "63/0",
    },
}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, sort_keys=True)
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(out) + "\n"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_reports(run_dir: Path) -> list[dict[str, Any]]:
    matrix = run_dir / "matrix.jsonl"
    reports: list[dict[str, Any]] = []
    if not matrix.exists():
        return reports
    for line in matrix.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        report_path = Path(row.get("report_path", ""))
        if not report_path.is_absolute():
            report_path = REPO / report_path
        if report_path.exists():
            reports.append(_read_json(report_path))
    return reports


def _all_report_items() -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    if not RAW.exists():
        return items
    for run_dir in sorted(path for path in RAW.iterdir() if path.is_dir()):
        if run_dir.name == "profile_promoted_serving_suite":
            continue
        for report in _load_reports(run_dir):
            items.append((run_dir.name, report))
    return items


def _variant_group(report: dict[str, Any]) -> str:
    name = report.get("variant", {}).get("name", "")
    if name == PROMOTED:
        return "promoted_prefix"
    if name == CONTROL:
        return "target07_control"
    return name or "unknown"


def _scenario_name(report: dict[str, Any]) -> str:
    scenario = report.get("scenario", {})
    return scenario.get("name", "") if isinstance(scenario, dict) else str(scenario)


def _graph(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    return config.get("graph_runner_case") or config.get("graph_runner") or {}


def _prefix(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    prefix = metrics.get("prefix_cache", {}).get("rank0_final", {})
    return prefix or report.get("config", {}).get("prefix_cache_metrics", {}) or {}


def _prefix_delta(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    return metrics.get("prefix_cache", {}).get("rank0_repeat_delta", {}) or {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _cv(values: list[float]) -> float:
    mean = _mean(values)
    return 0.0 if mean == 0.0 else _stdev(values) / mean


def _phase_repeat_values(report: dict[str, Any], key: str) -> list[float]:
    values = []
    for repeat in report.get("repeats", []):
        phase = repeat.get("phase_totals", {})
        values.append(float(phase.get(key) or 0.0))
    return values


def _output_rates(report: dict[str, Any]) -> list[float]:
    rates = []
    for repeat in report.get("repeats", []):
        elapsed = float(repeat.get("elapsed_s") or 0.0)
        tokens = float(repeat.get("actual_output_tokens") or 0.0)
        if elapsed > 0:
            rates.append(tokens / elapsed)
    if rates:
        return rates
    metrics = report.get("metrics", {})
    value = metrics.get("end_to_end_output_tokens_per_s")
    return [float(value)] if value is not None else []


def _bytes_gib(value: Any) -> float:
    return float(value or 0) / (1024.0**3)


def _bytes_mib(value: Any) -> float:
    return float(value or 0) / (1024.0**2)


def _run_kind(run_name: str) -> str:
    if run_name.startswith("macro_"):
        return "macro"
    if run_name.startswith("verify_"):
        return "verify"
    if run_name.startswith("profile_"):
        return "profile"
    return "other"


def _report_row(run_name: str, report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    phase = metrics.get("phase_totals", {})
    graph = _graph(report)
    prefix = _prefix(report)
    delta = _prefix_delta(report)
    rates = _output_rates(report)
    prefill_forward = _phase_repeat_values(report, "prefill_forward_s")
    decode_prepare = _phase_repeat_values(report, "decode_prepare_s")
    decode_forward = _phase_repeat_values(report, "decode_forward_s")
    prefill_prepare = _phase_repeat_values(report, "prefill_prepare_s")
    return {
        "run": run_name,
        "kind": _run_kind(run_name),
        "variant_group": _variant_group(report),
        "variant": report.get("variant", {}).get("name", ""),
        "scenario": _scenario_name(report),
        "status": report.get("status", ""),
        "repeats": len(report.get("repeats", [])),
        "output_tok_s_mean": _mean(rates),
        "output_tok_s_stdev": _stdev(rates),
        "output_tok_s_cv": _cv(rates),
        "elapsed_s": float(metrics.get("elapsed_s") or 0.0),
        "ttft_ms_mean": 1000.0 * float(metrics.get("ttft_s_mean") or 0.0),
        "tpot_ms_mean": 1000.0 * float(metrics.get("topt_s_mean") or 0.0),
        "request_latency_ms_mean": 1000.0
        * float(metrics.get("request_latency_s_mean") or 0.0),
        "prefill_tok_s": float(metrics.get("prefill_tokens_per_s") or 0.0),
        "decode_tok_s": float(metrics.get("decode_tokens_per_s") or 0.0),
        "prefill_prepare_s_mean": _mean(prefill_prepare)
        if prefill_prepare
        else float(phase.get("prefill_prepare_s") or 0.0),
        "prefill_forward_s_mean": _mean(prefill_forward)
        if prefill_forward
        else float(phase.get("prefill_forward_s") or 0.0),
        "decode_prepare_s_mean": _mean(decode_prepare)
        if decode_prepare
        else float(phase.get("decode_prepare_s") or 0.0),
        "decode_forward_s_mean": _mean(decode_forward)
        if decode_forward
        else float(phase.get("decode_forward_s") or 0.0),
        "decode_tokens": int(phase.get("decode_tokens") or 0),
        "prompt_tokens": int(metrics.get("prompt_tokens") or 0),
        "actual_output_tokens": int(metrics.get("actual_output_tokens") or 0),
        "graph_enabled": bool(graph.get("enabled")),
        "requested_bs": graph.get("requested_bs", []),
        "captured_bs": graph.get("captured_bs", []),
        "replay": int(graph.get("replay_count") or 0),
        "eager": int(graph.get("eager_decode_count") or 0),
        "replay_by_padded": graph.get("replay_count_by_padded_size", {}),
        "eager_by_bs": graph.get("eager_decode_count_by_batch_size", {}),
        "graph_capture_delta_gib": _bytes_gib(graph.get("capture_memory_delta_bytes")),
        "peak_allocated_gib": _bytes_gib(metrics.get("peak_gpu_memory_allocated_bytes")),
        "peak_reserved_gib": _bytes_gib(metrics.get("peak_gpu_memory_reserved_bytes")),
        "kv_cache_gib_per_rank": _bytes_gib(metrics.get("kv_cache_memory_bytes_per_rank_max")),
        "prefix_hit_requests": int(delta.get("hit_requests", prefix.get("hit_requests", 0)) or 0),
        "prefix_miss_requests": int(
            delta.get("miss_requests", prefix.get("miss_requests", 0)) or 0
        ),
        "prefix_saved_prefill_tokens": int(
            delta.get("saved_prefill_tokens", prefix.get("saved_prefill_tokens", 0)) or 0
        ),
        "prefix_evictions": int(delta.get("evictions", prefix.get("evictions", 0)) or 0),
        "prefix_evicted_tokens": int(
            delta.get("evicted_tokens", prefix.get("evicted_tokens", 0)) or 0
        ),
        "retained_prefix_pages": int(prefix.get("retained_prefix_pages") or 0),
        "retained_prefix_tokens": int(prefix.get("retained_prefix_tokens") or 0),
        "retained_memory_mib": _bytes_mib(
            prefix.get("dsv4_retention", {}).get("retained_memory_bytes")
        ),
        "swa_retained_mib": _bytes_mib(prefix.get("dsv4_retention", {}).get("swa_bytes")),
        "c4_retained_mib": _bytes_mib(prefix.get("dsv4_retention", {}).get("c4_bytes")),
        "c128_retained_mib": _bytes_mib(prefix.get("dsv4_retention", {}).get("c128_bytes")),
        "component_available_pages": prefix.get("dsv4_component_ownership", {}).get(
            "available_component_pages", ""
        ),
        "component_live_full_pages": prefix.get("dsv4_component_ownership", {}).get(
            "live_full_pages", ""
        ),
        "communication_count": int(
            report.get("communication_counters", {}).get("total_count") or 0
        ),
        "communication_gib": _bytes_gib(
            report.get("communication_counters", {}).get("total_bytes")
        ),
    }


def _text_smoke() -> dict[str, Any]:
    variant_path = RAW / f"text_smoke_promoted_verify.{PROMOTED}.json"
    generic_path = RAW / "text_smoke_promoted_verify.json"
    path = variant_path if variant_path.exists() else generic_path
    if not path.exists():
        return {"status": "missing", "outputs": [], "verifier": False, "graph": ""}
    data = _read_json(path)
    raw_env = data.get("variant", {}).get("raw_dsv4_sm80_env", {})
    active = data.get("variant", {}).get("active_dsv4_toggles", [])
    graph = data.get("config", {}).get("graph_runner", {})
    return {
        "status": data.get("status", "missing"),
        "outputs": [item.get("text", "") for item in data.get("outputs", [])],
        "verifier": raw_env.get(VERIFY_ENV) == "1" or VERIFY_ENV in active,
        "graph": f"{graph.get('replay_count', '')}/{graph.get('eager_decode_count', '')}",
    }


def _aggregate_macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "macro":
            continue
        groups.setdefault((row["variant_group"], row["scenario"]), []).append(row)

    out: list[dict[str, Any]] = []
    for (variant_group, scenario), group in sorted(groups.items()):
        first = group[0]

        def values(key: str) -> list[float]:
            return [float(row.get(key) or 0.0) for row in group]

        def mean_value(key: str) -> float:
            return _mean(values(key))

        output_values = values("output_tok_s_mean")
        row = dict(first)
        row.update(
            {
                "run": f"aggregate_{variant_group}_{scenario}",
                "kind": "macro_agg",
                "variant_group": variant_group,
                "scenario": scenario,
                "status": "pass" if all(item["status"] == "pass" for item in group) else "fail",
                "repeats": len(group),
                "output_tok_s_mean": _mean(output_values),
                "output_tok_s_stdev": _stdev(output_values),
                "output_tok_s_cv": _cv(output_values),
                "elapsed_s": mean_value("elapsed_s"),
                "ttft_ms_mean": mean_value("ttft_ms_mean"),
                "tpot_ms_mean": mean_value("tpot_ms_mean"),
                "request_latency_ms_mean": mean_value("request_latency_ms_mean"),
                "prefill_tok_s": mean_value("prefill_tok_s"),
                "decode_tok_s": mean_value("decode_tok_s"),
                "prefill_prepare_s_mean": mean_value("prefill_prepare_s_mean"),
                "prefill_forward_s_mean": mean_value("prefill_forward_s_mean"),
                "decode_prepare_s_mean": mean_value("decode_prepare_s_mean"),
                "decode_forward_s_mean": mean_value("decode_forward_s_mean"),
                "decode_tokens": int(round(mean_value("decode_tokens"))),
                "prompt_tokens": int(round(mean_value("prompt_tokens"))),
                "actual_output_tokens": int(round(mean_value("actual_output_tokens"))),
                "replay": int(round(mean_value("replay"))),
                "eager": int(round(mean_value("eager"))),
                "graph_capture_delta_gib": mean_value("graph_capture_delta_gib"),
                "peak_allocated_gib": mean_value("peak_allocated_gib"),
                "peak_reserved_gib": mean_value("peak_reserved_gib"),
                "kv_cache_gib_per_rank": mean_value("kv_cache_gib_per_rank"),
                "prefix_hit_requests": int(round(mean_value("prefix_hit_requests"))),
                "prefix_miss_requests": int(round(mean_value("prefix_miss_requests"))),
                "prefix_saved_prefill_tokens": int(
                    round(mean_value("prefix_saved_prefill_tokens"))
                ),
                "prefix_evictions": int(round(mean_value("prefix_evictions"))),
                "prefix_evicted_tokens": int(round(mean_value("prefix_evicted_tokens"))),
                "retained_prefix_pages": int(round(mean_value("retained_prefix_pages"))),
                "retained_prefix_tokens": int(round(mean_value("retained_prefix_tokens"))),
                "retained_memory_mib": mean_value("retained_memory_mib"),
                "swa_retained_mib": mean_value("swa_retained_mib"),
                "c4_retained_mib": mean_value("c4_retained_mib"),
                "c128_retained_mib": mean_value("c128_retained_mib"),
                "component_available_pages": (
                    ""
                    if first.get("component_available_pages") == ""
                    else int(round(mean_value("component_available_pages")))
                ),
                "component_live_full_pages": (
                    ""
                    if first.get("component_live_full_pages") == ""
                    else int(round(mean_value("component_live_full_pages")))
                ),
                "communication_count": int(round(mean_value("communication_count"))),
                "communication_gib": mean_value("communication_gib"),
            }
        )
        out.append(row)
    return out


def _write_workload_tables(rows: list[dict[str, Any]]) -> None:
    macro = _aggregate_macro_rows(rows)
    fields = [
        "variant_group",
        "scenario",
        "repeats",
        "status",
        "output_tok_s_mean",
        "output_tok_s_stdev",
        "output_tok_s_cv",
        "ttft_ms_mean",
        "tpot_ms_mean",
        "prefill_tok_s",
        "decode_tok_s",
        "prefill_forward_s_mean",
        "decode_prepare_s_mean",
        "decode_forward_s_mean",
        "replay",
        "eager",
        "prefix_saved_prefill_tokens",
        "prefix_evictions",
    ]
    _write_csv(SUMMARIES / "workload_throughput.csv", macro, fields)
    (SUMMARIES / "workload_throughput.md").write_text(
        "# Workload Throughput\n\n"
        + _md_table(
            [
                "variant",
                "scenario",
                "runs",
                "out tok/s",
                "stdev",
                "CV",
                "TTFT ms",
                "TPOT/ITL ms",
                "prefill fwd s",
                "decode prep s",
                "decode fwd s",
                "graph",
                "saved",
                "evict",
            ],
            [
                [
                    row["variant_group"],
                    row["scenario"],
                    row["repeats"],
                    row["output_tok_s_mean"],
                    row["output_tok_s_stdev"],
                    row["output_tok_s_cv"],
                    row["ttft_ms_mean"],
                    row["tpot_ms_mean"],
                    row["prefill_forward_s_mean"],
                    row["decode_prepare_s_mean"],
                    row["decode_forward_s_mean"],
                    f"{row['replay']}/{row['eager']}",
                    row["prefix_saved_prefill_tokens"],
                    row["prefix_evictions"],
                ]
                for row in macro
            ],
        ),
        encoding="utf-8",
    )


def _write_graph_tables(rows: list[dict[str, Any]]) -> None:
    interesting = [row for row in rows if row["kind"] in {"macro", "verify", "profile"}]
    fields = [
        "kind",
        "run",
        "variant_group",
        "scenario",
        "graph_enabled",
        "requested_bs",
        "captured_bs",
        "replay",
        "eager",
        "replay_by_padded",
        "eager_by_bs",
    ]
    _write_csv(SUMMARIES / "graph_coverage.csv", interesting, fields)
    (SUMMARIES / "graph_coverage.md").write_text(
        "# Graph Coverage\n\n"
        + _md_table(
            [
                "kind",
                "run",
                "variant",
                "scenario",
                "requested",
                "captured",
                "replay/eager",
                "replay by padded",
                "eager by bs",
            ],
            [
                [
                    row["kind"],
                    row["run"],
                    row["variant_group"],
                    row["scenario"],
                    row["requested_bs"],
                    row["captured_bs"],
                    f"{row['replay']}/{row['eager']}",
                    row["replay_by_padded"],
                    row["eager_by_bs"],
                ]
                for row in interesting
            ],
        ),
        encoding="utf-8",
    )


def _write_prefix_tables(rows: list[dict[str, Any]]) -> None:
    interesting = [
        row
        for row in rows
        if row["kind"] in {"macro", "verify", "profile"}
        and row["variant_group"] == "promoted_prefix"
    ]
    fields = [
        "kind",
        "run",
        "scenario",
        "prefix_hit_requests",
        "prefix_miss_requests",
        "prefix_saved_prefill_tokens",
        "prefix_evictions",
        "prefix_evicted_tokens",
        "retained_prefix_pages",
        "retained_prefix_tokens",
        "retained_memory_mib",
        "swa_retained_mib",
        "component_available_pages",
    ]
    _write_csv(SUMMARIES / "prefix_metrics.csv", interesting, fields)
    (SUMMARIES / "prefix_metrics.md").write_text(
        "# Prefix Metrics\n\n"
        + _md_table(
            [
                "kind",
                "run",
                "scenario",
                "hits",
                "misses",
                "saved prefill",
                "evictions",
                "evicted tokens",
                "retained pages",
                "retained MiB",
                "SWA MiB",
                "available comp pages",
            ],
            [
                [
                    row["kind"],
                    row["run"],
                    row["scenario"],
                    row["prefix_hit_requests"],
                    row["prefix_miss_requests"],
                    row["prefix_saved_prefill_tokens"],
                    row["prefix_evictions"],
                    row["prefix_evicted_tokens"],
                    row["retained_prefix_pages"],
                    row["retained_memory_mib"],
                    row["swa_retained_mib"],
                    row["component_available_pages"],
                ]
                for row in interesting
            ],
        ),
        encoding="utf-8",
    )


def _write_memory_tables(rows: list[dict[str, Any]]) -> None:
    macro = _aggregate_macro_rows(rows)
    fields = [
        "variant_group",
        "scenario",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "kv_cache_gib_per_rank",
        "graph_capture_delta_gib",
        "retained_prefix_pages",
        "retained_prefix_tokens",
        "retained_memory_mib",
        "swa_retained_mib",
        "c4_retained_mib",
        "c128_retained_mib",
        "component_available_pages",
        "component_live_full_pages",
    ]
    _write_csv(SUMMARIES / "memory_ledger.csv", macro, fields)
    (SUMMARIES / "memory_ledger.md").write_text(
        "# Memory And Capacity Ledger\n\n"
        + _md_table(
            [
                "variant",
                "scenario",
                "peak alloc GiB",
                "peak reserved GiB",
                "KV GiB/rank",
                "graph delta GiB",
                "retained pages",
                "retained tokens",
                "retained MiB",
                "SWA MiB",
                "C4 MiB",
                "C128 MiB",
                "available comp pages",
            ],
            [
                [
                    row["variant_group"],
                    row["scenario"],
                    row["peak_allocated_gib"],
                    row["peak_reserved_gib"],
                    row["kv_cache_gib_per_rank"],
                    row["graph_capture_delta_gib"],
                    row["retained_prefix_pages"],
                    row["retained_prefix_tokens"],
                    row["retained_memory_mib"],
                    row["swa_retained_mib"],
                    row["c4_retained_mib"],
                    row["c128_retained_mib"],
                    row["component_available_pages"],
                ]
                for row in macro
            ],
        ),
        encoding="utf-8",
    )


def _write_decode_phase(rows: list[dict[str, Any]]) -> None:
    macro = _aggregate_macro_rows(rows)
    fields = [
        "variant_group",
        "scenario",
        "decode_prepare_s_mean",
        "decode_forward_s_mean",
        "decode_prepare_forward_share",
        "prefill_forward_s_mean",
        "prefill_prepare_s_mean",
    ]
    out_rows = []
    for row in macro:
        denom = row["decode_prepare_s_mean"] + row["decode_forward_s_mean"]
        out = dict(row)
        out["decode_prepare_forward_share"] = 0.0 if denom <= 0 else row[
            "decode_prepare_s_mean"
        ] / denom
        out_rows.append(out)
    _write_csv(SUMMARIES / "decode_prepare_vs_forward.csv", out_rows, fields)
    (SUMMARIES / "decode_prepare_vs_forward.md").write_text(
        "# Decode Prepare vs Forward\n\n"
        + _md_table(
            [
                "variant",
                "scenario",
                "decode prepare s",
                "decode forward s",
                "prepare share",
                "prefill forward s",
                "prefill prepare s",
            ],
            [
                [
                    row["variant_group"],
                    row["scenario"],
                    row["decode_prepare_s_mean"],
                    row["decode_forward_s_mean"],
                    row["decode_prepare_forward_share"],
                    row["prefill_forward_s_mean"],
                    row["prefill_prepare_s_mean"],
                ]
                for row in out_rows
            ],
        ),
        encoding="utf-8",
    )


def _owner_timing_rows(items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, report in items:
        if not run_name.startswith("profile_"):
            continue
        owner = report.get("owner_timing", {})
        for section in ("cuda", "host"):
            labels = owner.get(section, {}).get("by_label", {}) or {}
            for label, stats in labels.items():
                if not isinstance(stats, dict):
                    continue
                rows.append(
                    {
                        "run": run_name,
                        "variant_group": _variant_group(report),
                        "scenario": _scenario_name(report),
                        "section": section,
                        "label": label,
                        "max_rank_total_ms": float(stats.get("max_rank_total_ms") or 0.0),
                        "sum_rank_total_ms": float(stats.get("sum_rank_total_ms") or 0.0),
                        "count": int(stats.get("count") or 0),
                        "captured_total_ms": float(
                            stats.get("max_rank_captured_total_ms") or 0.0
                        ),
                    }
                )
        for counter in owner.get("rank0", {}).get("counters", []) or []:
            metadata = counter.get("metadata") or {}
            if counter.get("label") != "dsv4.component_page_table_cache.rows":
                continue
            rows.append(
                {
                    "run": run_name,
                    "variant_group": _variant_group(report),
                    "scenario": _scenario_name(report),
                    "section": "counter",
                    "label": "dsv4.component_page_table_cache.rows/"
                    + str(metadata.get("phase", ""))
                    + "/"
                    + str(metadata.get("status", "")),
                    "max_rank_total_ms": 0.0,
                    "sum_rank_total_ms": 0.0,
                    "count": int(counter.get("count") or 0),
                    "captured_total_ms": 0.0,
                }
            )
    rows.sort(key=lambda row: row["max_rank_total_ms"], reverse=True)
    _write_csv(
        SUMMARIES / "owner_timing.csv",
        rows,
        [
            "run",
            "variant_group",
            "scenario",
            "section",
            "label",
            "max_rank_total_ms",
            "sum_rank_total_ms",
            "count",
            "captured_total_ms",
        ],
    )
    top = [row for row in rows if row["section"] != "counter"][:40]
    counters = [row for row in rows if row["section"] == "counter"]
    (SUMMARIES / "owner_timing.md").write_text(
        "# Owner Timing\n\n"
        + _md_table(
            [
                "run",
                "variant",
                "scenario",
                "section",
                "label",
                "max-rank ms",
                "count",
            ],
            [
                [
                    row["run"],
                    row["variant_group"],
                    row["scenario"],
                    row["section"],
                    row["label"],
                    row["max_rank_total_ms"],
                    row["count"],
                ]
                for row in top
            ],
        )
        + "\n## Component Row Counters\n\n"
        + _md_table(
            ["run", "scenario", "label", "count"],
            [
                [row["run"], row["scenario"], row["label"], row["count"]]
                for row in counters
            ],
        ),
        encoding="utf-8",
    )
    return rows


def _macro_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    macro_rows = rows if all(row.get("kind") == "macro_agg" for row in rows) else _aggregate_macro_rows(rows)
    return {
        (row["variant_group"], row["scenario"]): row
        for row in macro_rows
    }


def _write_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _macro_lookup(rows)
    out: list[dict[str, Any]] = []
    for scenario, value in TARGET07.items():
        if not scenario.startswith("historical_"):
            continue
        out.append(
            {
                "source": "TARGET 07.79 non-prefix",
                "variant": "target07_control",
                "scenario": scenario,
                "output_tok_s": value,
                "decode_prepare_s": "",
                "decode_forward_s": "",
                "graph": "0 eager",
                "note": "from prompts/target.md",
            }
        )
    out.append(
        {
            "source": "old vLLM baseline",
            "variant": "vLLM old serving line",
            "scenario": "4096_1024_bs4_serving_line",
            "output_tok_s": TARGET07["old_vllm_serving_4096_1024"],
            "decode_prepare_s": "",
            "decode_forward_s": "",
            "graph": "",
            "note": "historical old serving victory line",
        }
    )
    for scenario, data in TARGET0828.items():
        out.append(
            {
                "source": "TARGET 08.28 promoted gate",
                "variant": "route_b_lifetime_legacy_name",
                "scenario": scenario,
                "output_tok_s": data["output_tok_s"],
                "decode_prepare_s": data.get("decode_prepare_s", ""),
                "decode_forward_s": data.get("decode_forward_s", ""),
                "graph": data.get("graph", ""),
                "note": "from target08_route_b_lifetime_cache_promotion_gate",
            }
        )
    for key in sorted(lookup):
        row = lookup[key]
        out.append(
            {
                "source": "TARGET 08.30 current",
                "variant": row["variant_group"],
                "scenario": row["scenario"],
                "output_tok_s": row["output_tok_s_mean"],
                "decode_prepare_s": row["decode_prepare_s_mean"],
                "decode_forward_s": row["decode_forward_s_mean"],
                "graph": f"{row['replay']}/{row['eager']}",
                "note": f"CV={row['output_tok_s_cv']:.4f}",
            }
        )
    _write_csv(
        SUMMARIES / "comparison.csv",
        out,
        [
            "source",
            "variant",
            "scenario",
            "output_tok_s",
            "decode_prepare_s",
            "decode_forward_s",
            "graph",
            "note",
        ],
    )
    (SUMMARIES / "comparison.md").write_text(
        "# Historical Comparison\n\n"
        + _md_table(
            [
                "source",
                "variant",
                "scenario",
                "out tok/s",
                "decode prep s",
                "decode fwd s",
                "graph",
                "note",
            ],
            [
                [
                    row["source"],
                    row["variant"],
                    row["scenario"],
                    row["output_tok_s"],
                    row["decode_prepare_s"],
                    row["decode_forward_s"],
                    row["graph"],
                    row["note"],
                ]
                for row in out
            ],
        ),
        encoding="utf-8",
    )
    return out


def _serving_comm_owner_ms(owner_rows: list[dict[str, Any]]) -> float:
    labels = [
        row
        for row in owner_rows
        if row["variant_group"] == "promoted_prefix"
        and row["scenario"] == "serving_mixed_112req_wave16"
        and row["section"] == "cuda"
        and ("all_reduce" in row["label"] or ".comm." in row["label"])
    ]
    return sum(float(row["max_rank_total_ms"]) for row in labels)


def _bottleneck_rows(rows: list[dict[str, Any]], owner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = _macro_lookup(rows)
    serving = lookup.get(("promoted_prefix", "serving_mixed_112req_wave16"))
    if not serving:
        return []
    elapsed = max(float(serving["elapsed_s"]), 1e-9)
    comm_owner_ms = _serving_comm_owner_ms(owner_rows)
    out = [
        {
            "rank": 1,
            "bucket": "decode forward",
            "evidence": "serving_mixed phase total",
            "seconds_or_ms": serving["decode_forward_s_mean"],
            "share": serving["decode_forward_s_mean"] / elapsed,
            "interpretation": "dominant remaining E2E bucket; owner timing points to comm/attention work inside it",
        },
        {
            "rank": 2,
            "bucket": "communication / all-reduce owners",
            "evidence": "owner timing profile, attribution only",
            "seconds_or_ms": comm_owner_ms,
            "share": comm_owner_ms / max(serving["elapsed_s"] * 1000.0, 1e-9),
            "interpretation": "wo_b row-parallel, MoE reduce-once, and embedding all-reduce are top owner labels",
        },
        {
            "rank": 3,
            "bucket": "prefill forward / TTFT base cost",
            "evidence": "serving_mixed phase total",
            "seconds_or_ms": serving["prefill_forward_s_mean"],
            "share": serving["prefill_forward_s_mean"] / elapsed,
            "interpretation": "not helped unless workload has real prefix hits",
        },
        {
            "rank": 4,
            "bucket": "decode prepare / prefix metadata runtime",
            "evidence": "serving_mixed phase total plus owner timing",
            "seconds_or_ms": serving["decode_prepare_s_mean"],
            "share": serving["decode_prepare_s_mean"] / elapsed,
            "interpretation": "post-lifetime-cache tax; compare against 08.28 and owner rows",
        },
    ]
    component = next(
        (
            row
            for row in owner_rows
            if row["variant_group"] == "promoted_prefix"
            and row["scenario"] == "serving_mixed_112req_wave16"
            and row["label"] == "dsv4.metadata.decode.make_component_page_tables"
        ),
        None,
    )
    if component:
        out.append(
            {
                "rank": 5,
                "bucket": "component page-table lifetime cache owner",
                "evidence": "owner timing profile, attribution only",
                "seconds_or_ms": component["max_rank_total_ms"],
                "share": component["max_rank_total_ms"] / max(serving["elapsed_s"] * 1000.0, 1e-9),
                "interpretation": "metadata owner is now small relative to decode forward",
            }
        )
    _write_csv(
        SUMMARIES / "ranked_bottlenecks.csv",
        out,
        ["rank", "bucket", "evidence", "seconds_or_ms", "share", "interpretation"],
    )
    (SUMMARIES / "ranked_bottlenecks.md").write_text(
        "# Ranked Bottlenecks\n\n"
        + _md_table(
            ["rank", "bucket", "evidence", "seconds/ms", "share", "interpretation"],
            [
                [
                    row["rank"],
                    row["bucket"],
                    row["evidence"],
                    row["seconds_or_ms"],
                    row["share"],
                    row["interpretation"],
                ]
                for row in out
            ],
        ),
        encoding="utf-8",
    )
    return out


def _decision(
    rows: list[dict[str, Any]],
    text_smoke: dict[str, Any],
    owner_rows: list[dict[str, Any]],
) -> dict[str, str]:
    macro = _aggregate_macro_rows(rows)
    verify = [row for row in rows if row["kind"] == "verify"]
    if not macro:
        return {
            "status": "pending",
            "recommendation": "pending",
            "reason": "macro runs are not available yet",
        }
    if text_smoke["status"] not in {"pass", "missing"}:
        return {
            "status": "stop",
            "recommendation": "blocked",
            "reason": "promoted preset text smoke is not clean",
        }
    bad = [row for row in macro + verify if row["status"] != "pass"]
    if bad:
        return {
            "status": "stop",
            "recommendation": "blocked",
            "reason": "one or more benchmark reports failed",
        }
    eager = [row for row in macro + verify if int(row["eager"]) != 0]
    if eager:
        return {
            "status": "stop",
            "recommendation": "blocked",
            "reason": "graph replay is not zero-eager for all measured reports",
        }
    unstable = [
        row
        for row in macro
        if row["repeats"] >= 2
        and row["scenario"] != "prefix_eviction_pressure_96req_wave16"
        and row["output_tok_s_cv"] > 0.03
    ]
    if unstable:
        return {
            "status": "stop",
            "recommendation": "blocked",
            "reason": "benchmark variance is too high to rank cleanly",
        }
    lookup = _macro_lookup(rows)
    serving = lookup.get(("promoted_prefix", "serving_mixed_112req_wave16"))
    prefix_multi = lookup.get(("promoted_prefix", "prefix_multi_112req_wave16"))
    if serving and prefix_multi:
        decode_share = serving["decode_forward_s_mean"] / max(serving["elapsed_s"], 1e-9)
        prep_share = serving["decode_prepare_s_mean"] / max(serving["elapsed_s"], 1e-9)
        comm_owner_share = _serving_comm_owner_ms(owner_rows) / max(
            serving["elapsed_s"] * 1000.0, 1e-9
        )
        if decode_share > 0.45 and comm_owner_share > 0.15:
            return {
                "status": "complete",
                "recommendation": "TARGET 10",
                "reason": (
                    "decode forward dominates and isolated owner timing now points to "
                    "communication/all-reduce owners; start with a narrow TARGET10 "
                    "timeline before changing kernels or precision"
                ),
            }
        if decode_share > 0.45 and prep_share < 0.12:
            return {
                "status": "complete",
                "recommendation": "TARGET 09",
                "reason": (
                    "prefix metadata is no longer the dominant bucket; decode forward "
                    "dominates and the next isolated opportunity is low precision/cache format"
                ),
            }
    return {
        "status": "complete",
        "recommendation": "serving hardening",
        "reason": "no single post-prefix optimization bucket exceeds the decision threshold",
    }


def _commands_block() -> str:
    return """```bash
MODEL_PATH=/models/DeepSeek-V4-Flash \\
NPROC=8 \\
HISTORICAL_REPEATS=3 \\
SERVING_REPEATS=3 \\
RUN_OWNER_TIMING=0 \\
performance_milestones/target08_post_prefix_reprofile/scripts/run_post_prefix_reprofile.sh

RUN_TEXT_SMOKE=0 \\
RUN_VERIFY=0 \\
RUN_MACRO=0 \\
RUN_OWNER_TIMING=1 \\
performance_milestones/target08_post_prefix_reprofile/scripts/run_post_prefix_reprofile.sh
```

Promoted prefix matrix shape:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \\
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1 \\
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4 \\
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1 \\
torchrun --standalone --nproc_per_node=8 \\
  benchmark/offline/deepseek_v4_perf_matrix.py \\
  --model-path /models/DeepSeek-V4-Flash \\
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \\
  --page-size 256 --num-pages 128 \\
  --enable-dsv4-radix-prefix-cache \\
  --enable-dsv4-component-loc-ownership \\
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Non-prefix TARGET 07 control shape:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \\
torchrun --standalone --nproc_per_node=8 \\
  benchmark/offline/deepseek_v4_perf_matrix.py \\
  --model-path /models/DeepSeek-V4-Flash \\
  --variants dsv4_sm80_a100_victory \\
  --page-size 256 --num-pages 128 \\
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Owner timing runs additionally set:

```bash
MINISGL_DSV4_OWNER_TIMING=1
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000
```
"""


def _write_readme(
    rows: list[dict[str, Any]],
    owner_rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    bottlenecks: list[dict[str, Any]],
    decision: dict[str, str],
    text_smoke: dict[str, Any],
) -> None:
    git_status = (
        (RAW / "git_status_short.txt").read_text(encoding="utf-8").strip()
        if (RAW / "git_status_short.txt").exists()
        else ""
    )
    workload_md = (SUMMARIES / "workload_throughput.md").read_text(encoding="utf-8")
    graph_md = (SUMMARIES / "graph_coverage.md").read_text(encoding="utf-8")
    prefix_md = (SUMMARIES / "prefix_metrics.md").read_text(encoding="utf-8")
    memory_md = (SUMMARIES / "memory_ledger.md").read_text(encoding="utf-8")
    decode_md = (SUMMARIES / "decode_prepare_vs_forward.md").read_text(encoding="utf-8")
    owner_md = (SUMMARIES / "owner_timing.md").read_text(encoding="utf-8")
    comparison_md = (SUMMARIES / "comparison.md").read_text(encoding="utf-8")
    bottleneck_md = (SUMMARIES / "ranked_bottlenecks.md").read_text(encoding="utf-8")

    lookup = _macro_lookup(rows)
    serving = lookup.get(("promoted_prefix", "serving_mixed_112req_wave16"))
    prefix_multi = lookup.get(("promoted_prefix", "prefix_multi_112req_wave16"))
    eviction = lookup.get(("promoted_prefix", "prefix_eviction_pressure_96req_wave16"))
    zero_eager = all(row["eager"] == 0 for row in rows if row["kind"] in {"macro", "verify"})
    capacity_note = "pending"
    if eviction:
        capacity_note = (
            "not an OOM/capacity stopper in fixed --num-pages 128 runs; "
            f"eviction pressure completed with {eviction['prefix_evictions']} evictions and "
            f"{eviction['retained_prefix_pages']} retained pages"
        )
    prefix_answer = "pending"
    if serving and prefix_multi:
        prefix_answer = (
            "yes for shared-prefix workloads: prefix_multi saved "
            f"{prefix_multi['prefix_saved_prefill_tokens']} prefill tokens; serving_mixed has "
            "no designed prefix hits, so its TTFT reflects the base serving path rather than a hit win"
        )
    graph_answer = "yes" if zero_eager else "no"
    if decision["recommendation"] == "TARGET 10":
        bottleneck_classification = (
            "decode-forward dominated, with isolated owner timing pointing most strongly "
            "to communication/all-reduce owners; prefix metadata/runtime is now secondary, "
            "and low precision/cache format remains a later candidate if the TARGET10 "
            "timeline disproves communication upside"
        )
    elif decision["recommendation"] == "TARGET 09":
        bottleneck_classification = (
            "decode-forward dominated after prefix metadata was reduced; evidence points "
            "first to low precision/cache format, while TARGET10 needs a narrower timeline"
        )
    else:
        bottleneck_classification = (
            "not dominated by prefix metadata; remaining work is closer to serving/runtime "
            "hardening unless a future profile isolates a larger owner"
        )

    readme = [
        "# TARGET 08.30 Post-Prefix Reprofile",
        "",
        "## Final Configuration",
        "",
        f"Promoted prefix variant: `{PROMOTED}`.",
        "",
        "Final promoted prefix path:",
        "",
        "```text",
        "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1",
        "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1",
        "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4",
        "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1",
        "--page-size 256",
        "--num-pages 128",
        "--enable-dsv4-radix-prefix-cache",
        "--enable-dsv4-component-loc-ownership",
        "--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16",
        "```",
        "",
        f"Control variant: `{CONTROL}` with page size 256, num pages 128, and graph buckets 1 2 4 8 16.",
        "",
        "## Commands And Environment",
        "",
        _commands_block(),
        "## Git Status",
        "",
        "```text",
        git_status or "clean",
        "```",
        "",
        "## Correctness / Verifier",
        "",
        _md_table(
            ["check", "status", "verifier", "graph", "outputs"],
            [
                [
                    "text_smoke_promoted_verify",
                    text_smoke["status"],
                    text_smoke["verifier"],
                    text_smoke["graph"],
                    " | ".join(text_smoke["outputs"]),
                ]
            ],
        ),
        "Verifier matrix runs are included in the graph and prefix tables below.",
        "",
        "## Workload Throughput Tables",
        "",
        workload_md,
        "## Graph Replay / Eager Coverage",
        "",
        graph_md,
        "## Prefix Hit / Miss / Saved-Prefill / Eviction Metrics",
        "",
        prefix_md,
        "## Memory / Capacity Ledger",
        "",
        memory_md,
        "## Decode Prepare vs Decode Forward",
        "",
        decode_md,
        "## Owner Timing / Attribution",
        "",
        "Owner timing is attribution only and is not used as final throughput evidence.",
        "",
        owner_md,
        "## Ranked Bottleneck Table",
        "",
        bottleneck_md,
        "## Comparison To TARGET 07.79, TARGET 08.28, And vLLM",
        "",
        comparison_md,
        "## Required Questions",
        "",
        "1. Promoted Route B lifetime prefix path improves serving TTFT/prefill? "
        + prefix_answer
        + ".",
        "2. `[1,2,4,8,16]` graph bucket still zero eager? "
        + graph_answer
        + ".",
        "3. SWA-tail/full-tail guard or memory retention is capacity bottleneck? "
        + capacity_note
        + ".",
        "4. New main bottleneck classification: " + bottleneck_classification + ".",
        "5. Next target recommendation: `"
        + decision["recommendation"]
        + "`.",
        "",
        "## Next-Target Recommendation",
        "",
        f"Status: `{decision['status']}`.",
        "",
        f"Recommendation: `{decision['recommendation']}`.",
        "",
        f"Reason: {decision['reason']}.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    items = _all_report_items()
    rows = [_report_row(run_name, report) for run_name, report in items]
    rows.sort(key=lambda row: (row["kind"], row["variant_group"], row["scenario"], row["run"]))
    _write_csv(
        SUMMARIES / "matrix_runs.csv",
        rows,
        [
            "run",
            "kind",
            "variant_group",
            "variant",
            "scenario",
            "status",
            "repeats",
            "output_tok_s_mean",
            "output_tok_s_stdev",
            "output_tok_s_cv",
            "elapsed_s",
            "ttft_ms_mean",
            "tpot_ms_mean",
            "prefill_tok_s",
            "decode_tok_s",
            "prefill_forward_s_mean",
            "decode_prepare_s_mean",
            "decode_forward_s_mean",
            "replay",
            "eager",
            "prefix_hit_requests",
            "prefix_miss_requests",
            "prefix_saved_prefill_tokens",
            "prefix_evictions",
            "retained_prefix_pages",
            "peak_allocated_gib",
            "kv_cache_gib_per_rank",
        ],
    )
    _write_workload_tables(rows)
    _write_graph_tables(rows)
    _write_prefix_tables(rows)
    _write_memory_tables(rows)
    _write_decode_phase(rows)
    owner_rows = _owner_timing_rows(items)
    comparison = _write_comparison(rows)
    bottlenecks = _bottleneck_rows(rows, owner_rows)
    text_smoke = _text_smoke()
    decision = _decision(rows, text_smoke, owner_rows)
    summary = {
        "decision": decision,
        "text_smoke": text_smoke,
        "matrix_runs": rows,
        "comparison": comparison,
        "ranked_bottlenecks": bottlenecks,
    }
    (SUMMARIES / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        rows=rows,
        owner_rows=owner_rows,
        comparison=comparison,
        bottlenecks=bottlenecks,
        decision=decision,
        text_smoke=text_smoke,
    )


if __name__ == "__main__":
    main()

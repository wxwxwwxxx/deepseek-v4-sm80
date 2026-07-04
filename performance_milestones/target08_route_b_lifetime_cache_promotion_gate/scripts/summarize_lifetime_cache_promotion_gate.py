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
PREV_LIFETIME = REPO / "performance_milestones" / "target08_sglang_aligned_route_b_metadata_lifetime"
SUMMARIES.mkdir(parents=True, exist_ok=True)

VARIANT = "dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime"
VERIFY_ENV = "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY"
TARGET_0827_SERVING_TOK_S = 162.4726

THROUGHPUT_GROUPS = {
    "serving_mixed": "serving_mixed_r*_lifetime",
    "prefix_multi": "prefix_multi_r*_lifetime",
    "prefix_eviction": "prefix_eviction_r*_lifetime",
    "decode_ladder": "decode_ladder_lifetime",
}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is None:
            return ""
        return str(value).replace("|", "\\|")

    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(out) + "\n"


def _load_reports(run_dir: Path) -> list[dict[str, Any]]:
    matrix = run_dir / "matrix.jsonl"
    reports: list[dict[str, Any]] = []
    if not matrix.exists():
        return reports
    for line in matrix.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(row["report_path"])
        if not path.is_absolute():
            path = REPO / path
        if path.exists():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def _load_single_report(run_dir: Path) -> dict[str, Any] | None:
    reports = _load_reports(run_dir)
    return reports[0] if reports else None


def _prefix_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    prefix = metrics.get("prefix_cache", {}).get("rank0_final", {})
    if not prefix:
        prefix = report.get("config", {}).get("prefix_cache_metrics", {})
    return prefix or {}


def _graph(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    return config.get("graph_runner_case") or config.get("graph_runner") or {}


def _cache_counters(report: dict[str, Any]) -> dict[str, int]:
    out = {"dirty_rows": 0, "clean_rows": 0, "total_rows": 0}
    counters = report.get("owner_timing", {}).get("rank0", {}).get("counters", [])
    for counter in counters:
        if counter.get("label") != "dsv4.component_page_table_cache.rows":
            continue
        metadata = counter.get("metadata") or {}
        if metadata.get("phase") != "decode":
            continue
        count = int(counter.get("count") or 0)
        if metadata.get("status") == "dirty":
            out["dirty_rows"] += count
        elif metadata.get("status") == "clean":
            out["clean_rows"] += count
    out["total_rows"] = out["dirty_rows"] + out["clean_rows"]
    return out


def _timing(report: dict[str, Any], section: str, label: str) -> float:
    stats = report.get("owner_timing", {}).get(section, {}).get("by_label", {}).get(label, {})
    return float(stats.get("max_rank_total_ms") or 0.0)


def _metrics(report: dict[str, Any], run_name: str) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    phase = metrics.get("phase_totals", {})
    graph = _graph(report)
    prefix = _prefix_metrics(report)
    counters = _cache_counters(report)
    raw_env = report.get("variant", {}).get("raw_dsv4_sm80_env", {})
    active = report.get("variant", {}).get("active_dsv4_toggles", [])
    scenario = report.get("scenario", {})
    if isinstance(scenario, dict):
        scenario_name = scenario.get("name", "")
    else:
        scenario_name = str(scenario)
    return {
        "run": run_name,
        "scenario": scenario_name,
        "status": report.get("status", ""),
        "verifier_enabled": raw_env.get(VERIFY_ENV) == "1" or VERIFY_ENV in active,
        "output_tok_s": float(metrics.get("end_to_end_output_tokens_per_s") or 0.0),
        "decode_tok_s": float(metrics.get("decode_tokens_per_s") or 0.0),
        "prefill_tok_s": float(metrics.get("prefill_tokens_per_s") or 0.0),
        "decode_prepare_s": float(phase.get("decode_prepare_s") or 0.0),
        "decode_forward_s": float(phase.get("decode_forward_s") or 0.0),
        "prefill_forward_s": float(phase.get("prefill_forward_s") or 0.0),
        "elapsed_s": float(metrics.get("elapsed_s") or 0.0),
        "replay": int(graph.get("replay_count") or 0),
        "eager": int(graph.get("eager_decode_count") or 0),
        "captured_bs": graph.get("captured_bs", []),
        "requested_bs": graph.get("requested_bs", []),
        "hit_requests": int(prefix.get("hit_requests") or 0),
        "saved_prefill_tokens": int(prefix.get("saved_prefill_tokens") or 0),
        "evictions": int(prefix.get("evictions") or 0),
        "evicted_tokens": int(prefix.get("evicted_tokens") or 0),
        "retained_prefix_pages": int(prefix.get("retained_prefix_pages") or 0),
        "dirty_rows": counters["dirty_rows"],
        "clean_rows": counters["clean_rows"],
        "component_table_ms": _timing(
            report,
            "cuda",
            "dsv4.metadata.decode.make_component_page_tables",
        ),
        "raw_env": raw_env,
    }


def _all_matrix_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in RAW.iterdir() if path.is_dir()):
        for report in _load_reports(run_dir):
            rows.append(_metrics(report, run_dir.name))
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate_group(name: str, pattern: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(RAW.glob(pattern)):
        if run_dir.is_dir():
            report = _load_single_report(run_dir)
            if report is not None:
                rows.append(_metrics(report, run_dir.name))
    values = {
        key: [float(row[key]) for row in rows]
        for key in [
            "output_tok_s",
            "decode_tok_s",
            "decode_prepare_s",
            "decode_forward_s",
            "elapsed_s",
        ]
    }
    return {
        "group": name,
        "scenario": rows[0]["scenario"] if rows else "",
        "runs": len(rows),
        "all_pass": bool(rows) and all(row["status"] == "pass" for row in rows),
        "output_tok_s_mean": _mean(values["output_tok_s"]),
        "output_tok_s_stdev": _stdev(values["output_tok_s"]),
        "decode_tok_s_mean": _mean(values["decode_tok_s"]),
        "decode_prepare_s_mean": _mean(values["decode_prepare_s"]),
        "decode_forward_s_mean": _mean(values["decode_forward_s"]),
        "elapsed_s_mean": _mean(values["elapsed_s"]),
        "replay": max((int(row["replay"]) for row in rows), default=0),
        "eager": max((int(row["eager"]) for row in rows), default=0),
        "hit_requests": max((int(row["hit_requests"]) for row in rows), default=0),
        "saved_prefill_tokens": max(
            (int(row["saved_prefill_tokens"]) for row in rows),
            default=0,
        ),
        "evictions": max((int(row["evictions"]) for row in rows), default=0),
        "evicted_tokens": max((int(row["evicted_tokens"]) for row in rows), default=0),
    }


def _write_matrix_runs(rows: list[dict[str, Any]]) -> None:
    fields = [
        "run",
        "scenario",
        "status",
        "verifier_enabled",
        "output_tok_s",
        "decode_tok_s",
        "decode_prepare_s",
        "decode_forward_s",
        "elapsed_s",
        "replay",
        "eager",
        "captured_bs",
        "requested_bs",
        "hit_requests",
        "saved_prefill_tokens",
        "evictions",
        "evicted_tokens",
        "retained_prefix_pages",
        "dirty_rows",
        "clean_rows",
        "component_table_ms",
    ]
    _write_csv(SUMMARIES / "matrix_runs.csv", rows, fields)
    (SUMMARIES / "graph_replay.md").write_text(
        "# Graph Replay\n\n"
        + _md_table(
            [
                "run",
                "scenario",
                "captured bs",
                "requested bs",
                "replay/eager",
                "verifier",
            ],
            [
                [
                    row["run"],
                    row["scenario"],
                    row["captured_bs"],
                    row["requested_bs"],
                    f"{row['replay']}/{row['eager']}",
                    row["verifier_enabled"],
                ]
                for row in rows
            ],
        ),
        encoding="utf-8",
    )


def _write_throughput() -> list[dict[str, Any]]:
    rows = [_aggregate_group(name, pattern) for name, pattern in THROUGHPUT_GROUPS.items()]
    fields = [
        "group",
        "scenario",
        "runs",
        "all_pass",
        "output_tok_s_mean",
        "output_tok_s_stdev",
        "decode_tok_s_mean",
        "decode_prepare_s_mean",
        "decode_forward_s_mean",
        "elapsed_s_mean",
        "replay",
        "eager",
        "hit_requests",
        "saved_prefill_tokens",
        "evictions",
        "evicted_tokens",
    ]
    _write_csv(SUMMARIES / "throughput_by_workload.csv", rows, fields)
    (SUMMARIES / "throughput_by_workload.md").write_text(
        "# Throughput By Workload\n\n"
        + _md_table(
            [
                "group",
                "scenario",
                "runs",
                "output tok/s",
                "stdev",
                "decode prepare s",
                "decode forward s",
                "graph replay/eager",
                "saved prefill",
                "evictions",
            ],
            [
                [
                    row["group"],
                    row["scenario"],
                    row["runs"],
                    row["output_tok_s_mean"],
                    row["output_tok_s_stdev"],
                    row["decode_prepare_s_mean"],
                    row["decode_forward_s_mean"],
                    f"{row['replay']}/{row['eager']}",
                    row["saved_prefill_tokens"],
                    row["evictions"],
                ]
                for row in rows
            ],
        ),
        encoding="utf-8",
    )
    return rows


def _write_baseline_comparison(current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prior = _read_csv(PREV_LIFETIME / "summaries" / "throughput_repeat.csv")
    wanted = {
        "phase1 prefix on",
        "Route B graph baseline",
        "Route B direct C4",
        "Route B direct C4 + lifetime cache",
    }
    rows: list[dict[str, Any]] = []
    for row in prior:
        if row.get("mode") not in wanted:
            continue
        rows.append(
            {
                "mode": row["mode"],
                "scenario": "serving_mixed_112req_wave16",
                "runs": int(row["runs"]),
                "output_tok_s_mean": float(row["output_tok_s_mean"]),
                "decode_prepare_s_mean": float(row["decode_prepare_s_mean"]),
                "decode_forward_s_mean": float(row["decode_forward_s_mean"]),
                "replay": int(row["replay"]),
                "eager": int(row["eager"]),
                "source": "08.27/08.26 frozen summary",
            }
        )
    serving = next((row for row in current if row["group"] == "serving_mixed"), None)
    if serving:
        rows.append(
            {
                "mode": "08.28 Route B direct C4 + lifetime cache",
                "scenario": serving["scenario"],
                "runs": serving["runs"],
                "output_tok_s_mean": serving["output_tok_s_mean"],
                "decode_prepare_s_mean": serving["decode_prepare_s_mean"],
                "decode_forward_s_mean": serving["decode_forward_s_mean"],
                "replay": serving["replay"],
                "eager": serving["eager"],
                "source": "08.28 current gate",
            }
        )
    _write_csv(
        SUMMARIES / "serving_comparison.csv",
        rows,
        [
            "mode",
            "scenario",
            "runs",
            "output_tok_s_mean",
            "decode_prepare_s_mean",
            "decode_forward_s_mean",
            "replay",
            "eager",
            "source",
        ],
    )
    return rows


def _write_prefix_metrics(rows: list[dict[str, Any]]) -> None:
    interesting = [
        row
        for row in rows
        if row["scenario"]
        in {
            "serving_mixed_112req_wave16",
            "prefix_multi_112req_wave16",
            "prefix_eviction_pressure_96req_wave16",
            "decode_ladder_bs16",
        }
    ]
    fields = [
        "run",
        "scenario",
        "hit_requests",
        "saved_prefill_tokens",
        "evictions",
        "evicted_tokens",
        "retained_prefix_pages",
    ]
    _write_csv(SUMMARIES / "prefix_eviction_metrics.csv", interesting, fields)
    (SUMMARIES / "prefix_eviction_metrics.md").write_text(
        "# Prefix And Eviction Metrics\n\n"
        + _md_table(
            [
                "run",
                "scenario",
                "hits",
                "saved prefill",
                "evictions",
                "evicted tokens",
                "retained pages",
            ],
            [
                [
                    row["run"],
                    row["scenario"],
                    row["hit_requests"],
                    row["saved_prefill_tokens"],
                    row["evictions"],
                    row["evicted_tokens"],
                    row["retained_prefix_pages"],
                ]
                for row in interesting
            ],
        ),
        encoding="utf-8",
    )


def _write_component_counters(rows: list[dict[str, Any]]) -> None:
    profiled = [row for row in rows if row["dirty_rows"] or row["clean_rows"]]
    fields = [
        "run",
        "scenario",
        "dirty_rows",
        "clean_rows",
        "component_table_ms",
        "replay",
        "eager",
    ]
    _write_csv(SUMMARIES / "component_row_counters.csv", profiled, fields)
    (SUMMARIES / "component_row_counters.md").write_text(
        "# Component Row Counters\n\n"
        + _md_table(
            [
                "run",
                "scenario",
                "dirty rows",
                "clean rows",
                "component table ms",
                "graph replay/eager",
            ],
            [
                [
                    row["run"],
                    row["scenario"],
                    row["dirty_rows"],
                    row["clean_rows"],
                    row["component_table_ms"],
                    f"{row['replay']}/{row['eager']}",
                ]
                for row in profiled
            ],
        ),
        encoding="utf-8",
    )


def _text_smoke() -> dict[str, Any]:
    variant_path = RAW / f"text_smoke_routeb_lifetime_verify.{VARIANT}.json"
    generic_path = RAW / "text_smoke_routeb_lifetime_verify.json"
    path = variant_path if variant_path.exists() else generic_path
    if not path.exists():
        return {"status": "missing", "outputs": [], "verifier_enabled": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_env = data.get("variant", {}).get("raw_dsv4_sm80_env", {})
    active = data.get("variant", {}).get("active_dsv4_toggles", [])
    return {
        "status": data.get("status", "missing"),
        "outputs": [item.get("text", "") for item in data.get("outputs", [])],
        "verifier_enabled": raw_env.get(VERIFY_ENV) == "1" or VERIFY_ENV in active,
    }


def _write_verifier(rows: list[dict[str, Any]], text: dict[str, Any]) -> list[dict[str, Any]]:
    verifier_runs = [
        row
        for row in rows
        if row["run"] in {"verify_serving_mixed_lifetime", "verify_prefix_eviction_lifetime"}
    ]
    out_rows: list[dict[str, Any]] = [
        {
            "check": "text_smoke",
            "scenario": "text_smoke",
            "status": text["status"],
            "verifier_enabled": text["verifier_enabled"],
            "replay": "",
            "eager": "",
            "output": " | ".join(text["outputs"]),
        }
    ]
    for row in verifier_runs:
        out_rows.append(
            {
                "check": row["run"],
                "scenario": row["scenario"],
                "status": row["status"],
                "verifier_enabled": row["verifier_enabled"],
                "replay": row["replay"],
                "eager": row["eager"],
                "output": "",
            }
        )
    fields = ["check", "scenario", "status", "verifier_enabled", "replay", "eager", "output"]
    _write_csv(SUMMARIES / "verifier_results.csv", out_rows, fields)
    (SUMMARIES / "verifier_results.md").write_text(
        "# Verifier Results\n\n"
        + _md_table(
            ["check", "scenario", "status", "verifier", "graph replay/eager", "output"],
            [
                [
                    row["check"],
                    row["scenario"],
                    row["status"],
                    row["verifier_enabled"],
                    (
                        ""
                        if row["replay"] == ""
                        else f"{row['replay']}/{row['eager']}"
                    ),
                    row["output"],
                ]
                for row in out_rows
            ],
        ),
        encoding="utf-8",
    )
    return out_rows


def _decision(
    throughput: list[dict[str, Any]],
    verifier: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    required_names = {
        "verify_serving_mixed_lifetime",
        "serving_mixed_r01_lifetime",
        "serving_mixed_r02_lifetime",
        "serving_mixed_r03_lifetime",
        "prefix_multi_r01_lifetime",
        "prefix_multi_r02_lifetime",
        "prefix_multi_r03_lifetime",
        "verify_prefix_eviction_lifetime",
        "prefix_eviction_r01_lifetime",
        "prefix_eviction_r02_lifetime",
        "decode_ladder_lifetime",
    }
    seen = {row["run"] for row in rows}
    if not required_names <= seen or any(row["status"] == "missing" for row in verifier):
        return "pending"
    if any(row["status"] != "pass" for row in rows):
        return "split fix target"
    if any(row["eager"] != 0 for row in rows):
        return "reject"
    if any(row["verifier_enabled"] is False for row in verifier):
        return "keep experimental"
    serving = next((row for row in throughput if row["group"] == "serving_mixed"), None)
    if not serving:
        return "pending"
    if serving["output_tok_s_mean"] < TARGET_0827_SERVING_TOK_S * 0.97:
        return "keep experimental"
    if serving["decode_prepare_s_mean"] > 2.0:
        return "keep experimental"
    return "promote"


def _commands_block() -> str:
    return """```bash
MODEL_PATH=/models/DeepSeek-V4-Flash \\
NPROC=8 \\
SERVING_REPEATS=3 \\
PREFIX_MULTI_REPEATS=3 \\
EVICTION_REPEATS=2 \\
performance_milestones/target08_route_b_lifetime_cache_promotion_gate/scripts/run_lifetime_cache_promotion_gate.sh
```

All matrix runs use:

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \\
torchrun --standalone --nproc_per_node=8 \\
  benchmark/offline/deepseek_v4_perf_matrix.py \\
  --model-path /models/DeepSeek-V4-Flash \\
  --variants dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime \\
  --page-size 256 --num-pages 128 \\
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \\
  --enable-dsv4-radix-prefix-cache \\
  --enable-dsv4-component-loc-ownership \\
  --keep-going
```

Verifier runs additionally set:

```bash
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1
```

Counter profile runs additionally set:

```bash
MINISGL_DSV4_OWNER_TIMING=1
MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000
```
"""


def _write_readme(
    *,
    throughput: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    verifier: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    decision: str,
) -> None:
    git_status = (RAW / "git_status_short.txt").read_text(encoding="utf-8") if (
        RAW / "git_status_short.txt"
    ).exists() else ""
    fixes = [
        "Preserved `MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY` across benchmark/text-smoke variant env reset.",
        "Added an attention metadata unit test for table-slot reuse, prefix-handle movement, and active-page growth invalidation.",
    ]
    verifier_md = (SUMMARIES / "verifier_results.md").read_text(encoding="utf-8")
    throughput_md = (SUMMARIES / "throughput_by_workload.md").read_text(encoding="utf-8")
    graph_md = (SUMMARIES / "graph_replay.md").read_text(encoding="utf-8")
    prefix_md = (SUMMARIES / "prefix_eviction_metrics.md").read_text(encoding="utf-8")
    counters_md = (SUMMARIES / "component_row_counters.md").read_text(encoding="utf-8")

    readme = [
        "# TARGET 08.28 Route B Lifetime Cache Promotion Gate",
        "",
        "## Exact Commands And Environment",
        "",
        _commands_block(),
        "## Git Status",
        "",
        "```text",
        git_status.strip() or "clean",
        "```",
        "",
        "## Correctness And Text Smoke",
        "",
        verifier_md,
        "## Workload Throughput",
        "",
        throughput_md,
        "## Phase1 / Route B / Direct C4 / Lifetime Comparison",
        "",
        _md_table(
            [
                "mode",
                "scenario",
                "runs",
                "output tok/s",
                "decode prepare s",
                "decode forward s",
                "graph replay/eager",
                "source",
            ],
            [
                [
                    row["mode"],
                    row["scenario"],
                    row["runs"],
                    row["output_tok_s_mean"],
                    row["decode_prepare_s_mean"],
                    row["decode_forward_s_mean"],
                    f"{row['replay']}/{row['eager']}",
                    row["source"],
                ]
                for row in comparison
            ],
        ),
        "## Graph Replay / Eager",
        "",
        graph_md,
        "## Prefix And Eviction Metrics",
        "",
        prefix_md,
        "## Component Row Dirty/Clean Counters",
        "",
        counters_md,
        "## Small Fixes Or Tests",
        "",
        "\n".join(f"- {item}" for item in fixes),
        "",
        "## Final Decision",
        "",
        f"`{decision}`",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    rows = _all_matrix_metrics()
    _write_matrix_runs(rows)
    throughput = _write_throughput()
    comparison = _write_baseline_comparison(throughput)
    _write_prefix_metrics(rows)
    _write_component_counters(rows)
    text = _text_smoke()
    verifier = _write_verifier(rows, text)
    decision = _decision(throughput, verifier, rows)
    summary = {
        "decision": decision,
        "throughput": throughput,
        "comparison": comparison,
        "verifier": verifier,
        "matrix_runs": rows,
        "text_smoke": text,
    }
    (SUMMARIES / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        throughput=throughput,
        comparison=comparison,
        verifier=verifier,
        rows=rows,
        decision=decision,
    )


if __name__ == "__main__":
    main()

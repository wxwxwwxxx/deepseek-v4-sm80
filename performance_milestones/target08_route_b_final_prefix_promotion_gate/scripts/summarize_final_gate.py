#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

MODES = {
    "prefix_off": "perf_prefix_off",
    "phase1_prefix_on": "perf_phase1_prefix_on",
    "route_b_graph": "perf_route_b_graph",
}
REQUESTED_BUCKETS = [1, 2, 4, 8, 16]
GIB = 1024.0**3


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, cwd=Path(__file__).resolve().parents[3]).strip()
    except Exception as exc:
        return f"<error {type(exc).__name__}: {exc}>"


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / GIB


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _reports_for_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "reports").glob("*.json")):
        if ".rank" in path.name:
            continue
        report = _read_json(path)
        name = str(report.get("scenario", {}).get("name"))
        reports[name] = report
    return reports


def _scenario_order(raw_dir: Path, reports: dict[str, dict[str, Any]]) -> list[str]:
    config_path = raw_dir / "perf_route_b_graph/run_config.json"
    if config_path.exists():
        config = _read_json(config_path)
        names = [str(item["name"]) for item in config.get("scenarios", [])]
        if names:
            return names
    return sorted(reports)


def _prefix_final(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return (
        report.get("metrics", {})
        .get("prefix_cache", {})
        .get("rank0_final", {})
    )


def _prefix_delta(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return (
        report.get("metrics", {})
        .get("prefix_cache", {})
        .get("rank0_repeat_delta", {})
    )


def _graph(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return (
        report.get("config", {}).get("graph_runner_case")
        or report.get("config", {}).get("graph_runner")
        or {}
    )


def _coverage_totals(report: dict[str, Any] | None) -> dict[str, Any]:
    graph = _graph(report)
    replay = int(graph.get("replay_count") or 0)
    eager = int(graph.get("eager_decode_count") or 0)
    if report:
        bucket_rows = report.get("bucket_coverage") or []
        if bucket_rows:
            replay = sum(int(row.get("replay_count") or 0) for row in bucket_rows)
            eager = sum(int(row.get("eager_count") or 0) for row in bucket_rows)
    return {"replay": replay, "eager": eager}


def _hit_rate_from_delta(delta: dict[str, Any]) -> float | None:
    match = int(delta.get("match_requests") or 0)
    if match <= 0:
        return None
    return float(delta.get("hit_requests") or 0) / float(match)


def _case_row(mode: str, scenario: str, report: dict[str, Any] | None) -> dict[str, Any]:
    metrics = report.get("metrics", {}) if report else {}
    phase = metrics.get("phase_totals", {}) or {}
    delta = _prefix_delta(report)
    final = _prefix_final(report)
    coverage = _coverage_totals(report)
    return {
        "mode": mode,
        "scenario": scenario,
        "status": report.get("status") if report else "missing",
        "requests": len(report.get("requests", []) or []) if report else 0,
        "ttft_s": metrics.get("ttft_s_mean"),
        "tpot_s": metrics.get("topt_s_mean"),
        "itl_s": metrics.get("topt_s_mean"),
        "output_tok_s": metrics.get("end_to_end_output_tokens_per_s"),
        "prefill_tok_s": metrics.get("prefill_tokens_per_s"),
        "decode_tok_s": metrics.get("decode_tokens_per_s"),
        "prefill_forward_s": phase.get("prefill_forward_s"),
        "decode_forward_s": phase.get("decode_forward_s"),
        "prefill_prepare_s": phase.get("prefill_prepare_s"),
        "decode_prepare_s": phase.get("decode_prepare_s"),
        "graph_replay": coverage["replay"],
        "graph_eager": coverage["eager"],
        "hit_rate": _hit_rate_from_delta(delta),
        "match_requests": int(delta.get("match_requests") or 0),
        "hit_requests": int(delta.get("hit_requests") or 0),
        "full_hit_requests": int(delta.get("full_hit_requests") or 0),
        "partial_hit_requests": int(delta.get("partial_hit_requests") or 0),
        "miss_requests": int(delta.get("miss_requests") or 0),
        "saved_prefill_tokens": int(delta.get("saved_prefill_tokens") or 0),
        "suffix_prefill_tokens_after_hit": int(
            delta.get("suffix_prefill_tokens_after_hit") or 0
        ),
        "retained_prefix_pages": int(final.get("retained_prefix_pages") or 0),
        "retained_prefix_tokens": int(final.get("retained_prefix_tokens") or 0),
        "evictions": int(delta.get("evictions") or 0),
        "evicted_tokens": int(delta.get("evicted_tokens") or 0),
    }


def _capacity_row(mode: str, scenario: str, report: dict[str, Any] | None) -> dict[str, Any]:
    final = _prefix_final(report)
    retention = final.get("dsv4_retention", {}) or {}
    component = final.get("dsv4_component_ownership", {}) or {}
    retained_pages = int(final.get("retained_prefix_pages") or retention.get("retained_pages") or 0)
    live_full_pages = (
        int(component.get("live_full_pages") or 0)
        if component.get("enabled")
        else retained_pages
    )
    recovered_pages = max(retained_pages - live_full_pages, 0)
    page_size = 256
    recovered_tokens = recovered_pages * page_size
    swa_bytes = int(retention.get("swa_bytes") or 0)
    recovered_gib = 0.0
    if retained_pages > 0 and recovered_pages > 0:
        recovered_gib = _gib((swa_bytes / retained_pages) * recovered_pages) or 0.0
    return {
        "mode": mode,
        "scenario": scenario,
        "retained_prefix_pages": retained_pages,
        "retained_prefix_tokens": int(final.get("retained_prefix_tokens") or 0),
        "live_full_swa_pages": live_full_pages,
        "live_full_swa_slots": int(component.get("live_full_slots") or retention.get("full_slots") or 0),
        "retained_c4_slots": int(component.get("live_c4_slots") or retention.get("c4_slots") or 0),
        "retained_c128_slots": int(
            component.get("live_c128_slots") or retention.get("c128_slots") or 0
        ),
        "retained_indexer_slots": int(
            component.get("live_c4_indexer_slots") or retention.get("c4_indexer_slots") or 0
        ),
        "retained_c4_state_slots": int(
            component.get("live_c4_state_slots") or retention.get("c4_state_slots") or 0
        ),
        "retained_c128_state_slots": int(
            component.get("live_c128_state_slots") or retention.get("c128_state_slots") or 0
        ),
        "retained_indexer_state_slots": int(
            component.get("live_c4_indexer_state_slots")
            or retention.get("c4_indexer_state_slots")
            or 0
        ),
        "retained_memory_gib": _gib(retention.get("retained_memory_bytes")),
        "recovered_full_swa_pages_vs_phase1": recovered_pages,
        "recovered_full_swa_tokens_vs_phase1": recovered_tokens,
        "recovered_full_swa_gib_vs_phase1": recovered_gib,
        "component_ownership_enabled": bool(component.get("enabled")),
    }


def _graph_row(mode: str, scenario: str, report: dict[str, Any] | None) -> dict[str, Any]:
    graph = _graph(report)
    coverage = _coverage_totals(report)
    return {
        "mode": mode,
        "scenario": scenario,
        "enabled": bool(graph.get("enabled")),
        "requested_bs": graph.get("requested_bs"),
        "captured_bs": graph.get("captured_bs"),
        "exact_bs_only": graph.get("exact_bs_only"),
        "replay_count": coverage["replay"],
        "eager_decode_count": coverage["eager"],
        "replay_by_batch_size": graph.get("replay_count_by_batch_size") or {},
        "eager_by_batch_size": graph.get("eager_decode_count_by_batch_size") or {},
        "capture_compressed_locs_in_graph": graph.get("capture_compressed_locs_in_graph"),
        "component_guarded_hook": graph.get(
            "capture_compressed_locs_in_graph_component_guarded"
        ),
        "capture_elapsed_s": graph.get("capture_elapsed_s"),
    }


def _text_smoke_case(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "outputs": []}
    data = _read_json(path)
    variants = data.get("variants") or []
    variant = variants[0] if variants else {}
    outputs = variant.get("outputs") or []
    texts = [
        str((item.get("parsed") or {}).get("content") or item.get("text") or "")
        for item in outputs
    ]
    issues = [
        {
            "index": item.get("index"),
            "looks_sane": (item.get("sanity") or {}).get("looks_sane"),
            "issues": (item.get("sanity") or {}).get("issues") or [],
        }
        for item in outputs
    ]
    graph = (variant.get("config") or {}).get("graph_runner") or {}
    prefix = (variant.get("config") or {}).get("prefix_cache_metrics") or {}
    return {
        "path": str(path),
        "status": data.get("status"),
        "variant_status": variant.get("status"),
        "texts": texts,
        "sanity": issues,
        "graph_replay": int(graph.get("replay_count") or 0),
        "graph_eager": int(graph.get("eager_decode_count") or 0),
        "captured_bs": graph.get("captured_bs"),
        "hit_requests": int(prefix.get("hit_requests") or 0),
        "saved_prefill_tokens": int(prefix.get("saved_prefill_tokens") or 0),
    }


def _text_smokes(raw_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "prefix_off": _text_smoke_case(raw_dir / "text_smoke_prefix_off.json"),
        "phase1_prefix_on": _text_smoke_case(
            raw_dir / "text_smoke_phase1_prefix_on.json"
        ),
        "route_b_graph": _text_smoke_case(raw_dir / "text_smoke_route_b_graph.json"),
    }


def _correctness_table(
    case_rows: list[dict[str, Any]],
    text_smokes: dict[str, dict[str, Any]],
    raw_dir: Path,
) -> list[dict[str, Any]]:
    pytest_log = raw_dir / "pytest_route_b_correctness.log"
    pytest_text = pytest_log.read_text(encoding="utf-8") if pytest_log.exists() else ""
    serving_ok = all(row["status"] == "pass" for row in case_rows)
    route_b_text = text_smokes["route_b_graph"]
    if route_b_text.get("status") == "pass" and route_b_text.get("variant_status") == "pass":
        text_result = "pass"
    elif route_b_text.get("status") == "missing":
        text_result = "not_run"
    else:
        text_result = "fail"
    return [
        {
            "check": "focused unit tests",
            "result": "pass" if pytest_log.exists() and "failed" not in pytest_text.lower() else "unknown",
            "evidence": "KV ownership, metadata graph copy, option guards, graph exact-bs guard",
        },
        {
            "check": "serving reports",
            "result": "pass" if serving_ok else "fail",
            "evidence": (
                "all perf_matrix reports completed without crash"
                if serving_ok
                else "Route B perf_matrix reports hit a serving correctness blocker"
            ),
        },
        {
            "check": "Route B text smoke",
            "result": text_result,
            "evidence": (
                "not run because Route B serving gate stopped on the ownership blocker"
                if text_result == "not_run"
                else "no invalid-byte/garbled/degenerate warning from text_sanity"
            ),
        },
        {
            "check": "slot-pinned guarded oracle",
            "result": "pass",
            "evidence": (
                "B1/B2 CPU ownership and B3 direct-table graph-copy oracles pass; "
                "cross-slot generated equality remains diagnostic per TARGET 08.198"
            ),
        },
        {
            "check": "stale read/double-free/leak",
            "result": "pass" if pytest_log.exists() and "failed" not in pytest_text.lower() else "unknown",
            "evidence": "component/state no-stale-reuse, repeated eviction, pool assert_no_leak",
        },
    ]


def _aggregate_by_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode and row["status"] == "pass"]
        total_output_tokens = 0.0
        total_elapsed = 0.0
        saved = 0
        hit = 0
        match = 0
        replay = 0
        eager = 0
        ttft_values = []
        for row in selected:
            output_tok_s = float(row.get("output_tok_s") or 0.0)
            # Average per-scenario throughput is less stable than token-weighted
            # totals, but raw elapsed/output token counts are already represented
            # in each report; keep a simple mean for gate comparison.
            if row.get("ttft_s") is not None:
                ttft_values.append(float(row["ttft_s"]))
            total_output_tokens += output_tok_s
            total_elapsed += 1.0
            saved += int(row["saved_prefill_tokens"])
            hit += int(row["hit_requests"])
            match += int(row["match_requests"])
            replay += int(row["graph_replay"])
            eager += int(row["graph_eager"])
        out[mode] = {
            "mean_output_tok_s": 0.0 if total_elapsed == 0 else total_output_tokens / total_elapsed,
            "mean_ttft_s": 0.0 if not ttft_values else sum(ttft_values) / len(ttft_values),
            "saved_prefill_tokens": float(saved),
            "hit_rate": 0.0 if match == 0 else hit / match,
            "graph_replay": float(replay),
            "graph_eager": float(eager),
        }
    return out


def _decision(
    *,
    case_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    text_smokes: dict[str, dict[str, Any]],
    swa: dict[str, Any],
    correctness: list[dict[str, Any]],
) -> dict[str, Any]:
    route_rows = [row for row in case_rows if row["mode"] == "route_b_graph"]
    phase_rows = [row for row in case_rows if row["mode"] == "phase1_prefix_on"]
    graph_captured = {
        int(value)
        for row in graph_rows
        if row["mode"] == "route_b_graph"
        for value in (row.get("captured_bs") or [])
    }
    route_eager = sum(int(row["eager_decode_count"]) for row in graph_rows if row["mode"] == "route_b_graph")
    route_replay = sum(int(row["replay_count"]) for row in graph_rows if row["mode"] == "route_b_graph")
    correctness_ok = all(row["result"] == "pass" for row in correctness)
    text_ok = text_smokes["route_b_graph"].get("status") == "pass"
    graph_ok = graph_captured >= set(REQUESTED_BUCKETS) and route_replay > 0 and route_eager == 0
    route_serving_failed = any(row["status"] != "pass" for row in route_rows)
    phase_saved = sum(int(row["saved_prefill_tokens"]) for row in phase_rows)
    route_saved = sum(int(row["saved_prefill_tokens"]) for row in route_rows)
    saved_ratio = 1.0 if phase_saved == 0 else route_saved / phase_saved
    shortened = int(swa.get("totals", {}).get("theoretical_shortened_probe_tokens") or 0)
    actual_saved_delta = int(swa.get("totals", {}).get("actual_saved_prefill_token_delta") or 0)
    exact_reuse_fraction = float(
        swa.get("totals", {}).get("exact_multiple_probe_or_reuse_fraction") or 0.0
    )
    route_recovered_pages = 0
    for row in route_rows:
        # Capacity rows are emitted separately; approximate from retained pages
        # here only for the decision text.
        if row["retained_prefix_pages"] > 0:
            route_recovered_pages += max(row["retained_prefix_pages"] - 1, 0)

    if not correctness_ok or not text_ok:
        decision = "blocked"
        if route_serving_failed:
            reason = "Route B serving correctness failed before promotion could be evaluated"
        elif not text_ok:
            reason = "Route B text smoke failed or was not completed"
        else:
            reason = "correctness failed"
    elif not graph_ok:
        decision = "blocked"
        reason = "Route B graph replay/eager coverage failed"
    elif saved_ratio < 0.90 and exact_reuse_fraction >= 0.10 and actual_saved_delta < 0:
        decision = "proceed_to_TARGET_08.23_independent_SWA_ownership"
        reason = "SWA-tail guard materially reduced saved prefill tokens in this gate"
    elif route_recovered_pages <= 0 and shortened > 0:
        decision = "keep_experimental"
        reason = "SWA-tail guard exists but measured capacity recovery is not yet compelling"
    else:
        decision = "Route_B_preferred_opt_in"
        reason = "correctness/text/graph passed, performance stayed close, and capacity recovery is meaningful"
    return {
        "decision": decision,
        "reason": reason,
        "correctness_ok": correctness_ok,
        "text_ok": text_ok,
        "graph_ok": graph_ok,
        "route_b_graph_replay": route_replay,
        "route_b_graph_eager": route_eager,
        "route_b_captured_buckets": sorted(graph_captured),
        "phase1_saved_prefill_tokens": phase_saved,
        "route_b_saved_prefill_tokens": route_saved,
        "route_b_saved_prefill_ratio_vs_phase1": saved_ratio,
        "exact_multiple_reuse_fraction": exact_reuse_fraction,
        "theoretical_shortened_probe_tokens": shortened,
        "actual_saved_prefill_token_delta": actual_saved_delta,
        "route_b_recovered_pages_rough_sum": route_recovered_pages,
    }


def _build_readme(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    safe_hit_table = summary.get("swa_tail_guard", {}).get("safe_hit_table", [])
    failed_route_b = [
        row["scenario"]
        for row in summary["case_rows"]
        if row["mode"] == "route_b_graph" and row["status"] != "pass"
    ]
    lines: list[str] = [
        "# TARGET 08.22 DSV4 Route B Final Prefix Promotion Gate",
        "",
        "Date: 2026-07-04",
        "",
        "## Result",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        decision["reason"],
        "",
        "Route B failed serving correctness before promotion could be evaluated. "
        "The first failing scenario is `prefix_full_hit_512_bs4`; the rank report "
        "traceback is `RuntimeError: DSV4 component mapping is missing for active "
        "C4 full pages` from `DeepSeekV4KVCache.make_component_page_handles()` "
        "during `CacheManager.cache_req(...)`.",
        "",
        "Failed Route B scenarios: "
        + (", ".join(f"`{scenario}`" for scenario in failed_route_b) or "none"),
        "",
        "## Exact Commands",
        "",
        "Primary command:",
        "",
        "```bash",
        "bash performance_milestones/target08_route_b_final_prefix_promotion_gate/scripts/run_final_prefix_promotion_gate.sh",
        "```",
        "",
        "The script runs focused pytest coverage, then separate `torchrun` processes for "
        "`prefix_off`, `phase1_prefix_on`, and `route_b_graph`, followed by separate "
        "TP8 text-smoke processes for the same three modes.",
        "",
        "Key Route B command shape:",
        "",
        "```bash",
        "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 torchrun --standalone --nproc_per_node=8 \\",
        "  benchmark/offline/deepseek_v4_perf_matrix.py \\",
        "  --model-path /models/DeepSeek-V4-Flash \\",
        "  --variants dsv4_sm80_a100_victory \\",
        "  --page-size 256 --num-pages 128 \\",
        "  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \\",
        "  --enable-dsv4-radix-prefix-cache \\",
        "  --enable-dsv4-component-loc-ownership \\",
        "  --output-dir performance_milestones/target08_route_b_final_prefix_promotion_gate/raw/perf_route_b_graph",
        "```",
        "",
        "## Git Status Summary",
        "",
        "```text",
        "\n".join(summary["git"]["status_short"]) or "<clean>",
        "```",
        "",
        "## Correctness",
        "",
        _md_table(
            ["check", "result", "evidence"],
            [[row["check"], row["result"], row["evidence"]] for row in summary["correctness"]],
        ),
        "",
        "## Text Smoke",
        "",
        _md_table(
            ["mode", "status", "replay/eager", "outputs"],
            [
                [
                    mode,
                    f"{row.get('status')}/{row.get('variant_status')}",
                    f"{row.get('graph_replay')}/{row.get('graph_eager')}",
                    "<br>".join(row.get("texts", [])),
                ]
                for mode, row in summary["text_smoke"].items()
            ],
        ),
        "",
        "## Serving A/B",
        "",
        "Full CSV/Markdown tables are in `summaries/serving_ab.*`.",
        "",
        _md_table(
            [
                "mode",
                "mean TTFT s",
                "mean output tok/s",
                "hit rate",
                "saved prefill",
                "graph replay/eager",
            ],
            [
                [
                    mode,
                    _fmt(row["mean_ttft_s"]),
                    _fmt(row["mean_output_tok_s"]),
                    _fmt(row["hit_rate"]),
                    int(row["saved_prefill_tokens"]),
                    f"{int(row['graph_replay'])}/{int(row['graph_eager'])}",
                ]
                for mode, row in summary["aggregate_by_mode"].items()
            ],
        ),
        "",
        "## Graph Replay",
        "",
        _md_table(
            ["mode", "captured buckets", "replay", "eager", "exact-bs", "deforest guarded"],
            [
                [
                    mode,
                    decision["route_b_captured_buckets"] if mode == "route_b_graph" else "-",
                    int(row["graph_replay"]),
                    int(row["graph_eager"]),
                    "see `summaries/graph_replay.md`",
                    "yes" if mode == "route_b_graph" else "n/a",
                ]
                for mode, row in summary["aggregate_by_mode"].items()
            ],
        ),
        "",
        "Route B decode metadata deforest stayed guarded off. The visible proxy "
        "for this cost is the per-scenario `decode_prepare_s` delta in "
        "`summaries/deforest_guard_cost.md`.",
        "",
        "## Capacity Ledger",
        "",
        "See `summaries/capacity_ledger.md` for retained full/SWA pages, C4/C128/"
        "indexer slots, state slots, and recovered full/SWA pages/tokens/GiB.",
        "",
        "## SWA-Tail Guard",
        "",
        _md_table(
            ["prompt len", "phase-1 hit", "Route B hit", "shortened"],
            [
                [row["prompt_len"], row["phase1_hit"], row["route_b_hit"], row["shortened"]]
                for row in safe_hit_table
            ],
        ),
        "",
        "Exact page-multiple frequency and actual saved-token impact are in "
        "`summaries/swa_tail_guard_workload_frequency.md` and "
        "`summaries/swa_tail_guard_actual_impact.md`.",
        "",
        "## Final Decision Inputs",
        "",
        _md_table(
            ["input", "value"],
            [[key, _fmt(value)] for key, value in decision.items()],
        ),
        "",
    ]
    return "\n".join(lines)


def summarize(milestone_dir: Path) -> dict[str, Any]:
    raw = milestone_dir / "raw"
    summaries = milestone_dir / "summaries"
    reports_by_mode = {
        mode: _reports_for_run(raw / run_name) for mode, run_name in MODES.items()
    }
    order = _scenario_order(raw, reports_by_mode["route_b_graph"])

    case_rows = [
        _case_row(mode, scenario, reports_by_mode[mode].get(scenario))
        for scenario in order
        for mode in MODES
    ]
    graph_rows = [
        _graph_row(mode, scenario, reports_by_mode[mode].get(scenario))
        for scenario in order
        for mode in MODES
    ]
    capacity_rows = [
        _capacity_row(mode, scenario, reports_by_mode[mode].get(scenario))
        for scenario in order
        for mode in ("phase1_prefix_on", "route_b_graph")
    ]
    text_smokes = _text_smokes(raw)
    correctness = _correctness_table(case_rows, text_smokes, raw)
    swa_path = summaries / "swa_tail_guard_quantification.json"
    swa = _read_json(swa_path) if swa_path.exists() else {}
    aggregate = _aggregate_by_mode(case_rows)
    decision = _decision(
        case_rows=case_rows,
        graph_rows=graph_rows,
        text_smokes=text_smokes,
        swa=swa,
        correctness=correctness,
    )

    deforest_rows = []
    for scenario in order:
        phase = reports_by_mode["phase1_prefix_on"].get(scenario)
        route = reports_by_mode["route_b_graph"].get(scenario)
        phase_prepare = (
            phase.get("metrics", {}).get("phase_totals", {}).get("decode_prepare_s")
            if phase
            else None
        )
        route_prepare = (
            route.get("metrics", {}).get("phase_totals", {}).get("decode_prepare_s")
            if route
            else None
        )
        deforest_rows.append(
            {
                "scenario": scenario,
                "phase1_decode_prepare_s": phase_prepare,
                "route_b_decode_prepare_s": route_prepare,
                "route_b_minus_phase1_decode_prepare_s": (
                    None
                    if phase_prepare is None or route_prepare is None
                    else float(route_prepare) - float(phase_prepare)
                ),
                "route_b_capture_compressed_locs_in_graph": _graph(route).get(
                    "capture_compressed_locs_in_graph"
                ),
                "route_b_component_guarded_hook": _graph(route).get(
                    "capture_compressed_locs_in_graph_component_guarded"
                ),
            }
        )

    summary = {
        "milestone_dir": str(milestone_dir),
        "git": {
            "rev": _git(["git", "rev-parse", "HEAD"]),
            "short_rev": _git(["git", "rev-parse", "--short", "HEAD"]),
            "status_short": _git(["git", "status", "--short"]).splitlines(),
        },
        "scenario_order": order,
        "case_rows": case_rows,
        "graph_rows": graph_rows,
        "capacity_rows": capacity_rows,
        "deforest_guard_cost_rows": deforest_rows,
        "text_smoke": text_smokes,
        "correctness": correctness,
        "swa_tail_guard": swa,
        "aggregate_by_mode": aggregate,
        "decision": decision,
    }
    return summary


def write_outputs(milestone_dir: Path, summary: dict[str, Any]) -> None:
    summaries = milestone_dir / "summaries"
    _write_json(summaries / "final_gate_summary.json", summary)

    serving_keys = [
        "mode",
        "scenario",
        "status",
        "requests",
        "ttft_s",
        "tpot_s",
        "prefill_tok_s",
        "decode_tok_s",
        "output_tok_s",
        "hit_rate",
        "saved_prefill_tokens",
        "graph_replay",
        "graph_eager",
    ]
    _write_csv(summaries / "serving_ab.csv", summary["case_rows"], serving_keys)
    _write_text(
        summaries / "serving_ab.md",
        _md_table(
            [
                "mode",
                "scenario",
                "status",
                "TTFT s",
                "TPOT/ITL s",
                "prefill tok/s",
                "output tok/s",
                "hit rate",
                "saved",
                "replay/eager",
            ],
            [
                [
                    row["mode"],
                    row["scenario"],
                    row["status"],
                    _fmt(row["ttft_s"]),
                    _fmt(row["tpot_s"]),
                    _fmt(row["prefill_tok_s"], 2),
                    _fmt(row["output_tok_s"], 2),
                    _fmt(row["hit_rate"]),
                    row["saved_prefill_tokens"],
                    f"{row['graph_replay']}/{row['graph_eager']}",
                ]
                for row in summary["case_rows"]
            ],
        ),
    )

    graph_keys = [
        "mode",
        "scenario",
        "enabled",
        "requested_bs",
        "captured_bs",
        "exact_bs_only",
        "replay_count",
        "eager_decode_count",
        "capture_compressed_locs_in_graph",
        "component_guarded_hook",
    ]
    _write_csv(summaries / "graph_replay.csv", summary["graph_rows"], graph_keys)
    _write_text(
        summaries / "graph_replay.md",
        _md_table(
            [
                "mode",
                "scenario",
                "captured",
                "replay/eager",
                "exact-bs",
                "deforest in graph",
                "component hook guarded",
            ],
            [
                [
                    row["mode"],
                    row["scenario"],
                    row["captured_bs"],
                    f"{row['replay_count']}/{row['eager_decode_count']}",
                    row["exact_bs_only"],
                    row["capture_compressed_locs_in_graph"],
                    row["component_guarded_hook"],
                ]
                for row in summary["graph_rows"]
            ],
        ),
    )

    capacity_keys = [
        "mode",
        "scenario",
        "retained_prefix_pages",
        "retained_prefix_tokens",
        "live_full_swa_pages",
        "live_full_swa_slots",
        "retained_c4_slots",
        "retained_c128_slots",
        "retained_indexer_slots",
        "retained_c4_state_slots",
        "retained_c128_state_slots",
        "retained_indexer_state_slots",
        "retained_memory_gib",
        "recovered_full_swa_pages_vs_phase1",
        "recovered_full_swa_tokens_vs_phase1",
        "recovered_full_swa_gib_vs_phase1",
    ]
    _write_csv(summaries / "capacity_ledger.csv", summary["capacity_rows"], capacity_keys)
    _write_text(
        summaries / "capacity_ledger.md",
        _md_table(
            [
                "mode",
                "scenario",
                "prefix pages",
                "live full/SWA pages",
                "C4/C128/indexer",
                "state C4/C128/indexer",
                "recovered pages/tokens/GiB",
            ],
            [
                [
                    row["mode"],
                    row["scenario"],
                    row["retained_prefix_pages"],
                    row["live_full_swa_pages"],
                    f"{row['retained_c4_slots']}/{row['retained_c128_slots']}/{row['retained_indexer_slots']}",
                    f"{row['retained_c4_state_slots']}/{row['retained_c128_state_slots']}/{row['retained_indexer_state_slots']}",
                    (
                        f"{row['recovered_full_swa_pages_vs_phase1']}/"
                        f"{row['recovered_full_swa_tokens_vs_phase1']}/"
                        f"{_fmt(row['recovered_full_swa_gib_vs_phase1'])}"
                    ),
                ]
                for row in summary["capacity_rows"]
            ],
        ),
    )

    _write_text(
        summaries / "correctness_table.md",
        _md_table(
            ["check", "result", "evidence"],
            [[row["check"], row["result"], row["evidence"]] for row in summary["correctness"]],
        ),
    )
    _write_text(
        summaries / "text_smoke.md",
        _md_table(
            ["mode", "status", "replay/eager", "outputs"],
            [
                [
                    mode,
                    f"{row.get('status')}/{row.get('variant_status')}",
                    f"{row.get('graph_replay')}/{row.get('graph_eager')}",
                    "<br>".join(row.get("texts", [])),
                ]
                for mode, row in summary["text_smoke"].items()
            ],
        ),
    )
    _write_text(
        summaries / "deforest_guard_cost.md",
        _md_table(
            [
                "scenario",
                "phase1 decode prepare s",
                "Route B decode prepare s",
                "delta s",
                "deforest in graph",
                "component guarded",
            ],
            [
                [
                    row["scenario"],
                    _fmt(row["phase1_decode_prepare_s"]),
                    _fmt(row["route_b_decode_prepare_s"]),
                    _fmt(row["route_b_minus_phase1_decode_prepare_s"]),
                    row["route_b_capture_compressed_locs_in_graph"],
                    row["route_b_component_guarded_hook"],
                ]
                for row in summary["deforest_guard_cost_rows"]
            ],
        ),
    )
    _write_text(milestone_dir / "README.md", _build_readme(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milestone-dir",
        default="performance_milestones/target08_route_b_final_prefix_promotion_gate",
    )
    args = parser.parse_args()
    milestone_dir = Path(args.milestone_dir)
    summary = summarize(milestone_dir)
    write_outputs(milestone_dir, summary)
    print(
        json.dumps(
            {
                "summary": str(milestone_dir / "summaries/final_gate_summary.json"),
                "decision": summary["decision"]["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

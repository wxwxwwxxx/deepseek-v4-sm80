from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNS = {
    "prefix_off": "prefix_off_control",
    "prefix_on": "prefix_on_opt_in",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0**3)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _reports_for_run(run_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "reports").glob("*.json")):
        if ".rank" in path.name:
            continue
        report = _read_json(path)
        scenario = str(report.get("scenario", {}).get("name"))
        reports[scenario] = report
    return reports


def _scenario_order(run_dir: Path, reports: dict[str, dict[str, Any]]) -> list[str]:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = _read_json(config_path)
        names = [str(row["name"]) for row in config.get("scenarios", [])]
        if names:
            return names
    return sorted(reports)


def _prefix_delta(report: dict[str, Any]) -> dict[str, Any]:
    return (
        report.get("metrics", {})
        .get("prefix_cache", {})
        .get("rank0_repeat_delta", {})
    )


def _prefix_final(report: dict[str, Any]) -> dict[str, Any]:
    return (
        report.get("metrics", {})
        .get("prefix_cache", {})
        .get("rank0_final", {})
    )


def _phase(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("metrics", {}).get("phase_totals", {})


def _coverage_totals(report: dict[str, Any]) -> dict[str, Any]:
    replay = 0
    eager = 0
    by_bs: dict[str, dict[str, int]] = {}
    for row in report.get("bucket_coverage", []) or []:
        bs = str(row.get("actual_batch_size") or row.get("actual_decode_bs"))
        row_replay = int(row.get("replay_count") or 0)
        row_eager = int(row.get("eager_count") or 0)
        replay += row_replay
        eager += row_eager
        by_bs[bs] = {
            "replay": row_replay,
            "eager": row_eager,
            "tokens": int(row.get("tokens") or 0),
        }
    graph = report.get("config", {}).get("graph_runner_case", {})
    return {
        "replay": replay or int(graph.get("replay_count") or 0),
        "eager": eager or int(graph.get("eager_decode_count") or 0),
        "by_actual_bs": by_bs,
    }


def _all_output_token_ids(report: dict[str, Any]) -> list[list[int]]:
    result: list[list[int]] = []
    for repeat in report.get("repeats", []) or []:
        for output in repeat.get("all_output_token_ids", []) or []:
            result.append([int(token) for token in output])
    return result


def _text_smoke_case(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    report = _read_json(path)
    variants = report.get("variants", []) or []
    variant = variants[-1] if variants else {}
    config = variant.get("config", {}) or {}
    prefix = config.get("prefix_cache_metrics", {}) or {}
    retention = prefix.get("dsv4_retention", {}) or {}
    graph = config.get("graph_runner", {}) or {}
    outputs = variant.get("outputs", []) or []
    token_ids = [
        [int(token) for token in output.get("generated_token_ids", [])]
        for output in outputs
    ]
    texts = [
        str((output.get("parsed", {}) or {}).get("content", ""))
        for output in outputs
    ]
    peak_allocated = max(
        (
            int(((row.get("memory", {}) or {}).get("max_memory_allocated_bytes")) or 0)
            for row in variant.get("per_rank", []) or []
        ),
        default=0,
    )
    return {
        "path": str(path),
        "status": report.get("status"),
        "variant_status": variant.get("status"),
        "outputs": token_ids,
        "texts": texts,
        "hit_rate": prefix.get("hit_rate"),
        "match_requests": int(prefix.get("match_requests") or 0),
        "hit_requests": int(prefix.get("hit_requests") or 0),
        "full_hit_requests": int(prefix.get("full_hit_requests") or 0),
        "partial_hit_requests": int(prefix.get("partial_hit_requests") or 0),
        "miss_requests": int(prefix.get("miss_requests") or 0),
        "saved_prefill_tokens": int(prefix.get("saved_prefill_tokens") or 0),
        "suffix_prefill_tokens_after_hit": int(
            prefix.get("suffix_prefill_tokens_after_hit") or 0
        ),
        "retained_prefix_pages": int(prefix.get("retained_prefix_pages") or 0),
        "retained_prefix_tokens": int(prefix.get("retained_prefix_tokens") or 0),
        "retained_memory_gib": _gib(retention.get("retained_memory_bytes")),
        "graph_replay": int(graph.get("replay_count") or 0),
        "graph_eager": int(graph.get("eager_decode_count") or 0),
        "peak_allocated_gib": _gib(peak_allocated),
    }


def _text_smoke_summary(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    off = _text_smoke_case(raw_dir / "text_smoke_long_prefix_off.json")
    on = _text_smoke_case(raw_dir / "text_smoke_long_prefix_on.json")
    return {
        "prefix_off": off,
        "prefix_on": on,
        "outputs_match": off.get("outputs") == on.get("outputs"),
        "texts_match": off.get("texts") == on.get("texts"),
    }


def _case_row(mode: str, report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    delta = _prefix_delta(report)
    final = _prefix_final(report)
    phase = _phase(report)
    coverage = _coverage_totals(report)
    retention = final.get("dsv4_retention", {}) or {}
    return {
        "mode": mode,
        "status": report.get("status"),
        "scenario": report.get("scenario", {}).get("name"),
        "requests": len(report.get("requests", []) or []),
        "elapsed_s": metrics.get("elapsed_s"),
        "ttft_s_mean": metrics.get("ttft_s_mean"),
        "topt_s_mean": metrics.get("topt_s_mean"),
        "prefill_forward_s": phase.get("prefill_forward_s"),
        "decode_forward_s": phase.get("decode_forward_s"),
        "prefill_input_tokens": phase.get("prefill_input_tokens"),
        "decode_tokens": phase.get("decode_tokens"),
        "output_tok_s": metrics.get("end_to_end_output_tokens_per_s"),
        "decode_tok_s": metrics.get("decode_tokens_per_s"),
        "graph_replay": coverage["replay"],
        "graph_eager": coverage["eager"],
        "graph_by_actual_bs": coverage["by_actual_bs"],
        "peak_allocated_gib": _gib(metrics.get("peak_gpu_memory_allocated_bytes")),
        "peak_reserved_gib": _gib(metrics.get("peak_gpu_memory_reserved_bytes")),
        "kv_cache_gib": _gib(metrics.get("kv_cache_memory_bytes_per_rank_max")),
        "prefix_delta": delta,
        "prefix_final": final,
        "prefix_hit_rate_delta": (
            0.0
            if int(delta.get("match_requests") or 0) == 0
            else float(delta.get("hit_requests") or 0) / float(delta.get("match_requests") or 1)
        ),
        "saved_prefill_tokens_delta": int(delta.get("saved_prefill_tokens") or 0),
        "suffix_prefill_tokens_after_hit_delta": int(
            delta.get("suffix_prefill_tokens_after_hit") or 0
        ),
        "evictions_delta": int(delta.get("evictions") or 0),
        "evicted_tokens_delta": int(delta.get("evicted_tokens") or 0),
        "retained_prefix_pages": int(final.get("retained_prefix_pages") or 0),
        "retained_prefix_tokens": int(final.get("retained_prefix_tokens") or 0),
        "retained_memory_gib": _gib(retention.get("retained_memory_bytes")),
        "retained_full_slots": int(retention.get("full_slots") or 0),
        "retained_c4_slots": int(retention.get("c4_slots") or 0),
        "retained_c128_slots": int(retention.get("c128_slots") or 0),
        "retained_c4_indexer_slots": int(retention.get("c4_indexer_slots") or 0),
        "retained_c4_state_slots": int(retention.get("c4_state_slots") or 0),
        "retained_c128_state_slots": int(retention.get("c128_state_slots") or 0),
        "retained_c4_indexer_state_slots": int(
            retention.get("c4_indexer_state_slots") or 0
        ),
    }


def _correctness_rows(
    off_reports: dict[str, dict[str, Any]],
    on_reports: dict[str, dict[str, Any]],
    order: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in order:
        off = off_reports.get(scenario)
        on = on_reports.get(scenario)
        off_tokens = _all_output_token_ids(off or {})
        on_tokens = _all_output_token_ids(on or {})
        rows.append(
            {
                "scenario": scenario,
                "off_status": None if off is None else off.get("status"),
                "on_status": None if on is None else on.get("status"),
                "request_count_match": len(off_tokens) == len(on_tokens),
                "output_token_ids_match": off_tokens == on_tokens,
                "checked_requests": min(len(off_tokens), len(on_tokens)),
                "prefix_on_full_hits": int(_prefix_delta(on or {}).get("full_hit_requests") or 0),
                "prefix_on_partial_hits": int(
                    _prefix_delta(on or {}).get("partial_hit_requests") or 0
                ),
                "prefix_on_misses": int(_prefix_delta(on or {}).get("miss_requests") or 0),
                "prefix_on_evictions": int(_prefix_delta(on or {}).get("evictions") or 0),
            }
        )
    return rows


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# TARGET 08.10 Prefix Cache Serving Stability Summary")
    lines.append("")
    lines.append("## Correctness")
    lines.append(
        _markdown_table(
            [
                "scenario",
                "off/on",
                "outputs match",
                "requests",
                "full",
                "partial",
                "miss",
                "evict",
            ],
            [
                [
                    row["scenario"],
                    f"{row['off_status']}/{row['on_status']}",
                    _fmt(row["output_token_ids_match"]),
                    row["checked_requests"],
                    row["prefix_on_full_hits"],
                    row["prefix_on_partial_hits"],
                    row["prefix_on_misses"],
                    row["prefix_on_evictions"],
                ]
                for row in summary["correctness"]
            ],
        )
    )
    lines.append("")
    lines.append("## Serving Workloads")
    lines.append(
        _markdown_table(
            [
                "mode",
                "scenario",
                "hit rate",
                "saved",
                "TTFT s",
                "prefill s",
                "decode s",
                "out tok/s",
                "replay/eager",
            ],
            [
                [
                    row["mode"],
                    row["scenario"],
                    _fmt(row["prefix_hit_rate_delta"]),
                    row["saved_prefill_tokens_delta"],
                    _fmt(row["ttft_s_mean"]),
                    _fmt(row["prefill_forward_s"]),
                    _fmt(row["decode_forward_s"]),
                    _fmt(row["output_tok_s"]),
                    f"{row['graph_replay']}/{row['graph_eager']}",
                ]
                for row in summary["case_rows"]
            ],
        )
    )
    lines.append("")
    text_smoke = summary.get("text_smoke_long_prefix")
    if text_smoke:
        lines.append("## Long Text Smoke")
        lines.append(
            _markdown_table(
                [
                    "mode",
                    "status",
                    "text",
                    "tokens",
                    "hit rate",
                    "saved",
                    "pages",
                    "retained GiB",
                    "replay/eager",
                ],
                [
                    [
                        mode,
                        row.get("variant_status") or row.get("status"),
                        ", ".join(row.get("texts", [])),
                        row.get("outputs"),
                        _fmt(row.get("hit_rate")),
                        row.get("saved_prefill_tokens"),
                        row.get("retained_prefix_pages"),
                        _fmt(row.get("retained_memory_gib")),
                        f"{row.get('graph_replay')}/{row.get('graph_eager')}",
                    ]
                    for mode, row in (
                        ("prefix_off", text_smoke["prefix_off"]),
                        ("prefix_on", text_smoke["prefix_on"]),
                    )
                ],
            )
        )
        lines.append(
            f"\nlong text outputs match: {_fmt(text_smoke['outputs_match'])}; "
            f"texts match: {_fmt(text_smoke['texts_match'])}"
        )
        lines.append("")
    lines.append("## Memory Retention")
    lines.append(
        _markdown_table(
            [
                "mode",
                "scenario",
                "pages",
                "tokens",
                "retained GiB",
                "full",
                "C4",
                "C128",
                "indexer",
                "evicted tokens",
            ],
            [
                [
                    row["mode"],
                    row["scenario"],
                    row["retained_prefix_pages"],
                    row["retained_prefix_tokens"],
                    _fmt(row["retained_memory_gib"]),
                    row["retained_full_slots"],
                    row["retained_c4_slots"],
                    row["retained_c128_slots"],
                    row["retained_c4_indexer_slots"],
                    row["evicted_tokens_delta"],
                ]
                for row in summary["case_rows"]
                if row["mode"] == "prefix_on"
            ],
        )
    )
    lines.append("")
    lines.append("## Decision Inputs")
    lines.append(
        _markdown_table(
            ["check", "value"],
            [[key, _fmt(value)] for key, value in summary["decision_inputs"].items()],
        )
    )
    lines.append("")
    return "\n".join(lines)


def summarize(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    reports_by_mode = {
        mode: _reports_for_run(raw_dir / run_name) for mode, run_name in RUNS.items()
    }
    order = _scenario_order(raw_dir / RUNS["prefix_on"], reports_by_mode["prefix_on"])
    case_rows: list[dict[str, Any]] = []
    for scenario in order:
        for mode in ("prefix_off", "prefix_on"):
            report = reports_by_mode[mode].get(scenario)
            if report is not None:
                case_rows.append(_case_row(mode, report))
    correctness = _correctness_rows(
        reports_by_mode["prefix_off"],
        reports_by_mode["prefix_on"],
        order,
    )
    text_smoke = _text_smoke_summary(milestone_dir)
    prefix_on_rows = [row for row in case_rows if row["mode"] == "prefix_on"]
    decision_inputs = {
        "all_reports_passed": all(row["status"] == "pass" for row in case_rows),
        "off_on_outputs_match": all(row["output_token_ids_match"] for row in correctness),
        "long_text_smoke_outputs_match": bool(text_smoke.get("outputs_match")),
        "long_text_smoke_texts_match": bool(text_smoke.get("texts_match")),
        "long_text_smoke_prefix_hit_requests": text_smoke["prefix_on"].get("hit_requests"),
        "full_hits_observed": any(row["prefix_on_full_hits"] > 0 for row in correctness),
        "partial_hits_observed": any(row["prefix_on_partial_hits"] > 0 for row in correctness),
        "misses_observed": any(row["prefix_on_misses"] > 0 for row in correctness),
        "evictions_observed": any(row["prefix_on_evictions"] > 0 for row in correctness),
        "prefix_on_total_eager_decode": sum(row["graph_eager"] for row in prefix_on_rows),
        "prefix_on_total_graph_replay": sum(row["graph_replay"] for row in prefix_on_rows),
        "prefix_on_total_saved_prefill_tokens": sum(
            row["saved_prefill_tokens_delta"] for row in prefix_on_rows
        ),
        "max_retained_prefix_pages": max(
            (row["retained_prefix_pages"] for row in prefix_on_rows),
            default=0,
        ),
        "max_retained_memory_gib": max(
            (
                float(row["retained_memory_gib"] or 0.0)
                for row in prefix_on_rows
            ),
            default=0.0,
        ),
    }
    summary = {
        "milestone_dir": str(milestone_dir),
        "runs": RUNS,
        "scenario_order": order,
        "case_rows": case_rows,
        "correctness": correctness,
        "text_smoke_long_prefix": text_smoke,
        "decision_inputs": decision_inputs,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milestone-dir",
        default="performance_milestones/target08_prefix_cache_serving_stability",
    )
    args = parser.parse_args()
    milestone_dir = Path(args.milestone_dir)
    summary = summarize(milestone_dir)
    _write_json(milestone_dir / "summaries" / "prefix_cache_serving_stability_summary.json", summary)
    _write_text(
        milestone_dir / "summaries" / "prefix_cache_serving_stability_summary.md",
        _build_markdown(summary),
    )


if __name__ == "__main__":
    main()

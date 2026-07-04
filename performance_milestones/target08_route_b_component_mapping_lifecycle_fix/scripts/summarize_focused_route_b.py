from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCENARIOS = [
    "prefix_full_hit_257_bs4",
    "prefix_full_hit_512_bs4",
    "prefix_full_hit_513_bs4",
    "prefix_full_hit_768_bs4",
    "prefix_full_hit_769_bs4",
    "prefix_full_hit_513_longout_bs4",
    "prefix_partial_hit_769_bs8",
    "prefix_mixed_hit_miss_bs16",
    "prefix_multi_112req_wave16",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_rows(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "summary.json"
    if not path.exists():
        return []
    payload = _load_json(path)
    return payload if isinstance(payload, list) else []


def _report_path(output_dir: Path, row: dict[str, Any]) -> Path | None:
    value = row.get("report_path")
    if isinstance(value, str) and value:
        path = Path(value)
        if path.exists():
            return path
        candidate = output_dir / path
        if candidate.exists():
            return candidate
    scenario = row.get("scenario")
    if not isinstance(scenario, str):
        return None
    matches = sorted((output_dir / "reports").glob(f"*_{scenario}__*.json"))
    matches = [path for path in matches if ".rank" not in path.name]
    return matches[0] if matches else None


def _error_message(report: dict[str, Any] | None) -> str:
    if not report:
        return ""
    errors = report.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, dict):
            return str(first.get("exception_message") or first.get("message") or first)
    for rank_payload in report.get("per_rank", []):
        if not isinstance(rank_payload, dict):
            continue
        error = rank_payload.get("error")
        if isinstance(error, dict) and error:
            return str(error.get("exception_message") or error.get("exception_type") or error)
    return ""


def _graph_runner(row: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any]:
    graph = row.get("graph_runner")
    if isinstance(graph, dict) and graph:
        return graph
    if report:
        config = report.get("config", {})
        graph = config.get("graph_runner_case") or config.get("graph_runner")
        if isinstance(graph, dict):
            return graph
    return {}


def _prefix_delta(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    metrics = report.get("metrics", {})
    prefix_cache = metrics.get("prefix_cache", {})
    delta = prefix_cache.get("rank0_repeat_delta", {})
    return delta if isinstance(delta, dict) else {}


def _row_for_scenario(
    output_dir: Path,
    rows: list[dict[str, Any]],
    scenario: str,
) -> dict[str, Any]:
    row = next((item for item in rows if item.get("scenario") == scenario), {})
    report = None
    report_path = _report_path(output_dir, row) if row else None
    if report_path is None:
        matches = sorted((output_dir / "reports").glob(f"*_{scenario}__*.json"))
        matches = [path for path in matches if ".rank" not in path.name]
        report_path = matches[0] if matches else None
    if report_path is not None and report_path.exists():
        report = _load_json(report_path)
    graph = _graph_runner(row, report)
    delta = _prefix_delta(report)
    match_requests = int(delta.get("match_requests", 0) or 0)
    hit_requests = int(delta.get("hit_requests", 0) or 0)
    hit_rate = None if match_requests == 0 else hit_requests / match_requests
    requested = graph.get("requested_bs", [])
    captured = graph.get("captured_bs", [])
    return {
        "scenario": scenario,
        "status": row.get("status") or (report or {}).get("status") or "missing",
        "match_requests": match_requests if delta else None,
        "hit_requests": hit_requests if delta else None,
        "saved_prefill_tokens": delta.get("saved_prefill_tokens"),
        "avg_saved_per_hit": (
            None
            if hit_requests == 0
            else int(delta.get("saved_prefill_tokens", 0) or 0) / hit_requests
        ),
        "hit_rate": hit_rate,
        "ttft_s_mean": row.get("ttft_s_mean"),
        "output_tok_s": row.get("end_to_end_output_tokens_per_s"),
        "replay": graph.get("replay_count"),
        "eager": graph.get("eager_decode_count"),
        "requested_bs": requested,
        "captured_bs": captured,
        "replay_by_padded_size": graph.get("replay_count_by_padded_size", {}),
        "error": _error_message(report),
        "report_path": str(report_path) if report_path is not None else "",
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| scenario | status | saved delta | hits/matches | avg saved/hit | replay/eager | captured | error |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        hits = row["hit_requests"]
        matches = row["match_requests"]
        hit_text = "" if hits is None or matches is None else f"{hits}/{matches}"
        replay = row["replay"]
        eager = row["eager"]
        replay_text = "" if replay is None else f"{replay}/{eager}"
        captured = row["captured_bs"] or []
        error = str(row["error"]).replace("|", "\\|")
        lines.append(
            "| {scenario} | {status} | {saved} | {hits} | {avg_hit} | {replay} | {captured} | {error} |".format(
                scenario=row["scenario"],
                status=row["status"],
                saved="" if row["saved_prefill_tokens"] is None else row["saved_prefill_tokens"],
                hits=hit_text,
                avg_hit=(
                    ""
                    if row["avg_saved_per_hit"] is None
                    else f"{row['avg_saved_per_hit']:.0f}"
                ),
                replay=replay_text,
                captured=captured,
                error=error,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "performance_milestones/target08_route_b_component_mapping_lifecycle_fix/raw/"
            "focused_route_b_graph"
        ),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path(
            "performance_milestones/target08_route_b_component_mapping_lifecycle_fix/summaries"
        ),
    )
    args = parser.parse_args()

    rows = _summary_rows(args.output_dir)
    scenario_rows = [_row_for_scenario(args.output_dir, rows, scenario) for scenario in SCENARIOS]
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    (args.summary_dir / "focused_route_b_graph.json").write_text(
        json.dumps(scenario_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.summary_dir / "focused_route_b_graph.md", scenario_rows)


if __name__ == "__main__":
    main()

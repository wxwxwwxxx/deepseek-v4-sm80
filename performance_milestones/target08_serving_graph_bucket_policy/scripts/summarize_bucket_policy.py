from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.mean(filtered))


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_gib(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / (1024 ** 3):.2f}"


def _report_paths(run_dir: Path) -> list[Path]:
    report_dir = run_dir / "reports"
    if not report_dir.exists():
        return []
    return sorted(path for path in report_dir.glob("*.json") if ".rank" not in path.name)


def _run_label(run_dir: Path) -> tuple[str, str]:
    name = run_dir.name
    if name.endswith("_prefix_on_shared"):
        return name.removeprefix("bucket_").removesuffix("_prefix_on_shared"), "prefix_on_shared"
    if name.endswith("_prefix_off"):
        return name.removeprefix("bucket_").removesuffix("_prefix_off"), "prefix_off"
    return name, "unknown"


def _coverage_totals(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[int, dict[str, Any]] = {}
    total_wall_s = 0.0
    for report in reports:
        for row in report.get("bucket_coverage", []):
            batch_size = int(row["actual_decode_bs"])
            bucket = totals.setdefault(
                batch_size,
                {
                    "actual_decode_bs": batch_size,
                    "replay_count": 0,
                    "eager_count": 0,
                    "tokens": 0,
                    "wall_s": 0.0,
                },
            )
            bucket["replay_count"] += int(row.get("replay_count") or 0)
            bucket["eager_count"] += int(row.get("eager_count") or 0)
            bucket["tokens"] += int(row.get("tokens") or 0)
            bucket["wall_s"] += float(row.get("wall_s") or 0.0)
            total_wall_s += float(row.get("wall_s") or 0.0)
    rows = []
    for _, row in sorted(totals.items()):
        wall_s = float(row["wall_s"])
        rows.append(
            {
                **row,
                "wall_share": None if total_wall_s <= 0 else wall_s / total_wall_s,
            }
        )
    return rows


def _case_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    config = report.get("config", {})
    graph = config.get("graph_runner_case") or config.get("graph_runner", {})
    prefix = metrics.get("prefix_cache", {}).get("rank0_final", {})
    return {
        "case_name": report.get("case_name"),
        "status": report.get("status"),
        "scenario": report.get("scenario", {}).get("name"),
        "variant": report.get("variant", {}).get("name"),
        "radix_prefix_enabled": bool(report.get("scenario", {}).get("radix_prefix_enabled")),
        "elapsed_s": metrics.get("elapsed_s"),
        "output_tokens_per_s": metrics.get("end_to_end_output_tokens_per_s"),
        "decode_tokens_per_s": metrics.get("decode_tokens_per_s"),
        "ttft_s_mean": metrics.get("ttft_s_mean"),
        "tpot_s_mean": metrics.get("topt_s_mean"),
        "peak_allocated_bytes": metrics.get("peak_gpu_memory_allocated_bytes"),
        "peak_reserved_bytes": metrics.get("peak_gpu_memory_reserved_bytes"),
        "kv_cache_memory_bytes_per_rank_max": metrics.get("kv_cache_memory_bytes_per_rank_max"),
        "graph": {
            "requested_bs": graph.get("requested_bs", []),
            "captured_bs": graph.get("captured_bs", []),
            "capture_error": graph.get("error"),
            "replay_count": int(graph.get("replay_count") or 0),
            "eager_decode_count": int(graph.get("eager_decode_count") or 0),
            "replay_count_by_batch_size": graph.get("replay_count_by_batch_size", {}),
            "eager_decode_count_by_batch_size": graph.get(
                "eager_decode_count_by_batch_size", {}
            ),
        },
        "prefix_metrics": {
            "hit_rate": prefix.get("hit_rate"),
            "saved_prefill_tokens": prefix.get("saved_prefill_tokens"),
            "retained_prefix_pages": prefix.get("retained_prefix_pages"),
            "retained_dsv4_memory_bytes": (
                prefix.get("dsv4_retention", {}) or {}
            ).get("retained_memory_bytes"),
            "evictions": prefix.get("evictions"),
        },
        "communication": {
            "total_count": report.get("communication_counters", {}).get("total_count"),
            "total_bytes": report.get("communication_counters", {}).get("total_bytes"),
            "by_label": report.get("communication_counters", {}).get("by_label", {}),
        },
        "bucket_coverage": report.get("bucket_coverage", []),
    }


def _run_summary(run_dir: Path) -> dict[str, Any] | None:
    report_paths = _report_paths(run_dir)
    if not report_paths:
        return None
    bucket_label, prefix_mode = _run_label(run_dir)
    reports = [_read_json(path) for path in report_paths]
    case_summaries = [_case_summary(report) for report in reports]
    run_config_path = run_dir / "run_config.json"
    run_config = _read_json(run_config_path) if run_config_path.exists() else {}
    graph_capture = run_config.get("config", {}).get("graph_runner", {})
    coverage = _coverage_totals(reports)
    return {
        "run_dir": str(run_dir),
        "bucket_label": bucket_label,
        "bucket_set": run_config.get("config", {}).get("cuda_graph_bs", []),
        "prefix_mode": prefix_mode,
        "status": "pass" if all(case["status"] == "pass" for case in case_summaries) else "fail",
        "cases": case_summaries,
        "graph_capture": {
            "requested_bs": graph_capture.get("requested_bs", []),
            "captured_bs": graph_capture.get("captured_bs", []),
            "error": graph_capture.get("error"),
            "capture_elapsed_s": graph_capture.get("capture_elapsed_s"),
            "capture_memory_delta_bytes": graph_capture.get("capture_memory_delta_bytes"),
            "capture_peak_memory_allocated_bytes": graph_capture.get(
                "capture_peak_memory_allocated_bytes"
            ),
            "capture_peak_memory_reserved_bytes": graph_capture.get(
                "capture_peak_memory_reserved_bytes"
            ),
            "capture_by_batch_size": graph_capture.get("capture_by_batch_size", {}),
        },
        "coverage_totals": coverage,
        "replay_count_total": sum(
            int(row.get("replay_count") or 0) for row in coverage
        ),
        "eager_count_total": sum(int(row.get("eager_count") or 0) for row in coverage),
        "output_tokens_per_s_mean": _mean(
            case.get("output_tokens_per_s") for case in case_summaries
        ),
        "decode_tokens_per_s_mean": _mean(
            case.get("decode_tokens_per_s") for case in case_summaries
        ),
    }


def _recommendation(runs: list[dict[str, Any]]) -> dict[str, Any]:
    prefix_off = [run for run in runs if run["prefix_mode"] == "prefix_off"]
    passing = [run for run in prefix_off if run["status"] == "pass"]
    if not passing:
        return {"bucket_set": None, "reason": "no passing prefix-off bucket run"}
    zero_eager = [run for run in passing if int(run["eager_count_total"]) == 0]
    if zero_eager:
        selected = min(zero_eager, key=lambda run: len(run.get("bucket_set") or []))
        return {
            "bucket_set": selected.get("bucket_set"),
            "reason": "smallest passing measured prefix-off set with zero eager decode",
        }
    selected = min(passing, key=lambda run: int(run["eager_count_total"]))
    return {
        "bucket_set": selected.get("bucket_set"),
        "reason": "no zero-eager set; selected the passing set with the fewest eager decodes",
    }


def _coverage_cell(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        wall = row.get("wall_share")
        wall_pct = "-" if wall is None else f"{100.0 * float(wall):.1f}%"
        parts.append(
            "bs{bs}: r{r}/e{e}/tok{tok}/{wall}".format(
                bs=row["actual_decode_bs"],
                r=int(row.get("replay_count") or 0),
                e=int(row.get("eager_count") or 0),
                tok=int(row.get("tokens") or 0),
                wall=wall_pct,
            )
        )
    return "<br>".join(parts) if parts else "-"


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TARGET 08.05 Bucket Policy Summary",
        "",
        f"Recommended bucket set: `{summary['recommendation'].get('bucket_set')}`.",
        "",
        "## Runs",
        "",
        (
            "| bucket | mode | status | captured | capture GiB | capture s | "
            "replay | eager | mean output tok/s | mean decode tok/s |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        graph = run["graph_capture"]
        lines.append(
            "| `{bucket}` | {mode} | {status} | `{captured}` | {cap_gib} | {cap_s} | "
            "{replay} | {eager} | {out} | {decode} |".format(
                bucket=run["bucket_set"],
                mode=run["prefix_mode"],
                status=run["status"],
                captured=graph.get("captured_bs", []),
                cap_gib=_fmt_gib(graph.get("capture_memory_delta_bytes")),
                cap_s=_fmt_float(graph.get("capture_elapsed_s"), 2),
                replay=int(run["replay_count_total"]),
                eager=int(run["eager_count_total"]),
                out=_fmt_float(run.get("output_tokens_per_s_mean"), 2),
                decode=_fmt_float(run.get("decode_tokens_per_s_mean"), 2),
            )
        )

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| bucket | mode | actual decode bs coverage |",
            "| --- | --- | --- |",
        ]
    )
    for run in summary["runs"]:
        lines.append(
            f"| `{run['bucket_set']}` | {run['prefix_mode']} | "
            f"{_coverage_cell(run['coverage_totals'])} |"
        )

    lines.extend(
        [
            "",
            "## Workloads",
            "",
            (
                "| bucket | mode | scenario | status | output tok/s | decode tok/s | "
                "TTFT s | TPOT s | replay | eager | peak alloc GiB | peak reserved GiB |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in summary["runs"]:
        for case in run["cases"]:
            graph = case["graph"]
            lines.append(
                "| `{bucket}` | {mode} | `{scenario}` | {status} | {out} | {decode} | "
                "{ttft} | {tpot} | {replay} | {eager} | {alloc} | {reserved} |".format(
                    bucket=run["bucket_set"],
                    mode=run["prefix_mode"],
                    scenario=case["scenario"],
                    status=case["status"],
                    out=_fmt_float(case.get("output_tokens_per_s"), 2),
                    decode=_fmt_float(case.get("decode_tokens_per_s"), 2),
                    ttft=_fmt_float(case.get("ttft_s_mean"), 2),
                    tpot=_fmt_float(case.get("tpot_s_mean"), 4),
                    replay=int(graph.get("replay_count") or 0),
                    eager=int(graph.get("eager_decode_count") or 0),
                    alloc=_fmt_gib(case.get("peak_allocated_bytes")),
                    reserved=_fmt_gib(case.get("peak_reserved_bytes")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def summarize(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    runs = [
        run
        for run in (_run_summary(path) for path in sorted(raw_dir.glob("bucket_*")))
        if run is not None
    ]
    summary = {
        "milestone_dir": str(milestone_dir),
        "runs": runs,
        "recommendation": _recommendation(runs),
    }
    _write_json(milestone_dir / "summaries" / "bucket_policy_summary.json", summary)
    _write_text(
        milestone_dir / "summaries" / "bucket_policy_summary.md",
        _markdown(summary),
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milestone-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(args.milestone_dir)
    print(json.dumps(summary["recommendation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

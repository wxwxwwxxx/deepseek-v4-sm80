from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


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
    return float(value) / (1024**3)


def _fmt(value: int | float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_gib(value: int | float | None) -> str:
    gib = _gib(value)
    return "-" if gib is None else f"{gib:.2f}"


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _first_report(run_dir: Path) -> dict[str, Any] | None:
    report_dir = run_dir / "reports"
    if not report_dir.exists():
        return None
    paths = sorted(path for path in report_dir.glob("*.json") if ".rank" not in path.name)
    if not paths:
        return None
    return _read_json(paths[0])


def _capture_statuses(report: dict[str, Any], run_config: dict[str, Any]) -> list[dict[str, Any]]:
    statuses = []
    for payload in report.get("per_rank", []):
        graph = payload.get("graph_runner_before_case") or payload.get("graph_runner_after_case")
        if graph:
            statuses.append(graph)
    if statuses:
        return statuses
    graph = run_config.get("config", {}).get("graph_runner", {})
    return [graph] if graph else []


def _sum_graph_counter(report: dict[str, Any], name: str) -> int:
    graph = report.get("config", {}).get("graph_runner_case") or {}
    return int(graph.get(name) or 0)


def _run_kind(name: str) -> str:
    if name.startswith("bucketset_"):
        return "bucket_set"
    if name.startswith("single_"):
        return "single_bucket"
    if name.startswith("greedy_"):
        return "greedy_ab"
    if name.startswith("metadata_"):
        return "metadata_ab"
    if name.startswith("seq"):
        return "seq_pages"
    return "unknown"


def _summarize_run(run_dir: Path) -> dict[str, Any] | None:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return None
    report = _first_report(run_dir)
    if report is None:
        return None
    run_config = _read_json(run_config_path)
    config = run_config.get("config", {})
    report_config = report.get("config", {})
    statuses = _capture_statuses(report, run_config)
    rank_deltas = [
        int(status.get("capture_memory_delta_bytes") or 0)
        for status in statuses
        if status.get("capture_memory_delta_bytes") is not None
    ]
    rank_free_before = [
        int(status.get("capture_free_memory_before_bytes") or 0)
        for status in statuses
        if status.get("capture_free_memory_before_bytes") is not None
    ]
    rank_free_after = [
        int(status.get("capture_free_memory_after_bytes") or 0)
        for status in statuses
        if status.get("capture_free_memory_after_bytes") is not None
    ]
    rank0_graph = statuses[0] if statuses else config.get("graph_runner", {})
    metrics = report.get("metrics", {})
    memory = {
        "peak_allocated_bytes": metrics.get("peak_gpu_memory_allocated_bytes"),
        "peak_reserved_bytes": metrics.get("peak_gpu_memory_reserved_bytes"),
        "kv_cache_memory_bytes_per_rank_max": metrics.get("kv_cache_memory_bytes_per_rank_max"),
    }
    return {
        "name": run_dir.name,
        "kind": _run_kind(run_dir.name),
        "status": report.get("status"),
        "bucket_set": config.get("cuda_graph_bs") or [],
        "captured_bs": rank0_graph.get("captured_bs", []),
        "max_seq_len": config.get("max_seq_len") or report_config.get("max_seq_len"),
        "num_pages": config.get("num_pages"),
        "page_size": config.get("page_size"),
        "greedy_sample": bool(rank0_graph.get("capture_greedy_sample")),
        "compressed_locs_in_graph": bool(rank0_graph.get("capture_compressed_locs_in_graph")),
        "compressed_locs_disabled_by_env": bool(
            rank0_graph.get("capture_compressed_locs_in_graph_disabled_by_env")
        ),
        "graph_pool_reuse_enabled": bool(rank0_graph.get("capture_graph_pool_reuse_enabled")),
        "graph_pool_reuse_anchor_bs": rank0_graph.get("capture_graph_pool_reuse_anchor_bs"),
        "capture_elapsed_s": rank0_graph.get("capture_elapsed_s"),
        "capture_buffer_bytes": rank0_graph.get("capture_buffer_bytes"),
        "capture_free_memory_before_bytes_rank0": rank0_graph.get(
            "capture_free_memory_before_bytes"
        ),
        "capture_free_memory_after_bytes_rank0": rank0_graph.get(
            "capture_free_memory_after_bytes"
        ),
        "capture_memory_delta_bytes_rank0": rank0_graph.get("capture_memory_delta_bytes"),
        "capture_memory_delta_bytes_rank_mean": (
            int(statistics.mean(rank_deltas)) if rank_deltas else None
        ),
        "capture_memory_delta_bytes_rank_min": min(rank_deltas) if rank_deltas else None,
        "capture_memory_delta_bytes_rank_max": max(rank_deltas) if rank_deltas else None,
        "capture_free_memory_before_bytes_rank_mean": (
            int(statistics.mean(rank_free_before)) if rank_free_before else None
        ),
        "capture_free_memory_after_bytes_rank_mean": (
            int(statistics.mean(rank_free_after)) if rank_free_after else None
        ),
        "capture_memory_allocated_before_bytes": rank0_graph.get(
            "capture_memory_allocated_before_bytes"
        ),
        "capture_memory_allocated_after_bytes": rank0_graph.get(
            "capture_memory_allocated_after_bytes"
        ),
        "capture_memory_reserved_before_bytes": rank0_graph.get(
            "capture_memory_reserved_before_bytes"
        ),
        "capture_memory_reserved_after_bytes": rank0_graph.get(
            "capture_memory_reserved_after_bytes"
        ),
        "capture_peak_memory_allocated_bytes": rank0_graph.get(
            "capture_peak_memory_allocated_bytes"
        ),
        "capture_peak_memory_reserved_bytes": rank0_graph.get(
            "capture_peak_memory_reserved_bytes"
        ),
        "capture_by_batch_size": rank0_graph.get("capture_by_batch_size", {}),
        "replay_count": _sum_graph_counter(report, "replay_count"),
        "eager_decode_count": _sum_graph_counter(report, "eager_decode_count"),
        "greedy_sample_replay_count": _sum_graph_counter(report, "greedy_sample_replay_count"),
        "replay_count_by_batch_size": (
            report.get("config", {}).get("graph_runner_case") or {}
        ).get("replay_count_by_batch_size", {}),
        "eager_decode_count_by_batch_size": (
            report.get("config", {}).get("graph_runner_case") or {}
        ).get("eager_decode_count_by_batch_size", {}),
        "output_tokens_per_s": metrics.get("end_to_end_output_tokens_per_s"),
        "decode_tokens_per_s": metrics.get("decode_tokens_per_s"),
        "memory": memory,
        "report_path": str(Path(report.get("report_path", ""))),
    }


def _baseline(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for run in runs:
        if run["name"] == "bucketset_1_2_4_8_16_np128_sl2048_greedy_on_metadata_on":
            return run
    return None


def _delta_from_baseline(
    runs: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    base = None if baseline is None else baseline.get("capture_memory_delta_bytes_rank0")
    out = []
    for run in runs:
        delta = run.get("capture_memory_delta_bytes_rank0")
        out.append(
            {
                **run,
                "capture_delta_vs_baseline_bytes": (
                    None if base is None or delta is None else int(delta) - int(base)
                ),
            }
        )
    return out


def _per_bucket_cell(run: dict[str, Any]) -> str:
    parts = []
    for bs, payload in sorted(
        run.get("capture_by_batch_size", {}).items(),
        key=lambda item: int(item[0]),
        reverse=True,
    ):
        parts.append(f"bs{bs}: {_fmt_gib(payload.get('memory_delta_bytes'))}")
    return "<br>".join(parts) if parts else "-"


def _markdown(summary: dict[str, Any]) -> str:
    runs = summary["runs"]
    lines = [
        "# TARGET 08.06 CUDA Graph Memory Attribution Summary",
        "",
        "## Runs",
        "",
        (
            "| run | kind | buckets | max_seq_len | pages | greedy | metadata in graph | "
            "captured | delta GiB rank0 | delta GiB mean | capture s | pool reuse | replay/eager |"
        ),
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for run in runs:
        lines.append(
            "| `{name}` | {kind} | `{buckets}` | {seq} | {pages} | {greedy} | {meta} | "
            "`{captured}` | {delta0} | {deltamean} | {elapsed} | {pool} | {replay}/{eager} |".format(
                name=run["name"],
                kind=run["kind"],
                buckets=run["bucket_set"],
                seq=run["max_seq_len"],
                pages=run["num_pages"],
                greedy="on" if run["greedy_sample"] else "off",
                meta="on" if run["compressed_locs_in_graph"] else "off",
                captured=run["captured_bs"],
                delta0=_fmt_gib(run.get("capture_memory_delta_bytes_rank0")),
                deltamean=_fmt_gib(run.get("capture_memory_delta_bytes_rank_mean")),
                elapsed=_fmt(run.get("capture_elapsed_s"), 2),
                pool="yes" if run["graph_pool_reuse_enabled"] else "single/none",
                replay=int(run.get("replay_count") or 0),
                eager=int(run.get("eager_decode_count") or 0),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Bucket Free-Memory Delta",
            "",
            "| run | per-bucket delta GiB |",
            "| --- | --- |",
        ]
    )
    for run in runs:
        lines.append(f"| `{run['name']}` | {_per_bucket_cell(run)} |")
    return "\n".join(lines) + "\n"


def summarize(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    runs = []
    for run_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        run = _summarize_run(run_dir)
        if run is not None:
            runs.append(run)
    baseline = _baseline(runs)
    runs_with_delta = _delta_from_baseline(runs, baseline)
    rank0_deltas = [
        int(run["capture_memory_delta_bytes_rank0"])
        for run in runs
        if run.get("capture_memory_delta_bytes_rank0") is not None
    ]
    summary = {
        "baseline_run": None if baseline is None else baseline["name"],
        "runs": runs_with_delta,
        "capture_delta_gib_rank0_range": {
            "min": None if not rank0_deltas else _gib(min(rank0_deltas)),
            "max": None if not rank0_deltas else _gib(max(rank0_deltas)),
        },
    }
    summaries_dir = milestone_dir / "summaries"
    _write_json(summaries_dir / "cuda_graph_memory_attribution_summary.json", summary)
    _write_text(
        summaries_dir / "cuda_graph_memory_attribution_summary.md",
        _markdown(summary),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.milestone_dir)


if __name__ == "__main__":
    main()

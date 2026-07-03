from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


OWNER_KEYS = {
    "q_wqb": "q_wqb_bf16_weight_cache",
    "wo_b": "wo_b_bf16_weight_cache",
    "wo_a": "wo_a_bf16_bmm_cache",
    "indexer_wq_b": "indexer_wq_b_bf16_weight_cache",
    "shared_expert": "shared_expert_bf16_weight_cache",
}

RUN_ORDER = {
    "single_full_victory": 0,
    "single_no_projection_bf16_caches": 1,
    "single_no_q_wqb_bf16_cache": 2,
    "single_no_wo_b_bf16_cache": 3,
    "single_no_wo_a_bf16_bmm_cache": 4,
    "single_no_indexer_wq_b_bf16_cache": 5,
    "single_no_shared_expert_bf16_cache": 6,
    "single_no_all_tested_bf16_caches": 7,
    "full_full_victory": 20,
    "full_no_projection_bf16_caches": 21,
    "full_no_q_wqb_bf16_cache": 22,
    "full_no_wo_b_bf16_cache": 23,
    "full_no_wo_a_bf16_bmm_cache": 24,
    "full_no_indexer_wq_b_bf16_cache": 25,
    "full_no_shared_expert_bf16_cache": 26,
    "full_no_all_tested_bf16_caches": 27,
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
    return float(value) / (1024**3)


def _fmt_gib(value: int | float | None, digits: int = 3) -> str:
    gib = _gib(value)
    return "-" if gib is None else f"{gib:.{digits}f}"


def _fmt(value: int | float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


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


def _owner_cache(report: dict[str, Any], owner: str) -> dict[str, Any]:
    prepare = report.get("config", {}).get("model_prepare_report_rank0", {})
    payload = prepare.get(OWNER_KEYS[owner], {})
    return {
        "enabled": bool(payload.get("enabled")),
        "layers_cached": int(payload.get("layers_cached") or 0),
        "total_bytes": int(payload.get("total_bytes") or 0),
        "total_pretransposed_bytes": int(payload.get("total_pretransposed_bytes") or 0),
    }


def _prepare_report(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("config", {}).get("model_prepare_report_rank0", {})


def _run_kind(name: str) -> str:
    if name.startswith("single_"):
        return "single_bs16"
    if name.startswith("full_"):
        return "full_buckets"
    return "unknown"


def _rank_stat(statuses: list[dict[str, Any]], field: str) -> dict[str, int | None]:
    values = [
        int(status[field])
        for status in statuses
        if status.get(field) is not None
    ]
    if not values:
        return {"rank0": None, "mean": None, "min": None, "max": None}
    return {
        "rank0": values[0],
        "mean": int(statistics.mean(values)),
        "min": min(values),
        "max": max(values),
    }


def _summarize_run(run_dir: Path) -> dict[str, Any] | None:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return None
    report = _first_report(run_dir)
    if report is None:
        return None
    run_config = _read_json(run_config_path)
    config = run_config.get("config", {})
    statuses = _capture_statuses(report, run_config)
    rank0_graph = statuses[0] if statuses else config.get("graph_runner", {})
    prepare = _prepare_report(report)
    owner_caches = {owner: _owner_cache(report, owner) for owner in OWNER_KEYS}
    projection_total = prepare.get("projection_bf16_weight_cache_total", {})
    disable = prepare.get("attribution_disable_toggles", {})
    raw_env = (
        report.get("variant", {})
        .get("raw_dsv4_sm80_env", {})
        .get("MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES", "")
    )
    memory_delta = rank0_graph.get("capture_memory_delta_bytes")
    alloc_before = rank0_graph.get("capture_memory_allocated_before_bytes")
    alloc_after = rank0_graph.get("capture_memory_allocated_after_bytes")
    reserved_before = rank0_graph.get("capture_memory_reserved_before_bytes")
    reserved_after = rank0_graph.get("capture_memory_reserved_after_bytes")
    return {
        "name": run_dir.name,
        "kind": _run_kind(run_dir.name),
        "status": report.get("status"),
        "bucket_set": config.get("cuda_graph_bs") or [],
        "captured_bs": rank0_graph.get("captured_bs", []),
        "max_seq_len": config.get("max_seq_len") or report.get("config", {}).get("max_seq_len"),
        "num_pages": config.get("num_pages"),
        "page_size": config.get("page_size"),
        "disable_toggles_raw": disable.get("raw", raw_env),
        "disabled_toggles": disable.get("disabled_toggles", []),
        "owner_caches": owner_caches,
        "enabled_bf16_cache_owners": projection_total.get("owners", []),
        "persistent_tested_bf16_cache_total_bytes": int(
            projection_total.get("total_bytes") or 0
        ),
        "bf16_small_gemm_pretranspose_total_bytes": int(
            prepare.get("bf16_small_gemm_pretranspose_cache_total", {}).get(
                "total_pretransposed_bytes", 0
            )
            or 0
        ),
        "dense_fp8_marlin_projection_cache": prepare.get("dense_fp8_marlin_projection_cache", {}),
        "capture_free_memory_before_bytes_rank0": rank0_graph.get(
            "capture_free_memory_before_bytes"
        ),
        "capture_free_memory_after_bytes_rank0": rank0_graph.get(
            "capture_free_memory_after_bytes"
        ),
        "capture_memory_delta_bytes_rank0": memory_delta,
        "capture_memory_delta_bytes_rank_stats": _rank_stat(
            statuses, "capture_memory_delta_bytes"
        ),
        "capture_memory_allocated_before_bytes": alloc_before,
        "capture_memory_allocated_after_bytes": alloc_after,
        "capture_memory_allocated_delta_bytes": (
            None if alloc_before is None or alloc_after is None else int(alloc_after) - int(alloc_before)
        ),
        "capture_memory_reserved_before_bytes": reserved_before,
        "capture_memory_reserved_after_bytes": reserved_after,
        "capture_memory_reserved_delta_bytes": (
            None
            if reserved_before is None or reserved_after is None
            else int(reserved_after) - int(reserved_before)
        ),
        "capture_peak_memory_allocated_bytes": rank0_graph.get(
            "capture_peak_memory_allocated_bytes"
        ),
        "capture_peak_memory_reserved_bytes": rank0_graph.get("capture_peak_memory_reserved_bytes"),
        "capture_elapsed_s": rank0_graph.get("capture_elapsed_s"),
        "capture_by_batch_size": rank0_graph.get("capture_by_batch_size", {}),
        "capture_graph_pool_reuse_enabled": bool(
            rank0_graph.get("capture_graph_pool_reuse_enabled")
        ),
        "capture_graph_pool_reuse_anchor_bs": rank0_graph.get(
            "capture_graph_pool_reuse_anchor_bs"
        ),
        "replay_count": _sum_graph_counter(report, "replay_count"),
        "eager_decode_count": _sum_graph_counter(report, "eager_decode_count"),
        "greedy_sample_replay_count": _sum_graph_counter(report, "greedy_sample_replay_count"),
        "output_tokens_per_s": report.get("metrics", {}).get("end_to_end_output_tokens_per_s"),
        "decode_tokens_per_s": report.get("metrics", {}).get("decode_tokens_per_s"),
        "report_path": str(Path(report.get("report_path", ""))),
    }


def _add_baseline_deltas(runs: list[dict[str, Any]]) -> None:
    baselines = {
        run["kind"]: run.get("capture_memory_delta_bytes_rank0")
        for run in runs
        if run["name"] in {"single_full_victory", "full_full_victory"}
    }
    single_base = baselines.get("single_bs16")
    for run in runs:
        delta = run.get("capture_memory_delta_bytes_rank0")
        base = baselines.get(run["kind"])
        run["capture_delta_vs_kind_baseline_bytes"] = (
            None if base is None or delta is None else int(delta) - int(base)
        )
        run["capture_delta_vs_single_baseline_bytes"] = (
            None if single_base is None or delta is None else int(delta) - int(single_base)
        )


def _material_phase2_candidates(runs: list[dict[str, Any]], threshold_bytes: int) -> list[str]:
    candidates = []
    for run in runs:
        if run["kind"] != "single_bs16" or run["name"] == "single_full_victory":
            continue
        delta = run.get("capture_delta_vs_kind_baseline_bytes")
        if delta is not None and abs(int(delta)) > threshold_bytes:
            candidates.append(run["name"].removeprefix("single_"))
    return candidates


def _small_fix_candidates(runs: list[dict[str, Any]], threshold_bytes: int) -> list[str]:
    candidates = []
    for run in runs:
        if run["kind"] != "single_bs16" or run["name"] == "single_full_victory":
            continue
        delta = run.get("capture_delta_vs_kind_baseline_bytes")
        if delta is not None and int(delta) <= -threshold_bytes:
            candidates.append(run["name"].removeprefix("single_"))
    return candidates


def _enabled_cell(run: dict[str, Any]) -> str:
    labels = []
    for owner in ("q_wqb", "wo_b", "wo_a", "indexer_wq_b", "shared_expert"):
        labels.append("Y" if run["owner_caches"][owner]["enabled"] else "N")
    return "/".join(labels)


def _owner_bytes_cell(run: dict[str, Any]) -> str:
    parts = []
    for owner in ("q_wqb", "wo_b", "wo_a", "indexer_wq_b", "shared_expert"):
        parts.append(f"{owner}:{_fmt_gib(run['owner_caches'][owner]['total_bytes'])}")
    return "<br>".join(parts)


def _main_table(runs: list[dict[str, Any]], kind: str) -> list[str]:
    rows = [
        (
            "| run | buckets | denylist | enabled q/woB/woA/idx/shared | persistent GiB | "
            "free before/after/delta GiB | alloc delta GiB | reserved delta GiB | "
            "vs baseline GiB | replay/eager |"
        ),
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in [item for item in runs if item["kind"] == kind]:
        rows.append(
            "| `{name}` | `{buckets}` | `{deny}` | `{enabled}` | {persistent} | "
            "{free_before}/{free_after}/{free_delta} | {alloc_delta} | {reserved_delta} | "
            "{vs_base} | {replay}/{eager} |".format(
                name=run["name"],
                buckets=run["bucket_set"],
                deny=run["disable_toggles_raw"] or "",
                enabled=_enabled_cell(run),
                persistent=_fmt_gib(run["persistent_tested_bf16_cache_total_bytes"]),
                free_before=_fmt_gib(run.get("capture_free_memory_before_bytes_rank0")),
                free_after=_fmt_gib(run.get("capture_free_memory_after_bytes_rank0")),
                free_delta=_fmt_gib(run.get("capture_memory_delta_bytes_rank0")),
                alloc_delta=_fmt_gib(run.get("capture_memory_allocated_delta_bytes")),
                reserved_delta=_fmt_gib(run.get("capture_memory_reserved_delta_bytes")),
                vs_base=_fmt_gib(run.get("capture_delta_vs_kind_baseline_bytes")),
                replay=int(run.get("replay_count") or 0),
                eager=int(run.get("eager_decode_count") or 0),
            )
        )
    return rows


def _cache_matrix(runs: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| run | disabled toggles | owner bytes GiB | enabled owners |",
        "| --- | --- | --- | --- |",
    ]
    for run in runs:
        rows.append(
            "| `{name}` | `{disabled}` | {bytes_cell} | `{owners}` |".format(
                name=run["name"],
                disabled=run.get("disabled_toggles", []),
                bytes_cell=_owner_bytes_cell(run),
                owners=run.get("enabled_bf16_cache_owners", []),
            )
        )
    return rows


def _markdown(summary: dict[str, Any]) -> str:
    runs = summary["runs"]
    lines = [
        "# TARGET 08.07 BF16 Cache Graph Memory Attribution Summary",
        "",
        "All GiB values use bytes / 2^30.",
        "",
        "## Single-Bucket Attribution",
        "",
        *_main_table(runs, "single_bs16"),
        "",
        "## Full-Bucket Confirmation",
        "",
        *_main_table(runs, "full_buckets"),
        "",
        "## Cache Owner Matrix",
        "",
        *_cache_matrix(runs),
        "",
        "## Materiality",
        "",
        f"- Phase-2 threshold: > {_fmt_gib(summary['phase2_threshold_bytes'])} GiB/rank.",
        f"- Small-fix threshold: graph-delta reduction > {_fmt_gib(summary['small_fix_threshold_bytes'])} GiB/rank.",
        f"- Phase-2 candidates from single-bucket data: `{summary['material_phase2_candidates']}`.",
        f"- Small-fix candidates from single-bucket data: `{summary['small_fix_candidates']}`.",
        "",
    ]
    return "\n".join(lines)


def summarize(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    runs = []
    for run_dir in sorted(raw_dir.iterdir() if raw_dir.exists() else []):
        if not run_dir.is_dir():
            continue
        run = _summarize_run(run_dir)
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda item: (RUN_ORDER.get(item["name"], 1000), item["name"]))
    _add_baseline_deltas(runs)
    phase2_threshold = 1 * 1024**3
    small_fix_threshold = 2 * 1024**3
    summary = {
        "milestone_dir": str(milestone_dir),
        "runs": runs,
        "phase2_threshold_bytes": phase2_threshold,
        "small_fix_threshold_bytes": small_fix_threshold,
        "material_phase2_candidates": _material_phase2_candidates(runs, phase2_threshold),
        "small_fix_candidates": _small_fix_candidates(runs, small_fix_threshold),
    }
    _write_json(
        milestone_dir / "summaries" / "bf16_cache_graph_memory_attribution_summary.json",
        summary,
    )
    _write_text(
        milestone_dir / "summaries" / "bf16_cache_graph_memory_attribution_summary.md",
        _markdown(summary),
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milestone-dir",
        type=Path,
        default=Path("performance_milestones/target08_bf16_cache_graph_memory_attribution"),
    )
    args = parser.parse_args()
    summarize(args.milestone_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

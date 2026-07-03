from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


BASELINE = "dsv4_sm80_a100_victory"
CANDIDATE = "dsv4_sm80_a100_victory_densefp8marlinproj"
LONG_SHAPE = "4096x1024"
SHORT_SHAPE = "4096x128"
SCENARIO = "decode_throughput_bs8"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report_path(raw_dir: Path, shape: str, variant: str) -> Path:
    report_dir = raw_dir / f"{shape}_{variant}" / "reports"
    matches = sorted(report_dir.glob(f"*{SCENARIO}__{variant}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one report for {shape} {variant}, found {matches}")
    return matches[0]


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.mean(filtered))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def _stats(values: list[float], *, higher_is_better: bool) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "best": None,
            "worst": None,
            "std": None,
            "cv": None,
        }
    mean = float(statistics.mean(values))
    std = _std(values)
    best = max(values) if higher_is_better else min(values)
    worst = min(values) if higher_is_better else max(values)
    return {
        "mean": mean,
        "median": _median(values),
        "best": float(best),
        "worst": float(worst),
        "std": std,
        "cv": None if mean == 0 else float(std / abs(mean)),
    }


def _repeat_row(repeat: dict[str, Any]) -> dict[str, Any]:
    phase = repeat.get("phase_totals", {})
    elapsed_s = float(repeat.get("elapsed_s") or 0.0)
    output_tokens = int(repeat.get("actual_output_tokens") or 0)
    decode_tokens = int(phase.get("decode_tokens") or 0)
    decode_forward_s = float(phase.get("decode_forward_s") or 0.0)
    requests = repeat.get("requests", [])
    return {
        "repeat_index": int(repeat.get("repeat_index", 0)),
        "elapsed_s": elapsed_s,
        "output_tokens_per_s": None if elapsed_s <= 0 else output_tokens / elapsed_s,
        "decode_tokens_per_s": None if decode_forward_s <= 0 else decode_tokens / decode_forward_s,
        "ttft_s_mean": _mean(request.get("ttft_s") for request in requests),
        "prefill_forward_s": float(phase.get("prefill_forward_s") or 0.0),
        "decode_forward_s": decode_forward_s,
        "prefill_prepare_s": float(phase.get("prefill_prepare_s") or 0.0),
        "decode_prepare_s": float(phase.get("decode_prepare_s") or 0.0),
        "actual_output_tokens": output_tokens,
        "target_output_tokens": int(repeat.get("target_output_tokens") or 0),
    }


def _shape_variant_summary(raw_dir: Path, shape: str, variant: str) -> dict[str, Any]:
    report_path = _report_path(raw_dir, shape, variant)
    report = _read_json(report_path)
    repeats = [_repeat_row(repeat) for repeat in report.get("repeats", [])]
    metric_keys = (
        "output_tokens_per_s",
        "decode_tokens_per_s",
        "ttft_s_mean",
        "prefill_forward_s",
        "decode_forward_s",
        "elapsed_s",
    )
    higher_is_better = {
        "output_tokens_per_s": True,
        "decode_tokens_per_s": True,
        "ttft_s_mean": False,
        "prefill_forward_s": False,
        "decode_forward_s": False,
        "elapsed_s": False,
    }
    stats = {
        key: _stats(
            [float(row[key]) for row in repeats if row.get(key) is not None],
            higher_is_better=higher_is_better[key],
        )
        for key in metric_keys
    }
    graph = report.get("config", {}).get("graph_runner", {})
    memory = report.get("metrics", {})
    return {
        "report_path": str(report_path),
        "status": report.get("status"),
        "variant": variant,
        "shape": shape,
        "warmup": report.get("per_rank", [{}])[0].get("warmup", {}),
        "repeats": repeats,
        "stats": stats,
        "graph": {
            "enabled": bool(graph.get("enabled")),
            "captured_bs": graph.get("captured_bs", []),
            "replay_count": int(graph.get("replay_count") or 0),
            "replay_count_by_batch_size": graph.get("replay_count_by_batch_size", {}),
            "replay_count_by_padded_size": graph.get("replay_count_by_padded_size", {}),
            "greedy_sample_replay_count": int(graph.get("greedy_sample_replay_count") or 0),
            "eager_decode_count": int(graph.get("eager_decode_count") or 0),
        },
        "memory": {
            "peak_gpu_memory_allocated_bytes": memory.get("peak_gpu_memory_allocated_bytes"),
            "peak_gpu_memory_reserved_bytes": memory.get("peak_gpu_memory_reserved_bytes"),
            "kv_cache_memory_bytes_per_rank_max": memory.get("kv_cache_memory_bytes_per_rank_max"),
        },
        "load_init": report.get("load_init", {}),
        "variant_env": report.get("variant", {}),
        "model_prepare_report_rank0": report.get("config", {}).get("model_prepare_report_rank0", {}),
    }


def _pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (candidate / baseline - 1.0) * 100.0


def _catastrophic_repeat_regressions(
    candidate: dict[str, Any], baseline: dict[str, Any], *, threshold_pct: float = -3.0
) -> list[dict[str, Any]]:
    regressions = []
    baseline_repeats = baseline["repeats"]
    candidate_repeats = candidate["repeats"]
    for idx in range(min(len(baseline_repeats), len(candidate_repeats))):
        b = baseline_repeats[idx].get("output_tokens_per_s")
        c = candidate_repeats[idx].get("output_tokens_per_s")
        delta = _pct(c, b)
        if delta is not None and delta < threshold_pct:
            regressions.append({"repeat_index": idx, "delta_pct": delta, "baseline": b, "candidate": c})
    return regressions


def _decision(summary: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    long_base = summary["shapes"][LONG_SHAPE][BASELINE]
    long_cand = summary["shapes"][LONG_SHAPE][CANDIDATE]
    short_base = summary["shapes"][SHORT_SHAPE][BASELINE]
    short_cand = summary["shapes"][SHORT_SHAPE][CANDIDATE]

    long_median_delta = _pct(
        long_cand["stats"]["output_tokens_per_s"]["median"],
        long_base["stats"]["output_tokens_per_s"]["median"],
    )
    long_mean_delta = _pct(
        long_cand["stats"]["output_tokens_per_s"]["mean"],
        long_base["stats"]["output_tokens_per_s"]["mean"],
    )
    short_median_delta = _pct(
        short_cand["stats"]["output_tokens_per_s"]["median"],
        short_base["stats"]["output_tokens_per_s"]["median"],
    )
    base_cv = long_base["stats"]["output_tokens_per_s"]["cv"]
    cand_cv = long_cand["stats"]["output_tokens_per_s"]["cv"]
    cv_threshold = None
    if base_cv is not None:
        cv_threshold = max(float(base_cv) * 1.5, 0.02)
    cv_ok = cand_cv is not None and cv_threshold is not None and float(cand_cv) <= cv_threshold
    catastrophic = _catastrophic_repeat_regressions(long_cand, long_base)

    smoke_variant = (smoke.get("variants") or [{}])[0]
    smoke_ok = smoke.get("status") == "pass" and smoke_variant.get("status") == "pass"
    long_graph_ok = all(
        item["graph"]["replay_count"] > 0 and item["graph"]["eager_decode_count"] == 0
        for item in (long_base, long_cand)
    )
    short_graph_ok = all(
        item["graph"]["replay_count"] > 0 and item["graph"]["eager_decode_count"] == 0
        for item in (short_base, short_cand)
    )

    promote = (
        smoke_ok
        and long_graph_ok
        and short_graph_ok
        and long_median_delta is not None
        and long_median_delta >= 2.0
        and long_mean_delta is not None
        and long_mean_delta >= 1.0
        and short_median_delta is not None
        and short_median_delta >= -1.0
        and cv_ok
        and not catastrophic
    )

    if promote:
        outcome = "promote"
        reason = "candidate clears all promotion thresholds"
    elif long_median_delta is not None and -1.0 <= long_median_delta <= 2.0:
        outcome = "keep opt-in"
        reason = "4096/1024 median delta is inside the neutral [-1%, +2%] band"
    else:
        outcome = "revert/repair"
        reason = "candidate misses the repeat-stable promotion gate outside the neutral band"

    return {
        "outcome": outcome,
        "reason": reason,
        "thresholds": {
            "long_median_output_delta_pct": long_median_delta,
            "long_mean_output_delta_pct": long_mean_delta,
            "short_median_output_delta_pct": short_median_delta,
            "baseline_long_output_cv": base_cv,
            "candidate_long_output_cv": cand_cv,
            "candidate_cv_threshold": cv_threshold,
            "candidate_cv_ok": cv_ok,
            "catastrophic_repeat_regressions": catastrophic,
            "smoke_ok": smoke_ok,
            "long_graph_ok": long_graph_ok,
            "short_graph_ok": short_graph_ok,
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(summary: dict[str, Any], shape: str, metric: str) -> str:
    rows = ["| Variant | Mean | Median | Best | Worst | Std | CV |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in (BASELINE, CANDIDATE):
        stats = summary["shapes"][shape][variant]["stats"][metric]
        rows.append(
            "| "
            + " | ".join(
                [
                    f"`{variant}`",
                    _fmt(stats["mean"]),
                    _fmt(stats["median"]),
                    _fmt(stats["best"]),
                    _fmt(stats["worst"]),
                    _fmt(stats["std"]),
                    _fmt(stats["cv"], 6),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    lines = [
        "# TARGET 07.78 Summary",
        "",
        f"Decision: `{decision['outcome']}`.",
        "",
        "4096/1024 output tok/s:",
        "",
        _markdown_table(summary, LONG_SHAPE, "output_tokens_per_s"),
        "",
        "4096/128 output tok/s:",
        "",
        _markdown_table(summary, SHORT_SHAPE, "output_tokens_per_s"),
        "",
        "Promotion thresholds:",
        "",
    ]
    for key, value in decision["thresholds"].items():
        lines.append(f"- `{key}`: `{_fmt(value, 6)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.milestone_dir / "raw"
    smoke_path = raw_dir / f"smoke_{CANDIDATE}" / "text_smoke.json"
    smoke = _read_json(smoke_path)

    summary: dict[str, Any] = {
        "lifecycle_route": "Option B: separate torchrun per variant",
        "smoke_path": str(smoke_path),
        "smoke": smoke,
        "shapes": {
            shape: {
                variant: _shape_variant_summary(raw_dir, shape, variant)
                for variant in (BASELINE, CANDIDATE)
            }
            for shape in (LONG_SHAPE, SHORT_SHAPE)
        },
    }
    summary["decision"] = _decision(summary, smoke)

    out_json = args.milestone_dir / "summaries" / "target0778_summary.json"
    out_md = args.milestone_dir / "summaries" / "target0778_summary.md"
    _write_json(out_json, summary)
    _write_markdown_summary(out_md, summary)
    print(json.dumps({"summary_json": str(out_json), "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BENCH_PATH = ROOT / "benchmark/offline/deepseek_v4_perf_matrix.py"
PAGE_SIZE = 256


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("deepseek_v4_perf_matrix", BENCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _align_down(value: int, page_size: int) -> int:
    return value // page_size * page_size


def _phase1_hit_for_identical_prompt(prompt_len: int, page_size: int) -> int:
    warm_insert_len = _align_down(prompt_len, page_size)
    matchable_probe_len = max(prompt_len - 1, 0)
    return min(warm_insert_len, _align_down(matchable_probe_len, page_size))


def _route_b_hit_for_identical_prompt(prompt_len: int, page_size: int) -> int:
    phase1 = _phase1_hit_for_identical_prompt(prompt_len, page_size)
    if prompt_len % page_size == 0:
        return 0
    return phase1


def _scenario_names(milestone_dir: Path) -> list[str]:
    run_config = milestone_dir / "raw/perf_route_b_graph/run_config.json"
    if run_config.exists():
        config = _read_json(run_config)
        names = [str(row["name"]) for row in config.get("scenarios", [])]
        if names:
            return names
    return [
        "decode_ladder_bs16",
        "serving_mixed_112req_wave16",
        "prefix_full_hit_257_bs4",
        "prefix_full_hit_512_bs4",
        "prefix_full_hit_513_bs4",
        "prefix_full_hit_768_bs4",
        "prefix_full_hit_769_bs4",
        "prefix_full_hit_513_longout_bs4",
        "prefix_partial_hit_769_bs8",
        "prefix_mixed_hit_miss_bs16",
        "prefix_multi_112req_wave16",
        "prefix_eviction_pressure_96req_wave16",
    ]


def _load_report(run_dir: Path, scenario: str) -> dict[str, Any] | None:
    reports_dir = run_dir / "reports"
    matches = sorted(reports_dir.glob(f"*_{scenario}__dsv4_sm80_a100_victory.json"))
    if not matches:
        matches = sorted(reports_dir.glob(f"*{scenario}__dsv4_sm80_a100_victory.json"))
    if not matches:
        return None
    return _read_json(matches[0])


def _prefix_delta(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    return (
        report.get("metrics", {})
        .get("prefix_cache", {})
        .get("rank0_repeat_delta", {})
    )


def _ttft(report: dict[str, Any] | None) -> float | None:
    if report is None:
        return None
    value = report.get("metrics", {}).get("ttft_s_mean")
    return None if value is None else float(value)


def _workload_rows(milestone_dir: Path, page_size: int) -> list[dict[str, Any]]:
    bench = _load_bench_module()
    scenario_map = bench._scenario_map()
    rows: list[dict[str, Any]] = []
    for name in _scenario_names(milestone_dir):
        scenario = scenario_map[name]
        prompts, params = bench.build_workload(
            scenario,
            vocab_size=4096,
            seed=0,
            token_id_range=1024,
        )
        parts = bench._generation_parts(scenario, prompts, params)
        probe_ids = set()
        if len(parts) > 1:
            offset = len(parts[0][0])
            for idx in range(offset, len(prompts)):
                probe_ids.add(idx)
        elif scenario.kind in {"prefix_multi_sustained", "prefix_eviction_pressure"}:
            # These scenarios have no explicit warm/probe split.  The first wave
            # establishes prefixes and later waves are the serving reuse/pressure phase.
            first_wave = int(scenario.wave_size or scenario.batch_size)
            for idx in range(first_wave, len(prompts)):
                probe_ids.add(idx)

        exact_all = 0
        exact_probe = 0
        phase1_probe_hit_tokens = 0
        route_b_probe_hit_tokens = 0
        shortened_probe_tokens = 0
        prompt_len_hist: dict[str, int] = {}
        exact_len_hist: dict[str, int] = {}
        for idx, prompt in enumerate(prompts):
            prompt_len = len(prompt)
            key = str(prompt_len)
            prompt_len_hist[key] = prompt_len_hist.get(key, 0) + 1
            is_exact = prompt_len % page_size == 0
            if is_exact:
                exact_all += 1
                exact_len_hist[key] = exact_len_hist.get(key, 0) + 1
            if idx in probe_ids:
                phase1 = _phase1_hit_for_identical_prompt(prompt_len, page_size)
                route_b = _route_b_hit_for_identical_prompt(prompt_len, page_size)
                phase1_probe_hit_tokens += phase1
                route_b_probe_hit_tokens += route_b
                shortened_probe_tokens += max(phase1 - route_b, 0)
                if is_exact:
                    exact_probe += 1

        rows.append(
            {
                "scenario": name,
                "kind": scenario.kind,
                "requests": len(prompts),
                "probe_or_reuse_requests": len(probe_ids),
                "exact_multiple_requests": exact_all,
                "exact_multiple_probe_or_reuse_requests": exact_probe,
                "exact_multiple_request_fraction": (
                    0.0 if not prompts else exact_all / len(prompts)
                ),
                "exact_multiple_probe_or_reuse_fraction": (
                    0.0 if not probe_ids else exact_probe / len(probe_ids)
                ),
                "prompt_len_histogram": dict(sorted(prompt_len_hist.items(), key=lambda item: int(item[0]))),
                "exact_multiple_len_histogram": dict(
                    sorted(exact_len_hist.items(), key=lambda item: int(item[0]))
                ),
                "theoretical_phase1_probe_hit_tokens": phase1_probe_hit_tokens,
                "theoretical_route_b_probe_hit_tokens": route_b_probe_hit_tokens,
                "theoretical_shortened_probe_tokens": shortened_probe_tokens,
            }
        )
    return rows


def _actual_impact_rows(milestone_dir: Path) -> list[dict[str, Any]]:
    raw = milestone_dir / "raw"
    phase1_dir = raw / "perf_phase1_prefix_on"
    route_b_dir = raw / "perf_route_b_graph"
    rows: list[dict[str, Any]] = []
    for scenario in _scenario_names(milestone_dir):
        phase1 = _load_report(phase1_dir, scenario)
        route_b = _load_report(route_b_dir, scenario)
        phase1_delta = _prefix_delta(phase1)
        route_b_delta = _prefix_delta(route_b)
        phase1_saved = int(phase1_delta.get("saved_prefill_tokens") or 0)
        route_b_saved = int(route_b_delta.get("saved_prefill_tokens") or 0)
        rows.append(
            {
                "scenario": scenario,
                "phase1_saved_prefill_tokens": phase1_saved,
                "route_b_saved_prefill_tokens": route_b_saved,
                "actual_saved_prefill_token_delta": route_b_saved - phase1_saved,
                "phase1_hit_requests": int(phase1_delta.get("hit_requests") or 0),
                "route_b_hit_requests": int(route_b_delta.get("hit_requests") or 0),
                "phase1_avg_hit_tokens": phase1_delta.get("avg_hit_tokens"),
                "route_b_avg_hit_tokens": route_b_delta.get("avg_hit_tokens"),
                "phase1_ttft_s": _ttft(phase1),
                "route_b_ttft_s": _ttft(route_b),
                "route_b_minus_phase1_ttft_s": (
                    None
                    if _ttft(phase1) is None or _ttft(route_b) is None
                    else float(_ttft(route_b)) - float(_ttft(phase1))
                ),
            }
        )
    return rows


def _safe_hit_table(page_size: int) -> list[dict[str, int]]:
    rows = []
    for prompt_len in (256, 257, 512, 513, 768, 769, 1024, 1025):
        phase1 = _phase1_hit_for_identical_prompt(prompt_len, page_size)
        route_b = _route_b_hit_for_identical_prompt(prompt_len, page_size)
        rows.append(
            {
                "prompt_len": prompt_len,
                "phase1_hit": phase1,
                "route_b_hit": route_b,
                "shortened": max(phase1 - route_b, 0),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarize(milestone_dir: Path, page_size: int) -> dict[str, Any]:
    workload = _workload_rows(milestone_dir, page_size)
    actual = _actual_impact_rows(milestone_dir)
    safe_hit = _safe_hit_table(page_size)
    totals = {
        "requests": sum(row["requests"] for row in workload),
        "probe_or_reuse_requests": sum(row["probe_or_reuse_requests"] for row in workload),
        "exact_multiple_requests": sum(row["exact_multiple_requests"] for row in workload),
        "exact_multiple_probe_or_reuse_requests": sum(
            row["exact_multiple_probe_or_reuse_requests"] for row in workload
        ),
        "theoretical_shortened_probe_tokens": sum(
            row["theoretical_shortened_probe_tokens"] for row in workload
        ),
        "actual_saved_prefill_token_delta": sum(
            int(row["actual_saved_prefill_token_delta"]) for row in actual
        ),
    }
    totals["exact_multiple_request_fraction"] = (
        0.0 if totals["requests"] == 0 else totals["exact_multiple_requests"] / totals["requests"]
    )
    totals["exact_multiple_probe_or_reuse_fraction"] = (
        0.0
        if totals["probe_or_reuse_requests"] == 0
        else totals["exact_multiple_probe_or_reuse_requests"] / totals["probe_or_reuse_requests"]
    )
    return {
        "page_size": page_size,
        "safe_hit_table": safe_hit,
        "workload_frequency": workload,
        "actual_impact": actual,
        "totals": totals,
    }


def write_outputs(milestone_dir: Path, summary: dict[str, Any]) -> None:
    raw = milestone_dir / "raw"
    out = milestone_dir / "summaries"
    _write_json(raw / "swa_tail_guard_quantification.json", summary)
    _write_json(out / "swa_tail_guard_quantification.json", summary)

    safe_keys = ["prompt_len", "phase1_hit", "route_b_hit", "shortened"]
    _write_csv(out / "swa_tail_guard_safe_hit_table.csv", summary["safe_hit_table"], safe_keys)
    (out / "swa_tail_guard_safe_hit_table.md").write_text(
        _md_table(
            ["prompt len", "phase-1 hit", "Route B hit", "shortened"],
            [
                [row["prompt_len"], row["phase1_hit"], row["route_b_hit"], row["shortened"]]
                for row in summary["safe_hit_table"]
            ],
        ),
        encoding="utf-8",
    )

    freq_keys = [
        "scenario",
        "requests",
        "probe_or_reuse_requests",
        "exact_multiple_requests",
        "exact_multiple_probe_or_reuse_requests",
        "exact_multiple_request_fraction",
        "exact_multiple_probe_or_reuse_fraction",
        "theoretical_shortened_probe_tokens",
    ]
    _write_csv(out / "swa_tail_guard_workload_frequency.csv", summary["workload_frequency"], freq_keys)
    (out / "swa_tail_guard_workload_frequency.md").write_text(
        _md_table(
            [
                "scenario",
                "requests",
                "reuse reqs",
                "exact all",
                "exact reuse",
                "exact reuse fraction",
                "theory shortened",
            ],
            [
                [
                    row["scenario"],
                    row["requests"],
                    row["probe_or_reuse_requests"],
                    row["exact_multiple_requests"],
                    row["exact_multiple_probe_or_reuse_requests"],
                    _fmt(row["exact_multiple_probe_or_reuse_fraction"]),
                    row["theoretical_shortened_probe_tokens"],
                ]
                for row in summary["workload_frequency"]
            ],
        ),
        encoding="utf-8",
    )

    impact_keys = [
        "scenario",
        "phase1_saved_prefill_tokens",
        "route_b_saved_prefill_tokens",
        "actual_saved_prefill_token_delta",
        "phase1_hit_requests",
        "route_b_hit_requests",
        "phase1_ttft_s",
        "route_b_ttft_s",
        "route_b_minus_phase1_ttft_s",
    ]
    _write_csv(out / "swa_tail_guard_actual_impact.csv", summary["actual_impact"], impact_keys)
    (out / "swa_tail_guard_actual_impact.md").write_text(
        _md_table(
            [
                "scenario",
                "phase1 saved",
                "Route B saved",
                "delta saved",
                "phase1 hits",
                "Route B hits",
                "TTFT delta s",
            ],
            [
                [
                    row["scenario"],
                    row["phase1_saved_prefill_tokens"],
                    row["route_b_saved_prefill_tokens"],
                    row["actual_saved_prefill_token_delta"],
                    row["phase1_hit_requests"],
                    row["route_b_hit_requests"],
                    _fmt(row["route_b_minus_phase1_ttft_s"]),
                ]
                for row in summary["actual_impact"]
            ],
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milestone-dir",
        default="performance_milestones/target08_route_b_final_prefix_promotion_gate",
    )
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    args = parser.parse_args()
    milestone_dir = Path(args.milestone_dir)
    summary = summarize(milestone_dir, args.page_size)
    write_outputs(milestone_dir, summary)
    print(
        json.dumps(
            {
                "summary": str(milestone_dir / "summaries/swa_tail_guard_quantification.json"),
                "totals": summary["totals"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


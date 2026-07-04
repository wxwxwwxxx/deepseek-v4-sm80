#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "performance_milestones/target08_route_b_metadata_deforest_copy_elision"

PERF_MODES = {
    "prefix_off": "perf_prefix_off",
    "phase1_prefix_on": "perf_phase1_prefix_on",
    "route_b_graph_baseline": "perf_route_b_graph_baseline",
    "route_b_metadata_deforest": "perf_route_b_metadata_deforest",
}
OWNER_PROFILE_MODE = "route_b_metadata_deforest_profile"
OWNER_PROFILE_DIR = "perf_route_b_metadata_deforest_profile"

TEXT_MODES = {
    "prefix_off": "text_smoke_prefix_off.json",
    "phase1_prefix_on": "text_smoke_phase1_prefix_on.json",
    "route_b_graph_baseline": "text_smoke_route_b_graph_baseline.json",
    "route_b_metadata_deforest": "text_smoke_route_b_metadata_deforest.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _safe_div(a: float | int | None, b: float | int | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return float(a) / float(b)


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.splitlines()


def _load_perf_mode(raw_dir: Path, mode: str, dirname: str) -> dict[str, Any]:
    rows = _read_jsonl(raw_dir / dirname / "matrix.jsonl")
    cases = []
    for row in rows:
        report = _read_json(Path(str(row.get("report_path", ""))))
        metrics = report.get("metrics", {})
        phase = metrics.get("phase_totals", {})
        graph = row.get("graph_runner", {})
        cases.append(
            {
                "mode": mode,
                "scenario": row.get("scenario"),
                "status": row.get("status"),
                "ttft_s_mean": row.get("ttft_s_mean"),
                "output_tok_s": row.get("end_to_end_output_tokens_per_s"),
                "decode_prepare_s": phase.get("decode_prepare_s"),
                "prefill_prepare_s": phase.get("prefill_prepare_s"),
                "decode_forward_s": phase.get("decode_forward_s"),
                "prefill_forward_s": phase.get("prefill_forward_s"),
                "saved_prefill_tokens": row.get("prefix_saved_prefill_tokens", 0),
                "hit_rate": row.get("prefix_hit_rate", 0.0),
                "graph_replay": graph.get("replay_count", 0),
                "graph_eager": graph.get("eager_decode_count", 0),
                "captured_buckets": graph.get("captured_bs", []),
                "report_path": row.get("report_path"),
            }
        )
    pass_cases = [case for case in cases if case["status"] == "pass"]
    aggregate = {
        "mode": mode,
        "case_count": len(cases),
        "pass_count": len(pass_cases),
        "mean_ttft_s": _mean([float(c["ttft_s_mean"]) for c in pass_cases if c["ttft_s_mean"] is not None]),
        "mean_output_tok_s": _mean(
            [float(c["output_tok_s"]) for c in pass_cases if c["output_tok_s"] is not None]
        ),
        "decode_prepare_s": sum(float(c.get("decode_prepare_s") or 0.0) for c in pass_cases),
        "saved_prefill_tokens": sum(int(c.get("saved_prefill_tokens") or 0) for c in pass_cases),
        "mean_hit_rate": _mean([float(c.get("hit_rate") or 0.0) for c in pass_cases]),
        "graph_replay": sum(int(c.get("graph_replay") or 0) for c in pass_cases),
        "graph_eager": sum(int(c.get("graph_eager") or 0) for c in pass_cases),
        "captured_buckets": sorted(
            {
                int(bucket)
                for c in pass_cases
                for bucket in (c.get("captured_buckets") or [])
            }
        ),
    }
    return {"aggregate": aggregate, "cases": cases}


def _load_text(raw_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for mode, filename in TEXT_MODES.items():
        payload = _read_json(raw_dir / filename)
        if not payload:
            out[mode] = {"status": "missing", "variant_status": "missing", "texts": []}
            continue
        variant = (payload.get("variants") or [{}])[0]
        graph = payload.get("config", {}).get("graph_runner", {})
        out[mode] = {
            "status": payload.get("status"),
            "variant_status": variant.get("status"),
            "texts": [
                str(item.get("text", ""))
                for item in variant.get("outputs", [])
            ],
            "graph_replay": graph.get("replay_count", 0),
            "graph_eager": graph.get("eager_decode_count", 0),
        }
    return out


def _collect_owner_counters(profile: dict[str, Any]) -> list[dict[str, Any]]:
    counters: list[dict[str, Any]] = []
    for case in profile["cases"]:
        report_path = case.get("report_path")
        if not report_path:
            continue
        report = _read_json(Path(report_path))
        timing = report.get("owner_timing", {})
        rank0 = timing.get("rank0", {})
        for counter in rank0.get("counters", []):
            enriched = dict(counter)
            enriched["scenario"] = case.get("scenario")
            counters.append(enriched)
    return counters


def _counter_table(
    counters: list[dict[str, Any]],
    label: str,
    *,
    by_scenario: bool = False,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    for counter in counters:
        if counter.get("label") != label:
            continue
        metadata = counter.get("metadata", {})
        field = str(metadata.get("field", "unknown"))
        stable = str(metadata.get("stable", "unknown"))
        if by_scenario:
            key = (str(counter.get("scenario", "unknown")), field, stable)
        else:
            key = (field, stable)
        bucket = buckets.setdefault(
            key,
            (
                {"scenario": key[0], "field": key[1], "stable": key[2], "count": 0}
                if by_scenario
                else {"field": key[0], "stable": key[1], "count": 0}
            ),
        )
        bucket["count"] += int(counter.get("count") or 0)
    sort_key = (
        (lambda row: (str(row["scenario"]), -int(row["count"])))
        if by_scenario
        else (lambda row: -int(row["count"]))
    )
    return sorted(buckets.values(), key=sort_key)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_tables(milestone_dir: Path, summary: dict[str, Any]) -> None:
    summaries = milestone_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    serving_rows = list(summary["aggregate_by_mode"].values())
    _write_csv(
        summaries / "serving_ab.csv",
        serving_rows,
        [
            "mode",
            "case_count",
            "pass_count",
            "mean_ttft_s",
            "mean_output_tok_s",
            "decode_prepare_s",
            "saved_prefill_tokens",
            "mean_hit_rate",
            "graph_replay",
            "graph_eager",
        ],
    )
    (summaries / "serving_ab.md").write_text(
        _md_table(
            [
                "mode",
                "pass",
                "mean TTFT s",
                "mean output tok/s",
                "decode prepare s",
                "saved prefill",
                "graph replay/eager",
            ],
            [
                [
                    row["mode"],
                    f"{row['pass_count']}/{row['case_count']}",
                    row["mean_ttft_s"],
                    row["mean_output_tok_s"],
                    row["decode_prepare_s"],
                    row["saved_prefill_tokens"],
                    f"{row['graph_replay']}/{row['graph_eager']}",
                ]
                for row in serving_rows
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    scenarios = sorted(
        {
            case["scenario"]
            for mode in summary["perf"].values()
            for case in mode["cases"]
            if case.get("scenario")
        }
    )
    by_mode_scenario = {
        mode: {case["scenario"]: case for case in payload["cases"]}
        for mode, payload in summary["perf"].items()
    }
    effect_rows = []
    for scenario in scenarios:
        phase1 = by_mode_scenario.get("phase1_prefix_on", {}).get(scenario, {})
        baseline = by_mode_scenario.get("route_b_graph_baseline", {}).get(scenario, {})
        deforest = by_mode_scenario.get("route_b_metadata_deforest", {}).get(scenario, {})
        base_prepare = baseline.get("decode_prepare_s")
        new_prepare = deforest.get("decode_prepare_s")
        effect_rows.append(
            {
                "scenario": scenario,
                "phase1_decode_prepare_s": phase1.get("decode_prepare_s"),
                "route_b_baseline_decode_prepare_s": base_prepare,
                "route_b_deforest_decode_prepare_s": new_prepare,
                "prepare_reduction_vs_route_b": (
                    None
                    if base_prepare in (None, 0)
                    else (float(base_prepare) - float(new_prepare or 0.0)) / float(base_prepare)
                ),
                "deforest_output_tok_s": deforest.get("output_tok_s"),
                "phase1_output_tok_s": phase1.get("output_tok_s"),
                "deforest_output_vs_phase1": _safe_div(
                    deforest.get("output_tok_s"),
                    phase1.get("output_tok_s"),
                ),
            }
        )
    _write_csv(
        summaries / "deforest_effect.csv",
        effect_rows,
        [
            "scenario",
            "phase1_decode_prepare_s",
            "route_b_baseline_decode_prepare_s",
            "route_b_deforest_decode_prepare_s",
            "prepare_reduction_vs_route_b",
            "deforest_output_tok_s",
            "phase1_output_tok_s",
            "deforest_output_vs_phase1",
        ],
    )
    (summaries / "deforest_effect.md").write_text(
        _md_table(
            [
                "scenario",
                "phase1 prepare s",
                "Route B base prepare s",
                "Route B deforest prepare s",
                "reduction",
                "deforest/phase1 output",
            ],
            [
                [
                    row["scenario"],
                    row["phase1_decode_prepare_s"],
                    row["route_b_baseline_decode_prepare_s"],
                    row["route_b_deforest_decode_prepare_s"],
                    row["prepare_reduction_vs_route_b"],
                    row["deforest_output_vs_phase1"],
                ]
                for row in effect_rows
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    for name, rows in (
        ("metadata_build_bytes", summary["metadata_build_bytes"]),
        ("replay_copy_bytes", summary["replay_copy_bytes"]),
        ("metadata_build_calls", summary["metadata_build_calls"]),
        ("replay_copy_calls", summary["replay_copy_calls"]),
    ):
        _write_csv(summaries / f"{name}.csv", rows, ["field", "stable", "count"])
        (summaries / f"{name}.md").write_text(
            _md_table(
                ["field", "stable", "count"],
                [[row["field"], row["stable"], row["count"]] for row in rows[:32]],
            )
            + "\n",
            encoding="utf-8",
        )

    for name, rows in (
        ("metadata_build_bytes_by_scenario", summary["metadata_build_bytes_by_scenario"]),
        ("replay_copy_bytes_by_scenario", summary["replay_copy_bytes_by_scenario"]),
        ("metadata_build_calls_by_scenario", summary["metadata_build_calls_by_scenario"]),
        ("replay_copy_calls_by_scenario", summary["replay_copy_calls_by_scenario"]),
    ):
        _write_csv(summaries / f"{name}.csv", rows, ["scenario", "field", "stable", "count"])
        (summaries / f"{name}.md").write_text(
            _md_table(
                ["scenario", "field", "stable", "count"],
                [
                    [row["scenario"], row["field"], row["stable"], row["count"]]
                    for row in rows[:64]
                ],
            )
            + "\n",
            encoding="utf-8",
        )

    (summaries / "text_smoke.md").write_text(
        _md_table(
            ["mode", "status", "variant", "replay/eager", "outputs"],
            [
                [
                    mode,
                    row.get("status"),
                    row.get("variant_status"),
                    f"{row.get('graph_replay')}/{row.get('graph_eager')}",
                    "<br>".join(row.get("texts", [])),
                ]
                for mode, row in summary["text_smoke"].items()
            ],
        )
        + "\n",
        encoding="utf-8",
    )


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate_by_mode"]
    phase1 = aggregate.get("phase1_prefix_on", {})
    baseline = aggregate.get("route_b_graph_baseline", {})
    deforest = aggregate.get("route_b_metadata_deforest", {})
    graph_ok = int(deforest.get("graph_eager") or 0) == 0 and set(
        deforest.get("captured_buckets") or []
    ) >= {1, 2, 4, 8, 16}
    correctness_ok = Path(
        summary["milestone_dir"],
        "raw/pytest_route_b_metadata_deforest_correctness.log",
    ).exists()
    text = summary["text_smoke"].get("route_b_metadata_deforest", {})
    text_ok = text.get("status") == "pass" and text.get("variant_status") == "pass"
    saved_ratio = _safe_div(
        deforest.get("saved_prefill_tokens"),
        phase1.get("saved_prefill_tokens"),
    )
    prepare_reduction = _safe_div(
        float(baseline.get("decode_prepare_s") or 0.0)
        - float(deforest.get("decode_prepare_s") or 0.0),
        baseline.get("decode_prepare_s"),
    )
    output_vs_phase1 = _safe_div(
        deforest.get("mean_output_tok_s"),
        phase1.get("mean_output_tok_s"),
    )
    success = bool(
        correctness_ok
        and text_ok
        and graph_ok
        and saved_ratio is not None
        and saved_ratio >= 0.90
        and (
            (prepare_reduction is not None and prepare_reduction >= 0.50)
            or (output_vs_phase1 is not None and output_vs_phase1 >= 0.90)
        )
    )
    return {
        "decision": "keep_route_b_metadata_deforest_opt_in" if success else "keep_experimental",
        "correctness_ok": correctness_ok,
        "text_ok": text_ok,
        "graph_ok": graph_ok,
        "saved_prefill_ratio_vs_phase1": saved_ratio,
        "decode_prepare_reduction_vs_route_b": prepare_reduction,
        "output_throughput_vs_phase1": output_vs_phase1,
        "baseline_output_tok_s": baseline.get("mean_output_tok_s"),
        "deforest_output_tok_s": deforest.get("mean_output_tok_s"),
    }


def _build_readme(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    git_status = "\n".join(summary["git_status"]) or "<clean>"
    return "\n".join(
        [
            "# TARGET 08.24 DSV4 Route B Metadata Deforest And Copy Elision",
            "",
            "Date: 2026-07-04",
            "",
            "## Result",
            "",
            f"Decision: **{decision['decision']}**.",
            "",
            "The implementation keeps Route B metadata deforest behind "
            "`MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1`.  The new path consumes "
            "`c4_page_table`, `c128_page_table`, and `c4_indexer_page_table` instead "
            "of deriving component locations with `full_loc // 4` or `full_loc // 128`.",
            "",
            "## Exact Command",
            "",
            "```bash",
            "bash performance_milestones/target08_route_b_metadata_deforest_copy_elision/scripts/run_route_b_metadata_deforest_copy_elision.sh",
            "```",
            "",
            "## Git Status Summary",
            "",
            "```text",
            git_status,
            "```",
            "",
            "## Overhead Attribution",
            "",
            "Route B baseline still builds eager decode metadata and stages it into graph "
            "buffers.  The largest repeated tensors are SWA indices, C4 sparse raw/page/full "
            "indices, C128 raw/page/full indices, and component page tables.  The deforest "
            "opt-in moves C4/C128 index assembly to one component-aware Triton kernel and "
            "generates replay write locs from component page tables.",
            "",
            "The perf decision uses the uninstrumented `perf_route_b_metadata_deforest` "
            "run.  Owner-timing counters come from the separate "
            "`perf_route_b_metadata_deforest_profile` run so profiling overhead does not "
            "pollute throughput comparisons.",
            "",
            "See `summaries/metadata_build_bytes.md` and `summaries/replay_copy_bytes.md` "
            "for field-level byte counters from the owner-timing profile run.  "
            "Per-scenario attribution is in `summaries/*_by_scenario.md`.",
            "",
            "## Field Stability",
            "",
            _md_table(
                ["class", "fields"],
                [
                    ["per-token", "`raw_out_loc`, `positions`, `seq_lens`, SWA indices/lengths, C4 sparse metadata, write locs"],
                    ["per-request/per-hit", "`page_table`, `c4_page_table`, `c128_page_table`, `c4_indexer_page_table`, C128 prefix metadata"],
                    ["per-bucket", "`cu_seqlens_q`, graph capture buffers"],
                ],
            ),
            "",
            "## Component Formula",
            "",
            "`component_loc = component_page_table[row, raw_index // component_page_size] "
            "* component_page_size + raw_index % component_page_size`.  Missing or "
            "tombstoned component pages yield `-1`; full-token tombstones only affect "
            "`*_full_indices`, not component page indices.",
            "",
            "## Serving A/B",
            "",
            "Full table: `summaries/serving_ab.md`.",
            "",
            _md_table(
                [
                    "mode",
                    "mean TTFT s",
                    "mean output tok/s",
                    "decode prepare s",
                    "saved prefill",
                    "graph replay/eager",
                ],
                [
                    [
                        row["mode"],
                        row["mean_ttft_s"],
                        row["mean_output_tok_s"],
                        row["decode_prepare_s"],
                        row["saved_prefill_tokens"],
                        f"{row['graph_replay']}/{row['graph_eager']}",
                    ]
                    for row in summary["aggregate_by_mode"].values()
                ],
            ),
            "",
            "## Deforest Effect",
            "",
            "Full table: `summaries/deforest_effect.md`.",
            "",
            "## Gate Notes",
            "",
            "This gate does **not** promote the Route B metadata deforest path.  "
            "Correctness, text smoke, saved-prefill, and graph replay all pass, but "
            "the performance threshold is not met: aggregate decode prepare increases "
            "versus Route B baseline and output throughput remains below 0.90x phase1.",
            "",
            "The component-aware helper is active and safe, but it mostly changes how "
            "decode metadata is generated.  It does not eliminate the dominant graph "
            "staging copies of `c4_sparse_*`, `swa_page_indices`, and `c128_*` buffers.  "
            "The next useful lever is direct graph-buffer generation or row reuse for "
            "stable request/prefix-hit fields, not SWA-tail ownership work or attention "
            "kernel algorithm tuning.",
            "",
            "## Text Smoke",
            "",
            "Full table: `summaries/text_smoke.md`.",
            "",
            "## Correctness And Safety",
            "",
            _md_table(
                ["check", "result"],
                [
                    ["focused unit/kernel tests", "pass if `raw/pytest_route_b_metadata_deforest_correctness.log` has exit 0"],
                    ["component tombstone fail-safe", "kernel test covers live component pages with tombstoned/missing full pages"],
                    ["graph buckets [1,2,4,8,16]", "see serving A/B graph replay/eager counts"],
                    ["text smoke", "see `summaries/text_smoke.md`"],
                ],
            ),
            "",
            "## Decision Inputs",
            "",
            _md_table(["input", "value"], [[key, value] for key, value in decision.items()]),
            "",
        ]
    )


def summarize(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    perf = {
        mode: _load_perf_mode(raw_dir, mode, dirname)
        for mode, dirname in PERF_MODES.items()
    }
    owner_profile = _load_perf_mode(raw_dir, OWNER_PROFILE_MODE, OWNER_PROFILE_DIR)
    counters = _collect_owner_counters(owner_profile)
    summary: dict[str, Any] = {
        "milestone_dir": str(milestone_dir),
        "git_status": _git_status(),
        "perf": perf,
        "owner_profile": owner_profile,
        "aggregate_by_mode": {
            mode: payload["aggregate"]
            for mode, payload in perf.items()
        },
        "text_smoke": _load_text(raw_dir),
        "metadata_build_bytes": _counter_table(counters, "dsv4.metadata_build.bytes"),
        "replay_copy_bytes": _counter_table(counters, "dsv4.replay_metadata_copy.bytes"),
        "metadata_build_calls": _counter_table(counters, "dsv4.metadata_build.calls"),
        "replay_copy_calls": _counter_table(counters, "dsv4.replay_metadata_copy.calls"),
        "metadata_build_bytes_by_scenario": _counter_table(
            counters,
            "dsv4.metadata_build.bytes",
            by_scenario=True,
        ),
        "replay_copy_bytes_by_scenario": _counter_table(
            counters,
            "dsv4.replay_metadata_copy.bytes",
            by_scenario=True,
        ),
        "metadata_build_calls_by_scenario": _counter_table(
            counters,
            "dsv4.metadata_build.calls",
            by_scenario=True,
        ),
        "replay_copy_calls_by_scenario": _counter_table(
            counters,
            "dsv4.replay_metadata_copy.calls",
            by_scenario=True,
        ),
    }
    summary["decision"] = _decision(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone-dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()
    milestone_dir = Path(args.milestone_dir)
    summary = summarize(milestone_dir)
    _write_tables(milestone_dir, summary)
    summaries = milestone_dir / "summaries"
    (summaries / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (milestone_dir / "README.md").write_text(_build_readme(summary), encoding="utf-8")
    print(json.dumps({"summary": str(summaries / "summary.json"), "decision": summary["decision"]}))


if __name__ == "__main__":
    main()

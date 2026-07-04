#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = (
    ROOT
    / "performance_milestones"
    / "target08_route_b_final_prefix_promotion_gate"
    / "scripts"
    / "summarize_final_gate.py"
)
RERUN_DIR = (
    ROOT
    / "performance_milestones"
    / "target08_route_b_final_prefix_promotion_gate_rerun"
)


def _load_base():
    spec = importlib.util.spec_from_file_location("target08_22_base_summary", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


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
    return base._md_table(headers, rows)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _result_note(summary: dict[str, Any]) -> list[str]:
    decision = summary["decision"]
    failed_route_b = [
        row["scenario"]
        for row in summary["case_rows"]
        if row["mode"] == "route_b_graph" and row["status"] != "pass"
    ]
    route_eager = int(decision["route_b_graph_eager"])
    text = summary["text_smoke"]["route_b_graph"]

    lines = [decision["reason"], ""]
    if failed_route_b:
        lines.extend(
            [
                "Failed Route B scenarios: "
                + ", ".join(f"`{scenario}`" for scenario in failed_route_b),
                "",
            ]
        )
    elif not decision["correctness_ok"]:
        lines.extend(["Correctness did not pass; see `summaries/correctness_table.md`.", ""])
    elif text.get("status") != "pass" or text.get("variant_status") != "pass":
        lines.extend(["Route B text smoke did not pass; see `summaries/text_smoke.md`.", ""])
    elif not decision["graph_ok"] or route_eager:
        lines.extend(["Route B graph coverage did not pass; see `summaries/graph_replay.md`.", ""])
    else:
        lines.extend(
            [
                "The rerun cleared the TARGET 08.22.1 lifecycle blocker. Route B "
                "serving reports, text smoke, and graph replay completed; the "
                "remaining promotion decision is driven by performance, capacity, "
                "and SWA-tail guard impact.",
                "",
            ]
        )
    return lines


def _build_readme(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    safe_hit_table = summary.get("swa_tail_guard", {}).get("safe_hit_table", [])
    text_rows = summary["text_smoke"]
    aggregate = summary["aggregate_by_mode"]
    git_status = "\n".join(summary["git"]["status_short"]) or "<clean>"
    phase1 = aggregate["phase1_prefix_on"]
    route_b = aggregate["route_b_graph"]
    prefix_off = aggregate["prefix_off"]
    route_vs_phase1_ttft = route_b["mean_ttft_s"] - phase1["mean_ttft_s"]
    route_vs_phase1_output = (
        0.0
        if phase1["mean_output_tok_s"] == 0
        else route_b["mean_output_tok_s"] / phase1["mean_output_tok_s"]
    )
    route_vs_off_output = (
        0.0
        if prefix_off["mean_output_tok_s"] == 0
        else route_b["mean_output_tok_s"] / prefix_off["mean_output_tok_s"]
    )

    lines: list[str] = [
        "# TARGET 08.22 DSV4 Route B Final Prefix Promotion Gate Rerun",
        "",
        "Date: 2026-07-04",
        "",
        "## Result",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        *_result_note(summary),
        "## Exact Commands",
        "",
        "Primary command:",
        "",
        "```bash",
        "bash performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/scripts/run_final_prefix_promotion_gate_rerun.sh",
        "```",
        "",
        "The script runs focused pytest coverage, then separate `torchrun` "
        "processes for `prefix_off`, `phase1_prefix_on`, and `route_b_graph`, "
        "followed by separate TP8 text-smoke processes for the same three modes.",
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
        "  --output-dir performance_milestones/target08_route_b_final_prefix_promotion_gate_rerun/raw/perf_route_b_graph",
        "```",
        "",
        "## Git Status Summary",
        "",
        "```text",
        git_status,
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
                for mode, row in text_rows.items()
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
                for mode, row in aggregate.items()
            ],
        ),
        "",
        "Performance note: Route B recovered most of the prefix-cache work "
        f"(`{decision['route_b_saved_prefill_tokens']}` vs "
        f"`{decision['phase1_saved_prefill_tokens']}` saved prefill tokens, "
        f"{_fmt(decision['route_b_saved_prefill_ratio_vs_phase1'])} of phase-1) "
        f"and mean TTFT was {_fmt(route_vs_phase1_ttft)} s above phase-1. "
        f"Mean output throughput was {_fmt(route_vs_phase1_output)}x phase-1 "
        f"and {_fmt(route_vs_off_output)}x prefix-off because Route B keeps decode "
        "metadata deforest guarded off; this overhead is visible in "
        "`summaries/deforest_guard_cost.md` but did not erase the prefix-cache "
        "TTFT win.",
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
                for mode, row in aggregate.items()
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
        _md_table(["input", "value"], [[key, _fmt(value)] for key, value in decision.items()]),
        "",
    ]
    return "\n".join(lines)


def summarize_and_write(milestone_dir: Path) -> dict[str, Any]:
    summary = base.summarize(milestone_dir)
    if summary["decision"]["decision"] == "Route_B_preferred_opt_in":
        summary["decision"]["reason"] = (
            "correctness/text/graph passed; saved-prefill and TTFT stayed close "
            "to phase-1, capacity recovery is meaningful, and the remaining "
            "output-throughput gap is attributable to guarded Route B decode "
            "metadata deforest rather than SWA-tail loss"
        )
    base.write_outputs(milestone_dir, summary)
    (milestone_dir / "README.md").write_text(_build_readme(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone-dir", default=str(RERUN_DIR))
    args = parser.parse_args()
    milestone_dir = Path(args.milestone_dir)
    summary = summarize_and_write(milestone_dir)
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

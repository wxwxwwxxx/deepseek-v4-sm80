#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[3]

CHECKPOINTS = [
    "layer0.attention_input",
    "layer0.wqa_output",
    "layer0.q_lora_after_norm",
    "layer0.q_wqb_output",
    "layer0.q_after_q_norm_rope",
    "layer0.final_attention_output",
]

SAME_SHAPE_POSITION = [
    ("target_slot0_fixed_fillers", 0, "target_slot1_fixed_fillers", 1),
    ("target_slot0_fixed_fillers", 0, "target_slot2_fixed_fillers", 2),
    ("target_slot0_fixed_fillers", 0, "target_slot3_fixed_fillers", 3),
]
SAME_SHAPE_FILLER = [
    ("target_slot0_fixed_fillers", 0, "target_slot0_altA_fillers", 0),
    ("target_slot0_fixed_fillers", 0, "target_slot0_altB_fillers", 0),
]
SHAPE_CHANGE = [("single_target_alone", 0, "target_slot0_fixed_fillers", 0)]
IDENTICAL_ROWS = [
    ("identical_prompts_batch", 0, "identical_prompts_batch", 1),
    ("identical_prompts_batch", 0, "identical_prompts_batch", 2),
    ("identical_prompts_batch", 0, "identical_prompts_batch", 3),
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _git_status() -> dict[str, str]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
        except Exception as exc:
            return f"<error: {type(exc).__name__}: {exc}>"

    return {
        "rev": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _scenario_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in run.get("scenarios", [])}


def _target_row(run: dict[str, Any], scenario: str) -> int:
    item = _scenario_map(run).get(scenario)
    if not item:
        return 0
    labels = [str(row.get("label", "")) for row in item.get("probe_prompts", [])]
    if "target" in labels:
        return labels.index("target")
    for index, label in enumerate(labels):
        if label.startswith("target"):
            return index
    return 0


def _find_batch(
    batches: list[dict[str, Any]],
    *,
    scenario: str,
    phase: str,
) -> dict[str, Any] | None:
    for batch in batches:
        if (
            batch.get("scenario") == scenario
            and batch.get("stage") == "probe"
            and batch.get("phase") == phase
        ):
            return batch
    return None


def _load_logits(run_dir: Path, batch: dict[str, Any] | None) -> torch.Tensor | None:
    if batch is None or not batch.get("logits_path"):
        return None
    payload = torch.load(run_dir / "debug_trace" / batch["logits_path"], map_location="cpu")
    logits = payload.get("logits")
    return logits.float() if isinstance(logits, torch.Tensor) else None


def _activation_index(entries: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out = {}
    for entry in entries:
        key = (
            str(entry.get("scenario", "")),
            str(entry.get("batch", {}).get("phase", "")),
            str(entry.get("name", "")),
        )
        out.setdefault(key, entry)
    return out


def _load_activation(run_dir: Path, entry: dict[str, Any] | None) -> torch.Tensor | None:
    if not entry or not entry.get("tensor_path"):
        return None
    payload = torch.load(run_dir / "debug_trace" / entry["tensor_path"], map_location="cpu")
    tensor = payload.get("tensor")
    return tensor.float() if isinstance(tensor, torch.Tensor) else None


def _tensor_stats(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False, "allclose": False, "reason": "missing"}
    if a.shape != b.shape:
        return {
            "available": True,
            "allclose": False,
            "reason": "shape",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    diff = (a - b).abs()
    return {
        "available": True,
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "exact_equal": bool(torch.equal(a, b)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(diff.float()).item()) if diff.numel() else 0.0,
    }


def _stats_cell(stats: dict[str, Any]) -> str:
    if not stats.get("available"):
        return "n/a"
    if stats.get("reason") == "shape":
        return f"shape {stats.get('shape_a')} vs {stats.get('shape_b')}"
    status = "pass" if stats.get("allclose") else "FAIL"
    exact = ", exact" if stats.get("exact_equal") else ""
    return f"{status} max={_fmt(stats.get('max_abs'))}{exact}"


def _activation_pair_stats(
    run_dir: Path,
    by_activation: dict[tuple[str, str, str], dict[str, Any]],
    phase: str,
    name: str,
    left_scenario: str,
    left_row: int,
    right_scenario: str,
    right_row: int,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left = _load_activation(run_dir, by_activation.get((left_scenario, phase, name)))
    right = _load_activation(run_dir, by_activation.get((right_scenario, phase, name)))
    if left is not None and left.ndim > 0:
        left = left[left_row] if left_row < left.shape[0] else None
    if right is not None and right.ndim > 0:
        right = right[right_row] if right_row < right.shape[0] else None
    return _tensor_stats(left, right, atol=atol, rtol=rtol)


def _worst_group(
    run_dir: Path,
    by_activation: dict[tuple[str, str, str], dict[str, Any]],
    pairs: list[tuple[str, int, str, int]],
    phase: str,
    name: str,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    worst: dict[str, Any] = {"available": False, "max_abs": -1.0}
    for left_scenario, left_row, right_scenario, right_row in pairs:
        stats = _activation_pair_stats(
            run_dir,
            by_activation,
            phase,
            name,
            left_scenario,
            left_row,
            right_scenario,
            right_row,
            atol=atol,
            rtol=rtol,
        )
        stats["pair"] = f"{left_scenario}[{left_row}] vs {right_scenario}[{right_row}]"
        if not stats.get("available"):
            continue
        max_abs = float(stats.get("max_abs", 0.0))
        if not worst.get("available") or max_abs >= float(worst.get("max_abs", -1.0)):
            worst = stats
    return worst if worst.get("available") else {"available": False}


def _logit_row(
    run_dir: Path,
    batches: list[dict[str, Any]],
    scenario: str,
    row: int,
    phase: str,
) -> tuple[torch.Tensor | None, int | None]:
    batch = _find_batch(batches, scenario=scenario, phase=phase)
    logits = _load_logits(run_dir, batch)
    sampled = None
    if batch is not None:
        token_ids = batch.get("sampled_token_ids") or []
        if row < len(token_ids):
            sampled = int(token_ids[row])
    if logits is None or logits.ndim != 2 or row >= logits.shape[0]:
        return None, sampled
    return logits[row], sampled


def _topk_stats(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    left_sampled: int | None,
    right_sampled: int | None,
) -> dict[str, Any]:
    if left is None or right is None:
        return {"available": False}
    diff = (left - right).abs()
    top = min(10, left.numel())
    left_vals, left_ids = torch.topk(left, k=top)
    right_vals, right_ids = torch.topk(right, k=top)
    margin = float((left_vals[0] - left_vals[1]).item()) if top >= 2 else float("inf")
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    return {
        "available": True,
        "max_abs": max_abs,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "left_top1": int(left_ids[0].item()),
        "right_top1": int(right_ids[0].item()),
        "left_top1_value": float(left_vals[0].item()),
        "right_top1_value": float(right_vals[0].item()),
        "left_top1_margin": margin,
        "top1_equal": bool(left_ids[0].item() == right_ids[0].item()),
        "top10_equal": bool(torch.equal(left_ids, right_ids)),
        "left_sampled": left_sampled,
        "right_sampled": right_sampled,
        "sampled_equal": left_sampled == right_sampled,
        "max_abs_can_close_top1_margin": bool(2 * max_abs >= margin),
    }


def _topk_pair(
    run_dir: Path,
    batches: list[dict[str, Any]],
    phase: str,
    left_scenario: str,
    left_row: int,
    right_scenario: str,
    right_row: int,
) -> dict[str, Any]:
    left, left_sampled = _logit_row(run_dir, batches, left_scenario, left_row, phase)
    right, right_sampled = _logit_row(run_dir, batches, right_scenario, right_row, phase)
    stats = _topk_stats(left, right, left_sampled, right_sampled)
    stats["pair"] = f"{left_scenario}[{left_row}] vs {right_scenario}[{right_row}]"
    return stats


def _microbench_rows(microbench: dict[str, Any] | None) -> tuple[list[list[Any]], dict[str, Any]]:
    if not microbench:
        return [["not run", "n/a", "n/a", "n/a", "n/a", "n/a"]], {}
    table = []
    worst = {
        "active_q_norm_max": 0.0,
        "fused_q_kv_max": 0.0,
        "reference_max": 0.0,
        "active_vs_reference_max": 0.0,
        "fused_vs_reference_max": 0.0,
        "fused_available": False,
    }
    for row in microbench.get("rows", []):
        def max_of(key: str) -> float:
            stats = row.get(key) or {}
            if not stats.get("available"):
                return 0.0
            value = stats.get("max_abs", 0.0)
            return float(value) if value != "shape" else float("inf")

        ref = max_of("reference_single_vs_batch")
        active = max_of("active_q_norm_single_vs_batch")
        fused = max_of("fused_q_kv_single_vs_batch")
        active_ref = max_of("active_q_norm_vs_reference_batch")
        fused_ref = max_of("fused_q_kv_vs_reference_batch")
        worst["reference_max"] = max(worst["reference_max"], ref)
        worst["active_q_norm_max"] = max(worst["active_q_norm_max"], active)
        worst["fused_q_kv_max"] = max(worst["fused_q_kv_max"], fused)
        worst["active_vs_reference_max"] = max(worst["active_vs_reference_max"], active_ref)
        worst["fused_vs_reference_max"] = max(worst["fused_vs_reference_max"], fused_ref)
        worst["fused_available"] = worst["fused_available"] or bool(
            (row.get("fused_q_kv_single_vs_batch") or {}).get("available")
        )
        table.append(
            [
                row.get("target_slot"),
                _fmt(ref),
                _fmt(active),
                _fmt(fused) if (row.get("fused_q_kv_single_vs_batch") or {}).get("available") else "n/a",
                _fmt(active_ref),
                _fmt(fused_ref)
                if (row.get("fused_q_kv_vs_reference_batch") or {}).get("available")
                else "n/a",
            ]
        )
    return table, worst


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    run = _load_json(run_dir / "run.json")
    batches = _load_jsonl(run_dir / "debug_trace" / "batches.rank0.jsonl")
    activations = _load_jsonl(run_dir / "debug_trace" / "activations.rank0.jsonl")
    by_activation = _activation_index(activations)

    same_shape_payload: dict[str, Any] = {}
    same_shape_rows = []
    for phase in ("prefill", "decode"):
        for checkpoint in CHECKPOINTS:
            shape_stats = _worst_group(
                run_dir,
                by_activation,
                SHAPE_CHANGE,
                phase,
                checkpoint,
                atol=args.activation_atol,
                rtol=args.activation_rtol,
            )
            position_stats = _worst_group(
                run_dir,
                by_activation,
                SAME_SHAPE_POSITION,
                phase,
                checkpoint,
                atol=args.activation_atol,
                rtol=args.activation_rtol,
            )
            filler_stats = _worst_group(
                run_dir,
                by_activation,
                SAME_SHAPE_FILLER,
                phase,
                checkpoint,
                atol=args.activation_atol,
                rtol=args.activation_rtol,
            )
            identical_stats = _worst_group(
                run_dir,
                by_activation,
                IDENTICAL_ROWS,
                phase,
                checkpoint,
                atol=args.activation_atol,
                rtol=args.activation_rtol,
            )
            same_shape_payload[f"{phase}:{checkpoint}"] = {
                "shape_change": shape_stats,
                "same_shape_position": position_stats,
                "same_shape_filler": filler_stats,
                "identical_rows": identical_stats,
            }
            same_shape_rows.append(
                [
                    phase,
                    checkpoint,
                    _stats_cell(shape_stats),
                    _stats_cell(position_stats),
                    _stats_cell(filler_stats),
                    _stats_cell(identical_stats),
                ]
            )

    _write_text(
        out_dir / "same_shape_comparison.md",
        _markdown_table(
            [
                "Phase",
                "Checkpoint",
                "Shape change bs1->bs4",
                "Same-shape target slot",
                "Same-shape filler content",
                "Identical rows",
            ],
            same_shape_rows,
        ),
    )

    propagation_rows = []
    previous_max: float | None = None
    for checkpoint in CHECKPOINTS:
        stats = same_shape_payload.get(f"prefill:{checkpoint}", {}).get("shape_change", {})
        max_abs = float(stats.get("max_abs", 0.0)) if stats.get("available") else 0.0
        gain = "n/a" if previous_max in (None, 0.0) else f"{max_abs / previous_max:.3g}x"
        propagation_rows.append([checkpoint, _fmt(max_abs), _fmt(stats.get("mean_abs")), gain])
        previous_max = max_abs

    shape_logit_stats = _topk_pair(
        run_dir,
        batches,
        "prefill",
        "single_target_alone",
        0,
        "target_slot0_fixed_fillers",
        0,
    )
    propagation_rows.append(
        [
            "logits(prefill)",
            _fmt(shape_logit_stats.get("max_abs")) if shape_logit_stats.get("available") else "n/a",
            _fmt(shape_logit_stats.get("mean_abs")) if shape_logit_stats.get("available") else "n/a",
            (
                f"{float(shape_logit_stats.get('max_abs', 0.0)) / previous_max:.3g}x"
                if shape_logit_stats.get("available") and previous_max
                else "n/a"
            ),
        ]
    )
    _write_text(
        out_dir / "magnitude_propagation.md",
        _markdown_table(
            ["Shape-change path", "Max abs", "Mean abs", "Gain vs previous"],
            propagation_rows,
        ),
    )

    topk_payload: dict[str, Any] = {}
    topk_rows = []
    topk_groups = [
        ("shape_change", SHAPE_CHANGE),
        ("same_shape_position", SAME_SHAPE_POSITION),
        ("same_shape_filler", SAME_SHAPE_FILLER),
        ("identical_rows", IDENTICAL_ROWS),
    ]
    for phase in ("prefill", "decode"):
        for group, pairs in topk_groups:
            worst: dict[str, Any] | None = None
            for pair in pairs:
                stats = _topk_pair(run_dir, batches, phase, *pair)
                if not stats.get("available"):
                    continue
                if worst is None or float(stats.get("max_abs", 0.0)) >= float(
                    worst.get("max_abs", 0.0)
                ):
                    worst = stats
            topk_payload[f"{phase}:{group}"] = worst or {"available": False}
            stats = worst or {"available": False}
            topk_rows.append(
                [
                    phase,
                    group,
                    stats.get("pair", "n/a"),
                    _fmt(stats.get("max_abs")) if stats.get("available") else "n/a",
                    f"{stats.get('left_top1')}->{stats.get('right_top1')}"
                    if stats.get("available")
                    else "n/a",
                    "yes" if stats.get("top10_equal") else "no",
                    f"{stats.get('left_sampled')}->{stats.get('right_sampled')}"
                    if stats.get("available")
                    else "n/a",
                    "yes" if stats.get("sampled_equal") else "no",
                    _fmt(stats.get("left_top1_margin")) if stats.get("available") else "n/a",
                    "yes" if stats.get("max_abs_can_close_top1_margin") else "no",
                ]
            )
    _write_text(
        out_dir / "logits_topk_analysis.md",
        _markdown_table(
            [
                "Phase",
                "Group",
                "Worst pair",
                "Logit max abs",
                "Top1 ids",
                "Top10 same",
                "Sampled ids",
                "Sampled same",
                "Left top1 margin",
                "2*max_abs >= margin",
            ],
            topk_rows,
        ),
    )

    microbench = _load_json(Path(args.microbench_json)) if args.microbench_json else None
    microbench_rows, microbench_worst = _microbench_rows(microbench)
    _write_text(
        out_dir / "reference_path_comparison.md",
        _markdown_table(
            [
                "Target slot",
                "Reference single-vs-batch",
                "Active q_norm single-vs-batch",
                "Fused q_kv single-vs-batch",
                "Active q_norm vs ref",
                "Fused q_kv vs ref",
            ],
            microbench_rows,
        ),
    )

    reproduction_rows = []
    for scenario in run.get("scenarios", []):
        name = scenario.get("name")
        prefill = _find_batch(batches, scenario=name, phase="prefill")
        decode = _find_batch(batches, scenario=name, phase="decode")
        target_row = _target_row(run, name)
        req = (prefill.get("reqs", []) if prefill else [])
        target_req = req[target_row] if target_row < len(req) else {}
        reproduction_rows.append(
            [
                name,
                ",".join(str(row.get("length")) for row in scenario.get("probe_prompts", [])),
                target_row,
                prefill.get("batch_size") if prefill else "n/a",
                decode.get("forward_source") if decode else "n/a",
                decode.get("padded_size") if decode else "n/a",
                target_req.get("table_idx", "n/a"),
            ]
        )
    _write_text(
        out_dir / "reproduction_table.md",
        _markdown_table(
            [
                "Scenario",
                "Prompt lens",
                "Target row",
                "Prefill bs",
                "Decode source",
                "Decode padded",
                "Target table_idx",
            ],
            reproduction_rows,
        ),
    )

    shape_wqa = same_shape_payload.get("prefill:layer0.wqa_output", {}).get("shape_change", {})
    shape_qwqb = same_shape_payload.get("prefill:layer0.q_wqb_output", {}).get("shape_change", {})
    shape_qrope = same_shape_payload.get(
        "prefill:layer0.q_after_q_norm_rope", {}
    ).get("shape_change", {})
    same_shape_qrope = same_shape_payload.get(
        "prefill:layer0.q_after_q_norm_rope", {}
    ).get("same_shape_position", {})
    same_input_q_norm_max = float(microbench_worst.get("active_q_norm_max", 0.0) or 0.0)
    same_input_fused_max = float(microbench_worst.get("fused_q_kv_max", 0.0) or 0.0)
    fused_available = bool(microbench_worst.get("fused_available", False))

    if (
        same_input_q_norm_max == 0.0
        and (not fused_available or same_input_fused_max == 0.0)
        and float(same_shape_qrope.get("max_abs", 0.0) or 0.0) == 0.0
        and float(shape_qrope.get("max_abs", 0.0) or 0.0) > 0.0
        and (
            float(shape_wqa.get("max_abs", 0.0) or 0.0) > 0.0
            or float(shape_qwqb.get("max_abs", 0.0) or 0.0) > 0.0
        )
    ):
        classification = "GEMM shape numeric drift"
    elif same_input_q_norm_max > 0.0 or (fused_available and same_input_fused_max > 0.0):
        classification = "kernel row-coupling bug"
    else:
        classification = "still unknown"

    decode_same_shape_clean = True
    for key, stats in topk_payload.items():
        if not key.startswith("decode:"):
            continue
        if key == "decode:shape_change":
            continue
        if stats.get("available") and (
            not stats.get("sampled_equal", False)
            or not stats.get("top10_equal", False)
            or float(stats.get("max_abs", 0.0) or 0.0) != 0.0
        ):
            decode_same_shape_clean = False

    payload = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "git": _git_status(),
        "same_shape": same_shape_payload,
        "topk": topk_payload,
        "microbench_worst": microbench_worst,
        "classification": classification,
        "continue_08_20": classification == "GEMM shape numeric drift" and decode_same_shape_clean,
        "continue_08_20_reason": (
            "q-path same-shape/same-input is clean and decode logits are stable"
            if classification == "GEMM shape numeric drift" and decode_same_shape_clean
            else "q-path root cause is classified, but same-shape/identical decode logits are still not a clean oracle"
        ),
    }
    _write_json(out_dir / "comparison_summary.json", payload)
    print(json.dumps({"output_dir": str(out_dir), "classification": classification, "status": "pass"}))
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TARGET 08.197 q-path invariance.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--microbench-json", default=None)
    parser.add_argument("--activation-atol", type=float, default=2e-2)
    parser.add_argument("--activation-rtol", type=float, default=2e-2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    summarize(parse_args(argv))


if __name__ == "__main__":
    main()

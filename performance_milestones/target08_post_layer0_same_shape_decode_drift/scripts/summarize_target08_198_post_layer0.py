#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[3]

Q_PATH_CHECKPOINTS = [
    "layer0.attention_input",
    "layer0.wqa_output",
    "layer0.q_lora_after_norm",
    "layer0.q_wqb_output",
    "layer0.q_after_q_norm_rope",
    "layer0.final_attention_output",
]

PAIR_GROUPS: dict[str, list[tuple[str, int, str, int]]] = {
    "target-slot": [
        ("target_slot0_fixed_fillers", 0, "target_slot1_fixed_fillers", 1),
        ("target_slot0_fixed_fillers", 0, "target_slot2_fixed_fillers", 2),
        ("target_slot0_fixed_fillers", 0, "target_slot3_fixed_fillers", 3),
    ],
    "filler-content": [
        ("target_slot0_fixed_fillers", 0, "target_slot0_altA_fillers", 0),
        ("target_slot0_fixed_fillers", 0, "target_slot0_altB_fillers", 0),
    ],
    "identical-row": [
        ("identical_prompts_batch", 0, "identical_prompts_batch", 1),
        ("identical_prompts_batch", 0, "identical_prompts_batch", 2),
        ("identical_prompts_batch", 0, "identical_prompts_batch", 3),
    ],
}

PHASE_ORDER = ["prefill", "decode0", "decode1"]

SEMANTIC_SKIP_SUBSTRINGS = (
    "page_table",
    "global_page_table",
    "selected_full_indices",
    "selected_page_indices",
    "topk_page_indices",
    "topk_full_indices",
)

LAYER0_Q_PATH_PREFIXES = (
    "layer0.input",
    "layer0.attention_input",
    "layer0.wqa_output",
    "layer0.wkv_shared_activation_output",
    "layer0.q_lora_after_norm",
    "layer0.q_wqb_output",
    "layer0.wkv_output",
    "layer0.q_after_q_norm_rope",
    "layer0.kv_after_kv_norm_rope",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


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


def _phase_label(phase: str, index: int) -> str:
    if phase == "prefill":
        return "prefill"
    return f"decode{index}"


def _label_batches(batches: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    counters: dict[tuple[str, str], int] = {}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for batch in sorted(batches, key=lambda row: int(row.get("batch_index", 0))):
        if batch.get("stage") != "probe":
            continue
        scenario = str(batch.get("scenario", ""))
        phase = str(batch.get("phase", ""))
        key = (scenario, phase)
        index = counters.get(key, 0)
        counters[key] = index + 1
        by_key[(scenario, _phase_label(phase, index))] = batch
    return by_key


def _label_activations(
    activations: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    counters: dict[tuple[str, str, str], int] = {}
    current: dict[tuple[str, str, str], int] = {}
    by_forward: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for entry in activations:
        if entry.get("stage") != "probe":
            continue
        scenario = str(entry.get("scenario", ""))
        stage = str(entry.get("stage", ""))
        batch = entry.get("batch") or {}
        phase = str(batch.get("phase", ""))
        counter_key = (scenario, stage, phase)
        name = str(entry.get("name", ""))
        if name == "embedding" or counter_key not in current:
            index = counters.get(counter_key, 0)
            current[counter_key] = index
            if name == "embedding":
                counters[counter_key] = index + 1
        index = current[counter_key]
        label = _phase_label(phase, index)
        by_forward.setdefault((scenario, label), {})[name] = entry
    return by_forward


def _load_tensor(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    tensor = payload.get("tensor")
    if not isinstance(tensor, torch.Tensor):
        tensor = payload.get("logits")
    if not isinstance(tensor, torch.Tensor):
        return None
    return tensor.detach().float().cpu()


def _load_activation_row(run_dir: Path, entry: dict[str, Any] | None, row: int) -> torch.Tensor | None:
    if not entry or not entry.get("tensor_path"):
        return None
    tensor = _load_tensor(run_dir / "debug_trace" / str(entry["tensor_path"]))
    if tensor is None:
        return None
    if tensor.ndim == 0:
        return tensor
    if row >= tensor.shape[0]:
        return None
    return tensor[row]


def _load_logits_row(
    run_dir: Path,
    batch_by_key: dict[tuple[str, str], dict[str, Any]],
    scenario: str,
    phase: str,
    row: int,
) -> tuple[torch.Tensor | None, int | None]:
    batch = batch_by_key.get((scenario, phase))
    sampled = None
    if batch is not None:
        sampled_ids = batch.get("sampled_token_ids") or []
        if row < len(sampled_ids):
            sampled = int(sampled_ids[row])
    if batch is None or not batch.get("logits_path"):
        return None, sampled
    tensor = _load_tensor(run_dir / "debug_trace" / str(batch["logits_path"]))
    if tensor is None or tensor.ndim != 2 or row >= tensor.shape[0]:
        return None, sampled
    return tensor[row], sampled


def _tensor_stats(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if left is None or right is None:
        return {"available": False, "reason": "missing"}
    if left.shape != right.shape:
        return {
            "available": True,
            "reason": "shape",
            "shape_left": list(left.shape),
            "shape_right": list(right.shape),
            "allclose": False,
            "exact_equal": False,
        }
    diff = (left - right).abs()
    return {
        "available": True,
        "allclose": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        "exact_equal": bool(torch.equal(left, right)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(diff.float()).item()) if diff.numel() else 0.0,
    }


def _stats_cell(stats: dict[str, Any]) -> str:
    if not stats.get("available"):
        return "n/a"
    if stats.get("reason") == "shape":
        return f"shape {stats.get('shape_left')} vs {stats.get('shape_right')}"
    status = "exact" if stats.get("exact_equal") else "diff"
    return f"{status} max={_fmt(stats.get('max_abs'))} mean={_fmt(stats.get('mean_abs'))}"


def _activation_pair_stats(
    run_dir: Path,
    by_forward: dict[tuple[str, str], dict[str, dict[str, Any]]],
    phase: str,
    checkpoint: str,
    pair: tuple[str, int, str, int],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left_scenario, left_row, right_scenario, right_row = pair
    left_entry = by_forward.get((left_scenario, phase), {}).get(checkpoint)
    right_entry = by_forward.get((right_scenario, phase), {}).get(checkpoint)
    left = _load_activation_row(run_dir, left_entry, left_row)
    right = _load_activation_row(run_dir, right_entry, right_row)
    stats = _tensor_stats(left, right, atol=atol, rtol=rtol)
    stats["pair"] = f"{left_scenario}[{left_row}] vs {right_scenario}[{right_row}]"
    return stats


def _worst_group_stats(
    run_dir: Path,
    by_forward: dict[tuple[str, str], dict[str, dict[str, Any]]],
    phase: str,
    checkpoint: str,
    pairs: list[tuple[str, int, str, int]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    worst: dict[str, Any] = {"available": False, "max_abs": -1.0}
    all_pair_stats = []
    for pair in pairs:
        stats = _activation_pair_stats(
            run_dir,
            by_forward,
            phase,
            checkpoint,
            pair,
            atol=atol,
            rtol=rtol,
        )
        all_pair_stats.append(stats)
        if not stats.get("available") or stats.get("reason") == "shape":
            continue
        max_abs = float(stats.get("max_abs", 0.0) or 0.0)
        if not worst.get("available") or max_abs > float(worst.get("max_abs", -1.0)):
            worst = dict(stats)
    if not worst.get("available"):
        worst = {"available": False}
    worst["all_pairs"] = all_pair_stats
    return worst


def _topk_stats(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    left_sampled: int | None,
    right_sampled: int | None,
) -> dict[str, Any]:
    if left is None or right is None or left.shape != right.shape:
        return {"available": False}
    diff = (left - right).abs()
    k = min(10, left.numel())
    left_vals, left_ids = torch.topk(left, k=k)
    right_vals, right_ids = torch.topk(right, k=k)
    margin = float((left_vals[0] - left_vals[1]).item()) if k >= 2 else float("inf")
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
        "top1_changed": bool(left_ids[0].item() != right_ids[0].item()),
        "top10_equal": bool(torch.equal(left_ids, right_ids)),
        "left_sampled": left_sampled,
        "right_sampled": right_sampled,
        "sampled_changed": left_sampled != right_sampled,
        "max_abs_can_close_top1_margin": bool(2 * max_abs >= margin),
    }


def _logit_pair_stats(
    run_dir: Path,
    batch_by_key: dict[tuple[str, str], dict[str, Any]],
    phase: str,
    pair: tuple[str, int, str, int],
) -> dict[str, Any]:
    left_scenario, left_row, right_scenario, right_row = pair
    left, left_sampled = _load_logits_row(run_dir, batch_by_key, left_scenario, phase, left_row)
    right, right_sampled = _load_logits_row(run_dir, batch_by_key, right_scenario, phase, right_row)
    stats = _topk_stats(left, right, left_sampled, right_sampled)
    stats["pair"] = f"{left_scenario}[{left_row}] vs {right_scenario}[{right_row}]"
    return stats


def _semantic_checkpoint(name: str) -> bool:
    return not any(part in name for part in SEMANTIC_SKIP_SUBSTRINGS)


def _post_layer0_q_scan_checkpoint(name: str) -> bool:
    if not _semantic_checkpoint(name):
        return False
    if name == "embedding":
        return False
    if name.startswith(LAYER0_Q_PATH_PREFIXES):
        return False
    return True


def _owner_bucket(name: str) -> str:
    if name == "lm_head_logits":
        return "lm_head logits"
    if name == "final_norm":
        return "HC/final norm"
    if name.startswith("layer0."):
        if ".moe" in name:
            return "layer0 MoE/HC"
        return "layer0 attention merge/WO"
    if name.startswith("layer"):
        if ".moe" in name:
            return "later-layer MoE"
        if (
            ".attention" in name
            or ".indexer" in name
            or ".q_" in name
            or ".wqa" in name
            or ".wkv" in name
            or ".merged_attention" in name
            or ".final_attention" in name
        ):
            return "later-layer attention/indexer"
        return "later-layer HC/residual"
    return "unknown"


def _first_owner_for_group(
    run_dir: Path,
    by_forward: dict[tuple[str, str], dict[str, dict[str, Any]]],
    phase: str,
    ordered_names: list[str],
    group: str,
    *,
    atol: float,
    rtol: float,
    epsilon: float,
) -> dict[str, Any]:
    for checkpoint in ordered_names:
        if not _post_layer0_q_scan_checkpoint(checkpoint):
            continue
        stats = _worst_group_stats(
            run_dir,
            by_forward,
            phase,
            checkpoint,
            PAIR_GROUPS[group],
            atol=atol,
            rtol=rtol,
        )
        if not stats.get("available") or stats.get("reason") == "shape":
            continue
        if float(stats.get("max_abs", 0.0) or 0.0) > epsilon:
            stats["checkpoint"] = checkpoint
            stats["owner"] = _owner_bucket(checkpoint)
            return stats
    return {"available": False, "checkpoint": "n/a", "owner": "n/a"}


def _decode_index(phase: str) -> int | None:
    if not phase.startswith("decode"):
        return None
    try:
        return int(phase[len("decode") :])
    except ValueError:
        return None


def _sampler_feedback_owner(
    run_dir: Path,
    batch_by_key: dict[tuple[str, str], dict[str, Any]],
    phase: str,
    group: str,
) -> dict[str, Any] | None:
    decode_index = _decode_index(phase)
    if decode_index is None or decode_index <= 0:
        return None
    previous_phase = f"decode{decode_index - 1}"
    worst_changed: dict[str, Any] | None = None
    for pair in PAIR_GROUPS[group]:
        stats = _logit_pair_stats(run_dir, batch_by_key, previous_phase, pair)
        if not stats.get("available") or not stats.get("sampled_changed"):
            continue
        if worst_changed is None or float(stats.get("max_abs", 0.0) or 0.0) > float(
            worst_changed.get("max_abs", 0.0) or 0.0
        ):
            worst_changed = stats
    if worst_changed is None:
        return None
    return {
        "available": True,
        "checkpoint": f"{previous_phase}.sampled_token_ids",
        "owner": "sampler feedback",
        "pair": worst_changed.get("pair", "n/a"),
        "max_abs": worst_changed.get("max_abs"),
        "mean_abs": worst_changed.get("mean_abs"),
        "sampled_ids": f"{worst_changed.get('left_sampled')}->{worst_changed.get('right_sampled')}",
        "note": (
            f"{phase} consumes different tokens because {previous_phase} sampled ids changed; "
            "do not attribute that later layer0 drift to the q-path."
        ),
    }


def _ordered_names_for_phase(
    by_forward: dict[tuple[str, str], dict[str, dict[str, Any]]],
    phase: str,
) -> list[str]:
    for scenario in (
        "target_slot0_fixed_fillers",
        "identical_prompts_batch",
        "target_slot1_fixed_fillers",
    ):
        forward = by_forward.get((scenario, phase))
        if forward:
            return list(forward.keys())
    return []


def _available_phases(
    batch_by_key: dict[tuple[str, str], dict[str, Any]],
    by_forward: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> list[str]:
    phases = set()
    for _, phase in batch_by_key:
        phases.add(phase)
    for _, phase in by_forward:
        phases.add(phase)
    ordered = [phase for phase in PHASE_ORDER if phase in phases]
    extra = sorted(phase for phase in phases if phase not in PHASE_ORDER)
    return ordered + extra


def _scenario_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item.get("name", "")): item for item in run.get("scenarios", [])}


def _target_row(scenario: dict[str, Any]) -> int:
    labels = [str(item.get("label", "")) for item in scenario.get("probe_prompts", [])]
    if "target" in labels:
        return labels.index("target")
    for index, label in enumerate(labels):
        if label.startswith("target"):
            return index
    return 0


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    run = _load_json(run_dir / "run.json")
    batches = _load_jsonl(run_dir / "debug_trace" / "batches.rank0.jsonl")
    activations = _load_jsonl(run_dir / "debug_trace" / "activations.rank0.jsonl")
    batch_by_key = _label_batches(batches)
    by_forward = _label_activations(activations)
    phases = _available_phases(batch_by_key, by_forward)

    reproduction_rows = []
    for scenario in run.get("scenarios", []):
        name = str(scenario.get("name", ""))
        prefill = batch_by_key.get((name, "prefill"))
        decode0 = batch_by_key.get((name, "decode0"))
        decode1 = batch_by_key.get((name, "decode1"))
        row = _target_row(scenario)
        reqs = prefill.get("reqs", []) if prefill else []
        target_req = reqs[row] if row < len(reqs) else {}
        reproduction_rows.append(
            [
                name,
                ",".join(str(item.get("length")) for item in scenario.get("probe_prompts", [])),
                row,
                prefill.get("batch_size") if prefill else "n/a",
                decode0.get("forward_source") if decode0 else "n/a",
                decode1.get("forward_source") if decode1 else "n/a",
                decode0.get("padded_size") if decode0 else "n/a",
                target_req.get("table_idx", "n/a"),
            ]
        )
    _write_text(
        out_dir / "same_shape_reproduction.md",
        _markdown_table(
            [
                "Scenario",
                "Prompt lens",
                "Target row",
                "Prefill bs",
                "Decode0 source",
                "Decode1 source",
                "Decode padded",
                "Target table_idx",
            ],
            reproduction_rows,
        ),
    )

    q_path_payload: dict[str, Any] = {}
    q_path_rows = []
    for phase in phases:
        for checkpoint in Q_PATH_CHECKPOINTS:
            row = [phase, checkpoint]
            payload_row = {}
            for group, pairs in PAIR_GROUPS.items():
                stats = _worst_group_stats(
                    run_dir,
                    by_forward,
                    phase,
                    checkpoint,
                    pairs,
                    atol=args.activation_atol,
                    rtol=args.activation_rtol,
                )
                payload_row[group] = stats
                row.append(_stats_cell(stats))
            q_path_payload[f"{phase}:{checkpoint}"] = payload_row
            q_path_rows.append(row)
    _write_text(
        out_dir / "q_path_classification_check.md",
        _markdown_table(
            ["Phase", "Checkpoint", "Target-slot", "Filler-content", "Identical-row"],
            q_path_rows,
        ),
    )

    first_owner_payload: dict[str, Any] = {}
    first_owner_rows = []
    first_owner_checkpoints: set[str] = set()
    for phase in phases:
        ordered_names = _ordered_names_for_phase(by_forward, phase)
        for group in PAIR_GROUPS:
            owner = _sampler_feedback_owner(run_dir, batch_by_key, phase, group)
            if owner is None:
                owner = _first_owner_for_group(
                    run_dir,
                    by_forward,
                    phase,
                    ordered_names,
                    group,
                    atol=args.activation_atol,
                    rtol=args.activation_rtol,
                    epsilon=args.first_owner_epsilon,
                )
            first_owner_payload[f"{phase}:{group}"] = owner
            if owner.get("checkpoint") not in (None, "n/a"):
                first_owner_checkpoints.add(str(owner["checkpoint"]))
            first_owner_rows.append(
                [
                    phase,
                    group,
                    owner.get("owner", "n/a"),
                    owner.get("checkpoint", "n/a"),
                    owner.get("pair", "n/a"),
                    _fmt(owner.get("max_abs")) if owner.get("available") else "n/a",
                    _fmt(owner.get("mean_abs")) if owner.get("available") else "n/a",
                ]
            )
    _write_text(
        out_dir / "first_owner_table.md",
        _markdown_table(
            ["Phase", "Group", "Owner bucket", "First checkpoint", "Worst pair", "Max abs", "Mean abs"],
            first_owner_rows,
        ),
    )

    selected_checkpoints = []
    for name in Q_PATH_CHECKPOINTS:
        if name not in selected_checkpoints:
            selected_checkpoints.append(name)
    for name in sorted(first_owner_checkpoints):
        if name not in selected_checkpoints:
            selected_checkpoints.append(name)
    for name in ("final_norm", "lm_head_logits"):
        if name not in selected_checkpoints:
            selected_checkpoints.append(name)

    checkpoint_payload: dict[str, Any] = {}
    checkpoint_rows = []
    for phase in phases:
        for checkpoint in selected_checkpoints:
            for group, pairs in PAIR_GROUPS.items():
                stats = _worst_group_stats(
                    run_dir,
                    by_forward,
                    phase,
                    checkpoint,
                    pairs,
                    atol=args.activation_atol,
                    rtol=args.activation_rtol,
                )
                checkpoint_payload[f"{phase}:{checkpoint}:{group}"] = stats
                checkpoint_rows.append(
                    [
                        phase,
                        group,
                        checkpoint,
                        stats.get("pair", "n/a"),
                        _fmt(stats.get("max_abs")) if stats.get("available") else "n/a",
                        _fmt(stats.get("mean_abs")) if stats.get("available") else "n/a",
                        "yes" if stats.get("exact_equal") else "no",
                    ]
                )
    _write_text(
        out_dir / "checkpoint_diff_selected.md",
        _markdown_table(
            ["Phase", "Group", "Checkpoint", "Worst pair", "Max abs", "Mean abs", "Exact"],
            checkpoint_rows,
        ),
    )

    topk_payload: dict[str, Any] = {}
    topk_rows = []
    for phase in phases:
        for group, pairs in PAIR_GROUPS.items():
            for pair in pairs:
                stats = _logit_pair_stats(run_dir, batch_by_key, phase, pair)
                key = f"{phase}:{group}:{stats.get('pair')}"
                topk_payload[key] = stats
                topk_rows.append(
                    [
                        phase,
                        group,
                        stats.get("pair", "n/a"),
                        _fmt(stats.get("max_abs")) if stats.get("available") else "n/a",
                        _fmt(stats.get("mean_abs")) if stats.get("available") else "n/a",
                        f"{stats.get('left_top1')}->{stats.get('right_top1')}"
                        if stats.get("available")
                        else "n/a",
                        "yes" if stats.get("top1_changed") else "no",
                        f"{stats.get('left_sampled')}->{stats.get('right_sampled')}"
                        if stats.get("available")
                        else "n/a",
                        "yes" if stats.get("sampled_changed") else "no",
                        _fmt(stats.get("left_top1_margin")) if stats.get("available") else "n/a",
                        "yes" if stats.get("max_abs_can_close_top1_margin") else "no",
                    ]
                )
    _write_text(
        out_dir / "topk_margin_sampled_analysis.md",
        _markdown_table(
            [
                "Phase",
                "Group",
                "Pair",
                "Logit max abs",
                "Logit mean abs",
                "Top1 ids",
                "Top1 changed",
                "Sampled ids",
                "Sampled changed",
                "Left top1 margin",
                "2*max_abs >= margin",
            ],
            topk_rows,
        ),
    )

    clean_layer0_q_path = True
    for key, group_payload in q_path_payload.items():
        phase = key.split(":", 1)[0]
        if phase not in ("prefill", "decode0"):
            continue
        if not key.endswith(":layer0.q_after_q_norm_rope") and not key.endswith(
            ":layer0.final_attention_output"
        ):
            continue
        for group_stats in group_payload.values():
            if group_stats.get("available") and float(group_stats.get("max_abs", 0.0) or 0.0) != 0.0:
                clean_layer0_q_path = False

    decode1_q_path_sampler_feedback = False
    for key, group_payload in q_path_payload.items():
        if not key.startswith("decode1:"):
            continue
        if not key.endswith(":layer0.q_after_q_norm_rope") and not key.endswith(
            ":layer0.final_attention_output"
        ):
            continue
        for group_name, group_stats in group_payload.items():
            if not group_stats.get("available") or float(group_stats.get("max_abs", 0.0) or 0.0) == 0.0:
                continue
            if _sampler_feedback_owner(run_dir, batch_by_key, "decode1", group_name) is not None:
                decode1_q_path_sampler_feedback = True

    payload = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "git": _git_status(),
        "config": run.get("config", {}),
        "phases": phases,
        "q_path_classification_still_holds": clean_layer0_q_path,
        "decode1_q_path_diff_is_sampler_feedback": decode1_q_path_sampler_feedback,
        "q_path": q_path_payload,
        "first_owner": first_owner_payload,
        "checkpoint_diff_selected": checkpoint_payload,
        "topk": topk_payload,
    }
    _write_json(out_dir / "comparison_summary.json", payload)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "q_path_classification_still_holds": clean_layer0_q_path,
                "status": "pass",
            }
        )
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TARGET 08.198 same-shape drift.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activation-atol", type=float, default=0.0)
    parser.add_argument("--activation-rtol", type=float, default=0.0)
    parser.add_argument("--first-owner-epsilon", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    summarize(parse_args(argv))


if __name__ == "__main__":
    main()

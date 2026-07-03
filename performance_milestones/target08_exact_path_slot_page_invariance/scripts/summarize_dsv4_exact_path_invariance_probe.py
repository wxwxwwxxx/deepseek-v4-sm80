#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_run(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "run.json")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_batches(run_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(run_dir / "debug_trace" / "batches.rank0.jsonl")


def _load_activations(run_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(run_dir / "debug_trace" / "activations.rank0.jsonl")


def _scenario_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in run.get("scenarios", [])}


def _prompt_labels(scenario: dict[str, Any]) -> list[str]:
    return [str(item["label"]) for item in scenario.get("probe_prompts", [])]


def _label_index(scenario: dict[str, Any], label: str) -> int:
    labels = _prompt_labels(scenario)
    if label in labels:
        return labels.index(label)
    for idx, item in enumerate(labels):
        if item.startswith(label):
            return idx
    raise KeyError(f"label {label!r} not found in scenario {scenario.get('name')}")


def _find_batch(
    batches: list[dict[str, Any]],
    *,
    scenario: str,
    phase: str,
    stage: str = "probe",
) -> dict[str, Any] | None:
    for batch in batches:
        if (
            batch.get("scenario") == scenario
            and batch.get("stage") == stage
            and batch.get("phase") == phase
        ):
            return batch
    return None


def _load_logits(run_dir: Path, batch: dict[str, Any] | None) -> torch.Tensor | None:
    if batch is None or not batch.get("logits_path"):
        return None
    payload = torch.load(run_dir / "debug_trace" / batch["logits_path"], map_location="cpu")
    return payload["logits"].float()


def _load_metadata(run_dir: Path, batch: dict[str, Any] | None) -> dict[str, torch.Tensor]:
    if batch is None or not batch.get("metadata_path"):
        return {}
    return torch.load(run_dir / "debug_trace" / batch["metadata_path"], map_location="cpu")


def _load_activation_tensor(run_dir: Path, entry: dict[str, Any]) -> torch.Tensor | None:
    path = entry.get("tensor_path")
    if not path:
        return None
    payload = torch.load(run_dir / "debug_trace" / path, map_location="cpu")
    tensor = payload.get("tensor")
    return tensor.float() if isinstance(tensor, torch.Tensor) else None


def _logits_stats(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False, "allclose": False, "reason": "missing logits"}
    if a.shape != b.shape:
        return {
            "available": True,
            "allclose": False,
            "reason": "shape mismatch",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    diff = (a - b).abs()
    topk = min(10, a.shape[-1]) if a.ndim > 0 else 0
    topk_equal = True
    argmax_equal = True
    if a.ndim == 2 and topk > 0:
        topk_equal = bool(torch.equal(torch.topk(a, topk, dim=-1).indices, torch.topk(b, topk, dim=-1).indices))
        argmax_equal = bool(torch.equal(torch.argmax(a, dim=-1), torch.argmax(b, dim=-1)))
    return {
        "available": True,
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "row_max_abs": [float(x) for x in diff.reshape(diff.shape[0], -1).max(dim=-1).values.tolist()] if diff.ndim >= 2 else [],
        "argmax_equal": argmax_equal,
        "top10_ids_equal": topk_equal,
        "shape": list(a.shape),
    }


def _row_stats(
    logits: torch.Tensor | None,
    row_a: int,
    row_b: int,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if logits is None:
        return {"available": False, "allclose": False, "reason": "missing logits"}
    if logits.ndim != 2 or max(row_a, row_b) >= logits.shape[0]:
        return {"available": False, "allclose": False, "reason": "row unavailable"}
    return _logits_stats(logits[row_a : row_a + 1], logits[row_b : row_b + 1], atol=atol, rtol=rtol)


def _target_row_stats(
    base_logits: torch.Tensor | None,
    other_logits: torch.Tensor | None,
    other_row: int,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if base_logits is None or other_logits is None:
        return {"available": False, "allclose": False, "reason": "missing logits"}
    if base_logits.ndim != 2 or other_logits.ndim != 2 or other_row >= other_logits.shape[0]:
        return {"available": False, "allclose": False, "reason": "row unavailable"}
    return _logits_stats(base_logits[0:1], other_logits[other_row : other_row + 1], atol=atol, rtol=rtol)


def _fmt_stats(stats: dict[str, Any]) -> str:
    if not stats.get("available"):
        return "n/a"
    status = "pass" if stats.get("allclose") else "FAIL"
    return f"{status} max={stats.get('max_abs', 0.0):.6g}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def _req_summary(
    run_dir: Path,
    batch: dict[str, Any] | None,
    row: int,
) -> dict[str, Any]:
    if batch is None:
        return {}
    reqs = batch.get("reqs", [])
    req = reqs[row] if row < len(reqs) else {}
    meta = _load_metadata(run_dir, batch)
    page_rows = meta.get("batch.global_page_table_rows")
    first_locs: list[int] = []
    if isinstance(page_rows, torch.Tensor) and row < page_rows.shape[0]:
        valid = page_rows[row][page_rows[row] >= 0]
        first_locs = [int(x) for x in valid[: min(8, valid.numel())].tolist()]
    return {
        "uid": req.get("uid"),
        "table_idx": req.get("table_idx"),
        "cached_len": req.get("cached_len"),
        "device_len": req.get("device_len"),
        "extend_len": req.get("extend_len"),
        "first_physical_locs": first_locs,
    }


def _reproduction_rows(
    run_dir: Path,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
) -> list[list[Any]]:
    rows = []
    scenarios = _scenario_map(run)
    for name, scenario in scenarios.items():
        prefill = _find_batch(batches, scenario=name, phase="prefill")
        decode = _find_batch(batches, scenario=name, phase="decode")
        labels = _prompt_labels(scenario)
        target_row = 0
        if any(label == "target" for label in labels):
            target_row = _label_index(scenario, "target")
        req = _req_summary(run_dir, prefill, target_row)
        rows.append(
            [
                name,
                ",".join(str(item["length"]) for item in scenario.get("probe_prompts", [])),
                ",".join(labels),
                prefill.get("batch_size") if prefill else "n/a",
                decode.get("forward_source") if decode else "n/a",
                decode.get("padded_size") if decode else "n/a",
                req.get("table_idx", "n/a"),
                req.get("first_physical_locs", [])[:4],
            ]
        )
    return rows


def _identical_logits(
    run_dir: Path,
    batches: list[dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        batch = _find_batch(batches, scenario="identical_prompts_batch", phase=phase)
        logits = _load_logits(run_dir, batch)
        row_stats = {}
        for row in (1, 2, 3):
            row_stats[f"row0_vs_row{row}"] = _row_stats(logits, 0, row, atol=atol, rtol=rtol)
        out[phase] = row_stats
    return out


def _single_vs_slot_logits(
    run_dir: Path,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    scenarios = _scenario_map(run)
    out: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        base_batch = _find_batch(batches, scenario="single_target_alone", phase=phase)
        base_logits = _load_logits(run_dir, base_batch)
        phase_out = {}
        for slot in range(4):
            name = f"target_in_batch_slot{slot}"
            scenario = scenarios.get(name)
            batch = _find_batch(batches, scenario=name, phase=phase)
            logits = _load_logits(run_dir, batch)
            row = _label_index(scenario, "target") if scenario else slot
            phase_out[name] = _target_row_stats(base_logits, logits, row, atol=atol, rtol=rtol)
        out[phase] = phase_out
    return out


def _single_group_logits(
    run_dir: Path,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    names: list[str],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    scenarios = _scenario_map(run)
    out: dict[str, Any] = {}
    base_name = names[0]
    for phase in ("prefill", "decode"):
        base_batch = _find_batch(batches, scenario=base_name, phase=phase)
        base_logits = _load_logits(run_dir, base_batch)
        phase_out = {}
        for name in names[1:]:
            scenario = scenarios.get(name)
            batch = _find_batch(batches, scenario=name, phase=phase)
            logits = _load_logits(run_dir, batch)
            row = _label_index(scenario, "target") if scenario else 0
            phase_out[name] = _target_row_stats(base_logits, logits, row, atol=atol, rtol=rtol)
        out[phase] = phase_out
    return out


def _graph_vs_eager_logits(
    eager_dir: Path,
    graph_dir: Path | None,
    eager_run: dict[str, Any],
    eager_batches: list[dict[str, Any]],
    graph_batches: list[dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if graph_dir is None:
        return {}
    out = {}
    for scenario in _scenario_map(eager_run):
        scenario_out = {}
        for phase in ("prefill", "decode"):
            eager_batch = _find_batch(eager_batches, scenario=scenario, phase=phase)
            graph_batch = _find_batch(graph_batches, scenario=scenario, phase=phase)
            stats = _logits_stats(
                _load_logits(eager_dir, eager_batch),
                _load_logits(graph_dir, graph_batch),
                atol=atol,
                rtol=rtol,
            )
            stats["eager_forward_source"] = eager_batch.get("forward_source") if eager_batch else None
            stats["graph_forward_source"] = graph_batch.get("forward_source") if graph_batch else None
            stats["graph_padded_size"] = graph_batch.get("padded_size") if graph_batch else None
            stats["graph_batch_size"] = graph_batch.get("batch_size") if graph_batch else None
            scenario_out[phase] = stats
        out[scenario] = scenario_out
    return out


def _activation_entries_by_key(entries: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (
            str(entry.get("scenario", "")),
            str(entry.get("stage", "")),
            str(entry.get("batch", {}).get("phase", "")),
            str(entry.get("name", "")),
        )
        out.setdefault(key, []).append(entry)
    return out


def _tensor_row_diff(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False, "allclose": False, "reason": "missing tensor"}
    if a.shape != b.shape:
        return {
            "available": True,
            "allclose": False,
            "reason": "shape mismatch",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    diff = (a - b).abs()
    return {
        "available": True,
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "shape": list(a.shape),
    }


def _first_identical_activation_diff(
    run_dir: Path,
    entries: list[dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    candidates = [
        entry
        for entry in entries
        if entry.get("scenario") == "identical_prompts_batch" and entry.get("stage") == "probe"
    ]
    candidates.sort(key=lambda item: int(item.get("activation_index", 0)))
    for entry in candidates:
        tensor = _load_activation_tensor(run_dir, entry)
        if tensor is None or tensor.ndim < 1 or tensor.shape[0] < 2:
            continue
        worst = {"available": False, "allclose": True, "max_abs": 0.0}
        for row in range(1, min(4, tensor.shape[0])):
            stats = _tensor_row_diff(tensor[0], tensor[row], atol=atol, rtol=rtol)
            if stats.get("available") and stats.get("max_abs", 0.0) >= worst.get("max_abs", 0.0):
                worst = stats
                worst["row_pair"] = f"row0-row{row}"
        if worst.get("available") and not worst.get("allclose"):
            return {
                "scenario": "identical_prompts_batch",
                "phase": entry.get("batch", {}).get("phase"),
                "name": entry.get("name"),
                "activation_index": entry.get("activation_index"),
                "stats": worst,
            }
    return {"scenario": "identical_prompts_batch", "name": "none", "stats": {"allclose": True}}


def _first_slot_activation_diff(
    run_dir: Path,
    run: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    scenarios = _scenario_map(run)
    by_key = _activation_entries_by_key(entries)
    base_entries = [
        entry
        for entry in entries
        if entry.get("scenario") == "single_target_alone" and entry.get("stage") == "probe"
    ]
    base_entries.sort(key=lambda item: int(item.get("activation_index", 0)))
    for base in base_entries:
        phase = str(base.get("batch", {}).get("phase", ""))
        name = str(base.get("name", ""))
        base_tensor = _load_activation_tensor(run_dir, base)
        if base_tensor is None or base_tensor.shape[0] < 1:
            continue
        base_row = base_tensor[0]
        for slot in range(4):
            scenario_name = f"target_in_batch_slot{slot}"
            scenario = scenarios.get(scenario_name)
            if scenario is None:
                continue
            target_row = _label_index(scenario, "target")
            matches = by_key.get((scenario_name, "probe", phase, name), [])
            if not matches:
                continue
            tensor = _load_activation_tensor(run_dir, matches[0])
            if tensor is None or tensor.shape[0] <= target_row:
                continue
            stats = _tensor_row_diff(base_row, tensor[target_row], atol=atol, rtol=rtol)
            if stats.get("available") and not stats.get("allclose"):
                return {
                    "scenario": scenario_name,
                    "phase": phase,
                    "name": name,
                    "activation_index": matches[0].get("activation_index"),
                    "target_row": target_row,
                    "stats": stats,
                }
    return {"scenario": "single_vs_slots", "name": "none", "stats": {"allclose": True}}


def _bool_word(value: bool) -> str:
    return "yes" if value else "no"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    eager_dir = Path(args.eager) if args.eager else None
    graph_dir = Path(args.graph) if args.graph else None
    if eager_dir is None and graph_dir is None:
        raise SystemExit("At least one of --eager or --graph is required.")
    primary_dir = eager_dir or graph_dir
    assert primary_dir is not None

    eager_run = _load_run(eager_dir) if eager_dir else None
    graph_run = _load_run(graph_dir) if graph_dir else None
    primary_run = eager_run or graph_run
    assert primary_run is not None
    eager_batches = _load_batches(eager_dir) if eager_dir else []
    graph_batches = _load_batches(graph_dir) if graph_dir else []
    primary_batches = eager_batches or graph_batches

    identical = _identical_logits(primary_dir, primary_batches, atol=args.atol, rtol=args.rtol)
    slots = _single_vs_slot_logits(primary_dir, primary_run, primary_batches, atol=args.atol, rtol=args.rtol)
    table_rows = _single_group_logits(
        primary_dir,
        primary_run,
        primary_batches,
        [
            "target_table_row_after_0_dummy",
            "target_table_row_after_2_dummy",
            "target_table_row_after_3_dummy",
        ],
        atol=args.atol,
        rtol=args.rtol,
    )
    page_rows = _single_group_logits(
        primary_dir,
        primary_run,
        primary_batches,
        [
            "target_physical_page_none",
            "target_physical_page_one_page",
            "target_physical_page_mixed_pages",
        ],
        atol=args.atol,
        rtol=args.rtol,
    )
    graph_vs_eager = (
        _graph_vs_eager_logits(
            eager_dir,
            graph_dir,
            eager_run,
            eager_batches,
            graph_batches,
            atol=args.atol,
            rtol=args.rtol,
        )
        if eager_dir and graph_dir and eager_run is not None
        else {}
    )

    activation_entries = _load_activations(primary_dir)
    activation_summary = {
        "identical_first_diff": _first_identical_activation_diff(
            primary_dir, activation_entries, atol=args.activation_atol, rtol=args.activation_rtol
        ),
        "slot_first_diff": _first_slot_activation_diff(
            primary_dir,
            primary_run,
            activation_entries,
            atol=args.activation_atol,
            rtol=args.activation_rtol,
        ),
    }

    reproduction = _reproduction_rows(primary_dir, primary_run, primary_batches)
    payload = {
        "primary_run_dir": str(primary_dir),
        "eager_run_dir": str(eager_dir) if eager_dir else None,
        "graph_run_dir": str(graph_dir) if graph_dir else None,
        "atol": args.atol,
        "rtol": args.rtol,
        "identical_logits": identical,
        "single_vs_slot_logits": slots,
        "table_row_logits": table_rows,
        "physical_page_logits": page_rows,
        "graph_vs_eager_logits": graph_vs_eager,
        "activation_summary": activation_summary,
    }

    out_dir = Path(args.output_dir)
    _write_json(out_dir / "comparison_summary.json", payload)
    _write_text(
        out_dir / "reproduction_table.md",
        _markdown_table(
            [
                "Scenario",
                "Prompt lens",
                "Labels",
                "Prefill bs",
                "Decode source",
                "Decode padded",
                "Target table_idx",
                "Target first physical locs",
            ],
            reproduction,
        ),
    )

    logits_rows = []
    for phase, row_stats in identical.items():
        logits_rows.append(
            [
                f"identical {phase}",
                _fmt_stats(row_stats.get("row0_vs_row1", {})),
                _fmt_stats(row_stats.get("row0_vs_row2", {})),
                _fmt_stats(row_stats.get("row0_vs_row3", {})),
            ]
        )
    for phase, mapping in slots.items():
        for name, stats in mapping.items():
            logits_rows.append([f"alone vs {name} {phase}", _fmt_stats(stats), "", ""])
    for phase, mapping in table_rows.items():
        for name, stats in mapping.items():
            logits_rows.append([f"table {name} {phase}", _fmt_stats(stats), "", ""])
    for phase, mapping in page_rows.items():
        for name, stats in mapping.items():
            logits_rows.append([f"page {name} {phase}", _fmt_stats(stats), "", ""])
    _write_text(
        out_dir / "logits_comparison.md",
        _markdown_table(["Comparison", "Primary", "Secondary", "Tertiary"], logits_rows),
    )

    graph_rows = []
    for scenario, phases in graph_vs_eager.items():
        decode = phases.get("decode", {})
        prefill = phases.get("prefill", {})
        graph_rows.append(
            [
                scenario,
                _fmt_stats(prefill),
                _fmt_stats(decode),
                decode.get("graph_batch_size", "n/a"),
                decode.get("graph_padded_size", "n/a"),
                decode.get("graph_forward_source", "n/a"),
            ]
        )
    _write_text(
        out_dir / "graph_bucket_analysis.md",
        _markdown_table(
            ["Scenario", "Prefill eager/graph", "Decode eager/graph", "Real bs", "Padded bs", "Graph source"],
            graph_rows or [["not run", "n/a", "n/a", "n/a", "n/a", "n/a"]],
        ),
    )

    first_rows = []
    for label, item in (
        ("identical rows", activation_summary["identical_first_diff"]),
        ("single vs slots", activation_summary["slot_first_diff"]),
    ):
        stats = item.get("stats", {})
        first_rows.append(
            [
                label,
                item.get("scenario", ""),
                item.get("phase", ""),
                item.get("name", "none"),
                "pass" if stats.get("allclose") else f"FAIL max={stats.get('max_abs', 0.0):.6g}",
            ]
        )
    _write_text(
        out_dir / "first_divergent_layer.md",
        _markdown_table(["Lens", "Scenario", "Phase", "First checkpoint", "Result"], first_rows),
    )

    metadata_rows = []
    for scenario_name in (
        "identical_prompts_batch",
        "single_target_alone",
        "target_in_batch_slot0",
        "target_in_batch_slot1",
        "target_in_batch_slot2",
        "target_in_batch_slot3",
        "target_physical_page_none",
        "target_physical_page_one_page",
        "target_physical_page_mixed_pages",
        "swa_boundary_127_128_129_bs3",
    ):
        batch = _find_batch(primary_batches, scenario=scenario_name, phase="decode")
        if batch is None:
            continue
        metadata_rows.append(
            [
                scenario_name,
                batch.get("batch_size"),
                batch.get("padded_size"),
                batch.get("forward_source"),
                [req.get("table_idx") for req in batch.get("reqs", [])],
                [req.get("device_len") for req in batch.get("reqs", [])],
            ]
        )
    _write_text(
        out_dir / "metadata_comparison.md",
        _markdown_table(
            ["Scenario", "Decode bs", "Padded", "Source", "table_idx rows", "device_lens"],
            metadata_rows,
        ),
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TARGET 08.195 invariance probe.")
    parser.add_argument("--eager", default=None, help="Run directory for eager probe.")
    parser.add_argument("--graph", default=None, help="Run directory for graph probe.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--activation-atol", type=float, default=2e-2)
    parser.add_argument("--activation-rtol", type=float, default=2e-2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    payload = summarize(parse_args(argv))
    print(json.dumps({"summary": payload["primary_run_dir"], "status": "pass"}))


if __name__ == "__main__":
    main()

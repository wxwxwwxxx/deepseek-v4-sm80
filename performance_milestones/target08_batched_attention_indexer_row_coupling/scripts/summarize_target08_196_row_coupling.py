#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[3]
EXACT_SUMMARY = (
    ROOT
    / "performance_milestones"
    / "target08_exact_path_slot_page_invariance"
    / "scripts"
    / "summarize_dsv4_exact_path_invariance_probe.py"
)


def _load_exact_summary_module():
    spec = importlib.util.spec_from_file_location("target08_exact_summary", EXACT_SUMMARY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {EXACT_SUMMARY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _load_activation_tensor(run_dir: Path, entry: dict[str, Any]) -> torch.Tensor | None:
    path = entry.get("tensor_path")
    if not path:
        return None
    payload = torch.load(run_dir / "debug_trace" / path, map_location="cpu")
    tensor = payload.get("tensor")
    return tensor.float() if isinstance(tensor, torch.Tensor) else None


def _tensor_diff(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "allclose": False,
            "max_abs": "shape",
            "shape": f"{list(a.shape)} vs {list(b.shape)}",
        }
    diff = (a - b).abs()
    return {
        "allclose": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "shape": list(a.shape),
    }


def _fmt(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "n/a"
    if stats.get("max_abs") == "shape":
        return f"FAIL shape {stats.get('shape')}"
    status = "pass" if stats.get("allclose") else "FAIL"
    return f"{status} max={stats.get('max_abs', 0.0):.6g}"


def _interesting_checkpoint(name: str) -> bool:
    parts = (
        "attention",
        "indexer",
        "compressor",
        "wqa",
        "wkv",
        "q_",
        "kv_",
        "rope",
        "topk",
        "swa",
        "c4",
        "c128",
        "merged",
    )
    return any(part in name for part in parts)


def _activation_entries_by_key(entries: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (
            str(entry.get("scenario", "")),
            str(entry.get("batch", {}).get("phase", "")),
            str(entry.get("name", "")),
        )
        out.setdefault(key, entry)
    return out


def _checkpoint_rows(run_dir: Path, *, atol: float, rtol: float) -> list[list[Any]]:
    entries = _load_jsonl(run_dir / "debug_trace" / "activations.rank0.jsonl")
    by_key = _activation_entries_by_key(entries)
    identical = [
        entry
        for entry in entries
        if entry.get("scenario") == "identical_prompts_batch"
        and entry.get("stage") == "probe"
        and _interesting_checkpoint(str(entry.get("name", "")))
    ]
    identical.sort(key=lambda item: int(item.get("activation_index", 0)))
    rows: list[list[Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in identical:
        phase = str(entry.get("batch", {}).get("phase", ""))
        name = str(entry.get("name", ""))
        key = (phase, name)
        if key in seen:
            continue
        seen.add(key)
        tensor = _load_activation_tensor(run_dir, entry)
        identical_stats = None
        if tensor is not None and tensor.ndim >= 1 and tensor.shape[0] >= 2:
            worst = {"allclose": True, "max_abs": 0.0}
            for row in range(1, min(4, tensor.shape[0])):
                stats = _tensor_diff(tensor[0], tensor[row], atol=atol, rtol=rtol)
                if stats.get("max_abs") == "shape" or float(stats.get("max_abs", 0.0)) >= float(
                    worst.get("max_abs", 0.0)
                ):
                    worst = stats
            identical_stats = worst

        slot_stats = []
        base_entry = by_key.get(("single_target_alone", phase, name))
        base_tensor = _load_activation_tensor(run_dir, base_entry) if base_entry else None
        for slot in range(4):
            slot_entry = by_key.get((f"target_in_batch_slot{slot}", phase, name))
            slot_tensor = _load_activation_tensor(run_dir, slot_entry) if slot_entry else None
            if (
                base_tensor is None
                or slot_tensor is None
                or base_tensor.shape[0] < 1
                or slot_tensor.shape[0] <= slot
            ):
                continue
            slot_stats.append(
                _tensor_diff(base_tensor[0], slot_tensor[slot], atol=atol, rtol=rtol)
            )
        worst_slot = None
        for stats in slot_stats:
            if worst_slot is None or stats.get("max_abs") == "shape" or float(
                stats.get("max_abs", 0.0)
            ) >= float(worst_slot.get("max_abs", 0.0)):
                worst_slot = stats
        rows.append(
            [
                phase,
                name,
                _fmt(identical_stats),
                _fmt(worst_slot),
                entry.get("activation_index", ""),
            ]
        )
    return rows


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    exact = _load_exact_summary_module()
    exact_payload = exact.summarize(
        argparse.Namespace(
            eager=args.eager,
            graph=args.graph,
            output_dir=args.output_dir,
            atol=args.atol,
            rtol=args.rtol,
            activation_atol=args.activation_atol,
            activation_rtol=args.activation_rtol,
        )
    )
    primary_dir = Path(exact_payload["primary_run_dir"])
    rows = _checkpoint_rows(primary_dir, atol=args.activation_atol, rtol=args.activation_rtol)
    _write_text(
        Path(args.output_dir) / "attention_indexer_checkpoint_diff.md",
        _markdown_table(
            ["Phase", "Checkpoint", "Identical Rows", "Alone Vs Slots", "Activation Index"],
            rows or [["not run", "n/a", "n/a", "n/a", "n/a"]],
        ),
    )
    graph_dir = Path(args.graph) if args.graph else None
    run_for_graph = _load_json((graph_dir or primary_dir) / "run.json")
    exact_guard = bool(
        run_for_graph.get("config", {}).get("graph_runner", {}).get("exact_bs_only", False)
    )
    _write_text(
        Path(args.output_dir) / "graph_bucket_decision.md",
        _markdown_table(
            ["Decision", "Value"],
            [
                ["graph exact-bs-only guard", "enabled" if exact_guard else "not enabled in primary run"],
                ["bs=3 without captured bucket 3", "eager fallback when guard is enabled"],
            ],
        ),
    )
    return exact_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TARGET 08.196 row-coupling probe.")
    parser.add_argument("--eager", default=None)
    parser.add_argument("--graph", default=None)
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

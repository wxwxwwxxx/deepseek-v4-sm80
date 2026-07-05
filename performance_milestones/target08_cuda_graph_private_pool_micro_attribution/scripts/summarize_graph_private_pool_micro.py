#!/usr/bin/env python3
"""Summarize TARGET 08.32 graph private-pool microbench raw JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1 << 30)


def _fmt_gib(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{_gib(value):.3f}"


def _fmt_mib(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / float(1 << 20):.1f}"


def _load_cases(raw_dir: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            cases.append({"case": path.stem, "load_error": f"{type(exc).__name__}: {exc}"})
            continue
        payload["_path"] = str(path)
        cases.append(payload)
    return cases


def _case_ok(row: dict[str, Any]) -> bool:
    if "errors" in row:
        return not row.get("errors") and not row.get("load_error")
    return not row.get("capture_error") and not row.get("replay_error") and not row.get("load_error")


def _projection(row: dict[str, Any]) -> int | None:
    delta = row.get("free_delta_bytes")
    if delta is None:
        return None
    shape = row.get("shape") or {}
    layers = int(shape.get("layers") or 1)
    category = row.get("category")
    if category == "scaling":
        case = str(row.get("case") or "")
        full_count = 21 if case.startswith("indexer_topk_only") else 43
        if layers == 1:
            return int(delta) * full_count
        if layers == full_count:
            return int(delta)
        return None
    count = row.get("projection_count")
    if count is None:
        return None
    return int(delta) * int(count)


def _row_common(row: dict[str, Any]) -> dict[str, Any]:
    if "ranks" in row:
        ranks = [r for r in row.get("ranks", []) if r is not None]
        first = ranks[0] if ranks else {}
        return {
            "case": row.get("case"),
            "category": "communication_control",
            "bs": None,
            "layers": 1,
            "variant": row.get("dtype"),
            "ok": _case_ok(row),
            "free_delta_bytes": row.get("max_free_delta_bytes"),
            "allocated_delta_bytes": row.get("max_allocated_delta_bytes"),
            "reserved_delta_bytes": row.get("max_reserved_delta_bytes"),
            "peak_allocated_bytes": max((int(r.get("peak_allocated_bytes") or 0) for r in ranks), default=None),
            "peak_reserved_bytes": max((int(r.get("peak_reserved_bytes") or 0) for r in ranks), default=None),
            "capture_elapsed_s": max((float(r.get("capture_elapsed_s") or 0.0) for r in ranks), default=None),
            "replay_elapsed_s": max((float(r.get("replay_elapsed_s") or 0.0) for r in ranks), default=None),
            "projection_count": 1,
            "projected_full_model_delta_bytes": row.get("max_free_delta_bytes"),
            "projection_note": "measured TP communication graph control",
            "explicit_input_bytes": first.get("explicit_input_output_workspace_bytes"),
            "explicit_output_bytes": 0,
            "explicit_workspace_bytes": 0,
            "explicit_weight_bytes": 0,
            "explicit_cache_bytes": 0,
            "explicit_metadata_bytes": 0,
            "explicit_total_bytes": first.get("explicit_input_output_workspace_bytes"),
            "first_output_shape": [row.get("elements")],
            "first_output_dtype": row.get("dtype"),
            "finite": True,
            "capture_error": None,
            "replay_error": None,
            "load_error": row.get("load_error"),
            "description": f"TP{row.get('world_size')} NCCL all_reduce graph capture, dtype={row.get('dtype')}.",
            "notes": [],
        }
    sanity = row.get("replay_sanity") or {}
    shape = row.get("shape") or {}
    return {
        "case": row.get("case"),
        "category": row.get("category"),
        "bs": shape.get("bs"),
        "layers": shape.get("layers"),
        "variant": shape.get("variant"),
        "ok": _case_ok(row),
        "free_delta_bytes": row.get("free_delta_bytes"),
        "allocated_delta_bytes": row.get("allocated_delta_bytes"),
        "reserved_delta_bytes": row.get("reserved_delta_bytes"),
        "peak_allocated_bytes": row.get("peak_allocated_bytes"),
        "peak_reserved_bytes": row.get("peak_reserved_bytes"),
        "capture_elapsed_s": row.get("capture_elapsed_s"),
        "replay_elapsed_s": row.get("replay_elapsed_s"),
        "projection_count": row.get("projection_count"),
        "projected_full_model_delta_bytes": _projection(row),
        "projection_note": (
            "single-layer projection"
            if row.get("category") == "scaling" and int((row.get("shape") or {}).get("layers") or 1) == 1
            else "measured repeated-43 skeleton"
            if row.get("category") == "scaling"
            and int((row.get("shape") or {}).get("layers") or 1) == 43
            and not str(row.get("case") or "").startswith("indexer_topk_only")
            else "measured repeated-21 skeleton"
            if row.get("category") == "scaling"
            and int((row.get("shape") or {}).get("layers") or 1) == 21
            and str(row.get("case") or "").startswith("indexer_topk_only")
            else "not projected for intermediate repeated N"
            if row.get("category") == "scaling"
            else "single-owner projection x owner count"
        ),
        "explicit_input_bytes": row.get("explicit_input_bytes"),
        "explicit_output_bytes": row.get("explicit_output_bytes"),
        "explicit_workspace_bytes": row.get("explicit_workspace_bytes"),
        "explicit_weight_bytes": row.get("explicit_weight_bytes"),
        "explicit_cache_bytes": row.get("explicit_cache_bytes"),
        "explicit_metadata_bytes": row.get("explicit_metadata_bytes"),
        "explicit_total_bytes": row.get("explicit_total_bytes"),
        "first_output_shape": sanity.get("first_output_shape"),
        "first_output_dtype": sanity.get("first_output_dtype"),
        "finite": sanity.get("finite"),
        "capture_error": row.get("capture_error"),
        "replay_error": row.get("replay_error"),
        "load_error": row.get("load_error"),
        "description": row.get("description"),
        "notes": row.get("notes") or [],
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if key.endswith("_gib"):
                values.append(f"{value:.3f}" if isinstance(value, float) else "n/a")
            elif key.endswith("_mib"):
                values.append(f"{value:.1f}" if isinstance(value, float) else "n/a")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            elif isinstance(value, bool):
                values.append("yes" if value else "no")
            elif value is None:
                values.append("n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _decorate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        decorated = dict(row)
        for key in (
            "free_delta_bytes",
            "allocated_delta_bytes",
            "reserved_delta_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "projected_full_model_delta_bytes",
        ):
            decorated[key.replace("_bytes", "_gib")] = _gib(row.get(key))
        for key in (
            "explicit_input_bytes",
            "explicit_output_bytes",
            "explicit_workspace_bytes",
            "explicit_weight_bytes",
            "explicit_cache_bytes",
            "explicit_metadata_bytes",
            "explicit_total_bytes",
        ):
            decorated[key.replace("_bytes", "_mib")] = (
                float(row[key]) / float(1 << 20) if row.get(key) is not None else None
            )
        out.append(decorated)
    return out


def summarize(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    summary_dir = milestone_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_cases = _load_cases(raw_dir)
    rows = [_row_common(row) for row in raw_cases]
    decorated = _decorate(rows)
    controls = [r for r in decorated if r.get("category") == "control"]
    communication = [r for r in decorated if r.get("category") == "communication_control"]
    dsv4 = [r for r in decorated if r.get("category") == "dsv4_subgraph"]
    scaling = [r for r in decorated if r.get("category") == "scaling"]
    failures = [r for r in decorated if not r.get("ok")]
    successful = [r for r in decorated if r.get("ok")]
    largest = sorted(
        successful,
        key=lambda r: float(r.get("free_delta_gib") or 0.0),
        reverse=True,
    )[:10]
    largest_projection = sorted(
        successful,
        key=lambda r: float(r.get("projected_full_model_delta_gib") or 0.0),
        reverse=True,
    )[:10]
    summary = {
        "case_count": len(rows),
        "success_count": len(successful),
        "failure_count": len(failures),
        "controls": controls,
        "communication_controls": communication,
        "dsv4_subgraphs": dsv4,
        "scaling": scaling,
        "failures": failures,
        "largest_free_delta": largest,
        "largest_projected_delta": largest_projection,
    }
    (summary_dir / "graph_private_pool_micro_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    cols = [
        ("case", "case"),
        ("ok", "ok"),
        ("bs", "bs"),
        ("N", "layers"),
        ("variant", "variant"),
        ("free GiB", "free_delta_gib"),
        ("alloc GiB", "allocated_delta_gib"),
        ("reserved GiB", "reserved_delta_gib"),
        ("capture s", "capture_elapsed_s"),
        ("explicit MiB", "explicit_total_mib"),
        ("projected GiB", "projected_full_model_delta_gib"),
    ]
    md = [
        "# TARGET 08.32 Graph Private-Pool Micro Summary",
        "",
        f"- cases: `{len(rows)}`",
        f"- successes: `{len(successful)}`",
        f"- failures: `{len(failures)}`",
        "",
        "## Largest Measured Free-Memory Deltas",
        "",
        _markdown_table(largest, cols),
        "",
        "## Largest Simple Projections",
        "",
        _markdown_table(largest_projection, cols),
        "",
        "## Controls",
        "",
        _markdown_table(controls, cols),
        "",
        "## Communication Controls",
        "",
        _markdown_table(communication, cols),
        "",
        "## DSV4 Subgraphs",
        "",
        _markdown_table(dsv4, cols),
        "",
        "## One-Layer / Repeated-Layer Scaling",
        "",
        _markdown_table(scaling, cols),
    ]
    if failures:
        md.extend(
            [
                "",
                "## Failures",
                "",
                _markdown_table(
                    failures,
                    [
                        ("case", "case"),
                        ("capture_error", "capture_error"),
                        ("replay_error", "replay_error"),
                        ("load_error", "load_error"),
                    ],
                ),
            ]
        )
    (summary_dir / "graph_private_pool_micro_summary.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--milestone-dir",
        default="performance_milestones/target08_cuda_graph_private_pool_micro_attribution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(Path(args.milestone_dir))
    print(
        json.dumps(
            {
                "case_count": summary["case_count"],
                "success_count": summary["success_count"],
                "failure_count": summary["failure_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

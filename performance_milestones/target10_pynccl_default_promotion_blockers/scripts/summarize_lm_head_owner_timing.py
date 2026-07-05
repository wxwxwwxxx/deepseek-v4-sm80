#!/usr/bin/env python3
"""Summarize lm_head_all_gather owner timing reports for TARGET 10.27."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


LABEL = "dsv4.owner.comm.dsv4.lm_head_all_gather"


def _rank_from_path(path: Path) -> int:
    return int(path.name.split(".rank", 1)[1].split(".", 1)[0])


def _shape(sample: dict[str, Any]) -> str:
    tensor = sample.get("metadata", {}).get("tensor", {})
    return str(tensor.get("shape", []))


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum_ms": 0.0, "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "sum_ms": float(sum(values)),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def summarize_report_dir(name: str, report_dir: Path, pattern: str) -> dict[str, Any]:
    rows = []
    shape_rows: dict[tuple[bool, str], list[float]] = {}
    outliers = []
    for path in sorted(report_dir.glob(pattern)):
        rank = _rank_from_path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        owner = payload["owner_timing"]
        samples = [sample for sample in owner.get("cuda_samples", []) if sample.get("label") == LABEL]
        timed = [sample for sample in samples if sample.get("elapsed_ms") is not None]
        captured = [sample for sample in timed if sample.get("captured")]
        non_captured = [sample for sample in timed if not sample.get("captured")]
        for sample in timed:
            key = (bool(sample.get("captured")), _shape(sample))
            shape_rows.setdefault(key, []).append(float(sample["elapsed_ms"]))
        top_non = max(non_captured, key=lambda sample: float(sample["elapsed_ms"]), default=None)
        rows.append(
            {
                "rank": rank,
                "sample_count": len(samples),
                "timed_count": len(timed),
                "captured_timed_count": len(captured),
                "noncaptured_timed_count": len(non_captured),
                "total_ms": float(sum(float(sample["elapsed_ms"]) for sample in timed)),
                "captured_ms": float(sum(float(sample["elapsed_ms"]) for sample in captured)),
                "noncaptured_ms": float(sum(float(sample["elapsed_ms"]) for sample in non_captured)),
                "top_noncaptured_seq": None if top_non is None else int(top_non["seq"]),
                "top_noncaptured_ms": None if top_non is None else float(top_non["elapsed_ms"]),
                "top_noncaptured_shape": None if top_non is None else _shape(top_non),
            }
        )
        outliers.extend(
            {
                "rank": rank,
                "seq": int(sample["seq"]),
                "captured": bool(sample.get("captured")),
                "elapsed_ms": float(sample["elapsed_ms"]),
                "shape": _shape(sample),
            }
            for sample in timed
        )

    shape_summary = [
        {
            "captured": captured,
            "shape": shape,
            **_stats(values),
        }
        for (captured, shape), values in sorted(shape_rows.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    top_outliers = sorted(outliers, key=lambda row: row["elapsed_ms"], reverse=True)[:16]
    return {
        "name": name,
        "report_dir": str(report_dir),
        "pattern": pattern,
        "rank_rows": rows,
        "totals": {
            "ranks": len(rows),
            "total_ms": float(sum(row["total_ms"] for row in rows)),
            "captured_ms": float(sum(row["captured_ms"] for row in rows)),
            "noncaptured_ms": float(sum(row["noncaptured_ms"] for row in rows)),
            "timed_count": int(sum(row["timed_count"] for row in rows)),
            "captured_timed_count": int(sum(row["captured_timed_count"] for row in rows)),
            "noncaptured_timed_count": int(sum(row["noncaptured_timed_count"] for row in rows)),
        },
        "shape_summary": shape_summary,
        "top_outliers": top_outliers,
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# lm_head_all_gather Owner Timing Summary", ""]
    for summary in summaries:
        totals = summary["totals"]
        lines.append(f"## {summary['name']}")
        lines.append("")
        lines.append(
            f"- total_ms={totals['total_ms']:.3f}, "
            f"noncaptured_ms={totals['noncaptured_ms']:.3f}, "
            f"captured_ms={totals['captured_ms']:.3f}, "
            f"timed={totals['timed_count']}"
        )
        lines.append("")
        lines.append("| rank | total ms | noncaptured ms | captured ms | top noncaptured ms | top seq | top shape |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in summary["rank_rows"]:
            lines.append(
                f"| {row['rank']} | {row['total_ms']:.3f} | {row['noncaptured_ms']:.3f} | "
                f"{row['captured_ms']:.3f} | {row['top_noncaptured_ms']:.3f} | "
                f"{row['top_noncaptured_seq']} | `{row['top_noncaptured_shape']}` |"
            )
        lines.append("")
        lines.append("Shape split:")
        lines.append("")
        lines.append("| captured | shape | count | sum ms | mean ms | max ms |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
        for row in summary["shape_summary"]:
            lines.append(
                f"| {int(row['captured'])} | `{row['shape']}` | {row['count']} | "
                f"{row['sum_ms']:.3f} | {row['mean_ms']:.3f} | {row['max_ms']:.3f} |"
            )
        lines.append("")
        lines.append("Top outliers:")
        lines.append("")
        lines.append("| rank | seq | captured | elapsed ms | shape |")
        lines.append("| ---: | ---: | ---: | ---: | --- |")
        for row in summary["top_outliers"][:8]:
            lines.append(
                f"| {row['rank']} | {row['seq']} | {int(row['captured'])} | "
                f"{row['elapsed_ms']:.3f} | `{row['shape']}` |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Name and report dir as NAME=DIR.",
    )
    parser.add_argument("--pattern", default="000_historical*.rank*.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for run in args.run:
        name, raw_dir = run.split("=", 1)
        summaries.append(summarize_report_dir(name, Path(raw_dir), args.pattern))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summaries)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any


GIB = 1024**3


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sum_nested(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0) or 0) for row in rows))


def _rank(row: dict[str, Any]) -> int | None:
    value = row.get("rank")
    if value is None:
        return None
    return int(value)


def _run_label(path: Path, prefix: str) -> str:
    name = path.name
    if name.startswith(prefix):
        name = name[len(prefix) :]
    match = re.match(r"(?P<label>.+)_rank(?P<rank>\d+)\.jsonl$", name)
    if match:
        return match.group("label")
    return path.stem


def _load_labeled_jsonl(path: Path, prefix: str) -> list[dict[str, Any]]:
    label = _run_label(path, prefix)
    rows = []
    for row in _load_jsonl(path):
        item = dict(row)
        item["run_label"] = label
        item["source_file"] = str(path)
        rows.append(item)
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("run_label", "unknown")), []).append(row)
    return dict(sorted(grouped.items()))


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def theoretical_ledger(
    config: dict[str, Any],
    *,
    tp_size: int,
    page_size: int,
    indexer_fp8_cache: bool,
) -> dict[str, Any]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    local_intermediate = intermediate // tp_size
    experts = int(config["n_routed_experts"])
    layers = int(config["num_hidden_layers"])
    scale_hidden = _ceil_div(hidden, 32)
    scale_intermediate = _ceil_div(local_intermediate, 32)

    w13_raw = experts * 2 * local_intermediate * (hidden // 2)
    w2_raw = experts * hidden * (local_intermediate // 2)
    w13_scale = experts * 2 * local_intermediate * scale_hidden
    w2_scale = experts * hidden * scale_intermediate

    # gptq_marlin_repack allocates int32 output
    # [size_k / 16, size_n * 16 / (32 / num_bits)] per expert. With 4-bit
    # weights this is byte-identical to mini's packed int8 source tensors.
    w13_repacked = w13_raw
    w2_repacked = w2_raw
    w13_repacked_scale = w13_scale
    w2_repacked_scale = w2_scale

    per_layer_source = w13_raw + w2_raw + w13_scale + w2_scale
    per_layer_repacked = w13_repacked + w2_repacked + w13_repacked_scale + w2_repacked_scale
    kv_bytes_per_page = estimate_dsv4_kv_bytes_per_page(
        config,
        page_size=page_size,
        indexer_fp8_cache=indexer_fp8_cache,
    )

    def item(bytes_: int) -> dict[str, Any]:
        return {
            "bytes": int(bytes_),
            "gib": bytes_ / GIB,
            "kv_pages": bytes_ / kv_bytes_per_page,
            "kv_tokens": bytes_ / kv_bytes_per_page * page_size,
        }

    return {
        "config": {
            "hidden_size": hidden,
            "moe_intermediate_size": intermediate,
            "local_intermediate": local_intermediate,
            "n_routed_experts": experts,
            "num_layers": layers,
            "tp_size": tp_size,
            "page_size": page_size,
            "indexer_fp8_cache": bool(indexer_fp8_cache),
            "kv_bytes_per_page": kv_bytes_per_page,
        },
        "per_layer": {
            "raw_packed_w13": item(w13_raw),
            "raw_packed_w2": item(w2_raw),
            "raw_w13_scale": item(w13_scale),
            "raw_w2_scale": item(w2_scale),
            "raw_total": item(per_layer_source),
            "repacked_w13": item(w13_repacked),
            "repacked_w2": item(w2_repacked),
            "repacked_w13_scale": item(w13_repacked_scale),
            "repacked_w2_scale": item(w2_repacked_scale),
            "repacked_total": item(per_layer_repacked),
        },
        "all_layers": {
            "raw_total": item(per_layer_source * layers),
            "repacked_total": item(per_layer_repacked * layers),
            "raw_plus_repacked_total": item((per_layer_source + per_layer_repacked) * layers),
        },
    }


def estimate_dsv4_kv_bytes_per_page(
    config: dict[str, Any],
    *,
    page_size: int,
    indexer_fp8_cache: bool,
) -> int:
    dtype_size = 2
    head_dim = int(config["head_dim"])
    index_head_dim = int(config.get("index_head_dim") or head_dim)
    num_layers = int(config["num_hidden_layers"])
    ratios = list(config.get("compress_ratios") or [0] * num_layers)
    if len(ratios) < num_layers:
        ratios.extend([0] * (num_layers - len(ratios)))
    ratios = ratios[:num_layers]
    c4_layers = sum(1 for ratio in ratios if ratio == 4)
    c128_layers = sum(1 for ratio in ratios if ratio == 128)

    def compressed_bytes(layers: int, dim: int, ratio: int, multiplier: int = 1) -> int:
        return _ceil_div(layers * page_size * dim * multiplier * dtype_size, ratio)

    swa_bytes = num_layers * page_size * head_dim * dtype_size
    c4_bytes = compressed_bytes(c4_layers, head_dim, 4)
    c128_bytes = compressed_bytes(c128_layers, head_dim, 128)
    indexer_bytes = compressed_bytes(c4_layers, index_head_dim, 4)
    indexer_fp8_extra_bytes = (
        _ceil_div(c4_layers * page_size * (index_head_dim + 4), 4)
        if indexer_fp8_cache
        else 0
    )
    c4_state_bytes = c4_layers * 256 * 4 * head_dim * dtype_size
    c4_indexer_state_bytes = c4_layers * 256 * 4 * index_head_dim * dtype_size
    c128_state_bytes = c128_layers * 128 * 2 * head_dim * dtype_size
    return int(
        swa_bytes
        + c4_bytes
        + c128_bytes
        + indexer_bytes
        + indexer_fp8_extra_bytes
        + c4_state_bytes
        + c4_indexer_state_bytes
        + c128_state_bytes
    )


def summarize_raw(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    marlin_files = sorted(raw_dir.glob("marlin_wna16_cache_*_rank*.jsonl"))
    warmup_files = sorted(raw_dir.glob("warmup_forward_memory_*_rank*.jsonl"))
    graph_stage_files = sorted(raw_dir.glob("graph_capture_stage_*_rank*.jsonl"))

    marlin_rows = [
        row
        for path in marlin_files
        for row in _load_labeled_jsonl(path, "marlin_wna16_cache_")
    ]
    warmup_rows = [
        row
        for path in warmup_files
        for row in _load_labeled_jsonl(path, "warmup_forward_memory_")
    ]
    graph_stage_rows = [
        row
        for path in graph_stage_files
        for row in _load_labeled_jsonl(path, "graph_capture_stage_")
    ]

    marlin_by_run = {
        label: _summarize_marlin_rows(rows)
        for label, rows in _group_rows(marlin_rows).items()
    }
    graph_stage_by_run = {
        label: _summarize_graph_stage_rows(rows)
        for label, rows in _group_rows(graph_stage_rows).items()
    }
    warmup_by_run = {
        label: _summarize_warmup_rows(rows)
        for label, rows in _group_rows(warmup_rows).items()
    }

    top_warmup_alloc = sorted(
        warmup_rows,
        key=lambda row: int(row.get("memory_allocated_delta_from_previous_bytes", 0) or 0),
        reverse=True,
    )[:20]
    top_graph_alloc = sorted(
        graph_stage_rows,
        key=lambda row: int(row.get("memory_allocated_delta_from_previous_bytes", 0) or 0),
        reverse=True,
    )[:20]

    return {
        "files": {
            "marlin_wna16_cache": [str(path) for path in marlin_files],
            "warmup_forward_memory": [str(path) for path in warmup_files],
            "graph_capture_stage": [str(path) for path in graph_stage_files],
        },
        "marlin_wna16_cache": {
            "rows": len(marlin_rows),
            "source_total_bytes": _sum_nested(marlin_rows, "source_total_bytes"),
            "repacked_total_bytes": _sum_nested(marlin_rows, "repacked_total_bytes"),
            "memory_allocated_delta_bytes": _sum_nested(
                marlin_rows, "memory_allocated_delta_bytes"
            ),
            "memory_reserved_delta_bytes": _sum_nested(marlin_rows, "memory_reserved_delta_bytes"),
            "free_delta_bytes": _sum_nested(marlin_rows, "free_delta_bytes"),
            "entries": marlin_rows,
            "by_run": marlin_by_run,
        },
        "warmup_forward_memory": {
            "rows": len(warmup_rows),
            "top_alloc_deltas": top_warmup_alloc,
            "by_run": warmup_by_run,
        },
        "graph_capture_stage": {
            "rows": len(graph_stage_rows),
            "top_alloc_deltas": top_graph_alloc,
            "by_run": graph_stage_by_run,
        },
    }


def _summarize_marlin_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rank0 = [row for row in rows if _rank(row) == 0]
    return {
        "rows": len(rows),
        "ranks": sorted({int(row["rank"]) for row in rows if row.get("rank") is not None}),
        "source_total_bytes": _sum_nested(rows, "source_total_bytes"),
        "repacked_total_bytes": _sum_nested(rows, "repacked_total_bytes"),
        "memory_allocated_delta_bytes": _sum_nested(rows, "memory_allocated_delta_bytes"),
        "memory_reserved_delta_bytes": _sum_nested(rows, "memory_reserved_delta_bytes"),
        "free_delta_bytes": _sum_nested(rows, "free_delta_bytes"),
        "rank0": {
            "rows": len(rank0),
            "source_total_bytes": _sum_nested(rank0, "source_total_bytes"),
            "repacked_total_bytes": _sum_nested(rank0, "repacked_total_bytes"),
            "memory_allocated_delta_bytes": _sum_nested(rank0, "memory_allocated_delta_bytes"),
            "memory_reserved_delta_bytes": _sum_nested(rank0, "memory_reserved_delta_bytes"),
            "free_delta_bytes": _sum_nested(rank0, "free_delta_bytes"),
            "first_owner": rank0[0].get("owner") if rank0 else None,
            "last_owner": rank0[-1].get("owner") if rank0 else None,
        },
    }


def _summarize_graph_stage_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rank0 = [row for row in rows if _rank(row) == 0]
    stages = {
        str(row.get("stage")): {
            key: row.get(key)
            for key in (
                "batch_size",
                "free_memory_bytes",
                "memory_allocated_bytes",
                "memory_reserved_bytes",
                "free_delta_from_previous_bytes",
                "memory_allocated_delta_from_previous_bytes",
                "memory_reserved_delta_from_previous_bytes",
                "free_delta_from_baseline_bytes",
                "memory_allocated_delta_from_baseline_bytes",
                "memory_reserved_delta_from_baseline_bytes",
            )
            if key in row
        }
        for row in rank0
    }
    return {
        "rows": len(rows),
        "rank0_rows": len(rank0),
        "rank0_stages": stages,
        "rank0_after_warmup_model_forward": stages.get("after_warmup_model.forward"),
        "rank0_after_actual_cuda_graph_capture": stages.get("after_actual_cuda_graph_capture"),
    }


def _summarize_warmup_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top = sorted(
        [row for row in rows if _rank(row) == 0],
        key=lambda row: int(row.get("memory_allocated_delta_from_previous_bytes", 0) or 0),
        reverse=True,
    )[:20]
    return {
        "rows": len(rows),
        "rank0_rows": sum(1 for row in rows if _rank(row) == 0),
        "rank0_top_alloc_deltas": top,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    theory = summary["theoretical"]
    raw = summary["raw_summary"]
    lines = [
        "# MoE Marlin WNA16 Lifecycle Summary",
        "",
        "## Theoretical Ledger",
        "",
        "| Item | Bytes/rank | GiB/rank | KV pages | KV tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in theory["per_layer"].items():
        lines.append(_ledger_row(f"per_layer.{name}", item))
    for name, item in theory["all_layers"].items():
        lines.append(_ledger_row(f"all_layers.{name}", item))
    lines.extend(
        [
            "",
            "## Parsed Raw Logs",
            "",
            f"- marlin rows: `{raw['marlin_wna16_cache']['rows']}`",
            f"- marlin repacked total: `{raw['marlin_wna16_cache']['repacked_total_bytes']}` bytes",
            f"- warmup rows: `{raw['warmup_forward_memory']['rows']}`",
            f"- graph stage rows: `{raw['graph_capture_stage']['rows']}`",
            "",
            "## Runs",
            "",
            "| Run | Marlin rank0 repacked GiB | Warmup rank0 alloc delta GiB | Graph-capture rank0 alloc delta GiB |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    run_labels = sorted(
        set(raw["graph_capture_stage"].get("by_run", {}))
        | set(raw["marlin_wna16_cache"].get("by_run", {}))
    )
    for label in run_labels:
        marlin_run = raw["marlin_wna16_cache"].get("by_run", {}).get(label, {})
        graph_run = raw["graph_capture_stage"].get("by_run", {}).get(label, {})
        marlin_rank0 = marlin_run.get("rank0", {}).get("repacked_total_bytes")
        warmup = graph_run.get("rank0_after_warmup_model_forward") or {}
        capture = graph_run.get("rank0_after_actual_cuda_graph_capture") or {}
        lines.append(
            "| "
            + f"`{label}` | "
            + _fmt_gib(marlin_rank0)
            + " | "
            + _fmt_gib(warmup.get("memory_allocated_delta_from_previous_bytes"))
            + " | "
            + _fmt_gib(capture.get("memory_allocated_delta_from_previous_bytes"))
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ledger_row(name: str, item: dict[str, Any]) -> str:
    return (
        f"| `{name}` | {int(item['bytes']):,} | {item['gib']:.4f} | "
        f"{item['kv_pages']:.2f} | {item['kv_tokens']:.0f} |"
    )


def _fmt_gib(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value) / GIB:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milestone-dir",
        type=Path,
        default=Path("performance_milestones/target08_moe_marlin_wna16_cache_lifecycle"),
    )
    parser.add_argument("--config", type=Path, default=Path("/models/DeepSeek-V4-Flash/config.json"))
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument(
        "--indexer-fp8-cache",
        action="store_true",
        default=os.environ.get("MINISGL_DSV4_SM80_INDEXER_FP8_CACHE") == "1",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = {
        "theoretical": theoretical_ledger(
            config,
            tp_size=args.tp_size,
            page_size=args.page_size,
            indexer_fp8_cache=args.indexer_fp8_cache,
        ),
        "raw_summary": summarize_raw(args.milestone_dir),
    }
    summaries_dir = args.milestone_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    json_path = summaries_dir / "moe_marlin_lifecycle_summary.json"
    md_path = summaries_dir / "moe_marlin_lifecycle_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


CATEGORIES: dict[str, tuple[str, ...]] = {
    "sparse_attention": (
        "sparse_attention",
        "sparse_attn",
        "dsv4_sparse",
    ),
    "indexer_cache": (
        "indexer",
        "topk_transform",
        "global_topk",
        "q_kv_norm_rope",
        "kv_rope",
        "cache_store",
        "idxstore",
        "masked_locs",
        "compress",
    ),
    "moe_route_w13_swiglu_w2_sum": (
        "marlin_moe_wna16",
        "grouped_fp4",
        "moe_route",
        "moe_gate",
        "swiglu",
        "silu",
        "clamp",
        "gptq_marlin_repack",
    ),
    "hc_rmsnorm_logits_sampling": (
        "_hc_",
        "rmsnorm",
        "rms_norm",
        "lm_head",
        "logits",
        "sample",
        "sampling",
        "argmax",
    ),
    "nccl": (
        "nccl",
    ),
    "runtime_memcpy_allocation_kernels": (
        "copy",
        "fill",
        "catarray",
        "index_elementwise",
        "arange",
        "memset",
    ),
    "dense_linear_other": (
        "gemm",
        "cublas",
        "ampere_bf16",
        "ampere_fp16",
        "cutlass",
    ),
}

RUNTIME_CATEGORIES: dict[str, tuple[str, ...]] = {
    "cuda_graph_runtime": ("graph", "capture"),
    "sync_runtime": ("synchronize", "event"),
    "launch_runtime": ("launch",),
    "memcpy_runtime": ("memcpy",),
    "allocation_runtime": ("malloc", "free", "hostalloc", "memalloc"),
    "module_runtime": ("module",),
}


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (name,),
    ).fetchone()
    return row is not None


def _safe_rows(
    cur: sqlite3.Cursor,
    query: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    try:
        out: list[dict[str, Any]] = []
        cols = None
        for row in cur.execute(query, tuple(params)):
            if cols is None:
                cols = [item[0] for item in cur.description]
            out.append({key: row[idx] for idx, key in enumerate(cols)})
        return out
    except sqlite3.Error:
        return []


def _safe_one(cur: sqlite3.Cursor, query: str, params: Sequence[Any] = ()) -> Any:
    try:
        return cur.execute(query, tuple(params)).fetchone()
    except sqlite3.Error:
        return None


def _find_nvtx_window(cur: sqlite3.Cursor, name: str | None) -> tuple[int, int] | None:
    if not name or not table_exists(cur, "NVTX_EVENTS"):
        return None
    row = _safe_one(
        cur,
        """
        select n.start, n.end
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where (n.text=? or s.value=?) and n.end is not null
        order by (n.end - n.start) desc
        limit 1
        """,
        (name, name),
    )
    if row is None:
        return None
    return int(row[0]), int(row[1])


def _where_window(alias: str, window: tuple[int, int] | None) -> tuple[str, list[Any]]:
    if window is None:
        return "", []
    return f" where {alias}.start>=? and {alias}.end<=?", [window[0], window[1]]


def _named_events(
    cur: sqlite3.Cursor,
    table: str,
    name_col: str,
    window: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if not table_exists(cur, table) or not table_exists(cur, "StringIds"):
        return []
    where, params = _where_window("e", window)
    return _safe_rows(
        cur,
        f"""
        select
          coalesce(s.value, cast(e.{name_col} as text)) as name,
          count(*) as count,
          coalesce(sum(e.end - e.start), 0) / 1000000000.0 as duration_s
        from {table} e
        left join StringIds s on s.id = e.{name_col}
        {where}
        group by e.{name_col}
        order by sum(e.end - e.start) desc
        """,
        params,
    )


def _duration_count(
    cur: sqlite3.Cursor,
    table: str,
    window: tuple[int, int] | None,
) -> dict[str, Any]:
    if not table_exists(cur, table):
        return {"present": False, "count": 0, "duration_s": 0.0}
    where, params = _where_window("e", window)
    row = _safe_one(
        cur,
        f"select count(*), coalesce(sum(e.end-e.start), 0) / 1000000000.0 from {table} e{where}",
        params,
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {"present": True, "count": int(row[0]), "duration_s": float(row[1])}


def _memcpy_summary(cur: sqlite3.Cursor, window: tuple[int, int] | None) -> dict[str, Any]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return {"present": False, "count": 0, "duration_s": 0.0, "bytes": 0}
    where, params = _where_window("e", window)
    row = _safe_one(
        cur,
        f"""
        select count(*), coalesce(sum(e.end-e.start), 0) / 1000000000.0,
               coalesce(sum(e.bytes), 0)
        from CUPTI_ACTIVITY_KIND_MEMCPY e{where}
        """,
        params,
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None, "bytes": None}
    return {
        "present": True,
        "count": int(row[0]),
        "duration_s": float(row[1]),
        "bytes": int(row[2]),
    }


def _classify_name(name: str, patterns: dict[str, tuple[str, ...]]) -> str:
    lowered = name.lower()
    for category, needles in patterns.items():
        if any(needle in lowered for needle in needles):
            return category
    return "other"


def _classify_events(
    events: Sequence[dict[str, Any]],
    patterns: dict[str, tuple[str, ...]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "duration_s": 0.0}
    )
    unclassified: list[dict[str, Any]] = []
    for row in events:
        name = str(row.get("name") or "")
        count = int(row.get("count") or 0)
        duration_s = float(row.get("duration_s") or 0.0)
        category = _classify_name(name, patterns)
        by_category[category]["count"] += count
        by_category[category]["duration_s"] += duration_s
        if category == "other":
            unclassified.append({"name": name, "count": count, "duration_s": duration_s})
    return dict(by_category), unclassified


def build_summary(
    sqlite_path: Path,
    *,
    nvtx_window: str | None,
    top: int,
) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    window = _find_nvtx_window(cur, nvtx_window)
    sections: dict[str, Any] = {}
    for label, active_window in (("total", None), ("nvtx_window", window)):
        kernels = _named_events(cur, "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName", active_window)
        runtime = _named_events(cur, "CUPTI_ACTIVITY_KIND_RUNTIME", "nameId", active_window)
        kernel_categories, unclassified = _classify_events(kernels, CATEGORIES)
        runtime_categories, runtime_unclassified = _classify_events(runtime, RUNTIME_CATEGORIES)
        sections[label] = {
            "window_ns": active_window,
            "window_found": None if label == "total" else active_window is not None,
            "kernel": _duration_count(cur, "CUPTI_ACTIVITY_KIND_KERNEL", active_window),
            "graph_trace": _duration_count(cur, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", active_window),
            "runtime": _duration_count(cur, "CUPTI_ACTIVITY_KIND_RUNTIME", active_window),
            "memcpy": _memcpy_summary(cur, active_window),
            "memset": _duration_count(cur, "CUPTI_ACTIVITY_KIND_MEMSET", active_window),
            "sync": _duration_count(cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", active_window),
            "kernel_categories": kernel_categories,
            "runtime_categories": runtime_categories,
            "top_kernels": kernels[:top],
            "top_runtime": runtime[:top],
            "top_unclassified_kernels": sorted(
                unclassified, key=lambda item: item["duration_s"], reverse=True
            )[:top],
            "top_unclassified_runtime": sorted(
                runtime_unclassified, key=lambda item: item["duration_s"], reverse=True
            )[:top],
        }
    graph_events = _duration_count(cur, "CUDA_GRAPH_EVENTS", None)
    graph_node_events = _duration_count(cur, "CUDA_GRAPH_NODE_EVENTS", None)
    return {
        "sqlite_path": str(sqlite_path),
        "requested_nvtx_window": nvtx_window,
        "category_patterns": {key: list(value) for key, value in CATEGORIES.items()},
        "runtime_category_patterns": {
            key: list(value) for key, value in RUNTIME_CATEGORIES.items()
        },
        "cuda_graph_events": graph_events,
        "cuda_graph_node_events": graph_node_events,
        **sections,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _pct(part: float | None, total: float | None) -> str:
    if not part or not total:
        return "n/a"
    return f"{100.0 * part / total:.2f}%"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Post-Marlin Nsight Classification: {Path(summary['sqlite_path']).name}")
    lines.append("")
    lines.append(f"- Requested NVTX window: `{summary.get('requested_nvtx_window')}`")
    lines.append("")
    for section_name in ("total", "nvtx_window"):
        section = summary[section_name]
        kernel_total = section["kernel"].get("duration_s") or 0.0
        runtime_total = section["runtime"].get("duration_s") or 0.0
        lines.append(f"## {section_name}")
        lines.append("")
        if section_name == "nvtx_window":
            lines.append(f"- Window found: {section.get('window_found')}")
            lines.append("")
        lines.append("| Metric | Count | Duration s | Extra |")
        lines.append("| --- | ---: | ---: | --- |")
        lines.append(
            f"| kernels | {_fmt(section['kernel'].get('count'))} | {_fmt(kernel_total)} | |"
        )
        lines.append(
            f"| graph trace | {_fmt(section['graph_trace'].get('count'))} | {_fmt(section['graph_trace'].get('duration_s'))} | |"
        )
        lines.append(
            f"| runtime | {_fmt(section['runtime'].get('count'))} | {_fmt(runtime_total)} | |"
        )
        lines.append(
            f"| memcpy | {_fmt(section['memcpy'].get('count'))} | {_fmt(section['memcpy'].get('duration_s'))} | bytes={_fmt(section['memcpy'].get('bytes'))} |"
        )
        lines.append("")
        lines.append("Kernel categories:")
        lines.append("")
        lines.append("| Category | Count | Kernel duration s | Kernel share |")
        lines.append("| --- | ---: | ---: | ---: |")
        for category, item in sorted(
            section["kernel_categories"].items(),
            key=lambda pair: pair[1].get("duration_s", 0.0),
            reverse=True,
        ):
            duration = float(item.get("duration_s") or 0.0)
            lines.append(
                f"| `{category}` | {item.get('count', 0)} | {_fmt(duration)} | {_pct(duration, kernel_total)} |"
            )
        lines.append("")
        lines.append("Runtime categories:")
        lines.append("")
        lines.append("| Category | Count | Runtime duration s | Runtime share |")
        lines.append("| --- | ---: | ---: | ---: |")
        for category, item in sorted(
            section["runtime_categories"].items(),
            key=lambda pair: pair[1].get("duration_s", 0.0),
            reverse=True,
        ):
            duration = float(item.get("duration_s") or 0.0)
            lines.append(
                f"| `{category}` | {item.get('count', 0)} | {_fmt(duration)} | {_pct(duration, runtime_total)} |"
            )
        lines.append("")
        lines.append("Top kernels:")
        lines.append("")
        lines.append("| Name | Count | Duration s |")
        lines.append("| --- | ---: | ---: |")
        for row in section["top_kernels"][:10]:
            name = str(row["name"]).replace("|", "\\|")
            lines.append(f"| `{name[:150]}` | {row['count']} | {_fmt(row['duration_s'])} |")
        lines.append("")
        lines.append("Top runtime APIs:")
        lines.append("")
        lines.append("| Name | Count | Duration s |")
        lines.append("| --- | ---: | ---: |")
        for row in section["top_runtime"][:10]:
            name = str(row["name"]).replace("|", "\\|")
            lines.append(f"| `{name[:150]}` | {row['count']} | {_fmt(row['duration_s'])} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify post-Marlin Nsight SQLite data.")
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--nvtx-window", default="repeat:smoke_debug:0")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_summary(args.sqlite_path, nvtx_window=args.nvtx_window, top=args.top)
    if args.output_json is None and args.output_md is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md is not None:
        write_markdown(args.output_md, summary)


if __name__ == "__main__":
    main()

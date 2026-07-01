#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (name,),
    ).fetchone()
    return row is not None


def _safe_fetchone(cur: sqlite3.Cursor, query: str, params: Sequence[Any] = ()) -> Any:
    try:
        return cur.execute(query, tuple(params)).fetchone()
    except sqlite3.Error:
        return None


def _rows(cur: sqlite3.Cursor, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    try:
        cols = None
        out = []
        for row in cur.execute(query, tuple(params)):
            if cols is None:
                cols = [item[0] for item in cur.description]
            out.append({key: row[idx] for idx, key in enumerate(cols)})
        return out
    except sqlite3.Error:
        return []


def _duration_count(
    cur: sqlite3.Cursor,
    table: str,
    window: tuple[int, int] | None,
) -> dict[str, Any]:
    if not table_exists(cur, table):
        return {"present": False}
    where = ""
    params: tuple[int, int] | tuple[()] = ()
    if window is not None:
        where = " where start>=? and end<=?"
        params = window
    row = _safe_fetchone(
        cur,
        f"select count(*), coalesce(sum(end-start),0) from {table}{where}",
        params,
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {
        "present": True,
        "count": int(row[0]),
        "duration_s": float(row[1]) / 1e9,
    }


def _top_named_events(
    cur: sqlite3.Cursor,
    table: str,
    name_col: str,
    *,
    top: int,
    window: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if not table_exists(cur, table) or not table_exists(cur, "StringIds"):
        return []
    where = ""
    params: list[Any] = []
    if window is not None:
        where = "where e.start>=? and e.end<=?"
        params.extend(window)
    params.append(top)
    return _rows(
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
        limit ?
        """,
        params,
    )


def _top_nvtx(
    cur: sqlite3.Cursor,
    *,
    top: int,
    window: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return []
    where = ""
    params: list[Any] = []
    if window is not None:
        where = "where n.start>=? and (n.end<=? or n.end is null)"
        params.extend(window)
    params.append(top)
    return _rows(
        cur,
        f"""
        select
          coalesce(n.text, s.value, cast(n.textId as text)) as name,
          count(*) as count,
          min(n.start) as start_ns,
          max(n.end) as end_ns
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        {where}
        group by coalesce(n.text, s.value, cast(n.textId as text))
        order by count(*) desc
        limit ?
        """,
        params,
    )


def _name_filter_summary(
    cur: sqlite3.Cursor,
    table: str,
    name_col: str,
    pattern: str,
    *,
    window: tuple[int, int] | None,
) -> dict[str, Any]:
    if not table_exists(cur, table) or not table_exists(cur, "StringIds"):
        return {"present": table_exists(cur, table), "count": 0, "duration_s": 0.0}
    where = "where lower(coalesce(s.value, '')) like ?"
    params: list[Any] = [pattern.lower()]
    if window is not None:
        where += " and e.start>=? and e.end<=?"
        params.extend(window)
    row = _safe_fetchone(
        cur,
        f"""
        select count(*), coalesce(sum(e.end - e.start), 0) / 1000000000.0
        from {table} e
        left join StringIds s on s.id = e.{name_col}
        {where}
        """,
        params,
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {"present": True, "count": int(row[0]), "duration_s": float(row[1])}


def _nvtx_filter_summary(
    cur: sqlite3.Cursor,
    pattern: str,
    *,
    window: tuple[int, int] | None,
) -> dict[str, Any]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return {"present": False, "count": 0}
    where = "where lower(coalesce(n.text, s.value, '')) like ?"
    params: list[Any] = [pattern.lower()]
    if window is not None:
        where += " and n.start>=? and (n.end<=? or n.end is null)"
        params.extend(window)
    row = _safe_fetchone(
        cur,
        f"""
        select count(*), min(n.start), max(n.end)
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        {where}
        """,
        params,
    )
    if row is None:
        return {"present": True, "count": None}
    return {
        "present": True,
        "count": int(row[0]),
        "start_ns": row[1],
        "end_ns": row[2],
    }


def _memcpy_summary(
    cur: sqlite3.Cursor,
    *,
    window: tuple[int, int] | None,
) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not table_exists(cur, table):
        return {"present": False}
    where = ""
    params: tuple[int, int] | tuple[()] = ()
    if window is not None:
        where = " where start>=? and end<=?"
        params = window
    row = _safe_fetchone(
        cur,
        f"""
        select count(*), coalesce(sum(end-start), 0), coalesce(sum(bytes), 0)
        from {table}{where}
        """,
        params,
    )
    if row is None:
        return {"present": True, "count": None}
    return {
        "present": True,
        "count": int(row[0]),
        "duration_s": float(row[1]) / 1e9,
        "bytes": int(row[2]),
    }


def _find_nvtx_window(cur: sqlite3.Cursor, name: str | None) -> tuple[int, int] | None:
    if not name or not table_exists(cur, "NVTX_EVENTS"):
        return None
    row = _safe_fetchone(
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


def _cuda_graph_summary(cur: sqlite3.Cursor) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cuda_graph_events_present": table_exists(cur, "CUDA_GRAPH_EVENTS"),
        "cuda_graph_events_count": 0,
        "kernel_graph_node_nonnull_count": None,
        "event_names": [],
        "runtime_graph_api": [],
    }
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        row = _safe_fetchone(
            cur,
            "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null",
        )
        out["kernel_graph_node_nonnull_count"] = None if row is None else int(row[0])
    if table_exists(cur, "CUDA_GRAPH_EVENTS"):
        row = _safe_fetchone(cur, "select count(*) from CUDA_GRAPH_EVENTS")
        out["cuda_graph_events_count"] = None if row is None else int(row[0])
        out["event_names"] = _rows(
            cur,
            """
            select coalesce(s.value, cast(g.nameId as text)) as name, count(*) as count
            from CUDA_GRAPH_EVENTS g
            left join StringIds s on s.id = g.nameId
            group by g.nameId
            order by count(*) desc
            """,
        )
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME") and table_exists(cur, "StringIds"):
        out["runtime_graph_api"] = _rows(
            cur,
            """
            select coalesce(s.value, cast(r.nameId as text)) as name,
                   count(*) as count,
                   coalesce(sum(r.end - r.start), 0) / 1000000000.0 as duration_s
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            left join StringIds s on s.id = r.nameId
            where lower(coalesce(s.value, '')) like '%graph%'
               or lower(coalesce(s.value, '')) like '%capture%'
            group by r.nameId
            order by count(*) desc
            """,
        )
    return out


def _skipped_summary(label: str) -> dict[str, Any]:
    return {"present": None, "count": None, "duration_s": None, "skipped": label}


def _missing_window_section(nvtx_window: str | None) -> dict[str, Any]:
    skipped = _skipped_summary("missing nvtx window")
    return {
        "window_name": nvtx_window,
        "window_ns": None,
        "window_found": False,
        "kernel": dict(skipped),
        "graph_trace": dict(skipped),
        "runtime": dict(skipped),
        "memcpy": {**skipped, "bytes": None},
        "memset": dict(skipped),
        "sync": dict(skipped),
        "top_kernels": [],
        "top_runtime": [],
        "top_nvtx": [],
        "nccl_kernel": dict(skipped),
        "nccl_nvtx": {**skipped, "start_ns": None, "end_ns": None},
    }


def build_summary(
    path: Path,
    *,
    nvtx_window: str | None,
    top: int,
    lite: bool,
) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    window = _find_nvtx_window(cur, nvtx_window)
    table_names = [
        row[0]
        for row in cur.execute(
            "select name from sqlite_master where type='table' order by name"
        )
    ]

    sections: dict[str, Any] = {}
    for label, window_value in (("total", None), ("nvtx_window", window)):
        if label == "nvtx_window" and nvtx_window and window_value is None:
            sections[label] = _missing_window_section(nvtx_window)
            continue
        sections[label] = {
            "window_name": nvtx_window if label == "nvtx_window" else None,
            "window_ns": window_value,
            "window_found": bool(window_value is not None)
            if label == "nvtx_window" and nvtx_window
            else None,
            "kernel": _duration_count(cur, "CUPTI_ACTIVITY_KIND_KERNEL", window_value),
            "graph_trace": _duration_count(
                cur, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", window_value
            ),
            "runtime": _duration_count(cur, "CUPTI_ACTIVITY_KIND_RUNTIME", window_value),
            "memcpy": _memcpy_summary(cur, window=window_value),
            "memset": _duration_count(cur, "CUPTI_ACTIVITY_KIND_MEMSET", window_value),
            "sync": _duration_count(
                cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", window_value
            ),
            "top_kernels": []
            if lite
            else _top_named_events(
                    cur,
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "demangledName",
                    top=top,
                    window=window_value,
                ),
            "top_runtime": []
            if lite
            else _top_named_events(
                    cur,
                    "CUPTI_ACTIVITY_KIND_RUNTIME",
                    "nameId",
                    top=top,
                    window=window_value,
                ),
            "top_nvtx": [] if lite else _top_nvtx(cur, top=top, window=window_value),
            "nccl_kernel": _skipped_summary("lite mode")
            if lite
            else _name_filter_summary(
                    cur,
                    "CUPTI_ACTIVITY_KIND_KERNEL",
                    "demangledName",
                    "%nccl%",
                    window=window_value,
                ),
            "nccl_nvtx": _nvtx_filter_summary(cur, "%nccl%", window=window_value),
        }

    return {
        "sqlite_path": str(path),
        "tables": table_names,
        "requested_nvtx_window": nvtx_window,
        "lite": lite,
        "cuda_graph": _cuda_graph_summary(cur),
        **sections,
    }


def _fmt_s(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Nsight Summary: {Path(summary['sqlite_path']).name}")
    lines.append("")
    lines.append(f"- Lite mode: {summary.get('lite', False)}")
    lines.append("")
    cg = summary["cuda_graph"]
    lines.append("## CUDA Graph")
    lines.append("")
    lines.append(f"- CUDA_GRAPH_EVENTS present: {cg['cuda_graph_events_present']}")
    lines.append(f"- CUDA_GRAPH_EVENTS count: {cg['cuda_graph_events_count']}")
    lines.append(f"- kernel graphNodeId non-null count: {cg['kernel_graph_node_nonnull_count']}")
    if cg["event_names"]:
        lines.append("- graph event names: " + ", ".join(
            f"{row['name']}={row['count']}" for row in cg["event_names"][:8]
        ))
    lines.append("")
    for section_name in ("total", "nvtx_window"):
        section = summary[section_name]
        lines.append(f"## {section_name}")
        lines.append("")
        if section_name == "nvtx_window":
            lines.append(f"- window name: {section.get('window_name')}")
            lines.append(f"- window found: {section.get('window_found')}")
            lines.append("")
        lines.append("| Metric | Count | Duration s | Extra |")
        lines.append("| --- | ---: | ---: | --- |")
        lines.append(
            f"| kernels | {_fmt_s(section['kernel'].get('count'))} | "
            f"{_fmt_s(section['kernel'].get('duration_s'))} | |"
        )
        lines.append(
            f"| graph trace | {_fmt_s(section['graph_trace'].get('count'))} | "
            f"{_fmt_s(section['graph_trace'].get('duration_s'))} | |"
        )
        lines.append(
            f"| runtime | {_fmt_s(section['runtime'].get('count'))} | "
            f"{_fmt_s(section['runtime'].get('duration_s'))} | |"
        )
        lines.append(
            f"| memcpy | {_fmt_s(section['memcpy'].get('count'))} | "
            f"{_fmt_s(section['memcpy'].get('duration_s'))} | "
            f"bytes={_fmt_s(section['memcpy'].get('bytes'))} |"
        )
        lines.append(
            f"| NCCL kernels | {_fmt_s(section['nccl_kernel'].get('count'))} | "
            f"{_fmt_s(section['nccl_kernel'].get('duration_s'))} | |"
        )
        lines.append(
            f"| NCCL NVTX | {_fmt_s(section['nccl_nvtx'].get('count'))} | n/a | "
            f"range={section['nccl_nvtx'].get('start_ns')}..{section['nccl_nvtx'].get('end_ns')} |"
        )
        lines.append("")
        lines.append("Top kernels:")
        lines.append("")
        lines.append("| Name | Count | Duration s |")
        lines.append("| --- | ---: | ---: |")
        for row in section["top_kernels"][:10]:
            name = str(row["name"]).replace("|", "\\|")
            lines.append(f"| `{name[:160]}` | {row['count']} | {_fmt_s(row['duration_s'])} |")
        lines.append("")
        lines.append("Top runtime APIs:")
        lines.append("")
        lines.append("| Name | Count | Duration s |")
        lines.append("| --- | ---: | ---: |")
        for row in section["top_runtime"][:10]:
            name = str(row["name"]).replace("|", "\\|")
            lines.append(f"| `{name[:160]}` | {row['count']} | {_fmt_s(row['duration_s'])} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an Nsight Systems sqlite export.")
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--nvtx-window", default=None)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Skip expensive top-kernel/runtime grouping for very large sqlite exports.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_summary(
        args.sqlite_path,
        nvtx_window=args.nvtx_window,
        top=args.top,
        lite=args.lite,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md is not None:
        write_markdown(args.output_md, summary)
    if args.output_json is None and args.output_md is None:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Sequence


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (name,),
    ).fetchone()
    return row is not None


def table_columns(cur: sqlite3.Cursor, name: str) -> set[str]:
    if not table_exists(cur, name):
        return set()
    return {str(row[1]) for row in cur.execute(f"pragma table_info({name})")}


def _safe_fetchone(
    cur: sqlite3.Cursor, query: str, params: Sequence[Any] = ()
) -> Any:
    try:
        return cur.execute(query, tuple(params)).fetchone()
    except sqlite3.Error:
        return None


def _rows(
    cur: sqlite3.Cursor, query: str, params: Sequence[Any] = ()
) -> list[dict[str, Any]]:
    try:
        out = []
        cols = None
        for row in cur.execute(query, tuple(params)):
            if cols is None:
                cols = [item[0] for item in cur.description]
            out.append({key: row[idx] for idx, key in enumerate(cols)})
        return out
    except sqlite3.Error:
        return []


def _nvtx_ranges(cur: sqlite3.Cursor, name: str) -> list[tuple[int, int]]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return []
    rows = cur.execute(
        """
        select n.start, n.end
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where (n.text=? or s.value=?) and n.end is not null
        order by n.start
        """,
        (name, name),
    ).fetchall()
    return [(int(row[0]), int(row[1])) for row in rows]


def _inside_any(window: tuple[int, int], parents: Sequence[tuple[int, int]]) -> bool:
    start, end = window
    return any(start >= parent_start and end <= parent_end for parent_start, parent_end in parents)


def _select_ranges(
    cur: sqlite3.Cursor,
    name: str,
    parent_ranges: Sequence[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    ranges = _nvtx_ranges(cur, name)
    if parent_ranges:
        ranges = [window for window in ranges if _inside_any(window, parent_ranges)]
    return ranges


def _load_temp_ranges(cur: sqlite3.Cursor, ranges: Sequence[tuple[int, int]]) -> None:
    cur.execute("drop table if exists temp.selected_ranges")
    cur.execute("create temp table selected_ranges(start integer not null, end integer not null)")
    if ranges:
        cur.executemany(
            "insert into temp.selected_ranges(start, end) values (?, ?)",
            ranges,
        )


def _string_ids(cur: sqlite3.Cursor) -> dict[int, str]:
    if not table_exists(cur, "StringIds"):
        return {}
    return {int(row[0]): str(row[1]) for row in cur.execute("select id, value from StringIds")}


def _nvtx_named_ranges_with_prefixes(
    cur: sqlite3.Cursor,
    prefixes: Sequence[str],
) -> list[tuple[int, int, str]]:
    if not prefixes or not table_exists(cur, "NVTX_EVENTS"):
        return []
    rows: list[tuple[int, int, str]] = []
    for start, end, text, value in cur.execute(
        """
        select n.start, n.end, n.text, s.value
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where n.end is not null
        order by n.start
        """
    ):
        name = text or value
        if name is None:
            continue
        name = str(name)
        if any(name.startswith(prefix) for prefix in prefixes):
            rows.append((int(start), int(end), name))
    return rows


def _build_graph_node_capture_map(
    cur: sqlite3.Cursor,
    prefixes: Sequence[str],
) -> dict[int, str]:
    if not prefixes or not table_exists(cur, "CUDA_GRAPH_NODE_EVENTS"):
        return {}
    ranges = _nvtx_named_ranges_with_prefixes(cur, prefixes)
    if not ranges:
        return {}
    ranges.sort(key=lambda item: item[0])
    order_map = _build_graph_node_capture_order_map(cur, ranges)
    if order_map:
        return order_map

    active: list[tuple[int, int, str]] = []
    range_idx = 0
    out: dict[int, str] = {}
    query = """
        select start, graphNodeId
        from CUDA_GRAPH_NODE_EVENTS
        order by start
        """
    for start, graph_node_id in cur.execute(query):
        timestamp = int(start)
        while range_idx < len(ranges) and ranges[range_idx][0] <= timestamp:
            active.append(ranges[range_idx])
            range_idx += 1
        if active:
            active = [item for item in active if item[1] >= timestamp]
        if not active:
            continue
        best = min(active, key=lambda item: item[1] - item[0])
        out[int(graph_node_id)] = best[2]
    return out


def _runtime_rows_by_name(
    cur: sqlite3.Cursor,
    name: str,
) -> list[tuple[int, int]]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    return [
        (int(start), int(end))
        for start, end in cur.execute(
            """
            select rt.start, rt.end
            from CUPTI_ACTIVITY_KIND_RUNTIME rt
            left join StringIds s on s.id = rt.nameId
            where s.value = ?
            order by rt.start
            """,
            (name,),
        )
    ]


def _innermost_range_labels_for_events(
    ranges: Sequence[tuple[int, int, str]],
    events: Sequence[tuple[int, int]],
) -> list[str]:
    labels: list[str] = []
    active: list[tuple[int, int, str]] = []
    range_idx = 0
    for start, end in events:
        while range_idx < len(ranges) and ranges[range_idx][0] <= start:
            active.append(ranges[range_idx])
            range_idx += 1
        if active:
            active = [item for item in active if item[1] >= end]
        if not active:
            continue
        best = min(active, key=lambda item: item[1] - item[0])
        labels.append(best[2])
    return labels


def _build_graph_node_capture_order_map(
    cur: sqlite3.Cursor,
    ranges: Sequence[tuple[int, int, str]],
) -> dict[int, str]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME") or not table_exists(
        cur, "CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        return {}
    begins = _runtime_rows_by_name(cur, "cudaStreamBeginCapture_v10000")
    ends = _runtime_rows_by_name(cur, "cudaStreamEndCapture_v10000")
    if not begins or len(begins) != len(ends):
        return {}
    kernel_graph_node_ids = {
        int(row[0])
        for row in cur.execute(
            """
            select distinct graphNodeId
            from CUPTI_ACTIVITY_KIND_KERNEL
            where graphNodeId is not null
            """
        )
    }
    if not kernel_graph_node_ids:
        return {}

    out: dict[int, str] = {}
    for idx, (begin, end_capture) in enumerate(zip(begins, ends)):
        _begin_start, begin_end = begin
        end_start, end_end = end_capture
        launch_events = [
            (int(start), int(end))
            for start, end in cur.execute(
                """
                select rt.start, rt.end
                from CUPTI_ACTIVITY_KIND_RUNTIME rt
                left join StringIds s on s.id = rt.nameId
                where rt.start >= ? and rt.end <= ?
                  and s.value in (
                    'cudaLaunchKernel_v7000',
                    'cuLaunchKernel',
                    'cuLaunchKernelEx'
                  )
                order by rt.start
                """,
                (begin_end, end_start),
            )
        ]
        labels = _innermost_range_labels_for_events(ranges, launch_events)
        if not labels:
            continue
        next_begin_start = begins[idx + 1][0] if idx + 1 < len(begins) else 2**63 - 1
        graph_node_ids = [
            int(row[0])
            for row in cur.execute(
                """
                select graphNodeId
                from CUDA_GRAPH_NODE_EVENTS
                where start >= ? and start < ? and originalGraphNodeId is not null
                order by start
                """,
                (end_end, next_begin_start),
            )
            if int(row[0]) in kernel_graph_node_ids
        ]
        for graph_node_id, label in zip(graph_node_ids, labels):
            out[graph_node_id] = label
    return out


def _in_selected_range(
    ranges: Sequence[tuple[int, int]],
    starts: Sequence[int],
    start: int,
    end: int,
) -> bool:
    idx = bisect_right(starts, start) - 1
    return idx >= 0 and start >= ranges[idx][0] and end <= ranges[idx][1]


def _event_bounds(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def _scan_duration_count(
    cur: sqlite3.Cursor,
    table: str,
    ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    if not table_exists(cur, table):
        return {"present": False}
    bounds = _event_bounds(ranges)
    if bounds is None:
        return {"present": True, "count": 0, "duration_s": 0.0}
    range_starts = [item[0] for item in ranges]
    count = 0
    duration_ns = 0
    for start, end in cur.execute(
        f"select start, end from {table} where start >= ? and end <= ?",
        bounds,
    ):
        start = int(start)
        end = int(end)
        if not _in_selected_range(ranges, range_starts, start, end):
            continue
        count += 1
        duration_ns += end - start
    return {"present": True, "count": count, "duration_s": duration_ns / 1e9}


def _scan_memcpy_summary(
    cur: sqlite3.Cursor,
    ranges: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not table_exists(cur, table):
        return {"present": False}
    bounds = _event_bounds(ranges)
    if bounds is None:
        return {"present": True, "count": 0, "duration_s": 0.0, "bytes": 0}
    range_starts = [item[0] for item in ranges]
    count = 0
    duration_ns = 0
    total_bytes = 0
    for start, end, nbytes in cur.execute(
        f"select start, end, bytes from {table} where start >= ? and end <= ?",
        bounds,
    ):
        start = int(start)
        end = int(end)
        if not _in_selected_range(ranges, range_starts, start, end):
            continue
        count += 1
        duration_ns += end - start
        total_bytes += int(nbytes)
    return {
        "present": True,
        "count": count,
        "duration_s": duration_ns / 1e9,
        "bytes": total_bytes,
    }


def _top_rows(
    stats: dict[int | None, list[int]],
    string_ids: dict[int, str],
    *,
    top: int,
) -> list[dict[str, Any]]:
    rows = []
    for name_id, (count, duration_ns) in stats.items():
        name = string_ids.get(name_id, str(name_id))
        rows.append(
            {
                "name": name,
                "count": count,
                "duration_s": duration_ns / 1e9,
            }
        )
    rows.sort(key=lambda row: row["duration_s"], reverse=True)
    return rows[:top]


def _top_capture_rows(
    stats: dict[str, dict[str, Any]],
    *,
    top: int,
) -> list[dict[str, Any]]:
    rows = []
    for name, values in stats.items():
        graph_nodes = values.get("graph_nodes", set())
        rows.append(
            {
                "name": name,
                "count": int(values.get("count", 0)),
                "duration_s": int(values.get("duration_ns", 0)) / 1e9,
                "distinct_graph_nodes": len(graph_nodes),
            }
        )
    rows.sort(key=lambda row: row["duration_s"], reverse=True)
    return rows[:top]


_LAYER_RE = re.compile(r"^dsv4\.layer\d+\.")


def _capture_group_name(name: str) -> str:
    return _LAYER_RE.sub("dsv4.layer*.", name)


def _scan_kernel_summary(
    cur: sqlite3.Cursor,
    ranges: Sequence[tuple[int, int]],
    *,
    top: int,
    graph_node_capture_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if not table_exists(cur, table):
        empty = {"present": False}
        return {
            "kernel": empty,
            "kernel_graph_node_nonnull": empty,
            "nccl_kernel": empty,
            "top_kernels": [],
            "top_graph_node_kernels": [],
            "top_non_graph_node_kernels": [],
            "top_graph_node_capture_ranges": [],
            "top_graph_node_capture_groups": [],
        }
    bounds = _event_bounds(ranges)
    if bounds is None:
        zero = {"present": True, "count": 0, "duration_s": 0.0}
        return {
            "kernel": zero,
            "kernel_graph_node_nonnull": zero,
            "nccl_kernel": zero,
            "top_kernels": [],
            "top_graph_node_kernels": [],
            "top_non_graph_node_kernels": [],
            "top_graph_node_capture_ranges": [],
            "top_graph_node_capture_groups": [],
        }
    cols = table_columns(cur, table)
    graph_expr = "graphNodeId" if "graphNodeId" in cols else "null"
    range_starts = [item[0] for item in ranges]
    string_ids = _string_ids(cur)
    total_count = 0
    total_duration_ns = 0
    graph_count = 0
    graph_duration_ns = 0
    nccl_count = 0
    nccl_duration_ns = 0
    top_stats: dict[int | None, list[int]] = defaultdict(lambda: [0, 0])
    graph_stats: dict[int | None, list[int]] = defaultdict(lambda: [0, 0])
    non_graph_stats: dict[int | None, list[int]] = defaultdict(lambda: [0, 0])
    capture_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "duration_ns": 0, "graph_nodes": set()}
    )
    capture_group_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "duration_ns": 0, "graph_nodes": set()}
    )
    graph_node_capture_map = graph_node_capture_map or {}
    for start, end, name_id, graph_node_id in cur.execute(
        f"""
        select start, end, demangledName, {graph_expr}
        from {table}
        where start >= ? and end <= ?
        """,
        bounds,
    ):
        start = int(start)
        end = int(end)
        if not _in_selected_range(ranges, range_starts, start, end):
            continue
        name_id = int(name_id) if name_id is not None else None
        duration = end - start
        total_count += 1
        total_duration_ns += duration
        top_stats[name_id][0] += 1
        top_stats[name_id][1] += duration
        if graph_node_id is not None:
            graph_node_id = int(graph_node_id)
            graph_count += 1
            graph_duration_ns += duration
            graph_stats[name_id][0] += 1
            graph_stats[name_id][1] += duration
            capture_name = graph_node_capture_map.get(graph_node_id)
            if capture_name is not None:
                capture_stats[capture_name]["count"] += 1
                capture_stats[capture_name]["duration_ns"] += duration
                capture_stats[capture_name]["graph_nodes"].add(graph_node_id)
                capture_group = _capture_group_name(capture_name)
                capture_group_stats[capture_group]["count"] += 1
                capture_group_stats[capture_group]["duration_ns"] += duration
                capture_group_stats[capture_group]["graph_nodes"].add(graph_node_id)
        else:
            non_graph_stats[name_id][0] += 1
            non_graph_stats[name_id][1] += duration
        if "nccl" in string_ids.get(name_id, "").lower():
            nccl_count += 1
            nccl_duration_ns += duration
    return {
        "kernel": {
            "present": True,
            "count": total_count,
            "duration_s": total_duration_ns / 1e9,
        },
        "kernel_graph_node_nonnull": {
            "present": True,
            "count": graph_count,
            "duration_s": graph_duration_ns / 1e9,
        },
        "nccl_kernel": {
            "present": True,
            "count": nccl_count,
            "duration_s": nccl_duration_ns / 1e9,
        },
        "top_kernels": _top_rows(top_stats, string_ids, top=top),
        "top_graph_node_kernels": _top_rows(graph_stats, string_ids, top=top),
        "top_non_graph_node_kernels": _top_rows(non_graph_stats, string_ids, top=top),
        "top_graph_node_capture_ranges": _top_capture_rows(capture_stats, top=top),
        "top_graph_node_capture_groups": _top_capture_rows(
            capture_group_stats,
            top=top,
        ),
    }


def _scan_runtime_summary(
    cur: sqlite3.Cursor,
    ranges: Sequence[tuple[int, int]],
    *,
    top: int,
) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if not table_exists(cur, table):
        empty = {"present": False}
        return {
            "runtime": empty,
            "cuda_graph_launch": empty,
            "cuda_launch_kernel": empty,
            "top_runtime": [],
        }
    bounds = _event_bounds(ranges)
    if bounds is None:
        zero = {"present": True, "count": 0, "duration_s": 0.0}
        return {
            "runtime": zero,
            "cuda_graph_launch": zero,
            "cuda_launch_kernel": zero,
            "top_runtime": [],
        }
    range_starts = [item[0] for item in ranges]
    string_ids = _string_ids(cur)
    total_count = 0
    total_duration_ns = 0
    graph_launch_count = 0
    graph_launch_duration_ns = 0
    launch_kernel_count = 0
    launch_kernel_duration_ns = 0
    top_stats: dict[int | None, list[int]] = defaultdict(lambda: [0, 0])
    for start, end, name_id in cur.execute(
        f"select start, end, nameId from {table} where start >= ? and end <= ?",
        bounds,
    ):
        start = int(start)
        end = int(end)
        if not _in_selected_range(ranges, range_starts, start, end):
            continue
        name_id = int(name_id) if name_id is not None else None
        name = string_ids.get(name_id, "")
        duration = end - start
        total_count += 1
        total_duration_ns += duration
        top_stats[name_id][0] += 1
        top_stats[name_id][1] += duration
        if name == "cudaGraphLaunch_v10000":
            graph_launch_count += 1
            graph_launch_duration_ns += duration
        if name == "cudaLaunchKernel_v7000":
            launch_kernel_count += 1
            launch_kernel_duration_ns += duration
    return {
        "runtime": {
            "present": True,
            "count": total_count,
            "duration_s": total_duration_ns / 1e9,
        },
        "cuda_graph_launch": {
            "present": True,
            "count": graph_launch_count,
            "duration_s": graph_launch_duration_ns / 1e9,
        },
        "cuda_launch_kernel": {
            "present": True,
            "count": launch_kernel_count,
            "duration_s": launch_kernel_duration_ns / 1e9,
        },
        "top_runtime": _top_rows(top_stats, string_ids, top=top),
    }


def _summarize_ranges_scan(
    cur: sqlite3.Cursor,
    name: str,
    ranges: Sequence[tuple[int, int]],
    *,
    top: int,
    graph_node_capture_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    duration_ns = sum(end - start for start, end in ranges)
    kernel = _scan_kernel_summary(
        cur,
        ranges,
        top=top,
        graph_node_capture_map=graph_node_capture_map,
    )
    runtime = _scan_runtime_summary(cur, ranges, top=top)
    return {
        "name": name,
        "range_count": len(ranges),
        "total_range_duration_s": duration_ns / 1e9,
        "first_range_ns": list(ranges[0]) if ranges else None,
        "last_range_ns": list(ranges[-1]) if ranges else None,
        "kernel": kernel["kernel"],
        "graph_trace": _scan_duration_count(
            cur, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", ranges
        ),
        "cuda_graph_node_event": _scan_duration_count(
            cur, "CUDA_GRAPH_NODE_EVENTS", ranges
        ),
        "kernel_graph_node_nonnull": kernel["kernel_graph_node_nonnull"],
        "runtime": runtime["runtime"],
        "memcpy": _scan_memcpy_summary(cur, ranges),
        "memset": _scan_duration_count(cur, "CUPTI_ACTIVITY_KIND_MEMSET", ranges),
        "sync": _scan_duration_count(
            cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", ranges
        ),
        "nccl_kernel": kernel["nccl_kernel"],
        "cuda_graph_launch": runtime["cuda_graph_launch"],
        "cuda_launch_kernel": runtime["cuda_launch_kernel"],
        "top_kernels": kernel["top_kernels"],
        "top_graph_node_kernels": kernel["top_graph_node_kernels"],
        "top_non_graph_node_kernels": kernel["top_non_graph_node_kernels"],
        "top_graph_node_capture_ranges": kernel["top_graph_node_capture_ranges"],
        "top_graph_node_capture_groups": kernel["top_graph_node_capture_groups"],
        "top_runtime": runtime["top_runtime"],
    }


def _duration_count(cur: sqlite3.Cursor, table: str) -> dict[str, Any]:
    if not table_exists(cur, table):
        return {"present": False}
    row = _safe_fetchone(
        cur,
        f"""
        select count(*), coalesce(sum(e.end - e.start), 0)
        from {table} e
        join temp.selected_ranges r on e.start >= r.start and e.end <= r.end
        """,
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {
        "present": True,
        "count": int(row[0]),
        "duration_s": float(row[1]) / 1e9,
    }


def _memcpy_summary(cur: sqlite3.Cursor) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not table_exists(cur, table):
        return {"present": False}
    row = _safe_fetchone(
        cur,
        f"""
        select count(*), coalesce(sum(e.end - e.start), 0), coalesce(sum(e.bytes), 0)
        from {table} e
        join temp.selected_ranges r on e.start >= r.start and e.end <= r.end
        """,
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None, "bytes": None}
    return {
        "present": True,
        "count": int(row[0]),
        "duration_s": float(row[1]) / 1e9,
        "bytes": int(row[2]),
    }


def _top_named_events(
    cur: sqlite3.Cursor,
    table: str,
    name_col: str,
    *,
    top: int,
    where: str = "",
) -> list[dict[str, Any]]:
    if not table_exists(cur, table) or not table_exists(cur, "StringIds"):
        return []
    extra_where = f"and ({where})" if where else ""
    return _rows(
        cur,
        f"""
        select
          coalesce(s.value, cast(e.{name_col} as text)) as name,
          count(*) as count,
          coalesce(sum(e.end - e.start), 0) / 1000000000.0 as duration_s
        from {table} e
        join temp.selected_ranges r on e.start >= r.start and e.end <= r.end
        left join StringIds s on s.id = e.{name_col}
        where 1=1 {extra_where}
        group by e.{name_col}
        order by sum(e.end - e.start) desc
        limit ?
        """,
        (top,),
    )


def _name_filter_summary(
    cur: sqlite3.Cursor,
    table: str,
    name_col: str,
    pattern: str,
) -> dict[str, Any]:
    if not table_exists(cur, table) or not table_exists(cur, "StringIds"):
        return {"present": table_exists(cur, table), "count": 0, "duration_s": 0.0}
    row = _safe_fetchone(
        cur,
        f"""
        select count(*), coalesce(sum(e.end - e.start), 0) / 1000000000.0
        from {table} e
        join temp.selected_ranges r on e.start >= r.start and e.end <= r.end
        left join StringIds s on s.id = e.{name_col}
        where lower(coalesce(s.value, '')) like ?
        """,
        (pattern.lower(),),
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {"present": True, "count": int(row[0]), "duration_s": float(row[1])}


def _runtime_name_summary(cur: sqlite3.Cursor, name: str) -> dict[str, Any]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME") or not table_exists(
        cur, "StringIds"
    ):
        return {"present": table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"), "count": 0}
    row = _safe_fetchone(
        cur,
        """
        select count(*), coalesce(sum(e.end - e.start), 0) / 1000000000.0
        from CUPTI_ACTIVITY_KIND_RUNTIME e
        join temp.selected_ranges r on e.start >= r.start and e.end <= r.end
        left join StringIds s on s.id = e.nameId
        where s.value = ?
        """,
        (name,),
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {"present": True, "count": int(row[0]), "duration_s": float(row[1])}


def _kernel_graph_node_summary(cur: sqlite3.Cursor) -> dict[str, Any]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return {"present": False}
    row = _safe_fetchone(
        cur,
        """
        select count(*), coalesce(sum(e.end - e.start), 0)
        from CUPTI_ACTIVITY_KIND_KERNEL e
        join temp.selected_ranges r on e.start >= r.start and e.end <= r.end
        where e.graphNodeId is not null
        """,
    )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {"present": True, "count": int(row[0]), "duration_s": float(row[1]) / 1e9}


def _summarize_ranges(
    cur: sqlite3.Cursor,
    name: str,
    ranges: Sequence[tuple[int, int]],
    *,
    top: int,
) -> dict[str, Any]:
    _load_temp_ranges(cur, ranges)
    duration_ns = sum(end - start for start, end in ranges)
    return {
        "name": name,
        "range_count": len(ranges),
        "total_range_duration_s": duration_ns / 1e9,
        "first_range_ns": list(ranges[0]) if ranges else None,
        "last_range_ns": list(ranges[-1]) if ranges else None,
        "kernel": _duration_count(cur, "CUPTI_ACTIVITY_KIND_KERNEL"),
        "graph_trace": _duration_count(cur, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"),
        "kernel_graph_node_nonnull": _kernel_graph_node_summary(cur),
        "runtime": _duration_count(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"),
        "memcpy": _memcpy_summary(cur),
        "memset": _duration_count(cur, "CUPTI_ACTIVITY_KIND_MEMSET"),
        "sync": _duration_count(cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"),
        "nccl_kernel": _name_filter_summary(
            cur, "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName", "%nccl%"
        ),
        "cuda_graph_launch": _runtime_name_summary(cur, "cudaGraphLaunch_v10000"),
        "cuda_launch_kernel": _runtime_name_summary(cur, "cudaLaunchKernel_v7000"),
        "top_kernels": _top_named_events(
            cur, "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName", top=top
        ),
        "top_graph_node_kernels": _top_named_events(
            cur,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "demangledName",
            top=top,
            where="e.graphNodeId is not null",
        ),
        "top_non_graph_node_kernels": _top_named_events(
            cur,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "demangledName",
            top=top,
            where="e.graphNodeId is null",
        ),
        "top_runtime": _top_named_events(
            cur, "CUPTI_ACTIVITY_KIND_RUNTIME", "nameId", top=top
        ),
    }


def build_summary(
    sqlite_path: Path,
    *,
    names: Sequence[str],
    parent_name: str | None,
    top: int,
    scan_events: bool = False,
    graph_node_capture_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    parent_ranges = _nvtx_ranges(cur, parent_name) if parent_name else []
    graph_node_capture_map = (
        _build_graph_node_capture_map(cur, graph_node_capture_prefixes)
        if scan_events and graph_node_capture_prefixes
        else {}
    )
    sections = []
    for name in names:
        ranges = _select_ranges(cur, name, parent_ranges if parent_name else None)
        if scan_events:
            sections.append(
                _summarize_ranges_scan(
                    cur,
                    name,
                    ranges,
                    top=top,
                    graph_node_capture_map=graph_node_capture_map,
                )
            )
        else:
            sections.append(_summarize_ranges(cur, name, ranges, top=top))
    return {
        "sqlite_path": str(sqlite_path),
        "parent_name": parent_name,
        "parent_range_count": len(parent_ranges),
        "event_summary_mode": "scan" if scan_events else "sql_join",
        "graph_node_capture_prefixes": list(graph_node_capture_prefixes),
        "graph_node_capture_map_count": len(graph_node_capture_map),
        "parent_ranges_ns": [list(item) for item in parent_ranges],
        "ranges": sections,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Nsight NVTX Range Summary: {Path(summary['sqlite_path']).name}")
    lines.append("")
    lines.append(f"- Parent NVTX: {summary.get('parent_name')}")
    lines.append(f"- Parent range count: {summary.get('parent_range_count')}")
    lines.append(f"- Event summary mode: {summary.get('event_summary_mode')}")
    prefixes = summary.get("graph_node_capture_prefixes") or []
    if prefixes:
        lines.append(f"- Graph-node capture prefixes: {prefixes}")
        lines.append(
            f"- Graph-node capture map count: "
            f"{summary.get('graph_node_capture_map_count')}"
        )
    lines.append("")
    for section in summary["ranges"]:
        lines.append(f"## {section['name']}")
        lines.append("")
        lines.append(f"- range count: {section['range_count']}")
        lines.append(f"- total range duration s: {_fmt(section['total_range_duration_s'])}")
        lines.append(
            f"- kernel count: {_fmt(section['kernel'].get('count'))}, "
            f"duration s: {_fmt(section['kernel'].get('duration_s'))}"
        )
        lines.append(
            f"- graph trace count: {_fmt(section['graph_trace'].get('count'))}, "
            f"duration s: {_fmt(section['graph_trace'].get('duration_s'))}"
        )
        graph_node_event = section.get("cuda_graph_node_event")
        if graph_node_event:
            lines.append(
                f"- CUDA graph node event count: "
                f"{_fmt(graph_node_event.get('count'))}, "
                f"duration s: {_fmt(graph_node_event.get('duration_s'))}"
            )
        lines.append(
            f"- kernel graphNodeId non-null count: "
            f"{_fmt(section['kernel_graph_node_nonnull'].get('count'))}, "
            f"duration s: {_fmt(section['kernel_graph_node_nonnull'].get('duration_s'))}"
        )
        lines.append(
            f"- runtime count: {_fmt(section['runtime'].get('count'))}, "
            f"duration s: {_fmt(section['runtime'].get('duration_s'))}"
        )
        lines.append(
            f"- NCCL kernel count: {_fmt(section['nccl_kernel'].get('count'))}, "
            f"duration s: {_fmt(section['nccl_kernel'].get('duration_s'))}"
        )
        lines.append(
            f"- cudaGraphLaunch count: "
            f"{_fmt(section['cuda_graph_launch'].get('count'))}"
        )
        lines.append(
            f"- cudaLaunchKernel count: "
            f"{_fmt(section['cuda_launch_kernel'].get('count'))}"
        )
        lines.append(
            f"- memcpy count: {_fmt(section['memcpy'].get('count'))}, "
            f"bytes: {_fmt(section['memcpy'].get('bytes'))}"
        )
        if section["top_kernels"]:
            lines.append("- top kernels:")
            for row in section["top_kernels"][:8]:
                lines.append(
                    f"  - {row['name']}: count={row['count']}, "
                    f"duration_s={_fmt(row['duration_s'])}"
                )
        if section["top_graph_node_kernels"]:
            lines.append("- top graph-node kernels:")
            for row in section["top_graph_node_kernels"][:8]:
                lines.append(
                    f"  - {row['name']}: count={row['count']}, "
                    f"duration_s={_fmt(row['duration_s'])}"
                )
        if section["top_non_graph_node_kernels"]:
            lines.append("- top non-graph-node kernels:")
            for row in section["top_non_graph_node_kernels"][:8]:
                lines.append(
                    f"  - {row['name']}: count={row['count']}, "
                    f"duration_s={_fmt(row['duration_s'])}"
                )
        if section.get("top_graph_node_capture_ranges"):
            lines.append("- top graph-node capture groups:")
            for row in section.get("top_graph_node_capture_groups", [])[:8]:
                lines.append(
                    f"  - {row['name']}: count={row['count']}, "
                    f"duration_s={_fmt(row['duration_s'])}, "
                    f"distinct_graph_nodes={row['distinct_graph_nodes']}"
                )
            lines.append("- top graph-node capture ranges:")
            for row in section["top_graph_node_capture_ranges"][:8]:
                lines.append(
                    f"  - {row['name']}: count={row['count']}, "
                    f"duration_s={_fmt(row['duration_s'])}, "
                    f"distinct_graph_nodes={row['distinct_graph_nodes']}"
                )
        if section["top_runtime"]:
            lines.append("- top runtime:")
            for row in section["top_runtime"][:8]:
                lines.append(
                    f"  - {row['name']}: count={row['count']}, "
                    f"duration_s={_fmt(row['duration_s'])}"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--parent-nvtx")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--scan-events",
        action="store_true",
        help="Scan event rows and match them to NVTX ranges in Python. This is slower "
        "than indexed SQL for small traces but avoids large range-join explosions.",
    )
    parser.add_argument(
        "--graph-node-capture-prefix",
        action="append",
        default=[],
        help="When used with --scan-events, map CUDA graph node creation events to "
        "the innermost capture-time NVTX range whose name starts with this prefix, "
        "then aggregate replay kernels by that mapped range. Can be passed more "
        "than once.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = build_summary(
        args.sqlite,
        names=args.name,
        parent_name=args.parent_nvtx,
        top=args.top,
        scan_events=args.scan_events,
        graph_node_capture_prefixes=args.graph_node_capture_prefix,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, summary)


if __name__ == "__main__":
    main()

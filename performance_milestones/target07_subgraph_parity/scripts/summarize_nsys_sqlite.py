#!/usr/bin/env python3
"""Small Nsight Systems SQLite summarizer for target07_subgraph_parity."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def exists(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "select count(*) from sqlite_master where type='table' and name=?", (table,)
        ).fetchone()[0]
        > 0
    )


def rows(con: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cur = con.execute(query, params)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def summarize(path: Path) -> dict[str, Any]:
    con = sqlite3.connect(path)
    out: dict[str, Any] = {"sqlite_path": str(path), "exists": path.exists()}
    if exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        out["kernel_total"] = rows(
            con,
            """
            select count(*) as count,
                   sum(end-start)/1e9 as duration_s,
                   min(start) as first_start_ns,
                   max(end) as last_end_ns,
                   count(distinct streamId) as stream_count,
                   count(distinct deviceId) as device_count
            from CUPTI_ACTIVITY_KIND_KERNEL
            """,
        )[0]
        out["top_streams"] = rows(
            con,
            """
            select streamId,
                   count(*) as count,
                   sum(end-start)/1e9 as duration_s,
                   (max(end)-min(start))/1e9 as span_s,
                   count(distinct deviceId) as device_count
            from CUPTI_ACTIVITY_KIND_KERNEL
            group by streamId
            order by duration_s desc
            limit 32
            """,
        )
        out["top_kernels"] = rows(
            con,
            """
            select s.value as name,
                   count(*) as count,
                   sum(k.end-k.start)/1e9 as duration_s,
                   count(distinct k.streamId) as stream_count,
                   count(distinct k.deviceId) as device_count
            from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on s.id = k.demangledName
            group by s.value
            order by duration_s desc
            limit 80
            """,
        )
    if exists(con, "NVTX_EVENTS"):
        out["top_nvtx_ranges"] = rows(
            con,
            """
            select coalesce(n.text, s.value) as name,
                   count(*) as count,
                   sum(n.end-n.start)/1e9 as duration_s
            from NVTX_EVENTS n
            left join StringIds s on s.id = n.textId
            where n.end is not null
            group by name
            order by duration_s desc
            limit 80
            """,
        )
    if exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        out["top_runtime"] = rows(
            con,
            """
            select s.value as name,
                   count(*) as count,
                   sum(r.end-r.start)/1e9 as duration_s
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            join StringIds s on s.id = r.nameId
            group by s.value
            order by duration_s desc
            limit 50
            """,
        )
    graph_table = "CUDA_GRAPH_NODE_EVENTS" if exists(con, "CUDA_GRAPH_NODE_EVENTS") else "CUDA_GRAPH_EVENTS"
    if exists(con, graph_table):
        out["cuda_graph_events"] = rows(
            con,
            f"""
            select count(*) as count,
                   min(start) as first_start_ns,
                   max(end) as last_end_ns
            from {graph_table}
            """,
        )[0]
        out["cuda_graph_table"] = graph_table
    con.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.sqlite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

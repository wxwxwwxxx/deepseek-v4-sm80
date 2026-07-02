#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence


NS = 1_000_000_000.0

SUB_BOUNDARY_ORDER = (
    "direct_copy",
    "index_elementwise_kernel",
    "CatArrayBatchedCopy",
    "gatherTopK",
    "arange_index_helper",
    "topk_lens_swa_compressed_index_assembly",
    "other_metadata_copy_cat_index",
)


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (name,),
    ).fetchone()
    return row is not None


def rows(
    cur: sqlite3.Cursor,
    query: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for row in cur.execute(query, tuple(params)):
        cols = [item[0] for item in cur.description]
        values.append({key: row[idx] for idx, key in enumerate(cols)})
    return values


def nvtx_name_expr(alias: str = "n") -> str:
    return f"coalesce({alias}.text, s.value)"


def find_nvtx_ranges(
    cur: sqlite3.Cursor,
    *,
    exact: str | None = None,
    like: str | None = None,
    parent: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return []
    if exact is None and like is None:
        raise ValueError("exact or like is required")
    predicate = f"{nvtx_name_expr()} = ?"
    params: list[Any] = [exact]
    if like is not None:
        predicate = f"{nvtx_name_expr()} like ?"
        params = [like]
    parent_clause = ""
    if parent is not None:
        parent_clause = " and n.start>=? and n.end<=?"
        params.extend([parent[0], parent[1]])
    out = rows(
        cur,
        f"""
        select n.start as start, n.end as end
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where {predicate} and n.end is not null{parent_clause}
        order by n.start
        """,
        params,
    )
    return [(int(row["start"]), int(row["end"])) for row in out]


def largest_range(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return max(ranges, key=lambda item: item[1] - item[0])


def envelope(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def classify_sub_boundary(name: str) -> str | None:
    lowered = name.lower()
    if "gathertopk" in lowered or "bitonicsortkv" in lowered or "persistent_topk" in lowered:
        return "gatherTopK"
    if "catarraybatchedcopy" in lowered:
        return "CatArrayBatchedCopy"
    if "index_elementwise" in lowered or "gpu_index_kernel" in lowered:
        return "index_elementwise_kernel"
    if (
        "arange_cuda" in lowered
        or "vectorized_gather" in lowered
        or "_scatter_gather" in lowered
        or "indexselect" in lowered
        or "masked_select" in lowered
        or "nonzero" in lowered
        or "where" in lowered
    ):
        return "arange_index_helper"
    if (
        "topk_transform" in lowered
        or "global_topk" in lowered
        or "_copy_decode_metadata_for_replay" in lowered
        or "_copy_masked_compressed_locs" in lowered
        or "_pad_indices" in lowered
        or "fillfunctor" in lowered
        or "deviceselect" in lowered
        or "devicescan" in lowered
        or "devicecompact" in lowered
        or "deviceradixsort" in lowered
    ):
        return "topk_lens_swa_compressed_index_assembly"
    if (
        "direct_copy" in lowered
        or "copy_kernel" in lowered
        or "bfloat16_copy" in lowered
        or "float8_copy" in lowered
    ):
        return "direct_copy"
    if (
        "copy" in lowered
        or "index" in lowered
        or "gather" in lowered
        or "cat" in lowered
    ):
        return "other_metadata_copy_cat_index"
    return None


def kernel_rows(
    cur: sqlite3.Cursor,
    range_: tuple[int, int],
) -> list[dict[str, Any]]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    return rows(
        cur,
        f"""
        select
          coalesce(s.value, cast(k.demangledName as text)) as name,
          count(*) as count,
          coalesce(sum(k.end-k.start), 0) / {NS} as duration_s,
          sum(case when k.graphNodeId is not null then 1 else 0 end) as graph_count,
          count(distinct k.graphNodeId) as graph_node_count
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.start>=? and k.end<=?
        group by k.demangledName
        order by sum(k.end-k.start) desc
        """,
        range_,
    )


def owner_ranges(cur: sqlite3.Cursor, decode_range: tuple[int, int]) -> list[dict[str, Any]]:
    return rows(
        cur,
        f"""
        select
          {nvtx_name_expr()} as name,
          n.start as start,
          n.end as end,
          (n.end-n.start) / {NS} as duration_s
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where n.end is not null
          and n.start>=? and n.end<=?
          and (
            {nvtx_name_expr()} like 'batch_prepare:decode:%'
            or {nvtx_name_expr()} like 'batch_forward_enqueue:decode:%'
            or {nvtx_name_expr()} like 'batch_forward:decode:%'
            or {nvtx_name_expr()} like 'dsv4.%'
          )
        order by n.start
        """,
        decode_range,
    )


def innermost_owner(
    owners: Sequence[dict[str, Any]],
    start: int,
    end: int,
) -> str | None:
    candidates = [
        owner
        for owner in owners
        if int(owner["start"]) <= start and int(owner["end"]) >= end
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: int(item["end"]) - int(item["start"]))
    return str(best["name"])


def owner_breakdown(
    cur: sqlite3.Cursor,
    decode_range: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return {}
    owners = owner_ranges(cur, decode_range)
    if not owners:
        return {}
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "duration_s": 0.0,
            "sub_boundaries": defaultdict(lambda: {"count": 0, "duration_s": 0.0}),
        }
    )
    events = rows(
        cur,
        f"""
        select
          coalesce(s.value, cast(k.demangledName as text)) as name,
          k.start as start,
          k.end as end,
          (k.end-k.start) / {NS} as duration_s
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.start>=? and k.end<=?
        """,
        decode_range,
    )
    for event in events:
        sub = classify_sub_boundary(str(event["name"]))
        if sub is None:
            continue
        owner = innermost_owner(owners, int(event["start"]), int(event["end"]))
        if owner is None:
            owner = "unowned_decode_envelope"
        out[owner]["count"] += 1
        out[owner]["duration_s"] += float(event["duration_s"])
        out[owner]["sub_boundaries"][sub]["count"] += 1
        out[owner]["sub_boundaries"][sub]["duration_s"] += float(event["duration_s"])
    rendered: dict[str, dict[str, Any]] = {}
    for owner, values in out.items():
        rendered[owner] = {
            "count": values["count"],
            "duration_s": values["duration_s"],
            "sub_boundaries": dict(values["sub_boundaries"]),
        }
    return dict(sorted(rendered.items(), key=lambda item: item[1]["duration_s"], reverse=True))


def build_summary(sqlite_path: Path, repeat_nvtx: str) -> dict[str, Any]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    repeat = largest_range(find_nvtx_ranges(cur, exact=repeat_nvtx))
    if repeat is None:
        raise RuntimeError(f"repeat NVTX range not found: {repeat_nvtx}")
    decode_ranges = find_nvtx_ranges(cur, like="batch_forward:decode:%", parent=repeat)
    decode_envelope = envelope(decode_ranges)
    if decode_envelope is None:
        raise RuntimeError("decode forward envelope not found under repeat range")

    sub: dict[str, dict[str, Any]] = {
        name: {"count": 0, "duration_s": 0.0, "graph_count": 0, "graph_node_count": 0}
        for name in SUB_BOUNDARY_ORDER
    }
    top_by_sub: dict[str, list[dict[str, Any]]] = {name: [] for name in SUB_BOUNDARY_ORDER}
    total_selected = {"count": 0, "duration_s": 0.0, "graph_count": 0, "graph_node_count": 0}
    for kernel in kernel_rows(cur, decode_envelope):
        sub_name = classify_sub_boundary(str(kernel["name"]))
        if sub_name is None:
            continue
        count = int(kernel["count"])
        duration_s = float(kernel["duration_s"])
        graph_count = int(kernel["graph_count"] or 0)
        graph_node_count = int(kernel["graph_node_count"] or 0)
        sub[sub_name]["count"] += count
        sub[sub_name]["duration_s"] += duration_s
        sub[sub_name]["graph_count"] += graph_count
        sub[sub_name]["graph_node_count"] += graph_node_count
        total_selected["count"] += count
        total_selected["duration_s"] += duration_s
        total_selected["graph_count"] += graph_count
        total_selected["graph_node_count"] += graph_node_count
        top_by_sub[sub_name].append(
            {
                "name": kernel["name"],
                "count": count,
                "duration_s": duration_s,
                "graph_count": graph_count,
                "graph_node_count": graph_node_count,
            }
        )
    for kernels in top_by_sub.values():
        kernels.sort(key=lambda item: item["duration_s"], reverse=True)
        del kernels[10:]

    decode_wall_s = (decode_envelope[1] - decode_envelope[0]) / NS
    for values in sub.values():
        values["share_of_decode_envelope_wall"] = (
            values["duration_s"] / decode_wall_s if decode_wall_s else None
        )
    total_selected["share_of_decode_envelope_wall"] = (
        total_selected["duration_s"] / decode_wall_s if decode_wall_s else None
    )
    payload = {
        "sqlite_path": str(sqlite_path),
        "repeat_nvtx": repeat_nvtx,
        "repeat_range": repeat,
        "decode_forward_count": len(decode_ranges),
        "decode_envelope": decode_envelope,
        "decode_envelope_wall_s": decode_wall_s,
        "sub_boundaries": sub,
        "total_selected_metadata_adjacent": total_selected,
        "top_kernels_by_sub_boundary": top_by_sub,
        "owner_breakdown": owner_breakdown(cur, decode_envelope),
        "notes": [
            "Sub-boundary split intentionally includes gatherTopK/topk_transform as adjacent topk-lens metadata even when the 07.63 coarse classifier placed gatherTopK in fp8_indexer.",
            "The 07.63 gate bucket graph_runtime_copy_cat_index is the subset excluding gatherTopK/topk_transform kernels classified elsewhere by the older script.",
        ],
    }
    con.close()
    return payload


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Decode Metadata Nsight Split: {Path(summary['sqlite_path']).name}",
        "",
        f"- repeat NVTX: `{summary['repeat_nvtx']}`",
        f"- decode forward ranges: `{summary['decode_forward_count']}`",
        f"- decode envelope wall s: `{summary['decode_envelope_wall_s']:.6f}`",
        "",
        "## Sub-Boundaries",
        "",
        "| Sub-boundary | Kernel s | Count | Graph events | Graph nodes | Share of decode envelope |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in SUB_BOUNDARY_ORDER:
        values = summary["sub_boundaries"][name]
        lines.append(
            f"| `{name}` | `{values['duration_s']:.6f}` | {values['count']} | "
            f"{values['graph_count']} | {values['graph_node_count']} | "
            f"{values['share_of_decode_envelope_wall']:.2%} |"
        )
    total = summary["total_selected_metadata_adjacent"]
    lines.extend(
        [
            f"| `total_selected_metadata_adjacent` | `{total['duration_s']:.6f}` | "
            f"{total['count']} | {total['graph_count']} | {total['graph_node_count']} | "
            f"{total['share_of_decode_envelope_wall']:.2%} |",
            "",
            "## Top Kernels By Sub-Boundary",
            "",
        ]
    )
    for name in SUB_BOUNDARY_ORDER:
        kernels = summary["top_kernels_by_sub_boundary"][name]
        if not kernels:
            continue
        lines.extend(
            [
                f"### `{name}`",
                "",
                "| Kernel | Kernel s | Count | Graph events | Graph nodes |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for kernel in kernels:
            kernel_name = str(kernel["name"])
            if len(kernel_name) > 160:
                kernel_name = kernel_name[:157] + "..."
            lines.append(
                f"| `{kernel_name}` | `{kernel['duration_s']:.6f}` | {kernel['count']} | "
                f"{kernel['graph_count']} | {kernel['graph_node_count']} |"
            )
        lines.append("")

    owners = summary["owner_breakdown"]
    if owners:
        lines.extend(
            [
                "## NVTX Owner Breakdown",
                "",
                "| Owner | Kernel s | Count | Dominant sub-boundaries |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for owner, values in list(owners.items())[:30]:
            sub_items = sorted(
                values["sub_boundaries"].items(),
                key=lambda item: item[1]["duration_s"],
                reverse=True,
            )
            dom = ", ".join(
                f"{sub_name}={sub_values['duration_s']:.4f}s"
                for sub_name, sub_values in sub_items[:3]
            )
            lines.append(
                f"| `{owner}` | `{values['duration_s']:.6f}` | {values['count']} | {dom} |"
            )
        lines.append("")

    lines.extend(["## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.sqlite, args.repeat_nvtx)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(summary) + "\n")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()

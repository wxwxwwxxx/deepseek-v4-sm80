#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence


NS = 1_000_000_000.0


KERNEL_CATEGORY_ORDER = (
    "legacy_prefill_sparse_attention",
    "decode_splitk_gather_split_combine",
    "indexer_logits_topk_cache",
    "runtime_copy_cat_index_kernels",
    "fp8_projection_gemm",
    "moe_marlin_route",
    "hc_rmsnorm_logits_sampling",
    "dense_linear_other",
    "nccl_communication",
    "elementwise_math_other",
    "other",
)

RUNTIME_CATEGORY_ORDER = (
    "sync_wait_runtime",
    "cuda_graph_launch_runtime",
    "kernel_launch_runtime",
    "memcpy_runtime",
    "allocation_runtime",
    "module_runtime",
    "other",
)


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (name,),
    ).fetchone()
    return row is not None


def safe_rows(
    cur: sqlite3.Cursor,
    query: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        cols: list[str] | None = None
        for row in cur.execute(query, tuple(params)):
            if cols is None:
                cols = [item[0] for item in cur.description]
            rows.append({key: row[idx] for idx, key in enumerate(cols)})
        return rows
    except sqlite3.Error:
        return []


def safe_one(
    cur: sqlite3.Cursor,
    query: str,
    params: Sequence[Any] = (),
) -> tuple[Any, ...] | None:
    try:
        return cur.execute(query, tuple(params)).fetchone()
    except sqlite3.Error:
        return None


def event_name_expr(alias: str, name_col: str) -> str:
    return f"coalesce(s.value, cast({alias}.{name_col} as text))"


def find_nvtx_ranges(
    cur: sqlite3.Cursor,
    *,
    name: str,
    parent: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return []
    parent_clause = ""
    params: list[Any] = [name, name]
    if parent is not None:
        parent_clause = " and n.start>=? and n.end<=?"
        params.extend([parent[0], parent[1]])
    rows = safe_rows(
        cur,
        f"""
        select n.start as start, n.end as end
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where (n.text=? or s.value=?) and n.end is not null{parent_clause}
        order by n.start
        """,
        params,
    )
    return [(int(row["start"]), int(row["end"])) for row in rows]


def largest_range(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return max(ranges, key=lambda item: item[1] - item[0])


def range_duration_s(ranges: Sequence[tuple[int, int]]) -> float:
    return sum(max(0, end - start) for start, end in ranges) / NS


def envelope_range(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def ranges_cte(ranges: Sequence[tuple[int, int]]) -> tuple[str, list[Any]]:
    if not ranges:
        return "(select 0 as start, -1 as end)", []
    values = ",".join(["(?, ?)"] * len(ranges))
    params: list[Any] = []
    for start, end in ranges:
        params.extend([start, end])
    return f"(values {values})", params


def named_events(
    cur: sqlite3.Cursor,
    *,
    table: str,
    name_col: str,
    ranges: Sequence[tuple[int, int]] | None,
    top: int | None = None,
) -> list[dict[str, Any]]:
    if not table_exists(cur, table) or not table_exists(cur, "StringIds"):
        return []
    limit = "" if top is None else f" limit {int(top)}"
    if ranges is None:
        if table == "CUPTI_ACTIVITY_KIND_KERNEL":
            graph_cols = (
                ", sum(case when e.graphNodeId is not null then 1 else 0 end) as graph_count"
                ", count(distinct e.graphNodeId) as graph_node_count"
            )
        else:
            graph_cols = ", 0 as graph_count, 0 as graph_node_count"
        return safe_rows(
            cur,
            f"""
            select
              {event_name_expr("e", name_col)} as name,
              count(*) as count,
              coalesce(sum(e.end - e.start), 0) / {NS} as duration_s
              {graph_cols}
            from {table} e
            left join StringIds s on s.id = e.{name_col}
            group by e.{name_col}
            order by sum(e.end - e.start) desc
            {limit}
            """,
        )

    cte, params = ranges_cte(ranges)
    graph_cols = ""
    if table == "CUPTI_ACTIVITY_KIND_KERNEL":
        graph_cols = (
            ", sum(case when e.graphNodeId is not null then 1 else 0 end) as graph_count"
            ", count(distinct e.graphNodeId) as graph_node_count"
        )
    else:
        graph_cols = ", 0 as graph_count, 0 as graph_node_count"
    return safe_rows(
        cur,
        f"""
        with ranges(start, end) as {cte}
        select
          {event_name_expr("e", name_col)} as name,
          count(*) as count,
          coalesce(sum(e.end - e.start), 0) / {NS} as duration_s
          {graph_cols}
        from {table} e
        join ranges r on e.start>=r.start and e.end<=r.end
        left join StringIds s on s.id = e.{name_col}
        group by e.{name_col}
        order by sum(e.end - e.start) desc
        {limit}
        """,
        params,
    )


def duration_count(
    cur: sqlite3.Cursor,
    *,
    table: str,
    ranges: Sequence[tuple[int, int]] | None,
) -> dict[str, Any]:
    if not table_exists(cur, table):
        return {"present": False, "count": 0, "duration_s": 0.0}
    if ranges is None:
        row = safe_one(
            cur,
            f"select count(*), coalesce(sum(end-start), 0) / {NS} from {table}",
        )
    else:
        cte, params = ranges_cte(ranges)
        row = safe_one(
            cur,
            f"""
            with ranges(start, end) as {cte}
            select count(*), coalesce(sum(e.end-e.start), 0) / {NS}
            from {table} e
            join ranges r on e.start>=r.start and e.end<=r.end
            """,
            params,
        )
    if row is None:
        return {"present": True, "count": None, "duration_s": None}
    return {"present": True, "count": int(row[0]), "duration_s": float(row[1])}


def memcpy_summary(
    cur: sqlite3.Cursor,
    *,
    ranges: Sequence[tuple[int, int]] | None,
) -> dict[str, Any]:
    table = "CUPTI_ACTIVITY_KIND_MEMCPY"
    if not table_exists(cur, table):
        return {"present": False, "count": 0, "duration_s": 0.0, "bytes": 0}
    if ranges is None:
        row = safe_one(
            cur,
            f"select count(*), coalesce(sum(end-start), 0) / {NS}, coalesce(sum(bytes), 0) from {table}",
        )
    else:
        cte, params = ranges_cte(ranges)
        row = safe_one(
            cur,
            f"""
            with ranges(start, end) as {cte}
            select count(*), coalesce(sum(e.end-e.start), 0) / {NS}, coalesce(sum(e.bytes), 0)
            from {table} e
            join ranges r on e.start>=r.start and e.end<=r.end
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


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "sparse_attention_kernel" in lowered:
        return "legacy_prefill_sparse_attention"
    if (
        "_sparse_bf16_gather_with_mask_kernel" in lowered
        or "_sparse_splitk_bf16_split_kernel" in lowered
        or "_sparse_splitk_bf16_combine_kernel" in lowered
    ):
        return "decode_splitk_gather_split_combine"
    if "nccl" in lowered:
        return "nccl_communication"
    if "marlin_moe_wna16" in lowered or "moe_route" in lowered or "gptq_marlin_repack" in lowered:
        return "moe_marlin_route"
    if "_quantized_linear_fp8_kernel" in lowered:
        return "fp8_projection_gemm"
    if (
        "_indexer_bf16_logits_kernel" in lowered
        or "topk_transform" in lowered
        or "global_topk" in lowered
        or "gathertopk" in lowered
        or "bitonicsortkv" in lowered
        or "_q_kv_norm_rope_cache" in lowered
        or "_compress_norm_rope_store" in lowered
        or "store_cache" in lowered
        or "store_indexer" in lowered
        or "masked_locs" in lowered
        or "indexer" in lowered
    ):
        return "indexer_logits_topk_cache"
    if (
        "direct_copy" in lowered
        or "copy_kernel" in lowered
        or "bfloat16_copy" in lowered
        or "float8_copy" in lowered
        or "catarraybatchedcopy" in lowered
        or "index_elementwise" in lowered
        or "vectorized_gather" in lowered
        or "_scatter_gather" in lowered
        or "arange_cuda" in lowered
        or "fillfunctor" in lowered
        or "deviceselect" in lowered
        or "devicescan" in lowered
        or "devicecompact" in lowered
        or "deviceradixsort" in lowered
    ):
        return "runtime_copy_cat_index_kernels"
    if (
        "_hc_" in lowered
        or "rms_norm" in lowered
        or "rmsnorm" in lowered
        or "lm_head" in lowered
        or "logits" in lowered
        or "softmax" in lowered
        or "sampler" in lowered
        or "sampling" in lowered
        or "argmax" in lowered
    ):
        return "hc_rmsnorm_logits_sampling"
    if (
        "gemm" in lowered
        or "cutlass" in lowered
        or "cublas" in lowered
        or "ampere_bf16" in lowered
        or "ampere_sgemm" in lowered
    ):
        return "dense_linear_other"
    if (
        "reduce_kernel" in lowered
        or "elementwise_kernel" in lowered
        or "vectorized_elementwise" in lowered
        or "pow_" in lowered
        or "clamp" in lowered
        or "rsqrt" in lowered
        or "log2" in lowered
        or "ceil" in lowered
        or "silu" in lowered
        or "softplus" in lowered
        or "mulfunctor" in lowered
        or "divfunctor" in lowered
    ):
        return "elementwise_math_other"
    return "other"


def classify_runtime(name: str) -> str:
    lowered = name.lower()
    if "synchronize" in lowered or "event" in lowered:
        return "sync_wait_runtime"
    if "graphlaunch" in lowered or "graph" in lowered:
        return "cuda_graph_launch_runtime"
    if "launch" in lowered:
        return "kernel_launch_runtime"
    if "memcpy" in lowered:
        return "memcpy_runtime"
    if "malloc" in lowered or "free" in lowered or "hostalloc" in lowered or "memalloc" in lowered:
        return "allocation_runtime"
    if "module" in lowered:
        return "module_runtime"
    return "other"


def aggregate_categories(
    events: Iterable[dict[str, Any]],
    *,
    classifier,
    order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        item: {"count": 0, "duration_s": 0.0, "graph_count": 0, "graph_node_count": 0}
        for item in order
    }
    node_sets: dict[str, set[Any]] = defaultdict(set)
    for row in events:
        name = str(row.get("name") or "")
        category = classifier(name)
        if category not in out:
            out[category] = {
                "count": 0,
                "duration_s": 0.0,
                "graph_count": 0,
                "graph_node_count": 0,
            }
        out[category]["count"] += int(row.get("count") or 0)
        out[category]["duration_s"] += float(row.get("duration_s") or 0.0)
        out[category]["graph_count"] += int(row.get("graph_count") or 0)
        graph_nodes = row.get("graph_nodes")
        if graph_nodes:
            node_sets[category].update(graph_nodes)
        else:
            out[category]["graph_node_count"] += int(row.get("graph_node_count") or 0)
    for category, nodes in node_sets.items():
        out[category]["graph_node_count"] = len(nodes)
    return {
        category: values
        for category, values in out.items()
        if values["count"] or values["duration_s"]
    }


def kernel_events_with_graph_nodes(
    cur: sqlite3.Cursor,
    *,
    ranges: Sequence[tuple[int, int]] | None,
    top: int | None = None,
) -> list[dict[str, Any]]:
    events = named_events(
        cur,
        table="CUPTI_ACTIVITY_KIND_KERNEL",
        name_col="demangledName",
        ranges=ranges,
        top=top,
    )
    if not events or top is not None:
        return events
    # Preserve exact graph-node cardinality per category for the summary.
    # The grouped event rows have only per-name distinct node counts; collect
    # node sets for names that have graph nodes without materializing kernels.
    return events


def build_section(
    cur: sqlite3.Cursor,
    *,
    label: str,
    ranges: Sequence[tuple[int, int]] | None,
    top: int,
) -> dict[str, Any]:
    kernels = kernel_events_with_graph_nodes(cur, ranges=ranges, top=None)
    runtime = named_events(
        cur,
        table="CUPTI_ACTIVITY_KIND_RUNTIME",
        name_col="nameId",
        ranges=ranges,
        top=None,
    )
    kernel_total = {
        "present": table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"),
        "count": sum(int(row.get("count") or 0) for row in kernels),
        "duration_s": sum(float(row.get("duration_s") or 0.0) for row in kernels),
    }
    runtime_total = {
        "present": table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"),
        "count": sum(int(row.get("count") or 0) for row in runtime),
        "duration_s": sum(float(row.get("duration_s") or 0.0) for row in runtime),
    }
    return {
        "label": label,
        "range_count": None if ranges is None else len(ranges),
        "wall_s_sum": None if ranges is None else range_duration_s(ranges),
        "kernel": kernel_total,
        "runtime": runtime_total,
        "memcpy": memcpy_summary(cur, ranges=ranges),
        "memset": duration_count(cur, table="CUPTI_ACTIVITY_KIND_MEMSET", ranges=ranges),
        "sync": duration_count(cur, table="CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", ranges=ranges),
        "kernel_categories": aggregate_categories(
            kernels,
            classifier=classify_kernel,
            order=KERNEL_CATEGORY_ORDER,
        ),
        "runtime_categories": aggregate_categories(
            runtime,
            classifier=classify_runtime,
            order=RUNTIME_CATEGORY_ORDER,
        ),
        "top_kernels": sorted(
            kernels,
            key=lambda row: float(row.get("duration_s") or 0.0),
            reverse=True,
        )[:top],
        "top_runtime": sorted(
            runtime,
            key=lambda row: float(row.get("duration_s") or 0.0),
            reverse=True,
        )[:top],
    }


def build_summary(sqlite_path: Path, *, repeat_nvtx: str, top: int) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    repeat = largest_range(find_nvtx_ranges(cur, name=repeat_nvtx))
    repeat_ranges = [] if repeat is None else [repeat]
    prefill_forward = find_nvtx_ranges(cur, name="batch_forward:prefill:bs4:padded4", parent=repeat)
    decode_forward = find_nvtx_ranges(cur, name="batch_forward:decode:bs4:padded4", parent=repeat)
    decode_envelope = envelope_range(decode_forward)
    decode_envelope_ranges = [] if decode_envelope is None else [decode_envelope]

    sections = {
        "total": build_section(cur, label="total", ranges=None, top=top),
        "repeat": build_section(cur, label="repeat", ranges=repeat_ranges, top=top),
        "repeat_prefill_forward": build_section(
            cur,
            label="repeat_prefill_forward",
            ranges=prefill_forward,
            top=top,
        ),
        "repeat_decode_forward_envelope": build_section(
            cur,
            label="repeat_decode_forward_envelope",
            ranges=decode_envelope_ranges,
            top=top,
        ),
    }
    tables = {
        table: table_exists(cur, table)
        for table in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "NVTX_EVENTS",
        )
    }
    return {
        "sqlite_path": str(sqlite_path),
        "repeat_nvtx": repeat_nvtx,
        "repeat_range": repeat,
        "nvtx_range_counts": {
            "repeat_prefill_forward": len(prefill_forward),
            "repeat_decode_forward": len(decode_forward),
            "repeat_decode_forward_wall_s_sum": range_duration_s(decode_forward),
            "repeat_decode_forward_envelope": decode_envelope,
            "repeat_decode_forward_envelope_wall_s": (
                None if decode_envelope is None else range_duration_s([decode_envelope])
            ),
        },
        "tables": tables,
        "sections": sections,
    }


def pct(value: float, total: float | None) -> str:
    if not total:
        return "n/a"
    return f"{100.0 * value / total:.2f}%"


def fmt_s(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def render_category_table(
    categories: dict[str, dict[str, Any]],
    *,
    total_s: float | None,
    order: Sequence[str],
    runtime: bool = False,
) -> list[str]:
    rows = [
        "| Category | Count | Duration s | Share | Graph events | Graph nodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    sorted_items = sorted(
        categories.items(),
        key=lambda item: float(item[1].get("duration_s") or 0.0),
        reverse=True,
    )
    ordered = [item for category in order for item in sorted_items if item[0] == category]
    ordered.extend(item for item in sorted_items if item[0] not in order)
    for category, values in ordered:
        duration = float(values.get("duration_s") or 0.0)
        rows.append(
            "| `{}` | {} | `{}` | {} | {} | {} |".format(
                category,
                int(values.get("count") or 0),
                fmt_s(duration),
                pct(duration, total_s),
                int(values.get("graph_count") or 0),
                int(values.get("graph_node_count") or 0),
            )
        )
    if runtime:
        rows[0] = "| Category | Count | Duration s | Share | Graph events | Graph nodes |"
    return rows


def render_top_table(rows_in: Sequence[dict[str, Any]]) -> list[str]:
    rows = [
        "| Name | Count | Duration s | Graph events | Graph nodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_in:
        name = str(row.get("name") or "").replace("|", "\\|")
        if len(name) > 140:
            name = name[:137] + "..."
        rows.append(
            "| `{}` | {} | `{}` | {} | {} |".format(
                name,
                int(row.get("count") or 0),
                fmt_s(row.get("duration_s")),
                int(row.get("graph_count") or 0),
                int(row.get("graph_node_count") or 0),
            )
        )
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    path = Path(summary["sqlite_path"]).name
    lines.append(f"# Post-SplitK Nsight Classification: {path}")
    lines.append("")
    lines.append(f"- Requested repeat NVTX: `{summary['repeat_nvtx']}`")
    lines.append(f"- Repeat range found: `{summary['repeat_range'] is not None}`")
    counts = summary.get("nvtx_range_counts", {})
    if counts:
        lines.append(
            "- Repeat child ranges: prefill_forward={}, decode_forward={}, "
            "decode_forward_sum_s=`{}`, decode_envelope_s=`{}`".format(
                counts.get("repeat_prefill_forward"),
                counts.get("repeat_decode_forward"),
                fmt_s(counts.get("repeat_decode_forward_wall_s_sum")),
                fmt_s(counts.get("repeat_decode_forward_envelope_wall_s")),
            )
        )
    lines.append("- Tables:")
    for table, present in summary["tables"].items():
        lines.append(f"  - `{table}`: `{present}`")
    lines.append("")

    for label, section in summary["sections"].items():
        kernel_total = section["kernel"].get("duration_s")
        runtime_total = section["runtime"].get("duration_s")
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "| Metric | Count | Duration s | Extra |"
        )
        lines.append("| --- | ---: | ---: | --- |")
        lines.append(
            "| NVTX wall sum | {} | `{}` | ranges={} |".format(
                section["range_count"] if section["range_count"] is not None else "n/a",
                fmt_s(section["wall_s_sum"]),
                section["range_count"],
            )
        )
        for key in ("kernel", "runtime", "memcpy", "memset", "sync"):
            item = section[key]
            extra = ""
            if key == "memcpy":
                extra = f"bytes={item.get('bytes')}"
            lines.append(
                "| {} | {} | `{}` | {} |".format(
                    key,
                    item.get("count"),
                    fmt_s(item.get("duration_s")),
                    extra,
                )
            )
        lines.append("")
        lines.append("Kernel categories:")
        lines.append("")
        lines.extend(
            render_category_table(
                section["kernel_categories"],
                total_s=kernel_total,
                order=KERNEL_CATEGORY_ORDER,
            )
        )
        lines.append("")
        lines.append("Runtime categories:")
        lines.append("")
        lines.extend(
            render_category_table(
                section["runtime_categories"],
                total_s=runtime_total,
                order=RUNTIME_CATEGORY_ORDER,
                runtime=True,
            )
        )
        lines.append("")
        lines.append("Top kernels:")
        lines.append("")
        lines.extend(render_top_table(section["top_kernels"]))
        lines.append("")
        lines.append("Top runtime APIs:")
        lines.append("")
        lines.extend(render_top_table(section["top_runtime"]))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    summary = build_summary(args.sqlite, repeat_nvtx=args.repeat_nvtx, top=args.top)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(summary) + "\n")


if __name__ == "__main__":
    main()

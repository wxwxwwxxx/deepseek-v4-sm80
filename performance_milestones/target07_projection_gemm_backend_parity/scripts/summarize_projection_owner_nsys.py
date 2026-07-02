#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Sequence


NS = 1_000_000_000.0


OWNER_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "attn.q_proj_wqa_wkv",
        r"^dsv4\.layer\d+\.attn\.(q_proj|kv_proj)$",
        "FP8 WQA/WKV projection; active fwqakvcache path may dequantize cached BF16 weights and run F.linear.",
        "keep if fused WQA/WKV owns >=0.50s; otherwise use as context.",
    ),
    (
        "attn.q_wqb",
        r"^dsv4\.layer\d+\.attn\.q_wqb$",
        "DSV4Linear ColumnParallel FP8: quantize_fp8_activation_ref/_fp8_activation_quantize_kernel + _quantized_linear_fp8_kernel.",
        "keep if >=0.50s; compare against vLLM lifted wq_b ColumnParallelLinear.",
    ),
    (
        "attn.wo_a",
        r"^dsv4\.layer\d+\.attn\.wo_a$",
        "Grouped output projection: wo_a_grouped_projection_fp8 when enabled, otherwise dequant/einsum fallback.",
        "keep if >=0.50s; compare against vLLM SM80 wo_a BMM/reference and fp8_einsum boundary.",
    ),
    (
        "attn.wo_b",
        r"^dsv4\.layer\d+\.attn\.wo_b$",
        "DSV4Linear RowParallel FP8: _quantized_linear_fp8_kernel plus row-parallel all-reduce.",
        "keep if >=0.50s; compare against vLLM RowParallelLinear quant path.",
    ),
    (
        "indexer.wq_b",
        r"^dsv4\.indexer\.wq_b$",
        "Indexer query projection DSV4Linear FP8: _quantized_linear_fp8_kernel plus activation quant.",
        "keep if >=0.50s; otherwise context for FP8 indexer cache path.",
    ),
    (
        "indexer.weights_proj",
        r"^dsv4\.indexer\.weights_proj$",
        "Indexer weights/logits projection: BF16 F.linear/sgemm plus scale multiply.",
        "keep only if BF16 projection dominates.",
    ),
    (
        "indexer.compressor",
        r"^dsv4\.indexer\.compressor$",
        "Indexer compressor projection/norm/cache-adjacent work.",
        "context; not the primary projection owner unless it dominates.",
    ),
    (
        "shared_experts.gate_up_proj",
        r"^dsv4\.shared_experts\.gate_up_proj$",
        "Shared expert FP8 gate/up projection through DSV4Linear.",
        "keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated.",
    ),
    (
        "shared_experts.down_proj",
        r"^dsv4\.shared_experts\.down_proj$",
        "Shared expert FP8 down projection through DSV4Linear plus optional all-reduce.",
        "keep if shared expert projection is >=0.50s and not already MoE/Marlin dominated.",
    ),
    (
        "mlp.routed_experts",
        r"^dsv4\.layer\d+\.mlp\.runner\.experts$",
        "Routed expert backend, usually Marlin WNA16 in the active variant.",
        "context; out of scope unless projection attribution shows shared/routed FFN dominates.",
    ),
    (
        "lm_head",
        r"^dsv4\.lm_head$",
        "Vocab-parallel output linear: BF16/FP32 F.linear plus all-gather.",
        "context for decode envelope; not a projection backend PoC unless dominant.",
    ),
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


def envelope_range(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def range_duration_s(ranges: Sequence[tuple[int, int]]) -> float:
    return sum(max(0, end - start) for start, end in ranges) / NS


def ranges_cte(ranges: Sequence[tuple[int, int]]) -> tuple[str, list[Any]]:
    if not ranges:
        return "(select 0 as start, -1 as end)", []
    values = ",".join(["(?, ?)"] * len(ranges))
    params: list[Any] = []
    for start, end in ranges:
        params.extend([start, end])
    return f"(values {values})", params


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "_fp8_activation_quantize_kernel" in lowered or "scaled_fp8_quant" in lowered:
        return "activation_quant"
    if (
        "_quantized_linear_fp8_kernel" in lowered
        or "_quantized_linear_fp4_kernel" in lowered
        or "_wo_a_grouped_projection_fp8_kernel" in lowered
        or "gemm" in lowered
        or "cutlass" in lowered
        or "cublas" in lowered
        or "ampere_bf16" in lowered
        or "ampere_sgemm" in lowered
        or "aten::bmm" in lowered
    ):
        return "intrinsic_gemm"
    if "nccl" in lowered:
        return "communication"
    if "marlin_moe_wna16" in lowered or "moe_route" in lowered or "gptq_marlin_repack" in lowered:
        return "moe_marlin"
    if (
        "_indexer_fp8" in lowered
        or "topk_transform" in lowered
        or "global_topk" in lowered
        or "gathertopk" in lowered
        or "bitonicsortkv" in lowered
        or "persistent_topk" in lowered
    ):
        return "indexer_cache_topk"
    if (
        "_sparse_bf16_gather_with_mask_kernel" in lowered
        or "_sparse_splitk_bf16_split_kernel" in lowered
        or "_sparse_splitk_bf16_combine_kernel" in lowered
        or "sparse_attention_kernel" in lowered
    ):
        return "sparse_attention"
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
        return "wrapper_copy_layout"
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
        return "sampling_logits_norm"
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
        or "absfunctor" in lowered
    ):
        return "elementwise_scale_math"
    return "other"


def projection_bucket_kernel(name: str) -> bool:
    return classify_kernel(name) == "intrinsic_gemm"


def nvtx_events(cur: sqlite3.Cursor, pattern: re.Pattern[str]) -> list[tuple[int, int, str]]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return []
    rows = safe_rows(
        cur,
        """
        select n.start as start, n.end as end, coalesce(s.value, n.text) as name
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where n.end is not null and coalesce(s.value, n.text) like 'dsv4.%'
        order by n.start
        """,
    )
    out: list[tuple[int, int, str]] = []
    for row in rows:
        name = str(row["name"] or "")
        if pattern.match(name):
            out.append((int(row["start"]), int(row["end"]), name))
    return out


def graph_nodes_in_ranges(
    cur: sqlite3.Cursor,
    ranges: Sequence[tuple[int, int]],
) -> set[int]:
    if not ranges or not table_exists(cur, "CUDA_GRAPH_NODE_EVENTS"):
        return set()
    cte, params = ranges_cte(ranges)
    rows = safe_rows(
        cur,
        f"""
        with ranges(start, end) as {cte}
        select distinct c.graphNodeId as graphNodeId
        from CUDA_GRAPH_NODE_EVENTS c
        join ranges r on c.start>=r.start and c.end<=r.end
        """,
        params,
    )
    captured_or_original = {
        int(row["graphNodeId"]) for row in rows if row["graphNodeId"] is not None
    }
    if not captured_or_original:
        return set()

    # Node creation inside Python/NVTX capture ranges often records the
    # original graph node id. Replay kernels use the instantiated graph node id,
    # with CUDA_GRAPH_NODE_EVENTS.originalGraphNodeId linking back to the
    # capture-time id. Include both sides so capture owner ranges map to replay.
    placeholders = ",".join(["?"] * len(captured_or_original))
    mapped_rows = safe_rows(
        cur,
        f"""
        select distinct c.graphNodeId as graphNodeId
        from CUDA_GRAPH_NODE_EVENTS c
        where c.originalGraphNodeId in ({placeholders})
           or c.graphNodeId in ({placeholders})
        """,
        [*captured_or_original, *captured_or_original],
    )
    mapped = {int(row["graphNodeId"]) for row in mapped_rows if row["graphNodeId"] is not None}
    return captured_or_original | mapped


def kernel_rows_in_range(
    cur: sqlite3.Cursor,
    range_: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if range_ is None or not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    rows = safe_rows(
        cur,
        f"""
        select
          k.start as start,
          k.end as end,
          k.graphNodeId as graphNodeId,
          {event_name_expr("k", "demangledName")} as name
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.start>=? and k.end<=?
        """,
        [range_[0], range_[1]],
    )
    for row in rows:
        row["duration_s"] = (int(row["end"]) - int(row["start"])) / NS
        if row["graphNodeId"] is not None:
            row["graphNodeId"] = int(row["graphNodeId"])
    return rows


def aggregate_kernel_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    categories: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "duration_s": 0.0, "graph_nodes": set()}
    )
    top_by_name: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "duration_s": 0.0, "graph_nodes": set()}
    )
    graph_nodes: set[int] = set()
    for row in rows:
        name = str(row.get("name") or "")
        duration = float(row.get("duration_s") or 0.0)
        node = row.get("graphNodeId")
        category = classify_kernel(name)
        categories[category]["count"] += 1
        categories[category]["duration_s"] += duration
        top_by_name[name]["count"] += 1
        top_by_name[name]["duration_s"] += duration
        if node is not None:
            categories[category]["graph_nodes"].add(int(node))
            top_by_name[name]["graph_nodes"].add(int(node))
            graph_nodes.add(int(node))

    categories_out = {
        category: {
            "count": values["count"],
            "duration_s": values["duration_s"],
            "graph_node_count": len(values["graph_nodes"]),
        }
        for category, values in sorted(
            categories.items(),
            key=lambda item: item[1]["duration_s"],
            reverse=True,
        )
    }
    top = [
        {
            "name": name,
            "count": values["count"],
            "duration_s": values["duration_s"],
            "graph_node_count": len(values["graph_nodes"]),
        }
        for name, values in sorted(
            top_by_name.items(),
            key=lambda item: item[1]["duration_s"],
            reverse=True,
        )
    ]
    return {
        "count": len(rows),
        "duration_s": sum(float(row.get("duration_s") or 0.0) for row in rows),
        "graph_node_count": len(graph_nodes),
        "categories": categories_out,
        "top_kernels": top,
    }


def summarize_owner(
    *,
    owner: str,
    pattern: str,
    backend_contract: str,
    decision_hint: str,
    ranges: list[tuple[int, int, str]],
    nodes: set[int],
    decode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    node_rows = [
        row for row in decode_rows if row.get("graphNodeId") is not None and row["graphNodeId"] in nodes
    ]
    aggregate = aggregate_kernel_rows(node_rows)
    categories = aggregate["categories"]
    intrinsic = float(categories.get("intrinsic_gemm", {}).get("duration_s") or 0.0)
    activation = float(categories.get("activation_quant", {}).get("duration_s") or 0.0)
    copy_layout = float(categories.get("wrapper_copy_layout", {}).get("duration_s") or 0.0)
    elementwise = float(categories.get("elementwise_scale_math", {}).get("duration_s") or 0.0)
    runtime_copy = activation + copy_layout + elementwise
    return {
        "owner": owner,
        "pattern": pattern,
        "nvtx_range_count": len(ranges),
        "nvtx_unique_names": sorted(Counter(name for _, _, name in ranges)),
        "capture_graph_node_count": len(nodes),
        "decode_kernel_count": aggregate["count"],
        "decode_kernel_s": aggregate["duration_s"],
        "intrinsic_gemm_s": intrinsic,
        "activation_quant_s": activation,
        "wrapper_copy_layout_s": copy_layout,
        "elementwise_scale_math_s": elementwise,
        "runtime_copy_s": runtime_copy,
        "graph_node_count": aggregate["graph_node_count"],
        "categories": categories,
        "top_kernels": aggregate["top_kernels"][:12],
        "backend_contract": backend_contract,
        "decision_hint": decision_hint,
    }


def fmt_s(value: Any) -> str:
    return f"{float(value):.6f}"


def fmt_short(value: Any) -> str:
    return f"{float(value):.4f}"


def short_kernel_name(name: str, max_len: int = 72) -> str:
    name = name.replace("|", "\\|")
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Projection/GEMM Owner Attribution: {Path(summary['sqlite_path']).name}")
    lines.append("")
    lines.append(f"- Requested repeat NVTX: `{summary['repeat_nvtx']}`")
    lines.append(f"- Repeat range found: `{summary['repeat_range'] is not None}`")
    lines.append(
        "- Decode forward ranges: `{}`; decode envelope s: `{}`".format(
            summary["decode_forward_count"],
            fmt_s(summary["decode_envelope_s"]) if summary["decode_envelope_s"] else "n/a",
        )
    )
    lines.append(f"- DSV4 owner NVTX ranges found: `{summary['dsv4_nvtx_range_count']}`")
    lines.append(
        "- Decode projection/GEMM intrinsic bucket from kernel names: `{}` s, `{}` graph nodes".format(
            fmt_s(summary["decode_projection_gemm_s"]),
            summary["decode_projection_gemm_graph_nodes"],
        )
    )
    lines.append(
        "- Owner-attributed projection/GEMM intrinsic: `{}` s; unattributed intrinsic: `{}` s".format(
            fmt_s(summary["owner_attributed_intrinsic_gemm_s"]),
            fmt_s(summary["unattributed_intrinsic_gemm_s"]),
        )
    )
    lines.append("")
    lines.append("## Owner Table")
    lines.append("")
    lines.append(
        "| Owner | Kernel s | Runtime/copy s | Graph nodes | Top kernels | Backend contract | Keep/Pivot |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | --- | --- |")
    owners = sorted(
        summary["owners"],
        key=lambda item: (
            float(item.get("intrinsic_gemm_s") or 0.0),
            float(item.get("decode_kernel_s") or 0.0),
        ),
        reverse=True,
    )
    for item in owners:
        top = ", ".join(
            f"`{short_kernel_name(row['name'], 48)}` {fmt_short(row['duration_s'])}s"
            for row in item["top_kernels"][:3]
        )
        keep = item["decision_hint"]
        if float(item.get("intrinsic_gemm_s") or 0.0) >= 0.50:
            keep = "primary candidate: passes 0.50s owner gate"
        elif float(item.get("decode_kernel_s") or 0.0) >= 0.50:
            keep = "large owner, but verify intrinsic vs staging"
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {} | {} | {} |".format(
                item["owner"],
                fmt_s(item["intrinsic_gemm_s"]),
                fmt_s(item["runtime_copy_s"]),
                item["graph_node_count"],
                top or "n/a",
                item["backend_contract"],
                keep,
            )
        )
    lines.append("")
    lines.append("## Owner Details")
    for item in owners:
        lines.append("")
        lines.append(f"### `{item['owner']}`")
        lines.append("")
        lines.append(
            "- NVTX ranges: `{}`; capture graph nodes: `{}`; replay graph nodes: `{}`".format(
                item["nvtx_range_count"],
                item["capture_graph_node_count"],
                item["graph_node_count"],
            )
        )
        lines.append(
            "- Replay kernel total: `{}` s; intrinsic GEMM: `{}` s; activation quant: `{}` s; copy/layout: `{}` s; elementwise/scale: `{}` s".format(
                fmt_s(item["decode_kernel_s"]),
                fmt_s(item["intrinsic_gemm_s"]),
                fmt_s(item["activation_quant_s"]),
                fmt_s(item["wrapper_copy_layout_s"]),
                fmt_s(item["elementwise_scale_math_s"]),
            )
        )
        lines.append("")
        lines.append("| Category | Count | Duration s | Graph nodes |")
        lines.append("| --- | ---: | ---: | ---: |")
        for category, values in item["categories"].items():
            lines.append(
                "| `{}` | {} | `{}` | {} |".format(
                    category,
                    values["count"],
                    fmt_s(values["duration_s"]),
                    values["graph_node_count"],
                )
            )
        lines.append("")
        lines.append("| Top kernel | Count | Duration s | Graph nodes |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in item["top_kernels"][:8]:
            lines.append(
                "| `{}` | {} | `{}` | {} |".format(
                    short_kernel_name(row["name"]),
                    row["count"],
                    fmt_s(row["duration_s"]),
                    row["graph_node_count"],
                )
            )
    return "\n".join(lines)


def build_summary(sqlite_path: Path, *, repeat_nvtx: str) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cur = con.cursor()
    repeat = largest_range(find_nvtx_ranges(cur, name=repeat_nvtx))
    decode_forward = find_nvtx_ranges(cur, name="batch_forward:decode:bs4:padded4", parent=repeat)
    decode_envelope = envelope_range(decode_forward)
    decode_rows = kernel_rows_in_range(cur, decode_envelope)

    projection_rows = [row for row in decode_rows if projection_bucket_kernel(str(row.get("name") or ""))]
    projection_nodes = {
        int(row["graphNodeId"])
        for row in projection_rows
        if row.get("graphNodeId") is not None
    }
    projection_s = sum(float(row.get("duration_s") or 0.0) for row in projection_rows)

    owner_summaries: list[dict[str, Any]] = []
    owner_nodes_union: set[int] = set()
    for owner, pattern, backend, decision_hint in OWNER_SPECS:
        compiled = re.compile(pattern)
        ranges = nvtx_events(cur, compiled)
        range_bounds = [(start, end) for start, end, _ in ranges]
        nodes = graph_nodes_in_ranges(cur, range_bounds)
        owner_nodes_union.update(nodes)
        owner_summaries.append(
            summarize_owner(
                owner=owner,
                pattern=pattern,
                backend_contract=backend,
                decision_hint=decision_hint,
                ranges=ranges,
                nodes=nodes,
                decode_rows=decode_rows,
            )
        )

    attributed_projection_rows = [
        row
        for row in projection_rows
        if row.get("graphNodeId") is not None and int(row["graphNodeId"]) in owner_nodes_union
    ]
    attributed_projection_s = sum(float(row.get("duration_s") or 0.0) for row in attributed_projection_rows)

    dsv4_nvtx_count_row = safe_one(
        cur,
        """
        select count(*)
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where n.end is not null and coalesce(s.value, n.text) like 'dsv4.%'
        """,
    )
    return {
        "sqlite_path": str(sqlite_path),
        "repeat_nvtx": repeat_nvtx,
        "repeat_range": repeat,
        "decode_forward_count": len(decode_forward),
        "decode_forward_wall_s_sum": range_duration_s(decode_forward),
        "decode_envelope": decode_envelope,
        "decode_envelope_s": None if decode_envelope is None else range_duration_s([decode_envelope]),
        "decode_kernel_count": len(decode_rows),
        "decode_projection_gemm_s": projection_s,
        "decode_projection_gemm_graph_nodes": len(projection_nodes),
        "owner_attributed_intrinsic_gemm_s": attributed_projection_s,
        "unattributed_intrinsic_gemm_s": max(0.0, projection_s - attributed_projection_s),
        "dsv4_nvtx_range_count": int(dsv4_nvtx_count_row[0]) if dsv4_nvtx_count_row else 0,
        "owners": owner_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.sqlite, repeat_nvtx=args.repeat_nvtx)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(summary) + "\n")


if __name__ == "__main__":
    main()

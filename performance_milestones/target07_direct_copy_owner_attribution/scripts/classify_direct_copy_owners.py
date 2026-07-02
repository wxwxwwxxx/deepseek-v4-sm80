#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence

NS = 1_000_000_000.0
DIRECT_NVTX_PREFIX = "dsv4.direct_copy."


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


def largest_range(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return max(ranges, key=lambda item: item[1] - item[0])


def envelope(ranges: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def find_nvtx_ranges(
    cur: sqlite3.Cursor,
    *,
    exact: str | None = None,
    like: str | None = None,
    parent: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
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


def is_direct_copy_kernel(name: str) -> bool:
    lowered = name.lower()
    return (
        "direct_copy_kernel" in lowered
        or "bfloat16_copy_kernel" in lowered
        or "float8_copy" in lowered
    )


def normalize_owner(label: str | None) -> str:
    if not label:
        return "unattributed"
    owner = label.split("|", 1)[0]
    if owner.startswith(DIRECT_NVTX_PREFIX):
        owner = owner[len(DIRECT_NVTX_PREFIX) :]
    owner = re.sub(r"\blayer\d+\b", "layer*", owner)
    owner = re.sub(r"\bexpert\d+\b", "expert*", owner)
    owner = re.sub(r"\bbs\d+\b", "bs*", owner)
    owner = re.sub(r"\bpadded\d+\b", "padded*", owner)
    owner = re.sub(r"\brows\d+\b", "rows*", owner)
    return owner


def owner_source(owner: str) -> tuple[str, str, bool]:
    if owner.startswith("graph_input_staging"):
        return (
            "python/minisgl/engine/graph.py:GraphCaptureBuffer.copy_from",
            "direct NVTX around graph input copy_from input_ids/out_loc/positions",
            True,
        )
    if owner.startswith("replay_metadata_copy"):
        return (
            "python/minisgl/attention/deepseek_v4.py:DSV4AttentionBackend._copy_metadata_for_replay",
            "direct NVTX around replay metadata helper/fallback copies",
            True,
        )
    if owner.startswith("static_graph_input_updates"):
        return (
            "python/minisgl/attention/deepseek_v4.py:DSV4AttentionBackend.stage_capture_metadata_for_graph",
            "direct NVTX around capture metadata graph input update",
            True,
        )
    if owner.startswith("attention_boundary"):
        return (
            "python/minisgl/models/deepseek_v4.py and python/minisgl/attention/deepseek_v4.py",
            "direct NVTX around attention positions/cache/index dtype or layout staging",
            True,
        )
    if owner.startswith("moe_shared_expert_staging"):
        return (
            "python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner/DSV4SharedExperts",
            "direct NVTX around MoE/shared expert dtype staging",
            True,
        )
    if owner.startswith("sampler_logits_staging"):
        return (
            "python/minisgl/engine/engine.py:Engine.forward_batch",
            "direct NVTX around sampler/logits token staging",
            True,
        )
    if owner.startswith("batch_forward_bridge"):
        return (
            "python/minisgl/scheduler/scheduler.py:Scheduler._prepare_batch/_forward",
            "direct NVTX around scheduler to engine bridge",
            True,
        )
    if owner.startswith("dsv4.shared_experts"):
        return (
            "python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward",
            "graphNodeId original creation under coarse shared_experts NVTX",
            True,
        )
    if owner.startswith("dsv4.layer*.mlp"):
        return (
            "python/minisgl/models/deepseek_v4.py:DSV4MoE/DSV4FusedMoERunner.forward",
            "graphNodeId original creation under coarse layer MLP/MoE NVTX",
            True,
        )
    if owner.startswith("dsv4.layer*.attn"):
        return (
            "python/minisgl/models/deepseek_v4.py:DSV4Attention.forward",
            "graphNodeId original creation under coarse layer attention NVTX",
            True,
        )
    if owner.startswith("dsv4.layer*.hc_"):
        return (
            "python/minisgl/models/deepseek_v4.py:DSV4DecoderLayer.forward",
            "graphNodeId original creation under hidden-carrier staging NVTX",
            True,
        )
    if owner.startswith("dsv4.indexer."):
        return (
            "python/minisgl/models/deepseek_v4.py:DSV4Indexer.forward",
            "graphNodeId original creation under coarse indexer NVTX",
            True,
        )
    if owner == "dsv4.lm_head":
        return (
            "python/minisgl/models/deepseek_v4.py:DeepseekV4ForCausalLM.forward",
            "graphNodeId original creation under lm_head NVTX",
            True,
        )
    if owner.startswith("dsv4.model"):
        return (
            "python/minisgl/models/deepseek_v4.py:DeepseekV4Model.forward",
            "graphNodeId original creation under coarse model NVTX",
            True,
        )
    if owner.startswith("static_graph_replay"):
        return (
            "python/minisgl/engine/graph.py:GraphRunner._replay_to_buffer",
            "outer replay envelope; add or recover inner graph-node NVTX for implementation choice",
            False,
        )
    if owner.startswith("batch_forward") or owner.startswith("batch_prepare"):
        return (
            "benchmark/offline/deepseek_v4_perf_matrix.py:BenchScheduler._forward",
            "coarse benchmark NVTX; pre-instrumentation only",
            False,
        )
    if owner == "unattributed":
        return (
            "unknown",
            "no enclosing owner NVTX or graph-node mapping found",
            False,
        )
    return ("unknown", "unrecognized owner label", False)


def owner_display(owner: str) -> str:
    if owner == "unattributed":
        return "unattributed"
    if owner.startswith("static_graph_replay"):
        return f"residual static graph replay envelope: {owner}"
    if owner.startswith("batch_forward_bridge"):
        return owner.replace("_", " ")
    if owner.startswith("dsv4."):
        return f"graph node source: {owner}"
    if owner.startswith("batch_forward") or owner.startswith("batch_prepare"):
        return f"residual coarse benchmark envelope: {owner}"
    return owner.replace("_", " ")


def owner_ranges(
    cur: sqlite3.Cursor,
    *,
    window: tuple[int, int] | None = None,
    direct_only: bool = False,
    include_dsv4: bool = False,
) -> list[dict[str, Any]]:
    if not table_exists(cur, "NVTX_EVENTS"):
        return []
    predicates = [f"{nvtx_name_expr()} like ?"]
    params: list[Any] = [f"{DIRECT_NVTX_PREFIX}%"]
    if not direct_only:
        predicates.extend(
            [
                f"{nvtx_name_expr()} like 'batch_forward:%'",
                f"{nvtx_name_expr()} like 'batch_forward_enqueue:%'",
                f"{nvtx_name_expr()} like 'batch_prepare:%'",
            ]
        )
    if include_dsv4:
        predicates.append(f"{nvtx_name_expr()} like 'dsv4.%'")
    where = "(" + " or ".join(predicates) + ")"
    if window is not None:
        where += " and n.start<=? and n.end>=?"
        params.extend([window[1], window[0]])
    return rows(
        cur,
        f"""
        select {nvtx_name_expr()} as name, n.start as start, n.end as end
        from NVTX_EVENTS n
        left join StringIds s on s.id = n.textId
        where n.end is not null and {where}
        order by n.start
        """,
        params,
    )


def direct_copy_kernel_rows(
    cur: sqlite3.Cursor,
    *,
    window: tuple[int, int] | None = None,
    graph_only: bool = False,
) -> list[dict[str, Any]]:
    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    where = [
        """
        (
          lower(coalesce(s.value, cast(k.demangledName as text))) like '%direct_copy_kernel%'
          or lower(coalesce(s.value, cast(k.demangledName as text))) like '%bfloat16_copy_kernel%'
          or lower(coalesce(s.value, cast(k.demangledName as text))) like '%float8_copy%'
        )
        """
    ]
    params: list[Any] = []
    if window is not None:
        where.append("k.start>=? and k.end<=?")
        params.extend([window[0], window[1]])
    if graph_only:
        where.append("k.graphNodeId is not null")
    return rows(
        cur,
        f"""
        select
          coalesce(s.value, cast(k.demangledName as text)) as name,
          k.start as start,
          k.end as end,
          k.graphNodeId as graph_node_id,
          (k.end-k.start) / {NS} as duration_s
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where {" and ".join(where)}
        order by k.start
        """,
        params,
    )


def assign_owner_labels(
    kernels: list[dict[str, Any]],
    ranges: list[dict[str, Any]],
) -> list[str | None]:
    ranges = sorted(ranges, key=lambda row: int(row["start"]))
    kernels_sorted = sorted(enumerate(kernels), key=lambda item: int(item[1]["start"]))
    out: list[str | None] = [None] * len(kernels)
    active: list[dict[str, Any]] = []
    range_idx = 0
    for original_idx, kernel in kernels_sorted:
        start = int(kernel["start"])
        end = int(kernel["end"])
        while range_idx < len(ranges) and int(ranges[range_idx]["start"]) <= start:
            active.append(ranges[range_idx])
            range_idx += 1
        active = [row for row in active if int(row["end"]) >= end]
        candidates = [
            row
            for row in active
            if int(row["start"]) <= start and int(row["end"]) >= end
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: int(row["end"]) - int(row["start"]))
        out[original_idx] = str(best["name"])
    return out


def graph_node_owner_map(
    kernels: list[dict[str, Any]],
    labels: list[str | None],
) -> dict[int, str]:
    grouped: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for kernel, label in zip(kernels, labels):
        graph_node_id = kernel.get("graph_node_id")
        if graph_node_id is None or label is None:
            continue
        owner = normalize_owner(label)
        _, _, named = owner_source(owner)
        if not named:
            continue
        grouped[int(graph_node_id)][owner] += float(kernel["duration_s"])
    out = {}
    for graph_node_id, values in grouped.items():
        out[graph_node_id] = max(values.items(), key=lambda item: item[1])[0]
    return out


def graph_node_creation_owner_map(cur: sqlite3.Cursor) -> dict[int, str]:
    if not table_exists(cur, "CUDA_GRAPH_NODE_EVENTS"):
        return {}
    original_events = rows(
        cur,
        """
        select graphNodeId as graph_node_id, min(start) as start
        from CUDA_GRAPH_NODE_EVENTS
        where originalGraphNodeId is null
        group by graphNodeId
        """,
    )
    if not original_events:
        return {}
    start_by_original = {
        int(row["graph_node_id"]): int(row["start"]) for row in original_events
    }
    original_min = min(start_by_original.values())
    original_max = max(start_by_original.values())
    clone_events = rows(
        cur,
        """
        select graphNodeId as graph_node_id, originalGraphNodeId as original_graph_node_id
        from CUDA_GRAPH_NODE_EVENTS
        where originalGraphNodeId is not null
        """,
    )
    owner_range_rows = owner_ranges(
        cur,
        window=(original_min, original_max),
        direct_only=True,
        include_dsv4=True,
    )
    fake_kernels = [
        {
            "start": start,
            "end": start,
            "graph_node_id": graph_node_id,
            "duration_s": 0.0,
            "name": "CUDA graph node creation",
        }
        for graph_node_id, start in start_by_original.items()
    ]
    labels = assign_owner_labels(fake_kernels, owner_range_rows)
    owner_by_original = {
        int(kernel["graph_node_id"]): normalize_owner(label)
        for kernel, label in zip(fake_kernels, labels)
        if label is not None
    }
    out: dict[int, str] = {}
    for row in clone_events:
        original = row.get("original_graph_node_id")
        if original is None:
            continue
        owner = owner_by_original.get(int(original))
        if owner is not None:
            out[int(row["graph_node_id"])] = owner
    for original, owner in owner_by_original.items():
        out.setdefault(original, owner)
    return out


def aggregate(
    kernels: list[dict[str, Any]],
    owner_labels: list[str | None],
    graph_owner_by_node: dict[int, str],
) -> dict[str, Any]:
    owners: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "duration_s": 0.0,
            "count": 0,
            "graph_count": 0,
            "examples": set(),
            "evidence": set(),
        }
    )
    total = 0.0
    total_count = 0
    for kernel, label in zip(kernels, owner_labels):
        owner = normalize_owner(label)
        graph_node_id = kernel.get("graph_node_id")
        mapped = None
        if graph_node_id is not None:
            mapped = graph_owner_by_node.get(int(graph_node_id))
        _, _, named = owner_source(owner)
        if mapped is not None and not named:
            owner = mapped
            if owner.startswith("dsv4."):
                evidence = "graphNodeId originalGraphNodeId mapped to capture-time dsv4 NVTX"
            else:
                evidence = "graphNodeId originalGraphNodeId mapped to capture-time direct-copy NVTX"
        elif label is None:
            evidence = "no owner NVTX"
        elif str(label).startswith(DIRECT_NVTX_PREFIX):
            evidence = "innermost direct-copy NVTX"
        else:
            evidence = "coarse benchmark NVTX"
        duration_s = float(kernel["duration_s"])
        values = owners[owner]
        values["duration_s"] += duration_s
        values["count"] += 1
        values["graph_count"] += int(kernel.get("graph_node_id") is not None)
        values["examples"].add(str(kernel["name"])[:180])
        values["evidence"].add(evidence)
        total += duration_s
        total_count += 1

    rendered = []
    named_s = 0.0
    residual_s = 0.0
    for owner, values in owners.items():
        source, source_evidence, named = owner_source(owner)
        if named:
            named_s += values["duration_s"]
        else:
            residual_s += values["duration_s"]
        rendered.append(
            {
                "owner": owner,
                "display_owner": owner_display(owner),
                "duration_s": values["duration_s"],
                "count": values["count"],
                "graph_count": values["graph_count"],
                "share_of_direct_copy": values["duration_s"] / total if total else 0.0,
                "source": source,
                "evidence": "; ".join(sorted(values["evidence"] | {source_evidence})),
                "named_owner": named,
                "example_kernels": sorted(values["examples"])[:3],
            }
        )
    rendered.sort(key=lambda row: float(row["duration_s"]), reverse=True)
    return {
        "total_direct_copy_s": total,
        "total_direct_copy_count": total_count,
        "named_owner_direct_copy_s": named_s,
        "residual_direct_copy_s": residual_s,
        "named_owner_share": named_s / total if total else 0.0,
        "residual_share": residual_s / total if total else 0.0,
        "owners": rendered,
    }


def build_from_sqlite(sqlite_path: Path, repeat_nvtx: str) -> dict[str, Any]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    if not table_exists(cur, "NVTX_EVENTS"):
        raise RuntimeError("NVTX_EVENTS table not found")
    repeat = largest_range(find_nvtx_ranges(cur, exact=repeat_nvtx))
    if repeat is None:
        raise RuntimeError(f"repeat NVTX range not found: {repeat_nvtx}")
    decode_ranges = find_nvtx_ranges(cur, like="batch_forward:decode:%", parent=repeat)
    decode_envelope = envelope(decode_ranges)
    if decode_envelope is None:
        raise RuntimeError("decode forward envelope not found under repeat range")

    graph_owner_by_node = graph_node_creation_owner_map(cur)

    decode_kernels = direct_copy_kernel_rows(cur, window=decode_envelope)
    decode_ranges_for_owners = owner_ranges(cur, window=decode_envelope)
    decode_labels = assign_owner_labels(decode_kernels, decode_ranges_for_owners)
    summary = aggregate(decode_kernels, decode_labels, graph_owner_by_node)
    summary.update(
        {
            "mode": "sqlite",
            "sqlite_path": str(sqlite_path),
            "repeat_nvtx": repeat_nvtx,
            "repeat_range": repeat,
            "decode_forward_count": len(decode_ranges),
            "decode_envelope": decode_envelope,
            "decode_envelope_wall_s": (decode_envelope[1] - decode_envelope[0]) / NS,
            "graph_node_owner_map_count": len(graph_owner_by_node),
            "notes": [
                "Classifier first uses innermost direct-copy NVTX in the decode envelope.",
                "If replay kernels are only under coarse graph replay ranges, graphNodeId is mapped through originalGraphNodeId to capture-time direct-copy or dsv4 source NVTX when Nsight exposes it.",
                "Residual static_graph_replay or batch_forward owners mean more graph-node/source NVTX is needed before an implementation target is safe.",
            ],
        }
    )
    con.close()
    return summary


def build_from_subboundary_summary(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text())
    direct = data.get("sub_boundaries", {}).get("direct_copy", {})
    total = float(direct.get("duration_s") or 0.0)
    owners = []
    for owner, values in data.get("owner_breakdown", {}).items():
        direct_values = values.get("sub_boundaries", {}).get("direct_copy")
        if not direct_values:
            continue
        duration_s = float(direct_values.get("duration_s") or 0.0)
        count = int(direct_values.get("count") or 0)
        source, evidence, named = owner_source(owner)
        owners.append(
            {
                "owner": owner,
                "display_owner": owner_display(owner),
                "duration_s": duration_s,
                "count": count,
                "graph_count": None,
                "share_of_direct_copy": duration_s / total if total else 0.0,
                "source": source,
                "evidence": f"existing sub-boundary JSON owner_breakdown; {evidence}",
                "named_owner": named,
                "example_kernels": [],
            }
        )
    owners.sort(key=lambda row: float(row["duration_s"]), reverse=True)
    residual_s = sum(float(row["duration_s"]) for row in owners if not row["named_owner"])
    named_s = sum(float(row["duration_s"]) for row in owners if row["named_owner"])
    return {
        "mode": "subboundary_summary",
        "summary_path": str(summary_path),
        "sqlite_path": data.get("sqlite_path"),
        "repeat_nvtx": data.get("repeat_nvtx"),
        "decode_forward_count": data.get("decode_forward_count"),
        "decode_envelope_wall_s": data.get("decode_envelope_wall_s"),
        "total_direct_copy_s": total,
        "total_direct_copy_count": int(direct.get("count") or 0),
        "named_owner_direct_copy_s": named_s,
        "residual_direct_copy_s": residual_s,
        "named_owner_share": named_s / total if total else 0.0,
        "residual_share": residual_s / total if total else 0.0,
        "owners": owners,
        "notes": [
            "Pre-instrumentation control built from existing sub-boundary summary because raw sqlite is not present in this workspace.",
            "Coarse batch_forward owners are intentionally residual for the 07.65 owner gate.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Direct-Copy Owner Attribution: {summary.get('mode')}",
        "",
        f"- total direct_copy: `{summary['total_direct_copy_s']:.6f}s` / `{summary['total_direct_copy_count']}` kernels",
        f"- named owner coverage: `{summary['named_owner_share']:.2%}`",
        f"- residual: `{summary['residual_direct_copy_s']:.6f}s` (`{summary['residual_share']:.2%}`)",
        "",
        "## Direct-Copy Owner Table",
        "",
        "| Direct-copy owner | Kernel s | Count | Share | Source file/function | Evidence |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["owners"]:
        count = "n/a" if row["count"] is None else str(row["count"])
        lines.append(
            f"| `{row['display_owner']}` | `{row['duration_s']:.6f}` | {count} | "
            f"`{row['share_of_direct_copy']:.2%}` | `{row['source']}` | {row['evidence']} |"
        )
    residual = [row for row in summary["owners"] if not row["named_owner"]]
    if residual:
        lines.extend(
            [
                "",
                "## Residual Table",
                "",
                "| Residual owner | Kernel s | Share | Needed NVTX |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for row in residual:
            needed = (
                "inner graph-node/source NVTX inside replayed graph"
                if row["owner"].startswith("static_graph_replay")
                else "narrow direct-copy NVTX around the source boundary"
            )
            lines.append(
                f"| `{row['display_owner']}` | `{row['duration_s']:.6f}` | "
                f"`{row['share_of_direct_copy']:.2%}` | {needed} |"
            )
    lines.extend(["", "## Notes", ""])
    for note in summary.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sqlite", type=Path)
    source.add_argument("--subboundary-summary", type=Path)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    if args.sqlite is not None:
        summary = build_from_sqlite(args.sqlite, args.repeat_nvtx)
    else:
        summary = build_from_subboundary_summary(args.subboundary_summary)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(summary) + "\n")
    print(render_markdown(summary))


if __name__ == "__main__":
    main()

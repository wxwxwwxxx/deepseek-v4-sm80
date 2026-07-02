#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BASE_CLASSIFIER = (
    ROOT
    / "performance_milestones"
    / "target07_post_splitk_reprofile"
    / "scripts"
    / "summarize_post_splitk_nsys.py"
)
DEFAULT_CLASSIFIED = (
    ROOT
    / "performance_milestones"
    / "target07_graph_layout_replay_deforestation"
    / "summaries"
    / "nsys_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0_classified.json"
)
DEFAULT_SQLITE = (
    ROOT
    / "performance_milestones"
    / "target07_graph_layout_replay_deforestation"
    / "raw"
    / "nsys_target0754_graph_layout_node_4096x128_bs4_np128_actqtriton_rank0.sqlite"
)


GRAPH_LAYOUT_CLUSTER_S = 1.1874898079999998 + 0.6396070620000002


def _load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("target0740_nsys_base", BASE_CLASSIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base classifier from {BASE_CLASSIFIER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_decode_envelope(
    cur: sqlite3.Cursor,
    base: Any,
    *,
    repeat_nvtx: str,
    decode_nvtx: str,
) -> tuple[int, int]:
    repeat = base.largest_range(base.find_nvtx_ranges(cur, name=repeat_nvtx))
    if repeat is None:
        raise RuntimeError(f"repeat NVTX range not found: {repeat_nvtx}")
    decode_ranges = base.find_nvtx_ranges(cur, name=decode_nvtx, parent=repeat)
    envelope = base.envelope_range(decode_ranges)
    if envelope is None:
        raise RuntimeError(f"decode NVTX ranges not found under {repeat_nvtx}: {decode_nvtx}")
    return envelope


def _match_kernel(name: str, include: list[str], exclude: list[str] | None = None) -> bool:
    lowered = name.lower()
    if not any(token in lowered for token in include):
        return False
    return not any(token in lowered for token in (exclude or []))


def _group_rows(
    rows: list[dict[str, Any]],
    *,
    include: list[str],
    exclude: list[str] | None = None,
) -> dict[str, Any]:
    matched: list[dict[str, Any]] = []
    duration_s = 0.0
    count = 0
    graph_events = 0
    graph_nodes = 0
    for row in rows:
        if not _match_kernel(str(row["name"]), include, exclude):
            continue
        matched.append(row)
        duration_s += float(row.get("duration_s") or 0.0)
        count += int(row.get("count") or 0)
        graph_events += int(row.get("graph_count") or 0)
        graph_nodes += int(row.get("graph_node_count") or 0)
    top = sorted(matched, key=lambda item: float(item.get("duration_s") or 0.0), reverse=True)
    return {
        "duration_s": duration_s,
        "count": count,
        "graph_events": graph_events,
        "graph_nodes": graph_nodes,
        "share_of_07_54_graph_layout_cluster": duration_s / GRAPH_LAYOUT_CLUSTER_S,
        "top_kernels": [
            {
                "name": str(item["name"]),
                "duration_s": float(item.get("duration_s") or 0.0),
                "count": int(item.get("count") or 0),
                "graph_nodes": int(item.get("graph_node_count") or 0),
            }
            for item in top[:8]
        ],
    }


def _render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# TARGET 07.55 Remaining Graph/Layout Candidate Summary",
        "",
        f"- Source classified JSON: `{summary['inputs']['classified_json']}`",
        f"- Source SQLite: `{summary['inputs']['sqlite']}`",
        f"- Decode envelope wall: `{summary['decode_envelope_wall_s']:.6f} s`",
        f"- 07.54 graph/layout cluster: `{summary['graph_layout_cluster_s']:.6f} s`",
        f"- 10% graph/layout gate: `{summary['ten_percent_cluster_gate_s']:.6f} s`",
        "",
        "Candidate groups below are evidence slices, not additive totals; some kernel-name groups overlap with bucket-level classifications.",
        "",
        "| Candidate group | Duration s | Cluster share | Count | Graph nodes | Top kernel evidence |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, values in summary["candidate_groups"].items():
        top = values["top_kernels"][0]["name"] if values["top_kernels"] else "n/a"
        if len(top) > 92:
            top = top[:89] + "..."
        lines.append(
            "| "
            + name
            + f" | `{values['duration_s']:.6f}`"
            + f" | `{100.0 * values['share_of_07_54_graph_layout_cluster']:.2f}%`"
            + f" | `{values['count']}`"
            + f" | `{values['graph_nodes']}`"
            + f" | `{top}` |"
        )
    lines.extend(
        [
            "",
            "## Bucket Baseline",
            "",
            "| Bucket | Kernel s | Count | Graph nodes |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, values in summary["bucket_baseline"].items():
        lines.append(
            f"| `{name}` | `{values['duration_s']:.6f}` | `{values['count']}` | `{values['graph_nodes']}` |"
        )
    lines.extend(["", f"Decision encoded by TARGET 07.55 README: `{summary['decision']}`", ""])
    return "\n".join(lines)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    with args.classified_json.open() as f:
        classified = json.load(f)
    section = classified["sections"]["repeat_decode_forward_envelope"]
    base = _load_base_module()
    with sqlite3.connect(args.sqlite) as con:
        envelope = _find_decode_envelope(
            con.cursor(),
            base,
            repeat_nvtx=args.repeat_nvtx,
            decode_nvtx=args.decode_nvtx,
        )
        rows = base.named_events(
            con.cursor(),
            table="CUPTI_ACTIVITY_KIND_KERNEL",
            name_col="demangledName",
            ranges=[envelope],
            top=None,
        )

    candidate_groups = {
        "remaining_direct_copy_kernels": _group_rows(rows, include=["direct_copy"]),
        "bf16_and_float8_copy_kernels": _group_rows(
            rows,
            include=["bfloat16_copy", "float8_copy"],
        ),
        "cat_index_gather_topk_assembly": _group_rows(
            rows,
            include=[
                "catarraybatchedcopy",
                "index_elementwise",
                "vectorized_gather",
                "_scatter_gather",
                "arange_cuda",
                "fillfunctor",
                "gathertopk",
                "bitonicsort",
            ],
        ),
        "pow_mean_mul_elementwise_nodes": _group_rows(
            rows,
            include=[
                "pow_",
                "meanops",
                "mulfunctor",
                "rsqrt",
                "clamp",
                "divfunctor",
                "absfunctor",
                "log2",
                "ceil",
                "reduce_kernel",
            ],
            exclude=["cublas", "cutlass", "gemm", "splitkreduce"],
        ),
        "projection_gemm_intrinsic": _group_rows(
            rows,
            include=[
                "_quantized_linear_fp8_kernel",
                "ampere_sgemm",
                "ampere_bf16",
                "cutlass",
                "cublas",
                "gemm",
            ],
        ),
        "fp8_activation_quant_poc_kernel": _group_rows(
            rows,
            include=["_fp8_activation_quantize_kernel"],
        ),
    }
    bucket_baseline = {
        name: {
            "duration_s": float(values.get("duration_s") or 0.0),
            "count": int(values.get("count") or 0),
            "graph_nodes": int(values.get("graph_node_count") or 0),
        }
        for name, values in section["kernel_categories"].items()
    }
    return {
        "target": "TARGET 07.55 DSV4 SM80 remaining graph/layout or projection pivot",
        "inputs": {
            "classified_json": str(args.classified_json),
            "sqlite": str(args.sqlite),
            "repeat_nvtx": args.repeat_nvtx,
            "decode_nvtx": args.decode_nvtx,
        },
        "decode_envelope_wall_s": float(section["wall_s_sum"]),
        "graph_layout_cluster_s": GRAPH_LAYOUT_CLUSTER_S,
        "ten_percent_cluster_gate_s": 0.10 * GRAPH_LAYOUT_CLUSTER_S,
        "candidate_groups": candidate_groups,
        "bucket_baseline": bucket_baseline,
        "decision": "pivot to projection/GEMM backend parity",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified-json", type=Path, default=DEFAULT_CLASSIFIED)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--decode-nvtx", default="batch_forward:decode:bs4:padded4")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(_render_md(summary))


if __name__ == "__main__":
    main()

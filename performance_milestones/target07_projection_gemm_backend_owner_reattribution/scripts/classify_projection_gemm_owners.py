#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence


NS = 1_000_000_000.0
ROOT = Path(__file__).resolve().parents[3]
DIRECT_COPY_CLASSIFIER = (
    ROOT
    / "performance_milestones"
    / "target07_post_shared_expert_reprofile"
    / "scripts"
    / "classify_direct_copy_owners.py"
)


def load_direct_copy_module() -> Any:
    spec = importlib.util.spec_from_file_location("target0769_direct_copy", DIRECT_COPY_CLASSIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {DIRECT_COPY_CLASSIFIER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(
    cur: sqlite3.Cursor,
    query: str,
    params: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in cur.execute(query, tuple(params)):
        cols = [item[0] for item in cur.description]
        out.append({key: row[idx] for idx, key in enumerate(cols)})
    return out


def normalize_owner(owner: str) -> str:
    owner = re.sub(r"\blayer\d+\b", "layer*", owner)
    owner = re.sub(r"\bbs\d+\b", "bs*", owner)
    owner = re.sub(r"\bpadded\d+\b", "padded*", owner)
    return owner


def owner_family(owner: str) -> str:
    if owner in {"dsv4.layer*.hc_attn_pre", "dsv4.layer*.hc_ffn_pre"}:
        return "HC pre linear"
    if owner == "dsv4.layer*.mlp.runner.route":
        return "MoE router / route projection"
    if owner in {"dsv4.layer*.attn.q_proj", "dsv4.layer*.attn.compress"}:
        return "attention WQA/WKV/compress"
    if owner == "dsv4.layer*.attn.q_wqb":
        return "attention q_wqb"
    if owner == "dsv4.layer*.attn.wo_a":
        return "attention wo_a"
    if owner == "dsv4.layer*.attn.wo_b":
        return "attention wo_b local"
    if owner == "dsv4.indexer.wq_b":
        return "indexer wq_b"
    if owner in {"dsv4.indexer.weights_proj", "dsv4.indexer.compressor"}:
        return "indexer weight/compressor projection"
    if owner in {"dsv4.shared_experts.gate_up_proj", "dsv4.shared_experts.down_proj"}:
        return "shared experts cached BF16"
    if owner.startswith("dsv4.layer*.mlp.runner.experts"):
        return "routed MoE projection/backend"
    if owner == "dsv4.lm_head":
        return "lm_head"
    if owner.startswith("dsv4.model.hc_"):
        return "model HC head/expand"
    if owner in {"unattributed"} or owner.startswith("batch_forward") or owner.startswith("sampler"):
        return "residual / coarse owner"
    return owner


def owner_source(owner: str) -> str:
    family = owner_family(owner)
    if family == "HC pre linear":
        return "python/minisgl/models/deepseek_v4.py:DeepseekV4DecoderLayer._hc_pre; python/minisgl/kernel/deepseek_v4.py:linear_bf16_fp32_fallback"
    if family == "MoE router / route projection":
        return "python/minisgl/models/deepseek_v4.py:DSV4FusedMoERunner.route / DSV4TopK.forward"
    if family == "attention WQA/WKV/compress":
        return "python/minisgl/models/deepseek_v4.py:DSV4Attention.forward q_proj/compress"
    if family == "attention q_wqb":
        return "python/minisgl/models/deepseek_v4.py:DSV4Attention.forward q_wqb cached BF16"
    if family == "attention wo_a":
        return "python/minisgl/models/deepseek_v4.py:DSV4Attention.forward wo_a cached BF16 BMM"
    if family == "attention wo_b local":
        return "python/minisgl/models/deepseek_v4.py:DSV4Attention.forward wo_b cached BF16 local GEMM"
    if family == "indexer wq_b":
        return "python/minisgl/models/deepseek_v4.py:DSV4Indexer._wq_b_forward cached BF16"
    if family == "indexer weight/compressor projection":
        return "python/minisgl/models/deepseek_v4.py:DSV4Indexer.forward / DSV4Compressor.forward"
    if family == "shared experts cached BF16":
        return "python/minisgl/models/deepseek_v4.py:DSV4SharedExperts.forward cached BF16"
    if family == "lm_head":
        return "python/minisgl/models/deepseek_v4.py:DeepseekV4ForCausalLM.forward / DSV4VocabParallelEmbedding.linear"
    return "profile graphNodeId owner mapping"


def vllm_analogue(owner: str) -> str:
    family = owner_family(owner)
    if family == "HC pre linear":
        return "vllm/model_executor/layers/mhc.py:mhc_pre SM80 reference uses FP32 matmul plus fused Triton pre kernel"
    if family == "MoE router / route projection":
        return "vllm/model_executor/models/deepseek_v4.py:DeepseekV4MoE GateLinear / FusedMoE router"
    if family == "attention WQA/WKV/compress":
        return "vllm/model_executor/models/deepseek_v4.py:fused_wqa_wkv; layers/deepseek_v4_attention.py wrapper"
    if family in {"attention q_wqb", "attention wo_a", "attention wo_b local"}:
        return "vllm/model_executor/layers/deepseek_v4_attention.py:DeepseekV4MLA wrapper"
    if family.startswith("indexer"):
        return "vllm/model_executor/layers/deepseek_v4_attention.py:DeepseekV4Indexer and fused_indexer_q_rope_quant"
    if family == "shared experts cached BF16":
        return "vllm/model_executor/models/deepseek_v4.py:DeepseekV4MoE shared_experts passed to FusedMoE"
    if family == "lm_head":
        return "vllm/model_executor/models/deepseek_v4.py:ParallelLMHead / logits_processor"
    return "no focused vLLM source analogue selected"


def is_projection_kernel(name: str) -> bool:
    lowered = name.lower()
    return (
        "_quantized_linear_fp8_kernel" in lowered
        or "gemm" in lowered
        or "cutlass" in lowered
        or "cublas" in lowered
        or "ampere_bf16" in lowered
        or "ampere_sgemm" in lowered
        or "aten::bmm" in lowered
        or " bmm" in lowered
    )


def backend_family(name: str) -> str:
    lowered = name.lower()
    if "_quantized_linear_fp8_kernel" in lowered:
        return "residual FP8 quantized linear"
    if "splitkreduce" in lowered or "splitkreduce_kernel" in lowered:
        return "cuBLASLt splitK/reduce"
    if "cutlass::kernel" in lowered:
        return "CUTLASS BF16 GEMM"
    if "ampere_bf16" in lowered or "s16816gemm_bf16" in lowered:
        return "cuBLASLt BF16 GEMM"
    if "ampere_sgemm" in lowered or "gemmsn_tn_kernel<float" in lowered:
        return "cuBLAS SGEMM/FP32 GEMM"
    if "gemm" in lowered and "bf16" in lowered:
        return "BF16 GEMM other"
    if "gemm" in lowered:
        return "GEMM other"
    return "projection/GEMM other"


def backend_cluster(backend: str) -> str:
    if backend in {"CUTLASS BF16 GEMM", "cuBLASLt BF16 GEMM", "cuBLASLt splitK/reduce"}:
        return "BF16 small-GEMM + splitK/reduce cluster"
    if backend == "cuBLAS SGEMM/FP32 GEMM":
        return "FP32/SGEMM small-GEMM cluster"
    return backend


def short_kernel_name(name: str, limit: int = 120) -> str:
    name = name.replace("|", "\\|")
    if len(name) > limit:
        return name[: limit - 3] + "..."
    return name


def find_decode_envelope(cur: sqlite3.Cursor, repeat_nvtx: str, dc: Any) -> tuple[int, int]:
    repeat = dc.largest_range(dc.find_nvtx_ranges(cur, exact=repeat_nvtx))
    if repeat is None:
        raise RuntimeError(f"repeat NVTX not found: {repeat_nvtx}")
    decode_ranges = dc.find_nvtx_ranges(cur, like="batch_forward:decode:%", parent=repeat)
    if not decode_ranges:
        raise RuntimeError("decode forward NVTX ranges not found")
    return min(start for start, _ in decode_ranges), max(end for _, end in decode_ranges)


def kernel_rows(cur: sqlite3.Cursor, window: tuple[int, int]) -> list[dict[str, Any]]:
    return rows(
        cur,
        f"""
        select
          coalesce(s.value, cast(k.demangledName as text)) as name,
          k.start as start,
          k.end as end,
          k.graphNodeId as graph_node_id,
          k.gridX as grid_x,
          k.gridY as grid_y,
          k.gridZ as grid_z,
          k.blockX as block_x,
          k.blockY as block_y,
          k.blockZ as block_z,
          (k.end-k.start) / {NS} as duration_s
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        where k.start>=? and k.end<=?
        order by k.start
        """,
        window,
    )


def mapped_owner(row: dict[str, Any], label: str | None, owner_map: dict[int, str], dc: Any) -> str:
    owner = normalize_owner(dc.normalize_owner(label))
    mapped = None
    graph_node_id = row.get("graph_node_id")
    if graph_node_id is not None:
        mapped = owner_map.get(int(graph_node_id))
    _, _, named = dc.owner_source(owner)
    if mapped is not None and (not named or owner == "unattributed" or owner.startswith("static_graph_replay")):
        return normalize_owner(mapped)
    return owner


def aggregate(sqlite_path: Path, repeat_nvtx: str) -> dict[str, Any]:
    dc = load_direct_copy_module()
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    decode = find_decode_envelope(cur, repeat_nvtx, dc)
    owner_map = dc.graph_node_creation_owner_map(cur)
    ranges = dc.owner_ranges(cur, window=decode, include_dsv4=True)
    kernels = kernel_rows(cur, decode)
    labels = dc.assign_owner_labels(kernels, ranges)

    owners: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "duration_s": 0.0,
            "count": 0,
            "graph_nodes": set(),
            "backends": defaultdict(float),
            "kernel_names": defaultdict(float),
            "grid_shapes": defaultdict(int),
        }
    )
    owner_groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "duration_s": 0.0,
            "count": 0,
            "owners": defaultdict(float),
            "backends": defaultdict(float),
            "kernel_names": defaultdict(float),
            "graph_nodes": set(),
        }
    )
    backends: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "duration_s": 0.0,
            "count": 0,
            "owners": defaultdict(float),
            "kernel_names": defaultdict(float),
        }
    )
    clusters: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "duration_s": 0.0,
            "count": 0,
            "backends": defaultdict(float),
            "owners": defaultdict(float),
        }
    )
    top_kernels: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"duration_s": 0.0, "count": 0, "graph_count": 0, "graph_nodes": set()}
    )
    total = 0.0
    total_count = 0

    for row, label in zip(kernels, labels):
        name = str(row["name"])
        if not is_projection_kernel(name):
            continue
        duration = float(row["duration_s"])
        owner = mapped_owner(row, label, owner_map, dc)
        backend = backend_family(name)
        cluster = backend_cluster(backend)
        family = owner_family(owner)
        total += duration
        total_count += 1

        owner_values = owners[owner]
        owner_values["duration_s"] += duration
        owner_values["count"] += 1
        owner_values["backends"][backend] += duration
        owner_values["kernel_names"][name] += duration
        owner_values["grid_shapes"][
            (
                row["grid_x"],
                row["grid_y"],
                row["grid_z"],
                row["block_x"],
                row["block_y"],
                row["block_z"],
            )
        ] += 1
        if row.get("graph_node_id") is not None:
            owner_values["graph_nodes"].add(int(row["graph_node_id"]))

        group_values = owner_groups[family]
        group_values["duration_s"] += duration
        group_values["count"] += 1
        group_values["owners"][owner] += duration
        group_values["backends"][backend] += duration
        group_values["kernel_names"][name] += duration
        if row.get("graph_node_id") is not None:
            group_values["graph_nodes"].add(int(row["graph_node_id"]))

        backend_values = backends[backend]
        backend_values["duration_s"] += duration
        backend_values["count"] += 1
        backend_values["owners"][family] += duration
        backend_values["kernel_names"][name] += duration

        cluster_values = clusters[cluster]
        cluster_values["duration_s"] += duration
        cluster_values["count"] += 1
        cluster_values["backends"][backend] += duration
        cluster_values["owners"][family] += duration

        kernel_values = top_kernels[name]
        kernel_values["duration_s"] += duration
        kernel_values["count"] += 1
        if row.get("graph_node_id") is not None:
            kernel_values["graph_count"] += 1
            kernel_values["graph_nodes"].add(int(row["graph_node_id"]))

    con.close()

    def render_owner(owner: str, values: dict[str, Any]) -> dict[str, Any]:
        return {
            "owner": owner,
            "owner_family": owner_family(owner),
            "duration_s": values["duration_s"],
            "count": values["count"],
            "share_of_projection_gemm": values["duration_s"] / total if total else 0.0,
            "graph_node_count": len(values["graph_nodes"]),
            "backend_families": dict(
                sorted(values["backends"].items(), key=lambda item: item[1], reverse=True)
            ),
            "top_kernels": [
                {"name": name, "duration_s": duration}
                for name, duration in sorted(
                    values["kernel_names"].items(), key=lambda item: item[1], reverse=True
                )[:5]
            ],
            "source_boundary": owner_source(owner),
            "vllm_analogue": vllm_analogue(owner),
        }

    def render_group(family: str, values: dict[str, Any]) -> dict[str, Any]:
        return {
            "owner": family,
            "duration_s": values["duration_s"],
            "count": values["count"],
            "share_of_projection_gemm": values["duration_s"] / total if total else 0.0,
            "graph_node_count": len(values["graph_nodes"]),
            "owners": dict(sorted(values["owners"].items(), key=lambda item: item[1], reverse=True)),
            "backend_families": dict(
                sorted(values["backends"].items(), key=lambda item: item[1], reverse=True)
            ),
            "top_kernels": [
                {"name": name, "duration_s": duration}
                for name, duration in sorted(
                    values["kernel_names"].items(), key=lambda item: item[1], reverse=True
                )[:5]
            ],
        }

    summary = {
        "sqlite_path": str(sqlite_path),
        "repeat_nvtx": repeat_nvtx,
        "decode_envelope_s": (decode[1] - decode[0]) / NS,
        "projection_gemm_s": total,
        "projection_gemm_count": total_count,
        "graph_node_owner_map_count": len(owner_map),
        "owners": [
            render_owner(owner, values)
            for owner, values in sorted(owners.items(), key=lambda item: item[1]["duration_s"], reverse=True)
        ],
        "owner_groups": [
            render_group(family, values)
            for family, values in sorted(
                owner_groups.items(), key=lambda item: item[1]["duration_s"], reverse=True
            )
        ],
        "backends": [
            {
                "backend_family": backend,
                "duration_s": values["duration_s"],
                "count": values["count"],
                "share_of_projection_gemm": values["duration_s"] / total if total else 0.0,
                "owner_groups": dict(
                    sorted(values["owners"].items(), key=lambda item: item[1], reverse=True)
                ),
                "top_kernels": [
                    {"name": name, "duration_s": duration}
                    for name, duration in sorted(
                        values["kernel_names"].items(), key=lambda item: item[1], reverse=True
                    )[:5]
                ],
            }
            for backend, values in sorted(backends.items(), key=lambda item: item[1]["duration_s"], reverse=True)
        ],
        "backend_clusters": [
            {
                "backend_cluster": cluster,
                "duration_s": values["duration_s"],
                "count": values["count"],
                "share_of_projection_gemm": values["duration_s"] / total if total else 0.0,
                "backend_families": dict(
                    sorted(values["backends"].items(), key=lambda item: item[1], reverse=True)
                ),
                "owner_groups": dict(
                    sorted(values["owners"].items(), key=lambda item: item[1], reverse=True)
                ),
            }
            for cluster, values in sorted(clusters.items(), key=lambda item: item[1]["duration_s"], reverse=True)
        ],
        "top_kernels": [
            {
                "name": name,
                "duration_s": values["duration_s"],
                "count": values["count"],
                "graph_count": values["graph_count"],
                "graph_node_count": len(values["graph_nodes"]),
                "backend_family": backend_family(name),
            }
            for name, values in sorted(top_kernels.items(), key=lambda item: item[1]["duration_s"], reverse=True)
        ],
    }
    return summary


def fmt_s(value: float) -> str:
    return f"{value:.6f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def kernel_list(kernels: list[dict[str, Any]]) -> str:
    return "<br>".join(
        f"`{fmt_s(float(item['duration_s']))}` {short_kernel_name(str(item['name']), 88)}"
        for item in kernels[:3]
    )


def backends_list(backends: dict[str, float]) -> str:
    return "<br>".join(f"`{name}` `{fmt_s(float(duration))}`" for name, duration in backends.items())


def render_owner_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Projection/GEMM Owner Table",
        "",
        f"- sqlite: `{summary['sqlite_path']}`",
        f"- repeat NVTX: `{summary['repeat_nvtx']}`",
        f"- decode envelope: `{summary['decode_envelope_s']:.6f}s`",
        f"- projection/GEMM: `{summary['projection_gemm_s']:.6f}s` / `{summary['projection_gemm_count']}` kernels",
        f"- graph-node owner map entries: `{summary['graph_node_owner_map_count']}`",
        "",
        "## Grouped Owners",
        "",
        "| Owner group | Kernel s | Count | Share | Backend family | Top kernels | Decision |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["owner_groups"]:
        duration = float(row["duration_s"])
        decision = "microbench / next target candidate" if duration >= 0.09 else "name and track"
        if row["owner"] == "residual / coarse owner":
            decision = "residual small; no recapture needed"
        lines.append(
            "| `{}` | `{}` | {} | `{}` | {} | {} | {} |".format(
                row["owner"],
                fmt_s(duration),
                int(row["count"]),
                pct(float(row["share_of_projection_gemm"])),
                backends_list(row["backend_families"]),
                kernel_list(row["top_kernels"]),
                decision,
            )
        )

    lines.extend(
        [
            "",
            "## Raw Graph Owners",
            "",
            "| Owner | Kernel s | Count | Share | Backend family | Source boundary | vLLM analogue | Top kernels |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in summary["owners"]:
        lines.append(
            "| `{}` | `{}` | {} | `{}` | {} | `{}` | `{}` | {} |".format(
                row["owner"],
                fmt_s(float(row["duration_s"])),
                int(row["count"]),
                pct(float(row["share_of_projection_gemm"])),
                backends_list(row["backend_families"]),
                row["source_boundary"],
                row["vllm_analogue"],
                kernel_list(row["top_kernels"]),
            )
        )
    return "\n".join(lines) + "\n"


def render_backend_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Projection/GEMM Backend Families",
        "",
        f"- projection/GEMM denominator: `{summary['projection_gemm_s']:.6f}s`",
        "",
        "## Backend Clusters",
        "",
        "| Backend cluster | Kernel s | Count | Share | Backend families | Owner groups | Gate read |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["backend_clusters"]:
        gate = "clears same-backend cluster gate" if float(row["duration_s"]) >= 0.35 else "below same-backend cluster gate"
        lines.append(
            "| `{}` | `{}` | {} | `{}` | {} | {} | {} |".format(
                row["backend_cluster"],
                fmt_s(float(row["duration_s"])),
                int(row["count"]),
                pct(float(row["share_of_projection_gemm"])),
                backends_list(row["backend_families"]),
                backends_list(row["owner_groups"]),
                gate,
            )
        )

    lines.extend(
        [
            "",
            "## Backend Families",
            "",
            "| Backend family | Kernel s | Count | Share | Owner groups | Top kernels |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["backends"]:
        lines.append(
            "| `{}` | `{}` | {} | `{}` | {} | {} |".format(
                row["backend_family"],
                fmt_s(float(row["duration_s"])),
                int(row["count"]),
                pct(float(row["share_of_projection_gemm"])),
                backends_list(row["owner_groups"]),
                kernel_list(row["top_kernels"]),
            )
        )
    return "\n".join(lines) + "\n"


def render_top_kernel_markdown(summary: dict[str, Any], top: int) -> str:
    lines = [
        "# Projection/GEMM Top Kernels",
        "",
        "| Kernel | Backend family | Kernel s | Count | Graph events | Graph nodes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["top_kernels"][:top]:
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {} | {} |".format(
                short_kernel_name(str(row["name"]), 150),
                row["backend_family"],
                fmt_s(float(row["duration_s"])),
                int(row["count"]),
                int(row["graph_count"]),
                int(row["graph_node_count"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    summary = aggregate(args.sqlite, args.repeat_nvtx)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "projection_gemm_owner_table.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (args.output_dir / "projection_gemm_owner_table.md").write_text(
        render_owner_markdown(summary)
    )
    (args.output_dir / "projection_gemm_backend_families.md").write_text(
        render_backend_markdown(summary)
    )
    (args.output_dir / "projection_gemm_top_kernels.md").write_text(
        render_top_kernel_markdown(summary, top=args.top)
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
BASE_CLASSIFIER = (
    ROOT
    / "performance_milestones"
    / "target07_post_splitk_reprofile"
    / "scripts"
    / "summarize_post_splitk_nsys.py"
)

KERNEL_CATEGORY_ORDER = (
    "sparse_attention",
    "nccl_communication",
    "direct_copy_layout",
    "hc_elementwise",
    "moe_routed_backend",
    "projection_gemm",
    "fp8_activation_quant",
    "index_cache_topk",
    "rmsnorm_rope_compress_store",
    "sampling_logits_other",
    "other",
)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("target0767_nsys_base", BASE_CLASSIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base classifier from {BASE_CLASSIFIER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        return "nccl_communication"
    if (
        "sparse_attention_kernel" in lowered
        or "_sparse_bf16_gather_with_mask_kernel" in lowered
        or "_sparse_splitk_bf16_split_kernel" in lowered
        or "_sparse_splitk_bf16_combine_kernel" in lowered
        or "paged_mqa" in lowered
        or "flash_mla" in lowered
    ):
        return "sparse_attention"
    if (
        "_fp8_activation_quantize_kernel" in lowered
        or "fp8_activation_quant" in lowered
        or "act_quant" in lowered
    ):
        return "fp8_activation_quant"
    if (
        "marlin_moe_wna16" in lowered
        or "moe_route" in lowered
        or "gptq_marlin_repack" in lowered
        or "fused_moe" in lowered
        or "swiglu" in lowered
    ):
        return "moe_routed_backend"
    if (
        "rms_norm" in lowered
        or "rmsnorm" in lowered
        or "rope" in lowered
        or "compress_norm_rope" in lowered
        or "_q_kv_norm_rope_cache" in lowered
        or "compress_quant_cache" in lowered
        or "store_cache" in lowered
        or "kv_cache" in lowered
        or "k_cache" in lowered
        or "v_cache" in lowered
        or "masked_locs" in lowered
    ):
        return "rmsnorm_rope_compress_store"
    if (
        "indexer" in lowered
        or "topk" in lowered
        or "gathertopk" in lowered
        or "bitonicsort" in lowered
        or "persistent_topk" in lowered
        or "global_topk" in lowered
        or "store_indexer" in lowered
        or "deviceselect" in lowered
        or "devicescan" in lowered
        or "devicecompact" in lowered
        or "deviceradixsort" in lowered
    ):
        return "index_cache_topk"
    if (
        "_quantized_linear_fp8_kernel" in lowered
        or "gemm" in lowered
        or "cutlass" in lowered
        or "cublas" in lowered
        or "ampere_bf16" in lowered
        or "ampere_sgemm" in lowered
        or "aten::bmm" in lowered
        or " bmm" in lowered
    ):
        return "projection_gemm"
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
    ):
        return "direct_copy_layout"
    if (
        "_hc_" in lowered
        or "reduce_kernel" in lowered
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
        or "mean" in lowered
    ):
        return "hc_elementwise"
    if (
        "lm_head" in lowered
        or "logits" in lowered
        or "softmax" in lowered
        or "sampler" in lowered
        or "sampling" in lowered
        or "argmax" in lowered
    ):
        return "sampling_logits_other"
    return "other"


def fmt_s(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def pct(value: float, total: float | None) -> str:
    if not total:
        return "n/a"
    return f"{100.0 * value / float(total):.2f}%"


def section_categories(summary: dict[str, Any], section_name: str) -> dict[str, dict[str, Any]]:
    return summary.get("sections", {}).get(section_name, {}).get("kernel_categories", {})


def category_duration(summary: dict[str, Any], section_name: str, category: str) -> float:
    return float(section_categories(summary, section_name).get(category, {}).get("duration_s") or 0.0)


def infer_phase(prefill_s: float, decode_s: float) -> str:
    if prefill_s <= 0.0 and decode_s <= 0.0:
        return "not observed"
    if decode_s >= max(0.02, prefill_s * 3.0):
        return "decode-heavy"
    if prefill_s >= max(0.02, decode_s * 3.0):
        return "prefill-heavy"
    return "mixed"


def enrich_summary(summary: dict[str, Any]) -> dict[str, Any]:
    decode_wall = summary.get("nvtx_range_counts", {}).get(
        "repeat_decode_forward_envelope_wall_s"
    )
    for section in summary.get("sections", {}).values():
        wall = section.get("wall_s_sum")
        for values in section.get("kernel_categories", {}).values():
            duration = float(values.get("duration_s") or 0.0)
            values["share_of_section_wall"] = None if not wall else duration / float(wall)
            values["share_of_decode_envelope_wall"] = (
                None if not decode_wall else duration / float(decode_wall)
            )

    phases = {}
    for category in KERNEL_CATEGORY_ORDER:
        prefill_s = category_duration(summary, "repeat_prefill_forward", category)
        decode_s = category_duration(summary, "repeat_decode_forward_envelope", category)
        phases[category] = {
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "phase": infer_phase(prefill_s, decode_s),
        }
    summary["bucket_phase"] = phases
    return summary


def ordered_categories(categories: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    sorted_items = sorted(
        categories.items(),
        key=lambda item: float(item[1].get("duration_s") or 0.0),
        reverse=True,
    )
    ordered = [item for category in KERNEL_CATEGORY_ORDER for item in sorted_items if item[0] == category]
    ordered.extend(item for item in sorted_items if item[0] not in KERNEL_CATEGORY_ORDER)
    return ordered


def render_category_table(
    categories: dict[str, dict[str, Any]],
    *,
    total_s: float | None,
) -> list[str]:
    lines = [
        "| Bucket | Count | Kernel s | Share | Graph events | Graph nodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, values in ordered_categories(categories):
        duration = float(values.get("duration_s") or 0.0)
        lines.append(
            "| `{}` | {} | `{}` | {} | {} | {} |".format(
                category,
                int(values.get("count") or 0),
                fmt_s(duration),
                pct(duration, total_s),
                int(values.get("graph_count") or 0),
                int(values.get("graph_node_count") or 0),
            )
        )
    return lines


def render_top_table(rows_in: Sequence[dict[str, Any]], *, top: int) -> list[str]:
    lines = [
        "| Kernel name | Count | Kernel s | Graph events | Graph nodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_in[:top]:
        name = str(row.get("name") or "").replace("|", "\\|")
        if len(name) > 150:
            name = name[:147] + "..."
        lines.append(
            "| `{}` | {} | `{}` | {} | {} |".format(
                name,
                int(row.get("count") or 0),
                fmt_s(row.get("duration_s")),
                int(row.get("graph_count") or 0),
                int(row.get("graph_node_count") or 0),
            )
        )
    return lines


def render_markdown(summary: dict[str, Any], *, top: int) -> str:
    lines: list[str] = []
    path = Path(summary["sqlite_path"]).name
    counts = summary.get("nvtx_range_counts", {})
    lines.append(f"# TARGET 07.67 Bucket Summary: {path}")
    lines.append("")
    lines.append(f"- Requested repeat NVTX: `{summary['repeat_nvtx']}`")
    lines.append(f"- Repeat range found: `{summary['repeat_range'] is not None}`")
    lines.append(
        "- Repeat child ranges: prefill_forward={}, decode_forward={}, "
        "decode_forward_sum_s=`{}`, decode_envelope_s=`{}`".format(
            counts.get("repeat_prefill_forward"),
            counts.get("repeat_decode_forward"),
            fmt_s(counts.get("repeat_decode_forward_wall_s_sum")),
            fmt_s(counts.get("repeat_decode_forward_envelope_wall_s")),
        )
    )
    lines.append("")
    for section_name in (
        "repeat_decode_forward_envelope",
        "repeat_prefill_forward",
        "repeat",
    ):
        section = summary.get("sections", {}).get(section_name, {})
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append(
            "- wall_s=`{}`, kernel_s=`{}`, runtime_s=`{}`".format(
                fmt_s(section.get("wall_s_sum")),
                fmt_s(section.get("kernel", {}).get("duration_s")),
                fmt_s(section.get("runtime", {}).get("duration_s")),
            )
        )
        lines.append("")
        lines.extend(
            render_category_table(
                section.get("kernel_categories", {}),
                total_s=section.get("kernel", {}).get("duration_s"),
            )
        )
        lines.append("")
    lines.append("## Bucket Phase")
    lines.append("")
    lines.append("| Bucket | Prefill kernel s | Decode envelope kernel s | Phase |")
    lines.append("| --- | ---: | ---: | --- |")
    for category in KERNEL_CATEGORY_ORDER:
        row = summary.get("bucket_phase", {}).get(category, {})
        lines.append(
            "| `{}` | `{}` | `{}` | {} |".format(
                category,
                fmt_s(row.get("prefill_s")),
                fmt_s(row.get("decode_s")),
                row.get("phase", "n/a"),
            )
        )
    lines.append("")
    lines.append("## Top Decode Kernels")
    lines.append("")
    decode_section = summary.get("sections", {}).get("repeat_decode_forward_envelope", {})
    lines.extend(render_top_table(decode_section.get("top_kernels", []), top=top))
    lines.append("")
    lines.append("## Top Repeat Kernels")
    lines.append("")
    repeat_section = summary.get("sections", {}).get("repeat", {})
    lines.extend(render_top_table(repeat_section.get("top_kernels", []), top=top))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    base = load_base_module()
    base.KERNEL_CATEGORY_ORDER = KERNEL_CATEGORY_ORDER
    base.classify_kernel = classify_kernel
    summary = base.build_summary(args.sqlite, repeat_nvtx=args.repeat_nvtx, top=args.top)
    summary["classifier"] = "target07_67_post_shared_expert_reprofile"
    summary["kernel_category_order"] = list(KERNEL_CATEGORY_ORDER)
    summary = enrich_summary(summary)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render_markdown(summary, top=args.top) + "\n")


if __name__ == "__main__":
    main()

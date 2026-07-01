#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
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


KERNEL_CATEGORY_ORDER = (
    "graph_runtime_copy_cat_index",
    "elementwise_graph_nodes",
    "fp8_indexer",
    "sparse_attention_decode",
    "prefill_sparse_attention",
    "kv_compressor_cache_store",
    "projection_gemm",
    "moe_marlin",
    "nccl_communication",
    "sampling_logits",
    "unknown",
)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("target0740_nsys_base", BASE_CLASSIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base classifier from {BASE_CLASSIFIER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        return "nccl_communication"
    if "sparse_attention_kernel" in lowered:
        return "prefill_sparse_attention"
    if (
        "_sparse_bf16_gather_with_mask_kernel" in lowered
        or "_sparse_splitk_bf16_split_kernel" in lowered
        or "_sparse_splitk_bf16_combine_kernel" in lowered
    ):
        return "sparse_attention_decode"
    if (
        "_indexer_fp8" in lowered
        or "fp8_paged_mqa" in lowered
        or "_indexer_bf16_logits_kernel" in lowered
        or "topk_transform" in lowered
        or "global_topk" in lowered
        or "gathertopk" in lowered
        or "bitonicsortkv" in lowered
        or "persistent_topk" in lowered
        or "store_indexer" in lowered
        or ("indexer" in lowered and "wqb" not in lowered)
    ):
        return "fp8_indexer"
    if (
        "_q_kv_norm_rope_cache" in lowered
        or "_compress_norm_rope_store" in lowered
        or "compress_quant_cache" in lowered
        or "quant_cache" in lowered
        or "cache_utils" in lowered
        or "store_cache" in lowered
        or "kv_cache" in lowered
        or "k_cache" in lowered
        or "v_cache" in lowered
        or "masked_locs" in lowered
    ):
        return "kv_compressor_cache_store"
    if (
        "_quantized_linear_fp8_kernel" in lowered
        or "gemm" in lowered
        or "cutlass" in lowered
        or "cublas" in lowered
        or "ampere_bf16" in lowered
        or "ampere_sgemm" in lowered
    ):
        return "projection_gemm"
    if "marlin_moe_wna16" in lowered or "moe_route" in lowered or "gptq_marlin_repack" in lowered:
        return "moe_marlin"
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
        return "graph_runtime_copy_cat_index"
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
        return "sampling_logits"
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
        return "elementwise_graph_nodes"
    return "unknown"


def enrich_summary(summary: dict[str, Any]) -> dict[str, Any]:
    for section in summary.get("sections", {}).values():
        wall = section.get("wall_s_sum")
        decode_wall = summary.get("nvtx_range_counts", {}).get(
            "repeat_decode_forward_envelope_wall_s"
        )
        for values in section.get("kernel_categories", {}).values():
            duration = float(values.get("duration_s") or 0.0)
            values["share_of_section_wall"] = None if not wall else duration / float(wall)
            values["share_of_decode_envelope_wall"] = (
                None if not decode_wall else duration / float(decode_wall)
            )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--repeat-nvtx", default="repeat:decode_throughput_bs8:0")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    base = load_base_module()
    base.KERNEL_CATEGORY_ORDER = KERNEL_CATEGORY_ORDER
    base.classify_kernel = classify_kernel
    summary = base.build_summary(args.sqlite, repeat_nvtx=args.repeat_nvtx, top=args.top)
    summary["classifier"] = "target07_53_fp8_indexer_reprofile"
    summary["kernel_category_order"] = list(KERNEL_CATEGORY_ORDER)
    summary = enrich_summary(summary)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(base.render_markdown(summary) + "\n")


if __name__ == "__main__":
    main()

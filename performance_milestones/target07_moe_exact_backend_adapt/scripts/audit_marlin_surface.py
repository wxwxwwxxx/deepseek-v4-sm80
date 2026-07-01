from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = Path("/workspace/vllm-dsv4-docker")


SOURCE_ANCHORS = {
    "vllm_fused_marlin_moe": VLLM_ROOT / "vllm/model_executor/layers/fused_moe/fused_marlin_moe.py",
    "vllm_mxfp4_quant": VLLM_ROOT / "vllm/model_executor/layers/quantization/mxfp4.py",
    "vllm_marlin_utils_fp4": VLLM_ROOT
    / "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py",
    "vllm_custom_ops": VLLM_ROOT / "vllm/_custom_ops.py",
    "vllm_moe_marlin_cuda": VLLM_ROOT / "csrc/moe/marlin_moe_wna16/ops.cu",
    "vllm_marlin_repack_cuda": VLLM_ROOT / "csrc/quantization/marlin/gptq_marlin_repack.cu",
    "mini_kernel_wrapper": ROOT / "python/minisgl/kernel/deepseek_v4.py",
    "mini_triton_moe": ROOT / "python/minisgl/kernel/triton/deepseek_v4.py",
    "mini_model_moe": ROOT / "python/minisgl/models/deepseek_v4.py",
}


REQUIRED_SURFACE = {
    "prepare_moe_mxfp4_layer_for_marlin": {
        "vllm_anchor": "vllm_marlin_utils_fp4",
        "category": "weight_layout",
    },
    "gptq_marlin_repack": {
        "vllm_anchor": "vllm_marlin_repack_cuda",
        "category": "custom_cuda_repack_op",
    },
    "moe_wna16_marlin_gemm": {
        "vllm_anchor": "vllm_moe_marlin_cuda",
        "category": "custom_cuda_expert_gemm_op",
    },
    "moe_align_block_size": {
        "vllm_anchor": "vllm_fused_marlin_moe",
        "category": "route_metadata_adapter",
    },
    "marlin_moe_intermediate_size": {
        "vllm_anchor": "vllm_fused_marlin_moe",
        "category": "packed_shape_contract",
    },
}


def _contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return False


def _mini_equivalent_hits(name: str, category: str) -> list[str]:
    if category.startswith("custom_cuda"):
        search_roots = [
            ROOT / "python/minisgl/kernel/csrc",
            ROOT / "python/minisgl/kernel/triton",
        ]
        paths = [
            path
            for search_root in search_roots
            if search_root.exists()
            for path in search_root.rglob("*")
            if path.is_file()
        ]
        return [str(path) for path in paths if _contains(path, name)]

    if name == "moe_align_block_size":
        path = SOURCE_ANCHORS["mini_kernel_wrapper"]
        return [str(path)] if _contains(path, "build_moe_route_plan") else []

    paths = [
        SOURCE_ANCHORS["mini_kernel_wrapper"],
        SOURCE_ANCHORS["mini_triton_moe"],
        SOURCE_ANCHORS["mini_model_moe"],
    ]
    return [str(path) for path in paths if _contains(path, name)]


def build_report() -> dict[str, object]:
    anchors = {
        name: {
            "path": str(path),
            "exists": path.exists(),
        }
        for name, path in SOURCE_ANCHORS.items()
    }
    required = {}
    for name, spec in REQUIRED_SURFACE.items():
        anchor_path = SOURCE_ANCHORS[spec["vllm_anchor"]]
        mini_hits = _mini_equivalent_hits(name, spec["category"])
        required[name] = {
            **spec,
            "vllm_anchor_path": str(anchor_path),
            "vllm_anchor_present": _contains(anchor_path, name),
            "mini_hits": mini_hits,
            "mini_equivalent_present": bool(mini_hits),
        }

    hard_blockers = [
        name
        for name, item in required.items()
        if item["category"].startswith("custom_cuda") and not item["mini_equivalent_present"]
    ]
    return {
        "target": "TARGET_07.38_dsv4_sm80_moe_exact_backend_adapt",
        "selected_backend": "MARLIN",
        "candidate_backend_env": "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_mxfp4_w4a16",
        "anchors": anchors,
        "required_surface": required,
        "hard_blockers": hard_blockers,
        "decision": (
            "reject_direct_marlin_port_for_this_cut"
            if hard_blockers
            else "narrow_adaptation_may_be_feasible"
        ),
        "silent_fallback_policy": "Marlin opt-in raises unsupported instead of running grouped FP4.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit vLLM Marlin MXFP4 W4A16 surface.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_moe_exact_backend_adapt/summaries/"
        / "marlin_feasibility_audit.json",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()

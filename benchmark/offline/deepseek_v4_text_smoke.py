from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")


DEFAULT_PROMPTS = (
    "请用一句中文回答：2 + 2 等于几？",
    "Answer in one short English sentence: what color is the sky on a clear day?",
    "用一句话介绍杭州，不要超过20个字。",
)
DEFAULT_EXPECTATIONS = {
    DEFAULT_PROMPTS[0]: ("4", "四"),
    DEFAULT_PROMPTS[1]: ("blue", "Blue"),
    DEFAULT_PROMPTS[2]: ("杭州",),
}
DSV4_V0_BF16_TOGGLE = "MINISGL_DSV4_SM80_V0_BF16"
DSV4_V1_MOE_TOGGLE = "MINISGL_DSV4_SM80_V1_MOE"
DSV4_MOE_V2_TOGGLE = "MINISGL_DSV4_SM80_MOE_V2"
DSV4_MOE_VLLM_RUNNER_TOGGLE = "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER"
DSV4_MOE_REDUCE_BF16_TOGGLE = "MINISGL_DSV4_SM80_MOE_REDUCE_BF16"
DSV4_MOE_EXPERT_BACKEND_ENV = "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"
DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16 = "marlin_wna16"
DSV4_MARLIN_WNA16_PREBUILD_ENV = "MINISGL_DSV4_MARLIN_WNA16_PREBUILD"
DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
)
DSV4_MARLIN_WNA16_RELEASE_TIMING_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING"
DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT"
)
DSV4_MARLIN_WNA16_QUARANTINE_BLOCKS_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS"
)
DSV4_MARLIN_WNA16_QUARANTINE_BYTES_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_BYTES"
)
DSV4_MARLIN_WNA16_QUARANTINE_PATTERN_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_PATTERN"
)
DSV4_MARLIN_WNA16_GUARD_INTEGRITY_DEBUG_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_GUARD_INTEGRITY_DEBUG"
)
DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC_ENV = "MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC"
DSV4_HC_TOGGLE = "MINISGL_DSV4_SM80_HC"
DSV4_RMSNORM_TOGGLE = "MINISGL_DSV4_SM80_RMSNORM"
DSV4_FP8_GEMM_TOGGLE = "MINISGL_DSV4_SM80_FP8_GEMM"
DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE = "MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT"
DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE = "MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE"
DSV4_FUSED_Q_KV_RMSNORM_TOGGLE = "MINISGL_DSV4_SM80_FUSED_Q_KV_RMSNORM"
DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE = "MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE"
DSV4_Q_WQA_FP8_GEMM_TOGGLE = "MINISGL_DSV4_SM80_Q_WQA_FP8_GEMM"
DSV4_Q_WQB_FP8_GEMM_TOGGLE = "MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM"
DSV4_WO_B_FP8_GEMM_TOGGLE = "MINISGL_DSV4_SM80_WO_B_FP8_GEMM"
DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE = "MINISGL_DSV4_SM80_INDEXER_WQB_FP8_GEMM"
DSV4_SHARED_FP8_GEMM_TOGGLE = "MINISGL_DSV4_SM80_SHARED_FP8_GEMM"
DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE = "MINISGL_DSV4_SM80_GATE_FP32_WEIGHT_CACHE"
DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE = (
    "MINISGL_DSV4_SM80_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE"
)
DSV4_WO_A_TOGGLE = "MINISGL_DSV4_SM80_WO_A_BF16"
DSV4_GLOBAL_TOPK_LENS_TOGGLE = "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS"
DSV4_SPARSE_SPLITK_BF16_TOGGLE = "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16"
DSV4_REPLAY_METADATA_COPY_TOGGLE = "MINISGL_DSV4_SM80_REPLAY_METADATA_COPY"
DSV4_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE = "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS"
DSV4_DIRECT_GRAPH_METADATA_GROUPS_ENV = "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS"
DSV4_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_TOGGLE = (
    "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE"
)
DSV4_SWA_INDEPENDENT_LIFECYCLE_ENV = "MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE"
DSV4_SWA_METADATA_PAGE_TABLE_CACHE_ENV = "MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE"
DSV4_SWA_DIRECT_TOKEN_METADATA_ENV = "MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA"
DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED_ENV = (
    "MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED"
)
DSV4_INDEXER_FP8_CACHE_TOGGLE = "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE"
DSV4_FP8_ACT_QUANT_TRITON_TOGGLE = "MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON"
DSV4_STATIC_SCALE_CACHE_TOGGLE = "MINISGL_DSV4_SM80_STATIC_SCALE_CACHE"
DSV4_BF16_PROJECTION_CACHE_TOGGLE = "MINISGL_DSV4_SM80_BF16_PROJECTION_CACHE"
DSV4_A100_VICTORY_BUNDLE_TOGGLE = "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"
DSV4_DECODE_METADATA_DEFOREST_TOGGLE = "MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST"
DSV4_PREP_METADATA_IN_GRAPH_TOGGLE = "MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH"
DSV4_PREP_METADATA_IN_GRAPH_ORACLE_ENV = "MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH_ORACLE"
DSV4_DISABLE_RELEASE_DEFAULTS_ENV = "MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS"
DSV4_HC_GRAPH_CLEANUP_TOGGLE = "MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP"
DSV4_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE = "MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE"
DSV4_WO_B_BF16_WEIGHT_CACHE_TOGGLE = "MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE"
DSV4_WO_A_BF16_BMM_CACHE_TOGGLE = "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE"
DSV4_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE = "MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE"
DSV4_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE = "MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE"
DSV4_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE = "MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE"
DSV4_DENSE_FP8_MARLIN_PROJECTION_TOGGLE = "MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION"
DSV4_VLLM_FP8_MARLIN_PROJECTION_TOGGLE = "MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION"
DSV4_PROMOTED_ROUTE_B_LIFETIME_VARIANT = "dsv4_sm80_a100_victory_prefix_routeb_lifetime"
DSV4_ROUTE_B_LIFETIME_MOE_REDUCE_BF16_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16"
)
DSV4_ROUTE_B_LIFETIME_MOE_REDUCE_BF16_INGRAPH_METADATA_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16_ingraphmetadata"
)
DSV4_ROUTE_B_LIFETIME_SWA_INDEPENDENT_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent"
)
DSV4_ROUTE_B_LIFETIME_LEGACY_VARIANT = (
    "dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime"
)
DSV4_A100_MARLIN_PREBUILD_VARIANT = "dsv4_sm80_a100_victory_marlin_prebuild"
DSV4_A100_MARLIN_RELEASE_VARIANT = "dsv4_sm80_a100_victory_marlin_release"
DSV4_RELEASE_DEFAULT_VARIANT = "dsv4_sm80_release_default"
DSV4_A100_MARLIN_RELEASE_SAFE_ARENA_VARIANT = (
    "dsv4_sm80_a100_victory_marlin_release_safe_arena"
)
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release"
)
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_INDEPENDENT_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent"
)
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_DIRECT_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent_swadirect"
)
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_REPLAY_METADATA_FUSED_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent_swadirect_replaymetafused"
)
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SAFE_ARENA_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_safe_arena"
)
DSV4_ROUTE_B_LIFETIME_ENV = {
    DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
    DSV4_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE: "1",
    DSV4_DIRECT_GRAPH_METADATA_GROUPS_ENV: "c4",
    DSV4_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_TOGGLE: "1",
    DSV4_PREP_METADATA_IN_GRAPH_TOGGLE: "1",
}
DSV4_A100_MARLIN_PREBUILD_ENV = {
    DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
    DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
    DSV4_MARLIN_WNA16_PREBUILD_ENV: "1",
}
DSV4_A100_MARLIN_RELEASE_ENV = {
    **DSV4_A100_MARLIN_PREBUILD_ENV,
    DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS_ENV: "1",
    DSV4_MARLIN_WNA16_RELEASE_TIMING_ENV: "before_kv_alloc",
    DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT_ENV: "1",
    DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC_ENV: "component",
}
DSV4_A100_MARLIN_RELEASE_SAFE_ARENA_ENV = {
    **DSV4_A100_MARLIN_RELEASE_ENV,
    DSV4_MARLIN_WNA16_QUARANTINE_BLOCKS_ENV: "1",
    DSV4_MARLIN_WNA16_QUARANTINE_BYTES_ENV: "3.1875GiB",
    DSV4_MARLIN_WNA16_QUARANTINE_PATTERN_ENV: "deterministic",
    DSV4_MARLIN_WNA16_GUARD_INTEGRITY_DEBUG_ENV: "1",
}
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_ENV = {
    **DSV4_ROUTE_B_LIFETIME_ENV,
    **DSV4_A100_MARLIN_RELEASE_ENV,
}
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_INDEPENDENT_ENV = {
    **DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_ENV,
    DSV4_SWA_INDEPENDENT_LIFECYCLE_ENV: "1",
}
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_DIRECT_ENV = {
    **DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_INDEPENDENT_ENV,
    DSV4_SWA_METADATA_PAGE_TABLE_CACHE_ENV: "1",
    DSV4_SWA_DIRECT_TOKEN_METADATA_ENV: "1",
}
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_REPLAY_METADATA_FUSED_ENV = {
    **DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_DIRECT_ENV,
    DSV4_DIRECT_GRAPH_METADATA_GROUPS_ENV: "swa,c4",
    DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED_ENV: "1",
}
DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SAFE_ARENA_ENV = {
    **DSV4_ROUTE_B_LIFETIME_ENV,
    **DSV4_A100_MARLIN_RELEASE_SAFE_ARENA_ENV,
}
DSV4_ROUTE_B_LIFETIME_SWA_INDEPENDENT_ENV = {
    **DSV4_ROUTE_B_LIFETIME_ENV,
    DSV4_SWA_INDEPENDENT_LIFECYCLE_ENV: "1",
}
BASELINE_TP_SIZE = 8


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]
    description: str
    use_pynccl: bool = False
    allow_dsv4_cuda_graph: bool = False
    cuda_graph_capture_greedy_sample: bool = False
    enable_dsv4_radix_prefix_cache: bool = False
    enable_dsv4_component_loc_ownership: bool = False
    enable_dsv4_swa_independent_lifecycle: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "fallback",
        {DSV4_DISABLE_RELEASE_DEFAULTS_ENV: "1"},
        "All MINISGL_DSV4_SM80_* toggles cleared; release defaults disabled.",
    ),
    Variant(
        DSV4_RELEASE_DEFAULT_VARIANT,
        {},
        (
            "TARGET 12.52 release-default smoke: leave DSV4 env empty so "
            "Engine injects the A100/sm80 release bundle with SWA independent "
            "lifecycle and SWA direct replay metadata."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant("v0_bf16", {DSV4_V0_BF16_TOGGLE: "1"}, "TARGET 05.7 v0 BF16 whitelist bundle."),
    Variant(
        "v1_moe",
        {DSV4_V1_MOE_TOGGLE: "1"},
        "V1 exact grouped MoE bundle: v0 BF16 whitelist plus grouped MoE route.",
    ),
    Variant(
        "v1_moe_v2",
        {DSV4_V1_MOE_TOGGLE: "1", DSV4_MOE_V2_TOGGLE: "1"},
        (
            "V2 exact MoE boundary: V1 exact grouped MoE plus explicit route "
            "execution plan and per-layer grouped-MoE workspace."
        ),
    ),
    Variant(
        "v1_moe_pynccl",
        {DSV4_V1_MOE_TOGGLE: "1"},
        "V1 exact grouped MoE with PyNCCL tensor-parallel collectives.",
        use_pynccl=True,
    ),
    Variant(
        "v1_moe_graph",
        {DSV4_V1_MOE_TOGGLE: "1"},
        "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture.",
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc",
        {DSV4_V1_MOE_TOGGLE: "1", DSV4_HC_TOGGLE: "1"},
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture "
            "and experimental sm80 HC split/post helpers."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC helpers, and experimental sm80 RMSNorm helper."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_fp8gemm",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FP8_GEMM_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, and experimental sm80 FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_wqb_fp8gemm",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, and selective attention wq_b FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_wqb_woa",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_A_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b FP8 GEMM, "
            "and selective attention wo_a projection."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b FP8 GEMM, "
            "and selective attention wo_b FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, and selective indexer wq_b FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_SHARED_FP8_GEMM_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, and selective shared-expert "
            "FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, and exact gate fp32 "
            "weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, exact gate fp32 weight "
            "caching, and exact indexer-store norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, vLLM-aligned shared-activation "
            "attention wq_a/wkv FP8 projection, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, exact gate fp32 weight "
            "caching, and exact indexer-store norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        ("v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_" "gatecache_idxstorecache"),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, vLLM-aligned shared-activation "
            "attention wq_a/wkv FP8 projection, vLLM-aligned fused q norm/rope "
            "plus KV norm/rope/cache-store, selective attention wq_b/wo_b FP8 "
            "GEMM, selective indexer wq_b FP8 GEMM, exact gate fp32 weight "
            "caching, and exact indexer-store norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        ("v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_" "gatecache_idxstorecache"),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, vLLM-aligned shared-activation "
            "attention wq_a/wkv FP8 projection with cached fused bf16 weights, "
            "vLLM-aligned fused q norm/rope plus KV norm/rope/cache-store, "
            "selective attention wq_b/wo_b FP8 GEMM, selective indexer wq_b "
            "FP8 GEMM, exact gate fp32 weight caching, and exact indexer-store "
            "norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        (
            "v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_"
            "gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "the current vLLM-aligned cached fused wq_a/wkv graph path, and "
            "greedy sampler captured in the graph."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_"
            "gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        "Current best exact smoke variant with the V2 MoE execution boundary enabled.",
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "Mini-owned vLLM-shaped exact MoE runner wrapping the current grouped "
            "FP4 W13/SwiGLU/W2 backend, with the current exact graph smoke bundle."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "TARGET 07.391 mini-owned Marlin WNA16 backend smoke variant. "
            "This is explicit opt-in and uses cached MXFP4-to-Marlin expert "
            "weights after the first transformed call."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
        },
        (
            "TARGET 07.394 exact bf16 Marlin WNA16 smoke path with opt-in "
            "global topk/lens consolidation."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
        },
        (
            "TARGET 07.395 exact bf16 Marlin WNA16 smoke path with global "
            "topk/lens and opt-in bf16 split-K sparse decode."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_graph_"
            "hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_"
            "idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
            DSV4_REPLAY_METADATA_COPY_TOGGLE: "1",
        },
        (
            "TARGET 07.41 exact bf16 smoke path with split-K sparse decode "
            "and opt-in fused decode replay metadata staging."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
            DSV4_REPLAY_METADATA_COPY_TOGGLE: "1",
            DSV4_INDEXER_FP8_CACHE_TOGGLE: "1",
        },
        (
            "TARGET 07.50 opt-in FP8 indexer cache/logits smoke path on the "
            "07.41 split-K/metacopy stack. MLA/SWA cache precision remains bf16."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_"
            "wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
            DSV4_REPLAY_METADATA_COPY_TOGGLE: "1",
            DSV4_INDEXER_FP8_CACHE_TOGGLE: "1",
            DSV4_FP8_ACT_QUANT_TRITON_TOGGLE: "1",
        },
        (
            "TARGET 07.54 graph-layout PoC smoke path with Triton fused FP8 "
            "activation fake-quant staging."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_qwqbbf16cache_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
            DSV4_REPLAY_METADATA_COPY_TOGGLE: "1",
            DSV4_INDEXER_FP8_CACHE_TOGGLE: "1",
            DSV4_FP8_ACT_QUANT_TRITON_TOGGLE: "1",
            DSV4_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "TARGET 07.58 q_wqb-only cached BF16 dequantized weight projection "
            "path on top of the promoted 07.54 graph-layout stack."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
            DSV4_REPLAY_METADATA_COPY_TOGGLE: "1",
            DSV4_INDEXER_FP8_CACHE_TOGGLE: "1",
            DSV4_FP8_ACT_QUANT_TRITON_TOGGLE: "1",
            DSV4_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_WO_B_BF16_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "TARGET 07.59 q_wqb plus row-parallel wo_b cached BF16 "
            "dequantized weight projection smoke path."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_"
            "graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_"
            "gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
            DSV4_REPLAY_METADATA_COPY_TOGGLE: "1",
            DSV4_INDEXER_FP8_CACHE_TOGGLE: "1",
            DSV4_FP8_ACT_QUANT_TRITON_TOGGLE: "1",
            DSV4_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_WO_B_BF16_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "TARGET 07.60 q_wqb, row-parallel wo_b, and indexer.wq_b cached "
            "BF16 dequantized weight projection smoke path."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory",
        {DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1"},
        (
            "Milestone smoke bundle for the TARGET 07.66 A100/sm80 victory "
            "stack: Marlin WNA16 MoE, graph replay, FP8 indexer cache, split-K "
            "sparse decode, four attention/indexer BF16 projection caches, "
            "and shared expert BF16 projection caches."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        DSV4_A100_MARLIN_PREBUILD_VARIANT,
        dict(DSV4_A100_MARLIN_PREBUILD_ENV),
        (
            "TARGET 08.35 smoke diagnostic: dsv4_sm80_a100_victory with "
            "backend fixed to marlin_wna16 and routed expert Marlin WNA16 "
            "caches prebuilt before KV capacity planning, retaining original "
            "routed FP4 expert tensors."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        DSV4_A100_MARLIN_RELEASE_VARIANT,
        dict(DSV4_A100_MARLIN_RELEASE_ENV),
        (
            "TARGET 08.35 smoke high-memory-efficiency preset: "
            "dsv4_sm80_a100_victory with backend fixed to marlin_wna16, "
            "routed expert caches prebuilt before KV capacity planning, and "
            "original routed FP4 expert weights/scales released after full "
            "prebuild succeeds."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        DSV4_A100_MARLIN_RELEASE_SAFE_ARENA_VARIANT,
        dict(DSV4_A100_MARLIN_RELEASE_SAFE_ARENA_ENV),
        (
            "TARGET 08.38 smoke candidate: Marlin WNA16 prebuild plus "
            "before-KV release capacity credit, with a 3.1875 GiB/rank "
            "sentinel guard arena held out of the released ranges."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory_fp8marlinproj",
        {
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_VLLM_FP8_MARLIN_PROJECTION_TOGGLE: "1",
        },
        (
            "Legacy TARGET 07.74 smoke alias: dsv4_sm80_a100_victory with "
            "the mini-owned dense FP8 Marlin W8A16 block projection runtime "
            "for attention q_wqb, attention wo_b local projection, and shared "
            "experts down."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory_densefp8marlinproj",
        {
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_DENSE_FP8_MARLIN_PROJECTION_TOGGLE: "1",
        },
        (
            "TARGET 07.76 smoke opt-in: dsv4_sm80_a100_victory with "
            "mini-owned dense FP8 Marlin W8A16 block linear for attention "
            "q_wqb, attention wo_b local projection, and shared experts down."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory_hccleanup",
        {
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_HC_GRAPH_CLEANUP_TOGGLE: "1",
        },
        (
            "TARGET 07.68 smoke opt-in: dsv4_sm80_a100_victory plus fused "
            "HC prenorm/split boundary cleanup."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory_bf16smallgemm",
        {
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE: "1",
        },
        (
            "TARGET 07.70 opt-in: dsv4_sm80_a100_victory plus pretransposed "
            "cached BF16 weights for small-M projection GEMMs."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory_sharedbf16",
        {
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "TARGET 07.66 smoke audit variant: dsv4_sm80_a100_victory with "
            "explicit shared expert gate/up and down cached BF16 dequantized "
            "weight projection env."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory_metadatadeforest",
        {
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_DECODE_METADATA_DEFOREST_TOGGLE: "1",
        },
        (
            "TARGET 07.64 smoke opt-in: dsv4_sm80_a100_victory plus fused "
            "decode metadata indices/lens assembly. Under Route B component "
            "loc ownership this uses component-owned page tables."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "dsv4_sm80_a100_victory_directgraphmetadata",
        {
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE: "1",
        },
        (
            "TARGET 08.25 smoke opt-in: Route B graph replay direct C4 sparse "
            "metadata generation into captured graph buffers."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        DSV4_PROMOTED_ROUTE_B_LIFETIME_VARIANT,
        dict(DSV4_ROUTE_B_LIFETIME_ENV),
        (
            "TARGET 08.29 promoted Route B prefix preset: A100 victory bundle, "
            "direct C4 graph metadata buffers, in-graph replay metadata prep, "
            "and request-slot keyed component page-table lifetime caching. "
            "Pair with --enable-dsv4-radix-prefix-cache, "
            "--enable-dsv4-component-loc-ownership, --page-size 256, "
            "--num-pages 128, and graph buckets 1 2 4 8 16."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_VARIANT,
        dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_ENV),
        (
            "TARGET 08.35 prefix smoke high-memory-efficiency preset: "
            "promoted Route B lifetime prefix preset plus backend fixed to "
            "marlin_wna16, MoE Marlin WNA16 prebuild before KV capacity "
            "planning, and original routed FP4 expert weights/scales release "
            "after successful prebuild."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_INDEPENDENT_VARIANT,
        dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_INDEPENDENT_ENV),
        (
            "TARGET 08.31/08.40 compatibility smoke preset: Route B lifetime "
            "plus Marlin WNA16 before-KV release capacity credit, component-slot "
            "clear on page allocation, and independent SWA lifecycle."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_DIRECT_VARIANT,
        dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_DIRECT_ENV),
        (
            "TARGET 08.50/12.50 smoke preset: Route B lifetime plus Marlin "
            "WNA16 release, independent SWA lifecycle, SWA page-table cache, "
            "and direct token-level SWA metadata."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_REPLAY_METADATA_FUSED_VARIANT,
        dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_REPLAY_METADATA_FUSED_ENV),
        (
            "TARGET 08.54/12.50 smoke preset: Marlin release plus independent "
            "SWA direct token metadata and fused replay metadata for direct "
            "SWA graph index buffers."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SAFE_ARENA_VARIANT,
        dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SAFE_ARENA_ENV),
        (
            "TARGET 08.38 prefix smoke candidate: Route B lifetime prefix "
            "preset plus Marlin WNA16 before-KV release capacity credit and "
            "a 3.1875 GiB/rank sentinel guard arena."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        DSV4_ROUTE_B_LIFETIME_MOE_REDUCE_BF16_VARIANT,
        {**DSV4_ROUTE_B_LIFETIME_ENV, DSV4_MOE_REDUCE_BF16_TOGGLE: "1"},
        (
            "TARGET 10.27 promoted A100/sm80 communication preset: promoted "
            "Route B lifetime preset plus BF16 MoE reduce-once input and "
            "default PyNCCL threshold32m tensor-parallel collectives."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        DSV4_ROUTE_B_LIFETIME_MOE_REDUCE_BF16_INGRAPH_METADATA_VARIANT,
        {
            **DSV4_ROUTE_B_LIFETIME_ENV,
            DSV4_MOE_REDUCE_BF16_TOGGLE: "1",
            DSV4_PREP_METADATA_IN_GRAPH_TOGGLE: "1",
        },
        (
            "Legacy TARGET 12.4 smoke alias: TARGET10.27 Route B lifetime "
            "BF16 MoE-reduce preset with in-graph decode metadata "
            "materialization now included by the Route B lifetime env."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        DSV4_ROUTE_B_LIFETIME_SWA_INDEPENDENT_VARIANT,
        dict(DSV4_ROUTE_B_LIFETIME_SWA_INDEPENDENT_ENV),
        (
            "TARGET 08.31 smoke opt-in: promoted Route B lifetime prefix preset "
            "plus independent SWA lifecycle. Pair with --enable-dsv4-radix-prefix-cache, "
            "--enable-dsv4-component-loc-ownership, and "
            "--enable-dsv4-swa-independent-lifecycle."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        DSV4_ROUTE_B_LIFETIME_LEGACY_VARIANT,
        dict(DSV4_ROUTE_B_LIFETIME_ENV),
        (
            "Historical TARGET 08.27/08.28 diagnostic alias for "
            f"{DSV4_PROMOTED_ROUTE_B_LIFETIME_VARIANT}. Kept for artifact and "
            "script reproduction; new runs should use the promoted preset name."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        "target0762_woabf16bmmcache",
        {DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1"},
        ("Legacy alias for dsv4_sm80_a100_victory kept for TARGET 07.62 " "artifacts and scripts."),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        (
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_scalecache_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN_WNA16,
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_SHARED_ACT_TOGGLE: "1",
            DSV4_FUSED_WQA_WKV_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_FUSED_Q_KV_NORM_ROPE_STORE_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_GLOBAL_TOPK_LENS_TOGGLE: "1",
            DSV4_SPARSE_SPLITK_BF16_TOGGLE: "1",
            DSV4_REPLAY_METADATA_COPY_TOGGLE: "1",
            DSV4_INDEXER_FP8_CACHE_TOGGLE: "1",
            DSV4_FP8_ACT_QUANT_TRITON_TOGGLE: "1",
            DSV4_STATIC_SCALE_CACHE_TOGGLE: "1",
        },
        ("TARGET 07.56 low-cost preflight smoke path with static FP32 " "projection scale cache."),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache",
        {
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQA_FP8_GEMM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_a/wq_b/"
            "wo_b FP8 GEMM, selective indexer wq_b FP8 GEMM, exact gate fp32 "
            "weight caching, and exact indexer-store norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        "v1_moe_graph_sample",
        {DSV4_V1_MOE_TOGGLE: "1"},
        (
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture "
            "and greedy sampler captured in the graph."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        "v1_moe_graph_pynccl",
        {DSV4_V1_MOE_TOGGLE: "1"},
        "V1 exact grouped MoE with PyNCCL and opt-in DSV4 decode CUDA graph capture.",
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
    ),
)


def _variant_map() -> dict[str, Variant]:
    return {variant.name: variant for variant in VARIANTS}


def _all_dsv4_sm80_env_names(dsv4_kernel) -> list[str]:
    names = set(getattr(dsv4_kernel, "DSV4_SM80_KNOWN_TOGGLES", ()))
    names.add(DSV4_SWA_INDEPENDENT_LIFECYCLE_ENV)
    names.add(DSV4_SWA_METADATA_PAGE_TABLE_CACHE_ENV)
    names.add(DSV4_SWA_DIRECT_TOKEN_METADATA_ENV)
    names.add(DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED_ENV)
    names.update(name for name in os.environ if name.startswith("MINISGL_DSV4_SM80_"))
    return sorted(names)


def _preserved_dsv4_sm80_env_names(dsv4_kernel) -> tuple[str, ...]:
    return (
        getattr(
            dsv4_kernel,
            "DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY_TOGGLE",
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY",
        ),
        DSV4_PREP_METADATA_IN_GRAPH_ORACLE_ENV,
        getattr(
            dsv4_kernel,
            "DSV4_MARLIN_WNA16_KV_SENTINEL_DEBUG_ENV",
            "MINISGL_DSV4_MARLIN_WNA16_KV_SENTINEL_DEBUG",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_MARLIN_WNA16_KV_SENTINEL_BYTES_ENV",
            "MINISGL_DSV4_MARLIN_WNA16_KV_SENTINEL_BYTES",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_MARLIN_WNA16_POISON_THEN_FREE_ENV",
            "MINISGL_DSV4_MARLIN_WNA16_DEBUG_POISON_THEN_FREE",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_MARLIN_WNA16_POISON_THEN_FREE_BYTES_ENV",
            "MINISGL_DSV4_MARLIN_WNA16_DEBUG_POISON_THEN_FREE_BYTES",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_MARLIN_WNA16_POISON_THEN_FREE_PATTERN_ENV",
            "MINISGL_DSV4_MARLIN_WNA16_DEBUG_POISON_THEN_FREE_PATTERN",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_MARLIN_WNA16_RELEASE_LAYER_FILTER_ENV",
            "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_LAYER_FILTER",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC_ENV",
            "MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC",
        ),
    )


def configure_variant(dsv4_kernel, variant: Variant) -> dict[str, Any]:
    preserved = {
        name: os.environ[name]
        for name in _preserved_dsv4_sm80_env_names(dsv4_kernel)
        if name in os.environ
    }
    cleared = sorted(
        set(_all_dsv4_sm80_env_names(dsv4_kernel)) | {DSV4_DISABLE_RELEASE_DEFAULTS_ENV}
    )
    for name in cleared:
        os.environ.pop(name, None)
    for name, value in variant.env.items():
        os.environ[name] = value
    for name, value in preserved.items():
        os.environ[name] = value
    return {
        "cleared_dsv4_sm80_env": cleared,
        "preserved_dsv4_sm80_env": preserved,
        "active_dsv4_toggles": active_dsv4_toggles(dsv4_kernel),
        "raw_dsv4_sm80_env": raw_dsv4_env(),
    }


def active_dsv4_toggles(dsv4_kernel) -> list[str]:
    return [
        name for name in _all_dsv4_sm80_env_names(dsv4_kernel) if dsv4_kernel.dsv4_env_flag(name)
    ]


def raw_dsv4_env() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in sorted(os.environ)
        if name.startswith("MINISGL_DSV4_")
    }


def _runtime_variant_env(dsv4_kernel, variant: Variant) -> dict[str, Any]:
    if variant.name == DSV4_RELEASE_DEFAULT_VARIANT and not variant.env:
        return {
            "cleared_dsv4_sm80_env": [],
            "preserved_dsv4_sm80_env": {},
            "active_dsv4_toggles": active_dsv4_toggles(dsv4_kernel),
            "raw_dsv4_sm80_env": raw_dsv4_env(),
        }
    return configure_variant(dsv4_kernel, variant)


def load_dsv4_encoding(model_path: str):
    path = Path(model_path) / "encoding" / "encoding_dsv4.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("deepseek_v4_encoding_dsv4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load DeepSeek V4 encoding from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_chat_prompt(
    prompt: str,
    *,
    model_path: str,
    system_prompt: str,
    thinking_mode: str,
) -> str:
    encoding = load_dsv4_encoding(model_path)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if encoding is not None:
        return encoding.encode_messages(messages, thinking_mode=thinking_mode)
    if system_prompt:
        return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"


def parse_completion_text(
    raw_text: str,
    *,
    model_path: str,
    thinking_mode: str,
) -> dict[str, Any] | None:
    encoding = load_dsv4_encoding(model_path)
    if encoding is None:
        return None
    text = raw_text
    if getattr(encoding, "eos_token", "") and encoding.eos_token not in text:
        text = text + encoding.eos_token
    try:
        return encoding.parse_message_from_completion_text(text, thinking_mode=thinking_mode)
    except Exception as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}"}


def text_sanity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    non_ws = [ch for ch in text if not ch.isspace()]
    content_chars = [ch for ch in non_ws if ch.isalnum()]
    printable = [ch for ch in text if ch.isprintable() or ch in "\n\r\t"]
    max_run = 0
    max_symbol_run = 0
    current_run = 0
    current_symbol_run = 0
    last = None
    for ch in text:
        if ch == last:
            current_run += 1
        else:
            last = ch
            current_run = 1
        max_run = max(max_run, current_run)
        if ch == last and not ch.isalnum() and not ch.isspace():
            current_symbol_run = current_run
        else:
            current_symbol_run = 0
        max_symbol_run = max(max_symbol_run, current_symbol_run)

    issues = []
    if not stripped:
        issues.append("empty_output")
    if "\ufffd" in text:
        issues.append("replacement_character")
    if "\x00" in text:
        issues.append("nul_character")
    if non_ws and len(printable) / max(len(text), 1) < 0.95:
        issues.append("many_non_printable_chars")
    if non_ws and len(content_chars) / len(non_ws) < 0.25:
        issues.append("mostly_punctuation_or_symbols")
    if max_run >= 12:
        issues.append("long_repeated_character_run")
    if max_symbol_run >= 6:
        issues.append("long_repeated_symbol_run")
    for token in ("<｜begin", "<｜User｜>", "<｜Assistant｜>", "<|"):
        if token in text:
            issues.append(f"special_token_leak:{token}")
            break

    return {
        "looks_sane": len(issues) == 0,
        "issues": issues,
        "char_count": len(text),
        "non_whitespace_char_count": len(non_ws),
        "content_char_fraction": len(content_chars) / max(len(non_ws), 1),
        "printable_fraction": len(printable) / max(len(text), 1),
        "max_repeated_char_run": max_run,
        "max_repeated_symbol_run": max_symbol_run,
    }


def _normalize_for_overlap(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def response_sanity(
    text: str,
    *,
    prompt: str,
    expected_substrings: Sequence[str] | None = None,
) -> dict[str, Any]:
    report = text_sanity(text)
    issues = list(report["issues"])
    prompt_norm = _normalize_for_overlap(prompt)
    text_norm = _normalize_for_overlap(text)
    overlap_fraction = 0.0
    if prompt_norm and len(text_norm) >= 6:
        match = difflib.SequenceMatcher(None, prompt_norm, text_norm).find_longest_match(
            0, len(prompt_norm), 0, len(text_norm)
        )
        overlap_fraction = match.size / max(len(text_norm), 1)
        if overlap_fraction >= 0.7:
            issues.append("prompt_echo_like")
    if expected_substrings and not any(expected in text for expected in expected_substrings):
        issues.append("missing_expected_substring")

    report["issues"] = issues
    report["looks_sane"] = len(issues) == 0
    report["prompt_overlap_fraction"] = overlap_fraction
    report["expected_substrings"] = list(expected_substrings or ())
    return report


def _tp_rank_size(args: argparse.Namespace) -> tuple[int, int, int]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
    tp_size = args.tensor_parallel_size or env_world_size
    tp_rank = args.tp_rank if args.tp_rank is not None else env_local_rank
    if tp_size <= 0:
        raise ValueError("tensor parallel size must be positive")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid TP rank {tp_rank} for TP size {tp_size}")
    return tp_rank, tp_size, env_world_size


def _distributed_init_method(args: argparse.Namespace, tp_size: int) -> str | None:
    if args.distributed_init_method is not None:
        return args.distributed_init_method
    if tp_size > 1 and "MASTER_ADDR" in os.environ:
        return "env://"
    return None


def _runtime_options(args: argparse.Namespace, variants: Sequence[Variant]) -> dict[str, Any]:
    variant_pynccl = any(variant.use_pynccl for variant in variants)
    variant_graph = any(variant.allow_dsv4_cuda_graph for variant in variants)
    variant_graph_greedy_sample = any(
        variant.cuda_graph_capture_greedy_sample for variant in variants
    )
    variant_prefix = any(variant.enable_dsv4_radix_prefix_cache for variant in variants)
    variant_component = any(variant.enable_dsv4_component_loc_ownership for variant in variants)
    variant_swa_independent = any(
        variant.enable_dsv4_swa_independent_lifecycle for variant in variants
    )
    if (
        variant_pynccl
        and not all(variant.use_pynccl for variant in variants)
        and not args.use_pynccl
    ):
        raise SystemExit("PyNCCL text-smoke variants must be run separately or with --use-pynccl.")
    if (
        variant_graph
        and not all(variant.allow_dsv4_cuda_graph for variant in variants)
        and not args.allow_dsv4_cuda_graph
    ):
        raise SystemExit(
            "DSV4 CUDA graph text-smoke variants must be run separately or with "
            "--allow-dsv4-cuda-graph."
        )
    allow_dsv4_cuda_graph = bool(args.allow_dsv4_cuda_graph or variant_graph)
    if getattr(args, "disable_dsv4_cuda_graph", False):
        allow_dsv4_cuda_graph = False
    cuda_graph_bs = args.cuda_graph_bs
    if allow_dsv4_cuda_graph and cuda_graph_bs is None:
        cuda_graph_bs = [1, 2, 4, 8, 16]
    if not allow_dsv4_cuda_graph:
        cuda_graph_bs = []
    if getattr(args, "disable_cuda_graph_greedy_sample", False):
        variant_graph_greedy_sample = False
    if (
        variant_prefix
        and not all(variant.enable_dsv4_radix_prefix_cache for variant in variants)
        and not args.enable_dsv4_radix_prefix_cache
    ):
        raise SystemExit(
            "DSV4 radix-prefix variants must be run separately or with "
            "--enable-dsv4-radix-prefix-cache."
        )
    if (
        variant_component
        and not all(variant.enable_dsv4_component_loc_ownership for variant in variants)
        and not args.enable_dsv4_component_loc_ownership
    ):
        raise SystemExit(
            "DSV4 component-ownership variants must be run separately or with "
            "--enable-dsv4-component-loc-ownership."
        )
    if (
        variant_swa_independent
        and not all(variant.enable_dsv4_swa_independent_lifecycle for variant in variants)
        and not args.enable_dsv4_swa_independent_lifecycle
    ):
        raise SystemExit(
            "DSV4 SWA independent lifecycle variants must be run separately or with "
            "--enable-dsv4-swa-independent-lifecycle."
        )
    enable_dsv4_radix_prefix_cache = bool(
        args.enable_dsv4_radix_prefix_cache or variant_prefix
    )
    enable_dsv4_component_loc_ownership = bool(
        args.enable_dsv4_component_loc_ownership or variant_component
    )
    enable_dsv4_swa_independent_lifecycle = bool(
        args.enable_dsv4_swa_independent_lifecycle or variant_swa_independent
    )
    return {
        "use_pynccl": bool(args.use_pynccl or variant_pynccl),
        "allow_dsv4_cuda_graph": allow_dsv4_cuda_graph,
        "cuda_graph_bs": cuda_graph_bs,
        "cuda_graph_capture_greedy_sample": variant_graph_greedy_sample,
        "enable_dsv4_radix_prefix_cache": enable_dsv4_radix_prefix_cache,
        "enable_dsv4_component_loc_ownership": enable_dsv4_component_loc_ownership,
        "enable_dsv4_swa_independent_lifecycle": enable_dsv4_swa_independent_lifecycle,
    }


def _graph_init_variant(variants: Sequence[Variant]) -> Variant:
    for variant in variants:
        if variant.env.get(DSV4_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return variant
    return variants[0]


def _gather_payloads(group, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size(group=group)
        gathered: list[Any] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, payload, group=group)
        return list(gathered)
    return [payload]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_variant(
    *,
    args: argparse.Namespace,
    llm,
    dsv4_kernel,
    variant: Variant,
    prompts: Sequence[str],
    formatted_prompts: Sequence[str],
    rank: int,
    tp_size: int,
    runtime_options: dict[str, Any],
) -> dict[str, Any] | None:
    from minisgl.core import SamplingParams

    variant_env = _runtime_variant_env(dsv4_kernel, variant)
    llm.sync_all_ranks()
    tic = time.perf_counter()
    error = None
    outputs: list[dict[str, Any]] = []
    try:
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            ignore_eos=False,
            max_tokens=args.max_tokens,
        )
        generated = llm.generate(list(formatted_prompts), sampling)
        torch.cuda.synchronize(llm.device)
        elapsed_s = time.perf_counter() - tic
        if rank == 0:
            for idx, (prompt, formatted, item) in enumerate(
                zip(prompts, formatted_prompts, generated)
            ):
                token_ids = list(item["token_ids"])
                raw_text = llm.tokenizer.decode(token_ids, skip_special_tokens=False)
                clean_text = llm.tokenizer.decode(token_ids, skip_special_tokens=True)
                parsed = parse_completion_text(
                    raw_text,
                    model_path=args.model_path,
                    thinking_mode=args.thinking_mode,
                )
                expected_substrings = DEFAULT_EXPECTATIONS.get(prompt)
                outputs.append(
                    {
                        "index": idx,
                        "prompt": prompt,
                        "formatted_prompt_preview": formatted[:240],
                        "generated_token_count": len(token_ids),
                        "generated_token_ids": token_ids,
                        "raw_text": raw_text,
                        "text": clean_text,
                        "parsed": parsed,
                        "sanity": response_sanity(
                            clean_text or raw_text,
                            prompt=prompt,
                            expected_substrings=expected_substrings,
                        ),
                    }
                )
    except BaseException as exc:
        elapsed_s = time.perf_counter() - tic
        error = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }

    rank_payload = {
        "rank": rank,
        "elapsed_s": elapsed_s,
        "error": error,
        "memory": {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(llm.device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(llm.device)),
        },
    }
    gathered = _gather_payloads(llm.tp_cpu_group, rank_payload)
    if rank != 0:
        return None

    errors = [payload["error"] for payload in gathered if payload.get("error")]
    status = "fail" if errors else "pass"
    if status == "pass" and any(not item["sanity"]["looks_sane"] for item in outputs):
        status = "warn"
    return {
        "status": status,
        "variant": {
            "name": variant.name,
            "description": variant.description,
            **variant_env,
        },
        "elapsed_s": max(float(payload["elapsed_s"]) for payload in gathered),
        "outputs": outputs,
        "errors": errors,
        "per_rank": gathered,
        "config": {
            "tensor_parallel_size": tp_size,
            "page_size": args.page_size,
            "num_pages": args.num_pages,
            "use_pynccl": runtime_options["use_pynccl"],
            "allow_dsv4_cuda_graph": runtime_options["allow_dsv4_cuda_graph"],
            "cuda_graph_bs": runtime_options["cuda_graph_bs"],
            "enable_dsv4_radix_prefix_cache": runtime_options[
                "enable_dsv4_radix_prefix_cache"
            ],
            "enable_dsv4_swa_tail_retention_v1": args.enable_dsv4_swa_tail_retention_v1,
            "enable_dsv4_component_loc_ownership": runtime_options[
                "enable_dsv4_component_loc_ownership"
            ],
            "enable_dsv4_swa_independent_lifecycle": runtime_options[
                "enable_dsv4_swa_independent_lifecycle"
            ],
            "prefix_cache_metrics": llm.cache_manager.prefix_metrics_snapshot(),
            "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}),
            "model_prepare_report_rank0": getattr(llm.engine, "model_prepare_report", {}),
            "kv_capacity_plan_report": getattr(llm.engine, "kv_capacity_plan_report", {}),
            "distributed_init_method": _distributed_init_method(args, tp_size),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_seq_len": args.max_seq_len,
            "max_extend_tokens": args.max_extend_tokens,
            "thinking_mode": args.thinking_mode,
        },
    }


def run_text_smoke(args: argparse.Namespace) -> int:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import deepseek_v4 as dsv4_kernel
    from minisgl.llm import LLM

    rank, tp_size, env_world_size = _tp_rank_size(args)
    if env_world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={env_world_size} does not match tensor parallel size {tp_size}; "
            "launch with torchrun --standalone --nproc_per_node=8 for TP8."
        )
    distributed_init_method = _distributed_init_method(args, tp_size)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    formatted_prompts = [
        format_chat_prompt(
            prompt,
            model_path=args.model_path,
            system_prompt=args.system_prompt,
            thinking_mode=args.thinking_mode,
        )
        for prompt in prompts
    ]
    variants = [_variant_map()[name] for name in (args.variants or ["fallback", "v0_bf16"])]
    runtime_options = _runtime_options(args, variants)
    configure_variant(dsv4_kernel, _graph_init_variant(variants))

    llm = None
    reports: list[dict[str, Any]] = []
    try:
        llm_kwargs: dict[str, Any] = {}
        if distributed_init_method is not None:
            llm_kwargs["distributed_init_method"] = distributed_init_method
        llm = LLM(
            args.model_path,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(rank, tp_size),
            max_running_req=max(len(prompts), 1),
            max_seq_len_override=args.max_seq_len,
            max_extend_tokens=args.max_extend_tokens,
            num_page_override=args.num_pages,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            use_pynccl=runtime_options["use_pynccl"],
            allow_dsv4_cuda_graph=runtime_options["allow_dsv4_cuda_graph"],
            cuda_graph_bs=runtime_options["cuda_graph_bs"],
            cuda_graph_capture_greedy_sample=runtime_options["cuda_graph_capture_greedy_sample"],
            enable_dsv4_radix_prefix_cache=runtime_options[
                "enable_dsv4_radix_prefix_cache"
            ],
            enable_dsv4_swa_tail_retention_v1=args.enable_dsv4_swa_tail_retention_v1,
            enable_dsv4_component_loc_ownership=runtime_options[
                "enable_dsv4_component_loc_ownership"
            ],
            enable_dsv4_swa_independent_lifecycle=runtime_options[
                "enable_dsv4_swa_independent_lifecycle"
            ],
            **llm_kwargs,
        )
        for variant in variants:
            report = run_variant(
                args=args,
                llm=llm,
                dsv4_kernel=dsv4_kernel,
                variant=variant,
                prompts=prompts,
                formatted_prompts=formatted_prompts,
                rank=rank,
                tp_size=tp_size,
                runtime_options=runtime_options,
            )
            if rank == 0 and report is not None:
                reports.append(report)
                _write_json(
                    Path(args.output).with_suffix(f".{variant.name}.json"),
                    report,
                )
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException:
                if rank == 0:
                    traceback.print_exc()

    if rank == 0:
        overall_status = "pass"
        if any(report["status"] == "fail" for report in reports):
            overall_status = "fail"
        elif any(report["status"] == "warn" for report in reports):
            overall_status = "warn"
        payload = {
            "status": overall_status,
            "model_path": args.model_path,
            "prompts": list(prompts),
            "variants": reports,
            "config": {
                "tensor_parallel_size": tp_size,
                "page_size": args.page_size,
                "num_pages": args.num_pages,
                "use_pynccl": runtime_options["use_pynccl"],
                "allow_dsv4_cuda_graph": runtime_options["allow_dsv4_cuda_graph"],
                "cuda_graph_bs": runtime_options["cuda_graph_bs"],
                "cuda_graph_capture_greedy_sample": runtime_options[
                    "cuda_graph_capture_greedy_sample"
                ],
                "enable_dsv4_radix_prefix_cache": runtime_options[
                    "enable_dsv4_radix_prefix_cache"
                ],
                "enable_dsv4_swa_tail_retention_v1": args.enable_dsv4_swa_tail_retention_v1,
                "enable_dsv4_component_loc_ownership": runtime_options[
                    "enable_dsv4_component_loc_ownership"
                ],
                "enable_dsv4_swa_independent_lifecycle": (
                    runtime_options["enable_dsv4_swa_independent_lifecycle"]
                ),
                "prefix_cache_metrics": llm.cache_manager.prefix_metrics_snapshot(),
                "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}),
                "model_prepare_report_rank0": getattr(llm.engine, "model_prepare_report", {}),
                "kv_capacity_plan_report": getattr(llm.engine, "kv_capacity_plan_report", {}),
                "distributed_init_method": distributed_init_method,
                "memory_ratio": args.memory_ratio,
                "max_seq_len": args.max_seq_len,
                "max_extend_tokens": args.max_extend_tokens,
                "max_tokens": args.max_tokens,
            },
        }
        _write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        if overall_status == "fail" or (args.fail_on_warning and overall_status == "warn"):
            return 1
        return 0
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek V4 text correctness smoke for human-readable output checks."
    )
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--variants", nargs="*", choices=tuple(_variant_map()))
    parser.add_argument("--prompt", action="append", help="Prompt to run. May be repeated.")
    parser.add_argument("--output", default="/tmp/dsv4_text_smoke.json")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=BASELINE_TP_SIZE,
        help=(
            "Tensor parallel size. Defaults to 8 to match the official TARGET 06 "
            "baseline; pass 1 or another value only for explicit debug runs."
        ),
    )
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=64, help="Use 0 for automatic capacity planning.")
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-extend-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--use-pynccl",
        action="store_true",
        help="Use the PyNCCL communicator for tensor-parallel collectives.",
    )
    parser.add_argument(
        "--allow-dsv4-cuda-graph",
        action="store_true",
        help="Opt in to DeepSeek V4 decode CUDA graph capture. Defaults to sizes 1,2,4.",
    )
    parser.add_argument(
        "--disable-dsv4-cuda-graph",
        action="store_true",
        help="Diagnostic override: disable DSV4 decode CUDA graph even for graph preset variants.",
    )
    parser.add_argument(
        "--disable-cuda-graph-greedy-sample",
        action="store_true",
        help=(
            "Diagnostic override: capture decode CUDA graphs but keep greedy sampling "
            "outside the graph so logits remain observable."
        ),
    )
    parser.add_argument(
        "--enable-dsv4-radix-prefix-cache",
        action="store_true",
        help="Explicitly opt in to DeepSeek V4 radix prefix cache.",
    )
    parser.add_argument(
        "--enable-dsv4-swa-tail-retention-v1",
        action="store_true",
        help=(
            "Explicitly request TARGET 08.20 DSV4 SWA tail/component retention V1. "
            "The runtime currently fails closed; see the target DESIGN.md."
        ),
    )
    parser.add_argument(
        "--enable-dsv4-component-loc-ownership",
        action="store_true",
        help=(
            "Explicitly enable TARGET 08.21.2 DSV4 Route B component loc ownership. "
            "Requires --enable-dsv4-radix-prefix-cache; Route B decode metadata "
            "deforest/direct graph metadata buffers remain separate env opt-ins."
        ),
    )
    parser.add_argument(
        "--enable-dsv4-swa-independent-lifecycle",
        action="store_true",
        help=(
            "Explicitly enable TARGET 08.31 DSV4 independent SWA lifecycle. "
            "Requires --enable-dsv4-radix-prefix-cache and "
            "--enable-dsv4-component-loc-ownership."
        ),
    )
    parser.add_argument(
        "--cuda-graph-bs",
        nargs="*",
        type=int,
        default=None,
        help="Explicit CUDA graph decode batch sizes for opt-in graph runs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero when text sanity checks warn about possible garbled output.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Answer briefly and clearly.",
    )
    args = parser.parse_args(argv)
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.num_pages == 0:
        args.num_pages = None
    elif args.num_pages <= 1:
        parser.error("--num-pages must be greater than 1, or 0 for automatic planning")
    if args.max_seq_len <= 0 or args.max_extend_tokens <= 0 or args.max_tokens <= 0:
        parser.error("--max-seq-len, --max-extend-tokens and --max-tokens must be positive")
    if args.memory_ratio <= 0:
        parser.error("--memory-ratio must be positive")
    if args.cuda_graph_bs is not None:
        if any(value <= 0 for value in args.cuda_graph_bs):
            parser.error("--cuda-graph-bs values must be positive")
        args.cuda_graph_bs = sorted(set(args.cuda_graph_bs))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_text_smoke(parse_args(argv)))


if __name__ == "__main__":
    main()

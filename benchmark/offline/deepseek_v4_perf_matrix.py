from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")


DSV4_V0_BF16_TOGGLE = "MINISGL_DSV4_SM80_V0_BF16"
DSV4_V1_MOE_TOGGLE = "MINISGL_DSV4_SM80_V1_MOE"
DSV4_MOE_V2_TOGGLE = "MINISGL_DSV4_SM80_MOE_V2"
DSV4_MOE_VLLM_RUNNER_TOGGLE = "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER"
DSV4_MOE_REDUCE_BF16_TOGGLE = "MINISGL_DSV4_SM80_MOE_REDUCE_BF16"
DSV4_MOE_EXPERT_BACKEND_ENV = "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"
DSV4_MOE_EXPERT_BACKEND_MARLIN = "marlin_mxfp4_w4a16"
DSV4_MOE_EXPERT_BACKEND_VLLM_MARLIN_BRIDGE = "vllm_marlin_bridge"
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
DSV4_CASE_BOUNDARY_DEBUG_ENV = "MINISGL_DSV4_CASE_BOUNDARY_DEBUG"
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
DSV4_ROUTE_B_LIFETIME_SWA_DIRECT_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent_swadirect"
)
DSV4_ROUTE_B_LIFETIME_SWA_REPLAY_METADATA_FUSED_VARIANT = (
    "dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent_swadirect_replaymetafused"
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
DSV4_ROUTE_B_LIFETIME_SWA_DIRECT_ENV = {
    **DSV4_ROUTE_B_LIFETIME_SWA_INDEPENDENT_ENV,
    DSV4_SWA_METADATA_PAGE_TABLE_CACHE_ENV: "1",
    DSV4_SWA_DIRECT_TOKEN_METADATA_ENV: "1",
}
DSV4_ROUTE_B_LIFETIME_SWA_REPLAY_METADATA_FUSED_ENV = {
    **DSV4_ROUTE_B_LIFETIME_SWA_DIRECT_ENV,
    DSV4_DIRECT_GRAPH_METADATA_GROUPS_ENV: "swa,c4",
    DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED_ENV: "1",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    batch_size: int
    prompt_len: int
    decode_len: int
    description: str
    repeats: int = 3
    warmup_repeats: int = 1
    shared_prefix_len: int = 0
    suffix_len: int = 0
    total_requests: int = 0
    wave_size: int = 0
    prompt_len_cycle: tuple[int, ...] = ()
    decode_len_cycle: tuple[int, ...] = ()

    @property
    def max_input_len(self) -> int:
        if self.kind in {"shared_prefix", "shared_prefix_reuse"}:
            return self.shared_prefix_len + self.suffix_len
        if self.kind in {
            "prefix_partial_hit_reuse",
            "prefix_multi_sustained",
            "prefix_eviction_pressure",
        }:
            return max(self.prompt_len, self.shared_prefix_len + self.suffix_len)
        if self.prompt_len_cycle:
            return max(self.prompt_len_cycle)
        if self.kind == "mixed_prefill_decode":
            return self.prompt_len
        return self.prompt_len

    @property
    def max_seq_len(self) -> int:
        decode_len = max(self.decode_len_cycle) if self.decode_len_cycle else self.decode_len
        return self.max_input_len + decode_len


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


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="long_prefill_bs1",
        kind="random",
        batch_size=1,
        prompt_len=4096,
        decode_len=1,
        description="Single-request long prefill with one generated token.",
    ),
    Scenario(
        name="batch_prefill_bs8",
        kind="random",
        batch_size=8,
        prompt_len=1024,
        decode_len=1,
        description="Batch prefill with one generated token per request.",
    ),
    Scenario(
        name="decode_throughput_bs8",
        kind="random",
        batch_size=8,
        prompt_len=128,
        decode_len=64,
        description="Decode-heavy batch throughput workload.",
    ),
    Scenario(
        name="mixed_prefill_decode_bs4",
        kind="mixed_prefill_decode",
        batch_size=4,
        prompt_len=1024,
        decode_len=32,
        description=(
            "Varied prompt lengths and decode budgets. The current offline scheduler "
            "does not inject new arrivals while decode is already running."
        ),
    ),
    Scenario(
        name="shared_prompt_no_radix_bs8",
        kind="shared_prefix",
        batch_size=8,
        prompt_len=1088,
        decode_len=16,
        shared_prefix_len=1024,
        suffix_len=64,
        description="Repeated shared prompt with DeepSeek V4 radix prefix cache disabled.",
    ),
    Scenario(
        name="shared_prompt_reuse_bs8",
        kind="shared_prefix_reuse",
        batch_size=8,
        prompt_len=1088,
        decode_len=16,
        shared_prefix_len=1024,
        suffix_len=64,
        description=(
            "Sequential shared-prefix reuse: one warm request fills the prefix cache, "
            "then the remaining requests reuse the shared prefix."
        ),
    ),
)


TARGET08_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="historical_4096_1024_bs4",
        kind="random",
        batch_size=4,
        prompt_len=4096,
        decode_len=1024,
        repeats=1,
        warmup_repeats=0,
        description="TARGET07 historical fixed benchmark: prompt 4096, decode 1024, batch 4.",
    ),
    Scenario(
        name="historical_4096_128_bs4",
        kind="random",
        batch_size=4,
        prompt_len=4096,
        decode_len=128,
        repeats=1,
        warmup_repeats=0,
        description="TARGET07 historical fixed benchmark: prompt 4096, decode 128, batch 4.",
    ),
    Scenario(
        name="decode_ladder_bs16",
        kind="decode_ladder",
        batch_size=16,
        prompt_len=128,
        decode_len=64,
        repeats=1,
        warmup_repeats=0,
        decode_len_cycle=(16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24, 32, 32, 48, 64),
        description=(
            "Single-wave decode ladder. Sixteen requests start together and mixed "
            "output lengths naturally step active decode batch sizes through "
            "16, 8, 4, 2, and 1."
        ),
    ),
    Scenario(
        name="cuda_graph_padding_boundaries_257",
        kind="cuda_graph_padding_boundaries",
        batch_size=257,
        prompt_len=16,
        decode_len=10,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET 12.60 single-wave boundary probe. Grouped output budgets make "
            "active decode M step through 257, 129, 65, 33, and 17 so upward "
            "CUDA-graph padding and live-row isolation can be checked in one engine."
        ),
    ),
    Scenario(
        name="cuda_graph_padding_live_rows_64",
        kind="cuda_graph_padding_live_rows",
        batch_size=64,
        prompt_len=16,
        decode_len=8,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET 12.602 candidate-bucket probe. Grouped output budgets make active "
            "decode M step through 64, 57, 33, and 17 for 57->64, 33->40, and 17->24."
        ),
    ),
    Scenario(
        name="cuda_graph_recipe_balanced_wave256",
        kind="cuda_graph_recipe_balanced_wave",
        batch_size=256,
        prompt_len=512,
        decode_len=40,
        repeats=1,
        warmup_repeats=0,
        total_requests=512,
        wave_size=256,
        prompt_len_cycle=(64, 128, 256, 512),
        description=(
            "TARGET 12.605 balanced recipe workload: two real waves of 256 mixed "
            "short/medium prompts. Completion groups hold decode at active M "
            "256, 192, 128, 16, and 1 before each wave drains."
        ),
    ),
    Scenario(
        name="cuda_graph_recipe_high_concurrency_wave512",
        kind="cuda_graph_recipe_high_concurrency_wave",
        batch_size=512,
        prompt_len=256,
        decode_len=20,
        repeats=1,
        warmup_repeats=0,
        total_requests=512,
        wave_size=512,
        prompt_len_cycle=(32, 64, 128, 256),
        description=(
            "TARGET 12.605 high-concurrency recipe workload: one real wave of 512 "
            "mixed short/medium prompts. Completion groups hold decode at active M "
            "512, 384, 256, 128, and 1 before the wave drains."
        ),
    ),
    Scenario(
        name="cuda_graph_shape_census_17",
        kind="random",
        batch_size=17,
        prompt_len=16,
        decode_len=2,
        repeats=1,
        warmup_repeats=0,
        description="TARGET 12.6025 isolated exact-M versus padded-M census at M=17.",
    ),
    Scenario(
        name="cuda_graph_shape_census_33",
        kind="random",
        batch_size=33,
        prompt_len=16,
        decode_len=2,
        repeats=1,
        warmup_repeats=0,
        description="TARGET 12.6025 isolated exact-M versus padded-M census at M=33.",
    ),
    Scenario(
        name="cuda_graph_shape_census_57",
        kind="random",
        batch_size=57,
        prompt_len=16,
        decode_len=2,
        repeats=1,
        warmup_repeats=0,
        description="TARGET 12.6025 isolated exact-M versus padded-M census at M=57.",
    ),
    Scenario(
        name="serving_mixed_112req_wave16",
        kind="serving_mixed",
        batch_size=16,
        prompt_len=128,
        decode_len=64,
        repeats=1,
        warmup_repeats=0,
        total_requests=112,
        wave_size=16,
        prompt_len_cycle=(64, 96, 128, 160, 192, 224, 256, 128),
        decode_len_cycle=(16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 24, 24, 32, 32, 48, 64),
        description=(
            "Offline serving-style substitute: 112 total requests issued as seven "
            "same-process waves of 16, with mixed prompt and output lengths. "
            "The current offline scheduler does not model timed arrivals or RPS."
        ),
    ),
    Scenario(
        name="prefix_full_hit_257_bs4",
        kind="prefix_full_hit_reuse",
        batch_size=4,
        prompt_len=257,
        decode_len=4,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET08.10 boundary full-hit workload: one 257-token warm request, "
            "then three identical requests hit the retained 256-token page."
        ),
    ),
    Scenario(
        name="prefix_full_hit_512_bs4",
        kind="prefix_full_hit_reuse",
        batch_size=4,
        prompt_len=512,
        decode_len=4,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET08.22 exact page-multiple stress: one 512-token warm request, "
            "then three identical requests. Route B SWA-tail guard should shorten "
            "the otherwise page-aligned hit from 256 to 0."
        ),
    ),
    Scenario(
        name="prefix_full_hit_513_bs4",
        kind="prefix_full_hit_reuse",
        batch_size=4,
        prompt_len=513,
        decode_len=4,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET08.22 neighboring control for 512: one 513-token warm request, "
            "then three identical requests should hit the retained 512-token prefix."
        ),
    ),
    Scenario(
        name="prefix_full_hit_768_bs4",
        kind="prefix_full_hit_reuse",
        batch_size=4,
        prompt_len=768,
        decode_len=4,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET08.22 exact page-multiple stress: one 768-token warm request, "
            "then three identical requests. Route B SWA-tail guard should shorten "
            "the otherwise page-aligned hit from 512 to 0."
        ),
    ),
    Scenario(
        name="prefix_full_hit_769_bs4",
        kind="prefix_full_hit_reuse",
        batch_size=4,
        prompt_len=769,
        decode_len=4,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET08.22 neighboring control for 768: one 769-token warm request, "
            "then three identical requests should hit the retained 768-token prefix."
        ),
    ),
    Scenario(
        name="prefix_full_hit_513_longout_bs4",
        kind="prefix_full_hit_reuse",
        batch_size=4,
        prompt_len=513,
        decode_len=32,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET08.22 longer-output shared-prefix control: one 513-token warm "
            "request, then three identical requests with 32 generated tokens each."
        ),
    ),
    Scenario(
        name="prefix_partial_hit_769_bs8",
        kind="prefix_partial_hit_reuse",
        batch_size=8,
        prompt_len=769,
        decode_len=8,
        repeats=1,
        warmup_repeats=0,
        shared_prefix_len=257,
        suffix_len=512,
        description=(
            "TARGET08.10 partial-hit workload: one 257-token warm request retains "
            "one page, then seven 769-token requests reuse that page and prefill "
            "the remaining suffix."
        ),
    ),
    Scenario(
        name="prefix_mixed_hit_miss_bs16",
        kind="prefix_mixed_hit_miss",
        batch_size=16,
        prompt_len=769,
        decode_len=8,
        repeats=1,
        warmup_repeats=0,
        description=(
            "TARGET08.10 mixed workload: one warm request followed by a batch with "
            "full hits and unrelated misses."
        ),
    ),
    Scenario(
        name="prefix_multi_112req_wave16",
        kind="prefix_multi_sustained",
        batch_size=16,
        prompt_len=576,
        decode_len=8,
        repeats=1,
        warmup_repeats=0,
        shared_prefix_len=512,
        suffix_len=64,
        total_requests=112,
        wave_size=16,
        description=(
            "TARGET08.10 sustained multi-prefix workload: 112 requests in seven "
            "waves of 16 cycling across eight 512-token shared prefixes."
        ),
    ),
    Scenario(
        name="prefix_eviction_pressure_96req_wave16",
        kind="prefix_eviction_pressure",
        batch_size=16,
        prompt_len=513,
        decode_len=2,
        repeats=1,
        warmup_repeats=0,
        shared_prefix_len=512,
        suffix_len=1,
        total_requests=96,
        wave_size=16,
        description=(
            "TARGET08.10 eviction-pressure workload: 96 distinct two-page prefixes "
            "under --num-pages 128, forcing safe radix eviction."
        ),
    ),
)


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="fallback",
        env={DSV4_DISABLE_RELEASE_DEFAULTS_ENV: "1"},
        description="All MINISGL_DSV4_SM80_* toggles cleared; release defaults disabled.",
    ),
    Variant(
        name="v0_bf16",
        env={DSV4_V0_BF16_TOGGLE: "1"},
        description="TARGET 05.7 v0 BF16 whitelist bundle.",
    ),
    Variant(
        name="v1_moe",
        env={DSV4_V1_MOE_TOGGLE: "1"},
        description="V1 exact grouped MoE bundle: v0 BF16 whitelist plus grouped MoE route.",
    ),
)


RUNTIME_VARIANTS: tuple[Variant, ...] = (
    Variant(
        name=DSV4_RELEASE_DEFAULT_VARIANT,
        env={},
        description=(
            "TARGET 12.52 release-default benchmark: leave DSV4 env empty so "
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
    Variant(
        name="v1_moe_v2",
        env={DSV4_V1_MOE_TOGGLE: "1", DSV4_MOE_V2_TOGGLE: "1"},
        description=(
            "V2 exact MoE boundary: V1 exact grouped MoE plus explicit route "
            "execution plan and per-layer grouped-MoE workspace."
        ),
    ),
    Variant(
        name="v1_moe_pynccl",
        env={DSV4_V1_MOE_TOGGLE: "1"},
        description="V1 exact grouped MoE with PyNCCL tensor-parallel collectives.",
        use_pynccl=True,
    ),
    Variant(
        name="v1_moe_graph",
        env={DSV4_V1_MOE_TOGGLE: "1"},
        description="V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture.",
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc",
        env={DSV4_V1_MOE_TOGGLE: "1", DSV4_HC_TOGGLE: "1"},
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture "
            "and experimental sm80 HC split/post helpers."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC helpers, and experimental sm80 RMSNorm helper."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_fp8gemm",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_FP8_GEMM_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, and experimental sm80 FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_wqb_fp8gemm",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, and selective attention wq_b FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_wqb_woa",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_A_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b FP8 GEMM, "
            "and selective attention wo_a projection."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_wqb_wob_fp8gemm",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b FP8 GEMM, "
            "and selective attention wo_b FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_fp8gemm",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, and selective indexer wq_b FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_shared_fp8gemm",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_SHARED_FP8_GEMM_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, and selective shared-expert "
            "FP8 GEMM."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, and exact gate fp32 "
            "weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_wqb_wob_idxwqb_gatecache_idxstorecache",
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_HC_TOGGLE: "1",
            DSV4_RMSNORM_TOGGLE: "1",
            DSV4_Q_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_WO_B_FP8_GEMM_TOGGLE: "1",
            DSV4_INDEXER_WQB_FP8_GEMM_TOGGLE: "1",
            DSV4_GATE_FP32_WEIGHT_CACHE_TOGGLE: "1",
            DSV4_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE_TOGGLE: "1",
        },
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, exact gate fp32 weight "
            "caching, and exact indexer-store norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_fwqakv_wqb_wob_idxwqb_gatecache_idxstorecache",
        env={
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
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, vLLM-aligned shared-activation "
            "attention wq_a/wkv FP8 projection, selective attention wq_b/wo_b "
            "FP8 GEMM, selective indexer wq_b FP8 GEMM, exact gate fp32 weight "
            "caching, and exact indexer-store norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name=("v1_moe_graph_hc_rmsnorm_fwqakv_qkvrope_wqb_wob_idxwqb_" "gatecache_idxstorecache"),
        env={
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
        description=(
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
        name=(
            "v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_wqb_wob_idxwqb_" "gatecache_idxstorecache"
        ),
        env={
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
        description=(
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
        name=(
            "v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_"
            "gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "the current vLLM-aligned cached fused wq_a/wkv graph path, and "
            "greedy sampler captured in the graph."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_v2_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_"
            "gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "Current best exact macro variant with the V2 MoE execution plan and "
            "per-layer grouped-MoE workspace enabled."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "Mini-owned vLLM-shaped exact MoE runner wrapping the current grouped "
            "FP4 W13/SwiGLU/W2 backend, with the current exact graph macro bundle."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_mxfp4_w4a16_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_MARLIN,
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
        description=(
            "TARGET 07.38 Marlin MXFP4 W4A16 exact expert-backend candidate. "
            "This variant is expected to fail explicitly until mini owns an "
            "equivalent Marlin WNA16 custom-op surface."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_vllm_marlin_bridge_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
            DSV4_V1_MOE_TOGGLE: "1",
            DSV4_MOE_V2_TOGGLE: "1",
            DSV4_MOE_VLLM_RUNNER_TOGGLE: "1",
            DSV4_MOE_EXPERT_BACKEND_ENV: DSV4_MOE_EXPERT_BACKEND_VLLM_MARLIN_BRIDGE,
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
        description=(
            "TARGET 07.39 external vLLM Marlin bridge marker. The actual bridge "
            "is probe-only and this mini runtime variant fails explicitly until "
            "a mini-owned narrow csrc port exists."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.391 mini-owned Marlin WNA16 backend. This variant is "
            "explicit opt-in and may JIT-build the vendored Marlin extension "
            "before using cached MXFP4-to-Marlin expert weights."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.394 exact bf16 Marlin WNA16 path with opt-in global "
            "topk/lens consolidation. Cache and activation precision remain bf16."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.395 exact bf16 Marlin WNA16 path with global topk/lens "
            "and opt-in bf16 two-scope gather/mask plus split-K sparse decode."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_graph_"
            "hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_"
            "idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.41 exact bf16 path with the 07.395 split-K stack plus "
            "opt-in fused decode replay metadata staging."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.50 opt-in vLLM-aligned FP8 indexer cache/logits lane "
            "on top of the 07.41 split-K/metacopy exact stack. MLA/SWA cache "
            "precision remains bf16."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_"
            "wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.54 graph-layout PoC: vLLM-aligned FP8 indexer plus "
            "Triton fused FP8 activation fake-quant staging."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_qwqbbf16cache_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.58 q_wqb-only cached BF16 dequantized weight projection "
            "path on top of the promoted 07.54 graph-layout stack."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.59 q_wqb plus row-parallel wo_b cached BF16 "
            "dequantized weight projection path."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_qwqbbf16cache_wobbf16cache_idxwqbbf16cache_"
            "graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_"
            "gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.60 q_wqb, row-parallel wo_b, and indexer.wq_b cached "
            "BF16 dequantized weight projection path."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory",
        env={DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1"},
        description=(
            "Milestone bundle for the TARGET 07.66 A100/sm80 victory stack: "
            "Marlin WNA16 MoE, graph replay, FP8 indexer cache, split-K sparse "
            "decode, four attention/indexer BF16 projection caches, and shared "
            "expert BF16 projection caches."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=DSV4_A100_MARLIN_PREBUILD_VARIANT,
        env=dict(DSV4_A100_MARLIN_PREBUILD_ENV),
        description=(
            "TARGET 08.35 diagnostic: dsv4_sm80_a100_victory with the MoE "
            "expert backend fixed to marlin_wna16 and all routed expert "
            "Marlin WNA16 caches prebuilt before KV capacity planning, while "
            "retaining original routed FP4 expert tensors."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=DSV4_A100_MARLIN_RELEASE_VARIANT,
        env=dict(DSV4_A100_MARLIN_RELEASE_ENV),
        description=(
            "TARGET 08.35 high-memory-efficiency preset: dsv4_sm80_a100_victory "
            "with backend fixed to marlin_wna16, routed expert caches prebuilt "
            "before KV capacity planning, and original routed FP4 expert "
            "weights/scales released after successful full-model prebuild."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=DSV4_A100_MARLIN_RELEASE_SAFE_ARENA_VARIANT,
        env=dict(DSV4_A100_MARLIN_RELEASE_SAFE_ARENA_ENV),
        description=(
            "TARGET 08.38 candidate: Marlin WNA16 prebuild plus before-KV "
            "release capacity credit, with a 3.1875 GiB/rank sentinel guard "
            "arena held out of the released ranges."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_fp8marlinproj",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_VLLM_FP8_MARLIN_PROJECTION_TOGGLE: "1",
        },
        description=(
            "Legacy TARGET 07.74 alias: dsv4_sm80_a100_victory with the "
            "mini-owned dense FP8 Marlin W8A16 block projection runtime for "
            "attention q_wqb, attention wo_b local projection, and shared "
            "experts down."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_densefp8marlinproj",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_DENSE_FP8_MARLIN_PROJECTION_TOGGLE: "1",
        },
        description=(
            "TARGET 07.76 opt-in: dsv4_sm80_a100_victory with mini-owned "
            "dense FP8 Marlin W8A16 block linear for attention q_wqb, "
            "attention wo_b local projection, and shared experts down."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_hccleanup",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_HC_GRAPH_CLEANUP_TOGGLE: "1",
        },
        description=(
            "TARGET 07.68 opt-in: dsv4_sm80_a100_victory plus fused HC "
            "prenorm/split boundary cleanup."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_bf16smallgemm",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE: "1",
        },
        description=(
            "TARGET 07.70 opt-in: dsv4_sm80_a100_victory plus pretransposed "
            "cached BF16 weights for small-M projection GEMMs."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_sharedbf16",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE: "1",
        },
        description=(
            "TARGET 07.66 audit variant: dsv4_sm80_a100_victory with explicit "
            "shared expert gate/up and down cached BF16 dequantized weight "
            "projection env."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_metadatadeforest",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_DECODE_METADATA_DEFOREST_TOGGLE: "1",
        },
        description=(
            "TARGET 07.64 opt-in: dsv4_sm80_a100_victory plus fused decode "
            "metadata indices/lens assembly. Under Route B component loc "
            "ownership this uses component-owned page tables."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_directgraphmetadata",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE: "1",
        },
        description=(
            "TARGET 08.25 opt-in: Route B graph replay writes C4 sparse "
            "metadata directly into captured graph buffers instead of "
            "materializing eager C4 source tensors and staging copies."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_directgraphmetadata_c4",
        env={
            DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1",
            DSV4_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE: "1",
            DSV4_DIRECT_GRAPH_METADATA_GROUPS_ENV: "c4",
        },
        description=(
            "TARGET 08.26 diagnostic: Route B graph replay writes only C4 sparse "
            "metadata directly into captured graph buffers; SWA and C128 keep "
            "the 08.22/08.25 eager-source staging path."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=DSV4_PROMOTED_ROUTE_B_LIFETIME_VARIANT,
        env=dict(DSV4_ROUTE_B_LIFETIME_ENV),
        description=(
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
        name=DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_VARIANT,
        env=dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_ENV),
        description=(
            "TARGET 08.35 prefix high-memory-efficiency preset: promoted Route B "
            "lifetime prefix preset plus backend fixed to marlin_wna16, MoE "
            "Marlin WNA16 prebuild before KV capacity planning, and original "
            "routed FP4 expert weights/scales release after successful prebuild."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        name=DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_INDEPENDENT_VARIANT,
        env=dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_INDEPENDENT_ENV),
        description=(
            "TARGET 08.31/08.40 compatibility preset: Route B lifetime plus "
            "Marlin WNA16 before-KV release capacity credit, component-slot "
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
        name=DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_DIRECT_VARIANT,
        env=dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_DIRECT_ENV),
        description=(
            "TARGET 08.50/12.50 opt-in: Route B lifetime plus Marlin WNA16 "
            "release, independent SWA lifecycle, SWA page-table cache, and "
            "direct token-level SWA metadata."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        name=DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_REPLAY_METADATA_FUSED_VARIANT,
        env=dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SWA_REPLAY_METADATA_FUSED_ENV),
        description=(
            "TARGET 08.54/12.50 opt-in: Marlin release plus independent SWA "
            "direct token metadata and fused replay metadata for direct SWA "
            "graph index buffers."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        name=DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SAFE_ARENA_VARIANT,
        env=dict(DSV4_PREFIX_ROUTE_B_LIFETIME_MARLIN_RELEASE_SAFE_ARENA_ENV),
        description=(
            "TARGET 08.38 prefix candidate: Route B lifetime prefix preset plus "
            "Marlin WNA16 before-KV release capacity credit and a 3.1875 "
            "GiB/rank sentinel guard arena."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        name="dsv4_sm80_a100_victory_prefix_routeb_lifetime_groupedfp4",
        env={
            **DSV4_ROUTE_B_LIFETIME_ENV,
            DSV4_MOE_EXPERT_BACKEND_ENV: "grouped_fp4",
        },
        description=(
            "TARGET 08.34 diagnostic A/B preset: promoted Route B lifetime preset "
            "with the routed expert backend forced to grouped_fp4 instead of "
            "the A100 victory default marlin_wna16."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        name=DSV4_ROUTE_B_LIFETIME_MOE_REDUCE_BF16_VARIANT,
        env={**DSV4_ROUTE_B_LIFETIME_ENV, DSV4_MOE_REDUCE_BF16_TOGGLE: "1"},
        description=(
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
        name=DSV4_ROUTE_B_LIFETIME_MOE_REDUCE_BF16_INGRAPH_METADATA_VARIANT,
        env={
            **DSV4_ROUTE_B_LIFETIME_ENV,
            DSV4_MOE_REDUCE_BF16_TOGGLE: "1",
            DSV4_PREP_METADATA_IN_GRAPH_TOGGLE: "1",
        },
        description=(
            "Legacy TARGET 12.4 alias: TARGET10.27 Route B lifetime BF16 "
            "MoE-reduce preset with SGLang-style raw decode metadata. The "
            "in-graph metadata path is now part of the Route B lifetime env."
        ),
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
    ),
    Variant(
        name=DSV4_ROUTE_B_LIFETIME_SWA_INDEPENDENT_VARIANT,
        env=dict(DSV4_ROUTE_B_LIFETIME_SWA_INDEPENDENT_ENV),
        description=(
            "TARGET 08.31 opt-in: promoted Route B lifetime prefix preset plus "
            "independent SWA lifecycle. Pair with --enable-dsv4-radix-prefix-cache, "
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
        name=DSV4_ROUTE_B_LIFETIME_SWA_DIRECT_VARIANT,
        env=dict(DSV4_ROUTE_B_LIFETIME_SWA_DIRECT_ENV),
        description=(
            "TARGET 08.50 opt-in: promoted Route B lifetime prefix preset plus "
            "independent SWA lifecycle, SWA page-table cache, and direct "
            "token-level SWA metadata."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        name=DSV4_ROUTE_B_LIFETIME_SWA_REPLAY_METADATA_FUSED_VARIANT,
        env=dict(DSV4_ROUTE_B_LIFETIME_SWA_REPLAY_METADATA_FUSED_ENV),
        description=(
            "TARGET 08.54 opt-in: SWA direct plus fused replay metadata for "
            "independent SWA write locs and direct SWA graph index buffers."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
        enable_dsv4_radix_prefix_cache=True,
        enable_dsv4_component_loc_ownership=True,
        enable_dsv4_swa_independent_lifecycle=True,
    ),
    Variant(
        name=DSV4_ROUTE_B_LIFETIME_LEGACY_VARIANT,
        env=dict(DSV4_ROUTE_B_LIFETIME_ENV),
        description=(
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
        name="target0762_woabf16bmmcache",
        env={DSV4_A100_VICTORY_BUNDLE_TOGGLE: "1"},
        description=(
            "Legacy alias for dsv4_sm80_a100_victory kept for TARGET 07.62 "
            "artifacts and scripts."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name=(
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_actqtriton_scalecache_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ),
        env={
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
        description=(
            "TARGET 07.56 low-cost preflight: 07.54 graph-layout stack plus "
            "opt-in static FP32 projection scale cache."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="v1_moe_graph_hc_rmsnorm_qwqa_wqb_wob_idxwqb_gatecache_idxstorecache",
        env={
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
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture, "
            "experimental sm80 HC/RMSNorm helpers, selective attention wq_a/wq_b/"
            "wo_b FP8 GEMM, selective indexer wq_b FP8 GEMM, exact gate fp32 "
            "weight caching, and exact indexer-store norm fp32 weight caching."
        ),
        allow_dsv4_cuda_graph=True,
    ),
    Variant(
        name="v1_moe_graph_sample",
        env={DSV4_V1_MOE_TOGGLE: "1"},
        description=(
            "V1 exact grouped MoE with opt-in DSV4 decode CUDA graph capture "
            "and greedy sampler captured in the graph."
        ),
        allow_dsv4_cuda_graph=True,
        cuda_graph_capture_greedy_sample=True,
    ),
    Variant(
        name="v1_moe_graph_pynccl",
        env={DSV4_V1_MOE_TOGGLE: "1"},
        description="V1 exact grouped MoE with PyNCCL and opt-in DSV4 decode CUDA graph capture.",
        use_pynccl=True,
        allow_dsv4_cuda_graph=True,
    ),
)


ALL_SCENARIOS: tuple[Scenario, ...] = (*DEFAULT_SCENARIOS, *TARGET08_SCENARIOS)


ALL_VARIANTS: tuple[Variant, ...] = (*DEFAULT_VARIANTS, *RUNTIME_VARIANTS)


FALLBACK_COUNTER_NAMES = {
    "apply_rotary_tail",
    "compress_forward_fallback",
    "compress_norm_rope_store_fallback",
    "compressor_plan_fallback",
    "dequant_fp4_weight",
    "dequant_fp8_weight",
    "dsv4_sparse_attention_two_source_bf16",
    "get_paged_mqa_logits_metadata_fallback",
    "hash_topk_fallback",
    "hc_head_fallback",
    "hc_post_fallback",
    "hc_pre_fallback",
    "indexer_bf16_logits_fallback",
    "indexer_kv_hadamard_fallback",
    "indexer_q_rope_hadamard_bf16_fallback",
    "indexer_select_bf16_fallback",
    "k_norm_rope_cache_fallback",
    "linear_bf16_fp32_fallback",
    "mega_moe_pre_dispatch_fallback",
    "moe_gate_fallback",
    "moe_route_dispatch_bf16_grouped",
    "norm_rope_inplace_fallback",
    "paged_mqa_attention_fallback",
    "plan_topk_v2_fallback",
    "q_norm_rope_fallback",
    "quantized_linear_ref",
    "sequence_mqa_attention_fallback",
    "silu_and_mul_clamp_fallback",
    "store_compressed_fallback",
    "store_indexer_fallback",
    "store_swa_fallback",
    "topk_transform_512_fallback",
    "topk_transform_512_full_fallback",
    "topk_transform_512_v2_fallback",
    "wo_a_grouped_projection_fallback",
}


OPTIONAL_NONE_MEANS_SKIP = {
    "dsv4_sparse_attention_two_source_bf16",
    "moe_route_dispatch_bf16_grouped",
}


BOTTLENECK_COUNTER_GROUPS: dict[str, tuple[str, ...]] = {
    "attention": (
        "apply_rotary_tail",
        "dsv4_sparse_attention_two_source_bf16",
        "indexer_bf16_logits_fallback",
        "indexer_select_bf16_fallback",
        "paged_mqa_attention_fallback",
        "q_norm_rope_fallback",
        "sequence_mqa_attention_fallback",
        "topk_transform_512_full_fallback",
    ),
    "MoE / expert GEMM": (
        "mega_moe_pre_dispatch_fallback",
        "moe_gate_fallback",
        "moe_route_dispatch_bf16_grouped",
        "quantized_linear_ref",
        "silu_and_mul_clamp_fallback",
    ),
    "fp4 expert handling": (
        "dequant_fp4_weight",
        "moe_route_dispatch_bf16_grouped",
        "quantized_linear_ref",
    ),
    "KV cache writes": (
        "compress_norm_rope_store_fallback",
        "k_norm_rope_cache_fallback",
        "store_compressed_fallback",
        "store_indexer_fallback",
        "store_swa_fallback",
    ),
    "metadata construction": (
        "compressor_plan_fallback",
        "get_paged_mqa_logits_metadata_fallback",
        "plan_topk_v2_fallback",
        "topk_transform_512_fallback",
        "topk_transform_512_full_fallback",
    ),
}


def _scenario_map() -> dict[str, Scenario]:
    return {scenario.name: scenario for scenario in ALL_SCENARIOS}


def _variant_map() -> dict[str, Variant]:
    return {variant.name: variant for variant in ALL_VARIANTS}


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return _dist_version(name)
    return getattr(module, "__version__", None) or _dist_version(name)


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def git_info() -> dict[str, Any]:
    status = _git_output(["status", "--short"]) or ""
    return {
        "branch": _git_output(["branch", "--show-current"]),
        "commit": _git_output(["rev-parse", "HEAD"]),
        "short_commit": _git_output(["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


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
            "DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV",
            "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY_TOGGLE",
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY",
        ),
        DSV4_PREP_METADATA_IN_GRAPH_ORACLE_ENV,
        getattr(
            dsv4_kernel,
            "DSV4_MARLIN_WNA16_CACHE_DEBUG_ENV",
            "MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG",
        ),
        getattr(
            dsv4_kernel,
            "DSV4_WARMUP_FORWARD_MEMORY_DEBUG_ENV",
            "MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG",
        ),
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


def runtime_variant_env(dsv4_kernel, variant: Variant) -> dict[str, Any]:
    if variant.name == DSV4_RELEASE_DEFAULT_VARIANT and not variant.env:
        return {
            "cleared_dsv4_sm80_env": [],
            "preserved_dsv4_sm80_env": {},
            "active_dsv4_toggles": active_dsv4_toggles(dsv4_kernel),
            "raw_dsv4_sm80_env": raw_dsv4_env(),
        }
    return configure_variant(dsv4_kernel, variant)


def active_dsv4_toggles(dsv4_kernel) -> list[str]:
    del dsv4_kernel
    return []


def raw_dsv4_env() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in sorted(os.environ)
        if name.startswith("MINISGL_DSV4_")
    }


def run_classification(*, tp_size: int, page_size: int, smoke: bool) -> str:
    if tp_size == 8 and page_size == 256 and not smoke:
        return "baseline"
    return "smoke_debug"


def _jsonify_dataclass(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def collect_runtime_environment(torch, dsv4_kernel, *, rank: int) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        device = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(device)
        cuda.update(
            {
                "rank": rank,
                "current_device": device,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(device),
                "capability": [int(cap[0]), int(cap[1])],
                "capability_name": f"sm{cap[0]}{cap[1]}",
                "runtime": torch.version.cuda,
            }
        )
        try:
            nccl_version = torch.cuda.nccl.version()
            if isinstance(nccl_version, tuple):
                nccl_version = ".".join(str(part) for part in nccl_version)
        except Exception:
            nccl_version = None
    else:
        nccl_version = None

    capabilities = dsv4_kernel.detect_dsv4_kernel_capabilities()
    return {
        "python": sys.version.split()[0],
        "packages": {
            "torch": torch.__version__,
            "triton": _module_version("triton"),
            "sgl_kernel": _module_version("sgl_kernel"),
            "flashinfer": _module_version("flashinfer"),
            "deep_gemm": _module_version("deep_gemm"),
            "tilelang": _module_version("tilelang"),
            "tvm_ffi": _module_version("tvm_ffi"),
        },
        "cuda": cuda,
        "nccl": {"version": nccl_version},
        "dsv4_kernel_capabilities": _jsonify_dataclass(capabilities),
    }


def _random_tokens(
    rng: random.Random,
    length: int,
    vocab_size: int,
    *,
    token_id_range: int = 1024,
) -> list[int]:
    low = 10 if vocab_size > 64 else 1
    high = min(max(low, int(token_id_range)), max(vocab_size - 1, low))
    usable = max(high - low + 1, 1)
    return [low + rng.randrange(usable) for _ in range(length)]


def _random_token_bank(
    rng: random.Random,
    count: int,
    length: int,
    vocab_size: int,
    *,
    token_id_range: int,
) -> list[list[int]]:
    return [
        _random_tokens(rng, length, vocab_size, token_id_range=token_id_range) for _ in range(count)
    ]


def build_workload(
    scenario: Scenario,
    *,
    vocab_size: int,
    seed: int,
    token_id_range: int = 1024,
) -> tuple[list[list[int]], list[Any]]:
    from minisgl.core import SamplingParams

    rng = random.Random(seed)
    prompts: list[list[int]] = []
    output_lens: list[int] = []

    if scenario.kind in {"shared_prefix", "shared_prefix_reuse"}:
        prefix = _random_tokens(
            rng,
            scenario.shared_prefix_len,
            vocab_size,
            token_id_range=token_id_range,
        )
        for _ in range(scenario.batch_size):
            suffix = _random_tokens(
                rng,
                scenario.suffix_len,
                vocab_size,
                token_id_range=token_id_range,
            )
            prompts.append(prefix + suffix)
            output_lens.append(scenario.decode_len)
    elif scenario.kind == "prefix_full_hit_reuse":
        prompt = _random_tokens(
            rng,
            scenario.prompt_len,
            vocab_size,
            token_id_range=token_id_range,
        )
        for _ in range(scenario.batch_size):
            prompts.append(list(prompt))
            output_lens.append(scenario.decode_len)
    elif scenario.kind == "prefix_partial_hit_reuse":
        warm_len = scenario.shared_prefix_len or max(1, scenario.prompt_len // 3)
        if warm_len >= scenario.prompt_len:
            warm_len = max(1, scenario.prompt_len - 1)
        warm_prefix = _random_tokens(
            rng,
            warm_len,
            vocab_size,
            token_id_range=token_id_range,
        )
        prompts.append(list(warm_prefix))
        output_lens.append(scenario.decode_len)
        for _ in range(max(0, scenario.batch_size - 1)):
            suffix_len = scenario.prompt_len - warm_len
            suffix = _random_tokens(
                rng,
                suffix_len,
                vocab_size,
                token_id_range=token_id_range,
            )
            prompts.append(warm_prefix + suffix)
            output_lens.append(scenario.decode_len)
    elif scenario.kind == "prefix_mixed_hit_miss":
        warm_prompt = _random_tokens(
            rng,
            scenario.prompt_len,
            vocab_size,
            token_id_range=token_id_range,
        )
        prompts.append(list(warm_prompt))
        output_lens.append(scenario.decode_len)
        remaining = max(0, scenario.batch_size - 1)
        hit_count = (remaining + 1) // 2
        miss_count = remaining - hit_count
        mixed_prompts: list[list[int]] = [list(warm_prompt) for _ in range(hit_count)]
        mixed_prompts.extend(
            _random_token_bank(
                rng,
                miss_count,
                scenario.prompt_len,
                vocab_size,
                token_id_range=token_id_range,
            )
        )
        for idx, prompt in enumerate(mixed_prompts):
            # Interleave hits and misses instead of grouping them in the batch.
            target_idx = idx // 2 if idx % 2 == 0 else hit_count + idx // 2
            if 0 <= target_idx < len(mixed_prompts):
                prompts.append(mixed_prompts[target_idx])
            else:
                prompts.append(prompt)
            output_lens.append(scenario.decode_len)
    elif scenario.kind == "prefix_multi_sustained":
        request_count = scenario.total_requests or scenario.batch_size
        prefix_count = 8
        prefix_len = scenario.shared_prefix_len or max(1, scenario.prompt_len - scenario.suffix_len)
        suffix_len = scenario.suffix_len or max(1, scenario.prompt_len - prefix_len)
        prefixes = _random_token_bank(
            rng,
            prefix_count,
            prefix_len,
            vocab_size,
            token_id_range=token_id_range,
        )
        for idx in range(request_count):
            suffix = _random_tokens(
                rng,
                suffix_len,
                vocab_size,
                token_id_range=token_id_range,
            )
            prompts.append(prefixes[idx % prefix_count] + suffix)
            output_lens.append(scenario.decode_len)
    elif scenario.kind == "prefix_eviction_pressure":
        request_count = scenario.total_requests or scenario.batch_size
        prefix_len = scenario.shared_prefix_len or max(1, scenario.prompt_len - scenario.suffix_len)
        suffix_len = scenario.suffix_len or max(1, scenario.prompt_len - prefix_len)
        prefixes = _random_token_bank(
            rng,
            request_count,
            prefix_len,
            vocab_size,
            token_id_range=token_id_range,
        )
        for idx, prefix in enumerate(prefixes):
            suffix = _random_tokens(
                rng,
                suffix_len,
                vocab_size,
                token_id_range=token_id_range,
            )
            # Make distinct prefixes very unlikely to share the first page.
            if prefix:
                prefix[0] = 10 + (idx % max(1, min(token_id_range, vocab_size - 10)))
            prompts.append(prefix + suffix)
            output_lens.append(scenario.decode_len)
    elif scenario.kind == "mixed_prefill_decode":
        min_prompt_len = max(1, scenario.prompt_len // 4)
        min_decode_len = max(1, scenario.decode_len // 4)
        for idx in range(scenario.batch_size):
            frac = idx / max(scenario.batch_size - 1, 1)
            prompt_len = int(round(min_prompt_len + frac * (scenario.prompt_len - min_prompt_len)))
            decode_len = int(round(min_decode_len + frac * (scenario.decode_len - min_decode_len)))
            prompts.append(
                _random_tokens(
                    rng,
                    prompt_len,
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(max(1, decode_len))
    elif scenario.kind == "cuda_graph_padding_boundaries":
        if scenario.batch_size != 257:
            raise ValueError(
                "cuda_graph_padding_boundaries currently requires batch_size=257"
            )
        # The prefill forward produces token one. These budgets then leave
        # 257/129/65/33/17 live requests on successive decode plateaus.
        boundary_output_lens = [2] * 128 + [4] * 64 + [6] * 32 + [8] * 16 + [10] * 17
        for output_len in boundary_output_lens:
            prompts.append(
                _random_tokens(
                    rng,
                    scenario.prompt_len,
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(output_len)
    elif scenario.kind == "cuda_graph_padding_live_rows":
        if scenario.batch_size != 64:
            raise ValueError("cuda_graph_padding_live_rows currently requires batch_size=64")
        # Prefill produces token one. Successive decode plateaus have exactly
        # 64, 57, 33, and 17 live rows.
        boundary_output_lens = [2] * 7 + [4] * 24 + [6] * 16 + [8] * 17
        for output_len in boundary_output_lens:
            prompts.append(
                _random_tokens(
                    rng,
                    scenario.prompt_len,
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(output_len)
    elif scenario.kind == "cuda_graph_recipe_balanced_wave":
        if scenario.batch_size != 256 or (scenario.total_requests or 0) % 256:
            raise ValueError(
                "cuda_graph_recipe_balanced_wave requires 256-request whole waves"
            )
        # Prefill emits token one. Each wave then spends eight decode steps at
        # M=256/192/128/16/1 before draining.
        output_cycle = [8] * 64 + [16] * 64 + [24] * 112 + [32] * 15 + [40]
        prompt_cycle = scenario.prompt_len_cycle or (scenario.prompt_len,)
        for idx in range(scenario.total_requests or scenario.batch_size):
            prompts.append(
                _random_tokens(
                    rng,
                    int(prompt_cycle[idx % len(prompt_cycle)]),
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(output_cycle[idx % 256])
    elif scenario.kind == "cuda_graph_recipe_high_concurrency_wave":
        if scenario.batch_size != 512 or (scenario.total_requests or 0) % 512:
            raise ValueError(
                "cuda_graph_recipe_high_concurrency_wave requires 512-request whole waves"
            )
        # Prefill emits token one. The wave spends four decode steps at each
        # M=512/384/256/128/1 before draining.
        output_cycle = [4] * 128 + [8] * 128 + [12] * 128 + [16] * 127 + [20]
        prompt_cycle = scenario.prompt_len_cycle or (scenario.prompt_len,)
        for idx in range(scenario.total_requests or scenario.batch_size):
            prompts.append(
                _random_tokens(
                    rng,
                    int(prompt_cycle[idx % len(prompt_cycle)]),
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(output_cycle[idx % 512])
    elif scenario.kind in {"decode_ladder", "serving_mixed"}:
        request_count = scenario.total_requests or scenario.batch_size
        prompt_cycle = scenario.prompt_len_cycle or (scenario.prompt_len,)
        decode_cycle = scenario.decode_len_cycle or (scenario.decode_len,)
        for idx in range(request_count):
            prompt_len = int(prompt_cycle[idx % len(prompt_cycle)])
            output_len = int(decode_cycle[idx % len(decode_cycle)])
            prompts.append(
                _random_tokens(
                    rng,
                    prompt_len,
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(max(1, output_len))
    else:
        for _ in range(scenario.batch_size):
            prompts.append(
                _random_tokens(
                    rng,
                    scenario.prompt_len,
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(scenario.decode_len)

    sampling_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=output_len)
        for output_len in output_lens
    ]
    return prompts, sampling_params


@dataclass
class BenchRequestStatus:
    uid: int
    input_ids: list[int]
    output_ids: list[int]
    requested_output_len: int
    admitted_output_len: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    started_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    token_times: list[float] | None = None

    def record_token(self, timestamp: float) -> None:
        if self.token_times is None:
            self.token_times = []
        if self.first_token_at is None:
            self.first_token_at = timestamp
        self.token_times.append(timestamp)

    def mark_finished(self, timestamp: float) -> None:
        self.finished_at = timestamp


def make_benchmark_llm_class():
    import torch
    from minisgl.llm.llm import LLM, RequestAllFinished
    from minisgl.message import BaseBackendMsg, DetokenizeMsg, UserMsg

    class BenchmarkLLM(LLM):
        def __init__(self, *args, **kwargs):
            self.bench_batch_trace: list[dict[str, Any]] = []
            self._bench_prepare_s: dict[int, float] = {}
            self._bench_batch_info: dict[int, dict[str, Any]] = {}
            self._active_generation_started_at: float | None = None
            self._active_generation_finished_at: float | None = None
            self.bench_admit_all_at_once = False
            super().__init__(*args, **kwargs)

        def offline_receive_msg(self, blocking: bool = False) -> list[BaseBackendMsg]:
            if blocking and len(self.pending_requests) == 0:
                raise RequestAllFinished()
            results: list[BaseBackendMsg] = []
            added, sum_input_len = 0, 0
            for tokens_or_prompt, sampling_params in self.pending_requests:
                if not self.bench_admit_all_at_once and sum_input_len >= self.prefill_budget:
                    break
                input_ids = self._tokenize_one(tokens_or_prompt)
                sum_input_len += len(input_ids)
                uid, added = self.counter + added, added + 1
                results.append(
                    UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
                )
                self.status_map[uid] = BenchRequestStatus(
                    uid=uid,
                    input_ids=(
                        input_ids.tolist()
                        if isinstance(tokens_or_prompt, str)
                        else list(tokens_or_prompt)
                    ),
                    output_ids=[],
                    requested_output_len=sampling_params.max_tokens,
                    started_at=self._active_generation_started_at,
                )
            self.counter += added
            self.pending_requests = self.pending_requests[added:]
            return results

        def offline_send_result(self, reply: list[DetokenizeMsg]) -> None:
            timestamp = time.perf_counter()
            for msg in reply:
                status = self.status_map[msg.uid]
                emitted_output_token = not (msg.finished and msg.next_token == self.eos_token_id)
                if emitted_output_token:
                    status.output_ids.append(msg.next_token)
                    status.record_token(timestamp)
                if msg.finished:
                    status.finish_reason = msg.finish_reason or "stop"
                    status.error = msg.error
                    status.mark_finished(timestamp)

        def _prepare_batch(self, batch):
            sync_prepare = os.environ.get("MINISGL_BENCH_SYNC_PREPARE_NVTX", "0") == "1"
            range_name = f"batch_prepare:{batch.phase}:bs{batch.size}"
            if sync_prepare:
                torch.cuda.synchronize(self.device)
            with torch.cuda.nvtx.range(range_name):
                tic = time.perf_counter()
                forward_input = super()._prepare_batch(batch)
                if sync_prepare:
                    torch.cuda.synchronize(self.device)
                toc = time.perf_counter()
            batch_id = id(forward_input.batch)
            self._bench_prepare_s[batch_id] = toc - tic
            self._bench_batch_info[batch_id] = {
                "phase": batch.phase,
                "batch_size": batch.size,
                "padded_size": batch.padded_size,
                "prepare_sync_profile": sync_prepare,
                "input_tokens": int(sum(req.extend_len for req in batch.reqs)),
                "decode_tokens": int(batch.size if batch.is_decode else 0),
                "max_extend_len": int(max((req.extend_len for req in batch.reqs), default=0)),
                "max_device_len": int(max((req.device_len for req in batch.reqs), default=0)),
                "reqs": [
                    {
                        "uid": req.uid,
                        "cached_len": int(req.cached_len),
                        "extend_len": int(req.extend_len),
                        "device_len": int(req.device_len),
                        "remain_len": int(req.remain_len),
                        "is_chunked": type(req).__name__ == "ChunkedReq",
                    }
                    for req in batch.reqs
                ],
            }
            return forward_input

        def _forward(self, forward_input):
            batch = forward_input.batch
            batch_id = id(batch)
            stderr_batch_trace = os.environ.get("MINISGL_BENCH_STDERR_BATCH_TRACE", "0") == "1"
            if stderr_batch_trace:
                stderr_batch_trace = int(os.environ.get("RANK", "0")) == 0
            range_name = f"batch_forward:{batch.phase}:bs{batch.size}:padded{batch.padded_size}"
            enqueue_range_name = (
                f"batch_forward_enqueue:{batch.phase}:" f"bs{batch.size}:padded{batch.padded_size}"
            )
            if stderr_batch_trace:
                max_device_len = max((int(req.device_len) for req in batch.reqs), default=0)
                max_cached_len = max((int(req.cached_len) for req in batch.reqs), default=0)
                print(
                    "[bench-batch-start] "
                    f"phase={batch.phase} bs={batch.size} padded={batch.padded_size} "
                    f"max_device_len={max_device_len} max_cached_len={max_cached_len}",
                    flush=True,
                )
            graph_before = copy.deepcopy(getattr(self.engine.graph_runner, "capture_status", {}))
            torch.cuda.synchronize(self.device)
            long_prefill_timing = (
                os.environ.get("MINISGL_DSV4_LONG_PREFILL_TIMING", "").strip().lower()
                in {"1", "true", "yes", "on"}
                and bool(batch.is_prefill)
            )
            chunk_memory_before = None
            if long_prefill_timing:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                chunk_memory_before = {
                    "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                }
                torch.cuda.reset_peak_memory_stats(self.device)
            profiler_checkpoint = False
            raw_profiler_contexts = os.environ.get(
                "MINISGL_DSV4_LONG_PREFILL_PROFILER_CHECKPOINTS", ""
            ).strip()
            if raw_profiler_contexts and batch.is_prefill:
                profiler_contexts = {
                    int(value.strip())
                    for value in raw_profiler_contexts.split(",")
                    if value.strip()
                }
                committed_context = max(
                    (int(req.device_len) for req in batch.reqs), default=0
                )
                profiler_checkpoint = committed_context in profiler_contexts
            if profiler_checkpoint:
                torch.cuda.profiler.start()
            with torch.cuda.nvtx.range(range_name):
                tic = time.perf_counter()
                enqueue_tic = time.perf_counter()
                with torch.cuda.nvtx.range(enqueue_range_name):
                    output = super()._forward(forward_input)
                enqueue_toc = time.perf_counter()
                torch.cuda.synchronize(self.device)
                toc = time.perf_counter()
            if profiler_checkpoint:
                torch.cuda.profiler.stop()
            graph_after = getattr(self.engine.graph_runner, "capture_status", {})
            info = self._bench_batch_info.pop(batch_id, {})
            replay_delta = int(graph_after.get("replay_count") or 0) - int(
                graph_before.get("replay_count") or 0
            )
            eager_delta = int(graph_after.get("eager_decode_count") or 0) - int(
                graph_before.get("eager_decode_count") or 0
            )
            info.update(
                {
                    "prepare_s": self._bench_prepare_s.pop(batch_id, 0.0),
                    "forward_s": toc - tic,
                    "forward_enqueue_s": enqueue_toc - enqueue_tic,
                    "graph_replay": bool(batch.is_decode and replay_delta > 0),
                    "graph_eager": bool(batch.is_decode and eager_delta > 0),
                    "graph_replay_delta": replay_delta,
                    "graph_eager_delta": eager_delta,
                }
            )
            if long_prefill_timing:
                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                allocated = int(torch.cuda.memory_allocated(self.device))
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
                info["long_prefill_memory"] = {
                    "before": chunk_memory_before,
                    "after": {
                        "allocated_bytes": allocated,
                        "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                        "free_bytes": int(free_bytes),
                        "total_bytes": int(total_bytes),
                    },
                    "peak_allocated_bytes": peak_allocated,
                    "temporary_high_water_bytes": max(0, peak_allocated - allocated),
                }
            if self._active_generation_started_at is not None:
                info["completed_since_generate_start_s"] = (
                    toc - self._active_generation_started_at
                )
            self.bench_batch_trace.append(info)
            if stderr_batch_trace:
                print(
                    "[bench-batch-done] "
                    f"phase={batch.phase} bs={batch.size} padded={batch.padded_size} "
                    f"forward_s={toc - tic:.6f} replay_delta={replay_delta} "
                    f"eager_delta={eager_delta}",
                    flush=True,
                )
            return output

        def generate(self, prompts, sampling_params):
            self.bench_batch_trace = []
            self._active_generation_started_at = time.perf_counter()
            self._active_generation_finished_at = None
            try:
                return super().generate(prompts, sampling_params)
            finally:
                self._active_generation_finished_at = time.perf_counter()

        def request_metrics(self) -> list[dict[str, Any]]:
            finished_at = self._active_generation_finished_at
            metrics = []
            for uid in sorted(self.status_map):
                status = self.status_map[uid]
                token_times = status.token_times or []
                req_finished_at = status.finished_at or finished_at
                started_at = status.started_at
                metrics.append(
                    {
                        "uid": uid,
                        "input_tokens": len(status.input_ids),
                        "output_tokens": len(status.output_ids),
                        "output_token_ids": list(status.output_ids),
                        "requested_output_len": status.requested_output_len,
                        "admitted_output_len": status.admitted_output_len,
                        "finish_reason": status.finish_reason,
                        "error": status.error,
                        "ttft_s": (
                            None
                            if started_at is None or status.first_token_at is None
                            else status.first_token_at - started_at
                        ),
                        "latency_s": (
                            None
                            if started_at is None or req_finished_at is None
                            else req_finished_at - started_at
                        ),
                        "topt_s": (
                            None
                            if len(token_times) <= 1
                            else (token_times[-1] - token_times[0]) / (len(token_times) - 1)
                        ),
                        "token_times_s": (
                            []
                            if started_at is None
                            else [timestamp - started_at for timestamp in token_times]
                        ),
                    }
                )
            return metrics

    return BenchmarkLLM


class KernelCallTracer:
    def __init__(self, module) -> None:
        self.module = module
        self.originals: dict[str, Callable[..., Any]] = {}
        self.call_counts: dict[str, int] = {}
        self.none_skip_counts: dict[str, int] = {}
        self.unsupported_counts: dict[str, int] = {}
        self.exception_counts: dict[str, int] = {}
        self.indexer_samples: list[dict[str, Any]] = []

    def install(self) -> None:
        for name in sorted(FALLBACK_COUNTER_NAMES):
            value = getattr(self.module, name, None)
            if callable(value):
                self._wrap(name, value)
        indexer_select = getattr(self.module, "indexer_select_fp8_paged_fallback", None)
        if callable(indexer_select):
            self._wrap_indexer_select(indexer_select)
        unsupported = getattr(self.module, "unsupported_kernel", None)
        if callable(unsupported):
            self._wrap_unsupported(unsupported)

    def reset(self) -> None:
        self.call_counts.clear()
        self.none_skip_counts.clear()
        self.unsupported_counts.clear()
        self.exception_counts.clear()
        self.indexer_samples.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "fallback_wrapper_calls_total": int(sum(self.call_counts.values())),
            "fallback_wrapper_calls": dict(sorted(self.call_counts.items())),
            "optional_kernel_none_skips_total": int(sum(self.none_skip_counts.values())),
            "optional_kernel_none_skips": dict(sorted(self.none_skip_counts.items())),
            "unsupported_kernel_skips_total": int(sum(self.unsupported_counts.values())),
            "unsupported_kernel_skips": dict(sorted(self.unsupported_counts.items())),
            "wrapper_exceptions": dict(sorted(self.exception_counts.items())),
            "indexer_profile": self._indexer_profile_snapshot(),
        }

    def _indexer_profile_snapshot(self) -> dict[str, Any]:
        by_backend_shape: dict[str, dict[str, Any]] = {}
        total_ms = 0.0
        timed_count = 0
        for sample in self.indexer_samples:
            elapsed_ms = None
            start = sample.get("start")
            end = sample.get("end")
            if start is not None and end is not None:
                try:
                    elapsed_ms = float(start.elapsed_time(end))
                except Exception:
                    elapsed_ms = None
            key = json.dumps(
                {
                    "backend": sample["backend"],
                    "rows": sample["rows"],
                    "page_table_shape": sample["page_table_shape"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            bucket = by_backend_shape.setdefault(
                key,
                {
                    "backend": sample["backend"],
                    "rows": sample["rows"],
                    "page_table_shape": sample["page_table_shape"],
                    "count": 0,
                    "timed_count": 0,
                    "total_ms": 0.0,
                    "max_ms": None,
                },
            )
            bucket["count"] += 1
            if elapsed_ms is not None:
                bucket["timed_count"] += 1
                bucket["total_ms"] += elapsed_ms
                bucket["max_ms"] = (
                    elapsed_ms
                    if bucket["max_ms"] is None
                    else max(float(bucket["max_ms"]), elapsed_ms)
                )
                timed_count += 1
                total_ms += elapsed_ms
        entries = sorted(
            by_backend_shape.values(),
            key=lambda entry: (
                entry["rows"],
                entry["page_table_shape"],
                entry["backend"],
            ),
        )
        for entry in entries:
            entry["mean_ms"] = (
                None
                if entry["timed_count"] == 0
                else entry["total_ms"] / entry["timed_count"]
            )
        budget_env = getattr(
            self.module,
            "DSV4_INDEXER_MAX_LOGITS_MB_ENV",
            "MINISGL_DSV4_INDEXER_MAX_LOGITS_MB",
        )
        default_budget = getattr(self.module, "DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT", 512)
        return {
            "call_count": len(self.indexer_samples),
            "timed_count": timed_count,
            "total_ms": total_ms,
            "configured_max_logits_mb": int(
                os.environ.get(budget_env, str(default_budget))
            ),
            "entries": entries,
        }

    def _wrap(self, name: str, func: Callable[..., Any]) -> None:
        if name in self.originals:
            return
        self.originals[name] = func

        def wrapper(*args, **kwargs):
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
            try:
                result = func(*args, **kwargs)
            except Exception:
                self.exception_counts[name] = self.exception_counts.get(name, 0) + 1
                raise
            if result is None and name in OPTIONAL_NONE_MEANS_SKIP:
                self.none_skip_counts[name] = self.none_skip_counts.get(name, 0) + 1
            return result

        setattr(self.module, name, wrapper)

    def _wrap_indexer_select(self, func: Callable[..., Any]) -> None:
        name = "indexer_select_fp8_paged_fallback"
        if name in self.originals:
            return
        self.originals[name] = func

        def wrapper(*args, **kwargs):
            start = None
            end = None
            if args:
                try:
                    import torch

                    if getattr(args[0], "is_cuda", False):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                except Exception:
                    start = None
                    end = None
            try:
                result = func(*args, **kwargs)
            except Exception:
                self.exception_counts[name] = self.exception_counts.get(name, 0) + 1
                raise
            if end is not None:
                try:
                    end.record()
                except Exception:
                    start = None
                    end = None
            self.indexer_samples.append(
                {
                    "backend": str(getattr(result, "backend", "unknown")),
                    "rows": int(args[0].shape[0]) if args else 0,
                    "page_table_shape": list(args[4].shape) if len(args) > 4 else [],
                    "start": start,
                    "end": end,
                }
            )
            return result

        setattr(self.module, name, wrapper)

    def _wrap_unsupported(self, func: Callable[..., Any]) -> None:
        name = "unsupported_kernel"
        if name in self.originals:
            return
        self.originals[name] = func

        def wrapper(kernel_name, detail):
            key = str(kernel_name)
            self.unsupported_counts[key] = self.unsupported_counts.get(key, 0) + 1
            return func(kernel_name, detail)

        setattr(self.module, name, wrapper)


def _dtype_from_name(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


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


def _select_scenarios(args: argparse.Namespace) -> list[Scenario]:
    if args.smoke:
        scenario = Scenario(
            name="smoke_debug",
            kind="random",
            batch_size=args.batch_size or 1,
            prompt_len=args.prompt_len or 16,
            decode_len=args.decode_len or 2,
            repeats=args.repeats or 1,
            warmup_repeats=args.warmup_repeats if args.warmup_repeats is not None else 0,
            description="Tiny smoke/debug workload, excluded from official baseline summaries.",
        )
        return [scenario]

    selected = args.scenarios or [scenario.name for scenario in DEFAULT_SCENARIOS]
    scenario_map = _scenario_map()
    scenarios = [scenario_map[name] for name in selected]
    output = []
    for scenario in scenarios:
        overrides: dict[str, int] = {}
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        if args.prompt_len is not None:
            overrides["prompt_len"] = args.prompt_len
            if scenario.kind == "shared_prefix":
                overrides["shared_prefix_len"] = max(args.prompt_len - scenario.suffix_len, 1)
        if args.decode_len is not None:
            overrides["decode_len"] = args.decode_len
        if args.repeats is not None:
            overrides["repeats"] = args.repeats
        if args.warmup_repeats is not None:
            overrides["warmup_repeats"] = args.warmup_repeats
        output.append(replace(scenario, **overrides))
    return output


def _select_variants(args: argparse.Namespace) -> list[Variant]:
    names = args.variants or [DSV4_RELEASE_DEFAULT_VARIANT]
    if names != [DSV4_RELEASE_DEFAULT_VARIANT]:
        raise SystemExit(
            "The release perf matrix supports only the canonical optimized runtime."
        )
    variant_map = _variant_map()
    return [variant_map[name] for name in names]


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
        raise SystemExit("PyNCCL variants must be run separately or with --use-pynccl.")
    if (
        variant_graph
        and not all(variant.allow_dsv4_cuda_graph for variant in variants)
        and not args.allow_dsv4_cuda_graph
    ):
        raise SystemExit(
            "DSV4 CUDA graph variants must be run separately or with --allow-dsv4-cuda-graph."
        )

    allow_dsv4_cuda_graph = bool(args.allow_dsv4_cuda_graph or variant_graph)
    cuda_graph_bs = args.cuda_graph_bs
    if args.cuda_graph_capture_greedy_sample is None:
        cuda_graph_capture_greedy_sample = variant_graph_greedy_sample
    else:
        cuda_graph_capture_greedy_sample = bool(args.cuda_graph_capture_greedy_sample)
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
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "cuda_graph_capture_greedy_sample": cuda_graph_capture_greedy_sample,
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


def _scenario_max_running_req(scenarios: Sequence[Scenario]) -> int:
    return max((scenario.batch_size for scenario in scenarios), default=1)


@dataclass(frozen=True)
class MaxRunningReqModeConfig:
    mode: str
    requested_override: int | None
    scenario_required_max_running_req: int


def _max_running_req_mode_config(
    args: argparse.Namespace, scenarios: Sequence[Scenario]
) -> MaxRunningReqModeConfig:
    scenario_required = _scenario_max_running_req(scenarios)
    if args.max_running_req is not None:
        return MaxRunningReqModeConfig(
            mode="explicit_override",
            requested_override=int(args.max_running_req),
            scenario_required_max_running_req=scenario_required,
        )
    if args.scenario_sized_max_running_req:
        return MaxRunningReqModeConfig(
            mode="scenario_sized_diagnostic",
            requested_override=scenario_required,
            scenario_required_max_running_req=scenario_required,
        )
    return MaxRunningReqModeConfig(
        mode="serving_model_default",
        requested_override=None,
        scenario_required_max_running_req=scenario_required,
    )


def _max_running_req_llm_kwargs(
    args: argparse.Namespace, scenarios: Sequence[Scenario]
) -> dict[str, int]:
    mode = _max_running_req_mode_config(args, scenarios)
    if mode.requested_override is None:
        return {}
    return {"max_running_req": mode.requested_override}


def _validate_max_running_req(
    report: dict[str, Any], scenarios: Sequence[Scenario]
) -> None:
    required = _scenario_max_running_req(scenarios)
    effective = int(report["effective"])
    if required > effective:
        raise RuntimeError(
            "Selected benchmark scenarios require decode/request batch capacity "
            f"{required}, but effective max_running_req is {effective} in "
            f"{report['mode']} mode. Use an explicit --max-running-req override; "
            "do not relabel a scenario-sized diagnostic as a serving baseline."
        )


def _max_seq_len(scenarios: Sequence[Scenario]) -> int:
    return max((scenario.max_seq_len for scenario in scenarios), default=1)


@dataclass(frozen=True)
class MaxSequenceModeConfig:
    mode: str
    requested_override: int | None
    scenario_required_total_seq_len: int


def _max_sequence_mode_config(
    args: argparse.Namespace, scenarios: Sequence[Scenario]
) -> MaxSequenceModeConfig:
    scenario_required = _max_seq_len(scenarios)
    if args.max_seq_len is not None:
        return MaxSequenceModeConfig(
            mode="explicit_override",
            requested_override=int(args.max_seq_len),
            scenario_required_total_seq_len=scenario_required,
        )
    if args.scenario_sized_max_seq_len:
        return MaxSequenceModeConfig(
            mode="scenario_sized_diagnostic",
            requested_override=scenario_required,
            scenario_required_total_seq_len=scenario_required,
        )
    return MaxSequenceModeConfig(
        mode="model_default",
        requested_override=None,
        scenario_required_total_seq_len=scenario_required,
    )


def _max_sequence_llm_kwargs(
    args: argparse.Namespace, scenarios: Sequence[Scenario]
) -> dict[str, int]:
    mode = _max_sequence_mode_config(args, scenarios)
    if mode.requested_override is None:
        return {}
    return {"max_seq_len_override": mode.requested_override}


def _max_extend_tokens(scenarios: Sequence[Scenario]) -> int:
    return max((scenario.batch_size * scenario.max_input_len for scenario in scenarios), default=1)


def _safe_mean(values: Sequence[float]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.mean(filtered))


def _safe_median(values: Sequence[float]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    filtered = sorted(float(value) for value in values if value is not None)
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]
    rank = (len(filtered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return filtered[lower]
    weight = rank - lower
    return filtered[lower] * (1.0 - weight) + filtered[upper] * weight


def _latency_percentiles(values: Sequence[float]) -> dict[str, float | None]:
    return {
        f"p{percentile}": _percentile(values, percentile)
        for percentile in (50, 90, 95, 99)
    }


def _decode_m_range(value: int) -> str:
    for low, high in ((1, 16), (17, 128), (129, 256), (257, 384), (385, 512)):
        if low <= value <= high:
            return f"{low}-{high}"
    return "above-512" if value > 512 else "zero"


def _decode_m_time_distribution(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    active: dict[str, dict[str, float | int]] = {}
    padded: dict[str, dict[str, float | int]] = {}
    dispatch: dict[str, dict[str, float | int]] = {}

    def add(target, key: str, *, elapsed: float) -> None:
        bucket = target.setdefault(key, {"count": 0, "forward_s": 0.0})
        bucket["count"] = int(bucket["count"]) + 1
        bucket["forward_s"] = float(bucket["forward_s"]) + elapsed

    for repeat in repeats:
        for row in repeat.get("schedule_trace", []):
            if row.get("phase") != "decode":
                continue
            elapsed = float(row.get("forward_s") or 0.0)
            active_m = int(row.get("batch_size") or 0)
            padded_m = int(row.get("padded_size") or active_m)
            active_range = _decode_m_range(active_m)
            add(active, str(active_m), elapsed=elapsed)
            add(padded, str(padded_m), elapsed=elapsed)
            mode = "replay" if row.get("graph_replay") else "eager"
            add(dispatch, f"{active_range}:{mode}", elapsed=elapsed)

    def sorted_m(mapping: dict[str, Any]) -> dict[str, Any]:
        return dict(sorted(mapping.items(), key=lambda item: int(item[0])))

    return {
        "active_m": sorted_m(active),
        "resolved_or_padded_m": sorted_m(padded),
        "dispatch_by_active_m_range": dict(sorted(dispatch.items())),
    }


def _sum_phase(trace: Sequence[dict[str, Any]], phase: str, key: str) -> float:
    return float(sum(float(row.get(key, 0.0)) for row in trace if row.get("phase") == phase))


def _sum_trace_int(trace: Sequence[dict[str, Any]], phase: str, key: str) -> int:
    return int(sum(int(row.get(key, 0)) for row in trace if row.get("phase") == phase))


def _schedule_summary(repeats: Sequence[dict[str, Any]]) -> dict[str, Any]:
    phase_counts: dict[str, int] = {}
    batch_counts: dict[str, int] = {}
    padded_counts: dict[str, int] = {}
    phase_batch_counts: dict[str, int] = {}
    phase_padded_counts: dict[str, int] = {}
    total_batches = 0
    max_batch_size = 0
    max_padded_size = 0
    for repeat in repeats:
        for row in repeat.get("schedule_trace", []):
            total_batches += 1
            phase = str(row.get("phase", "unknown"))
            batch_size = int(row.get("batch_size", 0))
            padded_size = int(row.get("padded_size", batch_size))
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            batch_key = str(batch_size)
            padded_key = str(padded_size)
            batch_counts[batch_key] = batch_counts.get(batch_key, 0) + 1
            padded_counts[padded_key] = padded_counts.get(padded_key, 0) + 1
            phase_batch_key = f"{phase}:bs{batch_size}"
            phase_padded_key = f"{phase}:padded{padded_size}"
            phase_batch_counts[phase_batch_key] = phase_batch_counts.get(phase_batch_key, 0) + 1
            phase_padded_counts[phase_padded_key] = phase_padded_counts.get(phase_padded_key, 0) + 1
            max_batch_size = max(max_batch_size, batch_size)
            max_padded_size = max(max_padded_size, padded_size)
    return {
        "total_batches": total_batches,
        "phase_counts": dict(sorted(phase_counts.items())),
        "batch_size_counts": dict(sorted(batch_counts.items(), key=lambda item: int(item[0]))),
        "padded_size_counts": dict(sorted(padded_counts.items(), key=lambda item: int(item[0]))),
        "phase_batch_size_counts": dict(sorted(phase_batch_counts.items())),
        "phase_padded_size_counts": dict(sorted(phase_padded_counts.items())),
        "max_batch_size": max_batch_size,
        "max_padded_size": max_padded_size,
    }


def _snapshot_graph_status(llm) -> dict[str, Any]:
    return copy.deepcopy(getattr(llm.engine.graph_runner, "capture_status", {}))


def _snapshot_c128_prefill_one_surface(llm) -> dict[str, Any]:
    snapshot = getattr(llm.engine.attn_backend, "c128_prefill_one_surface_status", None)
    return copy.deepcopy(snapshot()) if callable(snapshot) else {}


def _tensor_tree_report(value: Any, *, root: str) -> dict[str, Any]:
    """Describe tensor-owned allocation surfaces without copying tensor contents."""
    try:
        import torch
    except ImportError:
        return {"total_bytes": 0, "tensors": []}

    tensors: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int]] = set()

    def visit(current: Any, path: str, depth: int) -> None:
        if depth > 12:
            return
        if isinstance(current, torch.Tensor):
            storage_key = (str(current.device), int(current.untyped_storage().data_ptr()))
            if storage_key in seen_storages:
                return
            seen_storages.add(storage_key)
            tensors.append(
                {
                    "path": path,
                    "shape": list(current.shape),
                    "dtype": str(current.dtype).removeprefix("torch."),
                    "device": str(current.device),
                    "bytes": int(current.numel() * current.element_size()),
                }
            )
            return
        if current is None or isinstance(current, (str, bytes, int, float, bool)):
            return
        object_id = id(current)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if is_dataclass(current) and not isinstance(current, type):
            for field in fields(current):
                visit(getattr(current, field.name), f"{path}.{field.name}", depth + 1)
        elif isinstance(current, dict):
            for key, item in current.items():
                visit(item, f"{path}.{key}", depth + 1)
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]", depth + 1)
        elif hasattr(current, "__dict__"):
            for name, item in vars(current).items():
                if name.startswith("__"):
                    continue
                visit(item, f"{path}.{name}", depth + 1)

    visit(value, root, 0)
    tensors.sort(key=lambda row: row["path"])
    return {
        "total_bytes": int(sum(row["bytes"] for row in tensors)),
        "tensors": tensors,
    }


def _rope_tensor_report(model: Any) -> list[dict[str, Any]]:
    import torch

    tensors: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int]] = set()

    def visit(current: Any, path: str, depth: int) -> None:
        if depth > 12 or current is None:
            return
        if isinstance(current, torch.Tensor):
            if "cos_sin_cache" not in path and "rope_cache" not in path:
                return
            storage_key = (str(current.device), int(current.untyped_storage().data_ptr()))
            if storage_key in seen_storages:
                return
            seen_storages.add(storage_key)
            tensors.append(
                {
                    "path": path,
                    "shape": list(current.shape),
                    "dtype": str(current.dtype).removeprefix("torch."),
                    "device": str(current.device),
                    "bytes": int(current.numel() * current.element_size()),
                }
            )
            return
        if isinstance(current, (str, bytes, int, float, bool)):
            return
        object_id = id(current)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if isinstance(current, dict):
            iterator = current.items()
        elif isinstance(current, (list, tuple)):
            iterator = enumerate(current)
        elif hasattr(current, "__dict__"):
            iterator = vars(current).items()
        else:
            return
        for name, item in iterator:
            visit(item, f"{path}.{name}", depth + 1)

    visit(model, "model", 0)
    return sorted(tensors, key=lambda row: row["path"])


def _max_sequence_runtime_report(
    llm, *, args: argparse.Namespace, scenarios: Sequence[Scenario]
) -> dict[str, Any]:
    mode = _max_sequence_mode_config(args, scenarios)
    engine = llm.engine
    model_config = engine.attn_backend.config
    model_config_max = int(model_config.rotary_config.max_position)
    effective_max = int(engine.max_seq_len)
    rope_kind = str(getattr(engine, "rope_cache_kind", "materialized"))
    rope_tensors = _rope_tensor_report(engine.model)
    rope_bytes = int(sum(row["bytes"] for row in rope_tensors))
    physical_rope_lengths = [
        int(row["shape"][0]) for row in rope_tensors if row.get("shape")
    ]
    effective_rope_cache_len = int(
        getattr(
            engine,
            "effective_rope_cache_len",
            min(physical_rope_lengths) if physical_rope_lengths else effective_max,
        )
    )
    page_table = engine.page_table
    capture = getattr(engine.attn_backend, "capture", None)
    capture_report = _tensor_tree_report(capture, root="attention_capture")
    return {
        "model_config_max_seq_len": model_config_max,
        "max_seq_len_mode": mode.mode,
        "requested_max_seq_len_override": mode.requested_override,
        "scenario_required_total_seq_len": mode.scenario_required_total_seq_len,
        "effective_engine_max_seq_len": effective_max,
        "effective_rope_cache_len": effective_rope_cache_len,
        "rope_cache_kind": rope_kind,
        "rope_cache_bytes": rope_bytes,
        "rope_cache_tensors": rope_tensors,
        "prompt_len_requested_max": max(
            (scenario.max_input_len for scenario in scenarios), default=0
        ),
        "decode_len_requested_max": max(
            (
                max(scenario.decode_len_cycle)
                if scenario.decode_len_cycle
                else scenario.decode_len
                for scenario in scenarios
            ),
            default=0,
        ),
        "scenario_fits_effective_engine": (
            mode.scenario_required_total_seq_len <= effective_max
        ),
        "context_page_table": {
            "shape": list(page_table.shape),
            "bytes": int(page_table.numel() * page_table.element_size()),
        },
        "attention_graph_capture": capture_report,
    }


def _validate_selected_scenarios(
    report: dict[str, Any], scenarios: Sequence[Scenario]
) -> None:
    effective_max = int(report["effective_engine_max_seq_len"])
    rope_len = int(report["effective_rope_cache_len"])
    for scenario in scenarios:
        if scenario.max_seq_len > effective_max:
            raise RuntimeError(
                "benchmark max-sequence configuration error: scenario "
                f"{scenario.name!r} requires total sequence length {scenario.max_seq_len}, "
                f"but effective engine max is {effective_max} in "
                f"{report['max_seq_len_mode']} mode"
            )
        max_model_position = scenario.max_seq_len - 2
        if scenario.decode_len <= 0:
            max_model_position = scenario.max_input_len - 1
        if max_model_position >= rope_len:
            raise RuntimeError(
                "benchmark RoPE configuration error: scenario "
                f"{scenario.name!r} may schedule position {max_model_position}, but effective "
                f"RoPE cache length is {rope_len}"
            )


def _case_boundary_debug_enabled() -> bool:
    return os.environ.get(DSV4_CASE_BOUNDARY_DEBUG_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _case_boundary_debug_snapshot(llm, stage: str) -> dict[str, Any]:
    if not _case_boundary_debug_enabled():
        return {}
    try:
        if getattr(llm, "device", None) is not None and llm.device.type == "cuda":
            import torch

            torch.cuda.synchronize(llm.device)
        snapshot = getattr(llm.cache_manager, "debug_case_boundary_snapshot", None)
        if callable(snapshot):
            return snapshot(stage, graph_runner=_snapshot_graph_status(llm))
        return {"stage": stage, "graph_runner": _snapshot_graph_status(llm)}
    except BaseException as exc:
        raise RuntimeError(f"DSV4 case-boundary debug failed at {stage}") from exc


def _counter_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    key: str,
) -> int:
    return int(after.get(key) or 0) - int(before.get(key) or 0)


def _dict_counter_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    key: str,
) -> dict[str, int]:
    before_counter = before.get(key, {}) or {}
    after_counter = after.get(key, {}) or {}
    keys = set(before_counter) | set(after_counter)
    delta: dict[str, int] = {}
    for item_key in keys:
        value = int(after_counter.get(item_key, 0)) - int(before_counter.get(item_key, 0))
        if value:
            delta[str(item_key)] = value
    return dict(sorted(delta.items(), key=lambda item: int(item[0])))


def _replay_timing_bucket_delta(
    before_bucket: dict[str, Any],
    after_bucket: dict[str, Any],
) -> dict[str, Any] | None:
    count = int(after_bucket.get("count") or 0) - int(before_bucket.get("count") or 0)
    total_s = float(after_bucket.get("total_s") or 0.0) - float(
        before_bucket.get("total_s") or 0.0
    )
    if count <= 0:
        return None
    bucket = {
        "count": count,
        "total_s": total_s,
        "mean_s": total_s / count,
    }
    if int(before_bucket.get("count") or 0) == 0:
        bucket["min_s"] = after_bucket.get("min_s")
        bucket["max_s"] = after_bucket.get("max_s")
    else:
        bucket["min_s"] = None
        bucket["max_s"] = None
        bucket["min_max_case_delta_unavailable"] = True
    return bucket


def _replay_timing_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_timing = before.get("replay_timing", {}) or {}
    after_timing = after.get("replay_timing", {}) or {}
    if not isinstance(after_timing, dict):
        return {}
    if not isinstance(before_timing, dict):
        before_timing = {}

    status = copy.deepcopy(after_timing)
    before_count = int(before_timing.get("count") or 0)
    count = int(after_timing.get("count") or 0) - before_count
    total_s = float(after_timing.get("total_s") or 0.0) - float(
        before_timing.get("total_s") or 0.0
    )
    status["count"] = count
    status["total_s"] = total_s
    status["mean_s"] = total_s / count if count > 0 else None
    if before_count == 0:
        status["min_s"] = after_timing.get("min_s")
        status["max_s"] = after_timing.get("max_s")
    else:
        status["min_s"] = None
        status["max_s"] = None
        status["min_max_case_delta_unavailable"] = True
        status["samples"] = []
        status["samples_are_global_first_replays"] = True

    for section in ("by_batch_size", "by_padded_size"):
        before_buckets = before_timing.get(section, {}) or {}
        after_buckets = after_timing.get(section, {}) or {}
        bucket_delta: dict[str, Any] = {}
        for key in set(before_buckets) | set(after_buckets):
            delta = _replay_timing_bucket_delta(
                before_buckets.get(key, {}) or {},
                after_buckets.get(key, {}) or {},
            )
            if delta is not None:
                bucket_delta[str(key)] = delta
        status[section] = dict(sorted(bucket_delta.items(), key=lambda item: int(item[0])))
    return status


def _graph_status_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    status = copy.deepcopy(after)
    for key in (
        "replay_count",
        "greedy_sample_replay_count",
        "replay_input_copy_bytes",
        "eager_decode_count",
    ):
        status[key] = _counter_delta(before, after, key)
    for key in (
        "replay_count_by_batch_size",
        "replay_count_by_padded_size",
        "greedy_sample_replay_count_by_batch_size",
        "eager_decode_count_by_batch_size",
    ):
        status[key] = _dict_counter_delta(before, after, key)
    if "replay_timing" in after:
        status["replay_timing"] = _replay_timing_delta(before, after)
    return status


def _decode_row_used_replay(row: dict[str, Any], graph_status: dict[str, Any]) -> bool:
    if "graph_replay" in row:
        return bool(row.get("graph_replay"))
    captured = {int(value) for value in graph_status.get("captured_bs", [])}
    return bool(graph_status.get("enabled")) and int(row.get("padded_size", 0)) in captured


def _bucket_coverage_table(
    repeats: Sequence[dict[str, Any]],
    graph_status: dict[str, Any],
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, Any]] = {}
    total_decode_wall_s = 0.0
    for repeat in repeats:
        for row in repeat.get("schedule_trace", []):
            if row.get("phase") != "decode":
                continue
            batch_size = int(row.get("batch_size", 0))
            if batch_size <= 0:
                continue
            wall_s = float(row.get("forward_s") or 0.0)
            tokens = int(row.get("decode_tokens") or batch_size)
            total_decode_wall_s += wall_s
            bucket = buckets.setdefault(
                batch_size,
                {
                    "actual_decode_bs": batch_size,
                    "replay_count": 0,
                    "eager_count": 0,
                    "tokens": 0,
                    "wall_s": 0.0,
                    "padded_size_counts": {},
                },
            )
            replay = _decode_row_used_replay(row, graph_status)
            eager = bool(row.get("graph_eager")) if "graph_eager" in row else not replay
            if replay:
                bucket["replay_count"] += 1
            if eager:
                bucket["eager_count"] += 1
            bucket["tokens"] += tokens
            bucket["wall_s"] += wall_s
            padded_key = str(int(row.get("padded_size", batch_size)))
            padded_counts = bucket["padded_size_counts"]
            padded_counts[padded_key] = int(padded_counts.get(padded_key, 0)) + 1

    replay_by_bs = graph_status.get("replay_count_by_batch_size", {}) or {}
    eager_by_bs = graph_status.get("eager_decode_count_by_batch_size", {}) or {}
    for key, value in replay_by_bs.items():
        batch_size = int(key)
        bucket = buckets.setdefault(
            batch_size,
            {
                "actual_decode_bs": batch_size,
                "replay_count": 0,
                "eager_count": 0,
                "tokens": 0,
                "wall_s": 0.0,
                "padded_size_counts": {},
            },
        )
        bucket["replay_count"] = int(value)
    for key, value in eager_by_bs.items():
        batch_size = int(key)
        bucket = buckets.setdefault(
            batch_size,
            {
                "actual_decode_bs": batch_size,
                "replay_count": 0,
                "eager_count": 0,
                "tokens": 0,
                "wall_s": 0.0,
                "padded_size_counts": {},
            },
        )
        bucket["eager_count"] = int(value)

    rows = []
    for batch_size, bucket in sorted(buckets.items()):
        wall_s = float(bucket["wall_s"])
        rows.append(
            {
                **bucket,
                "wall_s": wall_s,
                "wall_share": None if total_decode_wall_s <= 0 else wall_s / total_decode_wall_s,
            }
        )
    return rows


def _rank_memory_report(torch, llm) -> dict[str, int]:
    free_memory, total_memory = torch.cuda.mem_get_info(llm.device)
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(llm.device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(llm.device)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(llm.device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(llm.device)),
        "free_memory_bytes": int(free_memory),
        "total_memory_bytes": int(total_memory),
    }


def _estimate_kv_cache_bytes_from_config(llm, *, page_size: int, dtype, tp_size: int) -> int:
    from minisgl.kvcache import estimate_kvcache_bytes_per_page

    model_config = llm.engine.attn_backend.config
    pages = int(getattr(llm.engine.kv_cache, "_num_pages", llm.engine.num_pages))
    return int(
        pages
        * estimate_kvcache_bytes_per_page(
            model_config,
            page_size=page_size,
            dtype=dtype,
            tp_size=tp_size,
        )
    )


def _generation_parts(
    scenario: Scenario,
    prompts: list[list[int]],
    sampling_params: list[Any],
) -> tuple[tuple[list[list[int]], list[Any]], ...]:
    if (
        scenario.kind
        in {
            "shared_prefix_reuse",
            "prefix_full_hit_reuse",
            "prefix_partial_hit_reuse",
            "prefix_mixed_hit_miss",
        }
        and len(prompts) > 1
    ):
        return (
            (prompts[:1], sampling_params[:1]),
            (prompts[1:], sampling_params[1:]),
        )
    if (
        scenario.kind
        in {
            "serving_mixed",
            "cuda_graph_recipe_balanced_wave",
            "cuda_graph_recipe_high_concurrency_wave",
            "prefix_multi_sustained",
            "prefix_eviction_pressure",
        }
        and scenario.wave_size > 0
    ):
        return tuple(
            (
                prompts[start : start + scenario.wave_size],
                sampling_params[start : start + scenario.wave_size],
            )
            for start in range(0, len(prompts), scenario.wave_size)
        )
    return ((prompts, sampling_params),)


def _run_one_repeat(
    *,
    llm,
    torch,
    scenario: Scenario,
    vocab_size: int,
    seed: int,
    token_id_range: int,
    nvtx_name: str | None = None,
) -> dict[str, Any]:
    prompts, sampling_params = build_workload(
        scenario,
        vocab_size=vocab_size,
        seed=seed,
        token_id_range=token_id_range,
    )
    target_output_tokens = int(sum(param.max_tokens for param in sampling_params))
    prompt_tokens = int(sum(len(prompt) for prompt in prompts))
    torch.cuda.synchronize(llm.device)
    torch.cuda.reset_peak_memory_stats(llm.device)
    _case_boundary_debug_snapshot(llm, f"{nvtx_name or scenario.name}:before_prefix_metrics_before")
    prefix_metrics_before = llm.cache_manager.prefix_metrics_snapshot()
    _case_boundary_debug_snapshot(llm, f"{nvtx_name or scenario.name}:after_prefix_metrics_before")
    nvtx_context = torch.cuda.nvtx.range(nvtx_name) if nvtx_name else contextlib.nullcontext()
    outputs = []
    trace: list[dict[str, Any]] = []
    request_metrics: list[dict[str, Any]] = []
    with nvtx_context:
        tic = time.perf_counter()
        generation_parts = _generation_parts(scenario, prompts, sampling_params)
        for part_prompts, part_sampling_params in generation_parts:
            if not part_prompts:
                continue
            part_outputs = llm.generate(part_prompts, part_sampling_params)
            _case_boundary_debug_snapshot(
                llm,
                f"{nvtx_name or scenario.name}:after_generate_part{len(outputs)}",
            )
            outputs.extend(part_outputs)
            trace.extend(llm.bench_batch_trace)
            request_metrics.extend(llm.request_metrics())
        torch.cuda.synchronize(llm.device)
        elapsed_s = time.perf_counter() - tic
    _case_boundary_debug_snapshot(llm, f"{nvtx_name or scenario.name}:before_prefix_metrics_after")
    prefix_metrics_after = llm.cache_manager.prefix_metrics_snapshot()
    _case_boundary_debug_snapshot(llm, f"{nvtx_name or scenario.name}:after_prefix_metrics_after")
    output_lens = [len(output["token_ids"]) for output in outputs]
    request_errors = [request for request in request_metrics if request.get("error")]
    if request_errors:
        raise RuntimeError(
            "offline benchmark request rejected by scheduler: "
            + "; ".join(str(request["error"]) for request in request_errors)
        )
    admitted_output_tokens = int(
        sum(int(request.get("admitted_output_len") or 0) for request in request_metrics)
    )
    actual_output_tokens = int(sum(output_lens))
    if actual_output_tokens != admitted_output_tokens:
        raise RuntimeError(
            "offline benchmark completion mismatch: "
            f"admitted_output_tokens={admitted_output_tokens}, "
            f"actual_output_tokens={actual_output_tokens}"
        )
    observed_max_position = max(
        (int(row.get("max_device_len", 0)) - 1 for row in trace), default=-1
    )
    effective_rope_cache_len = int(
        getattr(llm.engine, "effective_rope_cache_len", llm.engine.max_seq_len)
    )
    return {
        "elapsed_s": elapsed_s,
        "prompt_tokens": prompt_tokens,
        "target_output_tokens": target_output_tokens,
        "actual_output_tokens": actual_output_tokens,
        "admitted_output_tokens": admitted_output_tokens,
        "admitted_decode_lens": [
            int(request.get("admitted_output_len") or 0) for request in request_metrics
        ],
        "observed_max_position": observed_max_position,
        "observed_max_position_within_effective_engine": (
            observed_max_position < int(llm.engine.max_seq_len)
        ),
        "observed_max_position_within_rope_cache": (
            observed_max_position < effective_rope_cache_len
        ),
        "output_lens": output_lens,
        "sample_output_token_ids": [output["token_ids"][:16] for output in outputs[:2]],
        "all_output_token_ids": [output["token_ids"] for output in outputs],
        "requests": request_metrics,
        "schedule_trace": trace,
        "prefix_cache_metrics": prefix_metrics_after,
        "prefix_cache_metrics_delta": _prefix_metrics_delta(
            prefix_metrics_before, prefix_metrics_after
        ),
        "phase_totals": {
            "prefill_forward_s": _sum_phase(trace, "prefill", "forward_s"),
            "decode_forward_s": _sum_phase(trace, "decode", "forward_s"),
            "prefill_forward_enqueue_s": _sum_phase(trace, "prefill", "forward_enqueue_s"),
            "decode_forward_enqueue_s": _sum_phase(trace, "decode", "forward_enqueue_s"),
            "prefill_prepare_s": _sum_phase(trace, "prefill", "prepare_s"),
            "decode_prepare_s": _sum_phase(trace, "decode", "prepare_s"),
            "prefill_input_tokens": _sum_trace_int(trace, "prefill", "input_tokens"),
            "decode_tokens": _sum_trace_int(trace, "decode", "decode_tokens"),
        },
        "memory": _rank_memory_report(torch, llm),
    }


def _prefix_metrics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in after.items():
        old_value = before.get(key)
        if isinstance(value, bool) or isinstance(old_value, bool):
            continue
        if isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
            delta[key] = value - old_value
    return delta


def _run_warmups(
    *,
    llm,
    torch,
    scenario: Scenario,
    vocab_size: int,
    seed: int,
    token_id_range: int,
) -> dict[str, Any]:
    elapsed: list[float] = []
    for idx in range(scenario.warmup_repeats):
        prompts, sampling_params = build_workload(
            scenario,
            vocab_size=vocab_size,
            seed=seed + idx,
            token_id_range=token_id_range,
        )
        torch.cuda.synchronize(llm.device)
        with torch.cuda.nvtx.range(f"warmup:{scenario.name}:{idx}"):
            tic = time.perf_counter()
            for part_prompts, part_sampling_params in _generation_parts(
                scenario, prompts, sampling_params
            ):
                if part_prompts:
                    llm.generate(part_prompts, part_sampling_params)
            torch.cuda.synchronize(llm.device)
            elapsed.append(time.perf_counter() - tic)
        llm.sync_all_ranks()
    return {
        "repeats": scenario.warmup_repeats,
        "elapsed_s": elapsed,
        "total_elapsed_s": float(sum(elapsed)),
    }


def _counter_categories(calls: dict[str, int]) -> dict[str, int]:
    return {
        label: int(sum(calls.get(name, 0) for name in names))
        for label, names in BOTTLENECK_COUNTER_GROUPS.items()
    }


def _aggregate_communication_counters(
    rank_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_entries: dict[
        tuple[str, str, str, tuple[int, ...], tuple[int, ...]], dict[str, Any]
    ] = {}
    for payload in rank_payloads:
        for entry in payload.get("communication_counters", {}).get("entries", []):
            shape = tuple(int(dim) for dim in entry.get("shape", ()))
            output_shape = tuple(int(dim) for dim in entry.get("output_shape", shape))
            key = (
                str(entry.get("label", "unlabeled")),
                str(entry.get("op", "unknown")),
                str(entry.get("dtype", "unknown")),
                shape,
                output_shape,
            )
            aggregate = aggregate_entries.get(key)
            if aggregate is None:
                aggregate = {
                    "label": key[0],
                    "op": key[1],
                    "dtype": key[2],
                    "shape": shape,
                    "output_shape": output_shape,
                    "input_bytes": int(entry.get("input_bytes", 0)),
                    "output_bytes": int(entry.get("output_bytes", 0)),
                    "bytes": 0,
                    "count": 0,
                }
                aggregate_entries[key] = aggregate
            aggregate["bytes"] += int(entry.get("bytes", 0))
            aggregate["count"] += int(entry.get("count", 0))

    entries = []
    by_label: dict[str, dict[str, Any]] = {}
    by_op: dict[str, dict[str, Any]] = {}
    for entry in sorted(
        aggregate_entries.values(),
        key=lambda item: (item["label"], item["op"], item["dtype"], item["shape"]),
    ):
        serializable = dict(entry)
        serializable["shape"] = list(serializable["shape"])
        serializable["output_shape"] = list(serializable["output_shape"])
        entries.append(serializable)
        for target, key in ((by_label, serializable["label"]), (by_op, serializable["op"])):
            bucket = target.setdefault(key, {"count": 0, "bytes": 0})
            bucket["count"] += int(serializable["count"])
            bucket["bytes"] += int(serializable["bytes"])
    return {
        "total_count": int(sum(entry["count"] for entry in entries)),
        "total_bytes": int(sum(entry["bytes"] for entry in entries)),
        "entries": entries,
        "by_label": dict(sorted(by_label.items())),
        "by_op": dict(sorted(by_op.items())),
        "rank0": next(
            (
                payload.get("communication_counters", {})
                for payload in rank_payloads
                if payload.get("rank") == 0
            ),
            {},
        ),
    }


def _aggregate_owner_timing(rank_payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def empty_bucket() -> dict[str, Any]:
        return {
            "count": 0,
            "timed_count": 0,
            "captured_count": 0,
            "sum_rank_total_ms": 0.0,
            "max_rank_total_ms": 0.0,
            "sum_rank_captured_total_ms": 0.0,
            "max_rank_captured_total_ms": 0.0,
            "rank_totals_ms": {},
            "rank_captured_totals_ms": {},
            "metadata_examples": [],
        }

    def accumulate_stats(
        target: dict[str, dict[str, Any]],
        *,
        rank: int,
        label: str,
        stats: dict[str, Any],
    ) -> None:
        bucket = target.setdefault(label, empty_bucket())
        total_ms = float(stats.get("total_ms") or 0.0)
        captured_total_ms = float(stats.get("captured_total_ms") or 0.0)
        bucket["count"] = int(bucket["count"]) + int(stats.get("count") or 0)
        bucket["timed_count"] = int(bucket["timed_count"]) + int(stats.get("timed_count") or 0)
        bucket["captured_count"] = int(bucket["captured_count"]) + int(
            stats.get("captured_count") or 0
        )
        bucket["sum_rank_total_ms"] = float(bucket["sum_rank_total_ms"]) + total_ms
        bucket["max_rank_total_ms"] = max(float(bucket["max_rank_total_ms"]), total_ms)
        bucket["sum_rank_captured_total_ms"] = (
            float(bucket["sum_rank_captured_total_ms"]) + captured_total_ms
        )
        bucket["max_rank_captured_total_ms"] = max(
            float(bucket["max_rank_captured_total_ms"]),
            captured_total_ms,
        )
        bucket["rank_totals_ms"][str(rank)] = total_ms
        bucket["rank_captured_totals_ms"][str(rank)] = captured_total_ms
        examples = bucket["metadata_examples"]
        for example in stats.get("metadata_examples", []):
            if len(examples) >= 4:
                break
            if example not in examples:
                examples.append(example)

    def accumulate_section(section_name: str) -> dict[str, Any]:
        labels: dict[str, dict[str, Any]] = {}
        label_shapes: dict[str, dict[str, Any]] = {}
        for payload in rank_payloads:
            rank = int(payload.get("rank", -1))
            timing = payload.get("owner_timing", {})
            for source, target in (
                (timing.get(f"{section_name}_by_label", {}), labels),
                (timing.get(f"{section_name}_by_label_shape", {}), label_shapes),
            ):
                for label, stats in source.items():
                    accumulate_stats(target, rank=rank, label=label, stats=stats)
        return {
            "by_label": dict(sorted(labels.items())),
            "by_label_shape": dict(sorted(label_shapes.items())),
        }

    def accumulate_host() -> dict[str, Any]:
        labels: dict[str, dict[str, Any]] = {}
        for payload in rank_payloads:
            rank = int(payload.get("rank", -1))
            for label, stats in payload.get("owner_timing", {}).get("host_by_label", {}).items():
                accumulate_stats(labels, rank=rank, label=label, stats=stats)
        return {"by_label": dict(sorted(labels.items()))}

    rank0 = next(
        (payload.get("owner_timing", {}) for payload in rank_payloads if payload.get("rank") == 0),
        {},
    )
    enabled = any(payload.get("owner_timing", {}).get("enabled") for payload in rank_payloads)
    return {
        "enabled": bool(enabled),
        "cuda": accumulate_section("cuda"),
        "host": accumulate_host(),
        "rank0": rank0,
    }


def _label_bottlenecks(
    *,
    metrics: dict[str, Any],
    counters: dict[str, Any],
) -> list[dict[str, Any]]:
    calls = counters.get("fallback_wrapper_calls", {})
    categories = _counter_categories(calls)
    phase = metrics.get("phase_totals", {})
    elapsed = float(metrics.get("elapsed_s") or 0.0)
    prefill = float(phase.get("prefill_forward_s") or 0.0)
    decode = float(phase.get("decode_forward_s") or 0.0)
    prepare = float(phase.get("prefill_prepare_s") or 0.0) + float(
        phase.get("decode_prepare_s") or 0.0
    )
    scheduler_overhead = max(0.0, elapsed - prefill - decode - prepare)

    labels: list[dict[str, Any]] = []
    dominant_phase = "prefill" if prefill >= decode else "decode"
    labels.append(
        {
            "label": f"{dominant_phase} dominated",
            "evidence": {
                "prefill_forward_s": prefill,
                "decode_forward_s": decode,
                "elapsed_s": elapsed,
            },
        }
    )
    for label, count in categories.items():
        if count > 0:
            labels.append({"label": label, "evidence": {"wrapper_calls": count}})
    if elapsed > 0 and prepare / elapsed >= 0.05:
        labels.append(
            {
                "label": "metadata construction",
                "evidence": {"prepare_s": prepare, "fraction_of_elapsed": prepare / elapsed},
            }
        )
    if elapsed > 0 and scheduler_overhead / elapsed >= 0.10:
        labels.append(
            {
                "label": "scheduler overhead",
                "evidence": {
                    "estimated_overhead_s": scheduler_overhead,
                    "fraction_of_elapsed": scheduler_overhead / elapsed,
                },
            }
        )
    return labels


def _aggregate_case_report(
    *,
    base: dict[str, Any],
    rank_payloads: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rank0 = next(
        (payload for payload in rank_payloads if payload.get("rank") == 0), rank_payloads[0]
    )
    rank_elapsed = [sum(r["elapsed_s"] for r in payload["repeats"]) for payload in rank_payloads]
    elapsed_s = float(max(rank_elapsed) if rank_elapsed else 0.0)
    repeats0 = rank0["repeats"]
    prompt_tokens = int(sum(repeat["prompt_tokens"] for repeat in repeats0))
    actual_output_tokens = int(sum(repeat["actual_output_tokens"] for repeat in repeats0))
    target_output_tokens = int(sum(repeat["target_output_tokens"] for repeat in repeats0))
    all_requests = [request for repeat in repeats0 for request in repeat["requests"]]
    ttft_values = [
        request["ttft_s"] for request in all_requests if request.get("ttft_s") is not None
    ]
    topt_values = [
        request["topt_s"] for request in all_requests if request.get("topt_s") is not None
    ]
    latency_values = [
        request["latency_s"] for request in all_requests if request.get("latency_s") is not None
    ]
    phase_totals: dict[str, float | int] = {
        "prefill_forward_s": max(
            sum(repeat["phase_totals"]["prefill_forward_s"] for repeat in payload["repeats"])
            for payload in rank_payloads
        ),
        "decode_forward_s": max(
            sum(repeat["phase_totals"]["decode_forward_s"] for repeat in payload["repeats"])
            for payload in rank_payloads
        ),
        "prefill_forward_enqueue_s": max(
            sum(
                repeat["phase_totals"].get("prefill_forward_enqueue_s", 0.0)
                for repeat in payload["repeats"]
            )
            for payload in rank_payloads
        ),
        "decode_forward_enqueue_s": max(
            sum(
                repeat["phase_totals"].get("decode_forward_enqueue_s", 0.0)
                for repeat in payload["repeats"]
            )
            for payload in rank_payloads
        ),
        "prefill_prepare_s": max(
            sum(repeat["phase_totals"]["prefill_prepare_s"] for repeat in payload["repeats"])
            for payload in rank_payloads
        ),
        "decode_prepare_s": max(
            sum(repeat["phase_totals"]["decode_prepare_s"] for repeat in payload["repeats"])
            for payload in rank_payloads
        ),
        "prefill_input_tokens": int(
            sum(repeat["phase_totals"]["prefill_input_tokens"] for repeat in repeats0)
        ),
        "decode_tokens": int(sum(repeat["phase_totals"]["decode_tokens"] for repeat in repeats0)),
    }

    prefill_forward_s = float(phase_totals["prefill_forward_s"])
    decode_forward_s = float(phase_totals["decode_forward_s"])
    prepare_s = float(phase_totals["prefill_prepare_s"]) + float(phase_totals["decode_prepare_s"])
    scheduler_overhead_s = max(0.0, elapsed_s - prefill_forward_s - decode_forward_s - prepare_s)

    aggregate_counters: dict[str, int] = {}
    aggregate_none_skips: dict[str, int] = {}
    aggregate_unsupported: dict[str, int] = {}
    for payload in rank_payloads:
        counters = payload.get("kernel_counters", {})
        for name, count in counters.get("fallback_wrapper_calls", {}).items():
            aggregate_counters[name] = aggregate_counters.get(name, 0) + int(count)
        for name, count in counters.get("optional_kernel_none_skips", {}).items():
            aggregate_none_skips[name] = aggregate_none_skips.get(name, 0) + int(count)
        for name, count in counters.get("unsupported_kernel_skips", {}).items():
            aggregate_unsupported[name] = aggregate_unsupported.get(name, 0) + int(count)

    kernel_counters = {
        "fallback_wrapper_calls_total": int(sum(aggregate_counters.values())),
        "fallback_wrapper_calls": dict(sorted(aggregate_counters.items())),
        "optional_kernel_none_skips_total": int(sum(aggregate_none_skips.values())),
        "optional_kernel_none_skips": dict(sorted(aggregate_none_skips.items())),
        "unsupported_kernel_skips_total": int(sum(aggregate_unsupported.values())),
        "unsupported_kernel_skips": dict(sorted(aggregate_unsupported.items())),
        "rank0": rank0.get("kernel_counters", {}),
        "indexer_profile_per_rank": [
            {
                "rank": int(payload.get("rank", -1)),
                **payload.get("kernel_counters", {}).get("indexer_profile", {}),
            }
            for payload in rank_payloads
        ],
    }
    communication_counters = _aggregate_communication_counters(rank_payloads)
    owner_timing = _aggregate_owner_timing(rank_payloads)
    c128_prefill_one_surface = {
        "rank0": rank0.get("c128_prefill_one_surface", {}),
        "per_rank": [
            {
                "rank": int(payload.get("rank", -1)),
                **payload.get("c128_prefill_one_surface", {}),
            }
            for payload in rank_payloads
        ],
    }
    peak_allocated = max(
        repeat["memory"]["max_memory_allocated_bytes"]
        for payload in rank_payloads
        for repeat in payload["repeats"]
    )
    peak_reserved = max(
        repeat["memory"]["max_memory_reserved_bytes"]
        for payload in rank_payloads
        for repeat in payload["repeats"]
    )
    kv_cache_per_rank = [int(payload["kv_cache_memory_bytes"]) for payload in rank_payloads]
    prefix_delta_rank0: dict[str, float | int] = {}
    for repeat in repeats0:
        for key, value in repeat.get("prefix_cache_metrics_delta", {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                prefix_delta_rank0[key] = int(prefix_delta_rank0.get(key, 0)) + value
            elif isinstance(value, float):
                prefix_delta_rank0[key] = float(prefix_delta_rank0.get(key, 0.0)) + value
    repeat_summary: list[dict[str, Any]] = []
    for repeat_index, repeat0 in enumerate(repeats0):
        elapsed = max(
            float(payload["repeats"][repeat_index]["elapsed_s"])
            for payload in rank_payloads
        )
        prefill_s = max(
            float(payload["repeats"][repeat_index]["phase_totals"]["prefill_forward_s"])
            for payload in rank_payloads
        )
        decode_s = max(
            float(payload["repeats"][repeat_index]["phase_totals"]["decode_forward_s"])
            for payload in rank_payloads
        )
        prefill_tokens = int(repeat0["phase_totals"]["prefill_input_tokens"])
        decode_tokens = int(repeat0["phase_totals"]["decode_tokens"])
        output_tokens = int(repeat0["actual_output_tokens"])
        repeat_summary.append(
            {
                "repeat_index": repeat_index,
                "elapsed_s": elapsed,
                "output_tokens": output_tokens,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "prefill_tokens_per_s": None if prefill_s <= 0 else prefill_tokens / prefill_s,
                "decode_tokens_per_s": None if decode_s <= 0 else decode_tokens / decode_s,
                "end_to_end_output_tokens_per_s": None if elapsed <= 0 else output_tokens / elapsed,
            }
        )

    def stable_metric(name: str) -> dict[str, Any]:
        values = [float(row[name]) for row in repeat_summary if row[name] is not None]
        median = float(statistics.median(values)) if values else None
        return {
            "values": values,
            "min": min(values) if values else None,
            "median": median,
            "max": max(values) if values else None,
            "relative_span": (
                None
                if median in (None, 0.0)
                else (max(values) - min(values)) / median
            ),
        }

    metrics = {
        "elapsed_s": elapsed_s,
        "completed_requests": len(all_requests),
        "requests_per_s": None if elapsed_s <= 0 else len(all_requests) / elapsed_s,
        "prompt_tokens": prompt_tokens,
        "actual_output_tokens": actual_output_tokens,
        "target_output_tokens": target_output_tokens,
        "admitted_output_tokens": int(
            sum(repeat.get("admitted_output_tokens", 0) for repeat in repeats0)
        ),
        "observed_max_position": max(
            (int(repeat.get("observed_max_position", -1)) for repeat in repeats0),
            default=-1,
        ),
        "ttft_s_mean": _safe_mean(ttft_values),
        "ttft_s_median": _safe_median(ttft_values),
        "ttft_s_percentiles": _latency_percentiles(ttft_values),
        "topt_s_mean": _safe_mean(topt_values),
        "tpot_s_percentiles": _latency_percentiles(topt_values),
        "request_latency_s_mean": _safe_mean(latency_values),
        "request_latency_s_percentiles": _latency_percentiles(latency_values),
        "prefill_tokens_per_s": (
            None
            if prefill_forward_s <= 0
            else float(phase_totals["prefill_input_tokens"]) / prefill_forward_s
        ),
        "decode_tokens_per_s": (
            None
            if decode_forward_s <= 0
            else float(phase_totals["decode_tokens"]) / decode_forward_s
        ),
        "end_to_end_output_tokens_per_s": (
            None if elapsed_s <= 0 else actual_output_tokens / elapsed_s
        ),
        "end_to_end_total_tokens_per_s": (
            None if elapsed_s <= 0 else (prompt_tokens + actual_output_tokens) / elapsed_s
        ),
        "phase_totals": phase_totals,
        "scheduler_overhead_s": scheduler_overhead_s,
        "peak_gpu_memory_allocated_bytes": int(peak_allocated),
        "peak_gpu_memory_reserved_bytes": int(peak_reserved),
        "kv_cache_memory_bytes_per_rank_max": max(kv_cache_per_rank),
        "kv_cache_memory_bytes_per_rank": kv_cache_per_rank,
        "kv_cache_memory_bytes_total_tp": int(sum(kv_cache_per_rank)),
        "prefix_cache": {
            "rank0_final": rank0.get("prefix_cache_metrics", {}),
            "rank0_repeat_delta": prefix_delta_rank0,
        },
        "repeat_summary": repeat_summary,
        "repeat_stable_median": {
            "repeat_count": len(repeat_summary),
            "prefill_tokens_per_s": stable_metric("prefill_tokens_per_s"),
            "decode_tokens_per_s": stable_metric("decode_tokens_per_s"),
            "end_to_end_output_tokens_per_s": stable_metric(
                "end_to_end_output_tokens_per_s"
            ),
        },
    }
    graph_status_case = (
        base.get("config", {}).get("graph_runner_case")
        or base.get("config", {}).get("graph_runner", {})
        or {}
    )
    max_sequence = copy.deepcopy(base.get("config", {}).get("max_sequence", {}))
    max_sequence.update(
        {
            "prompt_len_requested": base.get("scenario", {}).get("max_input_len"),
            "decode_len_requested": base.get("scenario", {}).get("decode_len"),
            "admitted_decode_lens": [
                int(request.get("admitted_output_len") or 0) for request in all_requests
            ],
            "observed_max_position": metrics["observed_max_position"],
            "observed_max_position_within_effective_engine": all(
                bool(repeat.get("observed_max_position_within_effective_engine", False))
                for repeat in repeats0
            ),
            "observed_max_position_within_rope_cache": all(
                bool(repeat.get("observed_max_position_within_rope_cache", False))
                for repeat in repeats0
            ),
        }
    )
    report = {
        **base,
        "status": "pass",
        "max_sequence": max_sequence,
        "metrics": metrics,
        "schedule_summary": _schedule_summary(repeats0),
        "decode_m_time_distribution": _decode_m_time_distribution(repeats0),
        "bucket_coverage": _bucket_coverage_table(repeats0, graph_status_case),
        "kernel_counters": kernel_counters,
        "communication_counters": communication_counters,
        "owner_timing": owner_timing,
        "c128_prefill_one_surface": c128_prefill_one_surface,
        "bottlenecks": _label_bottlenecks(metrics=metrics, counters=kernel_counters),
        "requests": all_requests,
        "repeats": repeats0,
        "per_rank": list(rank_payloads),
    }
    return report


def _summary_row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    prefix_metrics = metrics.get("prefix_cache", {}).get("rank0_final", {})
    config = report.get("config", {})
    return {
        "status": report.get("status"),
        "classification": report.get("classification"),
        "variant": report.get("variant", {}).get("name"),
        "scenario": report.get("scenario", {}).get("name"),
        "report_path": report.get("report_path"),
        "max_sequence": report.get("max_sequence")
        or report.get("config", {}).get("max_sequence", {}),
        "elapsed_s": metrics.get("elapsed_s"),
        "requests_per_s": metrics.get("requests_per_s"),
        "ttft_s_mean": metrics.get("ttft_s_mean"),
        "prefill_tokens_per_s": metrics.get("prefill_tokens_per_s"),
        "decode_tokens_per_s": metrics.get("decode_tokens_per_s"),
        "end_to_end_output_tokens_per_s": metrics.get("end_to_end_output_tokens_per_s"),
        "repeat_stable_median": metrics.get("repeat_stable_median", {}),
        "prefix_hit_rate": prefix_metrics.get("hit_rate"),
        "prefix_saved_prefill_tokens": prefix_metrics.get("saved_prefill_tokens"),
        "prefix_retained_pages": prefix_metrics.get("retained_prefix_pages"),
        "prefix_evictions": prefix_metrics.get("evictions"),
        "peak_gpu_memory_allocated_bytes": metrics.get("peak_gpu_memory_allocated_bytes"),
        "kv_cache_memory_bytes_per_rank_max": metrics.get("kv_cache_memory_bytes_per_rank_max"),
        "fallback_wrapper_calls_total": report.get("kernel_counters", {}).get(
            "fallback_wrapper_calls_total"
        ),
        "unsupported_kernel_skips_total": report.get("kernel_counters", {}).get(
            "unsupported_kernel_skips_total"
        ),
        "communication_total_count": report.get("communication_counters", {}).get("total_count"),
        "communication_total_bytes": report.get("communication_counters", {}).get("total_bytes"),
        "communication_by_label": report.get("communication_counters", {}).get("by_label", {}),
        "c128_prefill_one_surface": report.get("c128_prefill_one_surface", {}).get("rank0", {}),
        "graph_runner": config.get("graph_runner_case") or config.get("graph_runner", {}),
        "graph_runner_cumulative": config.get("graph_runner", {}),
        "bucket_coverage": report.get("bucket_coverage", []),
        "schedule_summary": report.get("schedule_summary", {}),
        "decode_m_time_distribution": report.get("decode_m_time_distribution", {}),
        "bottleneck_labels": [row["label"] for row in report.get("bottlenecks", [])],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _gather_payloads(torch, llm, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size(group=llm.tp_cpu_group)
        gathered: list[Any] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, payload, group=llm.tp_cpu_group)
        return list(gathered)
    return [payload]


def run_case(
    *,
    args: argparse.Namespace,
    llm,
    torch,
    dsv4_kernel,
    tracer: KernelCallTracer,
    scenario: Scenario,
    variant: Variant,
    case_index: int,
    output_dir: Path,
    rank: int,
    tp_size: int,
    distributed_init_method: str | None,
    communication_backend: str,
    runtime_options: dict[str, Any],
    load_init: dict[str, Any],
    runtime_environment: dict[str, Any],
    git: dict[str, Any],
    dtype,
) -> dict[str, Any] | None:
    case_name = f"{case_index:03d}_{scenario.name}__{variant.name}"
    report_path = output_dir / "reports" / f"{case_name}.json"
    rank_path = output_dir / "reports" / f"{case_name}.rank{rank}.json"
    variant_env = runtime_variant_env(dsv4_kernel, variant)
    llm.sync_all_ranks()
    tracer.reset()
    warmup = None
    repeats = []
    error: dict[str, Any] | None = None
    graph_status_before = _snapshot_graph_status(llm)
    graph_status_after_warmup = graph_status_before
    graph_status_after = graph_status_before
    case_boundary_debug: list[dict[str, Any]] = []
    try:
        case_boundary_debug.append(_case_boundary_debug_snapshot(llm, f"{case_name}:before_warmup"))
        warmup = _run_warmups(
            llm=llm,
            torch=torch,
            scenario=scenario,
            vocab_size=llm.engine.sampler.vocab_size,
            seed=args.seed + 100000 + case_index * 1000,
            token_id_range=args.token_id_range,
        )
        llm.sync_all_ranks()
        graph_status_after_warmup = _snapshot_graph_status(llm)
        case_boundary_debug.append(_case_boundary_debug_snapshot(llm, f"{case_name}:after_warmup"))
        for repeat_idx in range(scenario.repeats):
            case_boundary_debug.append(
                _case_boundary_debug_snapshot(llm, f"{case_name}:before_repeat_{repeat_idx}")
            )
            repeat_payload = _run_one_repeat(
                llm=llm,
                torch=torch,
                scenario=scenario,
                vocab_size=llm.engine.sampler.vocab_size,
                seed=args.seed + case_index * 1000 + repeat_idx,
                token_id_range=args.token_id_range,
                nvtx_name=f"repeat:{scenario.name}:{repeat_idx}",
            )
            repeat_payload["repeat_index"] = repeat_idx
            repeats.append(repeat_payload)
            case_boundary_debug.append(
                _case_boundary_debug_snapshot(llm, f"{case_name}:after_repeat_{repeat_idx}")
            )
            llm.sync_all_ranks()
        graph_status_after = _snapshot_graph_status(llm)
        case_boundary_debug.append(_case_boundary_debug_snapshot(llm, f"{case_name}:after_case"))
    except BaseException as exc:
        graph_status_after = _snapshot_graph_status(llm)
        error = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
        try:
            case_boundary_debug.append(
                _case_boundary_debug_snapshot(llm, f"{case_name}:after_exception")
            )
        except BaseException as debug_exc:
            case_boundary_debug.append(
                {
                    "stage": f"{case_name}:after_exception",
                    "error": f"{type(debug_exc).__name__}: {debug_exc}",
                }
            )
    graph_status_case = _graph_status_delta(graph_status_after_warmup, graph_status_after)
    graph_status_case_with_warmup = _graph_status_delta(graph_status_before, graph_status_after)

    kv_cache_memory_bytes = _estimate_kv_cache_bytes_from_config(
        llm,
        page_size=args.page_size,
        dtype=dtype,
        tp_size=tp_size,
    )
    replayed_padded_sizes = [
        int(size)
        for size, count in graph_status_case.get("replay_count_by_padded_size", {}).items()
        if int(count) > 0
    ]
    if error is None:
        case_boundary_debug.append(
            _case_boundary_debug_snapshot(llm, f"{case_name}:before_final_prefix_metrics")
        )
        final_prefix_cache_metrics = llm.cache_manager.prefix_metrics_snapshot()
        case_boundary_debug.append(
            _case_boundary_debug_snapshot(llm, f"{case_name}:after_final_prefix_metrics")
        )
    else:
        final_prefix_cache_metrics = {
            "error": "skipped_after_case_error",
            "case_error_type": error.get("exception_type"),
            "case_error_message": error.get("exception_message"),
        }
    rank_payload = {
        "rank": rank,
        "is_primary_rank": rank == 0,
        "warmup": warmup,
        "repeats": repeats,
        "kernel_counters": tracer.snapshot(),
        # The misc02 release runtime intentionally removed the historical
        # communication debug counter module.  Keep the benchmark payload
        # schema stable without restoring a production hot-path probe.
        "communication_counters": {},
        "c128_prefill_one_surface": _snapshot_c128_prefill_one_surface(llm),
        "graph_runner_before_case": graph_status_before,
        "graph_runner_after_warmup": graph_status_after_warmup,
        "graph_runner_after_case": graph_status_after,
        "graph_runner_case": graph_status_case,
        "graph_runner_case_with_warmup": graph_status_case_with_warmup,
        "case_boundary_debug": case_boundary_debug,
        "prefix_cache_metrics": final_prefix_cache_metrics,
        "memory_after_case": _rank_memory_report(torch, llm),
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "runtime_environment": runtime_environment,
        "max_sequence": next(
            (
                row
                for row in load_init.get("max_sequence_per_rank", [])
                if int(row.get("rank", -1)) == rank
            ),
            load_init.get("max_sequence_rank0", {}),
        ),
        "error": error,
    }
    if args.retain_full_rank_payloads:
        _write_json(rank_path, rank_payload)
    gathered = _gather_payloads(torch, llm, rank_payload)

    if rank != 0:
        return None

    base = {
        "case_name": case_name,
        "report_path": str(report_path),
        "model_path": args.model_path,
        "git": git,
        "variant": {
            "name": variant.name,
            "description": variant.description,
            "env": variant.env,
            **variant_env,
        },
        "scenario": {
            **asdict(scenario),
            "max_input_len": scenario.max_input_len,
            "scenario_required_total_seq_len": scenario.max_seq_len,
            "scheduler_supports_interleaved_arrivals": False,
            "radix_prefix_enabled": runtime_options["enable_dsv4_radix_prefix_cache"],
            "swa_tail_retention_v1_requested": bool(args.enable_dsv4_swa_tail_retention_v1),
            "component_loc_ownership_enabled": runtime_options[
                "enable_dsv4_component_loc_ownership"
            ],
            "swa_independent_lifecycle_enabled": runtime_options[
                "enable_dsv4_swa_independent_lifecycle"
            ],
        },
        "classification": run_classification(
            tp_size=tp_size,
            page_size=args.page_size,
            smoke=args.smoke,
        ),
        "max_sequence": load_init.get("max_sequence_rank0", {}),
        "config": {
            "tensor_parallel_size": tp_size,
            "rank_count": tp_size,
            "distributed_init_method": distributed_init_method,
            "communication_backend": communication_backend,
            "use_pynccl": runtime_options["use_pynccl"],
            "allow_dsv4_cuda_graph": runtime_options["allow_dsv4_cuda_graph"],
            "cuda_graph_bs": runtime_options["cuda_graph_bs"],
            "cuda_graph_max_bs": runtime_options["cuda_graph_max_bs"],
            "cuda_graph_bucket_policy": getattr(
                llm.engine, "cuda_graph_policy", None
            ).to_report(),
            "cuda_graph_capture_greedy_sample": runtime_options["cuda_graph_capture_greedy_sample"],
            "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}),
            "graph_runner_case": graph_status_case,
            "graph_runner_case_with_warmup": graph_status_case_with_warmup,
            "page_size": args.page_size,
            "num_pages": args.num_pages,
            "memory_ratio": args.memory_ratio,
            "dtype": args.dtype,
            "max_seq_len": args.max_seq_len,
            "max_sequence": load_init.get("max_sequence_rank0", {}),
            "max_extend_tokens": (
                int(llm.prefill_budget) if llm.prefill_budget is not None else None
            ),
            "requested_max_extend_tokens": args.max_extend_tokens,
            "use_serving_max_extend_tokens": args.use_serving_max_extend_tokens,
            "max_running_req": load_init.get("max_running_req_rank0", {}),
            "token_id_range": args.token_id_range,
            "radix_prefix_enabled": runtime_options["enable_dsv4_radix_prefix_cache"],
            "enable_dsv4_swa_tail_retention_v1": args.enable_dsv4_swa_tail_retention_v1,
            "enable_dsv4_component_loc_ownership": runtime_options[
                "enable_dsv4_component_loc_ownership"
            ],
            "enable_dsv4_swa_independent_lifecycle": (
                runtime_options["enable_dsv4_swa_independent_lifecycle"]
            ),
            "prefix_cache_metrics": final_prefix_cache_metrics,
            "model_prepare_report_rank0": getattr(llm.engine, "model_prepare_report", {}),
            "kv_capacity_plan_report": getattr(llm.engine, "kv_capacity_plan_report", {}),
        },
        "load_init": load_init,
        "runtime_environment_rank0": runtime_environment,
    }
    errors = [payload.get("error") for payload in gathered if payload.get("error")]
    if errors:
        report = {
            **base,
            "status": "fail",
            "errors": errors,
            "per_rank": gathered,
        }
    else:
        report = _aggregate_case_report(base=base, rank_payloads=gathered)
        if not args.retain_full_rank_payloads:
            report.pop("requests", None)
            report.pop("repeats", None)
            report.pop("per_rank", None)
            report["full_rank_payloads_retained"] = False
    _write_json(report_path, report)
    _append_jsonl(output_dir / "matrix.jsonl", _summary_row(report))
    return report


def _init_llm(
    *,
    args: argparse.Namespace,
    scenarios: Sequence[Scenario],
    rank: int,
    tp_size: int,
    distributed_init_method: str | None,
    runtime_options: dict[str, Any],
):
    import torch
    from minisgl.distributed import DistributedInfo

    BenchmarkLLM = make_benchmark_llm_class()
    dtype = _dtype_from_name(args.dtype)
    max_extend_tokens = args.max_extend_tokens
    if max_extend_tokens is None and not args.use_serving_max_extend_tokens:
        max_extend_tokens = _max_extend_tokens(scenarios)
    kwargs: dict[str, Any] = {}
    if distributed_init_method is not None:
        kwargs["distributed_init_method"] = distributed_init_method
    if max_extend_tokens is not None:
        kwargs["max_extend_tokens"] = max_extend_tokens
        kwargs["max_extend_tokens_explicit"] = True
    kwargs.update(_max_sequence_llm_kwargs(args, scenarios))
    kwargs.update(_max_running_req_llm_kwargs(args, scenarios))
    tic = time.perf_counter()
    llm = BenchmarkLLM(
        args.model_path,
        dtype=dtype,
        tp_info=DistributedInfo(rank, tp_size),
        dsv4_runtime_mode="optimized",
        dsv4_sm80_recipe=args.dsv4_sm80_recipe,
        num_page_override=args.num_pages,
        page_size=args.page_size,
        memory_ratio=args.memory_ratio,
        use_pynccl=runtime_options["use_pynccl"],
        allow_dsv4_cuda_graph=runtime_options["allow_dsv4_cuda_graph"],
        cuda_graph_bs=runtime_options["cuda_graph_bs"],
        cuda_graph_max_bs=runtime_options["cuda_graph_max_bs"],
        cuda_graph_capture_greedy_sample=runtime_options["cuda_graph_capture_greedy_sample"],
        enable_dsv4_radix_prefix_cache=runtime_options["enable_dsv4_radix_prefix_cache"],
        enable_dsv4_swa_tail_retention_v1=args.enable_dsv4_swa_tail_retention_v1,
        enable_dsv4_component_loc_ownership=runtime_options[
            "enable_dsv4_component_loc_ownership"
        ],
        enable_dsv4_swa_independent_lifecycle=runtime_options[
            "enable_dsv4_swa_independent_lifecycle"
        ],
        **kwargs,
    )
    llm.bench_admit_all_at_once = bool(args.admit_all_at_once)
    torch.cuda.synchronize(llm.device)
    load_init_s = time.perf_counter() - tic
    max_sequence = _max_sequence_runtime_report(llm, args=args, scenarios=scenarios)
    max_running_req_mode = _max_running_req_mode_config(args, scenarios)
    max_running_req = {
        "mode": max_running_req_mode.mode,
        "requested_override": max_running_req_mode.requested_override,
        "scenario_required_max_running_req": (
            max_running_req_mode.scenario_required_max_running_req
        ),
        # The Scheduler intentionally does not retain the config object. The
        # engine request table has one extra dummy row, so its materialized
        # row count is the authoritative effective request-slot capacity.
        "effective": int(llm.engine.page_table.shape[0] - 1),
        "request_table_shape": list(llm.engine.page_table.shape),
        "request_table_dtype": str(llm.engine.page_table.dtype).removeprefix("torch."),
        "request_table_bytes": int(
            llm.engine.page_table.numel() * llm.engine.page_table.element_size()
        ),
    }
    return (
        llm,
        torch,
        dtype,
        {
            "seconds": load_init_s,
            "rank": rank,
            "memory": _rank_memory_report(torch, llm),
            "model_prepare_report": getattr(llm.engine, "model_prepare_report", {}),
            "max_sequence": max_sequence,
            "max_running_req": max_running_req,
        },
    )


def run_matrix(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args)
    variants = _select_variants(args)
    runtime_options = _runtime_options(args, variants)
    rank, tp_size, env_world_size = _tp_rank_size(args)
    if env_world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={env_world_size} does not match tensor parallel size {tp_size}; "
            "launch TARGET 06 with torchrun --standalone --nproc_per_node=8."
        )
    distributed_init_method = _distributed_init_method(args, tp_size)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = output_dir / "matrix.jsonl"
        if matrix_path.exists():
            matrix_path.unlink()

    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    init_variant = (
        _graph_init_variant(variants)
        if runtime_options["allow_dsv4_cuda_graph"]
        else variants[0]
    )
    configure_variant(dsv4_kernel, init_variant)
    tracer = KernelCallTracer(dsv4_kernel)
    tracer.install()
    llm = None
    reports: list[dict[str, Any]] = []
    failures = 0
    try:
        llm, torch, dtype, local_load_init = _init_llm(
            args=args,
            scenarios=scenarios,
            rank=rank,
            tp_size=tp_size,
            distributed_init_method=distributed_init_method,
            runtime_options=runtime_options,
        )
        _validate_selected_scenarios(local_load_init["max_sequence"], scenarios)
        _validate_max_running_req(local_load_init["max_running_req"], scenarios)
        gathered_load_init = _gather_payloads(torch, llm, local_load_init)
        runtime_environment = collect_runtime_environment(torch, dsv4_kernel, rank=rank)
        communication_backend = (
            torch.distributed.get_backend()
            if torch.distributed.is_initialized()
            else "single_process"
        )
        load_init = {
            "seconds_max": max(float(payload["seconds"]) for payload in gathered_load_init),
            "seconds_per_rank": gathered_load_init,
            "max_sequence_rank0": gathered_load_init[0]["max_sequence"],
            "max_sequence_per_rank": [
                {
                    "rank": int(payload["rank"]),
                    **payload["max_sequence"],
                }
                for payload in gathered_load_init
            ],
            "max_running_req_rank0": gathered_load_init[0]["max_running_req"],
            "max_running_req_per_rank": [
                {
                    "rank": int(payload["rank"]),
                    **payload["max_running_req"],
                }
                for payload in gathered_load_init
            ],
        }
        git = git_info()
        if rank == 0:
            _write_json(
                output_dir / "run_config.json",
                {
                    "model_path": args.model_path,
                    "git": git,
                    "variants": [asdict(variant) for variant in variants],
                    "scenarios": [asdict(scenario) for scenario in scenarios],
                    "config": {
                        "tensor_parallel_size": tp_size,
                        "distributed_init_method": distributed_init_method,
                        "communication_backend": communication_backend,
                        "use_pynccl": runtime_options["use_pynccl"],
                        "allow_dsv4_cuda_graph": runtime_options["allow_dsv4_cuda_graph"],
                        "cuda_graph_bs": runtime_options["cuda_graph_bs"],
                        "cuda_graph_max_bs": runtime_options["cuda_graph_max_bs"],
                        "cuda_graph_bucket_policy": llm.engine.cuda_graph_policy.to_report(),
                        "dsv4_sm80_recipe": getattr(
                            llm.engine, "kv_capacity_plan_report", {}
                        ).get("dsv4_sm80_recipe"),
                        "cuda_graph_capture_greedy_sample": runtime_options[
                            "cuda_graph_capture_greedy_sample"
                        ],
                        "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}),
                        "page_size": args.page_size,
                        "num_pages": args.num_pages,
                        "max_sequence": load_init["max_sequence_rank0"],
                        "max_running_req": load_init["max_running_req_rank0"],
                        "max_extend_tokens": (
                            int(llm.prefill_budget) if llm.prefill_budget is not None else None
                        ),
                        "requested_max_extend_tokens": args.max_extend_tokens,
                        "use_serving_max_extend_tokens": args.use_serving_max_extend_tokens,
                        "enable_dsv4_radix_prefix_cache": runtime_options[
                            "enable_dsv4_radix_prefix_cache"
                        ],
                        "enable_dsv4_swa_tail_retention_v1": (
                            args.enable_dsv4_swa_tail_retention_v1
                        ),
                        "enable_dsv4_component_loc_ownership": (
                            runtime_options["enable_dsv4_component_loc_ownership"]
                        ),
                        "enable_dsv4_swa_independent_lifecycle": (
                            runtime_options["enable_dsv4_swa_independent_lifecycle"]
                        ),
                        "prefix_cache_metrics": llm.cache_manager.prefix_metrics_snapshot(),
                        "classification": run_classification(
                            tp_size=tp_size,
                            page_size=args.page_size,
                            smoke=args.smoke,
                        ),
                        "token_id_range": args.token_id_range,
                        "model_prepare_report_rank0": getattr(
                            llm.engine, "model_prepare_report", {}
                        ),
                        "kv_capacity_plan_report": getattr(
                            llm.engine, "kv_capacity_plan_report", {}
                        ),
                    },
                    "load_init": load_init,
                    "runtime_environment_rank0": runtime_environment,
                },
            )

        case_index = 0
        for scenario in scenarios:
            for variant in variants:
                report = run_case(
                    args=args,
                    llm=llm,
                    torch=torch,
                    dsv4_kernel=dsv4_kernel,
                    tracer=tracer,
                    scenario=scenario,
                    variant=variant,
                    case_index=case_index,
                    output_dir=output_dir,
                    rank=rank,
                    tp_size=tp_size,
                    distributed_init_method=distributed_init_method,
                    communication_backend=communication_backend,
                    runtime_options=runtime_options,
                    load_init=load_init,
                    runtime_environment=runtime_environment,
                    git=git,
                    dtype=dtype,
                )
                case_index += 1
                if rank == 0 and report is not None:
                    reports.append(report)
                    if report.get("status") != "pass":
                        failures += 1
                        if not args.keep_going:
                            raise RuntimeError(f"case failed: {report.get('case_name')}")
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException:
                if rank == 0:
                    traceback.print_exc()

    if rank == 0:
        summary = [_summary_row(report) for report in reports]
        _write_json(output_dir / "summary.json", summary)
        print(
            json.dumps(
                {"summary_path": str(output_dir / "summary.json"), "cases": summary}, indent=2
            )
        )
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Torchrun-native DeepSeek V4 TP8 sm80 baseline benchmark matrix."
    )
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument(
        "--dsv4-sm80-recipe",
        choices=(
            "dsv4_sm80_low_m64",
            "dsv4_sm80_mid_m128",
            "dsv4_sm80_balanced",
            "dsv4_sm80_long_context_512k",
            "dsv4_sm80_1m_smoke",
        ),
        default=None,
        help="Select a public DSV4 A100/sm80 recipe; omitted uses the no-env default.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=(DSV4_RELEASE_DEFAULT_VARIANT,),
        help="Compatibility label for the canonical optimized runtime only.",
    )
    parser.add_argument("--scenarios", nargs="*", choices=tuple(_scenario_map()))
    parser.add_argument("--output-dir", default="/tmp/dsv4_sm80_target06_tp8")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=None)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    max_seq_group = parser.add_mutually_exclusive_group()
    max_seq_group.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help=(
            "Explicit non-default maximum total sequence length override. Omit this "
            "option for the model/serving default."
        ),
    )
    max_seq_group.add_argument(
        "--scenario-sized-max-seq-len",
        action="store_true",
        help=(
            "Diagnostic only: size the engine to the largest selected prompt plus "
            "decode length. This is not a serving/release-default result."
        ),
    )
    parser.add_argument("--max-extend-tokens", type=int, default=None)
    parser.add_argument(
        "--use-serving-max-extend-tokens",
        action="store_true",
        help=(
            "Leave max_extend_tokens at the serving default instead of expanding it to "
            "the largest selected scenario input."
        ),
    )
    max_running_req_group = parser.add_mutually_exclusive_group()
    max_running_req_group.add_argument(
        "--max-running-req",
        type=int,
        default=None,
        help=(
            "Explicit non-default maximum concurrent request-slot override. Omit "
            "this option for the serving/model default."
        ),
    )
    max_running_req_group.add_argument(
        "--scenario-sized-max-running-req",
        action="store_true",
        help=(
            "Diagnostic only: size request slots to the largest selected scenario "
            "batch. This is not a serving/release-default result."
        ),
    )
    parser.add_argument(
        "--use-pynccl",
        action="store_true",
        help="Use the PyNCCL communicator for tensor-parallel collectives.",
    )
    parser.add_argument(
        "--allow-dsv4-cuda-graph",
        action="store_true",
        help="Opt in to DeepSeek V4 decode CUDA graph capture. Defaults to sizes 1,2,4,8,16.",
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
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help="Generate the authoritative CUDA graph bucket policy through this maximum.",
    )
    parser.add_argument(
        "--cuda-graph-capture-greedy-sample",
        dest="cuda_graph_capture_greedy_sample",
        action="store_true",
        default=None,
        help="Force graph capture of greedy argmax sampling.",
    )
    parser.add_argument(
        "--no-cuda-graph-capture-greedy-sample",
        dest="cuda_graph_capture_greedy_sample",
        action="store_false",
        help="Force greedy sampling outside the captured CUDA graph.",
    )
    parser.set_defaults(cuda_graph_capture_greedy_sample=None)
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--decode-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup-repeats", type=int, default=None)
    parser.add_argument(
        "--token-id-range",
        type=int,
        default=1024,
        help="Generate synthetic prompt token ids in [10, token-id-range] by default.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--retain-full-rank-payloads",
        action="store_true",
        help=(
            "Opt in to very large per-rank request/token/schedule payload files. "
            "Compact aggregate telemetry is the release-benchmark default."
        ),
    )
    parser.add_argument(
        "--admit-all-at-once",
        action="store_true",
        help=(
            "Benchmark contract: submit every selected request to the scheduler in "
            "one receive cycle while retaining the configured 8192-token chunk budget."
        ),
    )
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    args = parser.parse_args(argv)
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    for name in ("prompt_len", "decode_len", "batch_size", "repeats"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_repeats is not None and args.warmup_repeats < 0:
        parser.error("--warmup-repeats must be non-negative")
    if args.memory_ratio <= 0:
        parser.error("--memory-ratio must be positive")
    if args.max_seq_len is not None and args.max_seq_len <= 0:
        parser.error("--max-seq-len must be positive")
    if args.max_running_req is not None and args.max_running_req <= 0:
        parser.error("--max-running-req must be positive")
    if args.token_id_range <= 0:
        parser.error("--token-id-range must be positive")
    if args.num_pages == 0:
        args.num_pages = None
    elif args.num_pages is not None and args.num_pages <= 1:
        parser.error("--num-pages must be greater than 1, or 0 for automatic planning")
    if args.cuda_graph_bs is not None:
        if any(value <= 0 for value in args.cuda_graph_bs):
            parser.error("--cuda-graph-bs values must be positive")
        args.cuda_graph_bs = sorted(set(args.cuda_graph_bs))
    if args.cuda_graph_max_bs is not None and args.cuda_graph_max_bs < 0:
        parser.error("--cuda-graph-max-bs must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_scenarios:
        for scenario in ALL_SCENARIOS:
            print(f"{scenario.name}\t{scenario.kind}\t{scenario.description}")
        return
    if args.list_variants:
        for variant in ALL_VARIANTS:
            print(f"{variant.name}\t{variant.description}")
        return
    raise SystemExit(run_matrix(args))


if __name__ == "__main__":
    main()

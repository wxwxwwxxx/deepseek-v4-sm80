from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark" / "offline" / "deepseek_v4_perf_matrix.py"


@pytest.fixture(autouse=True)
def _restore_dsv4_sm80_env():
    original = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("MINISGL_DSV4_")
    }
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_"):
            os.environ.pop(name, None)
    yield
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_"):
            os.environ.pop(name, None)
    os.environ.update(original)


def _load_module():
    spec = importlib.util.spec_from_file_location("deepseek_v4_perf_matrix", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target06_defaults_are_tp8_page256_baseline_policy():
    bench = _load_module()

    args = bench.parse_args([])
    scenarios = bench._select_scenarios(args)
    variants = bench._select_variants(args)

    assert args.page_size == 256
    assert args.enable_dsv4_radix_prefix_cache is False
    assert args.enable_dsv4_swa_tail_retention_v1 is False
    assert args.enable_dsv4_component_loc_ownership is False
    assert [variant.name for variant in variants] == ["fallback", "v0_bf16", "v1_moe"]
    assert {scenario.name for scenario in scenarios} >= {
        "long_prefill_bs1",
        "batch_prefill_bs8",
        "decode_throughput_bs8",
        "shared_prompt_no_radix_bs8",
        "shared_prompt_reuse_bs8",
    }
    assert bench.run_classification(tp_size=8, page_size=256, smoke=False) == "baseline"

    enabled_args = bench.parse_args(["--enable-dsv4-radix-prefix-cache"])
    assert enabled_args.enable_dsv4_radix_prefix_cache is True

    retention_args = bench.parse_args(["--enable-dsv4-swa-tail-retention-v1"])
    assert retention_args.enable_dsv4_swa_tail_retention_v1 is True

    ownership_args = bench.parse_args(["--enable-dsv4-component-loc-ownership"])
    assert ownership_args.enable_dsv4_component_loc_ownership is True


def test_smoke_or_page_size_one_is_not_reported_as_baseline():
    bench = _load_module()

    args = bench.parse_args(["--smoke", "--page-size", "1"])
    scenarios = bench._select_scenarios(args)

    assert [scenario.name for scenario in scenarios] == ["smoke_debug"]
    assert bench.run_classification(tp_size=8, page_size=1, smoke=True) == "smoke_debug"
    assert bench.run_classification(tp_size=1, page_size=256, smoke=False) == "smoke_debug"


def test_target0810_prefix_scenarios_have_stable_prompt_shapes():
    bench = _load_module()

    names = {
        "prefix_full_hit_257_bs4",
        "prefix_full_hit_512_bs4",
        "prefix_full_hit_513_bs4",
        "prefix_full_hit_768_bs4",
        "prefix_full_hit_769_bs4",
        "prefix_full_hit_513_longout_bs4",
        "prefix_partial_hit_769_bs8",
        "prefix_mixed_hit_miss_bs16",
        "prefix_multi_112req_wave16",
        "prefix_eviction_pressure_96req_wave16",
    }
    assert names <= set(bench._scenario_map())

    partial = bench._scenario_map()["prefix_partial_hit_769_bs8"]
    prompts, params = bench.build_workload(
        partial,
        vocab_size=4096,
        seed=123,
        token_id_range=1024,
    )
    parts = bench._generation_parts(partial, prompts, params)
    assert [len(part[0]) for part in parts] == [1, 7]
    assert len(prompts[0]) == 257
    assert all(len(prompt) == 769 for prompt in prompts[1:])
    assert all(prompt[:257] == prompts[0] for prompt in prompts[1:])

    sustained = bench._scenario_map()["prefix_multi_112req_wave16"]
    prompts, params = bench.build_workload(
        sustained,
        vocab_size=4096,
        seed=123,
        token_id_range=1024,
    )
    parts = bench._generation_parts(sustained, prompts, params)
    assert len(prompts) == 112
    assert all(len(part[0]) == 16 for part in parts)
    assert prompts[0][:512] == prompts[8][:512]
    assert prompts[0][:512] == prompts[16][:512]

    exact_512 = bench._scenario_map()["prefix_full_hit_512_bs4"]
    prompts, params = bench.build_workload(
        exact_512,
        vocab_size=4096,
        seed=123,
        token_id_range=1024,
    )
    parts = bench._generation_parts(exact_512, prompts, params)
    assert [len(part[0]) for part in parts] == [1, 3]
    assert all(len(prompt) == 512 for prompt in prompts)
    assert all(prompt == prompts[0] for prompt in prompts)

    neighbor_769 = bench._scenario_map()["prefix_full_hit_769_bs4"]
    prompts, params = bench.build_workload(
        neighbor_769,
        vocab_size=4096,
        seed=123,
        token_id_range=1024,
    )
    assert all(len(prompt) == 769 for prompt in prompts)
    assert all(param.max_tokens == 4 for param in params)

    longout = bench._scenario_map()["prefix_full_hit_513_longout_bs4"]
    prompts, params = bench.build_workload(
        longout,
        vocab_size=4096,
        seed=123,
        token_id_range=1024,
    )
    assert all(len(prompt) == 513 for prompt in prompts)
    assert all(param.max_tokens == 32 for param in params)


def test_configure_variant_clears_existing_sm80_env_and_sets_v0(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V0_BF16",
            "MINISGL_DSV4_SM80_SWIGLU",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            return (
                name == "MINISGL_DSV4_SM80_SWIGLU"
                and os.environ.get("MINISGL_DSV4_SM80_V0_BF16") == "1"
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_SWIGLU", "1")
    monkeypatch.setenv("MINISGL_DSV4_SM80_FP8_GEMM", "1")

    result = bench.configure_variant(FakeKernel, bench._variant_map()["v0_bf16"])

    assert "MINISGL_DSV4_SM80_FP8_GEMM" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"] == {"MINISGL_DSV4_SM80_V0_BF16": "1"}
    assert result["active_dsv4_toggles"] == [
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V0_BF16",
    ]


def test_configure_variant_sets_v1_moe_without_int8_or_linear_experiment(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
            "MINISGL_DSV4_SM80_LINEAR_BF16_FP32",
            "MINISGL_DSV4_SM80_MOE_INT8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"}
                and os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_INT8", "1")
    result = bench.configure_variant(FakeKernel, bench._variant_map()["v1_moe"])

    assert "MINISGL_DSV4_SM80_MOE_INT8" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"] == {"MINISGL_DSV4_SM80_V1_MOE": "1"}
    assert result["active_dsv4_toggles"] == [
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V1_MOE",
    ]


def test_configure_variant_sets_moe_v2_without_int8_or_precision_lane(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
            "MINISGL_DSV4_SM80_LINEAR_BF16_FP32",
            "MINISGL_DSV4_SM80_MOE_INT8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1" or (
                os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_INT8", "1")
    result = bench.configure_variant(FakeKernel, bench._variant_map()["v1_moe_v2"])

    assert "MINISGL_DSV4_SM80_MOE_INT8" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"] == {
        "MINISGL_DSV4_SM80_MOE_V2": "1",
        "MINISGL_DSV4_SM80_V1_MOE": "1",
    }
    assert result["active_dsv4_toggles"] == [
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_MOE_V2",
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V1_MOE",
    ]


def test_configure_variant_sets_vllm_runner_without_precision_lane(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
            "MINISGL_DSV4_SM80_MOE_INT8",
            "MINISGL_DSV4_SM80_KV_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = (
                os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_VLLM_RUNNER") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_INT8", "1")
    monkeypatch.setenv("MINISGL_DSV4_SM80_KV_FP8", "1")
    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()[
            "v1_moe_vllm_runner_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ],
    )

    assert "MINISGL_DSV4_SM80_MOE_INT8" in result["cleared_dsv4_sm80_env"]
    assert "MINISGL_DSV4_SM80_KV_FP8" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_VLLM_RUNNER"] == "1"
    assert "MINISGL_DSV4_SM80_MOE_INT8" not in result["raw_dsv4_sm80_env"]
    assert "MINISGL_DSV4_SM80_KV_FP8" not in result["raw_dsv4_sm80_env"]
    active = set(result["active_dsv4_toggles"])
    assert {
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_MOE_V2",
        "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V1_MOE",
    } <= active
    assert "MINISGL_DSV4_SM80_MOE_INT8" not in active
    assert "MINISGL_DSV4_SM80_KV_FP8" not in active


def test_configure_variant_records_marlin_candidate_backend(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = (
                os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_VLLM_RUNNER") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()[
            "v1_moe_vllm_runner_marlin_mxfp4_w4a16_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_mxfp4_w4a16"
    )
    active = set(result["active_dsv4_toggles"])
    assert "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND" not in active
    assert {
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_MOE_V2",
        "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V1_MOE",
    } <= active


def test_configure_variant_records_vllm_marlin_bridge_backend(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = (
                os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_VLLM_RUNNER") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()[
            "v1_moe_vllm_runner_vllm_marlin_bridge_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "vllm_marlin_bridge"
    )
    assert "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND" not in result["active_dsv4_toggles"]


def test_configure_variant_records_marlin_wna16_backend(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = (
                os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_VLLM_RUNNER") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND" not in result["active_dsv4_toggles"]


def test_configure_variant_records_marlin_wna16_globaltopk(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_KV_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if name == "MINISGL_DSV4_SM80_KV_FP8":
                return False
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS"] == "1"
    assert "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_KV_FP8" not in result["raw_dsv4_sm80_env"]


def test_configure_variant_records_marlin_wna16_globaltopk_splitk(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16",
            "MINISGL_DSV4_SM80_KV_FP8",
            "MINISGL_DSV4_SM80_INDEXER_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if name in {"MINISGL_DSV4_SM80_KV_FP8", "MINISGL_DSV4_SM80_INDEXER_FP8"}:
                return False
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16"] == "1"
    assert "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_KV_FP8" not in result["raw_dsv4_sm80_env"]
    assert "MINISGL_DSV4_SM80_INDEXER_FP8" not in result["raw_dsv4_sm80_env"]


def test_configure_variant_records_marlin_wna16_indexer_fp8_cache(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16",
            "MINISGL_DSV4_SM80_REPLAY_METADATA_COPY",
            "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE",
            "MINISGL_DSV4_SM80_KV_FP8",
            "MINISGL_DSV4_SM80_INDEXER_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if name in {"MINISGL_DSV4_SM80_KV_FP8", "MINISGL_DSV4_SM80_INDEXER_FP8"}:
                return False
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_INDEXER_FP8_CACHE"] == "1"
    assert "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_KV_FP8" not in result["raw_dsv4_sm80_env"]
    assert "MINISGL_DSV4_SM80_INDEXER_FP8" not in result["raw_dsv4_sm80_env"]


def test_configure_variant_records_wo_a_bf16_bmm_cache(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16",
            "MINISGL_DSV4_SM80_REPLAY_METADATA_COPY",
            "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE",
            "MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON",
            "MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE",
            "MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE",
            "MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE",
            "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if (
                name == "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE"
                and os.environ.get("MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE") == "1"
            ):
                return True
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["target0762_woabf16bmmcache"],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE" not in result["raw_dsv4_sm80_env"]
    assert "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE" in result["active_dsv4_toggles"]


def test_configure_variant_records_direct_graph_metadata_c4_groups(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS":
                return False
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["dsv4_sm80_a100_victory_directgraphmetadata_c4"],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS"] == "c4"
    assert "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS" not in result["active_dsv4_toggles"]


def test_configure_variant_records_promoted_route_b_metadata_lifetime(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS",
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS":
                return False
            return os.environ.get(name) in {"1", "true"}

    variant = bench._variant_map()["dsv4_sm80_a100_victory_prefix_routeb_lifetime"]
    result = bench.configure_variant(
        FakeKernel,
        variant,
    )

    assert variant.env == {
        "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE": "1",
        "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS": "1",
        "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS": "c4",
        "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE": "1",
    }
    assert variant.allow_dsv4_cuda_graph is True
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS"] == "c4"
    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE"] == "1"
    )
    assert "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS" not in result["active_dsv4_toggles"]


def test_route_b_lifetime_legacy_variant_aliases_promoted_env():
    bench = _load_module()

    variants = bench._variant_map()
    promoted = variants["dsv4_sm80_a100_victory_prefix_routeb_lifetime"]
    legacy = variants["dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime"]

    assert legacy.env == promoted.env
    assert legacy.allow_dsv4_cuda_graph == promoted.allow_dsv4_cuda_graph
    assert legacy.cuda_graph_capture_greedy_sample == promoted.cuda_graph_capture_greedy_sample


def test_route_b_lifetime_moe_reduce_bf16_variant_extends_promoted_env(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS",
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE",
            "MINISGL_DSV4_SM80_MOE_REDUCE_BF16",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS":
                return False
            return os.environ.get(name) in {"1", "true"}

    variants = bench._variant_map()
    promoted = variants["dsv4_sm80_a100_victory_prefix_routeb_lifetime"]
    opt_in = variants["dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16"]
    result = bench.configure_variant(FakeKernel, opt_in)

    assert opt_in.env == {**promoted.env, "MINISGL_DSV4_SM80_MOE_REDUCE_BF16": "1"}
    assert opt_in.allow_dsv4_cuda_graph == promoted.allow_dsv4_cuda_graph
    assert opt_in.cuda_graph_capture_greedy_sample == promoted.cuda_graph_capture_greedy_sample
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_REDUCE_BF16"] == "1"
    assert "MINISGL_DSV4_SM80_MOE_REDUCE_BF16" in result["active_dsv4_toggles"]


def test_marlin_release_variants_expand_full_policy_env(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS",
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE",
            "MINISGL_DSV4_MARLIN_WNA16_PREBUILD",
            "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name in {
                "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
                "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS",
            }:
                return False
            return os.environ.get(name) in {"1", "true"}

    variants = bench._variant_map()
    prebuild = variants["dsv4_sm80_a100_victory_marlin_prebuild"]
    release = variants["dsv4_sm80_a100_victory_marlin_release"]
    prefix_release = variants[
        "dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release"
    ]

    assert prebuild.env == {
        "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE": "1",
        "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND": "marlin_wna16",
        "MINISGL_DSV4_MARLIN_WNA16_PREBUILD": "1",
    }
    assert release.env == {
        **prebuild.env,
        "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS": "1",
    }
    assert prefix_release.env == {
        **variants["dsv4_sm80_a100_victory_prefix_routeb_lifetime"].env,
        "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND": "marlin_wna16",
        "MINISGL_DSV4_MARLIN_WNA16_PREBUILD": "1",
        "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS": "1",
    }

    result = bench.configure_variant(FakeKernel, release)

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_MARLIN_WNA16_PREBUILD"] == "1"
    assert (
        result["raw_dsv4_sm80_env"][
            "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
        ]
        == "1"
    )
    assert "MINISGL_DSV4_MARLIN_WNA16_PREBUILD" in result["active_dsv4_toggles"]
    assert (
        "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
        in result["active_dsv4_toggles"]
    )
    assert release.allow_dsv4_cuda_graph is True
    assert prefix_release.allow_dsv4_cuda_graph is True


def test_marlin_release_env_is_not_preserved_across_variants(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_MARLIN_WNA16_PREBUILD",
            "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            return os.environ.get(name) in {"1", "true"}

    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_PREBUILD", "1")
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "1")

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["dsv4_sm80_a100_victory"],
    )

    assert "MINISGL_DSV4_MARLIN_WNA16_PREBUILD" not in result["preserved_dsv4_sm80_env"]
    assert (
        "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
        not in result["preserved_dsv4_sm80_env"]
    )
    assert "MINISGL_DSV4_MARLIN_WNA16_PREBUILD" not in result["raw_dsv4_sm80_env"]
    assert (
        "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
        not in result["raw_dsv4_sm80_env"]
    )


def test_configure_variant_preserves_route_b_lifetime_verifier(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY_TOGGLE = (
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY"
        )
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS",
            "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS",
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE",
            "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS":
                return False
            return os.environ.get(name) in {"1", "true"}

    monkeypatch.setenv("MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY", "1")
    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["dsv4_sm80_a100_victory_prefix_routeb_lifetime"],
    )

    assert result["preserved_dsv4_sm80_env"] == {
        "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY": "1"
    }
    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY"]
        == "1"
    )
    assert (
        "MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY"
        in result["active_dsv4_toggles"]
    )


def test_configure_variant_preserves_victory_disable_toggles(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV = (
            "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES"
        )
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES",
            "MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES":
                return False
            if (
                name == "MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE"
                and os.environ.get("MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES")
                == "q_wqb"
            ):
                return False
            if (
                name == "MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE"
                and os.environ.get("MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE") == "1"
            ):
                return True
            return os.environ.get(name) in {"1", "true"}

    monkeypatch.setenv("MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES", "q_wqb")
    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["dsv4_sm80_a100_victory"],
    )

    assert result["preserved_dsv4_sm80_env"] == {
        "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES": "q_wqb"
    }
    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES"]
        == "q_wqb"
    )
    assert "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE" not in result["active_dsv4_toggles"]


def test_configure_variant_records_shared_expert_bf16_cache(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["dsv4_sm80_a100_victory_sharedbf16"],
    )

    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE"]
        == "1"
    )
    assert (
        "MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE"
        in result["active_dsv4_toggles"]
    )


def test_configure_variant_records_bf16_small_gemm_pretranspose(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["dsv4_sm80_a100_victory_bf16smallgemm"],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE"]
        == "1"
    )
    assert (
        "MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE"
        in result["active_dsv4_toggles"]
    )


def test_graph_init_variant_prefers_bf16_small_gemm_pretranspose():
    bench = _load_module()
    variants = [
        bench._variant_map()["dsv4_sm80_a100_victory"],
        bench._variant_map()["dsv4_sm80_a100_victory_bf16smallgemm"],
    ]

    assert (
        bench._graph_init_variant(variants).name
        == "dsv4_sm80_a100_victory_bf16smallgemm"
    )
    assert (
        bench._graph_init_variant([bench._variant_map()["dsv4_sm80_a100_victory"]]).name
        == "dsv4_sm80_a100_victory"
    )


def test_configure_variant_records_hc_graph_cleanup(monkeypatch):
    bench = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            return os.environ.get(name) in {"1", "true"}

    result = bench.configure_variant(
        FakeKernel,
        bench._variant_map()["dsv4_sm80_a100_victory_hccleanup"],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP"] == "1"
    assert "MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP" in result["active_dsv4_toggles"]


def test_shared_prefix_workload_repeats_prefix_and_disables_radix_in_scenario():
    bench = _load_module()
    scenario_map = bench._scenario_map()

    for scenario_name in ("shared_prompt_no_radix_bs8", "shared_prompt_reuse_bs8"):
        scenario = scenario_map[scenario_name]
        prompts, sampling_params = bench.build_workload(scenario, vocab_size=1000, seed=7)

        assert len(prompts) == scenario.batch_size
        assert all(
            len(prompt) == scenario.shared_prefix_len + scenario.suffix_len
            for prompt in prompts
        )
        assert all(
            prompt[: scenario.shared_prefix_len] == prompts[0][: scenario.shared_prefix_len]
            for prompt in prompts
        )
        assert all(param.max_tokens == scenario.decode_len for param in sampling_params)
    assert scenario_map["shared_prompt_no_radix_bs8"].kind == "shared_prefix"
    assert scenario_map["shared_prompt_reuse_bs8"].kind == "shared_prefix_reuse"


def test_target08_serving_scenarios_are_selectable_without_changing_defaults():
    bench = _load_module()

    default_args = bench.parse_args([])
    default_names = [scenario.name for scenario in bench._select_scenarios(default_args)]
    assert "serving_mixed_112req_wave16" not in default_names

    args = bench.parse_args(["--scenarios", "decode_ladder_bs16", "serving_mixed_112req_wave16"])
    scenarios = bench._select_scenarios(args)
    assert [scenario.name for scenario in scenarios] == [
        "decode_ladder_bs16",
        "serving_mixed_112req_wave16",
    ]

    serving = scenarios[1]
    prompts, sampling_params = bench.build_workload(serving, vocab_size=1000, seed=11)
    assert len(prompts) == 112
    assert len(sampling_params) == 112
    assert serving.batch_size == 16
    assert serving.wave_size == 16
    assert {param.max_tokens for param in sampling_params} == {16, 24, 32, 48, 64}
    assert max(len(prompt) for prompt in prompts) == 256


def test_bucket_coverage_table_counts_replay_eager_tokens_and_wall_share():
    bench = _load_module()
    repeats = [
        {
            "schedule_trace": [
                {
                    "phase": "decode",
                    "batch_size": 7,
                    "padded_size": 8,
                    "decode_tokens": 7,
                    "forward_s": 0.2,
                    "graph_replay": True,
                    "graph_eager": False,
                },
                {
                    "phase": "decode",
                    "batch_size": 16,
                    "padded_size": 16,
                    "decode_tokens": 16,
                    "forward_s": 0.8,
                    "graph_replay": False,
                    "graph_eager": True,
                },
                {
                    "phase": "prefill",
                    "batch_size": 16,
                    "padded_size": 16,
                    "decode_tokens": 0,
                    "forward_s": 1.0,
                },
            ]
        }
    ]
    graph_status = {
        "enabled": True,
        "captured_bs": [8],
        "replay_count_by_batch_size": {"7": 1},
        "eager_decode_count_by_batch_size": {"16": 1},
    }

    table = bench._bucket_coverage_table(repeats, graph_status)
    by_bs = {row["actual_decode_bs"]: row for row in table}

    assert by_bs[7]["replay_count"] == 1
    assert by_bs[7]["eager_count"] == 0
    assert by_bs[7]["tokens"] == 7
    assert by_bs[7]["wall_share"] == pytest.approx(0.2)
    assert by_bs[16]["replay_count"] == 0
    assert by_bs[16]["eager_count"] == 1
    assert by_bs[16]["tokens"] == 16
    assert by_bs[16]["wall_share"] == pytest.approx(0.8)


def test_aggregate_case_report_has_required_schema_and_bottleneck_labels():
    bench = _load_module()
    base = {
        "status": "pass",
        "variant": {"name": "fallback"},
        "scenario": {"name": "decode_throughput_bs8"},
        "classification": "baseline",
        "report_path": "/tmp/report.json",
    }
    repeat = {
        "elapsed_s": 10.0,
        "prompt_tokens": 16,
        "target_output_tokens": 8,
        "actual_output_tokens": 8,
        "requests": [
            {"ttft_s": 2.0, "topt_s": 0.5, "latency_s": 5.0},
            {"ttft_s": 2.5, "topt_s": 0.6, "latency_s": 6.0},
        ],
        "phase_totals": {
            "prefill_forward_s": 3.0,
            "decode_forward_s": 5.0,
            "prefill_prepare_s": 0.2,
            "decode_prepare_s": 0.1,
            "prefill_input_tokens": 16,
            "decode_tokens": 7,
        },
        "memory": {
            "max_memory_allocated_bytes": 100,
            "max_memory_reserved_bytes": 128,
        },
    }
    payloads = [
        {
            "rank": 0,
            "repeats": [repeat],
            "kv_cache_memory_bytes": 64,
            "kernel_counters": {
                "fallback_wrapper_calls": {
                    "paged_mqa_attention_fallback": 3,
                    "quantized_linear_ref": 5,
                    "store_swa_fallback": 2,
                },
                "optional_kernel_none_skips": {},
                "unsupported_kernel_skips": {},
            },
        },
        {
            "rank": 1,
            "repeats": [repeat],
            "kv_cache_memory_bytes": 64,
            "kernel_counters": {
                "fallback_wrapper_calls": {
                    "paged_mqa_attention_fallback": 3,
                    "quantized_linear_ref": 5,
                    "store_swa_fallback": 2,
                },
                "optional_kernel_none_skips": {},
                "unsupported_kernel_skips": {},
            },
        },
    ]

    report = bench._aggregate_case_report(base=base, rank_payloads=payloads)

    assert report["metrics"]["decode_tokens_per_s"] == 7 / 5
    assert report["metrics"]["prefill_tokens_per_s"] == 16 / 3
    assert report["metrics"]["kv_cache_memory_bytes_total_tp"] == 128
    assert report["kernel_counters"]["fallback_wrapper_calls_total"] == 20
    labels = {row["label"] for row in report["bottlenecks"]}
    assert "decode dominated" in labels
    assert "attention" in labels
    assert "MoE / expert GEMM" in labels
    assert "KV cache writes" in labels

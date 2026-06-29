from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark" / "offline" / "deepseek_v4_perf_matrix.py"


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
    assert [variant.name for variant in variants] == ["fallback", "v0_bf16"]
    assert {scenario.name for scenario in scenarios} >= {
        "long_prefill_bs1",
        "batch_prefill_bs8",
        "decode_throughput_bs8",
        "shared_prompt_no_radix_bs8",
    }
    assert bench.run_classification(tp_size=8, page_size=256, smoke=False) == "baseline"


def test_smoke_or_page_size_one_is_not_reported_as_baseline():
    bench = _load_module()

    args = bench.parse_args(["--smoke", "--page-size", "1"])
    scenarios = bench._select_scenarios(args)

    assert [scenario.name for scenario in scenarios] == ["smoke_debug"]
    assert bench.run_classification(tp_size=8, page_size=1, smoke=True) == "smoke_debug"
    assert bench.run_classification(tp_size=1, page_size=256, smoke=False) == "smoke_debug"


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


def test_shared_prefix_workload_repeats_prefix_and_disables_radix_in_scenario():
    bench = _load_module()
    scenario = bench._scenario_map()["shared_prompt_no_radix_bs8"]

    prompts, sampling_params = bench.build_workload(scenario, vocab_size=1000, seed=7)

    assert len(prompts) == scenario.batch_size
    assert all(len(prompt) == scenario.shared_prefix_len + scenario.suffix_len for prompt in prompts)
    assert all(prompt[: scenario.shared_prefix_len] == prompts[0][: scenario.shared_prefix_len] for prompt in prompts)
    assert all(param.max_tokens == scenario.decode_len for param in sampling_params)
    assert scenario.kind == "shared_prefix"


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

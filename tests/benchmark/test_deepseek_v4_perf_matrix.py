from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "debug/dsv4/benchmark/offline/deepseek_v4_perf_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("deepseek_v4_perf_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
perf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = perf
SPEC.loader.exec_module(perf)


def test_historical_4096_128_bs4_is_retained():
    scenario = perf._scenario_map()["historical_4096_128_bs4"]
    assert scenario.prompt_len == 4096
    assert scenario.decode_len == 128
    assert scenario.batch_size == 4


def test_release_matrix_defaults_to_canonical_optimized_variant():
    args = perf.parse_args(["--scenarios", "historical_4096_128_bs4"])
    variants = perf._select_variants(args)
    assert [variant.name for variant in variants] == [perf.DSV4_RELEASE_DEFAULT_VARIANT]
    assert variants[0].allow_dsv4_cuda_graph is True
    assert variants[0].use_pynccl is True


def test_historical_feature_variant_is_rejected_by_parser():
    with pytest.raises(SystemExit):
        perf.parse_args(["--variants", "v1_moe"])


def test_release_matrix_keeps_page_size_256():
    assert perf.parse_args([]).page_size == 256


def test_scenario_override_keeps_macro_shape_explicit():
    args = perf.parse_args(
        [
            "--scenarios",
            "historical_4096_128_bs4",
            "--repeats",
            "3",
        ]
    )
    scenario = perf._select_scenarios(args)[0]
    assert scenario.repeats == 3
    assert (scenario.prompt_len, scenario.decode_len, scenario.batch_size) == (4096, 128, 4)


def test_target15_delayed_arrival_workload_has_one_long_request():
    scenario = perf._scenario_map()["target15_mixed_arrival_m4_64k"]
    prompts, sampling_params = perf.build_workload(
        scenario,
        vocab_size=129280,
        seed=0,
        token_id_range=1024,
    )
    assert [len(prompt) for prompt in prompts] == [128, 128, 128, 128, 65536]
    assert [params.max_tokens for params in sampling_params] == [160, 160, 160, 160, 8]
    assert scenario.initial_requests == 4
    assert scenario.arrival_after_decode_batches == 1


def test_target15_candidate_selector_is_benchmark_only_and_explicit():
    args = perf.parse_args(["--mixed-policy-candidate", "candidate-b"])
    assert args.mixed_policy_candidate == "candidate-b"


def test_target15_natural_text_workload_is_delayed_greedy_chat_plus_64k():
    scenario = perf._scenario_map()["target15_mixed_natural_text_m1_64k"]
    prompts, sampling_params = perf.build_workload(
        scenario,
        vocab_size=129280,
        seed=0,
        token_id_range=1024,
    )
    assert prompts[0] == perf.TARGET15_NATURAL_PROMPT
    assert isinstance(prompts[1], list) and len(prompts[1]) == 65536
    assert [params.temperature for params in sampling_params] == [0.0, 0.0]
    assert [params.max_tokens for params in sampling_params] == [256, 8]
    assert [params.ignore_eos for params in sampling_params] == [False, True]
    assert scenario.initial_requests == 1
    assert scenario.arrival_after_decode_batches == 1


def test_target15_natural_text_chat_formatter_has_safe_fallback(tmp_path):
    formatted = perf._format_target15_chat_prompt(
        perf.TARGET15_NATURAL_PROMPT,
        model_path=str(tmp_path),
    )
    assert formatted.startswith("System: ")
    assert perf.TARGET15_NATURAL_SYSTEM_PROMPT in formatted
    assert formatted.endswith("\nAssistant:")

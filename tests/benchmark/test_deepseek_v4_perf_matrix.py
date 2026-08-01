from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "debug/dsv4/benchmark/offline/deepseek_v4_perf_matrix.py"
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


@pytest.mark.parametrize(
    ("recipe", "batch_size"),
    [
        ("long_m8", 8),
        ("low_m64", 64),
        ("default_m128", 128),
        ("high_m256", 256),
    ],
)
@pytest.mark.parametrize(("prompt_label", "prompt_len"), [("p1k", 1024), ("p4k", 4096)])
def test_release_card_scenarios_use_recipe_max_m_and_fixed_decode(
    recipe, batch_size, prompt_label, prompt_len
):
    scenario = perf._scenario_map()[f"release_card_{recipe}_{prompt_label}_d1k"]
    assert scenario.batch_size == batch_size
    assert scenario.prompt_len == prompt_len
    assert scenario.decode_len == 1024
    assert scenario.repeats == 1
    assert scenario.warmup_repeats == 0


def test_removed_long_context_m4_recipe_is_rejected_by_benchmark_parser():
    with pytest.raises(SystemExit):
        perf.parse_args(["--recipe", "long_context_m4"])


def test_context_length_uses_release_spelling_for_max_sequence_override():
    args = perf.parse_args(["--context-length", "1048576"])
    assert args.max_seq_len == 1048576
    with pytest.raises(SystemExit):
        perf.parse_args(["--context-length", "1048576", "--max-seq-len", "1048576"])


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


@pytest.mark.parametrize(
    ("name", "batch_size", "prompt_len"),
    [
        ("long_context_pressure_512k_bs4", 4, 524288),
        ("long_context_pressure_1m_bs2", 2, 1048568),
    ],
)
def test_target16_long_pressure_prompts_diverge_at_token_zero(name, batch_size, prompt_len):
    scenario = perf._scenario_map()[name]
    prompts, sampling_params = perf.build_workload(
        scenario,
        vocab_size=129280,
        seed=0,
        token_id_range=1024,
    )
    assert len(prompts) == batch_size
    assert all(len(prompt) == prompt_len for prompt in prompts)
    assert len({prompt[0] for prompt in prompts}) == batch_size
    assert all(params.max_tokens == 8 for params in sampling_params)
    assert all(params.ignore_eos for params in sampling_params)


def test_target16_bounded_gate_shapes_are_retained():
    resident = perf._scenario_map()["target16_resident_64k_bs4"]
    control = perf._scenario_map()["target16_single_64k_control"]
    frontier = perf._scenario_map()["target16_budget_frontier_64k_bs8"]
    gate_a = perf._scenario_map()["target16_long_context_512k_bs8"]
    gate_b = perf._scenario_map()["target16_exact_1m_total_bs4"]
    mixed = perf._scenario_map()["target16_mixed_decode_m4_multi_prefill_32k"]

    assert (resident.batch_size, resident.prompt_len, resident.decode_len) == (
        4,
        65_536,
        8,
    )
    assert (control.batch_size, control.prompt_len, control.decode_len) == (
        1,
        65_536,
        8,
    )
    assert (frontier.batch_size, frontier.prompt_len, frontier.decode_len) == (
        8,
        65_536,
        8,
    )
    assert (gate_a.batch_size, gate_a.prompt_len, gate_a.decode_len) == (
        8,
        524_288,
        8,
    )
    assert (gate_b.batch_size, gate_b.prompt_len, gate_b.decode_len) == (
        4,
        1_048_568,
        8,
    )
    assert mixed.initial_requests == 4
    assert mixed.prompt_len_cycle == (128,) * 4 + (32_768,) * 4
    assert mixed.decode_len_cycle == (64,) * 4 + (8,) * 4


def test_benchmark_budget_defaults_to_release_value_without_auto_expansion():
    args = perf.parse_args(
        [
            "--scenarios",
            "target16_resident_64k_bs4",
        ]
    )
    scenarios = perf._select_scenarios(args)

    resolution = perf._resolve_benchmark_max_extend_tokens(args, scenarios)

    assert resolution.resolved_max_extend_tokens == 8192
    assert resolution.source == "benchmark_release_default"
    assert resolution.resolved_before_model_load is True


def test_schedule_summary_retains_resident_owner_and_budget_evidence():
    owner_reqs = [
        {
            "generation_id": generation,
            "c4_owner_generation": generation,
            "c128_owner_generation": generation,
        }
        for generation in range(4)
    ]
    repeats = [
        {
            "schedule_trace": [
                {
                    "phase": "prefill",
                    "batch_size": 4,
                    "padded_size": 4,
                    "input_tokens": 8192,
                    "sequence_ownership": {
                        "c4_owner_count": 4,
                        "c128_owner_count": 4,
                        "swa_ownership_version": 10,
                        "reqs": owner_reqs,
                    },
                },
                {
                    "phase": "decode",
                    "batch_size": 4,
                    "padded_size": 4,
                    "sequence_ownership": {
                        "c4_owner_count": 4,
                        "c128_owner_count": 4,
                        "swa_ownership_version": 11,
                        "reqs": owner_reqs,
                    },
                },
            ],
            "sequence_ownership_after_cleanup": {
                "c4_owner_count": 0,
                "c128_owner_count": 0,
                "swa_ownership_version": 12,
            },
        }
    ]

    summary = perf._schedule_summary(repeats)

    assert summary["max_prefill_request_batch"] == 4
    assert summary["max_decode_m"] == 4
    assert summary["max_prefill_input_tokens"] == 8192
    assert summary["max_c4_owner_count"] == 4
    assert summary["max_c128_owner_count"] == 4
    assert summary["owner_generation_matches"] is True
    assert summary["all_sequence_owners_released_after_cleanup"] is True
    assert summary["swa_ownership_version_range"] == [10, 12]


def test_512k_budget_guard_requires_explicit_serving_or_large_budget():
    scenario_name = "long_context_pressure_512k_bs4"
    args = perf.parse_args(["--scenarios", scenario_name])
    scenarios = perf._select_scenarios(args)
    with pytest.raises(ValueError, match="require --use-serving"):
        perf._resolve_benchmark_max_extend_tokens(args, scenarios)

    serving_args = perf.parse_args(
        ["--scenarios", scenario_name, "--use-serving-max-extend-tokens"]
    )
    serving_resolution = perf._resolve_benchmark_max_extend_tokens(
        serving_args, perf._select_scenarios(serving_args)
    )
    assert serving_resolution.resolved_max_extend_tokens == 8192
    assert serving_resolution.long_context_scenarios == (scenario_name,)


def test_512k_budget_guard_rejects_unintended_large_budget():
    args = perf.parse_args(
        [
            "--scenarios",
            "long_context_pressure_512k_bs4",
            "--max-extend-tokens",
            "9000",
        ]
    )

    with pytest.raises(ValueError, match="exactly 8192 or 16384"):
        perf._resolve_benchmark_max_extend_tokens(args, perf._select_scenarios(args))


@pytest.mark.parametrize(
    "scenario_name", ["long_context_pressure_512k_bs4", "long_context_pressure_1m_bs2"]
)
def test_512k_and_1m_budget_guard_rejects_256(
    scenario_name: str,
) -> None:
    args = perf.parse_args(
        [
            "--scenarios",
            scenario_name,
            "--max-extend-tokens",
            "256",
        ]
    )

    with pytest.raises(ValueError, match="256 is forbidden"):
        perf._resolve_benchmark_max_extend_tokens(args, perf._select_scenarios(args))


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


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("stop", False),
        ("length", True),
    ],
)
def test_terminal_eos_accounting_depends_on_finish_reason(finish_reason, expected):
    assert (
        perf._is_emitted_benchmark_output_token(
            next_token=100,
            eos_token_id=100,
            finished=True,
            finish_reason=finish_reason,
        )
        is expected
    )


def test_graph_status_delta_includes_dual_width_route_counters():
    before = {
        "replay_count": 7,
        "replay_count_by_graph_key": {"M1:short": 7},
        "unsupported_m_eager_count": 1,
        "unsupported_m_eager_count_by_batch_size": {"9": 1},
        "context_overflow_eager_count": 0,
        "context_overflow_eager_count_by_batch_size": {},
        "short_to_wide_transition_count": 2,
    }
    after = {
        "replay_count": 10,
        "replay_count_by_graph_key": {"M1:short": 7, "M4:wide": 3},
        "unsupported_m_eager_count": 2,
        "unsupported_m_eager_count_by_batch_size": {"9": 1, "17": 1},
        "context_overflow_eager_count": 0,
        "context_overflow_eager_count_by_batch_size": {},
        "short_to_wide_transition_count": 3,
    }

    delta = perf._graph_status_delta(before, after)

    assert delta["replay_count"] == 3
    assert delta["replay_count_by_graph_key"] == {"M4:wide": 3}
    assert delta["unsupported_m_eager_count"] == 1
    assert delta["unsupported_m_eager_count_by_batch_size"] == {"17": 1}
    assert delta["context_overflow_eager_count"] == 0
    assert delta["context_overflow_eager_count_by_batch_size"] == {}
    assert delta["short_to_wide_transition_count"] == 1

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmark/offline/deepseek_v4_perf_matrix.py"
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

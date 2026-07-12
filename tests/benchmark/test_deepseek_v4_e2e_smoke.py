from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "debug/dsv4/benchmark/offline/deepseek_v4_e2e_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("deepseek_v4_e2e_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def _parse_variant(variant: str):
    return smoke._parse_args(
        [
            "--model-path",
            "/unused/model",
            "--variant",
            variant,
            "--output",
            "/unused/output.json",
        ]
    )


@pytest.mark.parametrize("variant", ["optimized", "fallback"])
def test_cli_accepts_canonical_variants(variant):
    assert _parse_variant(variant).variant == variant


def test_variants_map_to_typed_runtime_without_historical_kernel_access():
    optimized = smoke._configure_variant("optimized")
    fallback = smoke._configure_variant("fallback")

    assert optimized.mode == "optimized"
    assert optimized.moe_expert_backend == "marlin_wna16"
    assert optimized.release_raw_expert_weights is True
    assert fallback.mode == "fallback"
    assert fallback.moe_expert_backend == "grouped_fp4"
    assert fallback.release_raw_expert_weights is False
    assert smoke._execution_settings(optimized, True) == (True, True)
    assert smoke._execution_settings(fallback, True) == (False, False)
    assert "dsv4_kernel" not in smoke.__dict__

    source = SCRIPT.read_text(encoding="utf-8")
    removed_names = (
        "DSV4_SM80_KNOWN_TOGGLES",
        "DSV4_SM80_V0_BF16_TOGGLE",
        "DSV4_SM80_V1_MOE_TOGGLE",
        "dsv4_env_flag",
        "MINISGL_DSV4_",
    )
    assert all(name not in source for name in removed_names)


@pytest.mark.parametrize("variant", ["v0_bf16", "v1_moe"])
def test_cli_rejects_historical_variants(variant):
    with pytest.raises(SystemExit):
        _parse_variant(variant)

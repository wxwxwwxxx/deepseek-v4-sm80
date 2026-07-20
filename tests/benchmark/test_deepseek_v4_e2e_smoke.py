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


def _parse_smoke():
    return smoke._parse_args(
        [
            "--model-path",
            "/unused/model",
            "--output",
            "/unused/output.json",
        ]
    )


def test_smoke_uses_only_optimized_release_without_historical_kernel_access():
    assert _parse_smoke().model_path == "/unused/model"
    assert smoke._execution_settings(True) == (True, True)
    assert smoke._execution_settings(False) == (False, True)
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
    with pytest.raises(SystemExit):
        smoke._parse_args(
            [
                "--model-path",
                "/unused/model",
                "--variant",
                "fallback",
                "--output",
                "/unused/output.json",
            ]
        )

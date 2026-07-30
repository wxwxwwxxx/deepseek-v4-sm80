from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "debug/dsv4/benchmark/offline/deepseek_v4_c4_production_poison.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deepseek_v4_c4_production_poison",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
production_poison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = production_poison
SPEC.loader.exec_module(production_poison)


@pytest.mark.skipif(
    not production_poison.torch.cuda.is_available(),
    reason="CUDA is required for the production Triton C4 proof",
)
def test_c4_production_candidate_a_poison_matrix():
    result = production_poison.run_production_poison()
    if result["status"] == "skip":
        pytest.skip(result["reason"])
    assert result["status"] == "pass"
    assert result["candidate"] == "A"
    assert result["phase_metadata_required"] is False
    assert result["separate_restore_or_materialization_launches"] == 0
    assert result["raw_page_inputs_to_producer"] == 0
    assert result["checkpoint_addressing"] == (
        "component_page_table[row, logical_full_page]"
    )
    assert len(result["cases"]) == 4
    for case in result["cases"]:
        assert case["status"] == "pass"
        assert len(case["component_page_contract_cases"]) == 2
        for contract in case["component_page_contract_cases"]:
            assert contract["raw_prefix_tombstoned"]
            assert contract["live_tombstoned_bitwise_identical"]
            assert contract["checkpoint_immutable"]
            assert contract["checkpoint_component_page"] == (
                contract["retained_component_pages"][-1]
            )
            assert contract["branch_component_pages"][:-1] == (
                contract["retained_component_pages"]
            )
        assert case["required_prefix_outputs"] == [
            259,
            263,
            515,
            519,
            259,
            259,
            263,
            263,
        ]

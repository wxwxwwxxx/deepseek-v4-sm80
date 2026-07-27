from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "debug/dsv4/benchmark/offline/deepseek_v4_c4_compact_checkpoint_proof.py"
)
SPEC = importlib.util.spec_from_file_location(
    "deepseek_v4_c4_compact_checkpoint_proof",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
proof = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = proof
SPEC.loader.exec_module(proof)


def test_c4_minimal_checkpoint_and_poison_matrix():
    result = proof.run_proof()
    assert result["status"] == "pass"
    assert result["cases_passed"] == result["cases_total"]
    assert result["compact_to_old_ratio"] == 0.25
    assert result["page_size_mod_ratio"] == 0
    assert result["phase_metadata_required"] is False
    assert result["required_checkpoint"] == {
        "rows": 4,
        "kv_half": "left",
        "score_half": "left",
        "dtype": "torch.float32",
        "ordering": "absolute positions page_end-4..page_end-1",
    }

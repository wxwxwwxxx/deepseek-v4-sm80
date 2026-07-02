#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

PERF_MATRIX = ROOT / "benchmark" / "offline" / "deepseek_v4_perf_matrix.py"
TEXT_SMOKE = ROOT / "benchmark" / "offline" / "deepseek_v4_text_smoke.py"

STALE_FP8_OPT_INS = (
    "MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM",
    "MINISGL_DSV4_SM80_WO_B_FP8_GEMM",
    "MINISGL_DSV4_SM80_INDEXER_WQB_FP8_GEMM",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def expansion(module: Any, dsv4_kernel: Any, variant_name: str) -> dict[str, Any]:
    variant = module._variant_map()[variant_name]
    env_report = module.configure_variant(dsv4_kernel, variant)
    active = env_report["active_dsv4_toggles"]
    return {
        "variant": variant_name,
        "variant_env": dict(variant.env),
        "description": variant.description,
        "allow_dsv4_cuda_graph": bool(variant.allow_dsv4_cuda_graph),
        "cuda_graph_capture_greedy_sample": bool(variant.cuda_graph_capture_greedy_sample),
        "raw_dsv4_sm80_env": env_report["raw_dsv4_sm80_env"],
        "active_dsv4_toggles": active,
        "moe_expert_backend": dsv4_kernel.dsv4_moe_expert_backend(),
        "stale_fp8_opt_ins": {
            name: bool(dsv4_kernel.dsv4_env_flag(name)) for name in STALE_FP8_OPT_INS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    perf = load_module("target0763_perf_matrix", PERF_MATRIX)
    smoke = load_module("target0763_text_smoke", TEXT_SMOKE)
    milestone = "dsv4_sm80_a100_victory"
    alias = "target0762_woabf16bmmcache"

    perf_milestone = expansion(perf, dsv4_kernel, milestone)
    perf_alias = expansion(perf, dsv4_kernel, alias)
    smoke_milestone = expansion(smoke, dsv4_kernel, milestone)
    smoke_alias = expansion(smoke, dsv4_kernel, alias)

    payload = {
        "cwd": str(ROOT),
        "milestone_variant": milestone,
        "legacy_alias": alias,
        "perf_matrix": {
            milestone: perf_milestone,
            alias: perf_alias,
            "env_expansion_equal": (
                perf_milestone["raw_dsv4_sm80_env"] == perf_alias["raw_dsv4_sm80_env"]
                and perf_milestone["active_dsv4_toggles"]
                == perf_alias["active_dsv4_toggles"]
                and perf_milestone["moe_expert_backend"] == perf_alias["moe_expert_backend"]
            ),
        },
        "text_smoke": {
            milestone: smoke_milestone,
            alias: smoke_alias,
            "env_expansion_equal": (
                smoke_milestone["raw_dsv4_sm80_env"] == smoke_alias["raw_dsv4_sm80_env"]
                and smoke_milestone["active_dsv4_toggles"]
                == smoke_alias["active_dsv4_toggles"]
                and smoke_milestone["moe_expert_backend"] == smoke_alias["moe_expert_backend"]
            ),
        },
        "a100_victory_bundle_whitelist": list(
            getattr(dsv4_kernel, "DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST", ())
        ),
        "stale_fp8_opt_ins_expected_inactive": list(STALE_FP8_OPT_INS),
        "ambient_dsv4_env_after_collection": {
            name: os.environ[name]
            for name in sorted(os.environ)
            if name.startswith("MINISGL_DSV4_SM80_")
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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
    active = set(env_report["active_dsv4_toggles"])
    return {
        "variant": variant_name,
        "variant_env": dict(variant.env),
        "description": variant.description,
        "allow_dsv4_cuda_graph": bool(variant.allow_dsv4_cuda_graph),
        "cuda_graph_capture_greedy_sample": bool(variant.cuda_graph_capture_greedy_sample),
        "raw_dsv4_sm80_env": env_report["raw_dsv4_sm80_env"],
        "active_dsv4_toggles": sorted(active),
        "shared_expert_bf16_weight_cache_active": (
            dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE in active
        ),
        "metadata_deforest_active": (
            dsv4_kernel.DSV4_SM80_DECODE_METADATA_DEFOREST_TOGGLE in active
        ),
        "moe_expert_backend": dsv4_kernel.dsv4_moe_expert_backend(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    perf = load_module("target0767_perf_matrix", PERF_MATRIX)
    smoke = load_module("target0767_text_smoke", TEXT_SMOKE)
    ambient_before = {
        name: os.environ[name]
        for name in sorted(os.environ)
        if name.startswith("MINISGL_DSV4_SM80_")
    }
    variants = (
        "dsv4_sm80_a100_victory",
        "dsv4_sm80_a100_victory_sharedbf16",
        "dsv4_sm80_a100_victory_metadatadeforest",
    )
    payload = {
        "cwd": str(ROOT),
        "variants": {
            "perf_matrix": {
                name: expansion(perf, dsv4_kernel, name) for name in variants
            },
            "text_smoke": {
                name: expansion(smoke, dsv4_kernel, name) for name in variants
            },
        },
        "a100_victory_bundle_whitelist": list(
            getattr(dsv4_kernel, "DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST", ())
        ),
        "bundle_contains_shared_expert_bf16_weight_cache": (
            dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
            in getattr(dsv4_kernel, "DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST", ())
        ),
        "bundle_contains_metadata_deforest": (
            dsv4_kernel.DSV4_SM80_DECODE_METADATA_DEFOREST_TOGGLE
            in getattr(dsv4_kernel, "DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST", ())
        ),
        "ambient_dsv4_env_at_process_start": ambient_before,
        "note": (
            "Variant expansion calls intentionally mutate this Python process env; "
            "the recorded source of truth is each variant's raw_dsv4_sm80_env and "
            "active_dsv4_toggles."
        ),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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
    return {
        "variant": variant_name,
        "variant_env": dict(variant.env),
        "description": variant.description,
        "allow_dsv4_cuda_graph": bool(variant.allow_dsv4_cuda_graph),
        "cuda_graph_capture_greedy_sample": bool(variant.cuda_graph_capture_greedy_sample),
        "raw_dsv4_sm80_env": env_report["raw_dsv4_sm80_env"],
        "active_dsv4_toggles": env_report["active_dsv4_toggles"],
        "moe_expert_backend": dsv4_kernel.dsv4_moe_expert_backend(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    perf = load_module("target0764_perf_matrix", PERF_MATRIX)
    smoke = load_module("target0764_text_smoke", TEXT_SMOKE)
    baseline = "dsv4_sm80_a100_victory"
    metadatadeforest = "dsv4_sm80_a100_victory_metadatadeforest"

    payload = {
        "cwd": str(ROOT),
        "baseline_variant": baseline,
        "metadata_deforest_variant": metadatadeforest,
        "perf_matrix": {
            baseline: expansion(perf, dsv4_kernel, baseline),
            metadatadeforest: expansion(perf, dsv4_kernel, metadatadeforest),
        },
        "text_smoke": {
            baseline: expansion(smoke, dsv4_kernel, baseline),
            metadatadeforest: expansion(smoke, dsv4_kernel, metadatadeforest),
        },
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

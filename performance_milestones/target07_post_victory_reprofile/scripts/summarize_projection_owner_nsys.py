#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT
    / "performance_milestones"
    / "target07_cached_bf16_indexer_wq_b_projection_backend"
    / "scripts"
    / "summarize_projection_owner_nsys.py"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("target0763_projection_owner_summary", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base summarizer from {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()

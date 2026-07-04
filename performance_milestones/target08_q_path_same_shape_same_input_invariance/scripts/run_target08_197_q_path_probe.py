#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXACT_PROBE = (
    ROOT
    / "performance_milestones"
    / "target08_exact_path_slot_page_invariance"
    / "scripts"
    / "run_dsv4_exact_path_invariance_probe.py"
)


def _load_exact_probe():
    spec = importlib.util.spec_from_file_location("target08_exact_path_probe", EXACT_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {EXACT_PROBE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_probe = _load_exact_probe()


def _fixed_fillers(
    *,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    seed: int,
    marker_base: int,
) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        _probe._random_tokens(
            rng,
            page_size + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=marker_base + index,
        )
        for index in range(3)
    ]


def _slot_scenario(slot: int):
    def build(
        bank: dict[str, list[int]],
        rng: random.Random,
        vocab_size: int,
        token_id_range: int,
        page_size: int,
    ):
        del rng
        target = list(bank["target_257"])
        fillers = _fixed_fillers(
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            page_size=page_size,
            seed=819700,
            marker_base=510,
        )
        prompts: list[list[int]] = []
        labels: list[str] = []
        filler_index = 0
        for row in range(4):
            if row == slot:
                prompts.append(target)
                labels.append("target")
            else:
                prompts.append(list(fillers[filler_index]))
                labels.append(f"fixed_filler{filler_index}")
                filler_index += 1
        return _probe.InvarianceScenario(
            name=f"target_slot{slot}_fixed_fillers",
            description=(
                "Controlled bs=4/len=257 same-shape batch with the target prompt in "
                f"slot {slot} and the same filler multiset as the other slot scenarios."
            ),
            coverage=[
                "same-shape oracle",
                "target row position changes",
                "filler multiset fixed",
            ],
            probe_prompts=prompts,
            probe_labels=labels,
            prelude_batches=[],
        )

    return build


def _alt_filler_scenario(tag: str, seed: int, marker_base: int):
    def build(
        bank: dict[str, list[int]],
        rng: random.Random,
        vocab_size: int,
        token_id_range: int,
        page_size: int,
    ):
        del rng
        fillers = _fixed_fillers(
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            page_size=page_size,
            seed=seed,
            marker_base=marker_base,
        )
        return _probe.InvarianceScenario(
            name=f"target_slot0_{tag}_fillers",
            description=(
                "Controlled bs=4/len=257 same-shape batch with target fixed in slot 0 "
                f"and alternate filler content set {tag}."
            ),
            coverage=[
                "same-shape oracle",
                "target row fixed",
                "filler content changes",
            ],
            probe_prompts=[list(bank["target_257"]), *fillers],
            probe_labels=["target", f"{tag}_filler0", f"{tag}_filler1", f"{tag}_filler2"],
            prelude_batches=[],
        )

    return build


def _install_target08_197_scenarios() -> None:
    builders: dict[str, Any] = {
        "single_target_alone": _probe._scenario_single_target,
        "identical_prompts_batch": _probe._scenario_identical_slots,
        "target_slot0_fixed_fillers": _slot_scenario(0),
        "target_slot1_fixed_fillers": _slot_scenario(1),
        "target_slot2_fixed_fillers": _slot_scenario(2),
        "target_slot3_fixed_fillers": _slot_scenario(3),
        "target_slot0_altA_fillers": _alt_filler_scenario("altA", 819701, 610),
        "target_slot0_altB_fillers": _alt_filler_scenario("altB", 819702, 710),
    }
    _probe._SCENARIO_BUILDERS = builders
    _probe._SCENARIO_ORDER = list(builders)


def main(argv: list[str] | None = None) -> None:
    _install_target08_197_scenarios()
    raise SystemExit(_probe.run(_probe.parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
"""No-weight phase-scheduling harness for TARGET 15.1.

This deliberately models only the production phase arbitration boundary.  It
does not construct requests, cache managers, the model, or CUDA state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from minisgl.scheduler.phase_policy import MixedPhaseFairPolicy

RELEASE_PREFILL_BUDGET = 8192


@dataclass(frozen=True)
class Decision:
    phase: str | None
    prefill_budget: int = 0


class PhasePolicy(Protocol):
    def select(self, *, prefill_runnable: bool, decode_runnable: bool) -> Decision: ...

    def record(self, phase: str) -> None: ...


class ProductionPrefillFirst:
    """Exact phase order in Scheduler._schedule_next_batch at the baseline SHA."""

    def select(self, *, prefill_runnable: bool, decode_runnable: bool) -> Decision:
        if prefill_runnable:
            return Decision("prefill", RELEASE_PREFILL_BUDGET)
        if decode_runnable:
            return Decision("decode")
        return Decision(None)

    def record(self, phase: str) -> None:
        del phase


class PermanentDecodeFirst:
    """Rejected endpoint used to prove reverse starvation is observable."""

    def select(self, *, prefill_runnable: bool, decode_runnable: bool) -> Decision:
        if decode_runnable:
            return Decision("decode")
        if prefill_runnable:
            return Decision("prefill", RELEASE_PREFILL_BUDGET)
        return Decision(None)

    def record(self, phase: str) -> None:
        del phase


class BoundedPolicyAdapter:
    def __init__(self, *, max_consecutive_decode: int, mixed_prefill_budget: int):
        self.policy = MixedPhaseFairPolicy(
            isolated_prefill_budget=RELEASE_PREFILL_BUDGET,
            max_consecutive_decode=max_consecutive_decode,
            mixed_prefill_budget=mixed_prefill_budget,
        )
        self._runnable = (False, False)

    def select(self, *, prefill_runnable: bool, decode_runnable: bool) -> Decision:
        self._runnable = (prefill_runnable, decode_runnable)
        decision = self.policy.choose(
            prefill_runnable=prefill_runnable,
            decode_runnable=decode_runnable,
        )
        return Decision(decision.phase, decision.prefill_budget)

    def record(self, phase: str) -> None:
        prefill_runnable, decode_runnable = self._runnable
        self.policy.record_scheduled(
            phase,
            prefill_runnable=prefill_runnable,
            decode_runnable=decode_runnable,
        )


@dataclass(frozen=True)
class Scenario:
    name: str
    prefill_tokens: tuple[int, ...]
    decode_steps: int
    decode_arrival_step: int | None = 0
    sustain_decode: bool = False
    max_steps: int = 256


def _consecutive(phase: str | None, previous: str | None, count: int) -> int:
    return count + 1 if phase == previous else (1 if phase is not None else 0)


def simulate(policy: PhasePolicy, scenario: Scenario) -> dict:
    prefills = list(scenario.prefill_tokens)
    decode_remaining = 0
    decode_arrived = False
    trace: list[dict] = []
    previous_phase: str | None = None
    consecutive = 0
    prefill_wait = 0
    decode_wait = 0
    prefill_progress = 0
    decode_progress = 0

    for step in range(scenario.max_steps):
        if (
            not decode_arrived
            and scenario.decode_arrival_step is not None
            and step >= scenario.decode_arrival_step
        ):
            decode_arrived = True
            decode_remaining = scenario.decode_steps

        prefill_runnable = bool(prefills)
        decode_runnable = decode_arrived and (
            scenario.sustain_decode or decode_remaining > 0
        )
        if not prefill_runnable and not decode_runnable:
            break

        decision = policy.select(
            prefill_runnable=prefill_runnable,
            decode_runnable=decode_runnable,
        )
        phase = decision.phase
        if phase is None:
            break
        if phase == "prefill" and not prefill_runnable:
            raise AssertionError("policy selected non-runnable prefill")
        if phase == "decode" and not decode_runnable:
            raise AssertionError("policy selected non-runnable decode")

        consecutive = _consecutive(phase, previous_phase, consecutive)
        previous_phase = phase
        chunk_size = 0
        if phase == "prefill":
            budget = decision.prefill_budget
            if budget <= 0:
                raise AssertionError("prefill decision must have a positive budget")
            # Match PrefillManager: fill one batch in queue order until the
            # budget is exhausted.  An unfinished request remains at the
            # front for the next pass.
            while prefills and budget > 0:
                consumed = min(prefills[0], budget)
                prefills[0] -= consumed
                budget -= consumed
                chunk_size += consumed
                prefill_progress += consumed
                if prefills[0] == 0:
                    prefills.pop(0)
                else:
                    break
        else:
            decode_progress += 1
            if not scenario.sustain_decode:
                decode_remaining -= 1

        policy.record(phase)

        if prefill_runnable and phase != "prefill":
            prefill_wait += 1
        if decode_runnable and phase != "decode":
            decode_wait += 1
        trace.append(
            {
                "step": step,
                "phase": phase,
                "prefill_budget": decision.prefill_budget,
                "actual_chunk_size": chunk_size,
                "prefill_queue_age": prefill_wait,
                "decode_queue_age": decode_wait,
                "consecutive_phase_count": consecutive,
                "both_runnable": prefill_runnable and decode_runnable,
                "prefill_remaining_tokens": sum(prefills),
                "decode_remaining_steps": (
                    None if scenario.sustain_decode else max(decode_remaining, 0)
                ),
            }
        )

    both_trace = [row for row in trace if row["both_runnable"]]
    max_consecutive = {"prefill": 0, "decode": 0}
    run_phase = None
    run_count = 0
    for row in both_trace:
        if row["phase"] == run_phase:
            run_count += 1
        else:
            run_phase = row["phase"]
            run_count = 1
        max_consecutive[run_phase] = max(max_consecutive[run_phase], run_count)
    return {
        "scenario": asdict(scenario),
        "phase_sequence": "".join(row["phase"][0].upper() for row in trace),
        "steps": len(trace),
        "prefill_progress_tokens": prefill_progress,
        "decode_progress_steps": decode_progress,
        "prefill_remaining_tokens": sum(prefills),
        "decode_remaining_steps": (
            None if scenario.sustain_decode else max(decode_remaining, 0)
        ),
        "max_consecutive_while_both_runnable": max_consecutive,
        "trace": trace,
    }


SCENARIOS = (
    Scenario(
        name="long_512k_equivalent_decode_runnable",
        prefill_tokens=(64 * RELEASE_PREFILL_BUDGET,),
        decode_steps=1,
        sustain_decode=True,
        max_steps=1400,
    ),
    Scenario(
        name="decode_becomes_runnable_after_prefill_started",
        prefill_tokens=(64 * RELEASE_PREFILL_BUDGET,),
        decode_steps=4,
        decode_arrival_step=8,
        max_steps=80,
    ),
    Scenario(
        name="continuous_decode_with_pending_prefill",
        prefill_tokens=(4 * RELEASE_PREFILL_BUDGET,),
        decode_steps=1,
        sustain_decode=True,
        max_steps=32,
    ),
    Scenario(
        name="ordinary_prefills_plus_chunked",
        prefill_tokens=(512, 1024, 2048, 5 * RELEASE_PREFILL_BUDGET),
        decode_steps=4,
        max_steps=24,
    ),
    Scenario(
        name="only_prefill_control",
        prefill_tokens=(512, 1024, 3 * RELEASE_PREFILL_BUDGET),
        decode_steps=0,
        decode_arrival_step=None,
        max_steps=16,
    ),
    Scenario(
        name="only_decode_control",
        prefill_tokens=(),
        decode_steps=8,
        max_steps=16,
    ),
)


def build_report() -> dict:
    policies: dict[str, Callable[[], PhasePolicy]] = {
        "production_prefill_first": ProductionPrefillFirst,
        "rejected_permanent_decode_first": PermanentDecodeFirst,
        "candidate_a_d4_p2048": lambda: BoundedPolicyAdapter(
            max_consecutive_decode=4, mixed_prefill_budget=2048
        ),
        "candidate_b_d8_p4096": lambda: BoundedPolicyAdapter(
            max_consecutive_decode=8, mixed_prefill_budget=4096
        ),
    }
    return {
        "harness": "no-weight phase arbitration model",
        "release_prefill_budget": RELEASE_PREFILL_BUDGET,
        "policies": {
            name: {scenario.name: simulate(factory(), scenario) for scenario in SCENARIOS}
            for name, factory in policies.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

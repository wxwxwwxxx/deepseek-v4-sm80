from __future__ import annotations

from types import SimpleNamespace

from minisgl.scheduler.phase_policy import MixedPhaseFairPolicy
from minisgl.scheduler.scheduler import Scheduler


def _run(
    policy: MixedPhaseFairPolicy,
    states: list[tuple[bool, bool]],
) -> list[tuple[str | None, int]]:
    result = []
    for prefill, decode in states:
        decision = policy.choose(prefill_runnable=prefill, decode_runnable=decode)
        result.append((decision.phase, decision.prefill_budget))
        policy.record_scheduled(
            decision.phase,
            prefill_runnable=prefill,
            decode_runnable=decode,
        )
    return result


def _policy(**kwargs) -> MixedPhaseFairPolicy:
    return MixedPhaseFairPolicy(
        isolated_prefill_budget=8192,
        mixed_prefill_budget=2048,
        max_consecutive_decode=4,
        **kwargs,
    )


def test_64_chunk_ladder_has_bounded_progress_in_both_directions() -> None:
    policy = _policy()
    decisions = _run(policy, [(True, True)] * 80)
    phases = [phase for phase, _ in decisions]
    assert "".join(phase[0].upper() for phase in phases[:15]) == "DDDDPDDDDPDDDDP"
    assert phases.count("prefill") == 16
    assert phases.count("decode") == 64
    assert max(
        sum(1 for item in group if item == "decode")
        for group in _groups(phases)
    ) == 4
    assert all(budget == 2048 for phase, budget in decisions if phase == "prefill")


def test_at_most_one_prefill_between_decode_opportunities() -> None:
    phases = [phase for phase, _ in _run(_policy(), [(True, True)] * 40)]
    assert all(not (a == b == "prefill") for a, b in zip(phases, phases[1:]))


def test_only_phase_controls_keep_release_behavior() -> None:
    policy = _policy()
    assert _run(policy, [(True, False)] * 4) == [("prefill", 8192)] * 4
    assert _run(policy, [(False, True)] * 4) == [("decode", 0)] * 4


def test_decode_enqueue_during_prefill_gets_first_conflict_slot() -> None:
    policy = _policy()
    decisions = _run(policy, [(True, False)] * 8 + [(True, True)] * 6)
    assert [phase for phase, _ in decisions[8:]] == [
        "decode",
        "decode",
        "decode",
        "decode",
        "prefill",
        "decode",
    ]


def test_queue_empty_resets_conflict_state() -> None:
    policy = _policy()
    _run(policy, [(True, True)] * 4)
    assert policy.snapshot()["consecutive_mixed_decode"] == 4
    _run(policy, [(False, True)])
    assert policy.snapshot()["mixed_active"] is False
    assert _run(policy, [(True, True)])[0][0] == "decode"


def test_abort_and_terminal_retirement_transitions_reset_state() -> None:
    policy = _policy()
    _run(policy, [(True, True)] * 4)
    # Pending prefill aborts: only decode remains.
    _run(policy, [(False, True)])
    assert policy.snapshot()["consecutive_mixed_decode"] == 0
    # Decode retires: isolated prefill must run at the full release budget.
    assert _run(policy, [(True, False)]) == [("prefill", 8192)]


def test_cache_hit_sized_prefill_consumes_one_bounded_opportunity() -> None:
    policy = _policy()
    phases = [phase for phase, _ in _run(policy, [(True, True)] * 6)]
    assert phases == ["decode", "decode", "decode", "decode", "prefill", "decode"]


def test_unconstructed_batch_does_not_advance_as_prefill() -> None:
    policy = _policy()
    _run(policy, [(True, True)] * 4)
    decision = policy.choose(prefill_runnable=True, decode_runnable=True)
    assert decision.phase == "prefill"
    # Scheduler's allocation fallback actually ran decode.
    policy.record_scheduled(
        "decode", prefill_runnable=True, decode_runnable=True
    )
    assert policy.choose(prefill_runnable=True, decode_runnable=True).phase == "prefill"


def test_candidate_first_prefill_is_still_bounded() -> None:
    policy = _policy(first_conflict="prefill")
    phases = [phase for phase, _ in _run(policy, [(True, True)] * 7)]
    assert phases == ["prefill", "decode", "decode", "decode", "decode", "prefill", "decode"]


def _groups(items: list[str | None]) -> list[list[str | None]]:
    groups: list[list[str | None]] = []
    for item in items:
        if not groups or groups[-1][0] != item:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


class _Manager:
    def __init__(self, phase: str, *, runnable: bool, construct: bool = True):
        self.phase = phase
        self.runnable = runnable
        self.construct = construct
        self.budgets: list[int] = []

    def schedule_next_batch(self, budget: int | None = None):
        if budget is not None:
            self.budgets.append(budget)
        if not self.runnable or not self.construct:
            return None
        return SimpleNamespace(phase=self.phase)


def _scheduler(*, prefill: _Manager, decode: _Manager) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 8192
    scheduler.prefill_manager = prefill
    scheduler.decode_manager = decode
    scheduler.phase_policy = _policy()
    scheduler._prepare_batch = lambda batch: batch
    return scheduler


def test_scheduler_integration_preserves_isolated_budget_and_order() -> None:
    prefill = _Manager("prefill", runnable=True)
    decode = _Manager("decode", runnable=False)
    scheduler = _scheduler(prefill=prefill, decode=decode)
    assert scheduler._schedule_next_batch().phase == "prefill"
    assert prefill.budgets == [8192]


def test_scheduler_integration_uses_bounded_mixed_budget() -> None:
    prefill = _Manager("prefill", runnable=True)
    decode = _Manager("decode", runnable=True)
    scheduler = _scheduler(prefill=prefill, decode=decode)
    phases = [scheduler._schedule_next_batch().phase for _ in range(5)]
    assert phases == ["decode", "decode", "decode", "decode", "prefill"]
    assert prefill.budgets == [2048]


def test_scheduler_integration_falls_back_when_forced_prefill_cannot_allocate() -> None:
    prefill = _Manager("prefill", runnable=True, construct=False)
    decode = _Manager("decode", runnable=True)
    scheduler = _scheduler(prefill=prefill, decode=decode)
    for _ in range(4):
        assert scheduler._schedule_next_batch().phase == "decode"
    assert scheduler._schedule_next_batch().phase == "decode"
    assert prefill.budgets == [2048]
    assert scheduler.phase_policy.snapshot()["consecutive_mixed_decode"] == 5

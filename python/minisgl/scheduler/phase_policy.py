from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Phase = Literal["prefill", "decode"]

# Release-owned policy.  The isolated prefill budget remains SchedulerConfig's
# max_extend_tokens; only a genuine prefill/decode conflict uses this budget.
DSV4_MIXED_PREFILL_BUDGET = 2048
DSV4_MAX_CONSECUTIVE_DECODE = 4


@dataclass(frozen=True)
class PhaseDecision:
    phase: Phase | None
    prefill_budget: int = 0
    mixed: bool = False


class MixedPhaseFairPolicy:
    """Bounded arbitration for Mini's separate prefill/decode forwards.

    State advances only after Scheduler confirms that a batch was actually
    constructed.  This keeps allocation failures and queue transitions from
    consuming a fairness opportunity that never ran.
    """

    def __init__(
        self,
        *,
        isolated_prefill_budget: int,
        mixed_prefill_budget: int = DSV4_MIXED_PREFILL_BUDGET,
        max_consecutive_decode: int = DSV4_MAX_CONSECUTIVE_DECODE,
        first_conflict: Phase = "decode",
    ) -> None:
        if isolated_prefill_budget <= 0:
            raise ValueError("isolated_prefill_budget must be positive")
        if mixed_prefill_budget <= 0:
            raise ValueError("mixed_prefill_budget must be positive")
        if max_consecutive_decode <= 0:
            raise ValueError("max_consecutive_decode must be positive")
        if first_conflict not in {"prefill", "decode"}:
            raise ValueError("first_conflict must be 'prefill' or 'decode'")
        self.isolated_prefill_budget = int(isolated_prefill_budget)
        self.mixed_prefill_budget = min(
            int(mixed_prefill_budget), self.isolated_prefill_budget
        )
        self.max_consecutive_decode = int(max_consecutive_decode)
        self.first_conflict = first_conflict
        self._mixed_active = False
        self._last_mixed_phase: Phase | None = None
        self._consecutive_mixed_decode = 0

    def choose(self, *, prefill_runnable: bool, decode_runnable: bool) -> PhaseDecision:
        if not prefill_runnable and not decode_runnable:
            return PhaseDecision(None)
        if prefill_runnable and not decode_runnable:
            return PhaseDecision("prefill", self.isolated_prefill_budget)
        if decode_runnable and not prefill_runnable:
            return PhaseDecision("decode")

        if not self._mixed_active:
            phase = self.first_conflict
        elif self._last_mixed_phase == "prefill":
            # Never allow two prefill forwards between protected decode slots.
            phase = "decode"
        elif self._consecutive_mixed_decode >= self.max_consecutive_decode:
            phase = "prefill"
        else:
            phase = "decode"
        return PhaseDecision(
            phase,
            self.mixed_prefill_budget if phase == "prefill" else 0,
            mixed=True,
        )

    def record_scheduled(
        self,
        phase: Phase | None,
        *,
        prefill_runnable: bool,
        decode_runnable: bool,
    ) -> None:
        mixed = prefill_runnable and decode_runnable
        if phase is None or not mixed:
            self.reset_mixed_state()
            return
        self._mixed_active = True
        self._last_mixed_phase = phase
        if phase == "decode":
            self._consecutive_mixed_decode += 1
        else:
            self._consecutive_mixed_decode = 0

    def reset_mixed_state(self) -> None:
        self._mixed_active = False
        self._last_mixed_phase = None
        self._consecutive_mixed_decode = 0

    def snapshot(self) -> dict[str, int | str | bool | None]:
        return {
            "mixed_active": self._mixed_active,
            "last_mixed_phase": self._last_mixed_phase,
            "consecutive_mixed_decode": self._consecutive_mixed_decode,
            "isolated_prefill_budget": self.isolated_prefill_budget,
            "mixed_prefill_budget": self.mixed_prefill_budget,
            "max_consecutive_decode": self.max_consecutive_decode,
            "first_conflict": self.first_conflict,
        }

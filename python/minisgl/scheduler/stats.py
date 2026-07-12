from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerStatsSnapshot:
    prompt_throughput: float
    generation_throughput: float


class SchedulerStatsTracker:
    def __init__(self, interval: float) -> None:
        if interval <= 0:
            raise ValueError("stats_log_interval must be positive")
        self.interval = float(interval)
        self._active = False
        self._last_time = time.monotonic()
        self._prompt_tokens = 0
        self._generation_tokens = 0

    def record(
        self,
        *,
        prompt_tokens: int = 0,
        generation_tokens: int = 0,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        if not self._active:
            self._active = True
            self._last_time = now
        self._prompt_tokens += int(prompt_tokens)
        self._generation_tokens += int(generation_tokens)

    def maybe_snapshot(self, now: float | None = None) -> SchedulerStatsSnapshot | None:
        if not self._active:
            return None
        now = time.monotonic() if now is None else now
        elapsed = now - self._last_time
        if elapsed < self.interval:
            return None
        snapshot = SchedulerStatsSnapshot(
            prompt_throughput=self._prompt_tokens / elapsed,
            generation_throughput=self._generation_tokens / elapsed,
        )
        self._last_time = now
        self._prompt_tokens = 0
        self._generation_tokens = 0
        return snapshot

    def reset_idle(self, now: float | None = None) -> None:
        self._active = False
        self._last_time = time.monotonic() if now is None else now
        self._prompt_tokens = 0
        self._generation_tokens = 0

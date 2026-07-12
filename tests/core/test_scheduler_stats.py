from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch
from minisgl.scheduler import scheduler as scheduler_module
from minisgl.scheduler.scheduler import Scheduler, _processed_prompt_tokens
from minisgl.scheduler.stats import SchedulerStatsTracker


def test_scheduler_stats_reports_interval_throughput():
    tracker = SchedulerStatsTracker(interval=10.0)
    tracker.record(prompt_tokens=1000, generation_tokens=200, now=5.0)

    assert tracker.maybe_snapshot(now=14.9) is None
    snapshot = tracker.maybe_snapshot(now=15.0)

    assert snapshot is not None
    assert snapshot.prompt_throughput == pytest.approx(100.0)
    assert snapshot.generation_throughput == pytest.approx(20.0)


def test_scheduler_stats_excludes_idle_time_between_workloads():
    tracker = SchedulerStatsTracker(interval=10.0)
    tracker.record(prompt_tokens=100, now=0.0)
    tracker.reset_idle(now=1.0)
    tracker.record(generation_tokens=100, now=100.0)

    snapshot = tracker.maybe_snapshot(now=110.0)

    assert snapshot is not None
    assert snapshot.prompt_throughput == 0.0
    assert snapshot.generation_throughput == pytest.approx(10.0)


def test_scheduler_stats_interval_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        SchedulerStatsTracker(interval=0.0)


def test_prefill_stats_use_forward_input_shape_after_request_state_advance():
    batch = SimpleNamespace(
        is_prefill=True,
        input_ids=torch.empty(4 * 4096, dtype=torch.int32),
        reqs=[SimpleNamespace(extend_len=1) for _ in range(4)],
    )

    assert _processed_prompt_tokens(batch) == 4 * 4096


def test_scheduler_stats_log_uses_cpu_owned_state(monkeypatch):
    scheduler = object.__new__(Scheduler)
    scheduler._stats_tracker = SchedulerStatsTracker(interval=10.0)
    scheduler._stats_tracker.record(
        prompt_tokens=1000,
        generation_tokens=200,
        now=time.monotonic() - 10.0,
    )
    scheduler.cache_manager = SimpleNamespace(num_pages=100, free_slots=[None] * 25)
    scheduler.decode_manager = SimpleNamespace(running_reqs={1, 2})
    scheduler.prefill_manager = SimpleNamespace(pending_list=[1, 2, 3])
    messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info_rank0",
        lambda message, *args, **kwargs: messages.append(message),
    )

    scheduler._maybe_log_stats()

    assert "running=2" in messages[0]
    assert "waiting=3" in messages[0]
    assert "KV=75.0% (75/100 pages)" in messages[0]

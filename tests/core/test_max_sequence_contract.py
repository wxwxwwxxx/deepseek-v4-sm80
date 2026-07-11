from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.engine.engine import _resolve_effective_max_seq_len
from minisgl.llm.llm import LLM, RequestStatus
from minisgl.message import UserMsg
from minisgl.scheduler import scheduler as scheduler_module
from minisgl.scheduler.scheduler import (
    Scheduler,
    decide_max_sequence_admission,
    validate_model_position_bound,
)


@pytest.mark.parametrize(
    ("input_len", "output_len", "accepted", "admitted"),
    [
        (4, 3, True, 3),
        (4, 4, True, 4),
        (4, 5, True, 4),
        (8, 1, False, 0),
        (9, 0, False, 0),
    ],
)
def test_max_seq_len_is_maximum_total_sequence_length(
    input_len: int, output_len: int, accepted: bool, admitted: int
):
    decision = decide_max_sequence_admission(
        input_len=input_len,
        requested_output_len=output_len,
        max_seq_len=8,
    )

    assert decision.accepted is accepted
    assert decision.admitted_output_len == admitted
    if accepted:
        assert input_len + decision.admitted_output_len <= 8
    assert (decision.rejection_reason is None) is accepted


def test_offline_llm_records_scheduler_rejection_as_observable_result():
    llm = object.__new__(LLM)
    llm.status_map = {
        7: RequestStatus(
            uid=7,
            input_ids=list(range(8)),
            output_ids=[],
            requested_output_len=1,
        )
    }
    decision = decide_max_sequence_admission(
        input_len=8,
        requested_output_len=1,
        max_seq_len=8,
    )

    llm._record_max_sequence_admission(7, decision)

    assert llm.status_map[7].finish_reason == "length_rejected"
    assert llm.status_map[7].admitted_output_len == 0
    assert "no model position remains" in (llm.status_map[7].error or "")


def test_scheduler_returns_terminal_rejection_instead_of_silent_drop(monkeypatch):
    monkeypatch.setattr(scheduler_module.logger, "debug_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler_module.logger, "warning_rank0", lambda *args, **kwargs: None)
    scheduler = object.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(max_seq_len=8)
    scheduler.eos_token_id = 2
    replies = []
    scheduler.send_result = replies.extend
    scheduler._record_max_sequence_admission = lambda uid, admission: None
    msg = UserMsg(
        uid=9,
        input_ids=torch.arange(8, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=1),
    )

    scheduler._process_one_msg(msg)

    assert len(replies) == 1
    assert replies[0].uid == 9
    assert replies[0].finished is True
    assert replies[0].finish_reason == "length_rejected"
    assert "no model position remains" in (replies[0].error or "")


def test_model_position_must_be_strictly_below_rope_cache_length():
    assert validate_model_position_bound(max_device_len=8, rope_cache_len=8) == 7
    with pytest.raises(RuntimeError, match="position=8, rope_cache_len=8"):
        validate_model_position_bound(max_device_len=9, rope_cache_len=8)


def test_materialized_rope_clamps_override_but_dsv4_on_the_fly_tracks_engine():
    assert _resolve_effective_max_seq_len(
        requested_max_seq_len=32,
        kv_capacity_tokens=64,
        model_rotary_max=16,
        rope_is_on_the_fly=False,
    ) == (16, 16, "materialized")
    assert _resolve_effective_max_seq_len(
        requested_max_seq_len=32,
        kv_capacity_tokens=24,
        model_rotary_max=16,
        rope_is_on_the_fly=True,
    ) == (24, 24, "on_the_fly")

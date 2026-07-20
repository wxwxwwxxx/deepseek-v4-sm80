from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from minisgl.core import Batch, Req, RequestLifecycleState, SamplingParams
from minisgl.message import AbortBackendMsg
from minisgl.reasoning import ReasoningState
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.scheduler import ForwardInput, IssuedReqRef, Scheduler


class FakeEvent:
    def __init__(self) -> None:
        self.synchronize_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1


class FakeStats:
    def __init__(self) -> None:
        self.records: list[dict[str, int]] = []

    def record(self, **kwargs: int) -> None:
        self.records.append(kwargs)


class FakeDecodeManager:
    def __init__(self, reqs: list[Req]) -> None:
        self.running_reqs = set(reqs)
        self.removed: list[int] = []

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)
        self.removed.append(req.uid)

    def abort_req(self, uid: int) -> Req | None:
        for req in tuple(self.running_reqs):
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None


class FakeCacheManager:
    def __init__(self) -> None:
        self.cached: list[tuple[int, bool]] = []
        self.swa_releases: list[int] = []

    @contextmanager
    def lazy_free_region(self):
        yield

    def cache_req(self, req: Req, *, finished: bool) -> None:
        self.cached.append((req.uid, finished))

    def release_active_dsv4_swa_out_of_window(self, req: Req) -> None:
        self.swa_releases.append(req.uid)


class FakePrefillManager:
    def abort_req(self, uid: int) -> Req | None:
        del uid
        return None


def make_req(
    uid: int,
    *,
    output_len: int = 8,
    reasoning_effort: str | None = None,
    generation_id: int | None = None,
) -> Req:
    req = Req(
        input_ids=torch.tensor([90], dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=object(),
        reasoning_effort=reasoning_effort,
    )
    req.lifecycle.generation_id = uid if generation_id is None else generation_id
    return req


def make_scheduler(reqs: list[Req]) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.eos_token_id = 1
    scheduler.think_end_token_id = 3
    scheduler.reasoning_sampler_contract_enabled = True
    scheduler.decode_manager = FakeDecodeManager(reqs)
    scheduler.prefill_manager = FakePrefillManager()
    scheduler.cache_manager = FakeCacheManager()
    scheduler.table_manager = SimpleNamespace(
        freed=[], free=lambda idx: scheduler.table_manager.freed.append(idx)
    )
    scheduler.engine = SimpleNamespace(
        released_events=[],
        release_copy_done_event=lambda event: scheduler.engine.released_events.append(event),
    )
    scheduler._stats_tracker = FakeStats()
    scheduler._retirement_owners = {}
    scheduler._live_reqs = {req.uid: req for req in reqs}
    scheduler.sent: list[list[object]] = []
    scheduler.send_result = lambda reply: scheduler.sent.append(list(reply))
    return scheduler


def issue(req: Req) -> IssuedReqRef:
    epoch = req.lifecycle.issue()
    req.complete_one()
    return IssuedReqRef(req, req.lifecycle.generation_id, epoch)


def completion(
    refs: list[IssuedReqRef], tokens: list[int], *, phase: str = "decode"
) -> tuple[tuple[ForwardInput, tuple[None, torch.Tensor, FakeEvent]], FakeEvent]:
    batch = Batch(reqs=[ref.req for ref in refs], phase=phase)
    batch.input_ids = torch.tensor([90] * len(refs), dtype=torch.int32)
    forward_input = ForwardInput(
        batch=batch,
        sample_args=None,
        input_tuple=(torch.empty(0), torch.empty(0)),
        write_tuple=(torch.empty(0), torch.empty(0)),
        issued_refs=tuple(refs),
    )
    event = FakeEvent()
    return (forward_input, (None, torch.tensor(tokens, dtype=torch.int32), event)), event


def process(
    scheduler: Scheduler,
    refs: list[IssuedReqRef],
    tokens: list[int],
    *,
    phase: str = "decode",
) -> FakeEvent:
    data, event = completion(refs, tokens, phase=phase)
    scheduler._process_last_data(data)
    return event


def flattened_replies(scheduler: Scheduler) -> list[object]:
    return [msg for batch in scheduler.sent for msg in batch]


def test_eos_discards_one_later_completion_and_retires_after_it() -> None:
    req = make_req(1, reasoning_effort="high")
    scheduler = make_scheduler([req])
    eos_ref = issue(req)
    stale_ref = issue(req)

    process(scheduler, [eos_ref], [1])
    assert req.lifecycle.state is RequestLifecycleState.TERMINAL_PENDING_RETIRE
    assert req.lifecycle.outstanding_epochs == {stale_ref.issue_epoch}
    assert scheduler.table_manager.freed == []
    assert [msg.next_token for msg in flattened_replies(scheduler)] == [1]

    reasoning_after_terminal = req.reasoning_state
    host_after_terminal = req.input_ids.clone()
    process(scheduler, [stale_ref], [3])
    assert req.lifecycle.state is RequestLifecycleState.RETIRED
    assert req.lifecycle.terminal_finish_reason == "stop"
    assert torch.equal(req.input_ids, host_after_terminal)
    assert req.reasoning_state is reasoning_after_terminal is ReasoningState.THINKING
    assert [msg.next_token for msg in flattened_replies(scheduler)] == [1]
    assert [msg.completion_tokens for msg in flattened_replies(scheduler)] == [1]
    assert scheduler.table_manager.freed == [req.table_idx]
    assert scheduler.cache_manager.cached == [(req.uid, True)]
    assert scheduler._stats_tracker.records == [{"generation_tokens": 1}]


def test_length_completion_discards_a_later_arbitrary_depth_epoch() -> None:
    req = make_req(2, output_len=2)
    scheduler = make_scheduler([req])
    first, terminal, stale = issue(req), issue(req), issue(req)

    process(scheduler, [first], [7])
    assert req.lifecycle.state is RequestLifecycleState.ACTIVE
    process(scheduler, [terminal], [8])
    assert req.lifecycle.state is RequestLifecycleState.TERMINAL_PENDING_RETIRE
    replies = flattened_replies(scheduler)
    assert [(msg.next_token, msg.finished, msg.finish_reason) for msg in replies] == [
        (7, False, None),
        (8, True, "length"),
    ]
    process(scheduler, [stale], [9])
    assert req.input_ids.tolist() == [90, 7, 8]
    assert req.lifecycle.state is RequestLifecycleState.RETIRED
    assert req.lifecycle.terminal_finish_reason == "length"
    assert scheduler.table_manager.freed == [req.table_idx]


def test_abort_discards_inflight_without_frontend_or_usage_mutation(monkeypatch) -> None:
    monkeypatch.setattr("minisgl.scheduler.scheduler.logger.debug_rank0", lambda *args: None)
    req = make_req(3, reasoning_effort="high")
    scheduler = make_scheduler([req])
    stale_ref = issue(req)

    scheduler._process_one_msg(AbortBackendMsg(uid=req.uid))
    assert req.lifecycle.state is RequestLifecycleState.TERMINAL_PENDING_RETIRE
    assert scheduler.table_manager.freed == []
    process(scheduler, [stale_ref], [3])
    assert req.lifecycle.state is RequestLifecycleState.RETIRED
    assert req.lifecycle.terminal_finish_reason == "abort"
    assert req.input_ids.tolist() == [90]
    assert req.reasoning_state is ReasoningState.THINKING
    assert flattened_replies(scheduler) == []
    assert scheduler._stats_tracker.records == []
    assert scheduler.table_manager.freed == [req.table_idx]


def test_mixed_batch_terminal_row_is_discarded_while_companion_continues() -> None:
    terminal_req, active_req = make_req(4), make_req(5)
    scheduler = make_scheduler([terminal_req, active_req])
    first = [issue(terminal_req), issue(active_req)]
    second = [issue(terminal_req), issue(active_req)]

    process(scheduler, first, [1, 40])
    process(scheduler, second, [41, 42])
    assert terminal_req.input_ids.tolist() == [90, 1]
    assert active_req.input_ids.tolist() == [90, 40, 42]
    assert terminal_req.lifecycle.state is RequestLifecycleState.RETIRED
    assert active_req.lifecycle.state is RequestLifecycleState.ACTIVE
    assert [msg.next_token for msg in flattened_replies(scheduler)] == [1, 40, 42]
    assert scheduler.table_manager.freed == [terminal_req.table_idx]


def test_adjacent_terminal_requests_each_wait_for_their_last_epoch() -> None:
    first_req, second_req = make_req(6), make_req(7)
    scheduler = make_scheduler([first_req, second_req])
    batch1 = [issue(first_req), issue(second_req)]
    batch2 = [issue(first_req), issue(second_req)]

    process(scheduler, batch1, [1, 50])
    batch3_second = issue(second_req)
    process(scheduler, batch2, [51, 1])
    assert first_req.lifecycle.state is RequestLifecycleState.RETIRED
    assert second_req.lifecycle.state is RequestLifecycleState.TERMINAL_PENDING_RETIRE
    assert scheduler.table_manager.freed == [first_req.table_idx]

    process(scheduler, [batch3_second], [52])
    assert second_req.lifecycle.state is RequestLifecycleState.RETIRED
    assert scheduler.table_manager.freed == [first_req.table_idx, second_req.table_idx]
    assert [msg.next_token for msg in flattened_replies(scheduler)] == [1, 50, 1]


def test_normal_non_overlap_terminal_retires_immediately_and_once() -> None:
    req = make_req(8)
    scheduler = make_scheduler([req])
    only_ref = issue(req)
    event = process(scheduler, [only_ref], [1], phase="prefill")

    assert event.synchronize_count == 1
    assert req.lifecycle.state is RequestLifecycleState.RETIRED
    assert req.lifecycle.resources_released
    assert scheduler.table_manager.freed == [req.table_idx]
    assert scheduler.cache_manager.cached == [(req.uid, True)]
    assert len(flattened_replies(scheduler)) == 1
    scheduler._maybe_release_terminal_resources(req.lifecycle)
    assert scheduler.table_manager.freed == [req.table_idx]
    assert scheduler.cache_manager.cached == [(req.uid, True)]


def test_chunked_incarnations_share_generation_and_owner_count() -> None:
    first = make_req(9, generation_id=99)
    second = ChunkedReq(
        input_ids=first.input_ids,
        table_idx=first.table_idx,
        cached_len=first.cached_len,
        output_len=first.output_len,
        uid=first.uid,
        sampling_params=first.sampling_params,
        cache_handle=first.cache_handle,
        lifecycle=first.lifecycle,
    )
    first_ref, second_ref = issue(first), issue(second)
    assert first_ref.generation_id == second_ref.generation_id == 99
    assert first.lifecycle.outstanding_epochs == {
        first_ref.issue_epoch,
        second_ref.issue_epoch,
    }

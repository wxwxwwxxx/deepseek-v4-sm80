from __future__ import annotations

import asyncio
import multiprocessing as mp

import pytest
from minisgl.message import AbortMsg, UserReply
from minisgl.server.api_server import BackendUnavailableError, FrontendManager
from minisgl.server.supervisor import ProcessSpec, ServerProcessSupervisor


class ControlledQueue:
    def __init__(self) -> None:
        self.failure_event = asyncio.Event()
        self.raise_on_get = False
        self.raise_on_put = False
        self.sent = []
        self.stopped = False

    async def get(self):
        await self.failure_event.wait()
        if self.raise_on_get:
            raise RuntimeError("injected listener loss")
        await asyncio.Future()

    async def put(self, msg) -> None:
        if self.raise_on_put:
            raise RuntimeError("injected send loss")
        self.sent.append(msg)

    def stop(self) -> None:
        self.stopped = True


def _ack_then_wait(role: str, ack_queue, release) -> None:
    ack_queue.put((role, f"{role} ready"))
    release.wait(10)


def _new_process_supervisor() -> tuple[ServerProcessSupervisor, object]:
    context = mp.get_context("fork")
    release = context.Event()
    supervisor = ServerProcessSupervisor(
        context=context,
        startup_timeout=1,
        ack_poll_interval=0.01,
        shutdown_grace=0.02,
        termination_grace=0.5,
    )
    supervisor.start(
        [
            ProcessSpec(
                role="runtime-worker",
                target=_ack_then_wait,
                args=("runtime-worker", supervisor.ack_queue, release),
            )
        ]
    )
    return supervisor, release


def _new_frontend(queue: ControlledQueue) -> FrontendManager:
    return FrontendManager(
        config=object(),
        send_tokenizer=queue,
        recv_tokenizer=queue,
        health_poll_interval=0.01,
    )


def test_runtime_worker_exit_wakes_waiter_rejects_new_work_and_reaps_child() -> None:
    supervisor, _ = _new_process_supervisor()
    queue = ControlledQueue()
    frontend = _new_frontend(queue)
    frontend.supervisor = supervisor

    async def run() -> None:
        await frontend.start_runtime_tasks()
        assert frontend.listener_task is not None
        assert frontend.health_task is not None

        uid = frontend.new_user()
        replies = frontend.wait_for_ack(uid)
        waiter = asyncio.create_task(anext(replies))
        supervisor.processes["runtime-worker"].terminate()
        supervisor.processes["runtime-worker"].join(timeout=1)

        with pytest.raises(BackendUnavailableError, match="runtime-worker.*runtime"):
            await asyncio.wait_for(waiter, timeout=1)
        await replies.aclose()
        assert uid not in frontend.ack_map
        assert uid not in frontend.event_map
        with pytest.raises(BackendUnavailableError, match="runtime-worker"):
            frontend.new_user()

        assert frontend.failure_cleanup_task is not None
        await asyncio.wait_for(frontend.failure_cleanup_task, timeout=1)
        await frontend.shutdown_async()
        await frontend.shutdown_async()

    asyncio.run(run())
    assert supervisor.shutdown_complete
    assert supervisor.live_roles() == []
    assert queue.stopped


def test_listener_loss_is_single_terminal_state_for_stream_and_non_stream_waiters() -> None:
    queue = ControlledQueue()
    queue.raise_on_get = True
    frontend = _new_frontend(queue)

    async def run() -> None:
        await frontend.start_runtime_tasks()
        non_stream_uid = frontend.new_user()
        stream_uid = frontend.new_user()
        non_stream_replies = frontend.wait_for_ack(non_stream_uid)
        non_stream = asyncio.create_task(anext(non_stream_replies))
        stream = asyncio.create_task(
            _collect_stream(
                frontend.stream_chat_completions(
                    stream_uid,
                    completion_id="chatcmpl-loss",
                    created=1,
                    model="test-model",
                    include_usage=False,
                )
            )
        )

        queue.failure_event.set()
        with pytest.raises(BackendUnavailableError, match="injected listener loss"):
            await asyncio.wait_for(non_stream, timeout=1)
        await non_stream_replies.aclose()
        chunks = await asyncio.wait_for(stream, timeout=1)
        assert sum(b'"code": "backend_unavailable"' in chunk for chunk in chunks) == 1
        assert chunks[-1] == b"data: [DONE]\n\n"
        assert frontend.ack_map == {}
        assert frontend.event_map == {}
        terminal = frontend.terminal_error
        frontend._mark_backend_unavailable("second failure must not replace the first")
        assert frontend.terminal_error is terminal
        with pytest.raises(BackendUnavailableError, match="injected listener loss"):
            frontend.new_user()
        await frontend.shutdown_async()

    asyncio.run(run())
    assert queue.stopped


def test_send_loss_rejects_submission_without_leaking_new_request_maps() -> None:
    queue = ControlledQueue()
    queue.raise_on_put = True
    frontend = _new_frontend(queue)

    async def run() -> None:
        uid = frontend.new_user()
        with pytest.raises(BackendUnavailableError, match="injected send loss"):
            await frontend.send_one(AbortMsg(uid=uid))
        assert frontend.ack_map == {}
        assert frontend.event_map == {}
        assert frontend.request_uids == set()
        await frontend.shutdown_async()

    asyncio.run(run())
    assert queue.stopped


async def _collect_stream(generator) -> list[bytes]:
    return [chunk async for chunk in generator]


def test_streaming_and_non_stream_cancellation_send_at_most_one_abort_each() -> None:
    queue = ControlledQueue()
    frontend = _new_frontend(queue)

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def cancel(task: asyncio.Task) -> None:
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def run() -> None:
        await frontend.start_runtime_tasks()

        before_first_uid = frontend.new_user()
        before_first_stream = frontend.stream_chat_completions(
            before_first_uid,
            completion_id="chatcmpl-before",
            created=1,
            model="test-model",
            include_usage=False,
        )
        before_first_wrapped = frontend.stream_with_cancellation(
            before_first_stream, ConnectedRequest(), before_first_uid
        )
        await cancel(asyncio.create_task(anext(before_first_wrapped)))
        frontend._schedule_abort(before_first_uid)

        during_uid = frontend.new_user()
        during_stream = frontend.stream_chat_completions(
            during_uid,
            completion_id="chatcmpl-during",
            created=1,
            model="test-model",
            include_usage=False,
        )
        during_wrapped = frontend.stream_with_cancellation(
            during_stream, ConnectedRequest(), during_uid
        )
        first = asyncio.create_task(anext(during_wrapped))
        await asyncio.sleep(0)
        frontend.ack_map[during_uid].append(
            UserReply(uid=during_uid, incremental_output="partial", finished=False)
        )
        frontend.event_map[during_uid].set()
        assert b"partial" in await asyncio.wait_for(first, timeout=1)
        await cancel(asyncio.create_task(anext(during_wrapped)))
        frontend._schedule_abort(during_uid)

        non_stream_uid = frontend.new_user()
        non_stream_replies = frontend.wait_for_ack(non_stream_uid)
        await cancel(asyncio.create_task(anext(non_stream_replies)))
        frontend._schedule_abort(non_stream_uid)

        await asyncio.sleep(0.12)
        abort_uids = [msg.uid for msg in queue.sent if isinstance(msg, AbortMsg)]
        assert abort_uids.count(before_first_uid) == 1
        assert abort_uids.count(during_uid) == 1
        assert abort_uids.count(non_stream_uid) == 1
        assert frontend.ack_map == {}
        assert frontend.event_map == {}
        await frontend.shutdown_async()

    asyncio.run(run())

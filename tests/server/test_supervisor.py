from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess

import pytest
from minisgl.server.supervisor import ProcessSpec, ServerProcessSupervisor


def _context():
    # The supervisor is context-agnostic; fork keeps these no-GPU lifecycle tests fast.
    return mp.get_context("fork")


def _ack_then_wait(role: str, ack_queue, release) -> None:
    ack_queue.put((role, f"{role} is ready"))
    release.wait(10)


def _wait_without_ack(release) -> None:
    release.wait(10)


def _exit_before_ack(exitcode: int) -> None:
    raise SystemExit(exitcode)


def _assert_all_reaped(supervisor: ServerProcessSupervisor) -> None:
    assert supervisor.live_roles() == []
    for process in supervisor.processes.values():
        assert not process.is_alive()
        assert process.exitcode is not None


def _new_supervisor(
    *,
    startup_timeout: float = 1.0,
    process_factory: Callable[..., BaseProcess] | None = None,
) -> ServerProcessSupervisor:
    return ServerProcessSupervisor(
        context=_context(),
        startup_timeout=startup_timeout,
        ack_poll_interval=0.02,
        shutdown_grace=0.05,
        termination_grace=0.5,
        process_factory=process_factory,
    )


def test_all_workers_acknowledge_and_repeated_shutdown_reaps_owned_children() -> None:
    context = _context()
    release = context.Event()
    supervisor = _new_supervisor()
    roles = ["scheduler-0", "tokenizer-0", "detokenizer-0"]
    specs = [
        ProcessSpec(
            role=role,
            target=_ack_then_wait,
            args=(role, supervisor.ack_queue, release),
        )
        for role in roles
    ]

    assert supervisor.start(specs) is supervisor
    assert supervisor.startup_complete
    assert supervisor.ready_roles == set(roles)
    assert set(supervisor.owned_pids()) == set(roles)
    assert set(supervisor.live_roles()) == set(roles)

    supervisor.set_graceful_shutdown(release.set)
    supervisor.shutdown()
    supervisor.shutdown()
    assert supervisor.shutdown_complete
    _assert_all_reaped(supervisor)


def test_worker_exit_before_ack_is_named_and_reaps_siblings_immediately() -> None:
    context = _context()
    release = context.Event()
    supervisor = _new_supervisor(startup_timeout=5.0)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="failed-scheduler.*code 23.*startup"):
        supervisor.start(
            [
                ProcessSpec(
                    role="waiting-tokenizer",
                    target=_wait_without_ack,
                    args=(release,),
                ),
                ProcessSpec(
                    role="failed-scheduler",
                    target=_exit_before_ack,
                    args=(23,),
                ),
            ]
        )

    assert time.monotonic() - started < 2.0
    _assert_all_reaped(supervisor)


def test_missing_ack_hits_injected_deadline_and_reaps_every_child() -> None:
    context = _context()
    release = context.Event()
    supervisor = _new_supervisor(startup_timeout=0.15)

    with pytest.raises(TimeoutError, match="never-ready"):
        supervisor.start(
            [
                ProcessSpec(
                    role="never-ready",
                    target=_wait_without_ack,
                    args=(release,),
                ),
                ProcessSpec(
                    role="also-waiting",
                    target=_wait_without_ack,
                    args=(release,),
                ),
            ]
        )

    _assert_all_reaped(supervisor)


def test_partial_spawn_failure_reaps_already_started_child() -> None:
    context = _context()
    release = context.Event()
    process_count = 0

    def fail_second_process(**kwargs) -> BaseProcess:
        nonlocal process_count
        process_count += 1
        if process_count == 2:
            raise OSError("injected spawn failure")
        return context.Process(**kwargs)

    supervisor = _new_supervisor(process_factory=fail_second_process)
    with pytest.raises(OSError, match="injected spawn failure"):
        supervisor.start(
            [
                ProcessSpec(
                    role="started-child",
                    target=_wait_without_ack,
                    args=(release,),
                ),
                ProcessSpec(
                    role="spawn-fails",
                    target=_wait_without_ack,
                    args=(release,),
                ),
            ]
        )

    assert set(supervisor.processes) == {"started-child"}
    _assert_all_reaped(supervisor)


def test_shutdown_after_empty_partial_startup_is_idempotent() -> None:
    supervisor = _new_supervisor()
    supervisor.shutdown(request_graceful=False)
    supervisor.shutdown(request_graceful=False)
    assert supervisor.shutdown_complete
    assert supervisor.owned_pids() == {}

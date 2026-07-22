from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from typing import Any

ReadyAck = tuple[str, str]


@dataclass(frozen=True)
class ProcessSpec:
    role: str
    target: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    expects_ready: bool = True


@dataclass(frozen=True)
class ProcessFailure:
    role: str
    name: str
    exitcode: int
    phase: str

    def describe(self) -> str:
        return (
            f"server child {self.role!r} ({self.name}) exited with code "
            f"{self.exitcode} during {self.phase}"
        )


class ServerProcessSupervisor:
    """Own the lifecycle of the local processes created for one API server."""

    def __init__(
        self,
        *,
        context: mp.context.BaseContext | None = None,
        startup_timeout: float = 15 * 60,
        ack_poll_interval: float = 0.1,
        shutdown_grace: float = 10.0,
        termination_grace: float = 5.0,
        process_factory: Callable[..., BaseProcess] | None = None,
    ) -> None:
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if ack_poll_interval <= 0:
            raise ValueError("ack_poll_interval must be positive")
        if shutdown_grace < 0 or termination_grace < 0:
            raise ValueError("shutdown grace periods must be non-negative")

        self.context = context or mp.get_context("spawn")
        self.startup_timeout = startup_timeout
        self.ack_poll_interval = ack_poll_interval
        self.shutdown_grace = shutdown_grace
        self.termination_grace = termination_grace
        self.ack_queue: mp.Queue[ReadyAck] = self.context.Queue()
        self.processes: dict[str, BaseProcess] = {}
        self.ready_roles: set[str] = set()

        self._process_factory = process_factory or self.context.Process
        self._expected_ready_roles: set[str] = set()
        self._graceful_shutdown: Callable[[], None] | None = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._startup_complete = False

    @property
    def startup_complete(self) -> bool:
        return self._startup_complete

    @property
    def shutdown_complete(self) -> bool:
        return self._shutdown_complete

    def set_graceful_shutdown(self, callback: Callable[[], None]) -> None:
        self._graceful_shutdown = callback

    def start(self, specs: Iterable[ProcessSpec]) -> ServerProcessSupervisor:
        if self.processes or self._startup_complete:
            raise RuntimeError("server process supervisor can only be started once")

        spec_list = list(specs)
        roles = [spec.role for spec in spec_list]
        if len(roles) != len(set(roles)):
            raise ValueError("server child process roles must be unique")
        self._expected_ready_roles = {spec.role for spec in spec_list if spec.expects_ready}

        try:
            for spec in spec_list:
                process = self._process_factory(
                    target=spec.target,
                    args=spec.args,
                    kwargs=spec.kwargs,
                    daemon=False,
                    name=spec.role,
                )
                self.processes[spec.role] = process
                process.start()
            self._wait_for_readiness()
            self._startup_complete = True
            return self
        except BaseException:
            self.shutdown(request_graceful=False)
            raise

    def _wait_for_readiness(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while self.ready_roles != self._expected_ready_roles:
            self._raise_for_child_exit(phase="startup")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = sorted(self._expected_ready_roles - self.ready_roles)
                raise TimeoutError(
                    "server startup timed out waiting for readiness from: " + ", ".join(missing)
                )
            try:
                ack = self.ack_queue.get(timeout=min(self.ack_poll_interval, remaining))
            except queue.Empty:
                continue

            if not isinstance(ack, tuple) or len(ack) != 2:
                raise RuntimeError(f"invalid server readiness acknowledgment: {ack!r}")
            role, message = ack
            if role not in self._expected_ready_roles:
                raise RuntimeError(f"unexpected server readiness acknowledgment from {role!r}")
            if role in self.ready_roles:
                raise RuntimeError(f"duplicate server readiness acknowledgment from {role!r}")
            self.ready_roles.add(role)
            # Preserve the existing useful per-worker readiness messages.
            logging.getLogger(__name__).info(message)

        self._raise_for_child_exit(phase="startup")

    def _raise_for_child_exit(self, *, phase: str) -> None:
        failure = self.unexpected_exit(phase=phase)
        if failure is not None:
            raise RuntimeError(failure.describe())

    def unexpected_exit(self, *, phase: str = "runtime") -> ProcessFailure | None:
        if self._shutdown_complete:
            return None
        for role, process in self.processes.items():
            exitcode = process.exitcode
            if exitcode is not None:
                return ProcessFailure(
                    role=role,
                    name=process.name,
                    exitcode=exitcode,
                    phase=phase,
                )
        return None

    def owned_pids(self) -> dict[str, int | None]:
        return {role: process.pid for role, process in self.processes.items()}

    def live_roles(self) -> list[str]:
        return [role for role, process in self.processes.items() if process.is_alive()]

    def shutdown(self, *, request_graceful: bool = True) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return

            if request_graceful and self._graceful_shutdown is not None:
                try:
                    self._graceful_shutdown()
                except Exception as exc:
                    # Cleanup must continue even when the existing message path is gone.
                    logging.getLogger("minisgl.server.launch").warning(
                        "Graceful scheduler shutdown signal failed: %s", exc
                    )

            if request_graceful:
                self._join_until(time.monotonic() + self.shutdown_grace)
            for process in self.processes.values():
                if process.is_alive():
                    process.terminate()
            self._join_until(time.monotonic() + self.termination_grace)

            live = self.live_roles()
            if live:
                raise RuntimeError(
                    "server children did not exit after termination: " + ", ".join(live)
                )

            self.ack_queue.close()
            self.ack_queue.join_thread()
            self._shutdown_complete = True

    def _join_until(self, deadline: float) -> None:
        while True:
            live = [process for process in self.processes.values() if process.is_alive()]
            if not live:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            timeout = min(self.ack_poll_interval, remaining)
            for process in live:
                process.join(timeout=timeout)

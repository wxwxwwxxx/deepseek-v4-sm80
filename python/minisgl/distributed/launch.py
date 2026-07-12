from __future__ import annotations

import multiprocessing as mp
import os
import socket
import time
from collections.abc import Callable
from typing import Any


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_rank(
    rank: int,
    world_size: int,
    master_port: int,
    target: Callable[..., Any],
    args: tuple[Any, ...],
) -> None:
    os.environ.update(
        {
            "LOCAL_RANK": str(rank),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )
    target(*args)


def launch_tensor_parallel(
    tensor_parallel_size: int,
    target: Callable[..., Any],
    *args: Any,
) -> None:
    """Run a synchronous offline entry once on every local TP rank."""
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "launch_tensor_parallel must run from a normal Python process, not inside torchrun"
        )
    if tensor_parallel_size == 1:
        target(*args)
        return

    context = mp.get_context("spawn")
    master_port = _find_free_port()
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, tensor_parallel_size, master_port, target, args),
            name=f"minisgl-TP{rank}-offline",
        )
        for rank in range(tensor_parallel_size)
    ]
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            failed = next(
                (
                    process
                    for process in processes
                    if process.exitcode is not None and process.exitcode != 0
                ),
                None,
            )
            if failed is not None:
                raise RuntimeError(
                    f"offline TP worker {failed.name} exited with code {failed.exitcode}"
                )
            time.sleep(0.05)
        failed = next((process for process in processes if process.exitcode != 0), None)
        if failed is not None:
            raise RuntimeError(
                f"offline TP worker {failed.name} exited with code {failed.exitcode}"
            )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()

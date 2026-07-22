from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import ReadyAck


def _run_scheduler(
    args: ServerArgs,
    ack_queue: mp.Queue[ReadyAck],
    ready_role: str,
) -> None:
    import torch
    from minisgl.scheduler import Scheduler

    # DSV4 prepares version-keyed weight caches during construction. Tensors
    # created by inference_mode do not expose version counters, so construct
    # under no_grad and let Scheduler.run_forever's inference_mode own the
    # steady-state execution loop.
    with torch.no_grad():
        scheduler = Scheduler(args)
    scheduler.sync_all_ranks()

    if args.tp_info.is_primary():
        ack_queue.put((ready_role, "Scheduler is ready"))

    if args.silent_output:
        logging.disable(logging.INFO)

    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger = init_logger(__name__)
        if args.tp_info.is_primary():
            print()  # for a clean newline after ^C
            logger.info("Scheduler exiting gracefully...")
        scheduler.shutdown()


def _request_scheduler_shutdown(args: ServerArgs) -> None:
    import zmq
    from minisgl.message import BaseBackendMsg, ExitMsg
    from minisgl.utils import ZmqPushQueue

    send_backend = ZmqPushQueue(
        args.zmq_backend_addr,
        create=False,
        encoder=BaseBackendMsg.encoder,
    )
    # A fresh local PUSH connection needs a brief bounded flush window for the
    # shutdown message; never let a broken backend make parent cleanup unbounded.
    send_backend.socket.setsockopt(zmq.SNDTIMEO, 1000)
    send_backend.socket.setsockopt(zmq.LINGER, 1000)
    try:
        send_backend.put(ExitMsg())
    finally:
        send_backend.stop()


def start_server_processes(
    server_args: ServerArgs,
    *,
    startup_timeout: float = 15 * 60,
    ack_poll_interval: float = 0.1,
    shutdown_grace: float = 10.0,
    termination_grace: float = 5.0,
):
    from minisgl.tokenizer import tokenize_worker

    from .supervisor import ProcessSpec, ServerProcessSupervisor

    context = mp.get_context("spawn")
    supervisor = ServerProcessSupervisor(
        context=context,
        startup_timeout=startup_timeout,
        ack_poll_interval=ack_poll_interval,
        shutdown_grace=shutdown_grace,
        termination_grace=termination_grace,
    )
    ack_queue = supervisor.ack_queue
    world_size = server_args.tp_info.size
    specs: list[ProcessSpec] = []

    for rank in range(world_size):
        role = f"minisgl-TP{rank}-scheduler"
        new_args = replace(
            server_args,
            tp_info=DistributedInfo(rank, world_size),
        )
        specs.append(
            ProcessSpec(
                role=role,
                target=_run_scheduler,
                args=(new_args, ack_queue, role),
                expects_ready=rank == 0,
            )
        )

    num_tokenizers = server_args.num_tokenizer
    detokenizer_role = "minisgl-detokenizer-0"
    specs.append(
        ProcessSpec(
            role=detokenizer_role,
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
                "ready_role": detokenizer_role,
            },
        )
    )
    for tokenizer_id in range(num_tokenizers):
        role = f"minisgl-tokenizer-{tokenizer_id}"
        specs.append(
            ProcessSpec(
                role=role,
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": tokenizer_id,
                    "ack_queue": ack_queue,
                    "ready_role": role,
                },
            )
        )

    supervisor.start(specs)
    supervisor.set_graceful_shutdown(lambda: _request_scheduler_shutdown(server_args))
    return supervisor


def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess():
        supervisor = start_server_processes(server_args)
        logger.info("Server-owned child PIDs: %s", supervisor.owned_pids())
        return supervisor

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()

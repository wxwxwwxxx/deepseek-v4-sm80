from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


def _is_local_model_path(model_path: str) -> bool:
    """Distinguish filesystem paths from Hugging Face repo IDs without loading config."""
    return (
        model_path.endswith("/")
        or Path(model_path).is_absolute()
        or model_path.startswith(("./", "../", "~"))
        or Path(model_path).exists()
    )


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    served_model_name: str | None = None

    @property
    def resolved_served_model_name(self) -> str:
        if self.served_model_name:
            return self.served_model_name
        model_path = self.model_path.rstrip("/")
        if not _is_local_model_path(self.model_path):
            return model_path
        return Path(model_path).name or model_path

    def accepts_model(self, model: str) -> bool:
        return model in {self.resolved_served_model_name, self.model_path}

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    max_extend_tokens_explicit = any(
        arg in {"--max-prefill-length", "--max-extend-length"}
        or arg.startswith("--max-prefill-length=")
        or arg.startswith("--max-extend-length=")
        for arg in args
    )
    max_running_req_explicit = any(
        arg == "--max-running-requests" or arg.startswith("--max-running-requests=") for arg in args
    )
    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--served-model-name",
        default=ServerArgs.served_model_name,
        help=(
            "Public model ID exposed by /v1/models and chat responses. "
            "Defaults to the input repo ID for Hugging Face models and the path basename "
            "for local models."
        ),
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--dsv4-runtime",
        dest="dsv4_runtime_mode",
        choices=["optimized", "fallback"],
        default=ServerArgs.dsv4_runtime_mode,
        help="Select the DeepSeek V4 optimized release runtime or reference fallback.",
    )

    parser.add_argument(
        "--dsv4-sm80-recipe",
        default=ServerArgs.dsv4_sm80_recipe,
        choices=[
            "dsv4_sm80_low_m64",
            "dsv4_sm80_mid_m128",
            "dsv4_sm80_balanced",
            "dsv4_sm80_long_context_512k",
            "dsv4_sm80_1m_smoke",
        ],
        help=(
            "Optionally apply a request/graph/context recipe validated on DGX A100 "
            "8x80GB. Explicit request, graph, and sequence settings override its fields."
        ),
    )

    parser.add_argument(
        "--context-length",
        "--max-model-len",
        dest="context_length",
        type=int,
        default=ServerArgs.context_length,
        help=(
            "The model's requested maximum context length, including prompt and generated "
            "tokens. Defaults to max_position_embeddings from the model config.json. "
            "The effective limit may be lower when constrained by KV-cache capacity or RoPE."
        ),
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--stats-log-interval",
        type=float,
        default=ServerArgs.stats_log_interval,
        help="Seconds between periodic scheduler throughput and load logs.",
    )
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="Disable periodic scheduler throughput and load logs.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()
    if kwargs["stats_log_interval"] <= 0:
        parser.error("--stats-log-interval must be positive")
    kwargs["max_extend_tokens_explicit"] = max_extend_tokens_explicit
    kwargs["max_running_req_explicit"] = max_running_req_explicit

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["max_running_req_explicit"] = True
        kwargs["silent_output"] = True
        kwargs["disable_log_stats"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell

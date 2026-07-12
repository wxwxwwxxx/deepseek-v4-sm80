from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.dsv4_runtime import DSV4RuntimeMode
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.engine.graph_policy import ResolvedCudaGraphBucketPolicy
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    max_running_req_explicit: bool = False
    dsv4_runtime_mode: DSV4RuntimeMode = "optimized"
    dsv4_sm80_recipe: str | None = None
    attention_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    disable_cuda_graph: bool = False
    cuda_graph_policy: ResolvedCudaGraphBucketPolicy | None = field(
        default=None, init=False, repr=False
    )
    allow_dsv4_cuda_graph: bool = False
    cuda_graph_capture_fail_open: bool = False
    cuda_graph_capture_greedy_sample: bool = False
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    distributed_init_method: str | None = None

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"

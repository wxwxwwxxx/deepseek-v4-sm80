from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, List

from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.engine.graph_policy import ResolvedCudaGraphBucketPolicy
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    max_running_req: int = 128
    max_running_req_explicit: bool = False
    enable_reasoning_sampler_contract: bool = False
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
    context_length: int | None = None
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
        if self.context_length is not None:
            return self.context_length
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def reasoning_sampler_contract_enabled(self) -> bool:
        """Return whether the optional reasoning grammar mask is active.

        The release runtime enables it only when explicitly requested.
        """

        return self.enable_reasoning_sampler_contract

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"

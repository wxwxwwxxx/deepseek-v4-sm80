from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool, estimate_kvcache_bytes_per_page
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import (
    dsv4_direct_copy_nvtx,
    dsv4_prefix_debug,
    init_logger,
    is_sm90_supported,
    is_sm100_supported,
    torch_dtype,
)

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        prepare_for_cuda_graph_capture = getattr(self.model, "prepare_for_cuda_graph_capture", None)
        if callable(prepare_for_cuda_graph_capture):
            self.model_prepare_report = prepare_for_cuda_graph_capture()
        else:
            self.model_prepare_report = {}

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up(self.max_seq_len, max(32, config.page_size))
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)
        self._copy_done_event_pool = [torch.cuda.Event() for _ in range(2)]
        self._copy_done_event_pool_ids = {id(event) for event in self._copy_done_event_pool}

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            capture_fail_open=config.cuda_graph_capture_fail_open,
            capture_greedy_sample=config.cuda_graph_capture_greedy_sample,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        init_method = config.distributed_init_method or config.distributed_addr
        init_kwargs = {
            "rank": config.tp_info.rank,
            "world_size": config.tp_info.size,
            "timeout": timedelta(seconds=config.distributed_timeout),
            "init_method": init_method,
        }
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                **init_kwargs,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                **init_kwargs,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            if config.model_config.is_deepseek_v4:
                return dict(load_weight(config.model_path, self.device))
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = estimate_kvcache_bytes_per_page(
            config.model_config,
            page_size=config.page_size,
            dtype=self.dtype,
            tp_size=config.tp_info.size,
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        next_tokens_gpu: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        forward_source = "unknown"
        with self.ctx.forward_batch(batch):
            if (
                args.temperatures is None
                and self.graph_runner.can_replay_greedy_sample(batch)
            ):
                forward_source = "cuda_graph_greedy_sample"
                next_tokens_gpu = self.graph_runner.replay_greedy_sample(batch)
            elif self.graph_runner.can_use_cuda_graph(batch):
                forward_source = "cuda_graph_replay"
                logits = self.graph_runner.replay(batch)
            else:
                forward_source = "eager"
                self.graph_runner.record_eager_decode(batch)
                logits = self.model.forward()

        debug_recorder = dsv4_prefix_debug.get_dsv4_prefix_debug_recorder()
        debug_snapshot = (
            debug_recorder.capture_pre_sample(
                batch=batch,
                logits=logits[: batch.size] if logits is not None else None,
                forward_source=forward_source,
            )
            if debug_recorder is not None
            else None
        )

        for req in batch.reqs:
            req.complete_one()

        if next_tokens_gpu is None:
            with dsv4_direct_copy_nvtx(
                f"sampler_logits_staging.sample_to_int32.bs{batch.size}",
                logits=logits[: batch.size],
            ):
                assert logits is not None
                next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        if debug_recorder is not None:
            debug_recorder.finish(
                debug_snapshot,
                next_tokens=next_tokens_gpu,
                graph_runner=getattr(self.graph_runner, "capture_status", {}),
            )
        with dsv4_direct_copy_nvtx(
            f"sampler_logits_staging.next_tokens_to_cpu.bs{batch.size}",
            next_tokens=next_tokens_gpu,
        ):
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = self._acquire_copy_done_event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def _acquire_copy_done_event(self) -> torch.cuda.Event:
        if self._copy_done_event_pool:
            event = self._copy_done_event_pool.pop()
            self._copy_done_event_pool_ids.remove(id(event))
            return event
        return torch.cuda.Event()

    def release_copy_done_event(self, event: torch.cuda.Event) -> None:
        event_id = id(event)
        if len(self._copy_done_event_pool) >= 2 or event_id in self._copy_done_event_pool_ids:
            return
        self._copy_done_event_pool.append(event)
        self._copy_done_event_pool_ids.add(event_id)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up(num: int, multiple: int) -> int:
    return (num + multiple - 1) // multiple * multiple


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.model_config.is_deepseek_v4:
        if config.attention_backend != "dsv4":
            override("attention_backend", "dsv4")
            logger.info_rank0("Using DSV4 attention backend for DeepSeek V4")
        if not config.allow_dsv4_cuda_graph:
            override("cuda_graph_bs", [])
            override("cuda_graph_max_bs", 0)
        else:
            if config.cuda_graph_bs is None:
                override("cuda_graph_bs", [1, 2, 4])
            if config.cuda_graph_max_bs is None:
                override("cuda_graph_max_bs", max(config.cuda_graph_bs or [0]))
            override("cuda_graph_capture_fail_open", True)
            logger.info_rank0(
                f"Opting in to DeepSeek V4 decode CUDA graph sizes: {config.cuda_graph_bs}"
            )
    elif config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

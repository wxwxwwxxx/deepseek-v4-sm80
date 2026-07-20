from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.dsv4_release import DSV4_RELEASE
from minisgl.kvcache import create_kvcache_pool, estimate_kvcache_bytes_per_page
from minisgl.models import create_model, load_weight
from minisgl.reasoning import ReasoningTokenIds, resolve_reasoning_token_ids
from minisgl.utils import (
    init_logger,
    load_tokenizer,
    torch_dtype,
)

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .graph_memory import (
    GraphMemoryEstimate,
    compare_graph_capture,
    empty_graph_memory_estimate,
    estimate_dsv4_sm80_graph_memory,
    select_num_pages,
)
from .graph_policy import (
    ResolvedCudaGraphBucketPolicy,
    resolve_cuda_graph_bucket_policy,
)
from .sample import BatchSamplingArgs, ReasoningSampler, Sampler

logger = init_logger(__name__)

_DSV4_SM80_DEFAULT_CUDA_GRAPH_MAX_BS = 128
_DSV4_SM80_RELEASE_CUDA_GRAPH_BS = (1, 2, 4, 8, 16)
_DSV4_SM80_RECIPES = {
    "default_m128": (128, 128, None),
    "low_m64": (64, 64, None),
    "high_m256": (256, 256, None),
    "long_context_m4": (4, 4, 524_288),
}
_GENERIC_DEFAULT_MAX_EXTEND_TOKENS = 8192
_DSV4_SM80_DEFAULT_MAX_EXTEND_TOKENS = 8192


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


def _resolve_effective_max_seq_len(
    *,
    requested_max_seq_len: int,
    kv_capacity_tokens: int,
    model_rotary_max: int,
    rope_is_on_the_fly: bool,
) -> tuple[int, int, str]:
    """Resolve admission and RoPE bounds before allocating max-width surfaces."""
    rope_limit = requested_max_seq_len if rope_is_on_the_fly else model_rotary_max
    effective_max = min(requested_max_seq_len, kv_capacity_tokens, rope_limit)
    effective_rope_len = effective_max if rope_is_on_the_fly else model_rotary_max
    rope_kind = "on_the_fly" if rope_is_on_the_fly else "materialized"
    return effective_max, effective_rope_len, rope_kind


def validate_graph_bucket_contract(
    *,
    resolved_bs: tuple[int, ...],
    estimated_bs: tuple[int, ...],
    runner_requested_bs: list[int],
    runner_captured_bs: list[int],
    capture_error: object | None,
) -> None:
    """Fail fast when planner and runner observe different graph policies."""

    requested = tuple(sorted(int(value) for value in runner_requested_bs))
    captured = tuple(sorted(int(value) for value in runner_captured_bs))
    if resolved_bs != estimated_bs or resolved_bs != requested:
        raise RuntimeError(
            "Programming error: CUDA graph bucket policy mismatch before/at capture: "
            f"resolved={list(resolved_bs)}, estimated={list(estimated_bs)}, "
            f"runner_requested={list(requested)}."
        )
    if capture_error is None and resolved_bs != captured:
        raise RuntimeError(
            "Programming error: CUDA graph captured bucket policy mismatch: "
            f"resolved={list(resolved_bs)}, runner_captured={list(captured)}."
        )


def resolve_engine_reasoning_token_ids(config: EngineConfig) -> ReasoningTokenIds | None:
    """Load protocol tokens only for the enabled production contract."""

    if not config.reasoning_sampler_contract_enabled:
        return None
    return resolve_reasoning_token_ids(load_tokenizer(config.model_path))


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
        self.reasoning_sampler_contract_enabled = (
            config.reasoning_sampler_contract_enabled
        )
        self.reasoning_token_ids = resolve_engine_reasoning_token_ids(config)
        # DeepSeek V4 SM80 uses BF16 activations while preserving model-defined
        # FP32 state and quantized checkpoint weights.
        self.dtype = torch.bfloat16
        self._marlin_wna16_release_done = False
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        self.cuda_graph_policy = _resolve_cuda_graph_policy(config, free_memory=init_free_memory)
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        with torch.device("meta"), torch_dtype(self.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        prepare_for_cuda_graph_capture = getattr(self.model, "prepare_for_cuda_graph_capture", None)
        if callable(prepare_for_cuda_graph_capture):
            self.model_prepare_report = prepare_for_cuda_graph_capture()
        else:
            self.model_prepare_report = {}

        # ======================= KV cache initialization ========================
        self.graph_memory_estimate = self._estimate_graph_memory(config)
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        self._maybe_release_marlin_wna16_for_timing(
            timing="before_kv_alloc",
            stage_label="before_kv_alloc_release",
        )
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            enable_dsv4_component_loc_ownership=bool(
                getattr(config, "enable_dsv4_component_loc_ownership", False)
            ),
            enable_dsv4_swa_independent_lifecycle=bool(
                getattr(config, "enable_dsv4_swa_independent_lifecycle", False)
            ),
            max_running_req=int(getattr(config, "max_running_req", 1)),
            dsv4_swa_num_pages=(
                self._planned_dsv4_swa_independent_pages(config)
                if self._dsv4_swa_independent_enabled(config)
                else None
            ),
            dsv4_dummy_token_start=num_tokens,
        )
        self._maybe_release_marlin_wna16_for_timing(
            timing="after_kv_alloc",
            stage_label="after_kv_alloc_release",
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        model_rotary_max = int(config.model_config.rotary_config.max_position)
        rope_is_on_the_fly = bool(config.model_config.is_deepseek_v4)
        (
            self.max_seq_len,
            self.effective_rope_cache_len,
            self.rope_cache_kind,
        ) = _resolve_effective_max_seq_len(
            requested_max_seq_len=config.max_seq_len,
            kv_capacity_tokens=num_tokens,
            model_rotary_max=model_rotary_max,
            rope_is_on_the_fly=rope_is_on_the_fly,
        )
        aligned_max_seq_len = _align_up(self.max_seq_len, max(32, config.page_size))
        self.kv_capacity_plan_report["effective_sequence_width"] = int(aligned_max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= DSV4 attention backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )

        # ======================= Sampler initialization ========================
        self.sampler = (
            ReasoningSampler(
                self.device,
                config.model_config.vocab_size,
                self.reasoning_token_ids,
            )
            if self.reasoning_token_ids is not None
            else Sampler(self.device, config.model_config.vocab_size)
        )
        self._copy_done_event_pool = [torch.cuda.Event() for _ in range(2)]
        self._copy_done_event_pool_ids = {id(event) for event in self._copy_done_event_pool}

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        dummy_position = 0
        dummy_input_ids = torch.zeros(1, dtype=torch.int32, device="cpu")
        self.dummy_req = Req(
            input_ids=dummy_input_ids,
            table_idx=config.max_running_req,
            cached_len=dummy_position,
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
            resolved_graph_bs=self.cuda_graph_policy.resolved_bs,
            graph_policy_report=self.cuda_graph_policy.to_report(),
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            capture_fail_open=config.cuda_graph_capture_fail_open,
            capture_greedy_sample=config.cuda_graph_capture_greedy_sample,
            reasoning_token_ids=self.reasoning_token_ids,
        )
        post_kv_prepare_report = self.graph_runner.capture_status.get(
            "post_kv_model_cache_prepare_report", {}
        )
        if post_kv_prepare_report:
            self.model_prepare_report["fused_wqa_wkv_bf16_weight_cache"] = (
                post_kv_prepare_report
            )
        self._finalize_graph_memory_ledger()
        self._maybe_release_marlin_wna16_for_timing(
            timing="after_graph_capture",
            stage_label="after_graph_capture_release",
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
            max_bytes = _pynccl_max_buffer_bytes(config)
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                **init_kwargs,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _marlin_wna16_release_timing(self) -> str:
        return DSV4_RELEASE.marlin_release_timing

    def _maybe_release_marlin_wna16_for_timing(self, *, timing: str, stage_label: str) -> None:
        if self._marlin_wna16_release_done:
            return
        if not DSV4_RELEASE.release_raw_expert_weights:
            return
        if self._marlin_wna16_release_timing() != timing:
            return
        release = getattr(self.model, "release_marlin_wna16_original_expert_weights", None)
        if not callable(release):
            return
        report = release(stage_label=stage_label)
        self._marlin_wna16_release_done = True
        self.model_prepare_report[f"moe_marlin_wna16_{stage_label}"] = report

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        return dict(load_weight(config.model_path, self.device))

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = estimate_kvcache_bytes_per_page(
            config.model_config,
            page_size=config.page_size,
            tp_size=config.tp_info.size,
        )
        fixed_swa_cache_bytes = 0
        legacy_cache_per_page = cache_per_page
        if config.model_config.is_deepseek_v4 and self._dsv4_swa_independent_enabled(config):
            dtype_size = torch.bfloat16.itemsize
            planned_swa_pages = self._planned_dsv4_swa_independent_pages(config)
            swa_per_page = (
                config.model_config.num_layers
                * config.page_size
                * config.model_config.head_dim
                * dtype_size
            )
            fixed_swa_cache_bytes = int(planned_swa_pages * swa_per_page)
            cache_per_page = int(max(cache_per_page - swa_per_page, 1))
        num_pages = config.num_page_override
        credit_report = self._marlin_wna16_release_capacity_credit_report(
            config=config,
            cache_per_page=cache_per_page,
        )
        model_memory = old_free_memory - new_free_memory
        requested_device_budget = int(config.memory_ratio * old_free_memory)
        credit_bytes = int(credit_report.get("net_release_credit_bytes", 0) or 0)
        applied_credit_bytes = credit_bytes if bool(credit_report.get("eligible", False)) else 0
        requested_width = _align_up(config.max_seq_len, max(32, config.page_size))
        request_table_bytes = int(
            (int(config.max_running_req) + 1) * requested_width * torch.int32.itemsize
        )
        graph_estimate_bytes = int(self.graph_memory_estimate.estimate_bytes)
        graph_margin_bytes = int(self.graph_memory_estimate.safety_margin_bytes)
        non_graph_activation_allowance_bytes = 0
        variable_kv_budget = (
            requested_device_budget
            - model_memory
            + applied_credit_bytes
            - fixed_swa_cache_bytes
            - request_table_bytes
            - non_graph_activation_allowance_bytes
            - graph_estimate_bytes
            - graph_margin_bytes
        )
        baseline_variable_kv_budget = (
            requested_device_budget
            - model_memory
            + applied_credit_bytes
            - fixed_swa_cache_bytes
            - request_table_bytes
            - non_graph_activation_allowance_bytes
        )
        try:
            num_pages, baseline_pages, lost_pages = select_num_pages(
                variable_kv_budget_bytes=variable_kv_budget,
                baseline_variable_kv_budget_bytes=baseline_variable_kv_budget,
                cache_per_page_bytes=cache_per_page,
                num_page_override=num_pages,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc} graph_estimate_bytes={graph_estimate_bytes}, "
                f"graph_safety_margin_bytes={graph_margin_bytes}."
            ) from exc

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page + fixed_swa_cache_bytes
        credit_report["planned_num_pages"] = int(num_pages)
        credit_report["planned_num_tokens"] = int(num_tokens)
        credit_report["planned_kv_bytes"] = int(real_kv_size)
        self.kv_capacity_plan_report = {
            "dsv4_sm80_recipe": getattr(config, "dsv4_sm80_recipe", None),
            "old_free_memory_bytes": int(old_free_memory),
            "new_free_memory_bytes": int(new_free_memory),
            "memory_ratio": float(config.memory_ratio),
            "cache_per_page_bytes": int(cache_per_page),
            "legacy_cache_per_page_bytes": int(legacy_cache_per_page),
            "fixed_swa_cache_bytes": int(fixed_swa_cache_bytes),
            "requested_device_budget_bytes": int(requested_device_budget),
            "weights_and_transformed_cache_bytes": int(model_memory),
            "request_page_table_bytes": int(request_table_bytes),
            "request_page_table_width": int(requested_width),
            "non_graph_activation_allowance_bytes": int(non_graph_activation_allowance_bytes),
            "unrequested_device_headroom_bytes": int(old_free_memory - requested_device_budget),
            "graph_memory": self.graph_memory_estimate.to_report(),
            "cuda_graph_bucket_policy": self.cuda_graph_policy.to_report(),
            "graph_memory_estimate_elapsed_s": float(
                getattr(self, "graph_memory_estimate_elapsed_s", 0.0)
            ),
            "variable_kv_budget_bytes": int(variable_kv_budget),
            "baseline_pages_without_graph_reserve": int(baseline_pages),
            "lost_pages_to_graph_reserve": int(lost_pages),
            "lost_tokens_to_graph_reserve": int(lost_pages * config.page_size),
            "final_num_pages": int(num_pages),
            "final_num_tokens": int(num_tokens),
            "num_page_override": config.num_page_override,
            "release_credit": credit_report,
        }
        if bool(credit_report.get("applied_to_num_pages", False)):
            logger.info_rank0(
                "Applied Marlin WNA16 release credit before KV allocation: "
                f"{mem_GB(int(credit_report['net_release_credit_bytes']))}, "
                f"equivalent_pages={credit_report.get('net_release_credit_pages')}"
            )
        logger.info_rank0(
            f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}"
        )
        return num_pages

    def _estimate_graph_memory(self, config: EngineConfig) -> GraphMemoryEstimate:
        started = time.perf_counter()
        graph_bs = self.cuda_graph_policy.resolved_bs
        if not graph_bs or not config.model_config.is_deepseek_v4:
            estimate = empty_graph_memory_estimate(graph_bs)
            self.graph_memory_estimate_elapsed_s = time.perf_counter() - started
            return estimate
        capability = tuple(int(part) for part in torch.cuda.get_device_capability(self.device))
        if capability != (8, 0):
            estimate = empty_graph_memory_estimate(graph_bs)
            self.graph_memory_estimate_elapsed_s = time.perf_counter() - started
            return estimate
        estimate = estimate_dsv4_sm80_graph_memory(
            graph_bs,
            metadata_width=int(config.max_seq_len),
            page_size=int(config.page_size),
            capture_greedy_sample=bool(config.cuda_graph_capture_greedy_sample),
            reasoning_sampler_contract_enabled=(
                config.reasoning_sampler_contract_enabled
            ),
        )
        estimate_bytes = self._sync_max_int(estimate.estimate_bytes)
        margin_bytes = self._sync_max_int(estimate.safety_margin_bytes)
        if (
            estimate_bytes != estimate.estimate_bytes
            or margin_bytes != estimate.safety_margin_bytes
        ):
            estimate = GraphMemoryEstimate(
                **{
                    **estimate.__dict__,
                    "estimate_bytes": estimate_bytes,
                    "safety_margin_bytes": margin_bytes,
                }
            )
        logger.info_rank0(
            "Reserving CUDA graph memory before KV planning: "
            f"estimate={mem_GB(estimate.estimate_bytes)}, "
            f"safety_margin={mem_GB(estimate.safety_margin_bytes)}, "
            f"buckets={list(estimate.graph_bs)}, metadata_width={estimate.metadata_width}"
        )
        self.graph_memory_estimate_elapsed_s = time.perf_counter() - started
        return estimate

    def _sync_max_int(self, value: int) -> int:
        # Capacity coordination is control-plane work.  Use the existing CPU
        # group so querying a scalar cannot initialize/resize PyNCCL device
        # buffers before the authoritative physical-memory snapshot.
        tensor = torch.tensor([int(value)], dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=self.tp_cpu_group,
        )
        return int(tensor.item())

    def _finalize_graph_memory_ledger(self) -> None:
        status = getattr(self.graph_runner, "capture_status", {})
        validate_graph_bucket_contract(
            resolved_bs=self.cuda_graph_policy.resolved_bs,
            estimated_bs=self.graph_memory_estimate.graph_bs,
            runner_requested_bs=list(status.get("requested_bs", [])),
            runner_captured_bs=list(status.get("captured_bs", [])),
            capture_error=status.get("error"),
        )
        actual_local = int(status.get("capture_memory_delta_bytes") or 0)
        actual = self._sync_max_int(actual_local)
        estimate = int(self.graph_memory_estimate.estimate_bytes)
        margin = int(self.graph_memory_estimate.safety_margin_bytes)
        post_capture_free = self._sync_get_memory()[0]
        graph_report = self.kv_capacity_plan_report.setdefault("graph_memory", {})
        post_kv_cache_report = status.get("post_kv_model_cache_prepare_report", {})
        post_kv_cache_bytes = (
            int(post_kv_cache_report.get("total_bytes", 0))
            if isinstance(post_kv_cache_report, dict)
            else 0
        )
        try:
            comparison = compare_graph_capture(
                estimate_bytes=estimate,
                safety_margin_bytes=margin,
                actual_physical_bytes=actual,
            )
        except RuntimeError:
            self.graph_runner.destroy_cuda_graphs()
            raise
        graph_report.update(comparison)
        graph_report["post_kv_persistent_cache_bytes"] = post_kv_cache_bytes
        graph_report["actual_physical_bytes_includes_post_kv_model_cache"] = bool(
            self.cuda_graph_policy.resolved_bs and post_kv_cache_bytes
        )
        graph_report["post_capture_free_bytes"] = int(post_capture_free)
        status["graph_memory_plan"] = dict(graph_report)

    def _dsv4_swa_independent_enabled(self, config: EngineConfig) -> bool:
        return bool(getattr(config, "enable_dsv4_swa_independent_lifecycle", False))

    def _planned_dsv4_swa_independent_pages(self, config: EngineConfig) -> int:
        window_size = int(getattr(config.model_config, "window_size", 128) or 128)
        tail_pages_per_req = max((window_size + config.page_size - 1) // config.page_size, 1)
        running_tail_pages = max(int(getattr(config, "max_running_req", 1)), 1)
        # A decode request can need one freshly allocated SWA page before the
        # previous page ages out of the sliding window and is tombstoned.
        running_tail_pages *= tail_pages_per_req + 1
        max_forward_len = int(getattr(config, "max_forward_len", config.page_size))
        max_forward_pages = max((max_forward_len + config.page_size - 1) // config.page_size, 1)
        planned = max_forward_pages + running_tail_pages + 1
        max_running_req = max(int(getattr(config, "max_running_req", 1)), 1)
        if max_running_req >= 16:
            planned = max(planned, max_running_req * 8)
        if config.num_page_override is not None:
            planned = min(planned, int(config.num_page_override))
        return planned

    def _marlin_wna16_release_capacity_credit_report(
        self,
        *,
        config: EngineConfig,
        cache_per_page: int,
    ) -> dict[str, Any]:
        requested = DSV4_RELEASE.marlin_capacity_credit
        release_requested = DSV4_RELEASE.release_raw_expert_weights
        timing = self._marlin_wna16_release_timing()
        moe_report = {}
        if isinstance(self.model_prepare_report, dict):
            maybe_report = self.model_prepare_report.get("moe_marlin_wna16_cache", {})
            if isinstance(maybe_report, dict):
                moe_report = maybe_report
        source_bytes = int(moe_report.get("total_source_bytes", 0) or 0)
        guard_bytes = self._planned_marlin_wna16_guard_bytes(source_bytes)
        safety_margin = 0
        gross_credit = max(0, source_bytes)
        net_credit = max(0, gross_credit - guard_bytes - int(safety_margin))
        eligible = (
            bool(config.model_config.is_deepseek_v4)
            and requested
            and release_requested
            and timing == "before_kv_alloc"
            and source_bytes > 0
        )
        applied = bool(eligible and config.num_page_override is None and net_credit > 0)
        return {
            "requested": bool(requested),
            "release_requested": bool(release_requested),
            "timing": timing,
            "eligible": bool(eligible),
            "applied_to_num_pages": applied,
            "ineligible_reason": None
            if eligible
            else _marlin_wna16_credit_ineligible_reason(
                config=config,
                requested=requested,
                release_requested=release_requested,
                timing=timing,
                source_bytes=source_bytes,
            ),
            "source_bytes": source_bytes,
            "gross_release_credit_bytes": gross_credit,
            "planned_guard_or_reserved_bytes": int(guard_bytes),
            "safety_margin_bytes": int(safety_margin),
            "net_release_credit_bytes": int(net_credit if eligible else 0),
            "theoretical_release_credit_pages": (
                float(gross_credit) / float(cache_per_page) if cache_per_page else 0.0
            ),
            "net_release_credit_pages": (
                float(net_credit) / float(cache_per_page) if cache_per_page else 0.0
            ),
            "net_release_credit_tokens": (
                int(net_credit // cache_per_page) * int(config.page_size) if cache_per_page else 0
            ),
        }

    def _planned_marlin_wna16_guard_bytes(self, source_bytes: int) -> int:
        return 0

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
        if not self.graph_runner.can_use_cuda_graph(batch):
            # Eager and prefill use the same semantic contract as graph replay.
            # Their input tensors are exact-sized, so no row masking occurs.
            batch.num_token_non_padded = torch.tensor(
                [batch.input_ids.numel()],
                dtype=torch.int32,
                device=batch.input_ids.device,
            )
        with self.ctx.forward_batch(batch):
            if args.temperatures is None and self.graph_runner.can_replay_greedy_sample(batch):
                next_tokens_gpu = self.graph_runner.replay_greedy_sample(batch)
            elif self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                self.graph_runner.record_eager_decode(batch)
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        if next_tokens_gpu is None:
            assert logits is not None
            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args, batch).to(torch.int32)
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


def _marlin_wna16_credit_ineligible_reason(
    *,
    config: EngineConfig,
    requested: bool,
    release_requested: bool,
    timing: str,
    source_bytes: int,
) -> str:
    if not config.model_config.is_deepseek_v4:
        return "not_deepseek_v4"
    if not requested:
        return "release_credit_not_requested"
    if not release_requested:
        return "release_not_requested"
    if timing == "model_prepare":
        return "model_prepare_release_already_reflected_in_free_memory"
    if timing != "before_kv_alloc":
        return "release_timing_cannot_back_pre_kv_pages"
    if source_bytes <= 0:
        return "no_releasable_source_bytes_reported"
    return "unknown"


def _pynccl_max_buffer_bytes(config: EngineConfig) -> int:
    max_bytes = (
        config.max_forward_len
        * config.model_config.hidden_size
        * torch.bfloat16.itemsize
    )
    if _use_dsv4_sm80_default_pynccl_threshold(config):
        return min(max_bytes, DSV4_RELEASE.pynccl_max_buffer_bytes)
    return max_bytes


def _use_dsv4_sm80_default_pynccl_threshold(config: EngineConfig) -> bool:
    if config.tp_info.size <= 1 or not config.use_pynccl:
        return False
    if not config.model_config.is_deepseek_v4:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        capability = torch.cuda.get_device_capability()
    except Exception:
        return False
    return tuple(int(part) for part in capability) == (8, 0)


def _resolve_cuda_graph_policy(
    config: EngineConfig, *, free_memory: int | None = None
) -> ResolvedCudaGraphBucketPolicy:
    existing = getattr(config, "cuda_graph_policy", None)
    if existing is not None:
        return existing
    is_dsv4 = bool(config.model_config.is_deepseek_v4)
    disabled = bool(getattr(config, "disable_cuda_graph", False)) or (
        is_dsv4 and not bool(config.allow_dsv4_cuda_graph)
    )
    legacy_default_max = None
    if not is_dsv4 and config.cuda_graph_bs is None and config.cuda_graph_max_bs is None:
        if free_memory is None:
            raise RuntimeError("generic CUDA graph auto policy requires the free-memory snapshot")
        legacy_default_max = 256 if free_memory / (1 << 30) > 80 else 160
    policy = resolve_cuda_graph_bucket_policy(
        cuda_graph_bs=config.cuda_graph_bs,
        cuda_graph_max_bs=config.cuda_graph_max_bs,
        effective_max_running_req=int(getattr(config, "max_running_req", 256)),
        graph_disabled=disabled,
        release_default_bs=(
            _DSV4_SM80_RELEASE_CUDA_GRAPH_BS
            if is_dsv4
            and bool(config.allow_dsv4_cuda_graph)
            and getattr(config, "dsv4_sm80_recipe", None) is None
            else None
        ),
        legacy_default_max_bs=legacy_default_max,
    )
    object.__setattr__(config, "cuda_graph_policy", policy)
    object.__setattr__(config, "cuda_graph_bs", list(policy.resolved_bs))
    object.__setattr__(config, "cuda_graph_max_bs", policy.resolved_max_bs)
    logger.info_rank0(
        "CUDA graph policy: "
        f"mode={policy.source_mode}, buckets={len(policy.resolved_bs)}, "
        f"max_bs={policy.resolved_max_bs}."
    )
    return policy


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if not config.model_config.is_deepseek_v4:
        raise ValueError("This release supports DeepSeek V4 Flash only.")
    if config.attention_backend not in ("auto", "dsv4"):
        raise ValueError(
            f"Attention backend {config.attention_backend!r} is not supported; "
            "this release supports the DSV4 attention backend only."
        )

    if config.model_config.is_deepseek_v4:
        requested_recipe = getattr(config, "dsv4_sm80_recipe", None)
        if requested_recipe is not None and requested_recipe not in _DSV4_SM80_RECIPES:
            raise ValueError(
                f"Unknown dsv4_sm80_recipe={requested_recipe!r}; supported recipes are "
                f"{sorted(_DSV4_SM80_RECIPES)}."
            )
        recipe_name = requested_recipe
        if recipe_name is not None:
            recipe_max_req, recipe_graph_max, recipe_max_seq = _DSV4_SM80_RECIPES[recipe_name]
            manual_overrides = []
            if bool(getattr(config, "max_running_req_explicit", False)):
                manual_overrides.append(f"max_running_req={config.max_running_req}")
            if config.cuda_graph_bs is not None:
                manual_overrides.append(f"cuda_graph_bs={config.cuda_graph_bs}")
            elif config.cuda_graph_max_bs is not None:
                manual_overrides.append(f"cuda_graph_max_bs={config.cuda_graph_max_bs}")
            if getattr(config, "context_length", None) is not None:
                manual_overrides.append(f"context_length={config.context_length}")
            if not bool(getattr(config, "max_running_req_explicit", False)):
                override("max_running_req", recipe_max_req)
            if config.cuda_graph_bs is None and config.cuda_graph_max_bs is None:
                override("cuda_graph_max_bs", min(recipe_graph_max, config.max_running_req))
            if recipe_max_seq is not None and getattr(config, "context_length", None) is None:
                override("context_length", recipe_max_seq)
            if recipe_name != "default_m128":
                logger.warning_rank0(
                    f"Applying DeepSeek V4 recipe {recipe_name!r}, validated on one "
                    "DGX A100 8x80GB system: "
                    f"max_running_req={recipe_max_req}, "
                    f"cuda_graph_max_bs={recipe_graph_max}, "
                    f"context_length={recipe_max_seq}. "
                    + (
                        "Explicit settings override recipe fields: "
                        + ", ".join(manual_overrides)
                        + "."
                        if manual_overrides
                        else "No explicit field overrides were detected."
                    )
                )
        elif config.cuda_graph_bs is None and config.cuda_graph_max_bs is None:
            override(
                "cuda_graph_max_bs",
                min(_DSV4_SM80_DEFAULT_CUDA_GRAPH_MAX_BS, config.max_running_req),
            )
        if config.page_size == 1:
            override("page_size", 256)
        if hasattr(config, "cache_type") and getattr(config, "cache_type") != "radix":
            override("cache_type", "radix")
        if hasattr(config, "enable_dsv4_radix_prefix_cache") and not getattr(
            config, "enable_dsv4_radix_prefix_cache"
        ):
            override("enable_dsv4_radix_prefix_cache", True)
        if hasattr(config, "enable_dsv4_component_loc_ownership") and not getattr(
            config, "enable_dsv4_component_loc_ownership"
        ):
            override("enable_dsv4_component_loc_ownership", True)
        if hasattr(config, "enable_dsv4_swa_independent_lifecycle") and not getattr(
            config, "enable_dsv4_swa_independent_lifecycle"
        ):
            override("enable_dsv4_swa_independent_lifecycle", True)
        max_extend_tokens = getattr(config, "max_extend_tokens", None)
        max_extend_tokens_explicit = bool(getattr(config, "max_extend_tokens_explicit", False))
        if max_extend_tokens is None or (
            max_extend_tokens == _GENERIC_DEFAULT_MAX_EXTEND_TOKENS
            and not max_extend_tokens_explicit
        ):
            override("max_extend_tokens", _DSV4_SM80_DEFAULT_MAX_EXTEND_TOKENS)
        graph_explicitly_disabled = (
            bool(getattr(config, "disable_cuda_graph", False))
            or config.cuda_graph_bs == []
            or config.cuda_graph_max_bs == 0
        )
        if not graph_explicitly_disabled and not config.allow_dsv4_cuda_graph:
            override("allow_dsv4_cuda_graph", True)
        communication = (
            "PyNCCL threshold=32 MiB" if bool(getattr(config, "use_pynccl", True)) else "NCCL"
        )
        logger.info_rank0(
            "Resolved DeepSeek V4 runtime parameters: "
            f"page_size={config.page_size}, max_running_req={config.max_running_req}, "
            f"cuda_graph_max_bs={config.cuda_graph_max_bs}, "
            f"max_prefill_tokens={config.max_extend_tokens}, communication={communication}."
        )
        reasoning_contract_enabled = bool(
            getattr(
                config,
                "reasoning_sampler_contract_enabled",
                getattr(config, "enable_reasoning_sampler_contract", False),
            )
        )
        if reasoning_contract_enabled:
            logger.warning_rank0(
                "DeepSeek V4 reasoning sampler contract is ENABLED. It masks "
                "protocol delimiters and EOS according to request state, changing "
                "the model's raw sampling distribution."
            )
        if config.attention_backend == "auto":
            override("attention_backend", "dsv4")
        if config.allow_dsv4_cuda_graph:
            override("cuda_graph_capture_fail_open", True)
        _resolve_cuda_graph_policy(config)

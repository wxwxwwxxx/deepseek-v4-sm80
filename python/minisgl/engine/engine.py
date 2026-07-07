from __future__ import annotations

import os
import time
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
    dsv4_memory_debug,
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

_PYNCCL_MAX_BUFFER_SIZE_ENV = "MINISGL_PYNCCL_MAX_BUFFER_SIZE"
_DSV4_SM80_DEFAULT_PYNCCL_MAX_BYTES = 32 * 1024 * 1024
_MARLIN_WNA16_RELEASE_ENV = "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
_MARLIN_WNA16_RELEASE_TIMING_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING"
_MARLIN_WNA16_RELEASE_AFTER_GRAPH_CAPTURE_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_AFTER_GRAPH_CAPTURE"
)
_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT_ENV = "MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT"
_MARLIN_WNA16_RELEASE_CREDIT_SAFETY_MARGIN_BYTES_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_RELEASE_CREDIT_SAFETY_MARGIN_BYTES"
)
_MARLIN_WNA16_QUARANTINE_BLOCKS_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS"
_MARLIN_WNA16_QUARANTINE_BYTES_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_BYTES"
_DSV4_CASE_BOUNDARY_DEBUG_ENV = "MINISGL_DSV4_CASE_BOUNDARY_DEBUG"
_DSV4_EXPERIMENTAL_MTP_ENV = "MINISGL_DSV4_EXPERIMENTAL_MTP"
_DSV4_MTP_SPECULATIVE_ENV = "MINISGL_DSV4_MTP_SPECULATIVE"
_DSV4_MTP_SPEC_DRAFT_LEN_ENV = "MINISGL_DSV4_MTP_SPEC_DRAFT_LEN"
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _case_boundary_debug_enabled() -> bool:
    return os.environ.get(_DSV4_CASE_BOUNDARY_DEBUG_ENV, "").strip().lower() in _TRUE_ENV_VALUES


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


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
        self._marlin_wna16_debug_release_done = False
        self._marlin_wna16_decode_guard_checks = 0
        self.enable_dsv4_mtp_speculative = bool(
            getattr(config, "enable_dsv4_mtp_speculative", False)
            or _env_flag(_DSV4_MTP_SPECULATIVE_ENV)
        )
        self.dsv4_mtp_spec_draft_len = int(
            getattr(config, "dsv4_mtp_spec_draft_len", 1) or 1
        )
        self._mtp_pending_draft_tokens: dict[int, int] = {}
        self.mtp_spec_stats: dict[str, Any] = {
            "enabled": bool(self.enable_dsv4_mtp_speculative),
            "draft_len": int(self.dsv4_mtp_spec_draft_len),
            "draft_tokens_proposed": 0,
            "draft_tokens_verified": 0,
            "draft_tokens_accepted": 0,
            "draft_tokens_rejected": 0,
            "acceptance_histogram": {"0": 0, "1": 0},
            "target_calls": 0,
            "draft_calls": 0,
            "target_latency_s": 0.0,
            "draft_latency_s": 0.0,
            "finite_failures": 0,
            "fallback_sampling_batches": 0,
            "fallback_missing_mtp_batches": 0,
            "last_batch": {},
        }
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
        self._record_marlin_wna16_owner_allocations("after_model_prepare")
        self._audit_marlin_wna16_cache_integrity("after_model_prepare")
        self._check_marlin_wna16_release_guards("after_model_prepare")

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        self._audit_marlin_wna16_cache_integrity("after_kv_capacity_empty_cache")
        self._maybe_release_marlin_wna16_for_timing(
            timing="before_kv_alloc",
            stage_label="before_kv_alloc_release",
        )
        self._check_marlin_wna16_release_guards("after_before_kv_alloc_release")
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
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
        self._record_marlin_wna16_owner_allocations("after_kv_alloc")
        self._check_marlin_wna16_release_guards("after_kv_alloc")
        self._check_marlin_wna16_kv_sentinels("after_kv_alloc")
        self._maybe_release_marlin_wna16_for_timing(
            timing="after_kv_alloc",
            stage_label="after_kv_alloc_release",
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
        dsv4_memory_debug.record_owner_tensor(
            owner_label="engine.page_table",
            stage="after_page_table_alloc",
            tensor=self.page_table,
            include_integrity=False,
            extra={
                "max_seq_len": int(self.max_seq_len),
                "aligned_max_seq_len": int(aligned_max_seq_len),
            },
        )
        self._check_marlin_wna16_release_guards("after_page_table_alloc")
        self._check_marlin_wna16_kv_sentinels("after_page_table_alloc")

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)
        self._check_marlin_wna16_release_guards("after_attention_backend_init")
        self._check_marlin_wna16_kv_sentinels("after_attention_backend_init")

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
        self._record_marlin_wna16_owner_allocations("after_graph_runner_init")
        self._maybe_release_marlin_wna16_for_timing(
            timing="after_graph_capture",
            stage_label="after_graph_capture_release",
        )
        self._audit_marlin_wna16_cache_integrity("after_graph_runner_init")
        self._check_marlin_wna16_release_guards("after_graph_runner_init")
        self._check_marlin_wna16_kv_sentinels("after_graph_runner_init")

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
            max_bytes = _pynccl_max_buffer_bytes(config, self.dtype)
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
        if dsv4_memory_debug.env_flag(_MARLIN_WNA16_RELEASE_AFTER_GRAPH_CAPTURE_ENV):
            return "after_graph_capture"
        raw = os.environ.get(_MARLIN_WNA16_RELEASE_TIMING_ENV, "model_prepare").strip().lower()
        aliases = {
            "": "model_prepare",
            "immediate": "model_prepare",
            "after_prebuild": "model_prepare",
            "after_full_model_prebuild": "model_prepare",
            "model_prepare": "model_prepare",
            "before_kv": "before_kv_alloc",
            "before_kv_alloc": "before_kv_alloc",
            "before_kv_allocation": "before_kv_alloc",
            "after_kv": "after_kv_alloc",
            "after_kv_alloc": "after_kv_alloc",
            "after_kv_allocation": "after_kv_alloc",
            "before_warmup": "before_warmup_forward",
            "before_warmup_forward": "before_warmup_forward",
            "after_warmup": "after_warmup_forward",
            "after_warmup_forward": "after_warmup_forward",
            "after_graph": "after_graph_capture",
            "after_graph_capture": "after_graph_capture",
            "after_first_decode": "after_first_decode",
            "after_decode_step1": "after_first_decode",
        }
        return aliases.get(raw, raw)

    def _maybe_release_marlin_wna16_for_timing(self, *, timing: str, stage_label: str) -> None:
        if self._marlin_wna16_debug_release_done:
            return
        if not dsv4_memory_debug.env_flag(_MARLIN_WNA16_RELEASE_ENV):
            return
        if self._marlin_wna16_release_timing() != timing:
            return
        release = getattr(self.model, "release_marlin_wna16_original_expert_weights", None)
        if not callable(release):
            return
        report = release(stage_label=stage_label)
        self._marlin_wna16_debug_release_done = True
        self.model_prepare_report[f"moe_marlin_wna16_{stage_label}"] = report
        self._record_marlin_wna16_owner_allocations(f"{stage_label}:after")
        self._check_marlin_wna16_release_guards(f"{stage_label}:after")

    def _check_marlin_wna16_release_guards(self, stage: str) -> None:
        check = getattr(self.model, "check_marlin_wna16_release_guards", None)
        if not callable(check):
            return
        report = check(stage)
        if not isinstance(report, dict) or not report.get("enabled", False):
            return
        key = f"moe_marlin_wna16_guard_{_sanitize_report_key(stage)}"
        self.model_prepare_report[key] = {k: v for k, v in report.items() if k != "records"}
        mutated = int(report.get("mutated_count", 0) or 0)
        if mutated:
            logger.error(
                "Marlin WNA16 release guard mutation detected at "
                f"{stage}: mutated_count={mutated}"
            )

    def _check_marlin_wna16_kv_sentinels(self, stage: str) -> None:
        kv_cache = getattr(self, "kv_cache", None)
        check = getattr(kv_cache, "check_marlin_wna16_kv_sentinels", None)
        if not callable(check):
            return
        report = check(stage)
        if not isinstance(report, dict) or not report.get("enabled", False):
            return
        key = f"moe_marlin_wna16_kv_sentinel_{_sanitize_report_key(stage)}"
        self.model_prepare_report[key] = {k: v for k, v in report.items() if k != "records"}
        mutated = int(report.get("mutated_count", 0) or 0)
        if mutated:
            logger.error(
                "Marlin WNA16 KV sentinel mutation detected at " f"{stage}: mutated_count={mutated}"
            )

    def _record_marlin_wna16_owner_allocations(self, stage: str) -> None:
        if not dsv4_memory_debug.marlin_wna16_release_ledger_enabled():
            return
        model_record = getattr(self.model, "record_marlin_wna16_owner_allocations", None)
        if callable(model_record):
            model_record(stage)
        kv_cache = getattr(self, "kv_cache", None)
        kv_record = getattr(kv_cache, "record_marlin_wna16_owner_allocations", None)
        if callable(kv_record):
            kv_record(stage)
        page_table = getattr(self, "page_table", None)
        if isinstance(page_table, torch.Tensor):
            dsv4_memory_debug.record_owner_tensor(
                owner_label="engine.page_table",
                stage=stage,
                tensor=page_table,
            )
        graph_runner = getattr(self, "graph_runner", None)
        graph_record = getattr(graph_runner, "record_marlin_wna16_owner_allocations", None)
        if callable(graph_record):
            graph_record(stage)

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            if config.model_config.is_deepseek_v4:
                return dict(
                    load_weight(
                        config.model_path,
                        self.device,
                        enable_dsv4_mtp=bool(
                            getattr(config, "enable_dsv4_mtp", False)
                            or _env_flag(_DSV4_EXPERIMENTAL_MTP_ENV)
                        ),
                    )
                )
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = estimate_kvcache_bytes_per_page(
            config.model_config,
            page_size=config.page_size,
            dtype=self.dtype,
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
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            credit_bytes = int(credit_report.get("net_release_credit_bytes", 0) or 0)
            if bool(credit_report.get("applied_to_num_pages", False)):
                available_memory += credit_bytes
            num_pages = (available_memory - fixed_swa_cache_bytes) // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page + fixed_swa_cache_bytes
        credit_report["planned_num_pages"] = int(num_pages)
        credit_report["planned_num_tokens"] = int(num_tokens)
        credit_report["planned_kv_bytes"] = int(real_kv_size)
        self.kv_capacity_plan_report = {
            "old_free_memory_bytes": int(old_free_memory),
            "new_free_memory_bytes": int(new_free_memory),
            "memory_ratio": float(config.memory_ratio),
            "cache_per_page_bytes": int(cache_per_page),
            "legacy_cache_per_page_bytes": int(legacy_cache_per_page),
            "fixed_swa_cache_bytes": int(fixed_swa_cache_bytes),
            "num_page_override": config.num_page_override,
            "release_credit": credit_report,
        }
        if bool(credit_report.get("applied_to_num_pages", False)):
            logger.info_rank0(
                "Applied Marlin WNA16 release credit before KV allocation: "
                f"{mem_GB(int(credit_report['net_release_credit_bytes']))}, "
                f"equivalent_pages={credit_report.get('net_release_credit_pages')}"
            )
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _dsv4_swa_independent_enabled(self, config: EngineConfig) -> bool:
        return bool(getattr(config, "enable_dsv4_swa_independent_lifecycle", False)) or (
            os.environ.get("MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    def _planned_dsv4_swa_independent_pages(self, config: EngineConfig) -> int:
        env_pages = os.environ.get("MINISGL_DSV4_SWA_INDEPENDENT_NUM_PAGES")
        if env_pages:
            try:
                parsed = int(env_pages)
            except ValueError:
                parsed = 0
            if parsed > 0:
                return parsed
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
        requested = dsv4_memory_debug.env_flag(_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT_ENV)
        release_requested = dsv4_memory_debug.env_flag(_MARLIN_WNA16_RELEASE_ENV)
        timing = self._marlin_wna16_release_timing()
        moe_report = {}
        if isinstance(self.model_prepare_report, dict):
            maybe_report = self.model_prepare_report.get("moe_marlin_wna16_cache", {})
            if isinstance(maybe_report, dict):
                moe_report = maybe_report
        source_bytes = int(moe_report.get("total_source_bytes", 0) or 0)
        guard_bytes = self._planned_marlin_wna16_guard_bytes(source_bytes)
        safety_margin = _env_bytes(_MARLIN_WNA16_RELEASE_CREDIT_SAFETY_MARGIN_BYTES_ENV, 0) or 0
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
            "ineligible_reason": (
                None
                if eligible
                else _marlin_wna16_credit_ineligible_reason(
                    config=config,
                    requested=requested,
                    release_requested=release_requested,
                    timing=timing,
                    source_bytes=source_bytes,
                )
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
        if source_bytes <= 0 or not dsv4_memory_debug.env_flag(_MARLIN_WNA16_QUARANTINE_BLOCKS_ENV):
            return 0
        raw = _env_bytes(_MARLIN_WNA16_QUARANTINE_BYTES_ENV, None)
        if raw is None:
            return int(source_bytes)
        return max(0, min(int(raw), int(source_bytes)))

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

    def _audit_marlin_wna16_cache_integrity(self, stage: str) -> None:
        audit = getattr(self.model, "audit_marlin_wna16_cache_integrity", None)
        if callable(audit):
            audit(stage)

    def _debug_forward_context(self, batch: Batch, forward_source: str) -> dict[str, Any]:
        def _tensor_range(tensor: torch.Tensor | None) -> tuple[int | None, int | None] | str:
            if tensor is None:
                return None, None
            try:
                if tensor.numel() == 0:
                    return None, None
                return int(tensor.min().item()), int(tensor.max().item())
            except Exception as exc:
                return f"{type(exc).__name__}: {exc}"

        reqs = []
        for req in getattr(batch, "reqs", [])[:4]:
            reqs.append(
                {
                    "uid": int(getattr(req, "uid", -1)),
                    "table_idx": int(getattr(req, "table_idx", -1)),
                    "cached_len": int(getattr(req, "cached_len", -1)),
                    "device_len": int(getattr(req, "device_len", -1)),
                    "extend_len": int(getattr(req, "extend_len", -1)),
                }
            )
        graph_status = getattr(getattr(self, "graph_runner", None), "capture_status", {})
        return {
            "stage_source": forward_source,
            "phase": getattr(batch, "phase", "unknown"),
            "bs": int(getattr(batch, "size", -1)),
            "padded": int(getattr(batch, "padded_size", getattr(batch, "size", -1))),
            "positions_range": _tensor_range(getattr(batch, "positions", None)),
            "out_loc_range": _tensor_range(getattr(batch, "out_loc", None)),
            "graph_replay_count": (
                int(graph_status.get("replay_count", 0)) if isinstance(graph_status, dict) else 0
            ),
            "graph_greedy_replay_count": (
                int(graph_status.get("greedy_sample_replay_count", 0))
                if isinstance(graph_status, dict)
                else 0
            ),
            "eager_decode_count": (
                int(graph_status.get("eager_decode_count", 0))
                if isinstance(graph_status, dict)
                else 0
            ),
            "reqs": reqs,
        }

    def _debug_sync_forward(self, stage: str, batch: Batch, forward_source: str) -> None:
        if not _case_boundary_debug_enabled() or self.device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            context = self._debug_forward_context(batch, forward_source)
            raise RuntimeError(
                "DSV4 forward failed during case-boundary debug synchronize: "
                f"stage={stage}, context={context}"
            ) from exc

    def _sync_device_for_mtp_spec(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _can_run_mtp_spec_greedy(self, batch: Batch, args: BatchSamplingArgs) -> bool:
        if not self.enable_dsv4_mtp_speculative:
            return False
        if args.temperatures is not None:
            self.mtp_spec_stats["fallback_sampling_batches"] = int(
                self.mtp_spec_stats["fallback_sampling_batches"]
            ) + 1
            return False
        has_mtp = callable(getattr(self.model, "mtp_forward_one_step", None)) and (
            getattr(self.model, "mtp", None) is not None
        )
        has_hidden_forward = callable(getattr(self.model, "forward_with_hidden", None))
        if not (has_mtp and has_hidden_forward):
            self.mtp_spec_stats["fallback_missing_mtp_batches"] = int(
                self.mtp_spec_stats["fallback_missing_mtp_batches"]
            ) + 1
            return False
        if int(self.dsv4_mtp_spec_draft_len) != 1:
            raise RuntimeError(
                "DeepSeek V4 MTP speculative V1 only supports draft_len=1, got "
                f"{self.dsv4_mtp_spec_draft_len}."
            )
        return batch.size > 0

    def _record_mtp_spec_verification(
        self,
        batch: Batch,
        next_tokens_gpu: torch.Tensor,
    ) -> dict[str, int]:
        target_tokens = [int(x) for x in next_tokens_gpu.detach().cpu().tolist()]
        accepted = 0
        rejected = 0
        verified = 0
        for req, target_token in zip(batch.reqs, target_tokens):
            draft_token = self._mtp_pending_draft_tokens.pop(int(req.uid), None)
            if draft_token is None:
                continue
            verified += 1
            if int(draft_token) == int(target_token):
                accepted += 1
                hist_key = "1"
            else:
                rejected += 1
                hist_key = "0"
            histogram = self.mtp_spec_stats["acceptance_histogram"]
            histogram[hist_key] = int(histogram.get(hist_key, 0)) + 1

        self.mtp_spec_stats["draft_tokens_verified"] = int(
            self.mtp_spec_stats["draft_tokens_verified"]
        ) + verified
        self.mtp_spec_stats["draft_tokens_accepted"] = int(
            self.mtp_spec_stats["draft_tokens_accepted"]
        ) + accepted
        self.mtp_spec_stats["draft_tokens_rejected"] = int(
            self.mtp_spec_stats["draft_tokens_rejected"]
        ) + rejected
        return {"verified": verified, "accepted": accepted, "rejected": rejected}

    def _propose_mtp_spec_drafts(
        self,
        batch: Batch,
        next_tokens_gpu: torch.Tensor,
        hidden_states_before_norm: torch.Tensor,
        mtp_positions: list[int],
    ) -> dict[str, Any]:
        eligible: list[tuple[int, Req]] = [
            (i, req) for i, req in enumerate(batch.reqs) if bool(req.can_decode)
        ]
        for req in batch.reqs:
            if not req.can_decode:
                self._mtp_pending_draft_tokens.pop(int(req.uid), None)
        if not eligible:
            return {"proposed": 0, "finite": True}

        indices = torch.tensor(
            [i for i, _ in eligible],
            dtype=torch.long,
            device=self.device,
        )
        eligible_reqs = [req for _, req in eligible]
        draft_input = next_tokens_gpu.index_select(0, indices).contiguous()
        draft_hidden = hidden_states_before_norm.index_select(0, indices).contiguous()

        mtp_batch = Batch(reqs=eligible_reqs, phase="decode")
        mtp_batch.padded_reqs = eligible_reqs
        mtp_batch.input_ids = draft_input
        mtp_batch.positions = torch.tensor(
            [int(mtp_positions[i]) for i, _ in eligible],
            dtype=torch.int32,
            device=self.device,
        )
        # No DSV4AttentionMetadata here: V1 lets the draft run as a no-store
        # sidecar so rejected draft tokens cannot mutate target KV/cache state.
        mtp_batch.out_loc = torch.empty(len(eligible_reqs), dtype=torch.int32, device=self.device)

        self._sync_device_for_mtp_spec()
        start_s = time.perf_counter()
        with self.ctx.forward_batch(mtp_batch):
            mtp_output = self.model.mtp_forward_one_step(draft_input, draft_hidden)
            self._debug_sync_forward("after_mtp_forward", mtp_batch, "mtp_spec_no_store_draft")
        self._sync_device_for_mtp_spec()
        self.mtp_spec_stats["draft_latency_s"] = float(
            self.mtp_spec_stats["draft_latency_s"]
        ) + (time.perf_counter() - start_s)
        self.mtp_spec_stats["draft_calls"] = int(self.mtp_spec_stats["draft_calls"]) + 1

        mtp_logits = mtp_output.logits[: len(eligible_reqs)]
        finite = bool(torch.isfinite(mtp_logits).all().item())
        if not finite:
            self.mtp_spec_stats["finite_failures"] = int(
                self.mtp_spec_stats["finite_failures"]
            ) + 1
            raise RuntimeError("DeepSeek V4 MTP speculative draft produced NaN/Inf logits.")

        draft_tokens = torch.argmax(mtp_logits, dim=-1).to(torch.int32)
        draft_tokens_cpu = [int(x) for x in draft_tokens.detach().cpu().tolist()]
        for req, draft_token in zip(eligible_reqs, draft_tokens_cpu):
            self._mtp_pending_draft_tokens[int(req.uid)] = int(draft_token)

        proposed = len(eligible_reqs)
        self.mtp_spec_stats["draft_tokens_proposed"] = int(
            self.mtp_spec_stats["draft_tokens_proposed"]
        ) + proposed
        return {
            "proposed": proposed,
            "finite": finite,
            "draft_tokens": draft_tokens_cpu,
        }

    def _forward_batch_mtp_spec_greedy(
        self,
        batch: Batch,
        args: BatchSamplingArgs,
    ) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        forward_source = "mtp_spec_eager_target"

        if batch.is_decode:
            self._marlin_wna16_decode_guard_checks += 1
            self._check_marlin_wna16_release_guards(
                f"before_decode_step_{self._marlin_wna16_decode_guard_checks}"
            )
            self._check_marlin_wna16_kv_sentinels(
                f"before_decode_step_{self._marlin_wna16_decode_guard_checks}"
            )

        mtp_positions = [int(req.device_len) - 1 for req in batch.reqs]
        self._sync_device_for_mtp_spec()
        start_s = time.perf_counter()
        with self.ctx.forward_batch(batch):
            self.graph_runner.record_eager_decode(batch)
            target_output = self.model.forward_with_hidden()
            self._debug_sync_forward("after_model_forward", batch, forward_source)
        self._sync_device_for_mtp_spec()
        self.mtp_spec_stats["target_latency_s"] = float(
            self.mtp_spec_stats["target_latency_s"]
        ) + (time.perf_counter() - start_s)
        self.mtp_spec_stats["target_calls"] = int(self.mtp_spec_stats["target_calls"]) + 1

        logits = target_output.logits[: batch.size]
        if not bool(torch.isfinite(logits).all().item()):
            self.mtp_spec_stats["finite_failures"] = int(
                self.mtp_spec_stats["finite_failures"]
            ) + 1
            raise RuntimeError("DeepSeek V4 MTP speculative target produced NaN/Inf logits.")

        debug_recorder = dsv4_prefix_debug.get_dsv4_prefix_debug_recorder()
        forward_stage = (
            f"{batch.phase}_bs{int(batch.size)}"
            f"_padded{int(getattr(batch, 'padded_size', batch.size))}_{forward_source}"
        )
        dsv4_memory_debug.record_owner_tensor(
            owner_label="engine.forward.logits",
            stage=forward_stage,
            tensor=logits,
            include_integrity=False,
        )
        dsv4_memory_debug.record_owner_tensors(
            owner_prefix="engine.sampler.args",
            stage=forward_stage,
            tensors={
                "temperatures": args.temperatures,
                "top_k": args.top_k,
                "top_p": args.top_p,
            },
        )
        debug_snapshot = (
            debug_recorder.capture_pre_sample(
                batch=batch,
                logits=logits,
                forward_source=forward_source,
            )
            if debug_recorder is not None
            else None
        )

        for req in batch.reqs:
            req.complete_one()

        with dsv4_direct_copy_nvtx(
            f"sampler_logits_staging.sample_to_int32.bs{batch.size}",
            logits=logits,
        ):
            next_tokens_gpu = self.sampler.sample(logits, args).to(torch.int32)
        self._debug_sync_forward("after_sampler", batch, forward_source)

        verification = self._record_mtp_spec_verification(batch, next_tokens_gpu)
        draft = self._propose_mtp_spec_drafts(
            batch,
            next_tokens_gpu,
            target_output.hidden_states_before_norm[: batch.size],
            mtp_positions,
        )
        self.mtp_spec_stats["last_batch"] = {
            "phase": batch.phase,
            "batch_size": int(batch.size),
            "verified": int(verification["verified"]),
            "accepted": int(verification["accepted"]),
            "rejected": int(verification["rejected"]),
            "proposed": int(draft["proposed"]),
        }

        dsv4_memory_debug.record_owner_tensor(
            owner_label="engine.sampler.next_tokens_gpu",
            stage=forward_stage,
            tensor=next_tokens_gpu,
            include_integrity=False,
        )
        if debug_recorder is not None:
            debug_recorder.finish(
                debug_snapshot,
                next_tokens=next_tokens_gpu,
                graph_runner=getattr(self.graph_runner, "capture_status", {}),
            )
        if batch.is_decode:
            self._check_marlin_wna16_release_guards(
                f"after_decode_step_{self._marlin_wna16_decode_guard_checks}"
            )
            self._check_marlin_wna16_kv_sentinels(
                f"after_decode_step_{self._marlin_wna16_decode_guard_checks}"
            )
            self._maybe_release_marlin_wna16_for_timing(
                timing="after_first_decode",
                stage_label="after_first_decode_release",
            )
        with dsv4_direct_copy_nvtx(
            f"sampler_logits_staging.next_tokens_to_cpu.bs{batch.size}",
            next_tokens=next_tokens_gpu,
        ):
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = self._acquire_copy_done_event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        if self._can_run_mtp_spec_greedy(batch, args):
            return self._forward_batch_mtp_spec_greedy(batch, args)

        next_tokens_gpu: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        forward_source = "unknown"
        with self.ctx.forward_batch(batch):
            if batch.is_decode:
                self._marlin_wna16_decode_guard_checks += 1
                self._check_marlin_wna16_release_guards(
                    f"before_decode_step_{self._marlin_wna16_decode_guard_checks}"
                )
                self._check_marlin_wna16_kv_sentinels(
                    f"before_decode_step_{self._marlin_wna16_decode_guard_checks}"
                )
            if args.temperatures is None and self.graph_runner.can_replay_greedy_sample(batch):
                forward_source = "cuda_graph_greedy_sample"
                next_tokens_gpu = self.graph_runner.replay_greedy_sample(batch)
            elif self.graph_runner.can_use_cuda_graph(batch):
                forward_source = "cuda_graph_replay"
                logits = self.graph_runner.replay(batch)
            else:
                forward_source = "eager"
                self.graph_runner.record_eager_decode(batch)
                logits = self.model.forward()
            self._debug_sync_forward("after_model_forward", batch, forward_source)

        debug_recorder = dsv4_prefix_debug.get_dsv4_prefix_debug_recorder()
        forward_stage = (
            f"{batch.phase}_bs{int(batch.size)}"
            f"_padded{int(getattr(batch, 'padded_size', batch.size))}_{forward_source}"
        )
        if logits is not None:
            dsv4_memory_debug.record_owner_tensor(
                owner_label="engine.forward.logits",
                stage=forward_stage,
                tensor=logits[: batch.size],
                include_integrity=False,
            )
        dsv4_memory_debug.record_owner_tensors(
            owner_prefix="engine.sampler.args",
            stage=forward_stage,
            tensors={
                "temperatures": args.temperatures,
                "top_k": args.top_k,
                "top_p": args.top_p,
            },
        )
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
            self._debug_sync_forward("after_sampler", batch, forward_source)
        dsv4_memory_debug.record_owner_tensor(
            owner_label="engine.sampler.next_tokens_gpu",
            stage=forward_stage,
            tensor=next_tokens_gpu,
            include_integrity=False,
        )
        if debug_recorder is not None:
            debug_recorder.finish(
                debug_snapshot,
                next_tokens=next_tokens_gpu,
                graph_runner=getattr(self.graph_runner, "capture_status", {}),
            )
        if batch.is_decode:
            self._check_marlin_wna16_release_guards(
                f"after_decode_step_{self._marlin_wna16_decode_guard_checks}"
            )
            self._check_marlin_wna16_kv_sentinels(
                f"after_decode_step_{self._marlin_wna16_decode_guard_checks}"
            )
            self._maybe_release_marlin_wna16_for_timing(
                timing="after_first_decode",
                stage_label="after_first_decode_release",
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


def _sanitize_report_key(stage: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in stage).strip("_") or "stage"


def _env_bytes(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    token = raw.strip().lower()
    multipliers = {
        "kib": 1 << 10,
        "kb": 1 << 10,
        "mib": 1 << 20,
        "mb": 1 << 20,
        "gib": 1 << 30,
        "gb": 1 << 30,
    }
    for suffix, multiplier in multipliers.items():
        if token.endswith(suffix):
            try:
                return int(float(token[: -len(suffix)]) * multiplier)
            except ValueError:
                return default
    try:
        return int(token)
    except ValueError:
        return default


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


def _pynccl_max_buffer_bytes(config: EngineConfig, dtype: torch.dtype) -> int:
    max_bytes = config.max_forward_len * config.model_config.hidden_size * dtype.itemsize
    if _use_dsv4_sm80_default_pynccl_threshold(config):
        logger.info_rank0(
            "Defaulting DeepSeek V4 sm80 PyNCCL max buffer size to 32 MiB; "
            f"set {_PYNCCL_MAX_BUFFER_SIZE_ENV} to override."
        )
        return min(max_bytes, _DSV4_SM80_DEFAULT_PYNCCL_MAX_BYTES)
    return max_bytes


def _use_dsv4_sm80_default_pynccl_threshold(config: EngineConfig) -> bool:
    if config.tp_info.size <= 1 or not config.use_pynccl:
        return False
    if _PYNCCL_MAX_BUFFER_SIZE_ENV in os.environ:
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


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.model_config.is_deepseek_v4:
        spec_enabled = bool(getattr(config, "enable_dsv4_mtp_speculative", False)) or _env_flag(
            _DSV4_MTP_SPECULATIVE_ENV
        )
        env_draft_len = os.environ.get(_DSV4_MTP_SPEC_DRAFT_LEN_ENV)
        draft_len = int(
            env_draft_len
            if env_draft_len not in (None, "")
            else getattr(config, "dsv4_mtp_spec_draft_len", 1)
        )
        if spec_enabled:
            if draft_len != 1:
                raise ValueError(
                    "DeepSeek V4 MTP speculative V1 only supports "
                    f"draft_len=1, got {draft_len}."
                )
            override("enable_dsv4_mtp_speculative", True)
            override("enable_dsv4_mtp", True)
            override("dsv4_mtp_spec_draft_len", 1)
            override("allow_dsv4_cuda_graph", False)
            override("cuda_graph_bs", [])
            override("cuda_graph_max_bs", 0)
            override("cuda_graph_capture_greedy_sample", False)
            os.environ[_DSV4_MTP_SPECULATIVE_ENV] = "1"
            os.environ[_DSV4_EXPERIMENTAL_MTP_ENV] = "1"
            logger.info_rank0(
                "Opting in to experimental DeepSeek V4 greedy MTP speculative "
                "sidecar: draft_len=1, sampling disabled, CUDA graph disabled."
            )
        if getattr(config, "enable_dsv4_mtp", False) or _env_flag(_DSV4_EXPERIMENTAL_MTP_ENV):
            os.environ[_DSV4_EXPERIMENTAL_MTP_ENV] = "1"
            logger.info_rank0(
                "Opting in to experimental DeepSeek V4 MTP weight loading and oracle helpers."
            )
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
            if getattr(config, "enable_dsv4_component_loc_ownership", False):
                logger.info_rank0(
                    "Opting in to DeepSeek V4 Route B decode CUDA graph metadata "
                    "copy; component-aware decode deforest and direct graph "
                    "metadata buffers remain explicit env opt-ins."
                )
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

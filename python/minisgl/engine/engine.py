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
    accepted_tokens_gpu: torch.Tensor | None = None
    accepted_tokens_cpu: torch.Tensor | None = None
    accepted_lens_cpu: torch.Tensor | None = None
    accepted_lens: tuple[int, ...] | None = None


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
        acceptance_histogram = {
            str(i): 0 for i in range(max(int(self.dsv4_mtp_spec_draft_len), 1) + 1)
        }
        self.mtp_spec_stats: dict[str, Any] = {
            "enabled": bool(self.enable_dsv4_mtp_speculative),
            "draft_len": int(self.dsv4_mtp_spec_draft_len),
            "draft_tokens_proposed": 0,
            "draft_tokens_verified": 0,
            "draft_tokens_accepted": 0,
            "draft_tokens_rejected": 0,
            "target_correction_tokens": 0,
            "target_fallback_tokens": 0,
            "emitted_tokens": 0,
            "acceptance_histogram": acceptance_histogram,
            "target_calls": 0,
            "target_verify_calls": 0,
            "target_commit_kv_copies": 0,
            "draft_calls": 0,
            "target_latency_s": 0.0,
            "target_verify_latency_s": 0.0,
            "target_commit_latency_s": 0.0,
            "target_rollback_latency_s": 0.0,
            "draft_latency_s": 0.0,
            "scheduler_overhead_s": 0.0,
            "finite_failures": 0,
            "fallback_sampling_batches": 0,
            "fallback_missing_mtp_batches": 0,
            "fallback_page_boundary_tokens": 0,
            "fallback_empty_draft_batches": 0,
            "fallback_temp_kv_unsupported_batches": 0,
            "fallback_temp_kv_capacity_batches": 0,
            "flattened_verify_tokens": 0,
            "target_verify_temp_kv_bytes": 0,
            "accepted_kv_copied_bytes": 0,
            "accepted_kv_copied_tokens": 0,
            "accepted_kv_commit_fail_closed": False,
            "accepted_kv_commit_blocker": "",
            "accepted_kv_commit_blocked_rows": 0,
            "draft_tokens_accept_candidates": 0,
            "target_correction_token_candidates": 0,
            "target_bonus_token_candidates": 0,
            "target_bonus_tokens": 0,
            "recompute_tokens": 0,
            "rejected_tail_isolation_checks": 0,
            "target_verify_batch_shapes": [],
            "target_verify_contract_trace": [],
            "c128_mtp_lifecycle_events": [],
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
        if int(self.dsv4_mtp_spec_draft_len) not in {1, 2, 4}:
            raise RuntimeError(
                "DeepSeek V4 MTP speculative frozen-KV runtime supports "
                "draft_len in {1, 2, 4}, got "
                f"{self.dsv4_mtp_spec_draft_len}."
            )
        return batch.size > 0

    def _allocated_kv_token_limit(self, req: Req) -> int:
        page_size = max(int(self.ctx.page_size), 1)
        return ((int(req.cached_len) + page_size - 1) // page_size) * page_size

    def _is_stop_token_for_req(self, req: Req, token: torch.Tensor) -> bool:
        eos_token_id = getattr(self, "eos_token_id", None)
        if eos_token_id is None or req.sampling_params.ignore_eos:
            return False
        return int(token.reshape(-1)[0].item()) == int(eos_token_id)

    def _make_mtp_frozen_batch(
        self,
        reqs: list[Req],
        input_ids: torch.Tensor,
        positions: list[int],
        *,
        read_only: bool,
    ) -> Batch:
        mtp_batch = Batch(reqs=reqs, phase="decode", frozen_kv_read_only=read_only)
        mtp_batch.padded_reqs = reqs
        mtp_batch.input_ids = input_ids.to(device=self.device, dtype=torch.int32)
        mtp_batch.positions = torch.tensor(positions, dtype=torch.int32, device=self.device)
        table = torch.tensor([req.table_idx for req in reqs], dtype=torch.long, device=self.device)
        pos = mtp_batch.positions.to(dtype=torch.long)
        mtp_batch.out_loc = self.page_table[table, pos].to(dtype=torch.int32)
        self.attn_backend.prepare_metadata(mtp_batch)
        return mtp_batch

    def _record_mtp_acceptance_histogram(self, accepted_prefix_lens: list[int]) -> None:
        histogram = self.mtp_spec_stats["acceptance_histogram"]
        for accepted_len in accepted_prefix_lens:
            key = str(int(accepted_len))
            histogram[key] = int(histogram.get(key, 0)) + 1

    def _mtp_has_c128_layers(self) -> bool:
        kv_cache = getattr(self, "kv_cache", None)
        layer_mapping = tuple(getattr(kv_cache, "layer_mapping", ()))
        return any(int(getattr(mapping, "compress_ratio", 0)) == 128 for mapping in layer_mapping)

    def _mtp_accepted_commit_blocker(self) -> str | None:
        if not self._mtp_has_c128_layers():
            return None
        kv_cache = getattr(self, "kv_cache", None)
        required = (
            "get_online_c128_mtp_pending_seq_lens",
            "get_online_c128_mtp_state_slot_offset",
            "get_online_c128_mtp_max_draft_tokens",
        )
        if kv_cache is None or not all(callable(getattr(kv_cache, name, None)) for name in required):
            return "c128_online_mtp_pending_write_commit_not_ported"
        return "c128_online_mtp_write_prefix_kernel_not_bound"

    def _record_mtp_c128_lifecycle_event(self, event: dict[str, Any]) -> None:
        events = self.mtp_spec_stats.setdefault("c128_mtp_lifecycle_events", [])
        events.append(event)
        if len(events) > 32:
            del events[:-32]

    def _record_mtp_target_verify_contract_trace(
        self,
        verify_batch: Batch,
        entries: list[dict[str, Any]],
    ) -> None:
        trace_log = self.mtp_spec_stats.setdefault("target_verify_contract_trace", [])
        core = getattr(getattr(verify_batch, "attn_metadata", None), "core_metadata", None)

        def _tolist(tensor: torch.Tensor | None, limit: int = 32) -> list[int]:
            if tensor is None:
                return []
            flat = tensor.detach().reshape(-1)[:limit].to("cpu")
            return [int(x) for x in flat.tolist()]

        metadata = getattr(verify_batch, "dsv4_target_verify_metadata", {})
        trace_log.append(
            {
                "mode": "target_verify",
                "speculative_num_draft_tokens": int(
                    metadata.get("speculative_num_draft_tokens", 0)
                ),
                "batch_size": int(len(entries)),
                "num_tokens": int(getattr(verify_batch, "input_ids").numel()),
                "verify_lens": [int(entry["verify_len"]) for entry in entries],
                "committed_seq_lens": [
                    int(entry.get("committed_seq_len", -1)) for entry in entries
                ],
                "input_tokens": _tolist(getattr(verify_batch, "input_ids", None)),
                "positions": _tolist(getattr(verify_batch, "positions", None)),
                "out_cache_loc": _tolist(getattr(verify_batch, "out_loc", None)),
                "seq_lens": _tolist(getattr(core, "seq_lens", None)),
                "extend_lens": _tolist(getattr(core, "extend_lens", None)),
                "req_seq_lens": _tolist(getattr(core, "req_seq_lens", None)),
                "c4_out_loc": _tolist(getattr(core, "c4_out_loc", None)),
                "c128_out_loc": _tolist(getattr(core, "c128_out_loc", None)),
                "c128_pending_write_commit": (
                    "not_applicable"
                    if not self._mtp_has_c128_layers()
                    else "not_ported_fail_closed"
                ),
            }
        )
        if len(trace_log) > 16:
            del trace_log[:-16]

    def _mtp_temp_kv_unsupported_reason(self) -> str | None:
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is None:
            return "missing_kv_cache"
        if bool(getattr(kv_cache, "component_loc_ownership_enabled", False)):
            return "route_b_component_loc_ownership"
        if bool(getattr(kv_cache, "swa_independent_lifecycle_enabled", False)):
            return "swa_independent_lifecycle"
        return None

    def _mtp_scratch_token_start(self) -> int:
        kv_cache = getattr(self, "kv_cache", None)
        return int(
            getattr(
                kv_cache,
                "dummy_token_start",
                int(self.num_pages) * int(self.ctx.page_size),
            )
        )

    def _make_mtp_flattened_temp_verify_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[Batch, list[tuple[Req, int, torch.Tensor, torch.Tensor]], int]:
        total_verify_tokens = sum(int(entry["verify_len"]) for entry in entries)
        scratch_capacity = int(self.ctx.page_size)
        if total_verify_tokens > scratch_capacity:
            self.mtp_spec_stats["fallback_temp_kv_capacity_batches"] = int(
                self.mtp_spec_stats["fallback_temp_kv_capacity_batches"]
            ) + 1
            raise RuntimeError(
                "DeepSeek V4 MTP flattened verify scratch page is too small: "
                f"tokens={total_verify_tokens}, capacity={scratch_capacity}."
            )

        scratch_start = self._mtp_scratch_token_start()
        restore: list[tuple[Req, int, torch.Tensor, torch.Tensor]] = []
        input_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        out_loc_chunks: list[torch.Tensor] = []
        cursor = 0
        layer_mapping = tuple(getattr(self.kv_cache, "layer_mapping", ()))
        has_c4 = any(int(m.compress_ratio) == 4 for m in layer_mapping)
        has_c128 = any(int(m.compress_ratio) == 128 for m in layer_mapping)
        used_scratch_offsets: set[int] = set()
        used_c4_slots: set[int] = set()
        used_c128_slots: set[int] = set()

        def _alloc_scratch_offset(position: int) -> int:
            is_c4_boundary = has_c4 and (int(position) + 1) % 4 == 0
            is_c128_boundary = has_c128 and (int(position) + 1) % 128 == 0

            def _valid(offset: int) -> bool:
                if offset in used_scratch_offsets:
                    return False
                if is_c4_boundary and (offset // 4) in used_c4_slots:
                    return False
                if is_c128_boundary and (offset // 128) in used_c128_slots:
                    return False
                return True

            candidates: range
            if is_c128_boundary:
                candidates = range(127, scratch_capacity, 128)
            else:
                candidates = range(int(position) % 4, scratch_capacity, 4)
            for offset in candidates:
                if _valid(offset):
                    used_scratch_offsets.add(offset)
                    if is_c4_boundary:
                        used_c4_slots.add(offset // 4)
                    if is_c128_boundary:
                        used_c128_slots.add(offset // 128)
                    return offset
            self.mtp_spec_stats["fallback_temp_kv_capacity_batches"] = int(
                self.mtp_spec_stats["fallback_temp_kv_capacity_batches"]
            ) + 1
            raise RuntimeError(
                "DeepSeek V4 MTP flattened verify scratch page cannot assign "
                f"a unique compressed temp slot for position={position}, "
                f"capacity={scratch_capacity}."
            )

        try:
            for entry in entries:
                req = entry["req"]
                verify_len = int(entry["verify_len"])
                base_pos = int(req.cached_len)
                old_device_len = int(req.device_len)
                positions = torch.arange(
                    base_pos,
                    base_pos + verify_len,
                    dtype=torch.long,
                    device=self.device,
                )
                if positions.numel() == 0:
                    continue
                if int(positions[-1].item()) >= int(self.page_table.shape[1]):
                    raise RuntimeError(
                        "DeepSeek V4 MTP flattened verify exceeded page-table width: "
                        f"position={int(positions[-1].item())}, "
                        f"width={int(self.page_table.shape[1])}."
                    )
                original_locs = self.page_table[int(req.table_idx), positions].clone()
                scratch_offsets = [
                    _alloc_scratch_offset(int(position)) for position in positions.tolist()
                ]
                temp_locs = torch.tensor(
                    [scratch_start + offset for offset in scratch_offsets],
                    dtype=torch.int32,
                    device=self.device,
                )
                self.page_table[int(req.table_idx), positions] = temp_locs
                req.device_len = base_pos + verify_len
                restore.append((req, old_device_len, positions, original_locs))

                entry["row_start"] = cursor
                entry["positions_tensor"] = positions.to(dtype=torch.int32)
                entry["temp_locs"] = temp_locs
                entry["real_locs"] = original_locs.to(dtype=torch.int32)
                input_chunks.extend(entry["verify_inputs"])
                position_chunks.append(entry["positions_tensor"])
                out_loc_chunks.append(temp_locs)
                cursor += verify_len

            verify_batch = Batch(
                reqs=[entry["req"] for entry in entries],
                phase="decode",
                frozen_kv_read_only=False,
            )
            verify_batch.padded_reqs = verify_batch.reqs
            verify_batch.input_ids = torch.cat(input_chunks, dim=0).to(
                device=self.device,
                dtype=torch.int32,
            )
            verify_batch.positions = torch.cat(position_chunks, dim=0).to(
                device=self.device,
                dtype=torch.int32,
            )
            verify_batch.out_loc = torch.cat(out_loc_chunks, dim=0).to(
                device=self.device,
                dtype=torch.int32,
            )
            self.attn_backend.prepare_metadata(verify_batch)
            return verify_batch, restore, total_verify_tokens
        except Exception:
            self._restore_mtp_flattened_temp_verify_batch(restore)
            raise

    def _restore_mtp_flattened_temp_verify_batch(
        self,
        restore: list[tuple[Req, int, torch.Tensor, torch.Tensor]],
    ) -> None:
        for req, old_device_len, positions, original_locs in restore:
            self.page_table[int(req.table_idx), positions] = original_locs
            req.device_len = int(old_device_len)

    def _make_mtp_flattened_verify_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[Batch, list[tuple[Req, int]], int]:
        total_verify_tokens = sum(int(entry["verify_len"]) for entry in entries)
        restore: list[tuple[Req, int]] = []
        input_chunks: list[torch.Tensor] = []
        position_chunks: list[torch.Tensor] = []
        out_loc_chunks: list[torch.Tensor] = []
        cursor = 0
        try:
            for entry in entries:
                req = entry["req"]
                verify_len = int(entry["verify_len"])
                base_pos = int(req.cached_len)
                old_device_len = int(req.device_len)
                positions = torch.arange(
                    base_pos,
                    base_pos + verify_len,
                    dtype=torch.long,
                    device=self.device,
                )
                if positions.numel() == 0:
                    continue
                if int(positions[-1].item()) >= int(self.page_table.shape[1]):
                    raise RuntimeError(
                        "DeepSeek V4 MTP flattened verify exceeded page-table width: "
                        f"position={int(positions[-1].item())}, "
                        f"width={int(self.page_table.shape[1])}."
                    )
                real_locs = self.page_table[int(req.table_idx), positions].clone()
                req.device_len = base_pos + verify_len
                restore.append((req, old_device_len))

                entry["row_start"] = cursor
                entry["committed_seq_len"] = base_pos
                entry["positions_tensor"] = positions.to(dtype=torch.int32)
                entry["temp_locs"] = real_locs.to(dtype=torch.int32)
                entry["real_locs"] = real_locs.to(dtype=torch.int32)
                input_chunks.extend(entry["verify_inputs"])
                position_chunks.append(entry["positions_tensor"])
                out_loc_chunks.append(entry["real_locs"])
                cursor += verify_len

            verify_batch = Batch(
                reqs=[entry["req"] for entry in entries],
                phase="decode",
                frozen_kv_read_only=False,
            )
            verify_batch.padded_reqs = verify_batch.reqs
            verify_batch.input_ids = torch.cat(input_chunks, dim=0).to(
                device=self.device,
                dtype=torch.int32,
            )
            verify_batch.positions = torch.cat(position_chunks, dim=0).to(
                device=self.device,
                dtype=torch.int32,
            )
            verify_batch.out_loc = torch.cat(out_loc_chunks, dim=0).to(
                device=self.device,
                dtype=torch.int32,
            )
            verify_lens = [int(entry["verify_len"]) for entry in entries]
            if len(set(verify_lens)) != 1:
                raise RuntimeError(
                    "DeepSeek V4 MTP target-verify metadata requires a fixed "
                    "active verify length per request in one flattened batch; "
                    f"got verify_lens={verify_lens}."
                )
            speculative_num_draft_tokens = int(verify_lens[0])
            verify_batch.dsv4_target_verify_metadata = {
                "speculative_num_draft_tokens": speculative_num_draft_tokens,
                "extend_lens": verify_lens,
                "committed_seq_lens": [
                    int(entry["committed_seq_len"]) for entry in entries
                ],
                "num_tokens": int(total_verify_tokens),
            }
            self.attn_backend.prepare_metadata(verify_batch)
            self._record_mtp_target_verify_contract_trace(verify_batch, entries)
            return verify_batch, restore, total_verify_tokens
        except Exception:
            self._restore_mtp_flattened_verify_batch(restore)
            raise

    def _restore_mtp_flattened_verify_batch(
        self,
        restore: list[tuple[Req, int]],
    ) -> None:
        for req, old_device_len in restore:
            req.device_len = int(old_device_len)

    def _forward_mtp_flattened_verify_with_hidden(
        self,
        verify_batch: Batch,
    ) -> dict[str, torch.Tensor]:
        inner_model = getattr(self.model, "model", None)
        lm_head = getattr(self.model, "lm_head", None)
        if inner_model is not None and lm_head is not None:
            output, hidden_before_norm = inner_model.forward(
                verify_batch.input_ids,
                return_hidden_states_before_norm=True,
            )
            logits = lm_head.linear(output)
            return {
                "logits": logits,
                "hidden_states": output,
                "hidden_states_before_norm": hidden_before_norm,
            }
        output = self.model.forward_with_hidden()
        return {
            "logits": output.logits,
            "hidden_states": output.hidden_states,
            "hidden_states_before_norm": output.hidden_states_before_norm,
        }

    def _estimate_mtp_kv_bytes(
        self,
        full_locs: torch.Tensor,
        positions: torch.Tensor,
    ) -> int:
        if full_locs.numel() == 0:
            return 0
        kv_cache = self.kv_cache
        rows = int(full_locs.numel())
        bytes_total = 0
        layer_mapping = tuple(getattr(kv_cache, "layer_mapping", ()))
        num_layers = len(layer_mapping) or int(getattr(kv_cache, "_num_layers", 0) or 0)
        if num_layers:
            sample = kv_cache.swa_cache(0)
            bytes_total += rows * num_layers * int(sample.shape[-1]) * int(sample.element_size())

        def _compressed_loc_count(ratio: int) -> int:
            locs = kv_cache.compressed_locs_from_full_locs(
                full_locs,
                ratio,  # type: ignore[arg-type]
                positions,
            )
            return int(locs.numel())

        c4_layers = [m.layer_id for m in layer_mapping if int(m.compress_ratio) == 4]
        c128_layers = [m.layer_id for m in layer_mapping if int(m.compress_ratio) == 128]
        c4_count = _compressed_loc_count(4) if c4_layers else 0
        c128_count = _compressed_loc_count(128) if c128_layers else 0
        for layer_id in c4_layers:
            cache = kv_cache.c4_cache(int(layer_id))
            bytes_total += c4_count * int(cache.shape[-1]) * int(cache.element_size())
            if self._mtp_indexer_uses_fp8_cache(kv_cache):
                bytes_total += c4_count * (
                    int(getattr(kv_cache, "_index_head_dim", cache.shape[-1])) + 4
                )
            else:
                indexer_cache = kv_cache.indexer_cache(int(layer_id))
                bytes_total += (
                    c4_count
                    * int(indexer_cache.shape[-1])
                    * int(indexer_cache.element_size())
                )
        for layer_id in c128_layers:
            cache = kv_cache.c128_cache(int(layer_id))
            bytes_total += c128_count * int(cache.shape[-1]) * int(cache.element_size())
        return int(bytes_total)

    def _snapshot_mtp_kv_rows(
        self,
        full_locs: torch.Tensor,
        positions: torch.Tensor,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"items": [], "bytes": 0}
        if full_locs.numel() == 0:
            return snapshot
        kv_cache = self.kv_cache
        full_locs = full_locs.to(device=self.device, dtype=torch.long)
        positions = positions.to(device=self.device, dtype=torch.long)

        def _add(cache: torch.Tensor, locs: torch.Tensor) -> None:
            valid_locs = locs.to(device=self.device, dtype=torch.long)
            valid_locs = valid_locs[(valid_locs >= 0) & (valid_locs < int(cache.shape[0]))]
            if valid_locs.numel() == 0:
                return
            valid_locs = torch.unique(valid_locs)
            values = cache[valid_locs].clone()
            snapshot["items"].append(("tensor", cache, valid_locs, values))
            snapshot["bytes"] = int(snapshot["bytes"]) + (
                int(values.numel()) * int(values.element_size())
            )

        layer_mapping = tuple(getattr(kv_cache, "layer_mapping", ()))
        num_layers = len(layer_mapping) or int(getattr(kv_cache, "_num_layers", 0) or 0)
        if num_layers:
            swa_locs = kv_cache.translate_full_locs_to_swa_locs(full_locs)
            for layer_id in range(num_layers):
                _add(kv_cache.swa_cache(int(layer_id)), swa_locs)

        c4_layers = [m.layer_id for m in layer_mapping if int(m.compress_ratio) == 4]
        c128_layers = [m.layer_id for m in layer_mapping if int(m.compress_ratio) == 128]
        if c4_layers:
            c4_locs = kv_cache.compressed_locs_from_full_locs(full_locs, 4, positions)
            c4_state_locs = self._mtp_state_locs_from_full_locs(
                kv_cache,
                full_locs,
                4,
                component="attention",
            )
            indexer_state_locs = self._mtp_state_locs_from_full_locs(
                kv_cache,
                full_locs,
                4,
                component="indexer",
            )
            for layer_id in c4_layers:
                _add(kv_cache.c4_cache(int(layer_id)), c4_locs)
                self._snapshot_mtp_indexer_kv(snapshot, kv_cache, int(layer_id), c4_locs)
                self._snapshot_mtp_state_pool(
                    snapshot,
                    getattr(kv_cache, "attention_compress_state", None),
                    int(layer_id),
                    c4_state_locs,
                )
                self._snapshot_mtp_state_pool(
                    snapshot,
                    getattr(kv_cache, "indexer_compress_state", None),
                    int(layer_id),
                    indexer_state_locs,
                )
        if c128_layers:
            c128_locs = kv_cache.compressed_locs_from_full_locs(full_locs, 128, positions)
            c128_state_locs = self._mtp_state_locs_from_full_locs(
                kv_cache,
                full_locs,
                128,
                component="attention",
            )
            for layer_id in c128_layers:
                _add(kv_cache.c128_cache(int(layer_id)), c128_locs)
                self._snapshot_mtp_state_pool(
                    snapshot,
                    getattr(kv_cache, "attention_compress_state", None),
                    int(layer_id),
                    c128_state_locs,
                )
        return snapshot

    def _mtp_state_locs_from_full_locs(
        self,
        kv_cache: Any,
        full_locs: torch.Tensor,
        ratio: int,
        *,
        component: str,
    ) -> torch.Tensor:
        state_locs = getattr(kv_cache, "state_locs_from_full_locs", None)
        if not callable(state_locs):
            return torch.empty(0, dtype=torch.long, device=self.device)
        try:
            return state_locs(full_locs, ratio, component=component).to(
                device=self.device,
                dtype=torch.long,
            )
        except Exception:
            return torch.empty(0, dtype=torch.long, device=self.device)

    def _snapshot_mtp_state_pool(
        self,
        snapshot: dict[str, Any],
        pool_getter: Any,
        layer_id: int,
        locs: torch.Tensor,
    ) -> None:
        if not callable(pool_getter) or locs.numel() == 0:
            return
        pool = pool_getter(layer_id)
        buffer = pool.kv_score_buffer.kv_score
        valid_locs = locs.to(device=buffer.device, dtype=torch.long)
        valid_locs = valid_locs[(valid_locs >= 0) & (valid_locs < int(buffer.shape[0]))]
        if valid_locs.numel() == 0:
            return
        valid_locs = torch.unique(valid_locs)
        values = buffer[valid_locs].clone()
        snapshot["items"].append(("tensor", buffer, valid_locs, values))
        snapshot["bytes"] = int(snapshot["bytes"]) + (
            int(values.numel()) * int(values.element_size())
        )

    def _snapshot_mtp_indexer_kv(
        self,
        snapshot: dict[str, Any],
        kv_cache: Any,
        layer_id: int,
        locs: torch.Tensor,
    ) -> None:
        valid_locs = locs.to(device=self.device, dtype=torch.long)
        valid_locs = valid_locs[valid_locs >= 0]
        if valid_locs.numel() == 0:
            return
        valid_locs = torch.unique(valid_locs)
        if self._mtp_indexer_uses_fp8_cache(kv_cache):
            if (
                hasattr(kv_cache, "has_indexer_fp8_paged_cache")
                and kv_cache.has_indexer_fp8_paged_cache()
            ):
                packed = kv_cache.indexer_fp8_paged_cache(layer_id)
                page_size = int(kv_cache.indexer_fp8_page_size)
                dim = int(getattr(kv_cache, "_index_head_dim", 0))
                page_bytes = int(packed.shape[-1])
                pages = valid_locs // page_size
                offsets = valid_locs - pages * page_size
                data = packed.as_strided(
                    (packed.shape[0], page_size, dim),
                    (page_bytes, dim, 1),
                )
                scales = packed.as_strided(
                    (packed.shape[0], page_size, 4),
                    (page_bytes, 4, 1),
                    storage_offset=page_size * dim,
                )
                data_values = data[pages, offsets].clone()
                scale_values = scales[pages, offsets].clone()
                snapshot["items"].append(
                    ("indexer_fp8_paged", packed, page_size, dim, pages, offsets, data_values, scale_values)
                )
                snapshot["bytes"] = int(snapshot["bytes"]) + (
                    int(data_values.numel()) * int(data_values.element_size())
                    + int(scale_values.numel()) * int(scale_values.element_size())
                )
                return

            values, scales = kv_cache.indexer_fp8_cache(layer_id)
            values_snapshot = values[valid_locs].clone()
            scales_snapshot = scales[valid_locs].clone()
            snapshot["items"].append(
                ("indexer_fp8", values, scales, valid_locs, values_snapshot, scales_snapshot)
            )
            snapshot["bytes"] = int(snapshot["bytes"]) + (
                int(values_snapshot.numel()) * int(values_snapshot.element_size())
                + int(scales_snapshot.numel()) * int(scales_snapshot.element_size())
            )
            return

        cache = kv_cache.indexer_cache(layer_id)
        valid_locs = valid_locs[valid_locs < int(cache.shape[0])]
        if valid_locs.numel() == 0:
            return
        values = cache[valid_locs].clone()
        snapshot["items"].append(("tensor", cache, valid_locs, values))
        snapshot["bytes"] = int(snapshot["bytes"]) + (
            int(values.numel()) * int(values.element_size())
        )

    def _restore_mtp_kv_snapshot(self, snapshot: dict[str, Any]) -> None:
        for item in snapshot.get("items", []):
            kind = item[0]
            if kind == "tensor":
                _, cache, locs, values = item
                cache[locs] = values
            elif kind == "indexer_fp8_paged":
                _, packed, page_size, dim, pages, offsets, data_values, scale_values = item
                page_bytes = int(packed.shape[-1])
                data = packed.as_strided(
                    (packed.shape[0], page_size, dim),
                    (page_bytes, dim, 1),
                )
                scales = packed.as_strided(
                    (packed.shape[0], page_size, 4),
                    (page_bytes, 4, 1),
                    storage_offset=page_size * dim,
                )
                data[pages, offsets] = data_values
                scales[pages, offsets] = scale_values
            elif kind == "indexer_fp8":
                _, values, scales, locs, values_snapshot, scales_snapshot = item
                values[locs] = values_snapshot
                scales[locs] = scales_snapshot

    def _mtp_indexer_uses_fp8_cache(self, kv_cache: Any) -> bool:
        has_fp8 = getattr(kv_cache, "has_indexer_fp8_cache", None)
        return bool(callable(has_fp8) and has_fp8())

    def _copy_mtp_temp_kv_to_committed(
        self,
        temp_locs: torch.Tensor,
        real_locs: torch.Tensor,
        positions: torch.Tensor,
    ) -> int:
        if temp_locs.numel() == 0:
            return 0
        kv_cache = self.kv_cache
        temp_locs = temp_locs.to(device=self.device, dtype=torch.long)
        real_locs = real_locs.to(device=self.device, dtype=torch.long)
        positions = positions.to(device=self.device, dtype=torch.long)
        if temp_locs.shape != real_locs.shape or temp_locs.shape != positions.shape:
            raise RuntimeError(
                "DeepSeek V4 MTP temp KV copy shape mismatch: "
                f"temp={tuple(temp_locs.shape)}, real={tuple(real_locs.shape)}, "
                f"positions={tuple(positions.shape)}."
            )
        if bool(torch.any(real_locs < 0).item()):
            raise RuntimeError("DeepSeek V4 MTP temp KV copy found negative committed loc.")

        copied_bytes = 0
        layer_mapping = tuple(getattr(kv_cache, "layer_mapping", ()))
        num_layers = len(layer_mapping) or int(getattr(kv_cache, "_num_layers", 0) or 0)
        if num_layers:
            src_swa = kv_cache.translate_full_locs_to_swa_locs(temp_locs)
            dst_swa = kv_cache.translate_full_locs_to_swa_locs(real_locs)
            if bool(torch.any((src_swa < 0) | (dst_swa < 0)).item()):
                raise RuntimeError("DeepSeek V4 MTP temp KV copy found invalid SWA loc.")
            src_swa = src_swa.to(dtype=torch.long)
            dst_swa = dst_swa.to(dtype=torch.long)
            for layer_id in range(num_layers):
                cache = kv_cache.swa_cache(int(layer_id))
                cache[dst_swa] = cache[src_swa]
                copied_bytes += (
                    int(dst_swa.numel()) * int(cache.shape[-1]) * int(cache.element_size())
                )

        c4_layers = [m.layer_id for m in layer_mapping if int(m.compress_ratio) == 4]
        c128_layers = [m.layer_id for m in layer_mapping if int(m.compress_ratio) == 128]
        if c4_layers:
            src_c4 = kv_cache.compressed_locs_from_full_locs(temp_locs, 4, positions).to(
                device=self.device,
                dtype=torch.long,
            )
            dst_c4 = kv_cache.compressed_locs_from_full_locs(real_locs, 4, positions).to(
                device=self.device,
                dtype=torch.long,
            )
            if src_c4.shape != dst_c4.shape:
                raise RuntimeError("DeepSeek V4 MTP C4 temp KV copy loc mismatch.")
            for layer_id in c4_layers:
                cache = kv_cache.c4_cache(int(layer_id))
                if dst_c4.numel() > 0:
                    cache[dst_c4] = cache[src_c4]
                copied_bytes += (
                    int(dst_c4.numel()) * int(cache.shape[-1]) * int(cache.element_size())
                )
                copied_bytes += self._copy_mtp_temp_indexer_kv(
                    kv_cache,
                    int(layer_id),
                    src_c4,
                    dst_c4,
                )
        if c128_layers:
            src_c128 = kv_cache.compressed_locs_from_full_locs(temp_locs, 128, positions).to(
                device=self.device,
                dtype=torch.long,
            )
            dst_c128 = kv_cache.compressed_locs_from_full_locs(real_locs, 128, positions).to(
                device=self.device,
                dtype=torch.long,
            )
            if src_c128.shape != dst_c128.shape:
                raise RuntimeError("DeepSeek V4 MTP C128 temp KV copy loc mismatch.")
            for layer_id in c128_layers:
                cache = kv_cache.c128_cache(int(layer_id))
                if dst_c128.numel() > 0:
                    cache[dst_c128] = cache[src_c128]
                copied_bytes += (
                    int(dst_c128.numel()) * int(cache.shape[-1]) * int(cache.element_size())
                )
        return int(copied_bytes)

    def _copy_mtp_temp_indexer_kv(
        self,
        kv_cache: Any,
        layer_id: int,
        src_locs: torch.Tensor,
        dst_locs: torch.Tensor,
    ) -> int:
        if src_locs.numel() == 0:
            return 0
        if self._mtp_indexer_uses_fp8_cache(kv_cache):
            if (
                hasattr(kv_cache, "has_indexer_fp8_paged_cache")
                and kv_cache.has_indexer_fp8_paged_cache()
            ):
                packed = kv_cache.indexer_fp8_paged_cache(layer_id)
                page_size = int(kv_cache.indexer_fp8_page_size)
                dim = int(getattr(kv_cache, "_index_head_dim", 0))
                page_bytes = int(packed.shape[-1])
                data = packed.as_strided(
                    (packed.shape[0], page_size, dim),
                    (page_bytes, dim, 1),
                )
                scales = packed.as_strided(
                    (packed.shape[0], page_size, 4),
                    (page_bytes, 4, 1),
                    storage_offset=page_size * dim,
                )
                src_pages = src_locs // page_size
                dst_pages = dst_locs // page_size
                src_offsets = src_locs - src_pages * page_size
                dst_offsets = dst_locs - dst_pages * page_size
                data[dst_pages, dst_offsets] = data[src_pages, src_offsets]
                scales[dst_pages, dst_offsets] = scales[src_pages, src_offsets]
                return int(src_locs.numel()) * (dim + 4)

            values, scales = kv_cache.indexer_fp8_cache(layer_id)
            values[dst_locs] = values[src_locs]
            scales[dst_locs] = scales[src_locs]
            return int(src_locs.numel()) * (
                int(values.shape[-1]) * int(values.element_size())
                + int(scales.shape[-1]) * int(scales.element_size())
            )

        cache = kv_cache.indexer_cache(layer_id)
        cache[dst_locs] = cache[src_locs]
        return int(src_locs.numel()) * int(cache.shape[-1]) * int(cache.element_size())

    def _propose_mtp_spec_drafts(
        self,
        batch: Batch,
        next_tokens_gpu: torch.Tensor,
        hidden_states_before_norm: torch.Tensor,
        mtp_positions: list[int],
    ) -> dict[str, Any]:
        draft_len = int(self.dsv4_mtp_spec_draft_len)
        entries: list[dict[str, Any]] = []
        for i, req in enumerate(batch.reqs):
            if self._is_stop_token_for_req(req, next_tokens_gpu[i : i + 1]):
                continue
            max_steps = min(draft_len, max(int(req.remain_len), 0))
            if max_steps <= 0:
                continue
            entries.append(
                {
                    "batch_index": i,
                    "req": req,
                    "max_steps": max_steps,
                    "position": int(mtp_positions[i]),
                    "prev_token": next_tokens_gpu[i : i + 1].contiguous(),
                    "prev_hidden": hidden_states_before_norm[i : i + 1].contiguous(),
                    "tokens": [],
                }
            )

        if not entries:
            self.mtp_spec_stats["fallback_empty_draft_batches"] = int(
                self.mtp_spec_stats["fallback_empty_draft_batches"]
            ) + 1
            return {"proposed": 0, "finite": True, "draft_tokens_by_index": {}}

        proposed = 0
        for step in range(draft_len):
            active = [entry for entry in entries if step < int(entry["max_steps"])]
            if not active:
                break
            draft_input = torch.cat([entry["prev_token"] for entry in active], dim=0).to(
                device=self.device, dtype=torch.int32
            )
            draft_hidden = torch.cat([entry["prev_hidden"] for entry in active], dim=0).contiguous()
            mtp_batch = self._make_mtp_frozen_batch(
                [entry["req"] for entry in active],
                draft_input,
                [int(entry["position"]) for entry in active],
                read_only=True,
            )

            self._sync_device_for_mtp_spec()
            start_s = time.perf_counter()
            with self.ctx.forward_batch(mtp_batch):
                mtp_output = self.model.mtp_forward_one_step(draft_input, draft_hidden)
                self._debug_sync_forward(
                    "after_mtp_forward",
                    mtp_batch,
                    "mtp_spec_frozen_kv_read_only_draft",
                )
            self._sync_device_for_mtp_spec()
            self.mtp_spec_stats["draft_latency_s"] = float(
                self.mtp_spec_stats["draft_latency_s"]
            ) + (time.perf_counter() - start_s)
            self.mtp_spec_stats["draft_calls"] = int(self.mtp_spec_stats["draft_calls"]) + 1

            mtp_logits = mtp_output.logits[: len(active)]
            finite = bool(torch.isfinite(mtp_logits).all().item())
            if not finite:
                self.mtp_spec_stats["finite_failures"] = int(
                    self.mtp_spec_stats["finite_failures"]
                ) + 1
                raise RuntimeError("DeepSeek V4 MTP speculative draft produced NaN/Inf logits.")

            draft_tokens = torch.argmax(mtp_logits, dim=-1).to(torch.int32)
            proposed += int(draft_tokens.numel())
            for row, entry in enumerate(active):
                token = draft_tokens[row : row + 1].contiguous()
                entry["tokens"].append(token)
                entry["prev_token"] = token
                entry["prev_hidden"] = mtp_output.hidden_states_before_norm[
                    row : row + 1
                ].contiguous()

        self.mtp_spec_stats["draft_tokens_proposed"] = int(
            self.mtp_spec_stats["draft_tokens_proposed"]
        ) + proposed
        draft_tokens_by_index = {
            int(entry["batch_index"]): [tok for tok in entry["tokens"]] for entry in entries
        }
        return {
            "proposed": proposed,
            "finite": True,
            "draft_tokens_by_index": draft_tokens_by_index,
        }

    def _verify_mtp_spec_drafts_flattened(
        self,
        batch: Batch,
        first_tokens_gpu: torch.Tensor,
        draft_tokens_by_index: dict[int, list[torch.Tensor]],
        emitted_tokens: list[list[torch.Tensor]],
    ) -> dict[str, Any]:
        unsupported_reason = self._mtp_temp_kv_unsupported_reason()
        if unsupported_reason is not None:
            self.mtp_spec_stats["fallback_temp_kv_unsupported_batches"] = int(
                self.mtp_spec_stats["fallback_temp_kv_unsupported_batches"]
            ) + 1
            raise RuntimeError(
                "DeepSeek V4 MTP flattened temp-KV verify is fail-closed for "
                f"{unsupported_reason}. Keep MTP speculative disabled for this "
                "ownership mode until accepted-KV movement covers Route-B "
                "component/SWA state."
            )

        entries: list[dict[str, Any]] = []
        page_boundary_stops = 0
        accepted_prefix_lens = [0] * len(batch.reqs)
        for i, req in enumerate(batch.reqs):
            drafts = draft_tokens_by_index.get(i, [])
            if not (drafts and req.can_decode):
                continue
            available_slots = max(
                int(self._allocated_kv_token_limit(req)) - int(req.cached_len),
                0,
            )
            if available_slots <= 0:
                page_boundary_stops += 1
                continue
            draft_count = min(len(drafts), max(int(req.remain_len), 0))
            has_bonus_row = int(req.remain_len) > draft_count
            verify_len = min(draft_count + (1 if has_bonus_row else 0), available_slots)
            if verify_len <= 0:
                page_boundary_stops += 1
                continue
            verify_inputs = [first_tokens_gpu[i : i + 1].contiguous()]
            verify_inputs.extend(tok.contiguous() for tok in drafts[: max(verify_len - 1, 0)])
            if len(verify_inputs) != verify_len:
                raise RuntimeError(
                    "DeepSeek V4 MTP flattened verify input construction mismatch: "
                    f"inputs={len(verify_inputs)}, verify_len={verify_len}."
                )
            entries.append(
                {
                    "batch_index": i,
                    "req": req,
                    "drafts": drafts[:draft_count],
                    "verify_len": verify_len,
                    "verify_inputs": verify_inputs,
                }
            )

        if not entries:
            self._record_mtp_acceptance_histogram(accepted_prefix_lens)
            self.mtp_spec_stats["fallback_page_boundary_tokens"] = int(
                self.mtp_spec_stats["fallback_page_boundary_tokens"]
            ) + page_boundary_stops
            return {
                "verified": 0,
                "accepted": 0,
                "rejected": 0,
                "correction_tokens": 0,
                "target_fallback_tokens": 0,
                "page_boundary_stops": page_boundary_stops,
                "accepted_prefix_lens": accepted_prefix_lens,
                "verify_shapes": [],
            }

        verified = 0
        accepted = 0
        rejected = 0
        correction_tokens = 0
        target_fallback_tokens = 0
        verify_shapes: list[list[int]] = []
        max_verify_len = max(int(entry["verify_len"]) for entry in entries)
        verify_batch, restore, total_verify_tokens = self._make_mtp_flattened_verify_batch(
            entries
        )
        all_real_locs = verify_batch.out_loc.to(device=self.device, dtype=torch.long)
        all_positions = verify_batch.positions.to(device=self.device, dtype=torch.long)
        pre_verify_snapshot = self._snapshot_mtp_kv_rows(all_real_locs, all_positions)
        temp_kv_bytes = int(pre_verify_snapshot.get("bytes", 0))
        start_s = time.perf_counter()
        target_output = None
        try:
            self._sync_device_for_mtp_spec()
            start_s = time.perf_counter()
            with self.ctx.forward_batch(verify_batch):
                self.graph_runner.record_eager_decode(verify_batch)
                target_output = self.model.forward_with_hidden()
                self._debug_sync_forward(
                    "after_mtp_flattened_temp_verify",
                    verify_batch,
                    "mtp_spec_flattened_temp_verify",
                )
            self._sync_device_for_mtp_spec()
        except Exception:
            self._restore_mtp_kv_snapshot(pre_verify_snapshot)
            self._sync_device_for_mtp_spec()
            raise
        finally:
            self._restore_mtp_flattened_verify_batch(restore)

        elapsed = time.perf_counter() - start_s
        self.mtp_spec_stats["target_latency_s"] = float(
            self.mtp_spec_stats["target_latency_s"]
        ) + elapsed
        self.mtp_spec_stats["target_verify_latency_s"] = float(
            self.mtp_spec_stats["target_verify_latency_s"]
        ) + elapsed
        self.mtp_spec_stats["target_calls"] = int(self.mtp_spec_stats["target_calls"]) + 1
        self.mtp_spec_stats["target_verify_calls"] = int(
            self.mtp_spec_stats["target_verify_calls"]
        ) + 1
        self.mtp_spec_stats["flattened_verify_tokens"] = int(
            self.mtp_spec_stats["flattened_verify_tokens"]
        ) + int(total_verify_tokens)
        self.mtp_spec_stats["target_verify_temp_kv_bytes"] = int(
            self.mtp_spec_stats["target_verify_temp_kv_bytes"]
        ) + int(temp_kv_bytes)
        verify_shapes.append([len(entries), max_verify_len, int(total_verify_tokens)])

        logits = target_output.logits[:total_verify_tokens]
        if not bool(torch.isfinite(logits).all().item()):
            self._restore_mtp_kv_snapshot(pre_verify_snapshot)
            self._sync_device_for_mtp_spec()
            self.mtp_spec_stats["finite_failures"] = int(
                self.mtp_spec_stats["finite_failures"]
            ) + 1
            raise RuntimeError("DeepSeek V4 MTP flattened target verify produced NaN/Inf logits.")
        target_tokens = torch.argmax(logits, dim=-1).to(torch.int32)

        commit_loc_chunks: list[torch.Tensor] = []
        commit_position_chunks: list[torch.Tensor] = []
        copy_rows_by_entry: list[tuple[Req, int]] = []
        bonus_tokens = 0
        accepted_candidates = 0
        correction_token_candidates = 0
        bonus_token_candidates = 0
        blocked_commit_rows = 0
        accepted_commit_blocker = self._mtp_accepted_commit_blocker()
        allow_accepted_commit = accepted_commit_blocker is None
        if accepted_commit_blocker is not None:
            self.mtp_spec_stats["accepted_kv_commit_fail_closed"] = True
            self.mtp_spec_stats["accepted_kv_commit_blocker"] = accepted_commit_blocker
            self._record_mtp_c128_lifecycle_event(
                {
                    "event": "accepted_commit_fail_closed",
                    "blocker": accepted_commit_blocker,
                    "verify_tokens": int(total_verify_tokens),
                    "reason": "missing OnlineC128MTPController pending/write/commit owner",
                }
            )
        trace_enabled = _env_flag("MINISGL_DSV4_MTP_SPEC_TRACE")
        trace_entries: list[dict[str, Any]] = []

        for entry in entries:
            batch_index = int(entry["batch_index"])
            req = entry["req"]
            drafts = entry["drafts"]
            verify_len = int(entry["verify_len"])
            row_start = int(entry["row_start"])
            row_tokens = target_tokens[row_start : row_start + verify_len]
            emitted_before = len(emitted_tokens[batch_index])
            accepted_prefix = 0
            copy_rows = 0
            candidate_copy_rows = 0
            candidate_emitted: list[torch.Tensor] = []
            mismatch_depth: int | None = None

            comparable = min(len(drafts), verify_len)
            for depth in range(comparable):
                target_token = row_tokens[depth : depth + 1].contiguous()
                draft_token = drafts[depth]
                verified += 1
                if bool((target_token == draft_token.to(device=self.device)).all().item()):
                    accepted_prefix += 1
                    candidate_emitted.append(draft_token.contiguous())
                    continue

                rejected += 1
                mismatch_depth = depth
                correction_token_candidates += 1
                candidate_emitted.append(target_token)
                candidate_copy_rows = depth + 1
                break

            if mismatch_depth is None and verify_len > len(drafts):
                target_token = row_tokens[len(drafts) : len(drafts) + 1].contiguous()
                bonus_token_candidates += 1
                candidate_emitted.append(target_token)
                candidate_copy_rows = min(verify_len, len(drafts) + 1)
            elif mismatch_depth is None:
                candidate_copy_rows = min(verify_len, accepted_prefix)

            accepted_candidates += accepted_prefix
            if allow_accepted_commit:
                accepted += accepted_prefix
                correction_tokens += 1 if mismatch_depth is not None else 0
                if mismatch_depth is None and verify_len > len(drafts):
                    bonus_tokens += 1
                accepted_prefix_lens[batch_index] = int(accepted_prefix)
                emitted_tokens[batch_index].extend(candidate_emitted)
                copy_rows = int(candidate_copy_rows)
            elif accepted_prefix > 0:
                blocked_commit_rows += int(max(candidate_copy_rows, accepted_prefix))

            if copy_rows > 0:
                commit_loc_chunks.append(entry["real_locs"][:copy_rows])
                commit_position_chunks.append(entry["positions_tensor"][:copy_rows])
                copy_rows_by_entry.append((req, copy_rows))
            if trace_enabled:
                emitted_now = emitted_tokens[batch_index][emitted_before:]
                trace_entries.append(
                    {
                        "uid": int(getattr(req, "uid", -1)),
                        "batch_index": int(batch_index),
                        "cached_len": int(getattr(req, "cached_len", -1)),
                        "device_len": int(getattr(req, "device_len", -1)),
                        "verify_len": int(verify_len),
                        "target_tokens": [int(x) for x in row_tokens.tolist()],
                        "draft_tokens": [int(tok.reshape(-1)[0].item()) for tok in drafts],
                        "accepted_prefix": int(accepted_prefix),
                        "accepted_commit_blocker": accepted_commit_blocker or "",
                        "candidate_copy_rows": int(candidate_copy_rows),
                        "copy_rows": int(copy_rows),
                        "emitted_tail": [
                            int(tok.reshape(-1)[0].item()) for tok in emitted_now
                        ],
                    }
                )

        commit_start_s = time.perf_counter()
        copied_bytes = 0
        copied_tokens = 0
        rollback_start_s = time.perf_counter()
        if commit_loc_chunks:
            commit_locs = torch.cat(commit_loc_chunks, dim=0).to(
                device=self.device,
                dtype=torch.long,
            )
            commit_positions = torch.cat(commit_position_chunks, dim=0).to(
                device=self.device,
                dtype=torch.long,
            )
            copied_tokens = int(commit_locs.numel())
            committed_snapshot = self._snapshot_mtp_kv_rows(commit_locs, commit_positions)
            copied_bytes = int(committed_snapshot.get("bytes", 0))
            self._restore_mtp_kv_snapshot(pre_verify_snapshot)
            self._restore_mtp_kv_snapshot(committed_snapshot)
            for req, copy_rows in copy_rows_by_entry:
                for _ in range(int(copy_rows)):
                    req.complete_one()
        else:
            self._restore_mtp_kv_snapshot(pre_verify_snapshot)
        self._sync_device_for_mtp_spec()
        rollback_elapsed = time.perf_counter() - rollback_start_s
        commit_elapsed = time.perf_counter() - commit_start_s
        self.mtp_spec_stats["target_commit_latency_s"] = float(
            self.mtp_spec_stats["target_commit_latency_s"]
        ) + commit_elapsed
        self.mtp_spec_stats["target_rollback_latency_s"] = float(
            self.mtp_spec_stats.get("target_rollback_latency_s", 0.0)
        ) + rollback_elapsed
        self.mtp_spec_stats["target_commit_kv_copies"] = int(
            self.mtp_spec_stats["target_commit_kv_copies"]
        ) + (1 if copied_tokens else 0)
        self.mtp_spec_stats["accepted_kv_copied_bytes"] = int(
            self.mtp_spec_stats["accepted_kv_copied_bytes"]
        ) + int(copied_bytes)
        self.mtp_spec_stats["accepted_kv_copied_tokens"] = int(
            self.mtp_spec_stats["accepted_kv_copied_tokens"]
        ) + int(copied_tokens)
        self.mtp_spec_stats["accepted_kv_commit_blocked_rows"] = int(
            self.mtp_spec_stats["accepted_kv_commit_blocked_rows"]
        ) + int(blocked_commit_rows)
        self.mtp_spec_stats["draft_tokens_accept_candidates"] = int(
            self.mtp_spec_stats["draft_tokens_accept_candidates"]
        ) + int(accepted_candidates)
        self.mtp_spec_stats["target_correction_token_candidates"] = int(
            self.mtp_spec_stats["target_correction_token_candidates"]
        ) + int(correction_token_candidates)
        self.mtp_spec_stats["target_bonus_token_candidates"] = int(
            self.mtp_spec_stats["target_bonus_token_candidates"]
        ) + int(bonus_token_candidates)
        self.mtp_spec_stats["target_bonus_tokens"] = int(
            self.mtp_spec_stats["target_bonus_tokens"]
        ) + int(bonus_tokens)
        self.mtp_spec_stats["rejected_tail_isolation_checks"] = int(
            self.mtp_spec_stats["rejected_tail_isolation_checks"]
        ) + int(len(entries))
        if trace_entries:
            trace_log = self.mtp_spec_stats.setdefault("debug_trace", [])
            trace_log.extend(trace_entries)
            if len(trace_log) > 64:
                del trace_log[:-64]

        self._record_mtp_acceptance_histogram(accepted_prefix_lens)
        self.mtp_spec_stats["draft_tokens_verified"] = int(
            self.mtp_spec_stats["draft_tokens_verified"]
        ) + verified
        self.mtp_spec_stats["draft_tokens_accepted"] = int(
            self.mtp_spec_stats["draft_tokens_accepted"]
        ) + accepted
        self.mtp_spec_stats["draft_tokens_rejected"] = int(
            self.mtp_spec_stats["draft_tokens_rejected"]
        ) + rejected
        self.mtp_spec_stats["target_correction_tokens"] = int(
            self.mtp_spec_stats["target_correction_tokens"]
        ) + correction_tokens
        self.mtp_spec_stats["target_fallback_tokens"] = int(
            self.mtp_spec_stats.get("target_fallback_tokens", 0)
        ) + target_fallback_tokens
        self.mtp_spec_stats["fallback_page_boundary_tokens"] = int(
            self.mtp_spec_stats["fallback_page_boundary_tokens"]
        ) + page_boundary_stops
        shape_log = self.mtp_spec_stats["target_verify_batch_shapes"]
        shape_log.extend(verify_shapes)
        if len(shape_log) > 64:
            del shape_log[:-64]
        return {
            "verified": verified,
            "accepted": accepted,
            "rejected": rejected,
            "correction_tokens": correction_tokens,
            "target_fallback_tokens": target_fallback_tokens,
            "page_boundary_stops": page_boundary_stops,
            "accepted_prefix_lens": accepted_prefix_lens,
            "verify_shapes": verify_shapes,
        }

    def _verify_mtp_spec_drafts(
        self,
        batch: Batch,
        first_tokens_gpu: torch.Tensor,
        draft_tokens_by_index: dict[int, list[torch.Tensor]],
        emitted_tokens: list[list[torch.Tensor]],
    ) -> dict[str, Any]:
        return self._verify_mtp_spec_drafts_flattened(
            batch,
            first_tokens_gpu,
            draft_tokens_by_index,
            emitted_tokens,
        )

    def _verify_mtp_spec_drafts_serial(
        self,
        batch: Batch,
        first_tokens_gpu: torch.Tensor,
        draft_tokens_by_index: dict[int, list[torch.Tensor]],
        emitted_tokens: list[list[torch.Tensor]],
    ) -> dict[str, Any]:
        draft_len = int(self.dsv4_mtp_spec_draft_len)
        states: dict[int, dict[str, Any]] = {}
        for i, req in enumerate(batch.reqs):
            drafts = draft_tokens_by_index.get(i, [])
            if drafts and req.can_decode:
                states[i] = {
                    "req": req,
                    "drafts": drafts,
                    "prev_token": first_tokens_gpu[i : i + 1].contiguous(),
                    "accepted_prefix": 0,
                    "draft_active": True,
                }

        verified = 0
        accepted = 0
        rejected = 0
        correction_tokens = 0
        target_fallback_tokens = 0
        page_boundary_stops = 0
        verify_shapes: list[list[int]] = []
        accepted_prefix_lens = [0] * len(batch.reqs)

        for depth in range(draft_len):
            rows: list[tuple[int, dict[str, Any]]] = []
            for batch_index, entry in states.items():
                req = entry["req"]
                if not req.can_decode:
                    continue
                if int(req.cached_len) >= self._allocated_kv_token_limit(req):
                    page_boundary_stops += 1
                    continue
                rows.append((batch_index, entry))
            if not rows:
                break

            input_ids = torch.cat([entry["prev_token"] for _, entry in rows], dim=0).to(
                device=self.device, dtype=torch.int32
            )
            verify_positions = [int(entry["req"].cached_len) for _, entry in rows]
            verify_batch = self._make_mtp_frozen_batch(
                [entry["req"] for _, entry in rows],
                input_ids,
                verify_positions,
                read_only=False,
            )

            self._sync_device_for_mtp_spec()
            start_s = time.perf_counter()
            with self.ctx.forward_batch(verify_batch):
                self.graph_runner.record_eager_decode(verify_batch)
                target_output = self.model.forward_with_hidden()
                self._debug_sync_forward(
                    "after_mtp_target_verify",
                    verify_batch,
                    "mtp_spec_target_verify",
                )
            self._sync_device_for_mtp_spec()
            elapsed = time.perf_counter() - start_s
            self.mtp_spec_stats["target_latency_s"] = float(
                self.mtp_spec_stats["target_latency_s"]
            ) + elapsed
            self.mtp_spec_stats["target_verify_latency_s"] = float(
                self.mtp_spec_stats["target_verify_latency_s"]
            ) + elapsed
            self.mtp_spec_stats["target_calls"] = int(self.mtp_spec_stats["target_calls"]) + 1
            self.mtp_spec_stats["target_verify_calls"] = int(
                self.mtp_spec_stats["target_verify_calls"]
            ) + 1
            verify_shapes.append([len(rows), 1])

            logits = target_output.logits[: len(rows)]
            if not bool(torch.isfinite(logits).all().item()):
                self.mtp_spec_stats["finite_failures"] = int(
                    self.mtp_spec_stats["finite_failures"]
                ) + 1
                raise RuntimeError("DeepSeek V4 MTP target verify produced NaN/Inf logits.")
            target_tokens = torch.argmax(logits, dim=-1).to(torch.int32)

            next_states: dict[int, dict[str, Any]] = {}
            for row, (batch_index, entry) in enumerate(rows):
                req = entry["req"]
                target_token = target_tokens[row : row + 1].contiguous()
                req.complete_one()

                draft_token = (
                    entry["drafts"][depth]
                    if bool(entry["draft_active"]) and depth < len(entry["drafts"])
                    else None
                )
                if draft_token is not None:
                    verified += 1
                    if bool((target_token == draft_token.to(device=self.device)).all().item()):
                        accepted += 1
                        entry["accepted_prefix"] = int(entry["accepted_prefix"]) + 1
                        accepted_prefix_lens[batch_index] = int(entry["accepted_prefix"])
                        emitted_tokens[batch_index].append(draft_token.contiguous())
                        entry["prev_token"] = draft_token.contiguous()
                        if (
                            req.can_decode
                            and depth + 1 < draft_len
                            and not self._is_stop_token_for_req(req, draft_token)
                        ):
                            next_states[batch_index] = entry
                    else:
                        rejected += 1
                        correction_tokens += 1
                        emitted_tokens[batch_index].append(target_token)
                        entry["prev_token"] = target_token
                        entry["draft_active"] = False
                        if (
                            req.can_decode
                            and depth + 1 < draft_len
                            and not self._is_stop_token_for_req(req, target_token)
                        ):
                            next_states[batch_index] = entry
                else:
                    target_fallback_tokens += 1
                    emitted_tokens[batch_index].append(target_token)
                    entry["prev_token"] = target_token
                    if (
                        req.can_decode
                        and depth + 1 < draft_len
                        and not self._is_stop_token_for_req(req, target_token)
                    ):
                        next_states[batch_index] = entry
            states = next_states

        self._record_mtp_acceptance_histogram(accepted_prefix_lens)
        self.mtp_spec_stats["draft_tokens_verified"] = int(
            self.mtp_spec_stats["draft_tokens_verified"]
        ) + verified
        self.mtp_spec_stats["draft_tokens_accepted"] = int(
            self.mtp_spec_stats["draft_tokens_accepted"]
        ) + accepted
        self.mtp_spec_stats["draft_tokens_rejected"] = int(
            self.mtp_spec_stats["draft_tokens_rejected"]
        ) + rejected
        self.mtp_spec_stats["target_correction_tokens"] = int(
            self.mtp_spec_stats["target_correction_tokens"]
        ) + correction_tokens
        self.mtp_spec_stats["target_fallback_tokens"] = int(
            self.mtp_spec_stats.get("target_fallback_tokens", 0)
        ) + target_fallback_tokens
        self.mtp_spec_stats["fallback_page_boundary_tokens"] = int(
            self.mtp_spec_stats["fallback_page_boundary_tokens"]
        ) + page_boundary_stops
        shape_log = self.mtp_spec_stats["target_verify_batch_shapes"]
        shape_log.extend(verify_shapes)
        if len(shape_log) > 64:
            del shape_log[:-64]
        return {
            "verified": verified,
            "accepted": accepted,
            "rejected": rejected,
            "correction_tokens": correction_tokens,
            "target_fallback_tokens": target_fallback_tokens,
            "page_boundary_stops": page_boundary_stops,
            "accepted_prefix_lens": accepted_prefix_lens,
            "verify_shapes": verify_shapes,
        }

    def _pack_accepted_tokens(
        self,
        emitted_tokens: list[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
        bs = len(emitted_tokens)
        max_len = max((len(tokens) for tokens in emitted_tokens), default=1)
        accepted_gpu = torch.zeros((bs, max_len), dtype=torch.int32, device=self.device)
        lens_values = tuple(int(len(tokens)) for tokens in emitted_tokens)
        lens = torch.tensor(lens_values, dtype=torch.int32, device=self.device)
        for i, tokens in enumerate(emitted_tokens):
            if tokens:
                accepted_gpu[i, : len(tokens)] = torch.cat(tokens, dim=0).to(
                    device=self.device,
                    dtype=torch.int32,
                )
        first_gpu = accepted_gpu[:, 0].contiguous()
        accepted_cpu = accepted_gpu.to("cpu", non_blocking=True)
        lens_cpu = lens.to("cpu", non_blocking=True)
        return first_gpu, accepted_gpu, accepted_cpu, lens_cpu, lens_values

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

        emitted_tokens: list[list[torch.Tensor]] = [
            [next_tokens_gpu[i : i + 1].contiguous()] for i in range(batch.size)
        ]
        draft = self._propose_mtp_spec_drafts(
            batch,
            next_tokens_gpu,
            target_output.hidden_states_before_norm[: batch.size],
            mtp_positions,
        )
        verification = self._verify_mtp_spec_drafts(
            batch,
            next_tokens_gpu,
            draft["draft_tokens_by_index"],
            emitted_tokens,
        )
        self.mtp_spec_stats["last_batch"] = {
            "phase": batch.phase,
            "batch_size": int(batch.size),
            "verified": int(verification["verified"]),
            "accepted": int(verification["accepted"]),
            "rejected": int(verification["rejected"]),
            "correction_tokens": int(verification["correction_tokens"]),
            "target_fallback_tokens": int(verification["target_fallback_tokens"]),
            "proposed": int(draft["proposed"]),
            "accepted_lens": [len(tokens) for tokens in emitted_tokens],
            "accepted_prefix_lens": verification["accepted_prefix_lens"],
        }
        self.mtp_spec_stats["emitted_tokens"] = int(self.mtp_spec_stats["emitted_tokens"]) + sum(
            len(tokens) for tokens in emitted_tokens
        )

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
            f"sampler_logits_staging.accepted_tokens_to_cpu.bs{batch.size}",
            next_tokens=next_tokens_gpu,
        ):
            (
                next_tokens_gpu,
                accepted_tokens_gpu,
                accepted_tokens_cpu,
                accepted_lens_cpu,
                accepted_lens,
            ) = self._pack_accepted_tokens(emitted_tokens)
            next_tokens_cpu = accepted_tokens_cpu[:, 0].contiguous()
        copy_done_event = self._acquire_copy_done_event()
        copy_done_event.record(self.stream)
        return ForwardOutput(
            next_tokens_gpu,
            next_tokens_cpu,
            copy_done_event,
            accepted_tokens_gpu=accepted_tokens_gpu,
            accepted_tokens_cpu=accepted_tokens_cpu,
            accepted_lens_cpu=accepted_lens_cpu,
            accepted_lens=accepted_lens,
        )

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
            if draft_len not in {1, 2, 4}:
                raise ValueError(
                    "DeepSeek V4 MTP speculative frozen-KV runtime supports "
                    f"draft_len in {{1, 2, 4}}, got {draft_len}."
                )
            override("enable_dsv4_mtp_speculative", True)
            override("enable_dsv4_mtp", True)
            override("dsv4_mtp_spec_draft_len", draft_len)
            override("allow_dsv4_cuda_graph", False)
            override("cuda_graph_bs", [])
            override("cuda_graph_max_bs", 0)
            override("cuda_graph_capture_greedy_sample", False)
            os.environ[_DSV4_MTP_SPECULATIVE_ENV] = "1"
            os.environ[_DSV4_EXPERIMENTAL_MTP_ENV] = "1"
            logger.info_rank0(
                "Opting in to experimental DeepSeek V4 greedy MTP speculative "
                f"frozen-KV runtime: draft_len={draft_len}, sampling disabled, "
                "CUDA graph disabled."
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

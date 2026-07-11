from __future__ import annotations

import gc
import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import dsv4_direct_copy_nvtx, dsv4_memory_debug, dsv4_owner_timing, init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
DSV4_CUDA_GRAPH_EXACT_BS_ONLY_ENV = "MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY"
DSV4_GRAPH_CAPTURE_STAGE_DEBUG_ENV = "MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG"
DSV4_GRAPH_REPLAY_TIMING_ENV = "MINISGL_DSV4_GRAPH_REPLAY_TIMING"
DSV4_GRAPH_REPLAY_TIMING_MAX_SAMPLES_ENV = "MINISGL_DSV4_GRAPH_REPLAY_TIMING_MAX_SAMPLES"
DSV4_AUDIT_LOG_DIR_ENV = "MINISGL_DSV4_AUDIT_LOG_DIR"
DSV4_AUDIT_RUN_LABEL_ENV = "MINISGL_DSV4_AUDIT_RUN_LABEL"
DSV4_CASE_BOUNDARY_DEBUG_ENV = "MINISGL_DSV4_CASE_BOUNDARY_DEBUG"
_MARLIN_WNA16_RELEASE_ENV = "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"
_MARLIN_WNA16_RELEASE_TIMING_ENV = "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING"
_MARLIN_WNA16_RELEASE_AFTER_GRAPH_CAPTURE_ENV = (
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_AFTER_GRAPH_CAPTURE"
)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    num_token_non_padded: torch.Tensor
    logits: torch.Tensor
    next_tokens: torch.Tensor | None

    @classmethod
    def init(
        cls,
        bs: int,
        vocab_size: int,
        device: torch.device,
        *,
        capture_greedy_sample: bool = False,
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            num_token_non_padded=torch.zeros(1, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            next_tokens=(
                torch.empty(bs, dtype=torch.int32, device=device)
                if capture_greedy_sample
                else None
            ),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        self.num_token_non_padded.fill_(batch.size)
        batch.num_token_non_padded = self.num_token_non_padded

    def nbytes(self) -> int:
        total = (
            self.input_ids.numel() * self.input_ids.element_size()
            + self.out_loc.numel() * self.out_loc.element_size()
            + self.positions.numel() * self.positions.element_size()
            + self.num_token_non_padded.numel()
            * self.num_token_non_padded.element_size()
            + self.logits.numel() * self.logits.element_size()
        )
        if self.next_tokens is not None:
            total += self.next_tokens.numel() * self.next_tokens.element_size()
        return int(total)

    def record_marlin_wna16_owner_allocations(self, stage: str) -> None:
        dsv4_memory_debug.record_owner_tensors(
            owner_prefix="graph.capture_buffer",
            stage=stage,
            tensors={
                "input_ids": self.input_ids,
                "out_loc": self.out_loc,
                "positions": self.positions,
                "num_token_non_padded": self.num_token_non_padded,
                "logits": self.logits,
                "next_tokens": self.next_tokens,
            },
        )

    def copy_from(self, batch: Batch) -> int:
        _slice = slice(batch.padded_size)
        timing_base = {
            "phase": "decode" if batch.is_decode else batch.phase,
            "batch_size": int(batch.size),
            "padded_size": int(batch.padded_size),
            "rows": int(batch.padded_size),
        }
        with dsv4_owner_timing.maybe_host_range("graph.copy_from.total", timing_base):
            # One graph input write per replay.  The captured kernels only see
            # this stable device address and never read the value back on CPU.
            self.num_token_non_padded.fill_(batch.size)
            batch.num_token_non_padded = self.num_token_non_padded
            with dsv4_direct_copy_nvtx(
                f"graph_input_staging.input_ids.bs{batch.size}.padded{batch.padded_size}",
                dst=self.input_ids[_slice],
                src=batch.input_ids,
            ):
                with dsv4_owner_timing.maybe_host_range(
                    "graph.copy_from.input_ids",
                    {**timing_base, "field": "input_ids"},
                ):
                    self.input_ids[_slice] = batch.input_ids
            with dsv4_direct_copy_nvtx(
                f"graph_input_staging.out_loc.bs{batch.size}.padded{batch.padded_size}",
                dst=self.out_loc[_slice],
                src=batch.out_loc,
            ):
                with dsv4_owner_timing.maybe_host_range(
                    "graph.copy_from.out_loc",
                    {**timing_base, "field": "out_loc"},
                ):
                    self.out_loc[_slice] = batch.out_loc
            with dsv4_direct_copy_nvtx(
                f"graph_input_staging.positions.bs{batch.size}.padded{batch.padded_size}",
                dst=self.positions[_slice],
                src=batch.positions,
            ):
                with dsv4_owner_timing.maybe_host_range(
                    "graph.copy_from.positions",
                    {**timing_base, "field": "positions"},
                ):
                    self.positions[_slice] = batch.positions
        copied_items = int(batch.padded_size)
        return copied_items * (
            self.input_ids.element_size()
            + self.out_loc.element_size()
            + self.positions.element_size()
        ) + self.num_token_non_padded.element_size()


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _marlin_wna16_release_timing() -> str:
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


def _sanitize_report_key(stage: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in stage).strip("_") or "stage"


def _audit_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(name)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
    return 0


def _audit_log_dir() -> str:
    return os.environ.get(
        DSV4_AUDIT_LOG_DIR_ENV,
        "performance_milestones/target08_indexer_capture_static_width_audit/raw",
    )


def _audit_run_label() -> str:
    raw = os.environ.get(DSV4_AUDIT_RUN_LABEL_ENV, "run").strip()
    return raw or "run"


def _append_audit_jsonl(kind: str, payload: dict) -> None:
    try:
        directory = _audit_log_dir()
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory,
            f"{kind}_{_audit_run_label()}_rank{_audit_rank()}.jsonl",
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
    except Exception:
        return


def _capture_stage_snapshot(
    device: torch.device,
    *,
    stage: str,
    batch_size: int | None,
    previous: dict | None,
    baseline: dict | None,
) -> dict:
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    payload = {
        "event": "dsv4_graph_capture_stage_memory",
        "stage": stage,
        "batch_size": None if batch_size is None else int(batch_size),
        "rank": _audit_rank(),
        "pid": os.getpid(),
        "free_memory_bytes": int(free_memory),
        "total_memory_bytes": int(total_memory),
        "memory_allocated_bytes": int(allocated),
        "memory_reserved_bytes": int(reserved),
        "max_memory_allocated_bytes": int(max_allocated),
        "max_memory_reserved_bytes": int(max_reserved),
    }
    if previous is not None:
        payload["free_delta_from_previous_bytes"] = int(
            previous["free_memory_bytes"] - free_memory
        )
        payload["memory_allocated_delta_from_previous_bytes"] = int(
            allocated - previous["memory_allocated_bytes"]
        )
        payload["memory_reserved_delta_from_previous_bytes"] = int(
            reserved - previous["memory_reserved_bytes"]
        )
    if baseline is not None:
        payload["free_delta_from_baseline_bytes"] = int(
            baseline["free_memory_bytes"] - free_memory
        )
        payload["memory_allocated_delta_from_baseline_bytes"] = int(
            allocated - baseline["memory_allocated_bytes"]
        )
        payload["memory_reserved_delta_from_baseline_bytes"] = int(
            reserved - baseline["memory_reserved_bytes"]
        )
    return payload


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        resolved_graph_bs: tuple[int, ...],
        graph_policy_report: dict[str, object],
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        capture_fail_open: bool = False,
        capture_greedy_sample: bool = False,
    ) -> None:
        cuda_graph_bs = resolved_graph_bs
        self.attn_backend = attn_backend
        self.model = model
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.capture_fail_open = capture_fail_open
        self.capture_greedy_sample = capture_greedy_sample
        self._marlin_wna16_debug_release_done = False
        self.exact_bs_only = (
            os.environ.get(DSV4_CUDA_GRAPH_EXACT_BS_ONLY_ENV, "").strip().lower()
            in _TRUE_ENV_VALUES
        )
        self._replay_timing_enabled = _truthy_env(DSV4_GRAPH_REPLAY_TIMING_ENV)
        self._replay_timing_max_samples = max(
            0, _int_env(DSV4_GRAPH_REPLAY_TIMING_MAX_SAMPLES_ENV, 32)
        )
        self.capture_status = {
            "enabled": bool(cuda_graph_bs),
            "bucket_policy": graph_policy_report,
            "exact_bs_only": bool(self.exact_bs_only),
            "requested_bs": list(self.graph_bs_list),
            "captured_bs": [],
            "capture_greedy_sample": bool(capture_greedy_sample),
            "capture_elapsed_s": None,
            "capture_free_memory_before_bytes": None,
            "capture_free_memory_after_bytes": None,
            "capture_memory_delta_bytes": None,
            "capture_memory_allocated_before_bytes": None,
            "capture_memory_allocated_after_bytes": None,
            "capture_memory_reserved_before_bytes": None,
            "capture_memory_reserved_after_bytes": None,
            "capture_peak_memory_allocated_bytes": None,
            "capture_peak_memory_reserved_bytes": None,
            "capture_buffer_bytes": None,
            "capture_stage_memory_ledger": [],
            "capture_graph_pool_reuse_enabled": False,
            "capture_graph_pool_reuse_anchor_bs": None,
            "capture_by_batch_size": {},
            "replay_count": 0,
            "replay_count_by_batch_size": {},
            "replay_count_by_padded_size": {},
            "greedy_sample_replay_count": 0,
            "greedy_sample_replay_count_by_batch_size": {},
            "replay_input_copy_bytes": 0,
            "replay_timing": {
                "enabled": bool(self._replay_timing_enabled),
                "sync_before_after_replay": bool(self._replay_timing_enabled),
                "max_samples": int(self._replay_timing_max_samples),
                "count": 0,
                "total_s": 0.0,
                "min_s": None,
                "max_s": None,
                "by_batch_size": {},
                "by_padded_size": {},
                "samples": [],
            },
            "eager_decode_count": 0,
            "eager_decode_count_by_batch_size": {},
            "error": None,
        }
        try:
            self._capture_graphs(max_seq_len, vocab_size, model)
        except BaseException as exc:
            self.capture_status["error"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
            self.graph_map = {}
            self.max_graph_bs = 0
            if capture_fail_open:
                logger.error(
                    "CUDA graph capture failed; falling back to eager decode because "
                    f"capture_fail_open=True. Blocker: {type(exc).__name__}: {exc}"
                )
            else:
                logger.error(f"CUDA graph capture failed: {type(exc).__name__}: {exc}")
                raise

    def _maybe_release_marlin_wna16_for_timing(self, *, timing: str, stage_label: str) -> None:
        if self._marlin_wna16_debug_release_done:
            return
        if not dsv4_memory_debug.env_flag(_MARLIN_WNA16_RELEASE_ENV):
            return
        if _marlin_wna16_release_timing() != timing:
            return
        release = getattr(self.model, "release_marlin_wna16_original_expert_weights", None)
        if not callable(release):
            return
        report = release(stage_label=stage_label)
        self._marlin_wna16_debug_release_done = True
        self.capture_status[f"moe_marlin_wna16_{stage_label}"] = report
        self._check_marlin_wna16_release_guards(f"{stage_label}:after")

    def record_marlin_wna16_owner_allocations(self, stage: str) -> None:
        buffer = getattr(self, "buffer", None)
        if buffer is not None:
            buffer.record_marlin_wna16_owner_allocations(stage)

    def _check_marlin_wna16_release_guards(self, stage: str) -> None:
        check = getattr(self.model, "check_marlin_wna16_release_guards", None)
        if not callable(check):
            return
        report = check(stage)
        if not isinstance(report, dict) or not report.get("enabled", False):
            return
        key = f"moe_marlin_wna16_guard_{_sanitize_report_key(stage)}"
        self.capture_status[key] = {
            k: v for k, v in report.items() if k != "records"
        }

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        stage_debug = _truthy_env(DSV4_GRAPH_CAPTURE_STAGE_DEBUG_ENV)
        stage_ledger: list[dict] = self.capture_status["capture_stage_memory_ledger"]

        def record_stage(stage: str, batch_size: int | None = None) -> None:
            if not stage_debug:
                return
            torch.cuda.synchronize(self.device)
            baseline = stage_ledger[0] if stage_ledger else None
            previous = stage_ledger[-1] if stage_ledger else None
            payload = _capture_stage_snapshot(
                self.device,
                stage=stage,
                batch_size=batch_size,
                previous=previous,
                baseline=baseline,
            )
            stage_ledger.append(payload)
            _append_audit_jsonl("graph_capture_stage", payload)

        record_stage("before_attn_backend.init_capture_graph")
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)
        record_stage("after_attn_backend.init_capture_graph")

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        record_stage("before_GraphCaptureBuffer.init")

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        capture_start_free_memory = free_memory
        capture_start_s = time.perf_counter()
        self.capture_status["capture_free_memory_before_bytes"] = int(free_memory)
        self.capture_status["capture_memory_allocated_before_bytes"] = int(
            torch.cuda.memory_allocated(self.device)
        )
        self.capture_status["capture_memory_reserved_before_bytes"] = int(
            torch.cuda.memory_reserved(self.device)
        )
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs,
            vocab_size,
            self.device,
            capture_greedy_sample=self.capture_greedy_sample,
        )
        self.buffer.record_marlin_wna16_owner_allocations("after_GraphCaptureBuffer.init")
        record_stage("after_GraphCaptureBuffer.init")
        self.capture_status["capture_buffer_bytes"] = self.buffer.nbytes()
        bind_capture_graph_inputs = getattr(
            self.attn_backend, "bind_capture_graph_inputs", None
        )
        if bind_capture_graph_inputs is not None:
            bind_capture_graph_inputs(
                input_ids=self.buffer.input_ids,
                out_loc=self.buffer.out_loc,
                positions=self.buffer.positions,
            )
        stage_capture_metadata = getattr(
            self.attn_backend, "stage_capture_metadata_for_graph", None
        )
        self.capture_status["capture_compressed_locs_in_graph"] = bool(
            getattr(self.attn_backend, "capture_compressed_locs_in_graph", False)
        )
        self.capture_status["capture_compressed_locs_in_graph_disabled_by_env"] = bool(
            getattr(self.attn_backend, "capture_compressed_locs_in_graph_disabled_by_env", False)
        )
        self.capture_status["capture_compressed_locs_in_graph_component_guarded"] = bool(
            getattr(
                self.attn_backend,
                "capture_compressed_locs_in_graph_component_guarded",
                False,
            )
        )
        self.capture_status["prep_metadata_in_graph_requested"] = bool(
            getattr(self.attn_backend, "prep_metadata_in_graph_requested", False)
        )
        self.capture_status["prep_metadata_in_graph"] = bool(
            getattr(self.attn_backend, "prep_metadata_in_graph", False)
        )
        self.capture_status["prep_metadata_in_graph_unsupported_reason"] = getattr(
            self.attn_backend,
            "prep_metadata_in_graph_unsupported_reason",
            None,
        )

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            bs_start_free_memory = free_memory
            bs_start_s = time.perf_counter()
            bs_start_allocated = torch.cuda.memory_allocated(self.device)
            bs_start_reserved = torch.cuda.memory_reserved(self.device)
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            record_stage("before_prepare_for_capture", bs)
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            record_stage("after_prepare_for_capture", bs)
            with get_global_ctx().forward_batch(batch):
                if stage_capture_metadata is not None:
                    stage_capture_metadata(batch)
                record_stage("after_stage_capture_metadata", bs)
                self._maybe_release_marlin_wna16_for_timing(
                    timing="before_warmup_forward",
                    stage_label=f"before_warmup_forward_bs{int(bs)}_release",
                )
                self._check_marlin_wna16_release_guards(
                    f"before_warmup_forward_bs{int(bs)}"
                )
                with dsv4_memory_debug.warmup_forward_context(
                    label="graph_capture_warmup_model_forward",
                    batch_size=bs,
                    device=self.device,
                ):
                    self.buffer.logits[:bs] = model.forward()
                self._check_marlin_wna16_release_guards(
                    f"after_warmup_forward_bs{int(bs)}"
                )
                self._maybe_release_marlin_wna16_for_timing(
                    timing="after_warmup_forward",
                    stage_label=f"after_warmup_forward_bs{int(bs)}_release",
                )
                if self.capture_greedy_sample:
                    assert self.buffer.next_tokens is not None
                    self.buffer.next_tokens[:bs] = torch.argmax(
                        self.buffer.logits[:bs], dim=-1
                    ).to(torch.int32)
                record_stage("after_warmup_model.forward", bs)
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    if stage_capture_metadata is not None:
                        stage_capture_metadata(batch)
                    self.buffer.logits[:bs] = model.forward()
                    if self.capture_greedy_sample:
                        assert self.buffer.next_tokens is not None
                        self.buffer.next_tokens[:bs] = torch.argmax(
                            self.buffer.logits[:bs], dim=-1
                        ).to(torch.int32)
            record_stage("after_actual_cuda_graph_capture", bs)
            self._check_marlin_wna16_release_guards(
                f"after_actual_cuda_graph_capture_bs{int(bs)}"
            )
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
                self.capture_status["capture_graph_pool_reuse_anchor_bs"] = int(bs)
            else:
                self.capture_status["capture_graph_pool_reuse_enabled"] = True
            self.graph_map[bs] = graph
            self.capture_status["captured_bs"].append(bs)
            torch.cuda.synchronize(self.device)
            if stage_debug:
                gc.collect()
                torch.cuda.empty_cache()
                record_stage("after_gc_empty_cache", bs)
            self._audit_marlin_wna16_cache_integrity(
                f"after_graph_capture_bs{int(bs)}_empty_cache"
            )
            bs_end_free_memory = get_free_memory(self.device)
            bs_end_allocated = torch.cuda.memory_allocated(self.device)
            bs_end_reserved = torch.cuda.memory_reserved(self.device)
            self.capture_status["capture_by_batch_size"][str(bs)] = {
                "elapsed_s": time.perf_counter() - bs_start_s,
                "free_memory_before_bytes": int(bs_start_free_memory),
                "free_memory_after_bytes": int(bs_end_free_memory),
                "memory_delta_bytes": int(bs_start_free_memory - bs_end_free_memory),
                "memory_allocated_before_bytes": int(bs_start_allocated),
                "memory_allocated_after_bytes": int(bs_end_allocated),
                "memory_allocated_delta_bytes": int(bs_end_allocated - bs_start_allocated),
                "memory_reserved_before_bytes": int(bs_start_reserved),
                "memory_reserved_after_bytes": int(bs_end_reserved),
                "memory_reserved_delta_bytes": int(bs_end_reserved - bs_start_reserved),
            }

        free_memory = get_free_memory(self.device)
        self.capture_status["capture_elapsed_s"] = time.perf_counter() - capture_start_s
        self.capture_status["capture_free_memory_after_bytes"] = int(free_memory)
        self.capture_status["capture_memory_delta_bytes"] = int(
            capture_start_free_memory - free_memory
        )
        self.capture_status["capture_memory_allocated_after_bytes"] = int(
            torch.cuda.memory_allocated(self.device)
        )
        self.capture_status["capture_memory_reserved_after_bytes"] = int(
            torch.cuda.memory_reserved(self.device)
        )
        self.capture_status["capture_peak_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(self.device)
        )
        self.capture_status["capture_peak_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(self.device)
        )
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")
        self._audit_marlin_wna16_cache_integrity("after_graph_capture_all")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if not batch.is_decode or batch.size > self.max_graph_bs:
            return False
        if self.exact_bs_only and batch.size not in self.graph_map:
            return False
        return True

    def _increment_status_counter(self, name: str, key: int) -> None:
        counter = self.capture_status[name]
        str_key = str(int(key))
        counter[str_key] = int(counter.get(str_key, 0)) + 1

    def _debug_sync_replay(self, stage: str, batch: Batch) -> None:
        if os.environ.get(DSV4_CASE_BOUNDARY_DEBUG_ENV, "").strip().lower() not in _TRUE_ENV_VALUES:
            return
        try:
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            raise RuntimeError(
                "DSV4 case-boundary debug CUDA sync failed during graph replay: "
                f"stage={stage}, phase={batch.phase}, bs={batch.size}, "
                f"padded={batch.padded_size}, replay_count={self.capture_status.get('replay_count')}, "
                f"captured_bs={self.capture_status.get('captured_bs')}"
            ) from exc

    def _debug_replay_context(self, batch: Batch) -> dict[str, object]:
        if os.environ.get(DSV4_CASE_BOUNDARY_DEBUG_ENV, "").strip().lower() not in _TRUE_ENV_VALUES:
            return {}

        def _range(tensor: torch.Tensor | None, rows: int) -> tuple[int | None, int | None]:
            if tensor is None or rows <= 0 or tensor.numel() == 0:
                return None, None
            view = tensor[: min(rows, int(tensor.numel()))]
            if view.numel() == 0:
                return None, None
            return int(view.min().item()), int(view.max().item())

        def _valid_range(tensor: torch.Tensor | None, rows: int) -> tuple[int | None, int | None]:
            if tensor is None or rows <= 0 or tensor.numel() == 0:
                return None, None
            if tensor.ndim == 1:
                view = tensor[: min(rows, int(tensor.shape[0]))]
            else:
                view = tensor[: min(rows, int(tensor.shape[0]))].reshape(-1)
            valid = view[view >= 0]
            if valid.numel() == 0:
                return None, None
            return int(valid.min().item()), int(valid.max().item())

        def _active_2d_range(
            tensor: torch.Tensor | None,
            lengths: torch.Tensor | None,
            rows: int,
        ) -> tuple[int | None, int | None]:
            if (
                tensor is None
                or lengths is None
                or rows <= 0
                or tensor.numel() == 0
                or lengths.numel() == 0
            ):
                return None, None
            rows = min(rows, int(tensor.shape[0]), int(lengths.shape[0]))
            if rows <= 0:
                return None, None
            width = int(tensor.shape[1]) if tensor.ndim == 2 else 0
            if width <= 0:
                return None, None
            lens = lengths[:rows].to(device=tensor.device, dtype=torch.long).clamp(min=0, max=width)
            cols = torch.arange(width, dtype=torch.long, device=tensor.device)
            active = cols[None, :] < lens[:, None]
            values = tensor[:rows][active]
            values = values[values >= 0]
            if values.numel() == 0:
                return None, None
            return int(values.min().item()), int(values.max().item())

        def _metadata_context(rows: int) -> dict[str, object]:
            attn_backend = getattr(self, "attn_backend", None)
            capture = getattr(attn_backend, "capture", None)
            core = getattr(capture, "core_metadata", None)
            if core is None or rows <= 0:
                return {}
            out: dict[str, object] = {
                "seq_lens_range": _range(getattr(core, "seq_lens", None), rows),
                "req_seq_lens_range": _range(getattr(core, "req_seq_lens", None), rows),
                "swa_ownership_version": int(getattr(core, "swa_ownership_version", -1)),
                "component_loc_ownership": bool(getattr(core, "component_loc_ownership", False)),
                "swa_source_elided": bool(getattr(core, "swa_source_elided_for_graph", False)),
                "c4_source_elided": bool(
                    getattr(core, "c4_sparse_source_elided_for_graph", False)
                ),
                "c128_source_elided": bool(getattr(core, "c128_source_elided_for_graph", False)),
                "swa_len_range": _range(getattr(core, "swa_topk_lengths", None), rows),
                "swa_active_range": _active_2d_range(
                    getattr(core, "swa_page_indices", None),
                    getattr(core, "swa_topk_lengths", None),
                    rows,
                ),
                "c4_len_range": _range(getattr(core, "c4_topk_lengths_raw", None), rows),
                "c4_sparse_len_range": _range(
                    getattr(core, "c4_sparse_topk_lengths", None),
                    rows,
                ),
                "c4_sparse_active_range": _active_2d_range(
                    getattr(core, "c4_sparse_page_indices", None),
                    getattr(core, "c4_sparse_topk_lengths", None),
                    rows,
                ),
                "c128_len_range": _range(
                    getattr(core, "c128_topk_lengths_clamp1", None),
                    rows,
                ),
                "c128_active_range": _valid_range(
                    getattr(core, "c128_page_indices", None),
                    rows,
                ),
                "c4_out_loc_range": _valid_range(getattr(core, "c4_out_loc", None), rows),
                "c128_out_loc_range": _valid_range(getattr(core, "c128_out_loc", None), rows),
                "c4_indexer_out_loc_range": _valid_range(
                    getattr(core, "c4_indexer_out_loc", None),
                    rows,
                ),
            }
            return out

        req_summaries = []
        for req in batch.padded_reqs[: min(batch.padded_size, 4)]:
            req_summaries.append(
                {
                    "uid": int(getattr(req, "uid", -1)),
                    "table_idx": int(getattr(req, "table_idx", -1)),
                    "cached_len": int(getattr(req, "cached_len", -1)),
                    "device_len": int(getattr(req, "device_len", -1)),
                    "extend_len": int(getattr(req, "extend_len", -1)),
                }
            )
        pos_min, pos_max = _range(getattr(batch, "positions", None), batch.padded_size)
        out_min, out_max = _range(getattr(batch, "out_loc", None), batch.padded_size)
        return {
            "phase": batch.phase,
            "bs": int(batch.size),
            "padded": int(batch.padded_size),
            "replay_count": int(self.capture_status.get("replay_count", 0)),
            "greedy_replay_count": int(
                self.capture_status.get("greedy_sample_replay_count", 0)
            ),
            "captured_bs": list(self.capture_status.get("captured_bs", [])),
            "positions_range": (pos_min, pos_max),
            "out_loc_range": (out_min, out_max),
            "reqs": req_summaries,
            "metadata": _metadata_context(int(batch.padded_size)),
        }

    def _record_replay_timing(self, batch: Batch, elapsed_s: float) -> None:
        timing = self.capture_status.get("replay_timing")
        if not isinstance(timing, dict):
            return
        count = int(timing.get("count") or 0) + 1
        total_s = float(timing.get("total_s") or 0.0) + float(elapsed_s)
        timing["count"] = count
        timing["total_s"] = total_s
        old_min = timing.get("min_s")
        old_max = timing.get("max_s")
        timing["min_s"] = (
            float(elapsed_s) if old_min is None else min(float(old_min), float(elapsed_s))
        )
        timing["max_s"] = (
            float(elapsed_s) if old_max is None else max(float(old_max), float(elapsed_s))
        )
        timing["mean_s"] = total_s / count if count else None

        def _update_bucket(section: str, key: int) -> None:
            buckets = timing.setdefault(section, {})
            bucket = buckets.setdefault(
                str(int(key)),
                {"count": 0, "total_s": 0.0, "min_s": None, "max_s": None},
            )
            bucket_count = int(bucket.get("count") or 0) + 1
            bucket_total_s = float(bucket.get("total_s") or 0.0) + float(elapsed_s)
            bucket_min = bucket.get("min_s")
            bucket_max = bucket.get("max_s")
            bucket["count"] = bucket_count
            bucket["total_s"] = bucket_total_s
            bucket["min_s"] = (
                float(elapsed_s)
                if bucket_min is None
                else min(float(bucket_min), float(elapsed_s))
            )
            bucket["max_s"] = (
                float(elapsed_s)
                if bucket_max is None
                else max(float(bucket_max), float(elapsed_s))
            )
            bucket["mean_s"] = bucket_total_s / bucket_count if bucket_count else None

        _update_bucket("by_batch_size", int(batch.size))
        _update_bucket("by_padded_size", int(batch.padded_size))

        samples = timing.setdefault("samples", [])
        if len(samples) < self._replay_timing_max_samples:
            samples.append(
                {
                    "replay_index": int(self.capture_status.get("replay_count", 0)) + 1,
                    "batch_size": int(batch.size),
                    "padded_size": int(batch.padded_size),
                    "elapsed_s": float(elapsed_s),
                }
            )

    def record_eager_decode(self, batch: Batch) -> None:
        if not batch.is_decode:
            return
        self.capture_status["eager_decode_count"] = int(
            self.capture_status["eager_decode_count"]
        ) + 1
        self._increment_status_counter("eager_decode_count_by_batch_size", batch.size)

    def _replay_to_buffer(self, batch: Batch) -> None:
        assert self.can_use_cuda_graph(batch)
        with dsv4_direct_copy_nvtx(
            f"graph_input_staging.copy_from.bs{batch.size}.padded{batch.padded_size}",
            input_ids=batch.input_ids,
            out_loc=batch.out_loc,
            positions=batch.positions,
        ):
            copied_bytes = self.buffer.copy_from(batch)
        if dsv4_owner_timing.enabled():
            metadata = {
                "phase": "decode",
                "rows": int(batch.size),
                "padded_rows": int(batch.padded_size),
                "field": "graph_input_staging",
                "group": "graph_input",
            }
            dsv4_owner_timing.record_counter(
                "dsv4.graph_replay.input_copy.bytes",
                metadata,
                value=int(copied_bytes),
            )
            dsv4_owner_timing.record_counter(
                "dsv4.graph_replay.input_copy.calls",
                metadata,
            )
        self._debug_sync_replay("after_input_staging", batch)
        g = self.graph_map[batch.padded_size]
        with dsv4_direct_copy_nvtx(
            f"replay_metadata_copy.prepare_for_replay.bs{batch.size}.padded{batch.padded_size}"
        ):
            with dsv4_owner_timing.maybe_host_range(
                "dsv4.graph_replay.prepare_for_replay",
                {
                    "phase": "decode",
                    "rows": int(batch.size),
                    "padded_rows": int(batch.padded_size),
                },
            ):
                self.attn_backend.prepare_for_replay(batch)
        self._debug_sync_replay("after_prepare_for_replay", batch)
        replay_context = self._debug_replay_context(batch)
        with dsv4_direct_copy_nvtx(
            f"static_graph_replay.g.replay.bs{batch.size}.padded{batch.padded_size}"
        ):
            try:
                if self._replay_timing_enabled:
                    torch.cuda.synchronize(self.device)
                    replay_start_s = time.perf_counter()
                g.replay()
                if self._replay_timing_enabled:
                    torch.cuda.synchronize(self.device)
                    self._record_replay_timing(batch, time.perf_counter() - replay_start_s)
            except Exception as exc:
                raise RuntimeError(
                    "DSV4 CUDA graph replay failed inside captured graph: "
                    f"context={replay_context}"
                ) from exc
        self._debug_sync_replay("after_graph_replay", batch)
        validate_after_replay = getattr(self.attn_backend, "validate_after_replay", None)
        if validate_after_replay is not None:
            validate_after_replay(batch)
        dump_padding_debug_boundaries = getattr(
            self.model, "dump_padding_debug_boundaries", None
        )
        if callable(dump_padding_debug_boundaries):
            dump_padding_debug_boundaries(batch)
        self.capture_status["replay_count"] = int(self.capture_status["replay_count"]) + 1
        self.capture_status["replay_input_copy_bytes"] = int(
            self.capture_status["replay_input_copy_bytes"]
        ) + copied_bytes
        self._increment_status_counter("replay_count_by_batch_size", batch.size)
        self._increment_status_counter("replay_count_by_padded_size", batch.padded_size)
        if int(self.capture_status["replay_count"]) <= 2:
            self._audit_marlin_wna16_cache_integrity(
                f"after_graph_replay_{int(self.capture_status['replay_count'])}"
            )

    def replay(self, batch: Batch) -> torch.Tensor:
        self._replay_to_buffer(batch)
        return self.buffer.logits[: batch.size]

    def can_replay_greedy_sample(self, batch: Batch) -> bool:
        return self.capture_greedy_sample and self.can_use_cuda_graph(batch)

    def replay_greedy_sample(self, batch: Batch) -> torch.Tensor:
        assert self.can_replay_greedy_sample(batch)
        self._replay_to_buffer(batch)
        assert self.buffer.next_tokens is not None
        self.capture_status["greedy_sample_replay_count"] = int(
            self.capture_status["greedy_sample_replay_count"]
        ) + 1
        self._increment_status_counter("greedy_sample_replay_count_by_batch_size", batch.size)
        return self.buffer.next_tokens[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        gc.collect()

    def _audit_marlin_wna16_cache_integrity(self, stage: str) -> None:
        audit = getattr(self.model, "audit_marlin_wna16_cache_integrity", None)
        if callable(audit):
            audit(stage)

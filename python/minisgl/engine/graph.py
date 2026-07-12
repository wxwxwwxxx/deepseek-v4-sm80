from __future__ import annotations

import gc
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


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
                torch.empty(bs, dtype=torch.int32, device=device) if capture_greedy_sample else None
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
            + self.num_token_non_padded.numel() * self.num_token_non_padded.element_size()
            + self.logits.numel() * self.logits.element_size()
        )
        if self.next_tokens is not None:
            total += self.next_tokens.numel() * self.next_tokens.element_size()
        return int(total)

    def copy_from(self, batch: Batch) -> int:
        _slice = slice(batch.padded_size)
        self.num_token_non_padded.fill_(batch.size)
        batch.num_token_non_padded = self.num_token_non_padded
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        copied_items = int(batch.padded_size)
        return (
            copied_items
            * (
                self.input_ids.element_size()
                + self.out_loc.element_size()
                + self.positions.element_size()
            )
            + self.num_token_non_padded.element_size()
        )


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


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
        self.exact_bs_only = False
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
            "capture_graph_pool_reuse_enabled": False,
            "capture_graph_pool_reuse_anchor_bs": None,
            "post_kv_model_cache_prepare_stage": None,
            "post_kv_model_cache_prepare_report": {},
            "capture_by_batch_size": {},
            "replay_count": 0,
            "replay_count_by_batch_size": {},
            "replay_count_by_padded_size": {},
            "greedy_sample_replay_count": 0,
            "greedy_sample_replay_count_by_batch_size": {},
            "replay_input_copy_bytes": 0,
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

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            self._prepare_post_kv_model_caches(model, stage="post_kv_allocation_graph_disabled")
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(
            f"Capturing {len(self.graph_bs_list)} CUDA graph buckets "
            f"through M={max(self.graph_bs_list)}."
        )
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

        self._prepare_post_kv_model_caches(
            model,
            stage="post_kv_allocation_pre_graph_warmup",
        )

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs,
            vocab_size,
            self.device,
            capture_greedy_sample=self.capture_greedy_sample,
        )
        self.capture_status["capture_buffer_bytes"] = self.buffer.nbytes()
        bind_capture_graph_inputs = getattr(self.attn_backend, "bind_capture_graph_inputs", None)
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
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                if stage_capture_metadata is not None:
                    stage_capture_metadata(batch)
                self.buffer.logits[:bs] = model.forward()
                if self.capture_greedy_sample:
                    assert self.buffer.next_tokens is not None
                    self.buffer.next_tokens[:bs] = torch.argmax(self.buffer.logits[:bs], dim=-1).to(
                        torch.int32
                    )
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    if stage_capture_metadata is not None:
                        stage_capture_metadata(batch)
                    self.buffer.logits[:bs] = model.forward()
                    if self.capture_greedy_sample:
                        assert self.buffer.next_tokens is not None
                        self.buffer.next_tokens[:bs] = torch.argmax(
                            self.buffer.logits[:bs], dim=-1
                        ).to(torch.int32)
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
                self.capture_status["capture_graph_pool_reuse_anchor_bs"] = int(bs)
            else:
                self.capture_status["capture_graph_pool_reuse_enabled"] = True
            self.graph_map[bs] = graph
            self.capture_status["captured_bs"].append(bs)
            torch.cuda.synchronize(self.device)
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

    def _prepare_post_kv_model_caches(self, model: BaseLLMModel, *, stage: str) -> None:
        prepare = getattr(model, "prepare_fused_wqa_wkv_bf16_weight_cache", None)
        report = prepare() if callable(prepare) else {}
        self.capture_status["post_kv_model_cache_prepare_stage"] = stage
        self.capture_status["post_kv_model_cache_prepare_report"] = report

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

    def record_eager_decode(self, batch: Batch) -> None:
        if not batch.is_decode:
            return
        self.capture_status["eager_decode_count"] = (
            int(self.capture_status["eager_decode_count"]) + 1
        )
        self._increment_status_counter("eager_decode_count_by_batch_size", batch.size)

    def _replay_to_buffer(self, batch: Batch) -> None:
        assert self.can_use_cuda_graph(batch)
        copied_bytes = self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        try:
            g.replay()
        except Exception as exc:
            raise RuntimeError("DSV4 CUDA graph replay failed inside captured graph") from exc
        validate_after_replay = getattr(self.attn_backend, "validate_after_replay", None)
        if validate_after_replay is not None:
            validate_after_replay(batch)
        self.capture_status["replay_count"] = int(self.capture_status["replay_count"]) + 1
        self.capture_status["replay_input_copy_bytes"] = (
            int(self.capture_status["replay_input_copy_bytes"]) + copied_bytes
        )
        self._increment_status_counter("replay_count_by_batch_size", batch.size)
        self._increment_status_counter("replay_count_by_padded_size", batch.padded_size)

    def replay(self, batch: Batch) -> torch.Tensor:
        self._replay_to_buffer(batch)
        return self.buffer.logits[: batch.size]

    def can_replay_greedy_sample(self, batch: Batch) -> bool:
        return self.capture_greedy_sample and self.can_use_cuda_graph(batch)

    def replay_greedy_sample(self, batch: Batch) -> torch.Tensor:
        assert self.can_replay_greedy_sample(batch)
        self._replay_to_buffer(batch)
        assert self.buffer.next_tokens is not None
        self.capture_status["greedy_sample_replay_count"] = (
            int(self.capture_status["greedy_sample_replay_count"]) + 1
        )
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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}



# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


@dataclass(frozen=True)
class MaxSequenceAdmission:
    input_len: int
    requested_output_len: int
    admitted_output_len: int
    max_seq_len: int
    accepted: bool
    rejection_reason: str | None = None


def decide_max_sequence_admission(
    *, input_len: int, requested_output_len: int, max_seq_len: int
) -> MaxSequenceAdmission:
    """Apply the maximum-total-sequence contract without model or GPU state."""
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if input_len < 0 or requested_output_len < 0:
        raise ValueError("input and requested output lengths must be non-negative")
    if input_len > max_seq_len:
        reason = (
            f"input sequence length {input_len} exceeds effective max sequence length "
            f"{max_seq_len}"
        )
        return MaxSequenceAdmission(
            input_len, requested_output_len, 0, max_seq_len, False, reason
        )
    available_output_len = max_seq_len - input_len
    if requested_output_len > 0 and available_output_len == 0:
        reason = (
            f"input sequence length {input_len} equals effective max sequence length "
            f"{max_seq_len}; no model position remains for generation"
        )
        return MaxSequenceAdmission(
            input_len, requested_output_len, 0, max_seq_len, False, reason
        )
    admitted_output_len = min(requested_output_len, available_output_len)
    return MaxSequenceAdmission(
        input_len,
        requested_output_len,
        admitted_output_len,
        max_seq_len,
        True,
    )


def validate_model_position_bound(*, max_device_len: int, rope_cache_len: int) -> int:
    """Return the largest model position, raising before an out-of-range forward."""
    if rope_cache_len <= 0:
        raise ValueError("rope_cache_len must be positive")
    observed_max_position = max_device_len - 1
    if observed_max_position >= rope_cache_len:
        raise RuntimeError(
            "model position exceeds the effective RoPE range: "
            f"position={observed_max_position}, rope_cache_len={rope_cache_len}"
        )
    return observed_max_position


def resolve_dsv4_cache_type(config: SchedulerConfig) -> str:
    cache_type = config.cache_type
    if not config.model_config.is_deepseek_v4:
        raise ValueError("This release supports DeepSeek V4 Flash only.")

    if config.enable_dsv4_swa_tail_retention_v1:
        raise RuntimeError(
            "TARGET 08.20 DSV4 SWA tail/component retention V1 is fail-closed. "
            "Mini's current DSV4 C4/C128/indexer/compression-state locations are "
            "derived from released full-token pages, so enabling this would risk "
            "dangling component reads or double frees. Keep using "
            "--enable-dsv4-radix-prefix-cache for the phase-1 full-page-owner "
            "baseline; see performance_milestones/target08_swa_tail_retention_v1/"
            "DESIGN.md."
        )
    if getattr(config, "enable_dsv4_swa_independent_lifecycle", False):
        if not config.enable_dsv4_radix_prefix_cache:
            raise ValueError(
                "DeepSeek V4 SWA independent lifecycle requires "
                "--enable-dsv4-radix-prefix-cache."
            )
        if not getattr(config, "enable_dsv4_component_loc_ownership", False):
            raise ValueError(
                "DeepSeek V4 SWA independent lifecycle requires "
                "--enable-dsv4-component-loc-ownership so C4/C128/indexer/state "
                "locations stay independent from released SWA/full rows."
            )
    if getattr(config, "enable_dsv4_component_loc_ownership", False):
        if not config.enable_dsv4_radix_prefix_cache:
            raise ValueError(
                "DeepSeek V4 component loc ownership requires the phase-1 radix "
                "prefix cache opt-in. Add --enable-dsv4-radix-prefix-cache."
            )
        window_size = int(getattr(config.model_config, "window_size", 128) or 128)
        if (
            window_size > config.page_size
            and not getattr(config, "enable_dsv4_swa_independent_lifecycle", False)
        ):
            raise ValueError(
                "DeepSeek V4 component loc ownership currently keeps one "
                "page-aligned SWA/full tail per retained node, so window_size "
                f"must be <= page_size. Got window_size={window_size}, "
                f"page_size={config.page_size}."
            )
    if config.enable_dsv4_radix_prefix_cache:
        if config.page_size % 128 != 0:
            raise ValueError(
                "DeepSeek V4 radix prefix cache requires a page size divisible "
                f"by 128, got page_size={config.page_size}. Use --page-size 256 "
                "for TARGET 08 runs."
            )
        if cache_type != "radix":
            raise ValueError(
                "DeepSeek V4 radix prefix cache opt-in requires "
                f"cache_type='radix', got {cache_type!r}."
            )
        logger.info("Opting in to DeepSeek V4 radix prefix cache.")
        if getattr(config, "enable_dsv4_component_loc_ownership", False):
            logger.info(
                "Opting in to DeepSeek V4 Route B component loc ownership "
                "for C4/C128/indexer components."
            )
        if getattr(config, "enable_dsv4_swa_independent_lifecycle", False):
            logger.info("Opting in to DeepSeek V4 independent SWA lifecycle.")
        return cache_type

    if cache_type != "naive":
        logger.info("Disabling radix prefix cache for DeepSeek V4 KV cache v1.")
    return "naive"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)
        # Engine resolves the immutable typed DSV4 mode into its cache/lifecycle
        # ownership fields. Select the prefix-cache implementation afterwards so
        # optimized cannot pair component ownership with NaivePrefixCache.
        cache_type = resolve_dsv4_cache_type(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # The final row is reserved for CUDA-graph padding.  Mirror the Engine's
        # explicitly selected diagnostic token without changing any live row.
        dummy_req = getattr(self.engine, "dummy_req", None)
        if dummy_req is not None:
            dummy_token = int(dummy_req.input_ids[-1].item())
            self.table_manager.token_pool[dummy_req.table_idx].fill_(dummy_token)
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            cache_type,
            kv_cache=self.engine.kv_cache,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        self.engine.release_copy_done_event(copy_done)
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    self.cache_manager.release_active_dsv4_swa_out_of_window(req)
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)
                else:
                    self.cache_manager.release_active_dsv4_swa_out_of_window(req)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            admission = decide_max_sequence_admission(
                input_len=len(msg.input_ids),
                requested_output_len=msg.sampling_params.max_tokens,
                max_seq_len=self.engine.max_seq_len,
            )
            if not admission.accepted:
                reason = admission.rejection_reason or "request rejected by max-sequence policy"
                logger.warning_rank0(
                    f"Request {msg.uid} rejected: {reason}."
                )
                self._record_max_sequence_admission(msg.uid, admission)
                self.send_result(
                    [
                        DetokenizeMsg(
                            uid=msg.uid,
                            next_token=self.eos_token_id,
                            finished=True,
                            finish_reason="length_rejected",
                            error=reason,
                        )
                    ]
                )
                return
            if admission.admitted_output_len != admission.requested_output_len:
                msg.sampling_params.max_tokens = admission.admitted_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {admission.admitted_output_len} for request "
                    f"{msg.uid}; requested total sequence length was "
                    f"{admission.input_len + admission.requested_output_len}, effective max is "
                    f"{admission.max_seq_len}."
                )
            self._record_max_sequence_admission(msg.uid, admission)
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _record_max_sequence_admission(
        self, uid: int, admission: MaxSequenceAdmission
    ) -> None:
        """Offline frontends override this to make clamps/rejections observable."""
        del uid, admission

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)



    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)

        self.cache_manager.allocate_paged(batch.reqs)

        batch.positions = _make_positions(batch, self.device)
        validate_model_position_bound(
            max_device_len=max((req.device_len for req in batch.reqs), default=0),
            rope_cache_len=int(
                getattr(self.engine, "effective_rope_cache_len", self.engine.max_seq_len)
            ),
        )

        input_mapping = _make_input_tuple(batch, self.device)

        write_mapping = _make_write_tuple(batch, self.device)

        batch.out_loc = self.engine.page_table[input_mapping]

        self.engine.attn_backend.prepare_metadata(batch)

        sample_args = self.engine.sampler.prepare(batch)

        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

        timing_metadata = {
            "phase": batch.phase,
            "batch_size": int(batch.size),
            "padded_size": int(getattr(batch, "padded_size", batch.size)),
        }
        self.engine.graph_runner.pad_batch(batch)

        timing_metadata["padded_size"] = int(batch.padded_size)
        self.cache_manager.allocate_paged(batch.reqs)

        batch.positions = _make_positions(batch, self.device)
        validate_model_position_bound(
            max_device_len=max((req.device_len for req in batch.reqs), default=0),
            rope_cache_len=int(
                getattr(self.engine, "effective_rope_cache_len", self.engine.max_seq_len)
            ),
        )

        input_mapping = _make_input_tuple(batch, self.device)

        write_mapping = _make_write_tuple(batch, self.device)

        batch.out_loc = self.engine.page_table[input_mapping]

        self.engine.attn_backend.prepare_metadata(batch)

        sample_args = self.engine.sampler.prepare(batch)

        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)

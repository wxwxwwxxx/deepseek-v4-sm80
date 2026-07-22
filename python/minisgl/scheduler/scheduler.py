from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req, RequestLifecycle, RequestLifecycleState
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
from .phase_policy import MixedPhaseFairPolicy
from .prefill import ChunkedReq, PrefillManager
from .stats import SchedulerStatsTracker
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _processed_prompt_tokens(batch: Batch) -> int:
    """Count this prefill forward after request lengths have advanced."""
    assert batch.is_prefill
    return int(batch.input_ids.numel())


# For overlap scheduling, we also need to cache some other data to avoid IMA
class IssuedReqRef(NamedTuple):
    req: Req
    generation_id: int
    issue_epoch: int


class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)
    issued_refs: Tuple[IssuedReqRef, ...]


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
            f"input sequence length {input_len} exceeds effective max sequence length {max_seq_len}"
        )
        return MaxSequenceAdmission(input_len, requested_output_len, 0, max_seq_len, False, reason)
    available_output_len = max_seq_len - input_len
    if requested_output_len > 0 and available_output_len == 0:
        reason = (
            f"input sequence length {input_len} equals effective max sequence length "
            f"{max_seq_len}; no model position remains for generation"
        )
        return MaxSequenceAdmission(input_len, requested_output_len, 0, max_seq_len, False, reason)
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

    if getattr(config, "enable_dsv4_swa_independent_lifecycle", False):
        if not config.enable_dsv4_radix_prefix_cache:
            raise ValueError(
                "DeepSeek V4 SWA independent lifecycle requires --enable-dsv4-radix-prefix-cache."
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
        if window_size > config.page_size and not getattr(
            config, "enable_dsv4_swa_independent_lifecycle", False
        ):
            raise ValueError(
                "DeepSeek V4 component loc ownership currently keeps one "
                "page-aligned SWA/full tail per retained node, so window_size "
                f"must be <= page_size. Got window_size={window_size}, "
                f"page_size={config.page_size}."
            )
    if not config.enable_dsv4_radix_prefix_cache:
        raise ValueError("The DeepSeek V4 release requires radix prefix caching.")
    if config.page_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 radix prefix cache requires a page size divisible "
            f"by 128, got page_size={config.page_size}. Use --page-size 256."
        )
    if cache_type != "radix":
        raise ValueError(
            "The DeepSeek V4 release requires "
            f"cache_type='radix', got {cache_type!r}."
        )
    return cache_type


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)
        # Engine resolves the immutable DSV4 release cache/lifecycle ownership
        # fields before the sole radix prefix-cache implementation is created.
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
        self._next_request_generation = 0
        self._live_reqs: dict[int, Req] = {}
        self._retirement_owners: dict[int, Req] = {}
        self.tokenizer = load_tokenizer(config.model_path)
        self.reasoning_sampler_contract_enabled = self.engine.reasoning_sampler_contract_enabled
        if self.engine.reasoning_token_ids is None:
            self.eos_token_id = self.tokenizer.eos_token_id
            self.think_end_token_id = None
        else:
            self.eos_token_id = self.engine.reasoning_token_ids.eos
            self.think_end_token_id = self.engine.reasoning_token_ids.think_end
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        self.phase_policy = MixedPhaseFairPolicy(
            isolated_prefill_budget=self.prefill_budget,
        )
        self._stats_tracker = (
            None if config.disable_log_stats else SchedulerStatsTracker(config.stats_log_interval)
        )

        # self.config = config

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        if self._stats_tracker is not None:
            self._stats_tracker.reset_idle()
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
        self._maybe_log_stats()
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
        self._maybe_log_stats()

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

        forward_input, (_, next_tokens_cpu, copy_done) = last_data
        batch = forward_input.batch
        if len(forward_input.issued_refs) != len(batch.reqs):
            raise RuntimeError("issued request references do not match forward batch rows")
        copy_done.synchronize()
        self.engine.release_copy_done_event(copy_done)
        reply: List[DetokenizeMsg] = []
        accepted_generation_tokens = 0
        with self.cache_manager.lazy_free_region():
            for i, (req, issued_ref) in enumerate(
                zip(batch.reqs, forward_input.issued_refs, strict=True)
            ):
                if req is not issued_ref.req:
                    raise RuntimeError("issued request identity does not match batch row")
                lifecycle = req.lifecycle
                if issued_ref.generation_id != lifecycle.generation_id:
                    raise RuntimeError("issued request generation changed before completion")
                if not lifecycle.accepts_completion:
                    self._complete_issued_ref(issued_ref)
                    continue
                if isinstance(req, ChunkedReq):
                    self._complete_issued_ref(issued_ref)
                    self.cache_manager.release_active_dsv4_swa_out_of_window(req)
                    continue
                next_token = next_tokens_cpu[i]
                if not req.can_commit_token:
                    raise RuntimeError(
                        f"active request {req.uid} has no remaining host token capacity"
                    )
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                if self.reasoning_sampler_contract_enabled:
                    assert self.think_end_token_id is not None
                    req.observe_generated_token(
                        next_token,
                        think_end_token_id=self.think_end_token_id,
                    )
                finished = not req.can_commit_token
                reached_eos = False
                if not req.sampling_params.ignore_eos:
                    reached_eos = next_token == self.eos_token_id
                    finished |= reached_eos
                prompt_tokens = None
                completion_tokens = None
                if finished:
                    prompt_tokens = req.max_device_len - req.output_len
                    completion_tokens = len(req.input_ids) - prompt_tokens
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=(
                            "stop" if finished and reached_eos else "length" if finished else None
                        ),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )

                if finished:
                    finish_reason = "stop" if reached_eos else "length"
                    self._commit_terminal(req, finish_reason)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)
                else:
                    self.cache_manager.release_active_dsv4_swa_out_of_window(req)
                self._complete_issued_ref(issued_ref)
                if batch.is_decode:
                    accepted_generation_tokens += 1

        if self._stats_tracker is not None:
            if batch.is_prefill:
                self._stats_tracker.record(prompt_tokens=_processed_prompt_tokens(batch))
            elif accepted_generation_tokens:
                self._stats_tracker.record(generation_tokens=accepted_generation_tokens)
        if reply:
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
                logger.warning_rank0(f"Request {msg.uid} rejected: {reason}.")
                self._record_max_sequence_admission(msg.uid, admission)
                self.send_result(
                    [
                        DetokenizeMsg(
                            uid=msg.uid,
                            next_token=self.eos_token_id,
                            finished=True,
                            finish_reason="length_rejected",
                            error=reason,
                            prompt_tokens=len(msg.input_ids),
                            completion_tokens=0,
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
            generation_id = self._next_request_generation
            self._next_request_generation += 1
            self.prefill_manager.add_one_req(msg, generation_id=generation_id)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self._live_reqs.get(msg.uid)
            if req_to_free is not None:
                if req_to_free.lifecycle.state is RequestLifecycleState.ACTIVE:
                    self._commit_terminal(req_to_free, "abort")
                self._maybe_release_terminal_resources(req_to_free.lifecycle)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _record_max_sequence_admission(self, uid: int, admission: MaxSequenceAdmission) -> None:
        """Offline frontends override this to make clamps/rejections observable."""
        del uid, admission

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _commit_terminal(self, req: Req, finish_reason: str) -> None:
        lifecycle = req.lifecycle
        lifecycle.commit_terminal(finish_reason)
        self.decode_manager.remove_req(req)
        self._retirement_owners[id(lifecycle)] = req

    def _complete_issued_ref(self, issued_ref: IssuedReqRef) -> None:
        lifecycle = issued_ref.req.lifecycle
        lifecycle.complete(
            generation_id=issued_ref.generation_id,
            issue_epoch=issued_ref.issue_epoch,
        )
        self._maybe_release_terminal_resources(lifecycle)

    def _maybe_release_terminal_resources(self, lifecycle: RequestLifecycle) -> None:
        if not lifecycle.ready_to_release_resources:
            return
        owner = self._retirement_owners.pop(id(lifecycle), None)
        if owner is None:
            raise RuntimeError("terminal request has no resource-retirement owner")
        self._free_req_resources(owner)
        lifecycle.mark_resources_released()
        live = self._live_reqs.get(owner.uid)
        if live is not None and live.lifecycle is lifecycle:
            self._live_reqs.pop(owner.uid)

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

        issued_refs = self._issue_batch_refs(batch)
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
            issued_refs=issued_refs,
        )

    def _issue_batch_refs(self, batch: Batch) -> Tuple[IssuedReqRef, ...]:
        refs: list[IssuedReqRef] = []
        for req in batch.reqs:
            lifecycle = req.lifecycle
            if lifecycle.generation_id < 0:
                lifecycle.generation_id = self._next_request_generation
                self._next_request_generation += 1
            existing = self._live_reqs.get(req.uid)
            if (
                existing is not None
                and existing.lifecycle is not lifecycle
                and existing.lifecycle.state is not RequestLifecycleState.RETIRED
            ):
                raise RuntimeError(
                    f"request uid {req.uid} was reused before its prior generation retired"
                )
            self._live_reqs[req.uid] = req
            issue_epoch = lifecycle.issue()
            refs.append(IssuedReqRef(req, lifecycle.generation_id, issue_epoch))
        return tuple(refs)

    def _schedule_next_batch(self) -> ForwardInput | None:
        prefill_runnable = self.prefill_manager.runnable
        decode_runnable = self.decode_manager.runnable
        decision = self.phase_policy.choose(
            prefill_runnable=prefill_runnable,
            decode_runnable=decode_runnable,
        )
        batch = None
        if decision.phase == "prefill":
            batch = self.prefill_manager.schedule_next_batch(decision.prefill_budget)
            # A pending request can be temporarily inadmissible.  Do not idle
            # runnable decode or consume a prefill fairness opportunity.
            if batch is None and decode_runnable:
                batch = self.decode_manager.schedule_next_batch()
        elif decision.phase == "decode":
            batch = self.decode_manager.schedule_next_batch()
            if batch is None and prefill_runnable:
                budget = (
                    self.phase_policy.mixed_prefill_budget
                    if decode_runnable
                    else self.prefill_budget
                )
                batch = self.prefill_manager.schedule_next_batch(budget)
        self.phase_policy.record_scheduled(
            batch.phase if batch is not None else None,
            prefill_runnable=prefill_runnable,
            decode_runnable=decode_runnable,
        )
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch = forward_input.batch
        sample_args = forward_input.sample_args
        input_mapping = forward_input.input_tuple
        output_mapping = forward_input.write_tuple
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def _maybe_log_stats(self) -> None:
        if self._stats_tracker is None:
            return
        snapshot = self._stats_tracker.maybe_snapshot()
        if snapshot is None:
            return
        total_pages = int(self.cache_manager.num_pages)
        # len(tensor) reads shape metadata only; this path never synchronizes CUDA.
        free_pages = len(self.cache_manager.free_slots)
        used_pages = max(total_pages - free_pages, 0)
        kv_usage = 0.0 if total_pages == 0 else 100.0 * used_pages / total_pages
        logger.info_rank0(
            "Engine stats: "
            f"P={snapshot.prompt_throughput:.1f} tok/s, "
            f"D={snapshot.generation_throughput:.1f} tok/s, "
            f"running={len(self.decode_manager.running_reqs)}, "
            f"waiting={len(self.prefill_manager.pending_list)}, "
            f"KV={kv_usage:.1f}% ({used_pages}/{total_pages} pages)."
        )


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

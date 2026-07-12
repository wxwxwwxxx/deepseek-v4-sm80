from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import DSV4RuntimeMode
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.scheduler.scheduler import MaxSequenceAdmission


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]
    requested_output_len: int
    admitted_output_len: int | None = None
    finish_reason: str | None = None
    error: str | None = None


class LLM(Scheduler):
    def __init__(
        self,
        model_path: str,
        tp_info: DistributedInfo | None = None,
        dsv4_runtime_mode: DSV4RuntimeMode = "optimized",
        **kwargs,
    ):
        if tp_info is None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            tp_info = DistributedInfo(rank, world_size)
            if world_size > 1:
                kwargs.setdefault("distributed_init_method", "env://")
        kwargs.setdefault("disable_log_stats", True)
        kwargs.setdefault("max_extend_tokens_explicit", "max_extend_tokens" in kwargs)
        kwargs.setdefault("max_running_req_explicit", "max_running_req" in kwargs)
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=tp_info,
            dsv4_runtime_mode=dsv4_runtime_mode,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
                requested_output_len=sampling_params.max_tokens,
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)
            if msg.finished:
                status.finish_reason = msg.finish_reason or "stop"
                status.error = msg.error

    def _record_max_sequence_admission(self, uid: int, admission: MaxSequenceAdmission) -> None:
        status = self.status_map[uid]
        status.admitted_output_len = admission.admitted_output_len
        if not admission.accepted:
            status.finish_reason = "length_rejected"
            status.error = admission.rejection_reason

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, Any]]:
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, Any]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append(
                {
                    "text": output_text,
                    "token_ids": status.output_ids,
                    "finish_reason": status.finish_reason,
                    "error": status.error,
                    "requested_output_len": status.requested_output_len,
                    "admitted_output_len": status.admitted_output_len,
                }
            )
        return results

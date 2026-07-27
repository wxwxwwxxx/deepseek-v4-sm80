from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import RequestLifecycle

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    reasoning_effort: str | None = None
    lifecycle: RequestLifecycle = field(default_factory=RequestLifecycle)
    chunked_req: ChunkedReq | None = None
    # Host-only fairness ledger. Prefix hits are deliberately excluded: this
    # counts only tokens granted by the scheduler for this request generation.
    prefill_scheduled_tokens: int = 0

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]

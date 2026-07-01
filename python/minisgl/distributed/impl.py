from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]
    _stats: ClassVar[dict[tuple[str, str, str, tuple[int, ...], tuple[int, ...]], dict[str, Any]]] = {}

    def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
        self._record("all_reduce", x, x.shape, label)
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
        output_shape = list(x.shape)
        output_shape[0] *= _world_size()
        self._record("all_gather", x, output_shape, label)
        return self.plugins[-1].all_gather(x)

    @classmethod
    def reset_stats(cls) -> None:
        cls._stats = {}

    @classmethod
    def snapshot_stats(cls) -> dict[str, Any]:
        entries = []
        by_label: dict[str, dict[str, Any]] = {}
        by_op: dict[str, dict[str, Any]] = {}
        for record in sorted(
            cls._stats.values(),
            key=lambda item: (item["label"], item["op"], item["dtype"], item["shape"]),
        ):
            entry = dict(record)
            entry["shape"] = list(entry["shape"])
            entry["output_shape"] = list(entry["output_shape"])
            entries.append(entry)
            _accumulate_comm_summary(by_label, entry["label"], entry)
            _accumulate_comm_summary(by_op, entry["op"], entry)
        return {
            "total_count": int(sum(entry["count"] for entry in entries)),
            "total_bytes": int(sum(entry["bytes"] for entry in entries)),
            "entries": entries,
            "by_label": dict(sorted(by_label.items())),
            "by_op": dict(sorted(by_op.items())),
        }

    @classmethod
    def _record(
        cls,
        op: str,
        x: torch.Tensor,
        output_shape: torch.Size | list[int] | tuple[int, ...],
        label: str | None,
    ) -> None:
        normalized_label = label or "unlabeled"
        shape = tuple(int(dim) for dim in x.shape)
        normalized_output_shape = tuple(int(dim) for dim in output_shape)
        dtype = str(x.dtype).removeprefix("torch.")
        input_bytes = int(x.numel() * x.element_size())
        output_numel = 1
        for dim in normalized_output_shape:
            output_numel *= dim
        output_bytes = int(output_numel * x.element_size())
        bytes_value = output_bytes if op == "all_gather" else input_bytes
        key = (normalized_label, op, dtype, shape, normalized_output_shape)
        record = cls._stats.get(key)
        if record is None:
            record = {
                "label": normalized_label,
                "op": op,
                "dtype": dtype,
                "shape": shape,
                "output_shape": normalized_output_shape,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "bytes": 0,
                "count": 0,
            }
            cls._stats[key] = record
        record["count"] += 1
        record["bytes"] += bytes_value


def _world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _accumulate_comm_summary(
    summaries: dict[str, dict[str, Any]],
    key: str,
    entry: dict[str, Any],
) -> None:
    summary = summaries.setdefault(key, {"count": 0, "bytes": 0})
    summary["count"] += int(entry["count"])
    summary["bytes"] += int(entry["bytes"])


def reset_communication_stats() -> None:
    DistributedCommunicator.reset_stats()


def snapshot_communication_stats() -> dict[str, Any]:
    return DistributedCommunicator.snapshot_stats()


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []

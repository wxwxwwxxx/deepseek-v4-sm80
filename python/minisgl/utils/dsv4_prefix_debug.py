from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ENV_DEBUG_DIR = "MINISGL_DSV4_PREFIX_DEBUG_DIR"
ENV_DEBUG_STAGE = "MINISGL_DSV4_PREFIX_DEBUG_STAGE"
ENV_DEBUG_SCENARIO = "MINISGL_DSV4_PREFIX_DEBUG_SCENARIO"
ENV_DEBUG_MODE = "MINISGL_DSV4_PREFIX_DEBUG_MODE"
ENV_DEBUG_TOPK = "MINISGL_DSV4_PREFIX_DEBUG_TOPK"
ENV_DEBUG_SAVE_FULL_LOGITS = "MINISGL_DSV4_PREFIX_DEBUG_SAVE_FULL_LOGITS"
ENV_DEBUG_MAX_BATCHES = "MINISGL_DSV4_PREFIX_DEBUG_MAX_BATCHES"
ENV_DEBUG_ALL_RANKS = "MINISGL_DSV4_PREFIX_DEBUG_ALL_RANKS"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_RECORDER: DSV4PrefixDebugRecorder | None = None


@dataclass
class DSV4PrefixDebugSnapshot:
    payload: dict[str, Any]


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _rank_info() -> tuple[int, int, bool]:
    try:
        from minisgl.distributed import try_get_tp_info

        info = try_get_tp_info()
        if info is not None:
            return int(info.rank), int(info.size), bool(info.is_primary())
    except Exception:
        pass
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world, rank == 0


def get_dsv4_prefix_debug_recorder() -> DSV4PrefixDebugRecorder | None:
    global _RECORDER
    debug_dir = os.environ.get(ENV_DEBUG_DIR)
    if not debug_dir:
        return None
    if _RECORDER is None:
        rank, world_size, is_primary = _rank_info()
        if not is_primary and not _truthy(os.environ.get(ENV_DEBUG_ALL_RANKS)):
            return None
        _RECORDER = DSV4PrefixDebugRecorder(
            root=Path(debug_dir),
            rank=rank,
            world_size=world_size,
            topk=max(1, _env_int(ENV_DEBUG_TOPK, 10)),
            save_full_logits=_truthy(os.environ.get(ENV_DEBUG_SAVE_FULL_LOGITS), default=True),
            max_batches=_env_int(ENV_DEBUG_MAX_BATCHES, 0),
        )
    return _RECORDER


class DSV4PrefixDebugRecorder:
    def __init__(
        self,
        *,
        root: Path,
        rank: int,
        world_size: int,
        topk: int,
        save_full_logits: bool,
        max_batches: int,
    ) -> None:
        self.root = root
        self.rank = rank
        self.world_size = world_size
        self.topk = topk
        self.save_full_logits = save_full_logits
        self.max_batches = max_batches
        self._counter = 0
        self.batch_dir = root / "batches"
        self.metadata_dir = root / "metadata"
        self.logits_dir = root / "logits"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.logits_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "rank": rank,
            "world_size": world_size,
            "topk": topk,
            "save_full_logits": save_full_logits,
            "max_batches": max_batches,
            "env": {
                ENV_DEBUG_DIR: str(root),
                ENV_DEBUG_TOPK: str(topk),
                ENV_DEBUG_SAVE_FULL_LOGITS: str(int(save_full_logits)),
            },
        }
        self._write_json(root / f"manifest.rank{rank}.json", manifest)

    def capture_pre_sample(
        self,
        *,
        batch: Any,
        logits: torch.Tensor | None,
        forward_source: str,
    ) -> DSV4PrefixDebugSnapshot | None:
        if self.max_batches > 0 and self._counter >= self.max_batches:
            return None

        batch_index = self._counter
        self._counter += 1
        stem = f"rank{self.rank}.batch{batch_index:06d}"

        metadata_tensors = self._metadata_tensors(batch)
        metadata_path = self.metadata_dir / f"{stem}.metadata.pt"
        torch.save(metadata_tensors, metadata_path)

        logits_summary: dict[str, Any] | None = None
        logits_path: Path | None = None
        if logits is not None:
            real_logits = logits[: batch.size].detach().float().cpu()
            if self.save_full_logits:
                logits_path = self.logits_dir / f"{stem}.logits.pt"
                torch.save({"logits": real_logits}, logits_path)
            logits_summary = self._logits_summary(real_logits)

        payload = {
            "batch_index": batch_index,
            "rank": self.rank,
            "world_size": self.world_size,
            "mode": os.environ.get(ENV_DEBUG_MODE, ""),
            "scenario": os.environ.get(ENV_DEBUG_SCENARIO, ""),
            "stage": os.environ.get(ENV_DEBUG_STAGE, ""),
            "phase": batch.phase,
            "forward_source": forward_source,
            "batch_size": int(batch.size),
            "padded_size": int(getattr(batch, "padded_size", batch.size)),
            "metadata_path": str(metadata_path.relative_to(self.root)),
            "logits_path": str(logits_path.relative_to(self.root)) if logits_path else None,
            "reqs": self._req_snapshots(batch),
            "padded_reqs": self._padded_req_snapshots(batch),
            "batch_tensors": self._batch_tensor_summaries(batch),
            "metadata": self._metadata_summary(metadata_tensors),
            "logits": logits_summary,
        }
        return DSV4PrefixDebugSnapshot(payload=payload)

    def finish(
        self,
        snapshot: DSV4PrefixDebugSnapshot | None,
        *,
        next_tokens: torch.Tensor,
        graph_runner: dict[str, Any] | None = None,
    ) -> None:
        if snapshot is None:
            return
        payload = snapshot.payload
        payload["sampled_token_ids"] = [
            int(x) for x in next_tokens[: payload["batch_size"]].detach().cpu().tolist()
        ]
        payload["graph_runner"] = graph_runner or {}
        stem = f"rank{self.rank}.batch{payload['batch_index']:06d}"
        batch_path = self.batch_dir / f"{stem}.json"
        payload["batch_path"] = str(batch_path.relative_to(self.root))
        self._write_json(batch_path, payload)
        with (self.root / f"batches.rank{self.rank}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def _metadata_tensors(self, batch: Any) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for name, value in self._batch_tensors(batch).items():
            tensors[f"batch.{name}"] = value.detach().cpu()

        attn_metadata = getattr(batch, "attn_metadata", None)
        core = getattr(attn_metadata, "core_metadata", None)
        if core is not None:
            for name in (
                "raw_out_loc",
                "page_table",
                "cu_seqlens_q",
                "seq_lens",
                "req_seq_lens",
                "extend_lens",
                "positions",
                "req_table_indices",
                "swa_page_indices",
                "swa_topk_lengths",
                "c4_out_loc",
                "c128_out_loc",
                "c4_indexer_out_loc",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_raw_indices",
                "c4_sparse_page_indices",
                "c4_sparse_full_indices",
                "c128_topk_lengths_clamp1",
                "c128_raw_indices",
                "c128_page_indices",
                "c128_full_indices",
            ):
                value = getattr(core, name, None)
                if isinstance(value, torch.Tensor):
                    tensors[f"core.{name}"] = value.detach().cpu()

        indexer = getattr(attn_metadata, "indexer_metadata", None)
        if indexer is not None:
            for name in ("page_table", "c4_seq_lens"):
                value = getattr(indexer, name, None)
                if isinstance(value, torch.Tensor):
                    tensors[f"indexer.{name}"] = value.detach().cpu()

        for prefix, compress in (
            ("c4_compress", getattr(attn_metadata, "c4_compress_metadata", None)),
            ("c128_compress", getattr(attn_metadata, "c128_compress_metadata", None)),
        ):
            if compress is None:
                continue
            for name in ("write_loc", "seq_lens", "positions"):
                value = getattr(compress, name, None)
                if isinstance(value, torch.Tensor):
                    tensors[f"{prefix}.{name}"] = value.detach().cpu()

        return tensors

    def _batch_tensors(self, batch: Any) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for name in ("input_ids", "positions", "out_loc"):
            value = getattr(batch, name, None)
            if isinstance(value, torch.Tensor):
                tensors[name] = value

        try:
            from minisgl.core import get_global_ctx

            page_table = get_global_ctx().page_table
            reqs = list(getattr(batch, "reqs", []))
            max_len = max((int(req.device_len) for req in reqs), default=0)
            rows = torch.full(
                (len(reqs), max(max_len, 1)),
                -1,
                dtype=torch.int32,
                device=page_table.device,
            )
            for row, req in enumerate(reqs):
                length = int(req.device_len)
                if length > 0:
                    rows[row, :length] = page_table[int(req.table_idx), :length]
            tensors["global_page_table_rows"] = rows
        except Exception:
            pass

        return tensors

    def _req_snapshots(self, batch: Any) -> list[dict[str, Any]]:
        return [self._req_snapshot(req) for req in getattr(batch, "reqs", [])]

    def _padded_req_snapshots(self, batch: Any) -> list[dict[str, Any]]:
        return [self._req_snapshot(req) for req in getattr(batch, "padded_reqs", [])]

    def _req_snapshot(self, req: Any) -> dict[str, Any]:
        cached_len = int(getattr(req, "cached_len", 0))
        device_len = int(getattr(req, "device_len", 0))
        extend_len = int(getattr(req, "extend_len", max(device_len - cached_len, 0)))
        return {
            "uid": int(getattr(req, "uid", -1)),
            "table_idx": int(getattr(req, "table_idx", -1)),
            "cached_len": cached_len,
            "device_len": device_len,
            "extend_len": extend_len,
            "max_device_len": int(getattr(req, "max_device_len", device_len)),
            "remain_len": int(getattr(req, "remain_len", 0)),
            "suffix_range": [cached_len, device_len],
            "is_chunked": type(req).__name__ == "ChunkedReq",
        }

    def _batch_tensor_summaries(self, batch: Any) -> dict[str, Any]:
        return {name: self._tensor_summary(value) for name, value in self._batch_tensors(batch).items()}

    def _metadata_summary(self, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
        return {name: self._tensor_summary(value) for name, value in tensors.items()}

    def _logits_summary(self, logits: torch.Tensor) -> dict[str, Any]:
        if logits.numel() == 0:
            return {"shape": list(logits.shape), "dtype": str(logits.dtype), "topk": []}
        k = min(self.topk, logits.shape[-1])
        values, indices = torch.topk(logits, k=k, dim=-1)
        return {
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "topk": [
                {
                    "token_ids": [int(x) for x in indices[row].tolist()],
                    "values": [float(x) for x in values[row].tolist()],
                }
                for row in range(logits.shape[0])
            ],
        }

    def _tensor_summary(self, tensor: torch.Tensor) -> dict[str, Any]:
        cpu = tensor.detach().cpu()
        flat = cpu.reshape(-1)
        head = flat[: min(12, flat.numel())].tolist()
        tail = flat[-min(12, flat.numel()) :].tolist() if flat.numel() else []
        summary: dict[str, Any] = {
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "numel": int(cpu.numel()),
            "head": _jsonable_list(head),
            "tail": _jsonable_list(tail),
        }
        if cpu.numel() > 0 and not cpu.is_complex():
            if cpu.dtype.is_floating_point:
                finite = cpu[torch.isfinite(cpu)]
                if finite.numel() > 0:
                    summary["min"] = float(finite.min().item())
                    summary["max"] = float(finite.max().item())
            else:
                summary["min"] = int(cpu.min().item())
                summary["max"] = int(cpu.max().item())
        if cpu.ndim >= 2 and cpu.shape[0] > 0:
            rows = min(4, cpu.shape[0])
            summary["row_heads"] = [
                _jsonable_list(cpu[row].reshape(-1)[: min(12, cpu[row].numel())].tolist())
                for row in range(rows)
            ]
            summary["row_tails"] = [
                _jsonable_list(cpu[row].reshape(-1)[-min(12, cpu[row].numel()) :].tolist())
                for row in range(rows)
            ]
        return summary

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def _jsonable_list(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if isinstance(value, bool):
            out.append(value)
        elif isinstance(value, int):
            out.append(int(value))
        elif isinstance(value, float):
            out.append(float(value))
        elif hasattr(value, "item"):
            item = value.item()
            if isinstance(item, bool):
                out.append(bool(item))
            elif isinstance(item, int):
                out.append(int(item))
            else:
                out.append(float(item))
        else:
            out.append(value)
    return out


__all__ = [
    "ENV_DEBUG_DIR",
    "ENV_DEBUG_MODE",
    "ENV_DEBUG_SCENARIO",
    "ENV_DEBUG_STAGE",
    "get_dsv4_prefix_debug_recorder",
]

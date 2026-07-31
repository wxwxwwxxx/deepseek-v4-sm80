#!/usr/bin/env python3
"""Metadata/no-weight attribution for TARGET 19 chunk-boundary stalls.

The harness builds production-width DSV4 prefill metadata with a two-layer
cache (one C4 and one C128 layer).  It deliberately avoids model weights while
retaining the release page size, total prefill budget, sparse widths, device
metadata ABI, and resident-request shapes.

The "candidate" is installed only on the harness-owned backend/cache objects.
It models the bounded production change under consideration:

* construct sparse C4 rows without ``Tensor.tolist()``;
* derive compressor-boundary rows from host-authoritative request lengths;
* gather component/SWA page mappings without CUDA boolean truth tests.

No runtime source is modified by this script.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import minisgl.core as core  # noqa: E402
import minisgl.distributed.info as dist_info  # noqa: E402
from minisgl.attention.deepseek_v4 import DSV4AttentionBackend, _pad_last_dim  # noqa: E402
from minisgl.core import Batch, Context, Req, SamplingParams  # noqa: E402
from minisgl.distributed import set_tp_info  # noqa: E402
from minisgl.kvcache import create_kvcache_pool  # noqa: E402
from minisgl.models.config import ModelConfig  # noqa: E402
from minisgl.utils import cached_load_hf_config, div_ceil  # noqa: E402

PAGE_SIZE = 256
PREFILL_BUDGET = 8192
MAX_CONTEXT = 524288
MAX_RESIDENT = 8


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p10_ms": _percentile(values, 0.10),
        "p90_ms": _percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def _tiny_production_config(model_path: str) -> ModelConfig:
    full = ModelConfig.from_hf(cached_load_hf_config(model_path))
    return dataclasses.replace(
        full,
        num_layers=2,
        compress_ratios=[4, 128],
    )


def _install_context(model_path: str) -> tuple[Context, DSV4AttentionBackend]:
    cfg = _tiny_production_config(model_path)
    device = torch.device("cuda:0")
    pages_per_row = div_ceil(MAX_CONTEXT, PAGE_SIZE)
    num_pages = MAX_RESIDENT * pages_per_row + 1
    ctx = Context(page_size=PAGE_SIZE)
    ctx.kv_cache = create_kvcache_pool(
        cfg,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        device=device,
        max_running_req=MAX_RESIDENT,
        dsv4_swa_num_pages=MAX_RESIDENT * 2 + 1,
        dsv4_dummy_token_start=(num_pages - 1) * PAGE_SIZE,
    )

    # Metadata only: install deterministic physical ownership without touching
    # the large payload buffers.  Component pages preserve the one-to-one
    # Route-B page mapping.  SWA pages may be reused outside a live window in
    # this harness because no attention kernel consumes the synthetic cache.
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    ctx.kv_cache._full_to_c4_page.copy_(page_ids)  # type: ignore[attr-defined]
    ctx.kv_cache._full_to_c128_page.copy_(page_ids)  # type: ignore[attr-defined]
    ctx.kv_cache._full_to_c4_indexer_page.copy_(page_ids)  # type: ignore[attr-defined]
    live_swa_pages = max(int(ctx.kv_cache._swa_dummy_page), 1)  # type: ignore[attr-defined]
    ctx.kv_cache._full_to_swa_page.copy_(page_ids.remainder(live_swa_pages))  # type: ignore[attr-defined]

    row_width = MAX_CONTEXT
    page_table = torch.empty(
        (MAX_RESIDENT + 1, row_width),
        dtype=torch.int32,
        device=device,
    )
    logical = torch.arange(row_width, dtype=torch.int32, device=device)
    for row in range(MAX_RESIDENT):
        page_table[row].copy_(logical + row * MAX_CONTEXT)
    page_table[MAX_RESIDENT].fill_((num_pages - 1) * PAGE_SIZE)
    ctx.page_table = page_table
    core.set_global_ctx(ctx)
    ctx.attn_backend = DSV4AttentionBackend(cfg)
    return ctx, ctx.attn_backend


def _make_batch(ctx: Context, *, context: int, resident: int) -> Batch:
    if PREFILL_BUDGET % resident:
        raise ValueError("resident count must divide the fixed prefill budget")
    rows_per_req = PREFILL_BUDGET // resident
    if context < rows_per_req:
        raise ValueError("context checkpoint is smaller than one request's chunk")
    shared_ids = torch.zeros(context, dtype=torch.int32)
    reqs: list[Req] = []
    for row in range(resident):
        reqs.append(
            Req(
                input_ids=shared_ids,
                table_idx=row,
                cached_len=context - rows_per_req,
                output_len=8,
                uid=row,
                sampling_params=SamplingParams(max_tokens=8),
                cache_handle=None,  # type: ignore[arg-type]
            )
        )
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    positions = torch.cat(
        [
            torch.arange(
                req.cached_len,
                req.device_len,
                dtype=torch.int32,
                device="cuda",
            )
            for req in reqs
        ]
    )
    table_indices = torch.cat(
        [
            torch.full(
                (req.extend_len,),
                req.table_idx,
                dtype=torch.long,
                device="cuda",
            )
            for req in reqs
        ]
    )
    batch.positions = positions
    batch.out_loc = ctx.page_table[table_indices, positions.to(torch.long)]
    batch.input_ids = torch.zeros(PREFILL_BUDGET, dtype=torch.int32, device="cuda")
    return batch


def _candidate_sparse_compressed_indices(
    self: DSV4AttentionBackend,
    table_indices: torch.Tensor,
    lengths: torch.Tensor,
    ratio: int,
    *,
    component_page_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(self.index_topk, 1)
    offsets = torch.arange(width, dtype=torch.int32, device=self.device)
    counts = lengths.clamp(min=0, max=width)
    starts = (lengths - width).clamp_min(0)
    raw_values = starts[:, None] + offsets[None, :]
    raw = torch.where(
        offsets[None, :] < counts[:, None],
        raw_values,
        torch.full_like(raw_values, -1),
    )
    full = self._compressed_raw_to_full_locs(table_indices, raw, ratio)
    if component_page_table is None:
        page = torch.where(full >= 0, full.div(ratio, rounding_mode="floor"), full)
    else:
        page = self._compressed_raw_to_component_locs(
            component_page_table,
            raw,
            ratio,
        )
    return (
        _pad_last_dim(raw, value=-1),
        _pad_last_dim(page.to(torch.int32), value=-1),
        _pad_last_dim(full.to(torch.int32), value=-1),
    )


def _candidate_component_write_locs(
    self: DSV4AttentionBackend,
    component_page_table: torch.Tensor | None,
    positions: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    if component_page_table is None or positions.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=self.device)
    reqs = getattr(self, "_target19_host_reqs")
    component_page_size = (
        self.kvcache.c4_component_page_size
        if ratio == 4
        else self.kvcache.c128_component_page_size
    )
    boundary_positions: list[torch.Tensor] = []
    source_rows: list[torch.Tensor] = []
    row_offset = 0
    for req in reqs:
        first = int(req.cached_len) + ((ratio - 1 - int(req.cached_len)) % ratio)
        count = max((int(req.device_len) - 1 - first) // ratio + 1, 0)
        if count:
            steps = torch.arange(count, dtype=torch.long, device=self.device)
            boundary_positions.append(first + steps * ratio)
            source_rows.append(row_offset + (first - int(req.cached_len)) + steps * ratio)
        row_offset += int(req.extend_len)
    if not boundary_positions:
        return torch.empty(0, dtype=torch.long, device=self.device)
    boundary_tensor = torch.cat(boundary_positions)
    source_tensor = torch.cat(source_rows)
    raw = boundary_tensor.div(ratio, rounding_mode="floor")
    logical_pages = raw.div(component_page_size, rounding_mode="floor")
    offsets = raw % component_page_size
    max_page = max(component_page_table.shape[1] - 1, 0)
    safe_pages = logical_pages.clamp(max=max_page)
    component_pages = component_page_table[source_tensor, safe_pages].to(torch.long)
    locs = component_pages * component_page_size + offsets
    valid = (logical_pages < component_page_table.shape[1]) & (component_pages >= 0)
    return torch.where(valid, locs, torch.full_like(locs, -1))


def _candidate_component_pages_from_full_page_starts(
    self,
    page_starts: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    page_starts = page_starts.to(device=self.device, dtype=torch.long)
    full_pages = torch.where(
        page_starts >= 0,
        page_starts.div(page_size, rounding_mode="floor"),
        torch.full_like(page_starts, -1),
    )
    valid = (full_pages >= 0) & (full_pages < self._num_pages)
    safe_pages = full_pages.clamp(min=0, max=max(self._num_pages - 1, 0))

    def gather(mapping: torch.Tensor, enabled: bool) -> torch.Tensor | None:
        if not enabled:
            return None
        gathered = mapping[safe_pages].to(torch.int32)
        return torch.where(valid, gathered, torch.full_like(gathered, -1))

    return (
        gather(self._full_to_c4_page, self._c4_layer_count > 0),
        gather(self._full_to_c128_page, self._c128_layer_count > 0),
        gather(self._full_to_c4_indexer_page, self._c4_layer_count > 0),
    )


def _candidate_swa_pages_from_full_page_starts(
    self,
    page_starts: torch.Tensor,
    page_size: int,
) -> torch.Tensor | None:
    page_starts = page_starts.to(device=self.device, dtype=torch.long)
    full_pages = torch.where(
        page_starts >= 0,
        page_starts.div(page_size, rounding_mode="floor"),
        torch.full_like(page_starts, -1),
    )
    valid = (full_pages >= 0) & (full_pages < self._num_pages)
    safe_pages = full_pages.clamp(min=0, max=max(self._num_pages - 1, 0))
    gathered = self._full_to_swa_page[safe_pages].to(torch.int32)
    out = torch.where(valid, gathered, torch.full_like(gathered, -1))
    dummy = page_starts == self._dummy_token_start
    return torch.where(dummy, int(self._swa_dummy_page), out)


class CandidatePatch:
    def __init__(self, backend: DSV4AttentionBackend):
        self.backend = backend
        self.kvcache = backend.kvcache
        self.saved = {
            "sparse": backend._make_sparse_compressed_indices,
            "write": backend._component_write_locs_from_page_table,
            "component_pages": self.kvcache.component_pages_from_full_page_starts,
            "swa_pages": self.kvcache.swa_pages_from_full_page_starts,
        }

    def install(self, reqs: Iterable[Req]) -> None:
        self.backend._target19_host_reqs = tuple(reqs)  # type: ignore[attr-defined]
        self.backend._make_sparse_compressed_indices = types.MethodType(  # type: ignore[method-assign]
            _candidate_sparse_compressed_indices,
            self.backend,
        )
        self.backend._component_write_locs_from_page_table = types.MethodType(  # type: ignore[method-assign]
            _candidate_component_write_locs,
            self.backend,
        )
        self.kvcache.component_pages_from_full_page_starts = types.MethodType(  # type: ignore[method-assign]
            _candidate_component_pages_from_full_page_starts,
            self.kvcache,
        )
        self.kvcache.swa_pages_from_full_page_starts = types.MethodType(  # type: ignore[method-assign]
            _candidate_swa_pages_from_full_page_starts,
            self.kvcache,
        )

    def restore(self) -> None:
        self.backend._make_sparse_compressed_indices = self.saved["sparse"]  # type: ignore[method-assign]
        self.backend._component_write_locs_from_page_table = self.saved["write"]  # type: ignore[method-assign]
        self.kvcache.component_pages_from_full_page_starts = self.saved[  # type: ignore[method-assign]
            "component_pages"
        ]
        self.kvcache.swa_pages_from_full_page_starts = self.saved["swa_pages"]  # type: ignore[method-assign]


def _tensor_fields(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        result[prefix] = value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            child = getattr(value, field.name)
            child_prefix = f"{prefix}.{field.name}" if prefix else field.name
            result.update(_tensor_fields(child, child_prefix))
    return result


def _compare_metadata(before: Any, after: Any) -> dict[str, Any]:
    before_fields = _tensor_fields(before)
    after_fields = _tensor_fields(after)
    paths_match = set(before_fields) == set(after_fields)
    failures: list[dict[str, Any]] = []
    compared = 0
    for path in sorted(set(before_fields) & set(after_fields)):
        lhs = before_fields[path]
        rhs = after_fields[path]
        compared += 1
        if (
            lhs.shape != rhs.shape
            or lhs.dtype != rhs.dtype
            or lhs.device != rhs.device
            or lhs.stride() != rhs.stride()
            or not torch.equal(lhs, rhs)
        ):
            failures.append(
                {
                    "path": path,
                    "before": {
                        "shape": list(lhs.shape),
                        "dtype": str(lhs.dtype),
                        "device": str(lhs.device),
                        "stride": list(lhs.stride()),
                    },
                    "after": {
                        "shape": list(rhs.shape),
                        "dtype": str(rhs.dtype),
                        "device": str(rhs.device),
                        "stride": list(rhs.stride()),
                    },
                    "values_equal": bool(
                        lhs.shape == rhs.shape
                        and lhs.dtype == rhs.dtype
                        and torch.equal(lhs, rhs)
                    ),
                }
            )
    return {
        "paths_match": paths_match,
        "tensor_fields_compared": compared,
        "failures": failures,
        "exact": paths_match and not failures,
    }


def _time_prepare(
    backend: DSV4AttentionBackend,
    batch: Batch,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        backend.prepare_metadata(batch)
    torch.cuda.synchronize()
    host_samples: list[float] = []
    gpu_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        host_start = time.perf_counter()
        backend.prepare_metadata(batch)
        host_samples.append((time.perf_counter() - host_start) * 1000.0)
        end_event.record()
        end_event.synchronize()
        wall_samples.append((time.perf_counter() - wall_start) * 1000.0)
        gpu_samples.append(float(start_event.elapsed_time(end_event)))
    return {
        "host_return": _distribution(host_samples),
        "gpu_stream": _distribution(gpu_samples),
        "wall_complete": _distribution(wall_samples),
    }


def _profile_prepare(
    backend: DSV4AttentionBackend,
    batch: Batch,
    trace_path: Path,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as profiler:
        backend.prepare_metadata(batch)
        torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    interesting: Counter[str] = Counter()
    cpu_us: defaultdict[str, float] = defaultdict(float)
    needles = (
        "item",
        "local_scalar",
        "tolist",
        "nonzero",
        "synchronize",
        "cudaMemcpy",
        "memcpy",
    )
    for event in profiler.events():
        name = str(event.name)
        if any(needle.lower() in name.lower() for needle in needles):
            interesting[name] += 1
            cpu_us[name] += float(getattr(event, "self_cpu_time_total", 0.0))
    return {
        "trace": str(trace_path),
        "interesting_events": [
            {
                "name": name,
                "count": count,
                "self_cpu_time_us": cpu_us[name],
            }
            for name, count in interesting.most_common()
        ],
    }


def _leaf_host_ledger(
    backend: DSV4AttentionBackend,
    batch: Batch,
) -> dict[str, Any]:
    owners: defaultdict[str, list[float]] = defaultdict(list)
    patches: list[tuple[object, str, Callable[..., Any]]] = []

    def wrap(owner: object, name: str, label: str) -> None:
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                owners[label].append((time.perf_counter() - started) * 1000.0)

        patches.append((owner, name, original))
        setattr(owner, name, measured)

    wrap(
        backend.kvcache,
        "component_pages_from_full_page_starts",
        "component_page_mapping",
    )
    wrap(
        backend.kvcache,
        "swa_pages_from_full_page_starts",
        "swa_page_mapping",
    )
    wrap(backend, "_make_sparse_compressed_indices", "c4_sparse_indices")
    wrap(backend, "_component_write_locs_from_page_table", "component_write_locs")
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        backend.prepare_metadata(batch)
        total_host_ms = (time.perf_counter() - started) * 1000.0
        torch.cuda.synchronize()
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
    return {
        "total_host_return_ms": total_host_ms,
        "owners": {
            label: {
                "calls": len(samples),
                "total_host_ms": sum(samples),
                "samples_ms": samples,
            }
            for label, samples in owners.items()
        },
        "note": "nested CUDA work may be charged to the first following host synchronization",
    }


def _one_case(
    ctx: Context,
    backend: DSV4AttentionBackend,
    *,
    context: int,
    resident: int,
    warmup: int,
    iters: int,
    trace_dir: Path | None,
) -> dict[str, Any]:
    batch = _make_batch(ctx, context=context, resident=resident)
    patch = CandidatePatch(backend)

    patch.restore()
    backend.prepare_metadata(batch)
    torch.cuda.synchronize()
    baseline_metadata = batch.attn_metadata
    baseline_timing = _time_prepare(backend, batch, warmup=warmup, iters=iters)
    baseline_ledger = _leaf_host_ledger(backend, batch)
    baseline_profile = None
    if trace_dir is not None and context == MAX_CONTEXT and resident == MAX_RESIDENT:
        baseline_profile = _profile_prepare(
            backend,
            batch,
            trace_dir / "baseline_m8_512k.json",
        )

    patch.install(batch.padded_reqs)
    backend.prepare_metadata(batch)
    torch.cuda.synchronize()
    candidate_metadata = batch.attn_metadata
    exactness = _compare_metadata(baseline_metadata, candidate_metadata)
    candidate_timing = _time_prepare(backend, batch, warmup=warmup, iters=iters)
    candidate_ledger = _leaf_host_ledger(backend, batch)
    candidate_profile = None
    if trace_dir is not None and context == MAX_CONTEXT and resident == MAX_RESIDENT:
        candidate_profile = _profile_prepare(
            backend,
            batch,
            trace_dir / "candidate_m8_512k.json",
        )
    patch.restore()

    baseline_host = baseline_timing["host_return"]["median_ms"]
    candidate_host = candidate_timing["host_return"]["median_ms"]
    baseline_wall = baseline_timing["wall_complete"]["median_ms"]
    candidate_wall = candidate_timing["wall_complete"]["median_ms"]
    return {
        "context_tokens_per_request": context,
        "resident_requests": resident,
        "total_prefill_forward_budget": PREFILL_BUDGET,
        "extend_tokens_per_request": PREFILL_BUDGET // resident,
        "baseline": {
            "timing": baseline_timing,
            "leaf_host_ledger": baseline_ledger,
            "profile": baseline_profile,
        },
        "candidate": {
            "timing": candidate_timing,
            "leaf_host_ledger": candidate_ledger,
            "profile": candidate_profile,
        },
        "delta": {
            "host_return_ms": candidate_host - baseline_host,
            "host_return_percent": (candidate_host / baseline_host - 1.0) * 100.0,
            "wall_complete_ms": candidate_wall - baseline_wall,
            "wall_complete_percent": (candidate_wall / baseline_wall - 1.0) * 100.0,
        },
        "metadata_exactness": exactness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--contexts", default="8192,65536,262144,524288")
    parser.add_argument("--residents", default="1,4,8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0):
        raise SystemExit("TARGET 19 metadata microbench requires CUDA sm80")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    old_ctx = core._GLOBAL_CTX
    old_tp = dist_info._TP_INFO
    core._GLOBAL_CTX = None
    dist_info._TP_INFO = None
    set_tp_info(0, 1)
    started = time.time()
    try:
        ctx, backend = _install_context(args.model_path)
        contexts = [int(value) for value in args.contexts.split(",") if value]
        residents = [int(value) for value in args.residents.split(",") if value]
        cases = [
            _one_case(
                ctx,
                backend,
                context=context,
                resident=resident,
                warmup=args.warmup,
                iters=args.iters,
                trace_dir=args.trace_dir,
            )
            for context in contexts
            for resident in residents
        ]
        output = {
            "suite": "target19_chunk_boundary_metadata_no_weight",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "contract": {
                "page_size": PAGE_SIZE,
                "total_prefill_forward_budget": PREFILL_BUDGET,
                "contexts": contexts,
                "resident_requests": residents,
                "metadata_layers": [4, 128],
                "candidate_scope": [
                    "C4 sparse construction without Tensor.tolist",
                    "host-authoritative C4/C128 compressor-boundary rows",
                    "component/SWA map gathers without CUDA bool reads",
                ],
                "production_source_modified": False,
            },
            "all_metadata_exact": all(
                case["metadata_exactness"]["exact"] for case in cases
            ),
            "cases": cases,
            "elapsed_s": time.time() - started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(output, indent=2))
    finally:
        core._GLOBAL_CTX = old_ctx
        dist_info._TP_INFO = old_tp


if __name__ == "__main__":
    main()

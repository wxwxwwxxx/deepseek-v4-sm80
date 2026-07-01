from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _copy_2d(dst: torch.Tensor, src: torch.Tensor, rows: int, *, fill: int) -> None:
    width = min(dst.shape[1], src.shape[1])
    if width > 0:
        dst[:rows, :width].copy_(src[:rows, :width])
    if width < dst.shape[1]:
        dst[:rows, width:].fill_(fill)


def _legacy_copy(state: dict[str, torch.Tensor], rows: int, *, graph_inputs_bound: bool) -> None:
    for name in (
        "raw_out_loc",
        "seq_lens",
        "req_seq_lens",
        "extend_lens",
        "positions",
        "req_table_indices",
        "swa_topk_lengths",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c4_sparse_topk_lengths",
        "c128_topk_lengths_clamp1",
    ):
        if graph_inputs_bound and name in {"raw_out_loc", "positions"}:
            continue
        state[f"dst_{name}"][:rows].copy_(state[f"src_{name}"][:rows])
    state["dst_cu_seqlens_q"][: rows + 1].copy_(state["src_cu_seqlens_q"][: rows + 1])
    _copy_2d(state["dst_page_table"], state["src_page_table"], rows, fill=0)
    for name in (
        "swa_page_indices",
        "c4_sparse_raw_indices",
        "c4_sparse_page_indices",
        "c4_sparse_full_indices",
        "c128_raw_indices",
        "c128_page_indices",
        "c128_full_indices",
    ):
        _copy_2d(state[f"dst_{name}"], state[f"src_{name}"], rows, fill=-1)


def _fused_copy(state: dict[str, torch.Tensor], rows: int, *, graph_inputs_bound: bool) -> None:
    ok = dsv4_kernel.copy_decode_metadata_for_replay(
        dst_raw_out_loc=state["dst_raw_out_loc"],
        src_raw_out_loc=state["src_raw_out_loc"],
        dst_seq_lens=state["dst_seq_lens"],
        src_seq_lens=state["src_seq_lens"],
        dst_req_seq_lens=state["dst_req_seq_lens"],
        src_req_seq_lens=state["src_req_seq_lens"],
        dst_extend_lens=state["dst_extend_lens"],
        src_extend_lens=state["src_extend_lens"],
        dst_positions=state["dst_positions"],
        src_positions=state["src_positions"],
        dst_req_table_indices=state["dst_req_table_indices"],
        src_req_table_indices=state["src_req_table_indices"],
        dst_swa_topk_lengths=state["dst_swa_topk_lengths"],
        src_swa_topk_lengths=state["src_swa_topk_lengths"],
        dst_c4_topk_lengths_raw=state["dst_c4_topk_lengths_raw"],
        src_c4_topk_lengths_raw=state["src_c4_topk_lengths_raw"],
        dst_c4_topk_lengths_clamp1=state["dst_c4_topk_lengths_clamp1"],
        src_c4_topk_lengths_clamp1=state["src_c4_topk_lengths_clamp1"],
        dst_c4_sparse_topk_lengths=state["dst_c4_sparse_topk_lengths"],
        src_c4_sparse_topk_lengths=state["src_c4_sparse_topk_lengths"],
        dst_c128_topk_lengths_clamp1=state["dst_c128_topk_lengths_clamp1"],
        src_c128_topk_lengths_clamp1=state["src_c128_topk_lengths_clamp1"],
        dst_cu_seqlens_q=state["dst_cu_seqlens_q"],
        src_cu_seqlens_q=state["src_cu_seqlens_q"],
        dst_page_table=state["dst_page_table"],
        src_page_table=state["src_page_table"],
        dst_swa_page_indices=state["dst_swa_page_indices"],
        src_swa_page_indices=state["src_swa_page_indices"],
        dst_c4_sparse_raw_indices=state["dst_c4_sparse_raw_indices"],
        src_c4_sparse_raw_indices=state["src_c4_sparse_raw_indices"],
        dst_c4_sparse_page_indices=state["dst_c4_sparse_page_indices"],
        src_c4_sparse_page_indices=state["src_c4_sparse_page_indices"],
        dst_c4_sparse_full_indices=state["dst_c4_sparse_full_indices"],
        src_c4_sparse_full_indices=state["src_c4_sparse_full_indices"],
        dst_c128_raw_indices=state["dst_c128_raw_indices"],
        src_c128_raw_indices=state["src_c128_raw_indices"],
        dst_c128_page_indices=state["dst_c128_page_indices"],
        src_c128_page_indices=state["src_c128_page_indices"],
        dst_c128_full_indices=state["dst_c128_full_indices"],
        src_c128_full_indices=state["src_c128_full_indices"],
        rows=rows,
        graph_inputs_bound=graph_inputs_bound,
    )
    if not ok:
        raise RuntimeError("copy_decode_metadata_for_replay did not take the fused path")


def _make_state(
    *,
    rows: int,
    page_table_width: int,
    c4_width: int,
    c128_width: int,
    graph_inputs_bound: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    del graph_inputs_bound
    state: dict[str, torch.Tensor] = {}
    one_d_names = (
        "raw_out_loc",
        "seq_lens",
        "req_seq_lens",
        "extend_lens",
        "positions",
        "req_table_indices",
        "swa_topk_lengths",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c4_sparse_topk_lengths",
        "c128_topk_lengths_clamp1",
    )
    counter = 1
    for name in one_d_names:
        state[f"src_{name}"] = torch.arange(
            counter, counter + rows, dtype=torch.int32, device=device
        )
        state[f"dst_{name}"] = torch.full((rows,), -9999, dtype=torch.int32, device=device)
        counter += 100

    state["src_cu_seqlens_q"] = torch.arange(
        counter, counter + rows + 1, dtype=torch.int32, device=device
    )
    state["dst_cu_seqlens_q"] = torch.full(
        (rows + 1,), -9999, dtype=torch.int32, device=device
    )
    counter += 100

    specs = {
        "page_table": (page_table_width, page_table_width, 0),
        "swa_page_indices": (128, 128, -1),
        "c4_sparse_raw_indices": (c4_width, c4_width, -1),
        "c4_sparse_page_indices": (c4_width, c4_width, -1),
        "c4_sparse_full_indices": (c4_width, c4_width, -1),
        "c128_raw_indices": (c128_width, c128_width, -1),
        "c128_page_indices": (c128_width, c128_width, -1),
        "c128_full_indices": (c128_width, c128_width, -1),
    }
    for name, (src_width, dst_width, _fill) in specs.items():
        values = torch.arange(
            counter,
            counter + rows * src_width,
            dtype=torch.int32,
            device=device,
        ).reshape(rows, src_width)
        state[f"src_{name}"] = values.contiguous()
        state[f"dst_{name}"] = torch.full(
            (rows, dst_width), -9999, dtype=torch.int32, device=device
        )
        counter += rows * src_width + 100
    return state


def _assert_same(reference: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
    for name, ref in reference.items():
        if not name.startswith("dst_"):
            continue
        got = actual[name]
        if not torch.equal(ref, got):
            raise AssertionError(f"{name} mismatch")


def run_case(
    *,
    rows: int,
    max_seq_len: int,
    page_size: int,
    warmup: int,
    iters: int,
    graph_inputs_bound: bool,
) -> dict[str, Any]:
    device = torch.device("cuda")
    page_table_width = (max_seq_len + page_size - 1) // page_size
    c4_width = 512
    c128_width = ((max((max_seq_len + 127) // 128, 1) + 63) // 64) * 64

    legacy_state = _make_state(
        rows=rows,
        page_table_width=page_table_width,
        c4_width=c4_width,
        c128_width=c128_width,
        graph_inputs_bound=graph_inputs_bound,
        device=device,
    )
    fused_state = {
        name: tensor.clone() if name.startswith("dst_") else tensor
        for name, tensor in legacy_state.items()
    }

    os.environ.pop(dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE, None)
    _legacy_copy(legacy_state, rows, graph_inputs_bound=graph_inputs_bound)

    os.environ[dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE] = "1"
    _fused_copy(fused_state, rows, graph_inputs_bound=graph_inputs_bound)
    torch.cuda.synchronize()
    _assert_same(legacy_state, fused_state)

    os.environ.pop(dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE, None)
    legacy_ms = _time_cuda(
        lambda: _legacy_copy(legacy_state, rows, graph_inputs_bound=graph_inputs_bound),
        warmup=warmup,
        iters=iters,
    )
    os.environ[dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE] = "1"
    fused_ms = _time_cuda(
        lambda: _fused_copy(fused_state, rows, graph_inputs_bound=graph_inputs_bound),
        warmup=warmup,
        iters=iters,
    )
    os.environ.pop(dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE, None)

    return {
        "rows": rows,
        "max_seq_len": max_seq_len,
        "page_size": page_size,
        "page_table_width": page_table_width,
        "c4_width": c4_width,
        "c128_width": c128_width,
        "graph_inputs_bound": graph_inputs_bound,
        "legacy_ms": legacy_ms,
        "fused_ms": fused_ms,
        "speedup": legacy_ms / fused_ms if fused_ms > 0 else None,
        "legacy_launches_per_replay": 18 if graph_inputs_bound else 20,
        "fused_launches_per_replay": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=5120)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-graph-inputs-bound", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")

    result = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case": run_case(
            rows=args.rows,
            max_seq_len=args.max_seq_len,
            page_size=args.page_size,
            warmup=args.warmup,
            iters=args.iters,
            graph_inputs_bound=not args.no_graph_inputs_bound,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

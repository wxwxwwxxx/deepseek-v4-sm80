from __future__ import annotations

import math

import pytest
import torch
from minisgl.kernel import deepseek_v4 as dsv4_kernel


def _sm80() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _reference(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    width: int,
    page_size: int,
    ratio: int,
):
    rows = scores.shape[0]
    raw = torch.full((rows, width), -1, dtype=torch.int32)
    page = torch.full_like(raw, -1)
    full = torch.full_like(raw, -1)
    lens = torch.empty(rows, dtype=torch.int32)
    for row in range(rows):
        length = int(seq_lens[row])
        values = scores[row, :length].float().tolist()
        if not all(math.isfinite(value) for value in values):
            lens[row] = -1
            continue
        ordered = sorted(range(length), key=lambda idx: (-values[idx], idx))[:width]
        lens[row] = len(ordered)
        for slot, idx in enumerate(ordered):
            physical = int(page_table[row, idx // page_size]) * page_size + idx % page_size
            raw[row, slot] = idx
            page[row, slot] = physical
            full[row, slot] = physical * ratio + ratio - 1
    return raw, page, full, lens


def _assert_contract(scores, seq_lens, page_table, *, width, page_size=64, ratio=4):
    expected = _reference(
        scores.cpu(), seq_lens.cpu(), page_table.cpu(),
        width=width, page_size=page_size, ratio=ratio,
    )
    before = (scores.clone(), seq_lens.clone(), page_table.clone())
    outputs = []
    for _ in range(20):
        out = dsv4_kernel.topk_transform_512_full_fallback(
            scores, seq_lens, page_table,
            page_size=page_size, width=width, ratio=ratio,
        )
        torch.cuda.synchronize()
        actual = (
            out.raw_indices.cpu(), out.page_indices.cpu(),
            out.full_indices.cpu(), out.topk_lens.cpu(),
        )
        for lhs, rhs in zip(actual, expected, strict=True):
            assert torch.equal(lhs, rhs)
        outputs.append(actual)
    for repeat in outputs[1:]:
        for lhs, rhs in zip(outputs[0], repeat, strict=True):
            assert torch.equal(lhs, rhs)
    assert torch.equal(scores.view(torch.uint8), before[0].view(torch.uint8))
    assert torch.equal(seq_lens, before[1])
    assert torch.equal(page_table, before[2])


@pytest.mark.skipif(not _sm80(), reason="requires an sm80 CUDA device")
def test_dsv4_topk_candidate_b_lengths_ties_and_noncontiguous_pages():
    width = 512
    max_len = 1537
    lengths = torch.tensor([0, 1, 511, 512, 513, 514, 1537], dtype=torch.int32)
    rows = len(lengths)
    base = torch.arange(max_len, dtype=torch.float32)
    scores = torch.stack(
        [
            torch.sin(base * 0.017 + row) + torch.cos(base * 0.031 - row) * 0.2
            for row in range(rows)
        ]
    )
    scores[2, :511] = 1.0  # all equal
    scores[3, :32] = 3.0  # ties fully above the cutoff
    scores[3, 32:512] = torch.linspace(2.0, -1.0, 480)
    scores[4, :10] = 2.0
    scores[4, 10:513] = 1.0  # exact tie crosses the cutoff
    scores[5, :500] = 1.0
    scores[5, 500:514:2] = 0.0
    scores[5, 501:514:2] = -0.0  # +/- zero is one numeric tie class
    repeated = torch.tensor(0.75, dtype=torch.float32).view(torch.int32).item()
    scores[6, 500:540].view(torch.int32).fill_(repeated)  # repeated bit pattern
    page_table = torch.arange((max_len + 63) // 64, dtype=torch.int32)[None].repeat(rows, 1)
    page_table = (page_table.flip(1) + torch.arange(rows, dtype=torch.int32)[:, None] * 97)
    _assert_contract(
        scores.cuda(), lengths.cuda(), page_table.cuda(), width=width
    )


@pytest.mark.skipif(not _sm80(), reason="requires an sm80 CUDA device")
def test_dsv4_topk_candidate_b_width_1024_and_nonfinite_reporting():
    lengths = torch.tensor([1023, 1024, 1025, 1537], dtype=torch.int32)
    base = torch.arange(1537, dtype=torch.float32)
    scores = torch.stack([torch.sin(base * 0.013 + row) for row in range(4)])
    scores[2, :1025] = 0.5
    page_table = torch.arange(25, dtype=torch.int32)[None].repeat(4, 1) + 200
    _assert_contract(
        scores.cuda(), lengths.cuda(), page_table.cuda(), width=1024
    )

    bad_scores = torch.zeros(2, 514, dtype=torch.float32, device="cuda")
    bad_scores[0, 100] = float("inf")
    bad_scores[1, 513] = float("nan")  # outside row-1's valid length
    bad_lens = torch.tensor([514, 513], dtype=torch.int32, device="cuda")
    bad_pages = torch.arange(9, dtype=torch.int32, device="cuda")[None].repeat(2, 1)
    _assert_contract(bad_scores, bad_lens, bad_pages, width=512)


@pytest.mark.skipif(not _sm80(), reason="requires an sm80 CUDA device")
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_dsv4_topk_candidate_b_cuda_graph_replay(rows: int):
    max_len = 1024
    base = torch.arange(max_len, dtype=torch.float32, device="cuda")
    scores = torch.stack([torch.sin(base * 0.019 + row) for row in range(rows)])
    seq_lens = torch.tensor(
        [513 + (row % 3) for row in range(rows)], dtype=torch.int32, device="cuda"
    )
    page_table = torch.arange(16, dtype=torch.int32, device="cuda")[None].repeat(rows, 1)

    # Build the extension and graph pool before capture, then require replay to
    # preserve exact outputs and stable graph-pool allocation.
    eager = dsv4_kernel.topk_transform_512_full_fallback(
        scores, seq_lens, page_table, page_size=64, width=512, ratio=4
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with torch.cuda.graph(graph):
        captured = dsv4_kernel.topk_transform_512_full_fallback(
            scores, seq_lens, page_table, page_size=64, width=512, ratio=4
        )
    torch.cuda.synchronize()
    after_capture = torch.cuda.memory_allocated()
    first = None
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        current = (
            captured.raw_indices.cpu(), captured.page_indices.cpu(),
            captured.full_indices.cpu(), captured.topk_lens.cpu(),
        )
        if first is None:
            first = current
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, current, strict=True))
    after_replay = torch.cuda.memory_allocated()
    assert after_replay == after_capture
    assert after_capture >= before  # required output tensors live in the graph pool
    assert torch.equal(captured.raw_indices, eager.raw_indices)
    assert torch.equal(captured.page_indices, eager.page_indices)
    assert torch.equal(captured.full_indices, eager.full_indices)
    assert torch.equal(captured.topk_lens, eager.topk_lens)

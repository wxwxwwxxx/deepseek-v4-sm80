from __future__ import annotations

from minisgl.core import Batch
from minisgl.engine.graph import GraphRunner


def _runner(*, exact_bs_only: bool) -> GraphRunner:
    runner = object.__new__(GraphRunner)
    runner.max_graph_bs = 4
    runner.graph_map = {1: object(), 2: object(), 4: object()}
    runner.exact_bs_only = exact_bs_only
    return runner


def _decode_batch(*, size: int, padded_size: int) -> Batch:
    reqs = [object()] * size
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs + [object()] * (padded_size - size)
    return batch


def test_cuda_graph_exact_bs_only_rejects_uncaptured_padded_bucket() -> None:
    batch = _decode_batch(size=3, padded_size=4)

    assert _runner(exact_bs_only=False).can_use_cuda_graph(batch)
    assert not _runner(exact_bs_only=True).can_use_cuda_graph(batch)


def test_cuda_graph_exact_bs_only_accepts_captured_exact_batch() -> None:
    batch = _decode_batch(size=4, padded_size=4)

    assert _runner(exact_bs_only=True).can_use_cuda_graph(batch)

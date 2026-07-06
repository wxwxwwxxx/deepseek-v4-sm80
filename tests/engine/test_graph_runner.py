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


def test_cuda_graph_replay_timing_records_batch_and_padded_buckets() -> None:
    runner = _runner(exact_bs_only=False)
    runner._replay_timing_max_samples = 1
    runner.capture_status = {
        "replay_count": 7,
        "replay_timing": {
            "enabled": True,
            "sync_before_after_replay": True,
            "max_samples": 1,
            "count": 0,
            "total_s": 0.0,
            "min_s": None,
            "max_s": None,
            "by_batch_size": {},
            "by_padded_size": {},
            "samples": [],
        },
    }
    batch = _decode_batch(size=3, padded_size=4)

    runner._record_replay_timing(batch, 0.001)
    runner._record_replay_timing(batch, 0.003)

    timing = runner.capture_status["replay_timing"]
    assert timing["count"] == 2
    assert timing["total_s"] == 0.004
    assert timing["mean_s"] == 0.002
    assert timing["by_batch_size"]["3"]["count"] == 2
    assert timing["by_padded_size"]["4"]["max_s"] == 0.003
    assert timing["samples"] == [
        {"replay_index": 8, "batch_size": 3, "padded_size": 4, "elapsed_s": 0.001}
    ]

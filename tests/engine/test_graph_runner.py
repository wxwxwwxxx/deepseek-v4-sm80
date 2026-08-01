from __future__ import annotations

from types import SimpleNamespace

import pytest
from minisgl.core import Batch
from minisgl.engine.graph import GraphContextClass, GraphKey, GraphRunner


def _runner(*, exact_bs_only: bool) -> GraphRunner:
    runner = object.__new__(GraphRunner)
    runner.max_graph_bs = 4
    runner.graph_map = {
        GraphKey(bs, context_class): object()
        for bs in (1, 2, 4)
        for context_class in ("short", "wide")
    }
    runner.exact_bs_only = exact_bs_only
    runner.graph_context_cap = 32_768
    runner.model_context_limit = 1_048_576
    runner.context_classes = (
        GraphContextClass("short", 32_768),
        GraphContextClass("wide", 1_048_576),
    )
    runner.graph_bs_list = [1, 2, 4]
    runner.dummy_req = object()
    runner._last_replay_context_class = None
    runner.capture_status = {
        "eager_decode_count": 0,
        "eager_decode_count_by_batch_size": {},
        "unsupported_m_eager_count": 0,
        "unsupported_m_eager_count_by_batch_size": {},
        "context_overflow_eager_count": 0,
        "context_overflow_eager_count_by_batch_size": {},
        "max_observed_rejected_model_context_length": 0,
    }
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


def _decode_batch_with_lengths(*lengths: int) -> Batch:
    reqs = [SimpleNamespace(device_len=length) for length in lengths]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    return batch


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize(
    "width,expected_class",
    [
        (32_767, "short"),
        (32_768, "short"),
        (32_769, "wide"),
        (64 * 1024, "wide"),
        (512 * 1024 - 1, "wide"),
        (512 * 1024, "wide"),
        (512 * 1024 + 1, "wide"),
        (1_048_575, "wide"),
        (1_048_576, "wide"),
    ],
)
def test_dual_width_dispatch_boundary_ladder(
    batch_size: int,
    width: int,
    expected_class: str,
) -> None:
    runner = _runner(exact_bs_only=False)
    runner.max_graph_bs = 8
    runner.graph_bs_list = [1, 2, 4, 8]
    batch = _decode_batch_with_lengths(*([width] * batch_size))

    runner.pad_batch(batch)

    assert runner.can_use_cuda_graph(batch)
    assert batch.graph_context_class == expected_class
    assert len(batch.padded_reqs) in runner.graph_bs_list


@pytest.mark.parametrize("batch_size", [16, 64, 128])
def test_default_large_m_ladder_has_structural_dual_width_coverage(batch_size: int) -> None:
    runner = _runner(exact_bs_only=False)
    runner.max_graph_bs = 128
    runner.graph_bs_list = [1, 2, 4, *range(8, 129, 8)]
    lengths = [4 * 1024] * batch_size
    lengths[-1] = 32_769
    batch = _decode_batch_with_lengths(*lengths)

    runner.pad_batch(batch)

    assert batch.graph_context_class == "wide"
    assert batch.padded_size == batch_size
    assert runner.can_use_cuda_graph(batch)


def test_cuda_graph_context_class_accepts_both_boundaries_and_rejects_model_overflow() -> None:
    runner = _runner(exact_bs_only=False)

    assert runner._context_class_for_batch(_decode_batch_with_lengths(32_767)).name == "short"
    assert runner._context_class_for_batch(_decode_batch_with_lengths(32_768)).name == "short"
    assert runner._context_class_for_batch(_decode_batch_with_lengths(32_769)).name == "wide"
    assert runner.can_use_cuda_graph(_decode_batch_with_lengths(1_048_575))
    assert runner.can_use_cuda_graph(_decode_batch_with_lengths(1_048_576))
    assert not runner.can_use_cuda_graph(_decode_batch_with_lengths(1_048_577))


def test_cuda_graph_mixed_batch_selects_wide_atomically_and_pads_only_m() -> None:
    runner = _runner(exact_bs_only=False)
    batch = _decode_batch_with_lengths(127, 128, 32_769)

    runner.pad_batch(batch)

    assert batch.graph_context_class == "wide"
    assert len(batch.padded_reqs) == 4
    assert batch.padded_reqs[:3] == batch.reqs
    assert runner.can_use_cuda_graph(batch)


def test_graph_supported_context_eager_is_a_failing_invariant() -> None:
    runner = _runner(exact_bs_only=False)
    batch = _decode_batch_with_lengths(32_769, 64 * 1024)

    with pytest.raises(RuntimeError, match="routed eager despite fitting"):
        runner.record_eager_decode(batch)

    assert runner.capture_status["eager_decode_count"] == 1
    assert runner.capture_status["context_overflow_eager_count"] == 1
    assert runner.capture_status["context_overflow_eager_count_by_batch_size"] == {"2": 1}


def test_unsupported_m_eager_is_counted_separately() -> None:
    runner = _runner(exact_bs_only=False)
    batch = _decode_batch_with_lengths(*([16_385] * 5))

    runner.record_eager_decode(batch)

    assert runner.capture_status["eager_decode_count"] == 1
    assert runner.capture_status["unsupported_m_eager_count"] == 1
    assert runner.capture_status["unsupported_m_eager_count_by_batch_size"] == {"5": 1}
    assert runner.capture_status["context_overflow_eager_count"] == 0


def test_exact_only_ladder_gap_is_counted_as_unsupported_m() -> None:
    runner = _runner(exact_bs_only=True)
    batch = _decode_batch_with_lengths(8_192, 8_192, 8_192)

    runner.record_eager_decode(batch)

    assert runner.capture_status["eager_decode_count"] == 1
    assert runner.capture_status["unsupported_m_eager_count"] == 1
    assert runner.capture_status["unsupported_m_eager_count_by_batch_size"] == {"3": 1}
    assert runner.capture_status["context_overflow_eager_count"] == 0


def test_post_kv_model_cache_prepare_has_explicit_single_lifecycle_entry() -> None:
    runner = object.__new__(GraphRunner)
    runner.capture_status = {}
    calls: list[str] = []

    class Model:
        def prepare_fused_wqa_wkv_bf16_weight_cache(self):
            calls.append("prepare")
            return {"enabled": True, "layers_cached": 2, "total_bytes": 64}

    runner._prepare_post_kv_model_caches(
        Model(),
        stage="post_kv_allocation_pre_graph_warmup",
    )

    assert calls == ["prepare"]
    assert (
        runner.capture_status["post_kv_model_cache_prepare_stage"]
        == "post_kv_allocation_pre_graph_warmup"
    )
    assert runner.capture_status["post_kv_model_cache_prepare_report"]["total_bytes"] == 64

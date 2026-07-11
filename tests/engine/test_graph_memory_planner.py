from minisgl.engine.graph_memory import (
    DSV4_SM80_GRAPH_SAFETY_MARGIN_BYTES,
    MIB,
    compare_graph_capture,
    estimate_dsv4_sm80_graph_memory,
    select_num_pages,
)


def _estimate(sizes: list[int], width: int = 1 << 20):
    return estimate_dsv4_sm80_graph_memory(
        sizes,
        metadata_width=width,
        page_size=256,
        capture_greedy_sample=False,
    )


def test_disabled_graph_has_no_reserve() -> None:
    estimate = _estimate([])
    assert estimate.estimate_bytes == 0
    assert estimate.safety_margin_bytes == 0
    assert estimate.reserve_bytes == 0


def test_dsv4_sm80_estimate_is_monotonic_for_target_ladders() -> None:
    max16 = _estimate([1, 2, 4, 8, 16])
    max64 = _estimate([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64])
    max128 = _estimate(
        [1, 2, 4, 8, 16, *range(24, 129, 8)]
    )
    assert max16.estimate_bytes < max64.estimate_bytes < max128.estimate_bytes
    assert max16.safety_margin_bytes == DSV4_SM80_GRAPH_SAFETY_MARGIN_BYTES
    assert max64.per_graph_bytes == 64 * MIB


def test_metadata_uses_requested_width_upper_bound() -> None:
    narrow = _estimate([1, 2, 4, 8, 16], width=8192)
    wide = _estimate([1, 2, 4, 8, 16], width=1 << 20)
    assert wide.metadata_allowance_bytes > narrow.metadata_allowance_bytes
    assert wide.estimate_bytes > narrow.estimate_bytes


def test_report_exposes_every_estimator_term() -> None:
    report = _estimate([1, 2, 4, 8, 16]).to_report()
    assert report["kind"] == "dsv4_sm80_target12_603_conservative"
    assert report["shared_pool_bytes"] > 0
    assert report["remaining_graph_bytes"] == 4 * 64 * MIB
    assert report["reserve_bytes"] == (
        report["estimate_bytes"] + report["safety_margin_bytes"]
    )


def test_explicit_page_override_is_authoritative_when_safe() -> None:
    selected, baseline, lost = select_num_pages(
        variable_kv_budget_bytes=1000,
        baseline_variable_kv_budget_bytes=1400,
        cache_per_page_bytes=100,
        num_page_override=7,
    )
    assert (selected, baseline, lost) == (7, 14, 7)


def test_unsafe_explicit_page_override_fails_without_modification() -> None:
    try:
        select_num_pages(
            variable_kv_budget_bytes=999,
            baseline_variable_kv_budget_bytes=1400,
            cache_per_page_bytes=100,
            num_page_override=10,
        )
    except RuntimeError as exc:
        assert "requested_pages=10" in str(exc)
        assert "override was not modified" in str(exc)
    else:
        raise AssertionError("unsafe explicit override did not fail closed")


def test_capture_overrun_fails_closed() -> None:
    try:
        compare_graph_capture(
            estimate_bytes=100,
            safety_margin_bytes=10,
            actual_physical_bytes=111,
        )
    except RuntimeError as exc:
        assert "overrun_bytes=1" in str(exc)
    else:
        raise AssertionError("capture overrun did not fail closed")


def test_capture_comparison_reports_remaining_margin() -> None:
    comparison = compare_graph_capture(
        estimate_bytes=100,
        safety_margin_bytes=10,
        actual_physical_bytes=90,
    )
    assert comparison["estimate_error_bytes"] == -10
    assert comparison["remaining_safety_margin_bytes"] == 20

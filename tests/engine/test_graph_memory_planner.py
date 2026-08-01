from minisgl.engine.graph_memory import (
    DSV4_SM80_ADDITIONAL_CONTEXT_CLASS_PER_GRAPH_BYTES,
    DSV4_SM80_GRAPH_SAFETY_MARGIN_BYTES,
    DSV4_SM80_POST_KV_PERSISTENT_CACHE_ALLOWANCE_BYTES,
    MIB,
    compare_graph_capture,
    estimate_dsv4_sm80_graph_memory,
    select_num_pages,
)
from minisgl.engine.graph_policy import generate_cuda_graph_buckets


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
    assert wide.shared_pool_bytes > narrow.shared_pool_bytes
    assert wide.estimate_bytes > narrow.estimate_bytes


def test_reference_width_preserves_target12_603_shared_calibration() -> None:
    estimate = _estimate([1, 2, 4, 128], width=1 << 20)
    assert estimate.shared_pool_bytes == (768 + 128 * 8) * MIB


def test_zero_width_legacy_sentinel_preserves_shared_calibration() -> None:
    estimate = _estimate([1, 2, 4, 128], width=0)
    assert estimate.shared_pool_bytes == (768 + 128 * 8) * MIB


def test_16k_width_scales_only_the_shared_max_row_term() -> None:
    estimate = _estimate([2, 4, 128, 256], width=16 * 1024)
    assert estimate.shared_pool_bytes == 768 * MIB + 256 * 128 * 1024
    assert estimate.remaining_graph_bytes == 3 * 64 * MIB


def test_report_exposes_every_estimator_term() -> None:
    report = _estimate([1, 2, 4, 8, 16]).to_report()
    assert report["kind"] == "dsv4_sm80_target20_1_width_aware_conservative"
    assert report["shared_pool_bytes"] > 0
    assert report["post_kv_persistent_cache_allowance_bytes"] == 256 * MIB
    assert report["remaining_graph_bytes"] == 4 * 64 * MIB
    assert report["reserve_bytes"] == (
        report["estimate_bytes"] + report["safety_margin_bytes"]
    )


def test_dual_width_long_context_m8_estimate_charges_exactly_eight_graphs() -> None:
    estimate = estimate_dsv4_sm80_graph_memory(
        [1, 2, 4, 8],
        metadata_width=1 << 20,
        metadata_widths=(32 * 1024, 1 << 20),
        page_size=256,
        capture_greedy_sample=False,
    )

    assert estimate.kind == "dsv4_sm80_target20_16_dual_width_conservative"
    assert estimate.graph_count == 8
    assert estimate.context_class_count == 2
    assert estimate.metadata_widths == (32 * 1024, 1 << 20)
    assert estimate.remaining_graph_bytes == (
        3 * 64 * MIB
        + 4 * DSV4_SM80_ADDITIONAL_CONTEXT_CLASS_PER_GRAPH_BYTES
    )
    assert (
        estimate.additional_context_class_per_graph_bytes
        == DSV4_SM80_ADDITIONAL_CONTEXT_CLASS_PER_GRAPH_BYTES
    )
    expected_metadata = 8 * ((32 * 1024) // 256 + (1 << 20) // 256) * 4 * 4
    assert estimate.metadata_allowance_bytes == expected_metadata


def test_dual_width_default_m128_stays_inside_added_capacity_gate() -> None:
    sizes = [1, 2, 4, 8, 16, *range(24, 129, 8)]
    single = estimate_dsv4_sm80_graph_memory(
        sizes,
        metadata_width=32 * 1024,
        page_size=256,
        capture_greedy_sample=True,
    )
    dual = estimate_dsv4_sm80_graph_memory(
        sizes,
        metadata_width=1 << 20,
        metadata_widths=(32 * 1024, 1 << 20),
        page_size=256,
        capture_greedy_sample=True,
    )
    cache_per_page_bytes = 2_024_704
    baseline_variable_bytes = 39_760_327_475
    single_pages = (
        baseline_variable_bytes - single.reserve_bytes
    ) // cache_per_page_bytes
    dual_pages = (
        baseline_variable_bytes - dual.reserve_bytes
    ) // cache_per_page_bytes

    assert dual.reserve_bytes - 3_869_245_440 > 256 * MIB
    assert (single_pages - dual_pages) / single_pages <= 0.05


def test_dual_width_high_m256_records_superseded_five_percent_frontier() -> None:
    cache_per_page_bytes = 2_024_704
    baseline_variable_bytes = 26_118_353_715

    def capacity_loss(max_graph_bs: int) -> float:
        sizes = generate_cuda_graph_buckets(max_graph_bs)
        single = estimate_dsv4_sm80_graph_memory(
            sizes,
            metadata_width=32 * 1024,
            page_size=256,
            capture_greedy_sample=True,
        )
        dual = estimate_dsv4_sm80_graph_memory(
            sizes,
            metadata_width=1 << 20,
            metadata_widths=(32 * 1024, 1 << 20),
            page_size=256,
            capture_greedy_sample=True,
        )
        single_pages = (
            baseline_variable_bytes - single.reserve_bytes
        ) // cache_per_page_bytes
        dual_pages = (
            baseline_variable_bytes - dual.reserve_bytes
        ) // cache_per_page_bytes
        return (single_pages - dual_pages) / single_pages

    # TARGET 20.17 supersedes this experiment gate.  Keep the frontier only as
    # a diagnostic so future planner changes cannot silently claim the old
    # bound; the 256-MiB persistent-cache allowance moves it from M88 to M86.
    assert capacity_loss(86) <= 0.05
    assert capacity_loss(87) > 0.05
    assert capacity_loss(96) > 0.05
    assert capacity_loss(256) > 0.14


def test_dual_width_high_m256_retains_half_the_capture_margin() -> None:
    sizes = generate_cuda_graph_buckets(256)
    estimate = estimate_dsv4_sm80_graph_memory(
        sizes,
        metadata_width=1 << 20,
        metadata_widths=(32 * 1024, 1 << 20),
        page_size=256,
        capture_greedy_sample=True,
    )
    measured_physical_bytes = 6_962_544_640

    assert (
        estimate.reserve_bytes - measured_physical_bytes
        >= DSV4_SM80_GRAPH_SAFETY_MARGIN_BYTES // 2
    )
    assert (
        estimate.post_kv_persistent_cache_allowance_bytes
        == DSV4_SM80_POST_KV_PERSISTENT_CACHE_ALLOWANCE_BYTES
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

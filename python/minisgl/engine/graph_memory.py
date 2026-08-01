from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

MIB = 1 << 20

# TARGET 12.603 calibration belongs here, not in generic KV-cache arithmetic.
# The repaired TARGET 12.6025 max64 graph measured an 806 MiB first capture and
# <= 48 MiB for every subsequent graph.  TARGET 20.1 then measured the same
# exact four capture buckets at the 1M model width and at 2K/4K/8K/16K graph
# widths.  The width-independent base and per-graph allowance stay unchanged;
# only the old max-row extrapolation scales with the actual captured width.
DSV4_SM80_SHARED_BASE_BYTES = 768 * MIB
DSV4_SM80_SHARED_PER_MAX_ROW_BYTES = 8 * MIB
DSV4_SM80_SHARED_ROW_REFERENCE_WIDTH = 1 << 20
DSV4_SM80_PER_GRAPH_BYTES = 64 * MIB
# A second context class reuses the first class's graph pool, stream, model
# buffers, and executable topology.  TARGET 20.16 measured a 195,035,136-byte
# graph1M increment for four simultaneously resident keys after graph32.  The
# existing width-scaled shared-pool term accounts for 62 MiB of that increment;
# charging 32 MiB for each additional class key predicts 190 MiB plus metadata.
# It also leaves the existing 512 MiB capture-overrun margin intact.
DSV4_SM80_ADDITIONAL_CONTEXT_CLASS_PER_GRAPH_BYTES = 32 * MIB
# A fresh physical M256 dual-width capture measured 541,065,216 bytes of
# persistent q-projection caches prepared after KV allocation.  The private
# graph estimate itself was within 5 MiB, but charging the entire persistent
# cache to the 512-MiB capture-overrun margin left less than 1 MiB of that
# margin.  Reserve half of the fixed cache here so a complete M256 ladder keeps
# at least 256 MiB of measured overrun margin without changing the cache owner
# or graph allocation order.
DSV4_SM80_POST_KV_PERSISTENT_CACHE_ALLOWANCE_BYTES = 256 * MIB
DSV4_SM80_GRAPH_SAFETY_MARGIN_BYTES = 512 * MIB


@dataclass(frozen=True)
class GraphMemoryEstimate:
    kind: str
    graph_bs: tuple[int, ...]
    max_graph_bs: int
    graph_count: int
    metadata_width: int
    metadata_widths: tuple[int, ...]
    context_class_count: int
    capture_greedy_sample: bool
    shared_pool_bytes: int
    per_graph_bytes: int
    additional_context_class_per_graph_bytes: int
    post_kv_persistent_cache_allowance_bytes: int
    remaining_graph_bytes: int
    metadata_allowance_bytes: int
    estimate_bytes: int
    safety_margin_bytes: int

    @property
    def reserve_bytes(self) -> int:
        return self.estimate_bytes + self.safety_margin_bytes

    def to_report(self) -> dict[str, object]:
        report = asdict(self)
        report["graph_bs"] = list(self.graph_bs)
        report["metadata_widths"] = list(self.metadata_widths)
        report["reserve_bytes"] = self.reserve_bytes
        return report


def estimate_dsv4_sm80_graph_memory(
    graph_bs: Iterable[int],
    *,
    metadata_width: int,
    metadata_widths: Iterable[int] | None = None,
    page_size: int,
    capture_greedy_sample: bool,
    reasoning_sampler_contract_enabled: bool = False,
) -> GraphMemoryEstimate:
    """Return the width-aware conservative graph estimate.

    ``metadata_widths`` contains the immutable context-class widths when more
    than one class is captured. Four int32 page-table-like surfaces are charged
    per class, and graph executables are charged per (M, class) key. The affine
    calibration remains deliberately DSV4/sm80-specific.
    """

    sizes = tuple(sorted({int(bs) for bs in graph_bs if int(bs) > 0}))
    max_bs = max(sizes, default=0)
    bucket_count = len(sizes)
    resolved_widths = (
        tuple(max(int(value), 0) for value in metadata_widths)
        if metadata_widths is not None
        else (max(int(metadata_width), 0),)
    )
    if not resolved_widths:
        resolved_widths = (max(int(metadata_width), 0),)
    context_class_count = len(resolved_widths)
    count = bucket_count * context_class_count
    width = max(resolved_widths, default=max(int(metadata_width), 0))
    page = max(int(page_size), 1)
    if count == 0:
        return GraphMemoryEstimate(
            kind="disabled",
            graph_bs=sizes,
            max_graph_bs=0,
            graph_count=0,
            metadata_width=width,
            metadata_widths=resolved_widths,
            context_class_count=context_class_count,
            capture_greedy_sample=bool(capture_greedy_sample),
            shared_pool_bytes=0,
            per_graph_bytes=0,
            additional_context_class_per_graph_bytes=0,
            post_kv_persistent_cache_allowance_bytes=0,
            remaining_graph_bytes=0,
            metadata_allowance_bytes=0,
            estimate_bytes=0,
            safety_margin_bytes=0,
        )

    # ``width == 0`` is the legacy generic/non-SM80 sentinel.  Preserve its
    # former conservative max-row term; DSV4/SM80 planning always supplies a
    # positive resolved capture width.
    row_calibration_width = width if width > 0 else DSV4_SM80_SHARED_ROW_REFERENCE_WIDTH
    per_max_row = (
        DSV4_SM80_SHARED_PER_MAX_ROW_BYTES * row_calibration_width
        + DSV4_SM80_SHARED_ROW_REFERENCE_WIDTH
        - 1
    ) // DSV4_SM80_SHARED_ROW_REFERENCE_WIDTH
    shared = DSV4_SM80_SHARED_BASE_BYTES + max_bs * per_max_row
    first_class_remaining = (bucket_count - 1) * DSV4_SM80_PER_GRAPH_BYTES
    additional_class_graphs = bucket_count * (context_class_count - 1)
    additional_class_remaining = (
        additional_class_graphs
        * DSV4_SM80_ADDITIONAL_CONTEXT_CLASS_PER_GRAPH_BYTES
    )
    remaining = first_class_remaining + additional_class_remaining
    metadata = sum(
        max_bs * ((class_width + page - 1) // page) * 4 * 4
        for class_width in resolved_widths
    )
    if reasoning_sampler_contract_enabled:
        metadata += max_bs * 4
    if capture_greedy_sample:
        metadata += max_bs * 4
    estimate = (
        shared
        + remaining
        + metadata
        + DSV4_SM80_POST_KV_PERSISTENT_CACHE_ALLOWANCE_BYTES
    )
    return GraphMemoryEstimate(
        kind=(
            "dsv4_sm80_target20_16_dual_width_conservative"
            if context_class_count == 2
            else "dsv4_sm80_target20_1_width_aware_conservative"
        ),
        graph_bs=sizes,
        max_graph_bs=max_bs,
        graph_count=count,
        metadata_width=width,
        metadata_widths=resolved_widths,
        context_class_count=context_class_count,
        capture_greedy_sample=bool(capture_greedy_sample),
        shared_pool_bytes=shared,
        per_graph_bytes=DSV4_SM80_PER_GRAPH_BYTES,
        additional_context_class_per_graph_bytes=(
            DSV4_SM80_ADDITIONAL_CONTEXT_CLASS_PER_GRAPH_BYTES
            if context_class_count > 1
            else 0
        ),
        post_kv_persistent_cache_allowance_bytes=(
            DSV4_SM80_POST_KV_PERSISTENT_CACHE_ALLOWANCE_BYTES
        ),
        remaining_graph_bytes=remaining,
        metadata_allowance_bytes=metadata,
        estimate_bytes=estimate,
        safety_margin_bytes=DSV4_SM80_GRAPH_SAFETY_MARGIN_BYTES,
    )


def empty_graph_memory_estimate(graph_bs: Iterable[int] = ()) -> GraphMemoryEstimate:
    return estimate_dsv4_sm80_graph_memory(
        graph_bs,
        metadata_width=0,
        page_size=1,
        capture_greedy_sample=False,
    )


def select_num_pages(
    *,
    variable_kv_budget_bytes: int,
    baseline_variable_kv_budget_bytes: int,
    cache_per_page_bytes: int,
    num_page_override: int | None,
) -> tuple[int, int, int]:
    """Select pages without silently changing an explicit override.

    Returns ``(selected_pages, baseline_pages_without_graph, lost_pages)``.
    """

    per_page = int(cache_per_page_bytes)
    if per_page <= 0:
        raise ValueError("cache_per_page_bytes must be positive")
    safe_pages = int(variable_kv_budget_bytes) // per_page
    baseline_pages = max(0, int(baseline_variable_kv_budget_bytes) // per_page)
    if num_page_override is None:
        selected = safe_pages
    else:
        selected = int(num_page_override)
        if selected * per_page > int(variable_kv_budget_bytes):
            raise RuntimeError(
                "Explicit num_pages override is unsafe after CUDA graph reserve planning: "
                f"requested_pages={selected}, requested_kv_bytes={selected * per_page}, "
                f"safe_variable_kv_budget_bytes={int(variable_kv_budget_bytes)}. "
                "The override was not modified."
            )
    return selected, baseline_pages, max(0, baseline_pages - selected)


def compare_graph_capture(
    *,
    estimate_bytes: int,
    safety_margin_bytes: int,
    actual_physical_bytes: int,
) -> dict[str, int | float]:
    estimate = int(estimate_bytes)
    margin = int(safety_margin_bytes)
    actual = int(actual_physical_bytes)
    error = actual - estimate
    remaining = estimate + margin - actual
    if remaining < 0:
        raise RuntimeError(
            "CUDA graph capture exceeded its pre-KV estimate plus safety margin: "
            f"actual_bytes={actual}, estimate_bytes={estimate}, margin_bytes={margin}, "
            f"overrun_bytes={-remaining}."
        )
    return {
        "actual_physical_bytes": actual,
        "estimate_error_bytes": error,
        "absolute_error_bytes": abs(error),
        "relative_error": (float(error) / float(actual)) if actual else 0.0,
        "remaining_safety_margin_bytes": remaining,
    }

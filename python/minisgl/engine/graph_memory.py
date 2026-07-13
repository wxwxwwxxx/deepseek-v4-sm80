from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

MIB = 1 << 20

# TARGET 12.603 calibration belongs here, not in generic KV-cache arithmetic.
# The repaired TARGET 12.6025 max64 graph measured an 806 MiB first capture and
# <= 48 MiB for every subsequent graph.  The affine shared term deliberately
# covers the wider production metadata surfaces and max128 extrapolation; the
# per-graph term rounds the observed maximum up to 64 MiB.
DSV4_SM80_SHARED_BASE_BYTES = 768 * MIB
DSV4_SM80_SHARED_PER_MAX_ROW_BYTES = 8 * MIB
DSV4_SM80_PER_GRAPH_BYTES = 64 * MIB
DSV4_SM80_GRAPH_SAFETY_MARGIN_BYTES = 512 * MIB


@dataclass(frozen=True)
class GraphMemoryEstimate:
    kind: str
    graph_bs: tuple[int, ...]
    max_graph_bs: int
    graph_count: int
    metadata_width: int
    capture_greedy_sample: bool
    shared_pool_bytes: int
    per_graph_bytes: int
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
        report["reserve_bytes"] = self.reserve_bytes
        return report


def estimate_dsv4_sm80_graph_memory(
    graph_bs: Iterable[int],
    *,
    metadata_width: int,
    page_size: int,
    capture_greedy_sample: bool,
    reasoning_sampler_contract_enabled: bool = False,
) -> GraphMemoryEstimate:
    """Return the TARGET 12.603 conservative repaired-graph estimate.

    ``metadata_width`` is the requested/model-width upper bound, rather than a
    post-KV effective width, so graph planning cannot become circular.  Four
    int32 page-table-like graph surfaces are charged explicitly.  The main
    affine calibration is intentionally DSV4/sm80-specific and replaceable by
    a future temporary-capture profiler implementing the same return contract.
    """

    sizes = tuple(sorted({int(bs) for bs in graph_bs if int(bs) > 0}))
    max_bs = max(sizes, default=0)
    count = len(sizes)
    width = max(int(metadata_width), 0)
    page = max(int(page_size), 1)
    if count == 0:
        return GraphMemoryEstimate(
            kind="disabled",
            graph_bs=sizes,
            max_graph_bs=0,
            graph_count=0,
            metadata_width=width,
            capture_greedy_sample=bool(capture_greedy_sample),
            shared_pool_bytes=0,
            per_graph_bytes=0,
            remaining_graph_bytes=0,
            metadata_allowance_bytes=0,
            estimate_bytes=0,
            safety_margin_bytes=0,
        )

    shared = DSV4_SM80_SHARED_BASE_BYTES + max_bs * DSV4_SM80_SHARED_PER_MAX_ROW_BYTES
    remaining = (count - 1) * DSV4_SM80_PER_GRAPH_BYTES
    pages_per_request = (width + page - 1) // page
    metadata = max_bs * pages_per_request * 4 * 4
    if reasoning_sampler_contract_enabled:
        metadata += max_bs * 4
    if capture_greedy_sample:
        metadata += max_bs * 4
    estimate = shared + remaining + metadata
    return GraphMemoryEstimate(
        kind="dsv4_sm80_target12_603_conservative",
        graph_bs=sizes,
        max_graph_bs=max_bs,
        graph_count=count,
        metadata_width=width,
        capture_greedy_sample=bool(capture_greedy_sample),
        shared_pool_bytes=shared,
        per_graph_bytes=DSV4_SM80_PER_GRAPH_BYTES,
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

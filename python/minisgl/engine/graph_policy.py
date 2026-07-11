from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

CUDA_GRAPH_BUCKET_GENERATION_RULE = "target12_60_sparse_v1"
CUDA_GRAPH_RELEASE_MAX_BS = 512

GraphBucketSourceMode = Literal[
    "release_default", "explicit_list", "explicit_max", "disabled"
]


@dataclass(frozen=True)
class ResolvedCudaGraphBucketPolicy:
    """Pure, pre-KV CUDA graph bucket policy contract."""

    enabled: bool
    source_mode: GraphBucketSourceMode
    requested_bs: tuple[int, ...] | None
    requested_max_bs: int | None
    effective_max_running_req: int
    resolved_bs: tuple[int, ...]
    resolved_max_bs: int
    generation_rule: str
    validation_or_cap_reason: str

    def to_report(self) -> dict[str, object]:
        report = asdict(self)
        report["requested_bs"] = (
            list(self.requested_bs) if self.requested_bs is not None else None
        )
        report["resolved_bs"] = list(self.resolved_bs)
        return report


def generate_cuda_graph_buckets(max_bs: int) -> tuple[int, ...]:
    """Generate the sole TARGET 12.60-aligned release bucket ladder."""

    endpoint = int(max_bs)
    if endpoint < 1:
        return ()
    if endpoint > CUDA_GRAPH_RELEASE_MAX_BS:
        raise ValueError(
            "CUDA graph generated-policy maximum exceeds the release limit: "
            f"requested_max_bs={endpoint}, release_limit={CUDA_GRAPH_RELEASE_MAX_BS}. "
            "Use an explicit bucket list only for isolated diagnostic captures."
        )
    sizes = [value for value in (1, 2, 4) if value <= endpoint]
    sizes.extend(range(8, min(endpoint, 256) + 1, 8))
    if endpoint >= 272:
        sizes.extend(range(272, endpoint + 1, 16))
    sizes.append(endpoint)
    return tuple(sorted(set(sizes)))


def resolve_cuda_graph_bucket_policy(
    *,
    cuda_graph_bs: Iterable[int] | None,
    cuda_graph_max_bs: int | None,
    effective_max_running_req: int,
    graph_disabled: bool = False,
    release_default_bs: Iterable[int] | None = None,
    legacy_default_max_bs: int | None = None,
) -> ResolvedCudaGraphBucketPolicy:
    """Resolve one authoritative graph policy without touching CUDA state."""

    effective_max = int(effective_max_running_req)
    if effective_max < 1:
        raise ValueError(
            f"effective max_running_req must be positive, got {effective_max}."
        )
    requested_bs = None if cuda_graph_bs is None else tuple(int(v) for v in cuda_graph_bs)
    requested_max = None if cuda_graph_max_bs is None else int(cuda_graph_max_bs)

    disabled_reason: str | None = None
    if graph_disabled:
        disabled_reason = "explicit graph-disable switch"
    elif requested_bs == ():
        disabled_reason = "explicit empty bucket list"
    elif requested_max == 0:
        disabled_reason = "explicit maximum zero"
    if disabled_reason is not None:
        return ResolvedCudaGraphBucketPolicy(
            False,
            "disabled",
            requested_bs,
            requested_max,
            effective_max,
            (),
            0,
            CUDA_GRAPH_BUCKET_GENERATION_RULE,
            disabled_reason,
        )

    if requested_max is not None and requested_max < 0:
        raise ValueError(f"cuda_graph_max_bs must be non-negative, got {requested_max}.")

    if requested_bs is not None:
        if any(value <= 0 for value in requested_bs):
            raise ValueError(
                f"cuda_graph_bs values must be positive: requested_bs={list(requested_bs)}."
            )
        resolved = tuple(sorted(set(requested_bs)))
        resolved_max = max(resolved)
        if requested_max is not None and requested_max != resolved_max:
            raise ValueError(
                "Conflicting explicit CUDA graph bucket list and maximum: "
                f"list_max={resolved_max}, requested_max_bs={requested_max}."
            )
        if resolved_max > effective_max:
            raise ValueError(
                "Explicit CUDA graph bucket policy exceeds effective max_running_req: "
                f"resolved_max_bs={resolved_max}, effective_max_running_req={effective_max}."
            )
        return ResolvedCudaGraphBucketPolicy(
            True,
            "explicit_list",
            requested_bs,
            requested_max,
            effective_max,
            resolved,
            resolved_max,
            CUDA_GRAPH_BUCKET_GENERATION_RULE,
            (
                "positive list normalized by sort/dedup"
                if resolved != requested_bs
                else "positive sorted unique list"
            ),
        )

    if requested_max is not None:
        if requested_max > effective_max:
            raise ValueError(
                "Explicit CUDA graph maximum exceeds effective max_running_req: "
                f"requested_max_bs={requested_max}, effective_max_running_req={effective_max}."
            )
        resolved = generate_cuda_graph_buckets(requested_max)
        return ResolvedCudaGraphBucketPolicy(
            True,
            "explicit_max",
            None,
            requested_max,
            effective_max,
            resolved,
            requested_max,
            CUDA_GRAPH_BUCKET_GENERATION_RULE,
            "explicit maximum validated; exact endpoint included",
        )

    if release_default_bs is not None:
        defaults = tuple(sorted({int(v) for v in release_default_bs}))
        if not defaults or any(value <= 0 for value in defaults):
            raise ValueError(f"release CUDA graph defaults are invalid: {list(defaults)}.")
        resolved = tuple(value for value in defaults if value <= effective_max)
        reason = "release default unchanged"
        if resolved != defaults:
            reason = f"release default capped by effective max_running_req={effective_max}"
    else:
        default_max = int(legacy_default_max_bs or 0)
        capped_max = min(default_max, effective_max, CUDA_GRAPH_RELEASE_MAX_BS)
        resolved = generate_cuda_graph_buckets(capped_max)
        reason = "legacy automatic maximum preserved"
        if capped_max != default_max:
            reason = f"automatic maximum capped to {capped_max}"
    return ResolvedCudaGraphBucketPolicy(
        bool(resolved),
        "release_default" if resolved else "disabled",
        None,
        None,
        effective_max,
        resolved,
        max(resolved, default=0),
        CUDA_GRAPH_BUCKET_GENERATION_RULE,
        reason,
    )

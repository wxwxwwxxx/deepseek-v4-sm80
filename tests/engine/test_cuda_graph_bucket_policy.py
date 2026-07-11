from __future__ import annotations

import pytest
from minisgl.engine.engine import validate_graph_bucket_contract
from minisgl.engine.graph_memory import estimate_dsv4_sm80_graph_memory
from minisgl.engine.graph_policy import (
    CUDA_GRAPH_BUCKET_GENERATION_RULE,
    resolve_cuda_graph_bucket_policy,
)

RELEASE = (1, 2, 4, 8, 16)


def _resolve(**kwargs):
    defaults = {
        "cuda_graph_bs": None,
        "cuda_graph_max_bs": None,
        "effective_max_running_req": 256,
        "release_default_bs": RELEASE,
    }
    defaults.update(kwargs)
    return resolve_cuda_graph_bucket_policy(**defaults)


def test_omitted_dsv4_release_policy_stays_max16() -> None:
    policy = _resolve()
    assert policy.enabled
    assert policy.source_mode == "release_default"
    assert policy.resolved_bs == RELEASE
    assert policy.generation_rule == CUDA_GRAPH_BUCKET_GENERATION_RULE


def test_explicit_list_is_authoritative_and_normalized() -> None:
    policy = _resolve(cuda_graph_bs=[16, 4, 4, 1])
    assert policy.source_mode == "explicit_list"
    assert policy.requested_bs == (16, 4, 4, 1)
    assert policy.resolved_bs == (1, 4, 16)
    assert "sort/dedup" in policy.validation_or_cap_reason


@pytest.mark.parametrize("endpoint", [16, 64, 128])
def test_max_only_generates_target12_60_ladder(endpoint: int) -> None:
    policy = _resolve(cuda_graph_max_bs=endpoint)
    assert policy.source_mode == "explicit_max"
    assert policy.resolved_max_bs == endpoint
    assert policy.resolved_bs[-1] == endpoint
    assert policy.resolved_bs[:3] == (1, 2, 4)


def test_max_only_includes_non_step_endpoint() -> None:
    policy = _resolve(cuda_graph_max_bs=67)
    assert policy.resolved_bs[-3:] == (56, 64, 67)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"cuda_graph_bs": []}, "empty bucket"),
        ({"cuda_graph_max_bs": 0}, "maximum zero"),
        ({"graph_disabled": True}, "disable switch"),
    ],
)
def test_all_disable_surfaces_resolve_empty(kwargs: dict, reason: str) -> None:
    policy = _resolve(**kwargs)
    assert not policy.enabled
    assert policy.source_mode == "disabled"
    assert policy.resolved_bs == ()
    assert reason in policy.validation_or_cap_reason


def test_explicit_list_and_conflicting_max_fail() -> None:
    with pytest.raises(ValueError, match="Conflicting explicit"):
        _resolve(cuda_graph_bs=[1, 2, 8], cuda_graph_max_bs=16)


def test_explicit_policy_above_effective_request_capacity_fails() -> None:
    with pytest.raises(ValueError, match="effective max_running_req"):
        _resolve(cuda_graph_max_bs=64, effective_max_running_req=63)


def test_generated_release_max_above_512_fails() -> None:
    with pytest.raises(ValueError, match="release limit"):
        _resolve(cuda_graph_max_bs=513, effective_max_running_req=1024)


def test_generic_legacy_default_preserves_a100_ladder() -> None:
    policy = resolve_cuda_graph_bucket_policy(
        cuda_graph_bs=None,
        cuda_graph_max_bs=None,
        effective_max_running_req=256,
        legacy_default_max_bs=160,
    )
    assert policy.source_mode == "release_default"
    assert policy.resolved_bs == (1, 2, 4, *range(8, 161, 8))


def test_estimator_consumes_exact_resolved_tuple() -> None:
    policy = _resolve(cuda_graph_max_bs=64)
    estimate = estimate_dsv4_sm80_graph_memory(
        policy.resolved_bs,
        metadata_width=1 << 20,
        page_size=256,
        capture_greedy_sample=False,
    )
    assert estimate.graph_bs == policy.resolved_bs


def test_post_capture_contract_accepts_descending_capture_order() -> None:
    validate_graph_bucket_contract(
        resolved_bs=(1, 2, 4),
        estimated_bs=(1, 2, 4),
        runner_requested_bs=[1, 2, 4],
        runner_captured_bs=[4, 2, 1],
        capture_error=None,
    )


def test_post_capture_contract_rejects_policy_mismatch_as_programming_error() -> None:
    with pytest.raises(RuntimeError, match="Programming error"):
        validate_graph_bucket_contract(
            resolved_bs=(1, 2, 4),
            estimated_bs=(1, 2, 4),
            runner_requested_bs=[1, 2, 8],
            runner_captured_bs=[8, 2, 1],
            capture_error=None,
        )

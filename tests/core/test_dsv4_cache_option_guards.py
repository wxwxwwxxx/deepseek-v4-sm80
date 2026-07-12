from __future__ import annotations

from types import SimpleNamespace

import pytest
from minisgl.scheduler.scheduler import resolve_dsv4_cache_type


def _config(
    *,
    is_deepseek_v4: bool = True,
    cache_type: str = "radix",
    page_size: int = 256,
    window_size: int = 128,
    enable_radix: bool = False,
    enable_swa_tail_v1: bool = False,
    enable_component_loc_ownership: bool = False,
    enable_swa_independent_lifecycle: bool = False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=is_deepseek_v4, window_size=window_size),
        cache_type=cache_type,
        page_size=page_size,
        enable_dsv4_radix_prefix_cache=enable_radix,
        enable_dsv4_swa_tail_retention_v1=enable_swa_tail_v1,
        enable_dsv4_component_loc_ownership=enable_component_loc_ownership,
        enable_dsv4_swa_independent_lifecycle=enable_swa_independent_lifecycle,
    )


def test_dsv4_swa_tail_retention_v1_fails_closed_before_runtime_cache_use():
    with pytest.raises(RuntimeError, match="fail-closed.*DESIGN.md"):
        resolve_dsv4_cache_type(_config(enable_swa_tail_v1=True))


def test_dsv4_cache_type_resolution_preserves_phase1_radix_guardrails():
    assert resolve_dsv4_cache_type(_config(enable_radix=False)) == "naive"
    assert resolve_dsv4_cache_type(_config(enable_radix=True)) == "radix"
    with pytest.raises(ValueError, match="DeepSeek V4 Flash only"):
        resolve_dsv4_cache_type(_config(is_deepseek_v4=False, cache_type="radix"))

    with pytest.raises(ValueError, match="page size divisible"):
        resolve_dsv4_cache_type(_config(enable_radix=True, page_size=64))

    with pytest.raises(ValueError, match="requires cache_type='radix'"):
        resolve_dsv4_cache_type(_config(cache_type="naive", enable_radix=True))


def test_dsv4_component_loc_ownership_requires_phase1_radix_and_safe_window():
    with pytest.raises(ValueError, match="requires the phase-1 radix"):
        resolve_dsv4_cache_type(_config(enable_component_loc_ownership=True))

    with pytest.raises(ValueError, match="window_size.*<= page_size"):
        resolve_dsv4_cache_type(
            _config(
                page_size=128,
                window_size=256,
                enable_radix=True,
                enable_component_loc_ownership=True,
            )
        )

    assert (
        resolve_dsv4_cache_type(
            _config(enable_radix=True, enable_component_loc_ownership=True)
        )
        == "radix"
    )


def test_dsv4_swa_independent_lifecycle_requires_radix_and_route_b():
    with pytest.raises(ValueError, match="requires --enable-dsv4-radix-prefix-cache"):
        resolve_dsv4_cache_type(_config(enable_swa_independent_lifecycle=True))

    with pytest.raises(ValueError, match="requires --enable-dsv4-component-loc-ownership"):
        resolve_dsv4_cache_type(
            _config(
                enable_radix=True,
                enable_swa_independent_lifecycle=True,
            )
        )

    assert (
        resolve_dsv4_cache_type(
            _config(
                page_size=128,
                window_size=256,
                enable_radix=True,
                enable_component_loc_ownership=True,
                enable_swa_independent_lifecycle=True,
            )
        )
        == "radix"
    )

from __future__ import annotations

from types import SimpleNamespace

import pytest

from minisgl.scheduler.scheduler import resolve_dsv4_cache_type


def _config(
    *,
    is_deepseek_v4: bool = True,
    cache_type: str = "radix",
    page_size: int = 256,
    enable_radix: bool = False,
    enable_swa_tail_v1: bool = False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=is_deepseek_v4),
        cache_type=cache_type,
        page_size=page_size,
        enable_dsv4_radix_prefix_cache=enable_radix,
        enable_dsv4_swa_tail_retention_v1=enable_swa_tail_v1,
    )


def test_dsv4_swa_tail_retention_v1_fails_closed_before_runtime_cache_use():
    with pytest.raises(RuntimeError, match="fail-closed.*DESIGN.md"):
        resolve_dsv4_cache_type(_config(enable_swa_tail_v1=True))


def test_dsv4_cache_type_resolution_preserves_phase1_radix_guardrails():
    assert resolve_dsv4_cache_type(_config(enable_radix=False)) == "naive"
    assert resolve_dsv4_cache_type(_config(enable_radix=True)) == "radix"
    assert resolve_dsv4_cache_type(_config(is_deepseek_v4=False, cache_type="radix")) == "radix"

    with pytest.raises(ValueError, match="page size divisible"):
        resolve_dsv4_cache_type(_config(enable_radix=True, page_size=64))

    with pytest.raises(ValueError, match="requires cache_type='radix'"):
        resolve_dsv4_cache_type(_config(cache_type="naive", enable_radix=True))

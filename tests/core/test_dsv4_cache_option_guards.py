from __future__ import annotations

from types import SimpleNamespace

import pytest
from minisgl.scheduler.scheduler import validate_dsv4_release_cache_contract


def _config(*, is_deepseek_v4: bool = True, page_size: int = 256):
    return SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=is_deepseek_v4),
        page_size=page_size,
    )


def test_dsv4_release_cache_contract_accepts_canonical_config():
    assert validate_dsv4_release_cache_contract(_config()) is None


def test_dsv4_release_cache_contract_rejects_non_dsv4_models():
    with pytest.raises(ValueError, match="DeepSeek V4 Flash only"):
        validate_dsv4_release_cache_contract(_config(is_deepseek_v4=False))


@pytest.mark.parametrize("page_size", [1, 64, 128, 512])
def test_dsv4_release_cache_contract_requires_public_page_size_256(page_size):
    with pytest.raises(ValueError, match="page_size=256"):
        validate_dsv4_release_cache_contract(_config(page_size=page_size))

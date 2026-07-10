from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from minisgl.engine import engine as engine_module


def _fake_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=True, is_moe=True),
        attention_backend="auto",
        moe_backend="auto",
        allow_dsv4_cuda_graph=False,
        cuda_graph_bs=None,
        cuda_graph_max_bs=None,
        cuda_graph_capture_fail_open=False,
        cuda_graph_capture_greedy_sample=False,
        page_size=1,
        max_extend_tokens=8192,
        max_extend_tokens_explicit=False,
        cache_type="radix",
        enable_dsv4_radix_prefix_cache=False,
        enable_dsv4_component_loc_ownership=False,
        enable_dsv4_swa_independent_lifecycle=False,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _clear_dsv4_env(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _restore_dsv4_env():
    original = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("MINISGL_DSV4_")
    }
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_"):
            os.environ.pop(name, None)
    yield
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_"):
            os.environ.pop(name, None)
    os.environ.update(original)


def test_deepseek_v4_release_defaults_make_llm_path_recipe_free(monkeypatch):
    _clear_dsv4_env(monkeypatch)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)

    config = _fake_config()

    engine_module._adjust_config(config)

    assert config.attention_backend == "dsv4"
    assert config.page_size == 256
    assert config.cache_type == "radix"
    assert config.enable_dsv4_radix_prefix_cache is True
    assert config.enable_dsv4_component_loc_ownership is True
    assert config.enable_dsv4_swa_independent_lifecycle is True
    assert config.max_extend_tokens == 8192
    assert config.allow_dsv4_cuda_graph is True
    assert config.cuda_graph_bs == [1, 2, 4, 8, 16]
    assert config.cuda_graph_max_bs == 16
    assert config.cuda_graph_capture_fail_open is True
    assert config.moe_backend == "fused"

    for name, value in engine_module._DSV4_SM80_RELEASE_DEFAULT_ENV.items():
        assert os.environ[name] == value
    assert os.environ["MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS"] == "swa,c4"
    assert os.environ["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert os.environ["MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP"] == "1"
    assert os.environ["MINISGL_DSV4_SM80_LINEAR_BF16_FP32"] == "1"
    assert os.environ["MINISGL_DSV4_MARLIN_WNA16_PREBUILD"] == "1"
    assert os.environ["MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS"] == "1"
    assert os.environ["MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING"] == "before_kv_alloc"
    assert os.environ["MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT"] == "1"
    assert os.environ["MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC"] == "component"
    assert os.environ["MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE"] == "1"
    assert os.environ["MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE"] == "1"
    assert os.environ["MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA"] == "1"
    assert os.environ["MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED"] == "1"
    assert "MINISGL_DSV4_SM80_KV_FP8" not in os.environ
    assert "MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION" not in os.environ


def test_deepseek_v4_release_defaults_can_be_disabled_for_fallback(monkeypatch):
    _clear_dsv4_env(monkeypatch)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)
    monkeypatch.setenv(engine_module._DSV4_DISABLE_RELEASE_DEFAULTS_ENV, "1")

    config = _fake_config()

    engine_module._adjust_config(config)

    assert config.attention_backend == "dsv4"
    assert config.page_size == 1
    assert config.max_extend_tokens == 8192
    assert config.enable_dsv4_radix_prefix_cache is False
    assert config.enable_dsv4_component_loc_ownership is False
    assert config.allow_dsv4_cuda_graph is False
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0
    for name in engine_module._DSV4_SM80_RELEASE_DEFAULT_ENV:
        assert name not in os.environ


def test_deepseek_v4_release_defaults_honor_explicit_sm80_env(monkeypatch):
    _clear_dsv4_env(monkeypatch)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)
    monkeypatch.setenv("MINISGL_DSV4_SM80_V0_BF16", "1")

    config = _fake_config()

    engine_module._adjust_config(config)

    assert config.page_size == 1
    assert config.max_extend_tokens == 8192
    assert config.enable_dsv4_radix_prefix_cache is False
    assert config.allow_dsv4_cuda_graph is False
    assert config.cuda_graph_bs == []
    assert os.environ["MINISGL_DSV4_SM80_V0_BF16"] == "1"
    assert "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE" not in os.environ


def test_deepseek_v4_release_defaults_allow_hc_cleanup_addon_env(monkeypatch):
    _clear_dsv4_env(monkeypatch)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)
    monkeypatch.setenv("MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP", "1")

    config = _fake_config()

    engine_module._adjust_config(config)

    assert config.page_size == 256
    assert config.enable_dsv4_radix_prefix_cache is True
    assert config.enable_dsv4_component_loc_ownership is True
    assert config.allow_dsv4_cuda_graph is True
    assert os.environ["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert os.environ["MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP"] == "1"
    assert os.environ["MINISGL_DSV4_SM80_LINEAR_BF16_FP32"] == "1"


def test_deepseek_v4_release_defaults_honor_explicit_max_extend_tokens(monkeypatch):
    _clear_dsv4_env(monkeypatch)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)

    config = _fake_config(max_extend_tokens=16384)

    engine_module._adjust_config(config)

    assert config.max_extend_tokens == 16384
    assert config.page_size == 256
    assert config.enable_dsv4_swa_independent_lifecycle is True


def test_deepseek_v4_release_defaults_honor_explicit_generic_max_extend_tokens(monkeypatch):
    _clear_dsv4_env(monkeypatch)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)

    config = _fake_config(max_extend_tokens=8192, max_extend_tokens_explicit=True)

    engine_module._adjust_config(config)

    assert config.max_extend_tokens == 8192
    assert config.page_size == 256
    assert config.enable_dsv4_swa_independent_lifecycle is True

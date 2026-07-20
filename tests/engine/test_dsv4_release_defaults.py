from __future__ import annotations

from types import SimpleNamespace

import pytest
from minisgl.dsv4_release import DSV4_RELEASE
from minisgl.engine import engine as engine_module


def _fake_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=True, is_moe=True),
        enable_reasoning_sampler_contract=False,
        dsv4_sm80_recipe=None,
        max_running_req=128,
        max_running_req_explicit=False,
        attention_backend="auto",
        allow_dsv4_cuda_graph=False,
        cuda_graph_bs=None,
        cuda_graph_max_bs=None,
        cuda_graph_capture_fail_open=False,
        cuda_graph_capture_greedy_sample=False,
        page_size=1,
        max_extend_tokens=8192,
        max_extend_tokens_explicit=False,
        context_length=None,
        cache_type="radix",
        enable_dsv4_radix_prefix_cache=False,
        enable_dsv4_component_loc_ownership=False,
        enable_dsv4_swa_independent_lifecycle=False,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_deepseek_v4_release_defaults_make_llm_path_recipe_free(monkeypatch):
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
    assert config.dsv4_sm80_recipe is None
    assert config.max_running_req == 128
    assert config.cuda_graph_max_bs == 128
    assert config.cuda_graph_policy.source_mode == "explicit_max"
    assert config.cuda_graph_policy.resolved_bs[-1] == 128
    assert config.cuda_graph_capture_fail_open is True
    assert DSV4_RELEASE.direct_graph_metadata_groups == frozenset({"swa", "c4"})
    assert "c128" not in DSV4_RELEASE.direct_graph_metadata_groups
    assert DSV4_RELEASE.marlin_release_timing == "before_kv_alloc"
    assert DSV4_RELEASE.clear_allocated_page_scope == "component"
    assert DSV4_RELEASE.pynccl_max_buffer_bytes == 32 * 1024 * 1024
    assert DSV4_RELEASE.release_raw_expert_weights is True
    assert DSV4_RELEASE.marlin_prebuild is True


@pytest.mark.parametrize("backend", ["fa", "fi", "trtllm", "fa,fi"])
def test_removed_attention_backends_fail_instead_of_falling_back(monkeypatch, backend):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    config = _fake_config(attention_backend=backend)

    with pytest.raises(ValueError, match="DSV4 attention backend only"):
        engine_module._adjust_config(config)


def test_optimized_explicit_contract_enable_emits_rank0_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        engine_module.logger,
        "warning_rank0",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    config = _fake_config(enable_reasoning_sampler_contract=True)

    engine_module._adjust_config(config)

    assert len(warnings) == 1
    assert "ENABLED" in warnings[0]
    assert "masks protocol delimiters and EOS" in warnings[0]
    assert "raw sampling distribution" in warnings[0]


def test_deepseek_v4_release_defaults_honor_explicit_max_extend_tokens(monkeypatch):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)

    config = _fake_config(max_extend_tokens=16384)

    engine_module._adjust_config(config)

    assert config.max_extend_tokens == 16384
    assert config.page_size == 256
    assert config.enable_dsv4_swa_independent_lifecycle is True


def test_deepseek_v4_release_defaults_honor_explicit_generic_max_extend_tokens(monkeypatch):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)

    config = _fake_config(max_extend_tokens=8192, max_extend_tokens_explicit=True)

    engine_module._adjust_config(config)

    assert config.max_extend_tokens == 8192
    assert config.page_size == 256
    assert config.enable_dsv4_swa_independent_lifecycle is True


@pytest.mark.parametrize(
    "recipe,max_req,graph_max,max_seq",
    [
        ("default_m128", 128, 128, None),
        ("low_m64", 64, 64, None),
        ("high_m256", 256, 256, None),
        ("long_context_m4", 4, 4, 524_288),
    ],
)
def test_public_dsv4_sm80_recipes_resolve_through_one_graph_policy(
    monkeypatch, recipe, max_req, graph_max, max_seq
):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "warning_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)
    config = _fake_config(dsv4_sm80_recipe=recipe)

    engine_module._adjust_config(config)

    assert config.max_running_req == max_req
    assert config.cuda_graph_policy.source_mode == "explicit_max"
    assert config.cuda_graph_policy.resolved_max_bs == graph_max
    assert config.context_length == max_seq


@pytest.mark.parametrize(
    "recipe",
    [
        "dsv4_sm80_low_m64",
        "dsv4_sm80_high_m256",
        "dsv4_sm80_long_context_512k",
        "dsv4_sm80_mid_m128",
        "dsv4_sm80_balanced",
        "dsv4_sm80_1m_smoke",
    ],
)
def test_removed_dsv4_sm80_recipe_names_fail(monkeypatch, recipe):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    config = _fake_config(dsv4_sm80_recipe=recipe)

    with pytest.raises(ValueError, match="Unknown dsv4_sm80_recipe"):
        engine_module._adjust_config(config)


def test_recipe_preserves_explicit_request_graph_and_sequence_overrides(monkeypatch):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    warnings = []
    monkeypatch.setattr(
        engine_module.logger,
        "warning_rank0",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)
    config = _fake_config(
        dsv4_sm80_recipe="long_context_m4",
        max_running_req=8,
        max_running_req_explicit=True,
        cuda_graph_max_bs=2,
        context_length=262_144,
    )

    engine_module._adjust_config(config)

    assert config.max_running_req == 8
    assert config.cuda_graph_policy.resolved_max_bs == 2
    assert config.context_length == 262_144
    assert "Explicit settings override recipe fields" in warnings[0]
    assert "max_running_req=8" in warnings[0]
    assert "cuda_graph_max_bs=2" in warnings[0]
    assert "context_length=262144" in warnings[0]


def test_release_defaults_cap_graph_to_explicit_request_capacity(monkeypatch):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)
    config = _fake_config(max_running_req=32, max_running_req_explicit=True)

    engine_module._adjust_config(config)

    assert config.max_running_req == 32
    assert config.cuda_graph_policy.resolved_max_bs == 32

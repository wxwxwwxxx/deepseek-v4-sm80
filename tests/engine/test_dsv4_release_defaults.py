from __future__ import annotations

from types import SimpleNamespace

import pytest
from minisgl.dsv4_runtime import (
    configure_dsv4_runtime,
    get_dsv4_runtime_config,
    resolve_dsv4_runtime_config,
)
from minisgl.engine import engine as engine_module


def _fake_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=True, is_moe=True),
        dsv4_runtime_mode="optimized",
        disable_reasoning_sampler_contract=False,
        dsv4_sm80_recipe=None,
        max_running_req=256,
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
    assert config.max_running_req == 256
    assert config.cuda_graph_max_bs == 256
    assert config.cuda_graph_policy.source_mode == "explicit_max"
    assert config.cuda_graph_policy.resolved_bs[-1] == 256
    assert config.cuda_graph_capture_fail_open is True
    runtime = resolve_dsv4_runtime_config(config.dsv4_runtime_mode)
    assert runtime.moe_expert_backend == "marlin_wna16"
    assert runtime.direct_graph_metadata_groups == frozenset({"swa", "c4"})
    assert "c128" not in runtime.direct_graph_metadata_groups
    assert runtime.marlin_release_timing == "before_kv_alloc"
    assert runtime.clear_allocated_page_scope == "component"
    assert runtime.pynccl_max_buffer_bytes == 32 * 1024 * 1024


@pytest.mark.parametrize("backend", ["fa", "fi", "trtllm", "fa,fi"])
def test_removed_attention_backends_fail_instead_of_falling_back(monkeypatch, backend):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    config = _fake_config(attention_backend=backend)

    with pytest.raises(ValueError, match="DSV4 attention backend only"):
        engine_module._adjust_config(config)


def test_deepseek_v4_mode_is_selected_only_by_typed_config(monkeypatch):
    try:
        configure_dsv4_runtime("optimized")
        assert get_dsv4_runtime_config().optimized is True

        configure_dsv4_runtime("fallback")
        assert get_dsv4_runtime_config().optimized is False
    finally:
        configure_dsv4_runtime("optimized")


def test_deepseek_v4_typed_fallback_contract(monkeypatch):
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.logger, "info", lambda *args, **kwargs: None)
    config = _fake_config(dsv4_runtime_mode="fallback")

    engine_module._adjust_config(config)

    assert config.attention_backend == "dsv4"
    assert config.page_size == 256
    assert config.max_extend_tokens == 8192
    assert config.enable_dsv4_radix_prefix_cache is False
    assert config.enable_dsv4_component_loc_ownership is False
    assert config.allow_dsv4_cuda_graph is False
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0
    assert config.cuda_graph_policy.source_mode == "disabled"
    assert config.cuda_graph_policy.resolved_bs == ()
    assert config.use_pynccl is False
    assert config.cache_type == "naive"
    runtime = resolve_dsv4_runtime_config(config.dsv4_runtime_mode)
    assert runtime.moe_expert_backend == "grouped_fp4"
    assert runtime.release_raw_expert_weights is False
    assert runtime.marlin_prebuild is False


def test_optimized_explicit_contract_disable_emits_rank0_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        engine_module.logger,
        "warning_rank0",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    config = _fake_config(disable_reasoning_sampler_contract=True)

    engine_module._adjust_config(config)

    assert len(warnings) == 1
    assert "DISABLED" in warnings[0]
    assert "stop-with-empty-content" in warnings[0]
    assert "multiple </think>" in warnings[0]
    assert "CHAT mode" in warnings[0]


def test_fallback_logs_oracle_disable_without_warning(monkeypatch):
    infos = []
    warnings = []
    monkeypatch.setattr(
        engine_module.logger,
        "info_rank0",
        lambda message, *args, **kwargs: infos.append(message),
    )
    monkeypatch.setattr(
        engine_module.logger,
        "warning_rank0",
        lambda message, *args, **kwargs: warnings.append(message),
    )
    config = _fake_config(dsv4_runtime_mode="fallback")

    engine_module._adjust_config(config)

    assert warnings == []
    assert any("oracle logits and sampling distributions" in message for message in infos)


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
        ("dsv4_sm80_low_m64", 256, 64, None),
        ("dsv4_sm80_mid_m128", 256, 128, None),
        ("dsv4_sm80_balanced", 256, 256, None),
        ("dsv4_sm80_long_context_512k", 4, 4, 524_288),
        ("dsv4_sm80_1m_smoke", 1, 1, 1_048_576),
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
        dsv4_sm80_recipe="dsv4_sm80_long_context_512k",
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

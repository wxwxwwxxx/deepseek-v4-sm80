from __future__ import annotations

from types import SimpleNamespace

import torch

from minisgl.engine import engine as engine_module


def _fake_config(*, num_page_override=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=True),
        page_size=256,
        tp_info=SimpleNamespace(size=8),
        memory_ratio=1.0,
        num_page_override=num_page_override,
    )


def _fake_engine(*, source_bytes: int):
    engine = object.__new__(engine_module.Engine)
    engine.dtype = torch.bfloat16
    engine.model_prepare_report = {
        "moe_marlin_wna16_cache": {
            "total_source_bytes": source_bytes,
        },
    }
    engine._sync_get_memory = lambda: (4_000, 4_000)
    return engine


def test_marlin_wna16_before_kv_release_credit_adds_net_pages(monkeypatch):
    monkeypatch.setattr(engine_module, "estimate_kvcache_bytes_per_page", lambda *a, **k: 1024)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "1")
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT", "1")
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING", "before_kv_alloc")
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS", "1")
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_BYTES", "1024")

    engine = _fake_engine(source_bytes=4096)

    pages = engine._determine_num_pages(10_000, _fake_config())

    assert pages == 6
    credit = engine.kv_capacity_plan_report["release_credit"]
    assert credit["applied_to_num_pages"] is True
    assert credit["source_bytes"] == 4096
    assert credit["planned_guard_or_reserved_bytes"] == 1024
    assert credit["net_release_credit_bytes"] == 3072


def test_marlin_wna16_release_credit_rejects_after_kv_timing(monkeypatch):
    monkeypatch.setattr(engine_module, "estimate_kvcache_bytes_per_page", lambda *a, **k: 1024)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS", "1")
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT", "1")
    monkeypatch.setenv("MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING", "after_kv_alloc")

    engine = _fake_engine(source_bytes=4096)

    pages = engine._determine_num_pages(10_000, _fake_config())

    assert pages == 3
    credit = engine.kv_capacity_plan_report["release_credit"]
    assert credit["applied_to_num_pages"] is False
    assert credit["net_release_credit_bytes"] == 0
    assert credit["ineligible_reason"] == "release_timing_cannot_back_pre_kv_pages"

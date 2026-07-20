from __future__ import annotations

from types import SimpleNamespace

import torch
from minisgl.engine import engine as engine_module
from minisgl.engine.graph_memory import empty_graph_memory_estimate
from minisgl.engine.graph_policy import resolve_cuda_graph_bucket_policy


def _fake_config(*, num_page_override=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_v4=True),
        page_size=256,
        tp_info=SimpleNamespace(size=8),
        memory_ratio=1.0,
        num_page_override=num_page_override,
        max_seq_len=0,
        max_running_req=0,
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
    engine.graph_memory_estimate = empty_graph_memory_estimate()
    engine.cuda_graph_policy = resolve_cuda_graph_bucket_policy(
        cuda_graph_bs=[],
        cuda_graph_max_bs=None,
        effective_max_running_req=1,
    )
    return engine


def test_marlin_wna16_before_kv_release_credit_adds_net_pages(monkeypatch):
    monkeypatch.setattr(engine_module, "estimate_kvcache_bytes_per_page", lambda *a, **k: 1024)
    monkeypatch.setattr(engine_module.logger, "info_rank0", lambda *args, **kwargs: None)

    engine = _fake_engine(source_bytes=4096)

    pages = engine._determine_num_pages(10_000, _fake_config())

    assert pages == 7
    credit = engine.kv_capacity_plan_report["release_credit"]
    assert credit["applied_to_num_pages"] is True
    assert credit["source_bytes"] == 4096
    assert credit["planned_guard_or_reserved_bytes"] == 0
    assert credit["net_release_credit_bytes"] == 4096

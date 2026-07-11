from types import SimpleNamespace

from minisgl.utils import dsv4_long_prefill_timing


def test_long_prefill_timing_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MINISGL_DSV4_LONG_PREFILL_TIMING", raising=False)
    dsv4_long_prefill_timing.reset()
    dsv4_long_prefill_timing.record_host("prepare", 1.0, {"committed_context": 8192})
    dsv4_long_prefill_timing.record_counter("work", {"committed_context": 8192})
    snapshot = dsv4_long_prefill_timing.snapshot(resolve_cuda=False)
    assert snapshot["enabled"] is False
    assert snapshot["host_by_owner"] == {}
    assert snapshot["counters"] == []


def test_long_prefill_host_and_counter_checkpoint_contract(monkeypatch):
    monkeypatch.setenv("MINISGL_DSV4_LONG_PREFILL_TIMING", "1")
    dsv4_long_prefill_timing.reset()
    batch = SimpleNamespace(
        is_prefill=True,
        phase="prefill",
        size=1,
        padded_size=1,
        reqs=[SimpleNamespace(device_len=16384, extend_len=8192)],
        padded_reqs=[SimpleNamespace(device_len=16384, extend_len=8192)],
    )
    with dsv4_long_prefill_timing.batch_context(batch):
        dsv4_long_prefill_timing.record_host("prepare", 2.5)
        dsv4_long_prefill_timing.record_counter(
            "indexer_slice", {"logits_bytes": 1024}, value=2
        )
    snapshot = dsv4_long_prefill_timing.snapshot(resolve_cuda=False)
    assert snapshot["host_by_owner"]["prepare"]["total_ms"] == 2.5
    assert snapshot["host_by_committed_context"]["16384"]["prepare"]["count"] == 1
    assert snapshot["counters"] == [
        {
            "label": "indexer_slice",
            "metadata": {
                "phase": "prefill",
                "batch_size": 1,
                "padded_size": 1,
                "rows": 8192,
                "committed_context": 16384,
                "logits_bytes": 1024,
            },
            "value": 2,
        }
    ]

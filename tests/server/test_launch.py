from pathlib import Path


def test_scheduler_construction_preserves_tensor_version_counters():
    launch = (
        Path(__file__).resolve().parents[2] / "python/minisgl/server/launch.py"
    ).read_text(encoding="utf-8")
    assert "with torch.no_grad():\n        scheduler = Scheduler(args)" in launch
    assert "with torch.inference_mode():\n        scheduler = Scheduler(args)" not in launch

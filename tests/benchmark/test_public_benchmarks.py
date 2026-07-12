from __future__ import annotations

import asyncio
import builtins
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SCRIPTS = (
    ROOT / "benchmark/offline/bench.py",
    ROOT / "benchmark/offline/bench_wildchat.py",
    ROOT / "benchmark/online/bench_simple.py",
    ROOT / "benchmark/online/bench_qwen.py",
)
DEBUG_DIR = ROOT / "debug/dsv4/benchmark/offline"
DEFAULT_MODEL = "/models/DeepSeek-V4-Flash"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


offline = load_script("public_offline_bench", PUBLIC_SCRIPTS[0])
wildchat = load_script("public_wildchat_bench", PUBLIC_SCRIPTS[1])
simple = load_script("public_online_simple_bench", PUBLIC_SCRIPTS[2])
trace = load_script("public_online_trace_bench", PUBLIC_SCRIPTS[3])


@pytest.mark.parametrize("script", PUBLIC_SCRIPTS)
def test_public_benchmark_help(script: Path):
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_public_defaults_are_dsv4_release_defaults():
    offline_args = offline.parse_args([])
    wildchat_args = wildchat.parse_args([])
    simple_args = simple.parse_args([])
    trace_args = trace.parse_args([])

    assert offline_args.model == wildchat_args.model == DEFAULT_MODEL
    assert offline_args.tp_size == wildchat_args.tp_size == 8
    assert offline_args.recipe == wildchat_args.recipe == "dsv4_sm80_balanced"
    assert offline_args.runtime_mode == wildchat_args.runtime_mode == "optimized"
    assert simple_args.expected_model == trace_args.expected_model == DEFAULT_MODEL
    assert simple_args.base_url == trace_args.base_url == "http://127.0.0.1:1919/v1"
    assert str(wildchat_args.dataset_cache).startswith(str(Path.home() / ".cache"))
    assert str(trace_args.trace_cache).startswith(str(Path.home() / ".cache"))


@pytest.mark.parametrize("module", [offline, wildchat])
def test_offline_tp_world_size_mismatch_is_explicit(module, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(SystemExit, match=r"WORLD_SIZE=1 does not match --tp-size=8"):
        module.distributed_info_from_env(8)


class FakeModels:
    def list(self):
        async def models():
            yield SimpleNamespace(id="/models/not-deepseek-v4")

        return models()


class FakeClient:
    models = FakeModels()


@pytest.mark.parametrize("module", [simple, trace])
def test_online_model_mismatch_is_explicit(module):
    with pytest.raises(RuntimeError, match="server model mismatch"):
        asyncio.run(module.verify_server_model(FakeClient(), DEFAULT_MODEL))


def test_online_connection_failure_returns_nonzero():
    result = subprocess.run(
        [
            sys.executable,
            str(PUBLIC_SCRIPTS[2]),
            "--base-url",
            "http://127.0.0.1:1/v1",
            "--timeout",
            "0.2",
            "--request-count",
            "1",
            "--batch-size",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        env={**os.environ, "NO_PROXY": "127.0.0.1"},
    )
    assert result.returncode != 0
    assert "online simple benchmark failed" in result.stderr


def test_wildchat_missing_pyarrow_has_install_hint(monkeypatch):
    real_import = builtins.__import__

    def reject_pyarrow(name, *args, **kwargs):
        if name == "pyarrow.parquet":
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pyarrow)
    with pytest.raises(RuntimeError, match=r"\.\[benchmark\]"):
        wildchat._load_pyarrow_parquet()


def test_all_moved_debug_scripts_exist_and_compile():
    scripts = sorted(DEBUG_DIR.glob("deepseek_v4_*.py"))
    scripts += [DEBUG_DIR / "dsv4_graph_reserve_lifecycle.py"]
    assert len(scripts) == 20
    for script in scripts:
        assert script.is_file()
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
    assert not list((ROOT / "benchmark/offline").glob("deepseek_v4_*.py"))


def test_readme_referenced_repository_paths_exist():
    for relative in (
        "benchmark/offline/bench.py",
        "benchmark/offline/bench_wildchat.py",
        "benchmark/online/bench_simple.py",
        "benchmark/online/bench_qwen.py",
        "debug/dsv4/README.md",
        "examples/offline_dsv4.py",
        "docs/features.md",
        "docs/structures.md",
        "prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md",
    ):
        assert (ROOT / relative).exists(), relative

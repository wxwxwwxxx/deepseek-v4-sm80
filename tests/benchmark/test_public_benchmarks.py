from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from minisgl.distributed import launch_tensor_parallel

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


def _record_local_tp_rank(output_dir: str) -> None:
    import os

    rank = os.environ["LOCAL_RANK"]
    world_size = os.environ["WORLD_SIZE"]
    Path(output_dir, f"rank-{rank}").write_text(world_size, encoding="utf-8")


@pytest.mark.parametrize("script", PUBLIC_SCRIPTS)
def test_public_benchmark_is_a_small_compilable_example(script: Path):
    source = script.read_text(encoding="utf-8")
    compile(source, str(script), "exec")
    assert "argparse" not in source
    assert "dsv4_sm80_recipe" not in source


def test_public_defaults_point_to_dsv4_release_surfaces():
    assert offline.MODEL == wildchat.MODEL == DEFAULT_MODEL
    assert offline.TP_SIZE == wildchat.TP_SIZE == 8
    assert simple.PORT == trace.PORT == 1919
    assert simple.TOKENIZER == trace.TOKENIZER == DEFAULT_MODEL
    assert str(wildchat.CACHE_DIR).startswith(str(Path.home() / ".cache"))
    assert str(trace.TRACE_PATH).startswith(str(Path.home() / ".cache"))


def test_synthetic_benchmarks_use_fixed_saturation_workloads():
    prompts, sampling_params = offline.make_workload()

    assert offline.NUM_SEQS == 128
    assert {len(prompt) for prompt in prompts} == {offline.INPUT_LEN}
    assert {params.max_tokens for params in sampling_params} == {offline.OUTPUT_LEN}
    assert all(params.is_greedy and params.ignore_eos for params in sampling_params)
    assert simple.TEST_BATCH_SIZES == [64]
    assert simple.INPUT_LEN == simple.OUTPUT_LEN == 1024


def test_wildchat_uses_non_reasoning_dsv4_formatter_contract():
    calls = []

    class Tokenizer:
        @staticmethod
        def encode(prompt):
            return [prompt]

    def formatter(messages, reasoning_effort):
        calls.append((messages, reasoning_effort))
        return "formatted"

    messages = [{"role": "user", "content": "hello"}]
    assert wildchat.encode_messages(Tokenizer(), formatter, messages) == ["formatted"]
    assert calls == [(messages, None)]


def test_online_benchmark_does_not_send_internal_input_length_override():
    class Response:
        def __init__(self):
            self.done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.done:
                raise StopAsyncIteration
            self.done = True
            return object()

    class Completions:
        kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    async def run_benchmark():
        client = Client()
        result = await simple.benchmark_one(
            client,
            "hello",
            2,
            "DeepSeek-V4-Flash",
            pbar=False,
            input_length=128,
        )
        return client, result

    client, result = asyncio.run(run_benchmark())

    assert result.input_len == 128
    assert client.chat.completions.kwargs["extra_body"] == {
        "ignore_eos": True,
        "top_k": 1,
    }


@pytest.mark.parametrize("module", [offline, wildchat])
def test_offline_examples_use_framework_tp_launcher(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "launch_tensor_parallel(TP_SIZE, main)" in source
    assert "WORLD_SIZE" not in source
    assert "LOCAL_RANK" not in source


def test_framework_tp_launcher_spawns_all_local_ranks(tmp_path):
    launch_tensor_parallel(2, _record_local_tp_rank, str(tmp_path))
    assert (tmp_path / "rank-0").read_text(encoding="utf-8") == "2"
    assert (tmp_path / "rank-1").read_text(encoding="utf-8") == "2"


def test_llm_uses_launcher_environment(monkeypatch):
    from minisgl.llm.llm import LLM, Scheduler

    configs = []
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(Scheduler, "__init__", lambda self, config: configs.append(config))

    LLM("/models/DeepSeek-V4-Flash")

    assert configs[0].tp_info.rank == 3
    assert configs[0].tp_info.size == 8
    assert configs[0].distributed_init_method == "env://"


def test_all_moved_debug_scripts_exist_and_compile():
    scripts = sorted(DEBUG_DIR.glob("deepseek_v4_*.py"))
    scripts += [DEBUG_DIR / "dsv4_graph_reserve_lifecycle.py"]
    assert len(scripts) == 18
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
    ):
        assert (ROOT / relative).exists(), relative

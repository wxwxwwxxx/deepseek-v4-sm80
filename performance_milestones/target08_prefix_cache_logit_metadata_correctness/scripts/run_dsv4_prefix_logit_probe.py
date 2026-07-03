#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")
os.environ.setdefault("MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE", "1")


@dataclass(frozen=True)
class ProbeScenario:
    name: str
    description: str
    warm_prompts: list[list[int]]
    probe_prompts: list[list[int]]
    expected_cached_lens: list[int]
    coverage: list[str]


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _git_info() -> dict[str, Any]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
        except Exception as exc:
            return f"<error: {type(exc).__name__}: {exc}>"

    return {
        "rev": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _tp_rank_size(args: argparse.Namespace) -> tuple[int, int, int]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
    tp_size = args.tensor_parallel_size or env_world_size
    tp_rank = args.tp_rank if args.tp_rank is not None else env_local_rank
    if env_world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={env_world_size} does not match tensor parallel size {tp_size}; "
            "launch with torchrun --standalone --nproc_per_node=8."
        )
    return tp_rank, tp_size, env_world_size


def _distributed_init_method(args: argparse.Namespace, tp_size: int) -> str | None:
    if args.distributed_init_method is not None:
        return args.distributed_init_method
    if tp_size > 1 and "MASTER_ADDR" in os.environ:
        return "env://"
    return None


def _dtype_from_name(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _random_tokens(
    rng: random.Random,
    length: int,
    *,
    vocab_size: int,
    token_id_range: int,
    marker: int,
) -> list[int]:
    low = 10 if vocab_size > 64 else 1
    high = min(max(low, int(token_id_range)), max(vocab_size - 1, low))
    tokens = [rng.randint(low, high) for _ in range(length)]
    if tokens:
        tokens[0] = min(high, low + marker)
    return tokens


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    return limit


def _expected_cached_len(probe: list[int], warm_prompts: list[list[int]], page_size: int) -> int:
    best = 0
    matchable_probe_len = max(len(probe) - 1, 0)
    for warm in warm_prompts:
        warm_insert_len = _align_down(len(warm), page_size)
        common = min(_common_prefix_len(probe, warm), matchable_probe_len)
        best = max(best, min(warm_insert_len, _align_down(common, page_size)))
    return best


def _make_builder(
    name: str,
    description: str,
    coverage: list[str],
    fn: Callable[[random.Random, int, int, int, int], tuple[list[list[int]], list[list[int]]]],
) -> Callable[[int, int, int, int], ProbeScenario]:
    def build(vocab_size: int, token_id_range: int, seed: int, page_size: int) -> ProbeScenario:
        scenario_index = _SCENARIO_ORDER.index(name)
        rng = random.Random(seed + scenario_index * 9973)
        warm, probe = fn(rng, vocab_size, token_id_range, page_size, scenario_index)
        expected = [_expected_cached_len(item, warm, page_size) for item in probe]
        return ProbeScenario(
            name=name,
            description=description,
            warm_prompts=warm,
            probe_prompts=probe,
            expected_cached_lens=expected,
            coverage=coverage,
        )

    return build


def _single_full_hit(
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    scenario_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    base = _random_tokens(
        rng,
        page_size + 1,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=10 + scenario_index * 17,
    )
    return [base], [list(base)]


def _single_partial_c128(
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    scenario_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    base = _random_tokens(
        rng,
        page_size + 1,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=10 + scenario_index * 17,
    )
    suffix = _random_tokens(
        rng,
        512,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=11 + scenario_index * 17,
    )
    return [base], [base + suffix]


def _identical_slots(
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    scenario_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    base = _random_tokens(
        rng,
        page_size + 1,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=10 + scenario_index * 17,
    )
    return [base], [list(base) for _ in range(4)]


def _mixed_hit_miss(
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    scenario_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    base = _random_tokens(
        rng,
        page_size + 1,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=10 + scenario_index * 17,
    )
    misses = [
        _random_tokens(
            rng,
            page_size + 1,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=200 + scenario_index * 17 + idx,
        )
        for idx in range(4)
    ]
    probe = []
    for miss in misses:
        probe.append(list(base))
        probe.append(miss)
    return [base], probe


def _swa_boundary_no_hit(
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    scenario_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    del page_size
    probe = [
        _random_tokens(
            rng,
            length,
            vocab_size=vocab_size,
            token_id_range=token_id_range,
            marker=10 + scenario_index * 17 + idx,
        )
        for idx, length in enumerate((127, 128, 129))
    ]
    return [], probe


def _c4_boundary_partial(
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    scenario_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    base = _random_tokens(
        rng,
        page_size + 1,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=10 + scenario_index * 17,
    )
    suffix = _random_tokens(
        rng,
        4,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=11 + scenario_index * 17,
    )
    return [base], [base + suffix]


def _page_boundary_mixed(
    rng: random.Random,
    vocab_size: int,
    token_id_range: int,
    page_size: int,
    scenario_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    base = _random_tokens(
        rng,
        page_size + 2,
        vocab_size=vocab_size,
        token_id_range=token_id_range,
        marker=10 + scenario_index * 17,
    )
    return [base], [base[: page_size - 1], base[:page_size], base[: page_size + 1], base]


_SCENARIO_BUILDERS = {
    "single_full_hit_page257": _make_builder(
        "single_full_hit_page257",
        "One warm 257-token request, then one identical request hits the retained 256-token page.",
        ["single-request full hit", "page boundary around 256"],
        _single_full_hit,
    ),
    "single_partial_hit_769_c128": _make_builder(
        "single_partial_hit_769_c128",
        "One warm 257-token request, then a 769-token request reuses only page 0 and crosses C128.",
        ["single-request partial hit", "C128 boundary around 128"],
        _single_partial_c128,
    ),
    "identical_prompts_batch_slots": _make_builder(
        "identical_prompts_batch_slots",
        "One warm 257-token request, then four identical probe prompts in separate batch slots.",
        ["identical prompts in batch slots", "single-request full hit"],
        _identical_slots,
    ),
    "mixed_hit_miss_batch": _make_builder(
        "mixed_hit_miss_batch",
        "One warm 257-token request, then interleaved full-hit and unrelated miss prompts.",
        ["mixed hit/miss batch"],
        _mixed_hit_miss,
    ),
    "swa_boundary_127_128_129_no_hit": _make_builder(
        "swa_boundary_127_128_129_no_hit",
        "Prefix-enabled miss path for 127/128/129-token prompts, centered on the SWA window.",
        ["SWA boundary around 128", "prefix-disabled equivalent miss path"],
        _swa_boundary_no_hit,
    ),
    "c4_boundary_partial261": _make_builder(
        "c4_boundary_partial261",
        "One warm 257-token request, then a 261-token request whose suffix crosses a C4 store point.",
        ["C4 boundary around 4"],
        _c4_boundary_partial,
    ),
    "page_boundary_255_256_257_258": _make_builder(
        "page_boundary_255_256_257_258",
        "One warm 258-token request, then probe lengths 255/256/257/258 around page matching.",
        ["page boundary around 256", "mixed hit/miss batch"],
        _page_boundary_mixed,
    ),
}
_SCENARIO_ORDER = list(_SCENARIO_BUILDERS)


def _build_scenarios(args: argparse.Namespace, vocab_size: int) -> list[ProbeScenario]:
    names = args.scenarios or _SCENARIO_ORDER
    return [
        _SCENARIO_BUILDERS[name](
            vocab_size,
            args.token_id_range,
            args.seed,
            args.page_size,
        )
        for name in names
    ]


def _sampling(max_tokens: int):
    from minisgl.core import SamplingParams

    return SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens)


def _set_debug_context(*, mode: str, scenario: str, stage: str) -> None:
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_MODE"] = mode
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_SCENARIO"] = scenario
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_STAGE"] = stage


def _prefix_metrics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, bool) or isinstance(old, bool):
            continue
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            delta[key] = value - old
    return delta


def _run_scenario(llm: Any, args: argparse.Namespace, scenario: ProbeScenario) -> dict[str, Any]:
    import torch

    before = llm.cache_manager.prefix_metrics_snapshot()
    warm_outputs: list[dict[str, Any]] = []
    probe_outputs: list[dict[str, Any]] = []
    tic = time.perf_counter()
    if scenario.warm_prompts:
        _set_debug_context(mode=args.mode, scenario=scenario.name, stage="warm")
        warm_outputs = llm.generate(
            scenario.warm_prompts,
            [_sampling(args.warm_max_tokens) for _ in scenario.warm_prompts],
        )
        torch.cuda.synchronize(llm.device)
    _set_debug_context(mode=args.mode, scenario=scenario.name, stage="probe")
    probe_outputs = llm.generate(
        scenario.probe_prompts,
        [_sampling(args.probe_max_tokens) for _ in scenario.probe_prompts],
    )
    torch.cuda.synchronize(llm.device)
    elapsed_s = time.perf_counter() - tic
    after = llm.cache_manager.prefix_metrics_snapshot()
    return {
        "name": scenario.name,
        "description": scenario.description,
        "coverage": scenario.coverage,
        "warm_prompt_lens": [len(prompt) for prompt in scenario.warm_prompts],
        "probe_prompt_lens": [len(prompt) for prompt in scenario.probe_prompts],
        "expected_cached_lens": scenario.expected_cached_lens,
        "warm_outputs": warm_outputs,
        "probe_outputs": probe_outputs,
        "prefix_cache_metrics_before": before,
        "prefix_cache_metrics_after": after,
        "prefix_cache_metrics_delta": _prefix_metrics_delta(before, after),
        "elapsed_s": elapsed_s,
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from minisgl.distributed import DistributedInfo
    from minisgl.llm.llm import LLM

    rank, tp_size, _ = _tp_rank_size(args)
    distributed_init_method = _distributed_init_method(args, tp_size)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug_trace"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_DIR"] = str(debug_dir)
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_TOPK"] = str(args.topk)
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_SAVE_FULL_LOGITS"] = "1"
    os.environ["MINISGL_DSV4_PREFIX_DEBUG_MODE"] = args.mode

    kwargs: dict[str, Any] = {}
    if distributed_init_method is not None:
        kwargs["distributed_init_method"] = distributed_init_method

    allow_graph = bool(args.allow_dsv4_cuda_graph)
    cuda_graph_bs = args.cuda_graph_bs if allow_graph else []
    dtype = _dtype_from_name(args.dtype)
    llm = None
    failures = 0
    try:
        load_tic = time.perf_counter()
        llm = LLM(
            args.model_path,
            dtype=dtype,
            tp_info=DistributedInfo(rank, tp_size),
            max_running_req=args.max_running_req,
            max_seq_len_override=args.max_seq_len,
            max_extend_tokens=args.max_extend_tokens,
            num_page_override=args.num_pages,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            allow_dsv4_cuda_graph=allow_graph,
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_capture_greedy_sample=False,
            enable_dsv4_radix_prefix_cache=args.mode != "prefix_off",
            **kwargs,
        )
        torch.cuda.synchronize(llm.device)
        load_s = time.perf_counter() - load_tic
        scenarios = _build_scenarios(args, llm.engine.sampler.vocab_size)
        scenario_reports = []
        for scenario in scenarios:
            scenario_reports.append(_run_scenario(llm, args, scenario))
            llm.sync_all_ranks()

        if rank == 0:
            payload = {
                "status": "pass",
                "mode": args.mode,
                "model_path": args.model_path,
                "git": _git_info(),
                "config": {
                    "tensor_parallel_size": tp_size,
                    "distributed_init_method": distributed_init_method,
                    "dtype": args.dtype,
                    "page_size": args.page_size,
                    "num_pages": args.num_pages,
                    "max_seq_len": args.max_seq_len,
                    "max_extend_tokens": args.max_extend_tokens,
                    "max_running_req": args.max_running_req,
                    "allow_dsv4_cuda_graph": allow_graph,
                    "cuda_graph_bs": cuda_graph_bs,
                    "enable_dsv4_radix_prefix_cache": args.mode != "prefix_off",
                    "probe_max_tokens": args.probe_max_tokens,
                    "warm_max_tokens": args.warm_max_tokens,
                    "token_id_range": args.token_id_range,
                    "seed": args.seed,
                    "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}),
                    "model_prepare_report_rank0": getattr(llm.engine, "model_prepare_report", {}),
                },
                "load_seconds": load_s,
                "scenarios": scenario_reports,
                "final_prefix_cache_metrics": llm.cache_manager.prefix_metrics_snapshot(),
            }
            with (output_dir / "run.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(json.dumps({"run_json": str(output_dir / "run.json"), "status": "pass"}))
    except BaseException as exc:
        failures += 1
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "status": "fail",
                "mode": args.mode,
                "error": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(limit=40),
                },
                "git": _git_info(),
            }
            with (output_dir / "run.json").open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            traceback.print_exc()
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException:
                if rank == 0:
                    traceback.print_exc()
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TARGET 08.19 DSV4 prefix-cache logits probe.")
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--mode", choices=("prefix_off", "prefix_on", "prefix_on_eager"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="*", choices=tuple(_SCENARIO_ORDER))
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-extend-tokens", type=int, default=20000)
    parser.add_argument("--max-running-req", type=int, default=16)
    parser.add_argument("--allow-dsv4-cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-bs", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--warm-max-tokens", type=int, default=1)
    parser.add_argument("--probe-max-tokens", type=int, default=2)
    parser.add_argument("--token-id-range", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=819)
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args(argv)
    if args.list_scenarios:
        for name, builder in _SCENARIO_BUILDERS.items():
            del builder
            print(name)
        raise SystemExit(0)
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.num_pages <= 1:
        parser.error("--num-pages must be greater than 1")
    if args.max_seq_len <= 0 or args.max_extend_tokens <= 0 or args.max_running_req <= 0:
        parser.error("max sequence/running settings must be positive")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.warm_max_tokens <= 0 or args.probe_max_tokens <= 1:
        parser.error("--warm-max-tokens must be positive and --probe-max-tokens must be > 1")
    args.cuda_graph_bs = sorted(set(args.cuda_graph_bs or []))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(parse_args(argv)))


if __name__ == "__main__":
    main()

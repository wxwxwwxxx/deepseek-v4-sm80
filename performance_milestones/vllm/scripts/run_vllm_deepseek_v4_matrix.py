#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    batch_size: int
    prompt_len: int
    decode_len: int
    description: str
    repeats: int = 3
    warmup_repeats: int = 1
    shared_prefix_len: int = 0
    suffix_len: int = 0

    @property
    def max_input_len(self) -> int:
        if self.kind == "shared_prefix":
            return self.shared_prefix_len + self.suffix_len
        return self.prompt_len

    @property
    def max_seq_len(self) -> int:
        return self.max_input_len + self.decode_len


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="long_prefill_bs1",
        kind="random",
        batch_size=1,
        prompt_len=4096,
        decode_len=1,
        description="Single-request long prefill with one generated token.",
    ),
    Scenario(
        name="batch_prefill_bs8",
        kind="random",
        batch_size=8,
        prompt_len=1024,
        decode_len=1,
        description="Batch prefill with one generated token per request.",
    ),
    Scenario(
        name="decode_throughput_bs8",
        kind="random",
        batch_size=8,
        prompt_len=128,
        decode_len=64,
        description="Decode-heavy batch throughput workload.",
    ),
    Scenario(
        name="mixed_prefill_decode_bs4",
        kind="mixed_prefill_decode",
        batch_size=4,
        prompt_len=1024,
        decode_len=32,
        description="Varied prompt lengths and decode budgets.",
    ),
    Scenario(
        name="shared_prompt_no_radix_bs8",
        kind="shared_prefix",
        batch_size=8,
        prompt_len=1088,
        decode_len=16,
        shared_prefix_len=1024,
        suffix_len=64,
        description="Repeated shared prompt with prefix caching disabled.",
    ),
)


def _scenario_map() -> dict[str, Scenario]:
    return {scenario.name: scenario for scenario in DEFAULT_SCENARIOS}


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(cwd: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def git_info(cwd: Path) -> dict[str, Any]:
    status = _git_output(cwd, ["status", "--short"]) or ""
    return {
        "root": str(cwd),
        "branch": _git_output(cwd, ["branch", "--show-current"]),
        "commit": _git_output(cwd, ["rev-parse", "HEAD"]),
        "short_commit": _git_output(cwd, ["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _select_scenarios(args: argparse.Namespace) -> list[Scenario]:
    if args.smoke:
        return [
            Scenario(
                name="smoke_debug",
                kind="random",
                batch_size=args.batch_size or 1,
                prompt_len=args.prompt_len or 16,
                decode_len=args.decode_len or 2,
                repeats=args.repeats if args.repeats is not None else 1,
                warmup_repeats=args.warmup_repeats
                if args.warmup_repeats is not None
                else 0,
                description="Tiny smoke/debug workload.",
            )
        ]

    selected = args.scenarios or [scenario.name for scenario in DEFAULT_SCENARIOS]
    scenario_map = _scenario_map()
    scenarios = [scenario_map[name] for name in selected]
    output = []
    for scenario in scenarios:
        overrides: dict[str, int] = {}
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        if args.prompt_len is not None:
            overrides["prompt_len"] = args.prompt_len
            if scenario.kind == "shared_prefix":
                overrides["shared_prefix_len"] = max(args.prompt_len - scenario.suffix_len, 1)
        if args.decode_len is not None:
            overrides["decode_len"] = args.decode_len
        if args.repeats is not None:
            overrides["repeats"] = args.repeats
        if args.warmup_repeats is not None:
            overrides["warmup_repeats"] = args.warmup_repeats
        output.append(replace(scenario, **overrides))
    return output


def _parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        return None
    return [int(item) for item in values]


def _max_running_req(scenarios: Sequence[Scenario]) -> int:
    return max((scenario.batch_size for scenario in scenarios), default=1)


def _max_seq_len(scenarios: Sequence[Scenario]) -> int:
    return max((scenario.max_seq_len for scenario in scenarios), default=1)


def _max_extend_tokens(scenarios: Sequence[Scenario]) -> int:
    return max((scenario.batch_size * scenario.max_input_len for scenario in scenarios), default=1)


def _actual_scheduler_config(llm) -> dict[str, Any]:
    try:
        scheduler_config = llm.llm_engine.vllm_config.scheduler_config
    except Exception:
        return {}
    keys = (
        "max_num_batched_tokens",
        "max_num_scheduled_tokens",
        "max_num_seqs",
        "enable_chunked_prefill",
        "max_num_partial_prefills",
        "max_long_partial_prefills",
        "long_prefill_token_threshold",
    )
    return {key: getattr(scheduler_config, key, None) for key in keys}


def _random_tokens(
    rng: random.Random,
    length: int,
    vocab_size: int,
    *,
    token_id_range: int,
) -> list[int]:
    low = 10 if vocab_size > 64 else 1
    high = min(max(low, int(token_id_range)), max(vocab_size - 1, low))
    usable = max(high - low + 1, 1)
    return [low + rng.randrange(usable) for _ in range(length)]


def build_workload(
    scenario: Scenario,
    *,
    vocab_size: int,
    seed: int,
    token_id_range: int,
):
    from vllm import SamplingParams

    rng = random.Random(seed)
    prompts: list[list[int]] = []
    output_lens: list[int] = []

    if scenario.kind == "shared_prefix":
        prefix = _random_tokens(
            rng,
            scenario.shared_prefix_len,
            vocab_size,
            token_id_range=token_id_range,
        )
        for _ in range(scenario.batch_size):
            suffix = _random_tokens(
                rng,
                scenario.suffix_len,
                vocab_size,
                token_id_range=token_id_range,
            )
            prompts.append(prefix + suffix)
            output_lens.append(scenario.decode_len)
    elif scenario.kind == "mixed_prefill_decode":
        min_prompt_len = max(1, scenario.prompt_len // 4)
        min_decode_len = max(1, scenario.decode_len // 4)
        for idx in range(scenario.batch_size):
            frac = idx / max(scenario.batch_size - 1, 1)
            prompt_len = int(round(min_prompt_len + frac * (scenario.prompt_len - min_prompt_len)))
            decode_len = int(round(min_decode_len + frac * (scenario.decode_len - min_decode_len)))
            prompts.append(
                _random_tokens(
                    rng,
                    prompt_len,
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(max(1, decode_len))
    else:
        for _ in range(scenario.batch_size):
            prompts.append(
                _random_tokens(
                    rng,
                    scenario.prompt_len,
                    vocab_size,
                    token_id_range=token_id_range,
                )
            )
            output_lens.append(scenario.decode_len)

    sampling_params = [
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=output_len,
            detokenize=False,
        )
        for output_len in output_lens
    ]
    return prompts, sampling_params


def _safe_mean(values: Sequence[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.mean(filtered))


def _safe_median(values: Sequence[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _get_vocab_size(llm) -> int:
    tokenizer = llm.get_tokenizer()
    for attr in ("vocab_size", "n_words"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
        if isinstance(vocab, dict) and vocab:
            return max(vocab.values()) + 1
    return 102400


def _cuda_sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _nvtx_push(torch, name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_pop(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def run_repeat(
    *,
    llm,
    torch,
    scenario: Scenario,
    vocab_size: int,
    seed: int,
    token_id_range: int,
    repeat_index: int,
) -> dict[str, Any]:
    prompts, sampling_params = build_workload(
        scenario,
        vocab_size=vocab_size,
        seed=seed,
        token_id_range=token_id_range,
    )
    prompt_tokens = int(sum(len(prompt) for prompt in prompts))
    target_output_tokens = int(sum(param.max_tokens or 0 for param in sampling_params))
    _cuda_sync(torch)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    _nvtx_push(torch, f"repeat:{scenario.name}:{repeat_index}")
    tic = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    _cuda_sync(torch)
    elapsed_s = time.perf_counter() - tic
    _nvtx_pop(torch)

    output_lens = [len(output.outputs[0].token_ids) for output in outputs]
    return {
        "repeat_index": repeat_index,
        "elapsed_s": elapsed_s,
        "prompt_tokens": prompt_tokens,
        "target_output_tokens": target_output_tokens,
        "actual_output_tokens": int(sum(output_lens)),
        "output_lens": output_lens,
        "sample_output_token_ids": [
            list(output.outputs[0].token_ids[:16]) for output in outputs[:2]
        ],
        "memory": {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None,
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved())
            if torch.cuda.is_available()
            else None,
        },
    }


def summarize_case(
    *,
    scenario: Scenario,
    repeats: Sequence[dict[str, Any]],
    warmup: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    elapsed_values = [float(repeat["elapsed_s"]) for repeat in repeats]
    elapsed_s = float(sum(elapsed_values))
    prompt_tokens = int(sum(repeat["prompt_tokens"] for repeat in repeats))
    actual_output_tokens = int(sum(repeat["actual_output_tokens"] for repeat in repeats))
    target_output_tokens = int(sum(repeat["target_output_tokens"] for repeat in repeats))
    peak_allocated = max(
        (repeat.get("memory", {}).get("max_memory_allocated_bytes") or 0)
        for repeat in repeats
    )
    peak_reserved = max(
        (repeat.get("memory", {}).get("max_memory_reserved_bytes") or 0)
        for repeat in repeats
    )
    metrics = {
        "elapsed_s": elapsed_s,
        "elapsed_s_mean": _safe_mean(elapsed_values),
        "elapsed_s_median": _safe_median(elapsed_values),
        "prompt_tokens": prompt_tokens,
        "actual_output_tokens": actual_output_tokens,
        "target_output_tokens": target_output_tokens,
        "end_to_end_output_tokens_per_s": None
        if elapsed_s <= 0
        else actual_output_tokens / elapsed_s,
        "end_to_end_total_tokens_per_s": None
        if elapsed_s <= 0
        else (prompt_tokens + actual_output_tokens) / elapsed_s,
        "peak_gpu_memory_allocated_bytes": int(peak_allocated),
        "peak_gpu_memory_reserved_bytes": int(peak_reserved),
    }
    return {
        "status": "pass",
        "scenario": asdict(scenario),
        "report_path": str(report_path),
        "metrics": metrics,
        "warmup": warmup,
        "repeats": list(repeats),
    }


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    scenario = report.get("scenario", {})
    return {
        "status": report.get("status"),
        "scenario": scenario.get("name"),
        "report_path": report.get("report_path"),
        "elapsed_s": metrics.get("elapsed_s"),
        "elapsed_s_mean": metrics.get("elapsed_s_mean"),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "actual_output_tokens": metrics.get("actual_output_tokens"),
        "end_to_end_output_tokens_per_s": metrics.get(
            "end_to_end_output_tokens_per_s"
        ),
        "end_to_end_total_tokens_per_s": metrics.get(
            "end_to_end_total_tokens_per_s"
        ),
        "peak_gpu_memory_allocated_bytes": metrics.get(
            "peak_gpu_memory_allocated_bytes"
        ),
    }


def collect_environment(torch, args: argparse.Namespace) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda: dict[str, Any] = {"available": cuda_available}
    if cuda_available:
        cap = torch.cuda.get_device_capability(0)
        cuda.update(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "capability": [int(cap[0]), int(cap[1])],
                "capability_name": f"sm{cap[0]}{cap[1]}",
                "runtime": torch.version.cuda,
            }
        )
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            "torch": torch.__version__,
            "vllm": _dist_version("vllm"),
            "triton": _dist_version("triton"),
            "flashinfer-python": _dist_version("flashinfer-python"),
            "flashinfer-cubin": _dist_version("flashinfer-cubin"),
            "tilelang": _dist_version("tilelang"),
        },
        "cuda": cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "args": vars(args),
    }


def run_matrix(args: argparse.Namespace) -> int:
    import torch
    from vllm import LLM

    scenarios = _select_scenarios(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "matrix.jsonl"
    if matrix_path.exists():
        matrix_path.unlink()

    max_model_len = args.max_model_len or _max_seq_len(scenarios)
    max_num_seqs = args.max_num_seqs or _max_running_req(scenarios)
    engine_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "tokenizer_mode": args.tokenizer_mode,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "block_size": args.block_size,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "seed": args.seed,
    }
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.enable_chunked_prefill is not None:
        engine_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.max_num_partial_prefills is not None:
        engine_kwargs["max_num_partial_prefills"] = args.max_num_partial_prefills
    if args.max_long_partial_prefills is not None:
        engine_kwargs["max_long_partial_prefills"] = args.max_long_partial_prefills
    if args.long_prefill_token_threshold is not None:
        engine_kwargs["long_prefill_token_threshold"] = args.long_prefill_token_threshold
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization

    compilation_config: dict[str, Any] = {}
    cudagraph_capture_sizes = _parse_int_csv(args.cudagraph_capture_sizes)
    if cudagraph_capture_sizes is not None:
        compilation_config["cudagraph_capture_sizes"] = cudagraph_capture_sizes
    if args.max_cudagraph_capture_size is not None:
        compilation_config["max_cudagraph_capture_size"] = args.max_cudagraph_capture_size
    if args.compilation_config_json:
        compilation_config.update(json.loads(args.compilation_config_json))
    if compilation_config:
        engine_kwargs["compilation_config"] = compilation_config

    if args.dry_run:
        print(
            json.dumps(
                {
                    "engine_kwargs": engine_kwargs,
                    "scenarios": [asdict(scenario) for scenario in scenarios],
                    "output_dir": str(output_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    load_tic = time.perf_counter()
    llm = LLM(**engine_kwargs)
    _cuda_sync(torch)
    load_s = time.perf_counter() - load_tic
    vocab_size = _get_vocab_size(llm)
    actual_scheduler_config = _actual_scheduler_config(llm)

    environment = collect_environment(torch, args)
    root = Path(args.vllm_root).resolve()
    run_config = {
        "model_path": args.model_path,
        "vllm_root": str(root),
        "git": git_info(root) if root.exists() else None,
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "engine_kwargs": engine_kwargs,
        "actual_scheduler_config": actual_scheduler_config,
        "load_init": {"seconds": load_s},
        "vocab_size": vocab_size,
        "runtime_environment": environment,
        "notes": [
            "Prompts are synthetic token-id lists matching mini-sglang's "
            "deepseek_v4_perf_matrix.py workload shapes.",
            "Offline vLLM generate returns full outputs, so this runner reports "
            "throughput but not per-request TTFT/TPOT.",
        ],
    }
    _write_json(output_dir / "run_config.json", run_config)

    reports: list[dict[str, Any]] = []
    for case_index, scenario in enumerate(scenarios):
        case_name = f"{case_index:03d}_{scenario.name}__vllm"
        report_path = output_dir / "reports" / f"{case_name}.json"
        warmup_elapsed = []
        _nvtx_push(torch, f"case:{scenario.name}")
        for warmup_idx in range(scenario.warmup_repeats):
            repeat = run_repeat(
                llm=llm,
                torch=torch,
                scenario=scenario,
                vocab_size=vocab_size,
                seed=args.seed + 100000 + case_index * 1000 + warmup_idx,
                token_id_range=args.token_id_range,
                repeat_index=-(warmup_idx + 1),
            )
            warmup_elapsed.append(float(repeat["elapsed_s"]))
        warmup = {
            "repeats": scenario.warmup_repeats,
            "elapsed_s": warmup_elapsed,
            "total_elapsed_s": float(sum(warmup_elapsed)),
        }

        repeats = []
        for repeat_idx in range(scenario.repeats):
            repeats.append(
                run_repeat(
                    llm=llm,
                    torch=torch,
                    scenario=scenario,
                    vocab_size=vocab_size,
                    seed=args.seed + case_index * 1000 + repeat_idx,
                    token_id_range=args.token_id_range,
                    repeat_index=repeat_idx,
                )
            )
        _nvtx_pop(torch)
        report = summarize_case(
            scenario=scenario,
            repeats=repeats,
            warmup=warmup,
            report_path=report_path,
        )
        report.update(
            {
                "case_name": case_name,
                "model_path": args.model_path,
                "config": {
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "block_size": args.block_size,
                    "max_model_len": max_model_len,
                    "max_num_seqs": max_num_seqs,
                    "requested_max_num_batched_tokens": args.max_num_batched_tokens,
                    "actual_scheduler_config": actual_scheduler_config,
                    "token_id_range": args.token_id_range,
                    "enable_prefix_caching": args.enable_prefix_caching,
                    "disable_custom_all_reduce": args.disable_custom_all_reduce,
                    "enforce_eager": args.enforce_eager,
                },
                "runtime_environment": environment,
            }
        )
        _write_json(report_path, report)
        _append_jsonl(matrix_path, summary_row(report))
        reports.append(report)

    summary = [summary_row(report) for report in reports]
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {"summary_path": str(output_dir / "summary.json"), "cases": summary},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM DeepSeek V4 with mini-sglang perf-matrix workloads."
    )
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--vllm-root", default="/workspace/vllm-dsv4-docker")
    parser.add_argument("--output-dir", default="/tmp/dsv4_vllm_matrix_tp8")
    parser.add_argument("--scenarios", nargs="*", choices=tuple(_scenario_map()))
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-partial-prefills", type=int, default=None)
    parser.add_argument("--max-long-partial-prefills", type=int, default=None)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=None)
    parser.add_argument("--prompt-len", type=int, default=None)
    parser.add_argument("--decode-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup-repeats", type=int, default=None)
    parser.add_argument("--token-id-range", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    chunked_prefill = parser.add_mutually_exclusive_group()
    chunked_prefill.add_argument(
        "--enable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_true",
        default=None,
    )
    chunked_prefill.add_argument(
        "--disable-chunked-prefill",
        dest="enable_chunked_prefill",
        action="store_false",
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        default=None,
        help="Comma-separated vLLM CUDA graph capture sizes, for example 1,2,4.",
    )
    parser.add_argument("--max-cudagraph-capture-size", type=int, default=None)
    parser.add_argument(
        "--compilation-config-json",
        default=None,
        help="JSON object merged into vLLM compilation_config.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args(argv)

    for name in (
        "tensor_parallel_size",
        "block_size",
        "token_id_range",
        "prompt_len",
        "decode_len",
        "batch_size",
        "repeats",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_num_partial_prefills",
        "max_long_partial_prefills",
        "max_cudagraph_capture_size",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.long_prefill_token_threshold is not None and args.long_prefill_token_threshold < 0:
        parser.error("--long-prefill-token-threshold must be non-negative")
    try:
        cudagraph_capture_sizes = _parse_int_csv(args.cudagraph_capture_sizes)
    except ValueError as exc:
        parser.error(f"--cudagraph-capture-sizes must be comma-separated integers: {exc}")
    if cudagraph_capture_sizes is not None and any(size <= 0 for size in cudagraph_capture_sizes):
        parser.error("--cudagraph-capture-sizes values must be positive")
    if args.compilation_config_json:
        try:
            parsed_compilation_config = json.loads(args.compilation_config_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--compilation-config-json is not valid JSON: {exc}")
        if not isinstance(parsed_compilation_config, dict):
            parser.error("--compilation-config-json must decode to a JSON object")
    if args.warmup_repeats is not None and args.warmup_repeats < 0:
        parser.error("--warmup-repeats must be non-negative")
    if args.gpu_memory_utilization <= 0:
        parser.error("--gpu-memory-utilization must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_scenarios:
        for scenario in DEFAULT_SCENARIOS:
            print(f"{scenario.name}\t{scenario.kind}\t{scenario.description}")
        return
    raise SystemExit(run_matrix(args))


if __name__ == "__main__":
    main()

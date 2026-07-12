from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL = "/models/DeepSeek-V4-Flash"
DEFAULT_RECIPE = "dsv4_sm80_balanced"
PUBLIC_RECIPES = (
    "dsv4_sm80_low_m64",
    "dsv4_sm80_mid_m128",
    "dsv4_sm80_balanced",
    "dsv4_sm80_long_context_512k",
    "dsv4_sm80_1m_smoke",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline DeepSeek V4 Flash benchmark (validated with torchrun TP8)."
    )
    parser.add_argument("--model", "--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--tp-size", "--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--request-count", "--num-requests", type=int, default=256)
    parser.add_argument("--min-input-length", type=int, default=100)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--min-output-length", type=int, default=100)
    parser.add_argument("--max-output-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recipe", choices=PUBLIC_RECIPES, default=DEFAULT_RECIPE)
    parser.add_argument(
        "--runtime-mode", choices=("optimized", "fallback"), default="optimized"
    )
    parser.add_argument("--warmup-output-length", type=int, default=2)
    parser.add_argument("--output", type=Path, help="Write the rank-0 JSON report here.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for low, high, label in (
        (args.min_input_length, args.max_input_length, "input"),
        (args.min_output_length, args.max_output_length, "output"),
    ):
        if low < 1 or high < low:
            raise SystemExit(f"invalid {label} length range: {low}..{high}")
    if args.request_count < 1:
        raise SystemExit("--request-count must be positive")
    if args.tp_size < 1:
        raise SystemExit("--tp-size must be positive")
    if args.runtime_mode == "fallback" and args.recipe != DEFAULT_RECIPE:
        raise SystemExit("--recipe cannot be combined with --runtime-mode fallback")
    return args


def distributed_info_from_env(tp_size: int) -> tuple[int, int, str | None]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={world_size} does not match --tp-size={tp_size}; "
            f"launch with torchrun --standalone --nproc_per_node={tp_size}."
        )
    if not 0 <= rank < world_size:
        raise SystemExit(f"LOCAL_RANK={rank} is invalid for WORLD_SIZE={world_size}")
    return rank, world_size, "env://" if world_size > 1 else None


def _valid_token_ids(tokenizer: Any) -> list[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    ids = sorted({int(token_id) for token_id in tokenizer.get_vocab().values()})
    ids = [token_id for token_id in ids if token_id >= 0 and token_id not in special_ids]
    if not ids:
        raise RuntimeError("the loaded tokenizer exposes no usable non-special token ids")
    return ids


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload, flush=True)
    if output is not None:
        output = output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM

    rank, world_size, init_method = distributed_info_from_env(args.tp_size)
    rng = random.Random(args.seed)
    kwargs: dict[str, Any] = {
        "tp_info": DistributedInfo(rank, world_size),
        "dsv4_runtime_mode": args.runtime_mode,
        "distributed_init_method": init_method,
    }
    if args.runtime_mode == "optimized":
        kwargs["dsv4_sm80_recipe"] = args.recipe
    llm = LLM(args.model, **kwargs)

    valid_ids = _valid_token_ids(llm.tokenizer)
    prompt_lengths = [
        rng.randint(args.min_input_length, args.max_input_length)
        for _ in range(args.request_count)
    ]
    output_lengths = [
        rng.randint(args.min_output_length, args.max_output_length)
        for _ in range(args.request_count)
    ]
    prompts = [rng.choices(valid_ids, k=length) for length in prompt_lengths]
    params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=length)
        for length in output_lengths
    ]

    warmup_id = valid_ids[0]
    llm.generate(
        [[warmup_id]],
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.warmup_output_length,
        ),
    )
    started = time.perf_counter()
    results = llm.generate(prompts, params)
    duration = time.perf_counter() - started
    actual_output_lengths = [len(result["token_ids"]) for result in results]
    failures = [result.get("error") for result in results if result.get("error")]
    if failures:
        raise RuntimeError(f"{len(failures)} requests failed; first error: {failures[0]}")

    if rank != 0:
        return None
    total_input = sum(prompt_lengths)
    total_output = sum(actual_output_lengths)
    report = {
        "benchmark": "offline_dsv4",
        "model": args.model,
        "runtime_mode": args.runtime_mode,
        "recipe": args.recipe if args.runtime_mode == "optimized" else None,
        "tp_size": world_size,
        "request_count": args.request_count,
        "seed": args.seed,
        "duration_seconds": duration,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "output_tokens_per_second": total_output / duration,
        "requests_per_second": args.request_count / duration,
        "input_length": {"min": min(prompt_lengths), "max": max(prompt_lengths)},
        "output_length": {
            "min": min(actual_output_lengths),
            "max": max(actual_output_lengths),
        },
    }
    _write_report(report, args.output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except SystemExit:
        raise
    except Exception as exc:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        if rank == 0:
            print(
                f"offline benchmark failed: {type(exc).__name__}: {exc}",
                file=__import__("sys").stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

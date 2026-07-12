from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

DEFAULT_MODEL = "/models/DeepSeek-V4-Flash"
DEFAULT_CACHE = Path("~/.cache/minisgl/benchmarks").expanduser()
DEFAULT_RECIPE = "dsv4_sm80_balanced"
WILDCHAT_FIRST_SHARD = "train-00000-of-00086.parquet"
WILDCHAT_BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/"
PUBLIC_RECIPES = (
    "dsv4_sm80_low_m64",
    "dsv4_sm80_mid_m128",
    "dsv4_sm80_balanced",
    "dsv4_sm80_long_context_512k",
    "dsv4_sm80_1m_smoke",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline WildChat benchmark for DeepSeek V4 Flash (validated with TP8)."
    )
    parser.add_argument("--model", "--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--tp-size", "--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--request-count", "--num-requests", type=int, default=256)
    parser.add_argument("--languages", nargs="+", default=["English", "Chinese"])
    parser.add_argument("--output-length", type=int, default=1024)
    parser.add_argument("--dataset-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--dataset-shard",
        type=Path,
        help="Use an existing WildChat parquet shard instead of downloading one.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recipe", choices=PUBLIC_RECIPES, default=DEFAULT_RECIPE)
    parser.add_argument(
        "--runtime-mode", choices=("optimized", "fallback"), default="optimized"
    )
    parser.add_argument("--output", type=Path, help="Write the rank-0 JSON report here.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.tp_size < 1 or args.request_count < 1 or args.output_length < 1:
        raise SystemExit("TP size, request count, and output length must be positive")
    if not args.languages:
        raise SystemExit("--languages must contain at least one language")
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
    return rank, world_size, "env://" if world_size > 1 else None


def _load_pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "WildChat benchmarking requires optional dependency pyarrow; "
            "install it with `pip install -e '.[benchmark]'`."
        ) from exc
    return parquet


def download_if_missing(url: str, path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading WildChat shard to user cache: {path}", flush=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def resolve_shard(args: argparse.Namespace) -> Path:
    if args.dataset_shard is not None:
        path = args.dataset_shard.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"WildChat shard does not exist: {path}")
        return path
    return download_if_missing(
        WILDCHAT_BASE_URL + WILDCHAT_FIRST_SHARD,
        args.dataset_cache.expanduser() / "wildchat" / WILDCHAT_FIRST_SHARD,
    )


def iter_filtered_prompt_ids(
    tokenizer: Any,
    shard_path: Path,
    languages: set[str],
    dsv4_chat_formatter: Callable[[list[dict]], str] | None = None,
) -> Iterator[list[int]]:
    parquet = _load_pyarrow_parquet()
    source = parquet.ParquetFile(shard_path)
    for batch in source.iter_batches(batch_size=256, columns=["conversation"]):
        for conversation in batch.to_pydict()["conversation"]:
            first_user = next(
                (turn for turn in (conversation or []) if turn.get("role") == "user"), None
            )
            if first_user is None:
                continue
            text = (first_user.get("content") or "").strip()
            if (
                not text
                or first_user.get("language") not in languages
                or bool(first_user.get("redacted"))
                or bool(first_user.get("toxic"))
            ):
                continue
            messages = [{"role": "user", "content": text}]
            if dsv4_chat_formatter is not None:
                yield tokenizer.encode(dsv4_chat_formatter(messages))
            else:
                yield tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload, flush=True)
    if output is not None:
        path = output.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM
    from minisgl.tokenizer.tokenize import load_dsv4_chat_formatter
    from transformers import AutoTokenizer

    rank, world_size, init_method = distributed_info_from_env(args.tp_size)
    shard_path = resolve_shard(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    chat_formatter = load_dsv4_chat_formatter(args.model)
    candidates = []
    for ids in iter_filtered_prompt_ids(
        tokenizer, shard_path, set(args.languages), chat_formatter
    ):
        candidates.append(ids)
        if len(candidates) == args.request_count * 4:
            break
    random.Random(args.seed).shuffle(candidates)
    prompts = candidates[: args.request_count]
    if not prompts:
        raise RuntimeError(
            f"no usable WildChat prompts for languages {args.languages!r} in {shard_path}"
        )
    if len(prompts) < args.request_count:
        raise RuntimeError(
            f"requested {args.request_count} prompts but found only {len(prompts)} in {shard_path}"
        )

    llm_kwargs: dict[str, Any] = {
        "tp_info": DistributedInfo(rank, world_size),
        "dsv4_runtime_mode": args.runtime_mode,
        "distributed_init_method": init_method,
    }
    if args.runtime_mode == "optimized":
        llm_kwargs["dsv4_sm80_recipe"] = args.recipe
    llm = LLM(args.model, **llm_kwargs)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.output_length)
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=2))
    started = time.perf_counter()
    results = llm.generate(prompts, sampling)
    duration = time.perf_counter() - started
    failures = [result.get("error") for result in results if result.get("error")]
    if failures:
        raise RuntimeError(f"{len(failures)} requests failed; first error: {failures[0]}")
    if rank != 0:
        return None

    output_lengths = [len(result["token_ids"]) for result in results]
    report = {
        "benchmark": "offline_wildchat_dsv4",
        "model": args.model,
        "runtime_mode": args.runtime_mode,
        "recipe": args.recipe if args.runtime_mode == "optimized" else None,
        "tp_size": world_size,
        "request_count": len(prompts),
        "seed": args.seed,
        "languages": args.languages,
        "dataset_shard": str(shard_path),
        "duration_seconds": duration,
        "total_input_tokens": sum(map(len, prompts)),
        "total_output_tokens": sum(output_lengths),
        "output_tokens_per_second": sum(output_lengths) / duration,
        "requests_per_second": len(prompts) / duration,
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
            print(f"WildChat benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:1919/v1"
DEFAULT_MODEL = "/models/DeepSeek-V4-Flash"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple OpenAI-compatible benchmark for a DeepSeek V4 Flash server."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer", "--tokenizer-path", help="Tokenizer path (defaults to expected model)."
    )
    parser.add_argument("--request-count", "--num-requests", type=int, default=64)
    parser.add_argument("--batch-size", "--concurrency", type=int, default=64)
    parser.add_argument("--min-input-length", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--min-output-length", type=int, default=16)
    parser.add_argument("--max-output-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, help="Write a JSON report here.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for value, name in (
        (args.request_count, "request count"),
        (args.batch_size, "batch size"),
        (args.timeout, "timeout"),
    ):
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if not 1 <= args.min_input_length <= args.max_input_length:
        raise SystemExit("invalid input length range")
    if not 2 <= args.min_output_length <= args.max_output_length:
        raise SystemExit("output lengths must be at least 2 and form a valid range")
    return args


async def verify_server_model(client: Any, expected_model: str) -> str:
    models = [model.id async for model in client.models.list()]
    if not models:
        raise RuntimeError("server /v1/models returned no models")
    if expected_model not in models:
        raise RuntimeError(
            f"server model mismatch: expected {expected_model!r}, /v1/models returned {models!r}"
        )
    return expected_model


def summarize(raw_results: list[Any], duration: float) -> dict[str, Any]:
    if not raw_results:
        raise RuntimeError("benchmark produced no results")
    if any(len(result.tics) < 2 for result in raw_results):
        raise RuntimeError("one or more responses produced no streamed output token")
    ttft = sorted(result.tics[1] - result.tics[0] for result in raw_results)
    total_output = sum(max(0, len(result.tics) - 1) for result in raw_results)
    return {
        "duration_seconds": duration,
        "request_count": len(raw_results),
        "total_output_tokens": total_output,
        "requests_per_second": len(raw_results) / duration,
        "output_tokens_per_second": total_output / duration,
        "ttft_ms": {
            "mean": 1000 * sum(ttft) / len(ttft),
            "p50": 1000 * ttft[len(ttft) // 2],
            "p90": 1000 * ttft[min(len(ttft) - 1, int(len(ttft) * 0.9))],
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from minisgl.benchmark.client import benchmark_one_batch, generate_prompt
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    random.seed(args.seed)
    async with AsyncOpenAI(
        base_url=args.base_url, api_key="dummy", timeout=args.timeout
    ) as client:
        model = await verify_server_model(client, args.expected_model)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.expected_model)
        input_lengths = [
            rng.randint(args.min_input_length, args.max_input_length)
            for _ in range(args.request_count)
        ]
        output_lengths = [
            rng.randint(args.min_output_length, args.max_output_length)
            for _ in range(args.request_count)
        ]
        prompts = [generate_prompt(tokenizer, length) for length in input_lengths]
        all_results = []
        started = time.perf_counter()
        for offset in range(0, args.request_count, args.batch_size):
            end = min(args.request_count, offset + args.batch_size)
            batch = await benchmark_one_batch(
                client,
                prompts[offset:end],
                output_lengths[offset:end],
                model,
                input_lengths=input_lengths[offset:end],
                pbar=False,
            )
            all_results.extend(batch)
        duration = time.perf_counter() - started

    report = {
        "benchmark": "online_simple_dsv4",
        "base_url": args.base_url,
        "model": model,
        "tokenizer": args.tokenizer or args.expected_model,
        "seed": args.seed,
        "batch_size": args.batch_size,
        **summarize(all_results, duration),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        asyncio.run(run(parse_args(argv)))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"online simple benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

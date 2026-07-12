from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

DEFAULT_BASE_URL = "http://127.0.0.1:1919/v1"
DEFAULT_MODEL = "/models/DeepSeek-V4-Flash"
DEFAULT_CACHE = Path("~/.cache/minisgl/benchmarks").expanduser()
TRACE_NAME = "qwen_traceA_blksz_16.jsonl"
TRACE_URL = (
    "https://media.githubusercontent.com/media/alibaba-edu/"
    "qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a Qwen-format request trace against a DeepSeek V4 Flash server. "
            "The trace format does not imply Qwen model support."
        )
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokenizer", "--tokenizer-path", help="Tokenizer path (defaults to expected model)."
    )
    parser.add_argument("--request-count", "--num-requests", type=int, default=1000)
    parser.add_argument("--max-concurrency", "--concurrency", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale trace inter-arrival times.")
    parser.add_argument("--trace-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--trace-file", type=Path, help="Use an existing trace JSONL file.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, help="Write a JSON report here.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.request_count < 1 or args.max_concurrency < 1:
        raise SystemExit("request count and concurrency must be positive")
    if args.scale <= 0 or args.timeout <= 0:
        raise SystemExit("scale and timeout must be positive")
    return args


def resolve_trace(args: argparse.Namespace) -> Path:
    if args.trace_file is not None:
        path = args.trace_file.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"trace file does not exist: {path}")
        return path
    path = args.trace_cache.expanduser() / "qwen-format" / TRACE_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Qwen-format request trace to user cache: {path}", flush=True)
        urllib.request.urlretrieve(TRACE_URL, path)
    return path


async def verify_server_model(client: Any, expected_model: str) -> str:
    models = [model.id async for model in client.models.list()]
    if not models:
        raise RuntimeError("server /v1/models returned no models")
    if expected_model not in models:
        raise RuntimeError(
            f"server model mismatch: expected {expected_model!r}, /v1/models returned {models!r}"
        )
    return expected_model


async def replay_with_limit(client: Any, traces: list[Any], model: str, limit: int) -> list[Any]:
    from minisgl.benchmark.client import benchmark_one

    semaphore = asyncio.Semaphore(limit)
    started = time.perf_counter()
    first_timestamp = min(trace.timestamp for trace in traces)

    async def one(trace: Any):
        target = started + trace.timestamp - first_timestamp
        await asyncio.sleep(max(0.0, target - time.perf_counter()))
        async with semaphore:
            return await benchmark_one(
                client,
                trace.message,
                trace.output_length,
                model,
                pbar=False,
                input_length=trace.input_length,
            )

    return await asyncio.gather(*(one(trace) for trace in traces))


def summarize(raw_results: list[Any], duration: float) -> dict[str, Any]:
    if not raw_results or any(len(result.tics) < 2 for result in raw_results):
        raise RuntimeError("one or more trace requests produced no streamed output token")
    total_output = sum(max(0, len(result.tics) - 1) for result in raw_results)
    ttft = sorted(result.tics[1] - result.tics[0] for result in raw_results)
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
    from minisgl.benchmark.client import read_qwen_trace, scale_traces
    from openai import AsyncOpenAI
    from transformers import AutoTokenizer

    random.seed(args.seed)
    trace_path = resolve_trace(args)
    async with AsyncOpenAI(
        base_url=args.base_url, api_key="dummy", timeout=args.timeout
    ) as client:
        model = await verify_server_model(client, args.expected_model)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.expected_model)
        traces = read_qwen_trace(
            str(trace_path), tokenizer, n=args.request_count, dummy=True
        )
        if not traces:
            raise RuntimeError(f"trace file contains no requests: {trace_path}")
        if len(traces) < args.request_count:
            raise RuntimeError(
                f"requested {args.request_count} trace rows but found only {len(traces)} "
                f"in {trace_path}"
            )
        traces = scale_traces(traces, args.scale)
        started = time.perf_counter()
        results = await replay_with_limit(client, traces, model, args.max_concurrency)
        duration = time.perf_counter() - started

    report = {
        "benchmark": "qwen_format_trace_replay_dsv4",
        "workload": "Qwen-format request trace replay",
        "base_url": args.base_url,
        "model": model,
        "tokenizer": args.tokenizer or args.expected_model,
        "trace_file": str(trace_path),
        "seed": args.seed,
        "scale": args.scale,
        "max_concurrency": args.max_concurrency,
        **summarize(results, duration),
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
        print(f"trace replay benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

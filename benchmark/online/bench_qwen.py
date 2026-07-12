import asyncio
import random
import urllib.request
from pathlib import Path

from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

PORT = 1919
NUM_REQUESTS = 1000
SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]
TRACE_PATH = Path("~/.cache/minisgl/benchmarks/qwen_traceA_blksz_16.jsonl").expanduser()
TRACE_URL = (
    "https://media.githubusercontent.com/media/alibaba-edu/"
    "qwen-bailian-usagetraces-anon/refs/heads/main/"
    "qwen_traceA_blksz_16.jsonl"
)


def download_trace() -> str:
    if not TRACE_PATH.exists():
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading trace from {TRACE_URL} to {TRACE_PATH}")
        urllib.request.urlretrieve(TRACE_URL, TRACE_PATH)
    return str(TRACE_PATH)


async def main():
    random.seed(42)
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        traces = read_qwen_trace(download_trace(), tokenizer, n=NUM_REQUESTS, dummy=True)
        logger.info(f"Benchmarking {NUM_REQUESTS} requests with model {model}")
        for scale in SCALES:
            results = await benchmark_trace(client, scale_traces(traces, scale), model)
            process_benchmark_results(results)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import random
from typing import List

from minisgl.benchmark.client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)

PORT = 1919
TOKENIZER = "/models/DeepSeek-V4-Flash"
TEST_BATCH_SIZES = [64]
INPUT_LEN = 1024
OUTPUT_LEN = 1024


async def main():
    random.seed(42)

    async def generate_tasks(max_batch_size: int) -> List[str]:
        result = []
        for _ in range(max_batch_size):
            result.append(generate_prompt(tokenizer, INPUT_LEN))
            await asyncio.sleep(0)
        return result

    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        logger.info(f"Using served model {model}; loaded tokenizer from {TOKENIZER}")

        tasks = asyncio.create_task(generate_tasks(max(TEST_BATCH_SIZES)))
        test_message = generate_prompt(tokenizer, 100)
        test_result = await benchmark_one(client, test_message, 2, model, pbar=False)
        if len(test_result.tics) <= 2:
            raise RuntimeError("Server connection test produced no output")

        messages = await tasks
        for batch_size in TEST_BATCH_SIZES:
            logger.info(
                f"Benchmarking fixed batch M={batch_size}, input={INPUT_LEN}tok, "
                f"output={OUTPUT_LEN}tok"
            )
            results = await benchmark_one_batch(
                client,
                messages[:batch_size],
                OUTPUT_LEN,
                model,
            )
            process_benchmark_results(results)


if __name__ == "__main__":
    asyncio.run(main())

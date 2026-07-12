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
TEST_BATCH_SIZES = [64]
MAX_INPUT_LEN = 8192


async def main():
    random.seed(42)

    async def generate_tasks(max_batch_size: int) -> List[str]:
        result = []
        for _ in range(max_batch_size):
            length = random.randint(1, MAX_INPUT_LEN)
            result.append(generate_prompt(tokenizer, length))
            await asyncio.sleep(0)
        return result

    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        logger.info(f"Loaded tokenizer from {model}")

        tasks = asyncio.create_task(generate_tasks(max(TEST_BATCH_SIZES)))
        test_message = generate_prompt(tokenizer, 100)
        test_result = await benchmark_one(client, test_message, 2, model, pbar=False)
        if len(test_result.tics) <= 2:
            raise RuntimeError("Server connection test produced no output")

        messages = await tasks
        output_lengths = [random.randint(16, 1024) for _ in messages]
        for batch_size in TEST_BATCH_SIZES:
            results = await benchmark_one_batch(
                client,
                messages[:batch_size],
                output_lengths[:batch_size],
                model,
            )
            process_benchmark_results(results)


if __name__ == "__main__":
    asyncio.run(main())

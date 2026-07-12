import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.distributed import get_tp_info, launch_tensor_parallel
from minisgl.llm import LLM

MODEL = "/models/DeepSeek-V4-Flash"
TP_SIZE = 8
NUM_SEQS = 256
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024


def main():
    seed(0)
    llm = LLM(MODEL)
    prompts = [
        [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))] for _ in range(NUM_SEQS)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, MAX_OUTPUT_LEN),
        )
        for _ in range(NUM_SEQS)
    ]

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1, max_tokens=2))
    start = time.time()
    results = llm.generate(prompts, sampling_params)
    duration = time.time() - start
    total_tokens = sum(len(result["token_ids"]) for result in results)
    if get_tp_info().is_primary():
        print(
            f"Total: {total_tokens}tok, Time: {duration:.2f}s, "
            f"Throughput: {total_tokens / duration:.2f}tok/s"
        )


if __name__ == "__main__":
    launch_tensor_parallel(TP_SIZE, main)

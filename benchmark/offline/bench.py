import time
from random import randint, seed

from minisgl.core import SamplingParams
from minisgl.distributed import get_tp_info, launch_tensor_parallel
from minisgl.llm import LLM

MODEL = "/models/DeepSeek-V4-Flash"
TP_SIZE = 8
NUM_SEQS = 128
INPUT_LEN = 1024
OUTPUT_LEN = 1024


def make_workload():
    seed(0)
    prompts = [[randint(0, 10000) for _ in range(INPUT_LEN)] for _ in range(NUM_SEQS)]
    sampling_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=OUTPUT_LEN)
        for _ in range(NUM_SEQS)
    ]
    return prompts, sampling_params


def main():
    llm = LLM(MODEL)
    prompts, sampling_params = make_workload()

    llm.generate(
        ["Benchmark: "],
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=2),
    )
    start = time.time()
    results = llm.generate(prompts, sampling_params)
    duration = time.time() - start
    input_tokens = sum(len(prompt) for prompt in prompts)
    output_tokens = sum(len(result["token_ids"]) for result in results)
    if get_tp_info().is_primary():
        print(
            f"Workload: M={NUM_SEQS}, input={INPUT_LEN}tok, output={OUTPUT_LEN}tok, "
            "greedy, ignore_eos=True"
        )
        print(
            f"Input: {input_tokens}tok, Output: {output_tokens}tok, "
            f"Time: {duration:.2f}s, Requests: {NUM_SEQS / duration:.2f}req/s, "
            f"Output throughput: {output_tokens / duration:.2f}tok/s"
        )


if __name__ == "__main__":
    launch_tensor_parallel(TP_SIZE, main)

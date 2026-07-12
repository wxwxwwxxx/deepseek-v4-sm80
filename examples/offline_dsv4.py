"""Run one DeepSeek V4 Flash generation with local TP8 workers.

Execute this file directly; ``launch_tensor_parallel`` starts all eight local
workers, so an external ``torchrun`` command is not required.
"""

from __future__ import annotations

from minisgl.core import SamplingParams
from minisgl.distributed import get_tp_info, launch_tensor_parallel
from minisgl.llm import LLM
from minisgl.tokenizer.tokenize import load_dsv4_chat_formatter

MODEL = "/models/DeepSeek-V4-Flash"


def main() -> None:
    llm = LLM(MODEL)
    chat_formatter = load_dsv4_chat_formatter(MODEL)
    if chat_formatter is None:
        raise RuntimeError("DeepSeek V4 chat formatter is missing from the model directory")
    prompt = chat_formatter([{"role": "user", "content": "Reply with only 4: 2 + 2 ="}])
    result = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=32),
    )
    if get_tp_info().is_primary():
        print(result[0]["text"])


if __name__ == "__main__":
    launch_tensor_parallel(8, main)

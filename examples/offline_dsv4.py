"""Minimal torchrun-native DeepSeek V4 Flash LLM example."""

from __future__ import annotations

import os

from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.llm import LLM


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if world_size != 8:
        raise SystemExit(f"this validated example requires WORLD_SIZE=8, got {world_size}")
    llm = LLM(
        "/models/DeepSeek-V4-Flash",
        tp_info=DistributedInfo(rank, world_size),
        distributed_init_method="env://",
        dsv4_runtime_mode="optimized",
        dsv4_sm80_recipe="dsv4_sm80_balanced",
    )
    result = llm.generate(
        ["Answer briefly: what is 2 + 2?"],
        SamplingParams(temperature=0.0, max_tokens=32),
    )
    if rank == 0:
        print(result[0]["text"])


if __name__ == "__main__":
    main()

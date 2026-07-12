import os
import random
import shutil
import time
import urllib.request
from pathlib import Path

from minisgl.core import SamplingParams
from minisgl.distributed import get_tp_info, launch_tensor_parallel
from minisgl.llm import LLM
from minisgl.tokenizer.tokenize import load_dsv4_chat_formatter
from transformers import AutoTokenizer

MODEL = "/models/DeepSeek-V4-Flash"
TP_SIZE = 8
NUM_SEQS = 256
MAX_OUTPUT_LEN = 4096
LANGUAGES = {"English", "Chinese"}

CACHE_DIR = Path("~/.cache/minisgl/benchmarks/wildchat").expanduser()
SHARD_NAME = "train-00000-of-00086.parquet"
SHARD_URL = f"https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/{SHARD_NAME}"


def download_if_missing() -> Path:
    path = CACHE_DIR / SHARD_NAME
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    print(f"Downloading {SHARD_URL} -> {path}")
    try:
        with (
            urllib.request.urlopen(SHARD_URL, timeout=300) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def iter_prompt_ids(tokenizer, chat_formatter):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "WildChat requires pyarrow; install it with `pip install -e '.[benchmark]'`."
        ) from exc
    parquet = pq.ParquetFile(download_if_missing())
    for batch in parquet.iter_batches(batch_size=256, columns=["conversation"]):
        for conversation in batch.to_pydict()["conversation"]:
            user = next(
                (turn for turn in (conversation or []) if turn.get("role") == "user"),
                None,
            )
            if user is None:
                continue
            text = (user.get("content") or "").strip()
            if (
                not text
                or user.get("language") not in LANGUAGES
                or user.get("redacted")
                or user.get("toxic")
            ):
                continue
            messages = [{"role": "user", "content": text}]
            if chat_formatter is not None:
                yield tokenizer.encode(chat_formatter(messages))
            else:
                yield tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )


def print_len_stats(name: str, lengths: list[int]) -> None:
    values = sorted(lengths)
    count = len(values)
    print(
        f"{name}: count={count}, min={values[0]}, p50={values[count // 2]}, "
        f"p90={values[int(count * 0.9)]}, max={values[-1]}"
    )


def main():
    random.seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    chat_formatter = load_dsv4_chat_formatter(MODEL)
    prompts = []
    for prompt_ids in iter_prompt_ids(tokenizer, chat_formatter):
        prompts.append(prompt_ids)
        if len(prompts) >= NUM_SEQS:
            break
    if not prompts:
        raise RuntimeError("No usable WildChat prompts were found")

    llm = LLM(MODEL)
    sampling_params = SamplingParams(
        temperature=0.6,
        ignore_eos=False,
        max_tokens=MAX_OUTPUT_LEN,
    )
    llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=2))
    start = time.time()
    results = llm.generate(prompts, sampling_params)
    duration = time.time() - start

    if get_tp_info().is_primary():
        output_lengths = [len(result["token_ids"]) for result in results]
        total_tokens = sum(output_lengths)
        print_len_stats("Input length", [len(prompt) for prompt in prompts])
        print_len_stats("Output length", output_lengths)
        print(
            f"Total: {total_tokens}tok, Time: {duration:.2f}s, "
            f"Throughput: {total_tokens / duration:.2f}tok/s"
        )


if __name__ == "__main__":
    launch_tensor_parallel(TP_SIZE, main)

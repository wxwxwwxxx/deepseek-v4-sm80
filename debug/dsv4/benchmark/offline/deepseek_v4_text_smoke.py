from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

os.environ.setdefault("MINISGL_DISABLE_OVERLAP_SCHEDULING", "1")

DEFAULT_PROMPTS = (
    "请用一句中文回答：2 + 2 等于几？",
    "Answer in one short English sentence: what color is the sky on a clear day?",
    "用一句话介绍杭州，不要超过20个字。",
)
DEFAULT_EXPECTATIONS = {
    DEFAULT_PROMPTS[0]: ("4", "四"),
    DEFAULT_PROMPTS[1]: ("blue",),
    DEFAULT_PROMPTS[2]: ("杭州",),
}


def load_dsv4_encoding(model_path: str):
    path = Path(model_path) / "encoding" / "encoding_dsv4.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("deepseek_v4_encoding_dsv4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load DeepSeek V4 encoding from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_chat_prompt(
    prompt: str,
    *,
    model_path: str,
    system_prompt: str,
    thinking_mode: str,
) -> str:
    encoding = load_dsv4_encoding(model_path)
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if encoding is not None:
        return encoding.encode_messages(messages, thinking_mode=thinking_mode)
    prefix = f"{system_prompt}\n\n" if system_prompt else ""
    return f"{prefix}User: {prompt}\nAssistant:"


def parse_completion_text(
    raw_text: str,
    *,
    model_path: str,
    thinking_mode: str,
) -> dict[str, Any] | None:
    encoding = load_dsv4_encoding(model_path)
    if encoding is None:
        return None
    text = raw_text
    if getattr(encoding, "eos_token", "") and encoding.eos_token not in text:
        text += encoding.eos_token
    try:
        return encoding.parse_message_from_completion_text(text, thinking_mode=thinking_mode)
    except Exception as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}"}


def _normalize(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def response_sanity(
    text: str,
    *,
    prompt: str,
    token_ids: Sequence[int],
    expected_substrings: Sequence[str],
) -> dict[str, Any]:
    issues: list[str] = []
    stripped = text.strip()
    printable = [ch for ch in text if ch.isprintable() or ch in "\n\r\t"]
    if not stripped:
        issues.append("empty_output")
    if "\ufffd" in text:
        issues.append("replacement_character")
    if "\x00" in text:
        issues.append("nul_character")
    printable_fraction = len(printable) / max(len(text), 1)
    if printable_fraction < 0.95:
        issues.append("many_non_printable_chars")
    if expected_substrings and not any(
        expected.casefold() in text.casefold() for expected in expected_substrings
    ):
        issues.append("missing_expected_substring")

    max_token_run = 1 if token_ids else 0
    token_run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        token_run = token_run + 1 if current == previous else 1
        max_token_run = max(max_token_run, token_run)
    repeated_pattern_length = 0
    for length in range(1, min(4, len(token_ids) // 3) + 1):
        tail = list(token_ids[-3 * length :])
        if tail == tail[:length] * 3:
            repeated_pattern_length = length
            break
    if max_token_run >= 4:
        issues.append("repeated_token_loop")
    elif repeated_pattern_length:
        issues.append("repeated_short_token_pattern")

    prompt_norm = _normalize(prompt)
    text_norm = _normalize(text)
    overlap_fraction = 0.0
    if prompt_norm and len(text_norm) >= 6:
        match = difflib.SequenceMatcher(None, prompt_norm, text_norm).find_longest_match(
            0, len(prompt_norm), 0, len(text_norm)
        )
        overlap_fraction = match.size / max(len(text_norm), 1)
        if overlap_fraction >= 0.7:
            issues.append("prompt_echo_like")
    return {
        "looks_sane": not issues,
        "issues": issues,
        "expected_substrings": list(expected_substrings),
        "printable_fraction": printable_fraction,
        "max_repeated_token_run": max_token_run,
        "repeated_pattern_length": repeated_pattern_length,
        "prompt_overlap_fraction": overlap_fraction,
    }


def _distributed_info(args: argparse.Namespace) -> tuple[int, int, str | None]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if world_size != args.tensor_parallel_size:
        raise SystemExit(
            f"WORLD_SIZE={world_size} does not match --tensor-parallel-size="
            f"{args.tensor_parallel_size}; launch with torchrun --nproc_per_node=8."
        )
    init_method = "env://" if world_size > 1 else None
    return rank, world_size, init_method


def _jsonable_runtime(runtime) -> dict[str, Any]:
    payload = asdict(runtime)
    payload["direct_graph_metadata_groups"] = sorted(payload["direct_graph_metadata_groups"])
    return payload


def run_text_smoke(args: argparse.Namespace) -> int:
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.dsv4_runtime import resolve_dsv4_runtime_config
    from minisgl.llm import LLM

    rank, world_size, init_method = _distributed_info(args)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    formatted = [
        format_chat_prompt(
            prompt,
            model_path=args.model_path,
            system_prompt=args.system_prompt,
            thinking_mode=args.thinking_mode,
        )
        for prompt in prompts
    ]
    fallback = args.dsv4_runtime == "fallback"
    use_pynccl = False if fallback else not args.disable_pynccl
    allow_graph = False if fallback else not args.disable_cuda_graph
    llm_kwargs: dict[str, Any] = {}
    if init_method is not None:
        llm_kwargs["distributed_init_method"] = init_method

    llm = None
    error = None
    outputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        llm = LLM(
            args.model_path,
            tp_info=DistributedInfo(rank, world_size),
            dsv4_runtime_mode=args.dsv4_runtime,
            dsv4_sm80_recipe=args.dsv4_sm80_recipe,
            max_running_req=args.max_running_req or max(len(prompts), 1),
            context_length=args.max_seq_len,
            max_extend_tokens=args.max_extend_tokens,
            num_page_override=args.num_pages,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            use_pynccl=use_pynccl,
            allow_dsv4_cuda_graph=allow_graph,
            disable_cuda_graph=not allow_graph,
            **llm_kwargs,
        )
        generated = llm.generate(
            formatted,
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                ignore_eos=False,
                max_tokens=args.max_tokens,
            ),
        )
        torch.cuda.synchronize(llm.device)
        if rank == 0:
            for prompt, formatted_prompt, item in zip(prompts, formatted, generated):
                token_ids = list(item["token_ids"])
                raw_text = llm.tokenizer.decode(token_ids, skip_special_tokens=False)
                text = llm.tokenizer.decode(token_ids, skip_special_tokens=True)
                expected = DEFAULT_EXPECTATIONS.get(prompt, ())
                outputs.append(
                    {
                        "prompt": prompt,
                        "formatted_prompt_preview": formatted_prompt[:240],
                        "generated_token_ids": token_ids,
                        "generated_token_count": len(token_ids),
                        "raw_text": raw_text,
                        "text": text,
                        "parsed": parse_completion_text(
                            raw_text,
                            model_path=args.model_path,
                            thinking_mode=args.thinking_mode,
                        ),
                        "sanity": response_sanity(
                            text or raw_text,
                            prompt=prompt,
                            token_ids=token_ids,
                            expected_substrings=expected,
                        ),
                    }
                )
    except BaseException as exc:
        error = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }

    elapsed_s = time.perf_counter() - started
    if rank == 0:
        status = "pass"
        if error or any(not output["sanity"]["looks_sane"] for output in outputs):
            status = "fail"
        runtime = resolve_dsv4_runtime_config(args.dsv4_runtime)
        payload = {
            "status": status,
            "model_path": args.model_path,
            "dsv4_runtime": _jsonable_runtime(runtime),
            "prompts": prompts,
            "outputs": outputs,
            "error": error,
            "elapsed_s": elapsed_s,
            "config": {
                "tensor_parallel_size": world_size,
                "page_size": args.page_size,
                "num_pages": args.num_pages,
                "use_pynccl": use_pynccl,
                "allow_dsv4_cuda_graph": allow_graph,
                "max_seq_len": args.max_seq_len,
                "max_running_req": args.max_running_req or max(len(prompts), 1),
                "max_extend_tokens": args.max_extend_tokens,
                "max_tokens": args.max_tokens,
                "graph_runner": getattr(llm.engine.graph_runner, "capture_status", {}) if llm else {},
                "model_prepare_report_rank0": getattr(llm.engine, "model_prepare_report", {}) if llm else {},
                "kv_capacity_plan_report": getattr(llm.engine, "kv_capacity_plan_report", {}) if llm else {},
            },
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        status = "pass" if error is None else "fail"

    if llm is not None:
        try:
            llm.shutdown()
        except BaseException:
            if rank == 0:
                traceback.print_exc()
    return 0 if status == "pass" else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical optimized/fallback DeepSeek V4 TP text correctness smoke."
    )
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--dsv4-runtime", choices=("optimized", "fallback"), default="optimized")
    parser.add_argument("--recipe", dest="dsv4_sm80_recipe", default=None)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--output", default="/tmp/dsv4_text_smoke.json")
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-running-req", type=int, default=None)
    parser.add_argument("--max-extend-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--disable-pynccl", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Answer briefly and clearly.",
    )
    args = parser.parse_args(argv)
    if args.page_size <= 0 or args.num_pages <= 1:
        parser.error("--page-size must be positive and --num-pages must exceed one")
    if args.max_tokens <= 0 or args.max_seq_len <= 0 or args.max_extend_tokens <= 0:
        parser.error("token and sequence limits must be positive")
    if args.max_running_req is not None and args.max_running_req <= 0:
        parser.error("--max-running-req must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_text_smoke(parse_args(argv)))


if __name__ == "__main__":
    main()

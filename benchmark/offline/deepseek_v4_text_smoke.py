from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
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
    DEFAULT_PROMPTS[1]: ("blue", "Blue"),
    DEFAULT_PROMPTS[2]: ("杭州",),
}
DSV4_V0_BF16_TOGGLE = "MINISGL_DSV4_SM80_V0_BF16"
DSV4_V1_MOE_TOGGLE = "MINISGL_DSV4_SM80_V1_MOE"
BASELINE_TP_SIZE = 8


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]
    description: str


VARIANTS: tuple[Variant, ...] = (
    Variant("fallback", {}, "All MINISGL_DSV4_SM80_* toggles cleared."),
    Variant("v0_bf16", {DSV4_V0_BF16_TOGGLE: "1"}, "TARGET 05.7 v0 BF16 whitelist bundle."),
    Variant(
        "v1_moe",
        {DSV4_V1_MOE_TOGGLE: "1"},
        "V1 exact grouped MoE bundle: v0 BF16 whitelist plus grouped MoE route.",
    ),
)


def _variant_map() -> dict[str, Variant]:
    return {variant.name: variant for variant in VARIANTS}


def _all_dsv4_sm80_env_names(dsv4_kernel) -> list[str]:
    names = set(getattr(dsv4_kernel, "DSV4_SM80_KNOWN_TOGGLES", ()))
    names.update(name for name in os.environ if name.startswith("MINISGL_DSV4_SM80_"))
    return sorted(names)


def configure_variant(dsv4_kernel, variant: Variant) -> dict[str, Any]:
    cleared = _all_dsv4_sm80_env_names(dsv4_kernel)
    for name in cleared:
        os.environ.pop(name, None)
    for name, value in variant.env.items():
        os.environ[name] = value
    return {
        "cleared_dsv4_sm80_env": cleared,
        "active_dsv4_toggles": [
            name for name in _all_dsv4_sm80_env_names(dsv4_kernel) if dsv4_kernel.dsv4_env_flag(name)
        ],
        "raw_dsv4_sm80_env": {
            name: os.environ[name]
            for name in sorted(os.environ)
            if name.startswith("MINISGL_DSV4_SM80_")
        },
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
    if system_prompt:
        return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant:"


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
        text = text + encoding.eos_token
    try:
        return encoding.parse_message_from_completion_text(text, thinking_mode=thinking_mode)
    except Exception as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}"}


def text_sanity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    non_ws = [ch for ch in text if not ch.isspace()]
    content_chars = [ch for ch in non_ws if ch.isalnum()]
    printable = [ch for ch in text if ch.isprintable() or ch in "\n\r\t"]
    max_run = 0
    max_symbol_run = 0
    current_run = 0
    current_symbol_run = 0
    last = None
    for ch in text:
        if ch == last:
            current_run += 1
        else:
            last = ch
            current_run = 1
        max_run = max(max_run, current_run)
        if ch == last and not ch.isalnum() and not ch.isspace():
            current_symbol_run = current_run
        else:
            current_symbol_run = 0
        max_symbol_run = max(max_symbol_run, current_symbol_run)

    issues = []
    if not stripped:
        issues.append("empty_output")
    if "\ufffd" in text:
        issues.append("replacement_character")
    if "\x00" in text:
        issues.append("nul_character")
    if non_ws and len(printable) / max(len(text), 1) < 0.95:
        issues.append("many_non_printable_chars")
    if non_ws and len(content_chars) / len(non_ws) < 0.25:
        issues.append("mostly_punctuation_or_symbols")
    if max_run >= 12:
        issues.append("long_repeated_character_run")
    if max_symbol_run >= 6:
        issues.append("long_repeated_symbol_run")
    for token in ("<｜begin", "<｜User｜>", "<｜Assistant｜>", "<|"):
        if token in text:
            issues.append(f"special_token_leak:{token}")
            break

    return {
        "looks_sane": len(issues) == 0,
        "issues": issues,
        "char_count": len(text),
        "non_whitespace_char_count": len(non_ws),
        "content_char_fraction": len(content_chars) / max(len(non_ws), 1),
        "printable_fraction": len(printable) / max(len(text), 1),
        "max_repeated_char_run": max_run,
        "max_repeated_symbol_run": max_symbol_run,
    }


def _normalize_for_overlap(text: str) -> str:
    return "".join(ch.casefold() for ch in text if ch.isalnum())


def response_sanity(
    text: str,
    *,
    prompt: str,
    expected_substrings: Sequence[str] | None = None,
) -> dict[str, Any]:
    report = text_sanity(text)
    issues = list(report["issues"])
    prompt_norm = _normalize_for_overlap(prompt)
    text_norm = _normalize_for_overlap(text)
    overlap_fraction = 0.0
    if prompt_norm and len(text_norm) >= 6:
        match = difflib.SequenceMatcher(None, prompt_norm, text_norm).find_longest_match(
            0, len(prompt_norm), 0, len(text_norm)
        )
        overlap_fraction = match.size / max(len(text_norm), 1)
        if overlap_fraction >= 0.7:
            issues.append("prompt_echo_like")
    if expected_substrings and not any(
        expected in text for expected in expected_substrings
    ):
        issues.append("missing_expected_substring")

    report["issues"] = issues
    report["looks_sane"] = len(issues) == 0
    report["prompt_overlap_fraction"] = overlap_fraction
    report["expected_substrings"] = list(expected_substrings or ())
    return report


def _tp_rank_size(args: argparse.Namespace) -> tuple[int, int, int]:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
    tp_size = args.tensor_parallel_size or env_world_size
    tp_rank = args.tp_rank if args.tp_rank is not None else env_local_rank
    if tp_size <= 0:
        raise ValueError("tensor parallel size must be positive")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid TP rank {tp_rank} for TP size {tp_size}")
    return tp_rank, tp_size, env_world_size


def _distributed_init_method(args: argparse.Namespace, tp_size: int) -> str | None:
    if args.distributed_init_method is not None:
        return args.distributed_init_method
    if tp_size > 1 and "MASTER_ADDR" in os.environ:
        return "env://"
    return None


def _gather_payloads(group, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size(group=group)
        gathered: list[Any] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, payload, group=group)
        return list(gathered)
    return [payload]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_variant(
    *,
    args: argparse.Namespace,
    llm,
    dsv4_kernel,
    variant: Variant,
    prompts: Sequence[str],
    formatted_prompts: Sequence[str],
    rank: int,
    tp_size: int,
) -> dict[str, Any] | None:
    from minisgl.core import SamplingParams

    variant_env = configure_variant(dsv4_kernel, variant)
    llm.sync_all_ranks()
    tic = time.perf_counter()
    error = None
    outputs: list[dict[str, Any]] = []
    try:
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            ignore_eos=False,
            max_tokens=args.max_tokens,
        )
        generated = llm.generate(list(formatted_prompts), sampling)
        torch.cuda.synchronize(llm.device)
        elapsed_s = time.perf_counter() - tic
        if rank == 0:
            for idx, (prompt, formatted, item) in enumerate(
                zip(prompts, formatted_prompts, generated)
            ):
                token_ids = list(item["token_ids"])
                raw_text = llm.tokenizer.decode(token_ids, skip_special_tokens=False)
                clean_text = llm.tokenizer.decode(token_ids, skip_special_tokens=True)
                parsed = parse_completion_text(
                    raw_text,
                    model_path=args.model_path,
                    thinking_mode=args.thinking_mode,
                )
                expected_substrings = DEFAULT_EXPECTATIONS.get(prompt)
                outputs.append(
                    {
                        "index": idx,
                        "prompt": prompt,
                        "formatted_prompt_preview": formatted[:240],
                        "generated_token_count": len(token_ids),
                        "generated_token_ids": token_ids,
                        "raw_text": raw_text,
                        "text": clean_text,
                        "parsed": parsed,
                        "sanity": response_sanity(
                            clean_text or raw_text,
                            prompt=prompt,
                            expected_substrings=expected_substrings,
                        ),
                    }
                )
    except BaseException as exc:
        elapsed_s = time.perf_counter() - tic
        error = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }

    rank_payload = {
        "rank": rank,
        "elapsed_s": elapsed_s,
        "error": error,
        "memory": {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(llm.device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(llm.device)),
        },
    }
    gathered = _gather_payloads(llm.tp_cpu_group, rank_payload)
    if rank != 0:
        return None

    errors = [payload["error"] for payload in gathered if payload.get("error")]
    status = "fail" if errors else "pass"
    if status == "pass" and any(not item["sanity"]["looks_sane"] for item in outputs):
        status = "warn"
    return {
        "status": status,
        "variant": {
            "name": variant.name,
            "description": variant.description,
            **variant_env,
        },
        "elapsed_s": max(float(payload["elapsed_s"]) for payload in gathered),
        "outputs": outputs,
        "errors": errors,
        "per_rank": gathered,
        "config": {
            "tensor_parallel_size": tp_size,
            "page_size": args.page_size,
            "num_pages": args.num_pages,
            "use_pynccl": False,
            "distributed_init_method": _distributed_init_method(args, tp_size),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_seq_len": args.max_seq_len,
            "max_extend_tokens": args.max_extend_tokens,
            "thinking_mode": args.thinking_mode,
        },
    }


def run_text_smoke(args: argparse.Namespace) -> int:
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM
    from minisgl.kernel import deepseek_v4 as dsv4_kernel

    rank, tp_size, env_world_size = _tp_rank_size(args)
    if env_world_size != tp_size:
        raise SystemExit(
            f"WORLD_SIZE={env_world_size} does not match tensor parallel size {tp_size}; "
            "launch with torchrun --standalone --nproc_per_node=8 for TP8."
        )
    distributed_init_method = _distributed_init_method(args, tp_size)
    prompts = args.prompt or list(DEFAULT_PROMPTS)
    formatted_prompts = [
        format_chat_prompt(
            prompt,
            model_path=args.model_path,
            system_prompt=args.system_prompt,
            thinking_mode=args.thinking_mode,
        )
        for prompt in prompts
    ]
    variants = [_variant_map()[name] for name in (args.variants or ["fallback", "v0_bf16"])]

    llm = None
    reports: list[dict[str, Any]] = []
    try:
        llm_kwargs: dict[str, Any] = {}
        if distributed_init_method is not None:
            llm_kwargs["distributed_init_method"] = distributed_init_method
        llm = LLM(
            args.model_path,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(rank, tp_size),
            max_running_req=max(len(prompts), 1),
            max_seq_len_override=args.max_seq_len,
            max_extend_tokens=args.max_extend_tokens,
            num_page_override=args.num_pages,
            page_size=args.page_size,
            memory_ratio=args.memory_ratio,
            use_pynccl=False,
            **llm_kwargs,
        )
        for variant in variants:
            report = run_variant(
                args=args,
                llm=llm,
                dsv4_kernel=dsv4_kernel,
                variant=variant,
                prompts=prompts,
                formatted_prompts=formatted_prompts,
                rank=rank,
                tp_size=tp_size,
            )
            if rank == 0 and report is not None:
                reports.append(report)
                _write_json(
                    Path(args.output).with_suffix(f".{variant.name}.json"),
                    report,
                )
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except BaseException:
                if rank == 0:
                    traceback.print_exc()

    if rank == 0:
        overall_status = "pass"
        if any(report["status"] == "fail" for report in reports):
            overall_status = "fail"
        elif any(report["status"] == "warn" for report in reports):
            overall_status = "warn"
        payload = {
            "status": overall_status,
            "model_path": args.model_path,
            "prompts": list(prompts),
            "variants": reports,
            "config": {
                "tensor_parallel_size": tp_size,
                "page_size": args.page_size,
                "num_pages": args.num_pages,
                "use_pynccl": False,
                "distributed_init_method": distributed_init_method,
                "memory_ratio": args.memory_ratio,
                "max_seq_len": args.max_seq_len,
                "max_extend_tokens": args.max_extend_tokens,
                "max_tokens": args.max_tokens,
            },
        }
        _write_json(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        if overall_status == "fail" or (
            args.fail_on_warning and overall_status == "warn"
        ):
            return 1
        return 0
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek V4 text correctness smoke for human-readable output checks."
    )
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--variants", nargs="*", choices=tuple(_variant_map()))
    parser.add_argument("--prompt", action="append", help="Prompt to run. May be repeated.")
    parser.add_argument("--output", default="/tmp/dsv4_text_smoke.json")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=BASELINE_TP_SIZE,
        help=(
            "Tensor parallel size. Defaults to 8 to match the official TARGET 06 "
            "baseline; pass 1 or another value only for explicit debug runs."
        ),
    )
    parser.add_argument("--tp-rank", type=int, default=None)
    parser.add_argument("--distributed-init-method", default=None)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-extend-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero when text sanity checks warn about possible garbled output.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Answer briefly and clearly.",
    )
    args = parser.parse_args(argv)
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.num_pages <= 1:
        parser.error("--num-pages must be greater than 1")
    if args.max_seq_len <= 0 or args.max_extend_tokens <= 0 or args.max_tokens <= 0:
        parser.error("--max-seq-len, --max-extend-tokens and --max-tokens must be positive")
    if args.memory_ratio <= 0:
        parser.error("--memory-ratio must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_text_smoke(parse_args(argv)))


if __name__ == "__main__":
    main()

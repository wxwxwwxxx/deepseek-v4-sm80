from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark" / "offline" / "deepseek_v4_text_smoke.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("deepseek_v4_text_smoke", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_text_smoke_defaults_match_target06_baseline_shape():
    smoke = _load_module()

    args = smoke.parse_args([])

    assert args.model_path == "/models/DeepSeek-V4-Flash"
    assert args.tensor_parallel_size == 8
    assert args.page_size == 256
    assert args.variants is None
    assert args.max_tokens == 64
    assert args.fail_on_warning is False


def test_text_sanity_accepts_readable_text_and_flags_garbage():
    smoke = _load_module()

    good = smoke.text_sanity("答案是4。 The sky is blue.")
    empty = smoke.text_sanity("   \n")
    replacement = smoke.text_sanity("answer\ufffd")
    repeated = smoke.text_sanity("aaaaaaaaaaaa")
    punctuation = smoke.text_sanity(".????????? ...")

    assert good["looks_sane"] is True
    assert empty["looks_sane"] is False
    assert "empty_output" in empty["issues"]
    assert replacement["looks_sane"] is False
    assert "replacement_character" in replacement["issues"]
    assert repeated["looks_sane"] is False
    assert "long_repeated_character_run" in repeated["issues"]
    assert punctuation["looks_sane"] is False
    assert "mostly_punctuation_or_symbols" in punctuation["issues"]
    assert "long_repeated_symbol_run" in punctuation["issues"]


def test_response_sanity_flags_prompt_echo_and_missing_expected_answer():
    smoke = _load_module()

    good = smoke.response_sanity(
        "答案是4。",
        prompt="请用一句中文回答：2 + 2 等于几？",
        expected_substrings=("4", "四"),
    )
    echoed = smoke.response_sanity(
        ": 2 + 2 等于几？ 2 + 2 ",
        prompt="请用一句中文回答：2 + 2 等于几？",
        expected_substrings=("4", "四"),
    )

    assert good["looks_sane"] is True
    assert echoed["looks_sane"] is False
    assert "prompt_echo_like" in echoed["issues"]
    assert "missing_expected_substring" in echoed["issues"]


def test_format_and_parse_use_model_encoding_when_available(tmp_path):
    smoke = _load_module()
    encoding_dir = tmp_path / "encoding"
    encoding_dir.mkdir()
    (encoding_dir / "encoding_dsv4.py").write_text(
        "\n".join(
            [
                "eos_token = '<eos>'",
                "def encode_messages(messages, thinking_mode):",
                "    return thinking_mode + '::' + '|'.join(m['role'] + '=' + m['content'] for m in messages)",
                "def parse_message_from_completion_text(text, thinking_mode):",
                "    return {'content': text, 'thinking_mode': thinking_mode}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    formatted = smoke.format_chat_prompt(
        "hello",
        model_path=str(tmp_path),
        system_prompt="system",
        thinking_mode="chat",
    )
    parsed = smoke.parse_completion_text("world", model_path=str(tmp_path), thinking_mode="chat")

    assert formatted == "chat::system=system|user=hello"
    assert parsed == {"content": "world<eos>", "thinking_mode": "chat"}


def test_format_chat_prompt_has_plain_fallback(tmp_path):
    smoke = _load_module()

    formatted = smoke.format_chat_prompt(
        "hello",
        model_path=str(tmp_path),
        system_prompt="",
        thinking_mode="chat",
    )

    assert formatted == "User: hello\nAssistant:"


def test_configure_variant_clears_existing_sm80_env_and_sets_v0(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V0_BF16",
            "MINISGL_DSV4_SM80_SWIGLU",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            return (
                name == "MINISGL_DSV4_SM80_SWIGLU"
                and os.environ.get("MINISGL_DSV4_SM80_V0_BF16") == "1"
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_SWIGLU", "1")
    monkeypatch.setenv("MINISGL_DSV4_SM80_FP8_GEMM", "1")

    result = smoke.configure_variant(FakeKernel, smoke._variant_map()["v0_bf16"])

    assert "MINISGL_DSV4_SM80_FP8_GEMM" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"] == {"MINISGL_DSV4_SM80_V0_BF16": "1"}
    assert result["active_dsv4_toggles"] == [
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V0_BF16",
    ]


def test_configure_variant_sets_v1_moe(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"}
                and os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_ROUTE", "1")
    result = smoke.configure_variant(FakeKernel, smoke._variant_map()["v1_moe"])

    assert result["raw_dsv4_sm80_env"] == {"MINISGL_DSV4_SM80_V1_MOE": "1"}
    assert result["active_dsv4_toggles"] == [
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V1_MOE",
    ]


def test_tp_rank_size_defaults_to_tp8_under_torchrun_env(monkeypatch):
    smoke = _load_module()
    args = smoke.parse_args([])

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    assert smoke._tp_rank_size(args) == (2, 8, 8)

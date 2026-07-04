from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark" / "offline" / "deepseek_v4_text_smoke.py"


@pytest.fixture(autouse=True)
def _restore_dsv4_sm80_env():
    original = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("MINISGL_DSV4_SM80_")
    }
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_SM80_"):
            os.environ.pop(name, None)
    yield
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_SM80_"):
            os.environ.pop(name, None)
    os.environ.update(original)


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
    assert args.enable_dsv4_radix_prefix_cache is False
    assert args.enable_dsv4_swa_tail_retention_v1 is False

    enabled_args = smoke.parse_args(["--enable-dsv4-radix-prefix-cache"])
    assert enabled_args.enable_dsv4_radix_prefix_cache is True

    retention_args = smoke.parse_args(["--enable-dsv4-swa-tail-retention-v1"])
    assert retention_args.enable_dsv4_swa_tail_retention_v1 is True


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


def test_configure_variant_sets_moe_v2(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
            "MINISGL_DSV4_SM80_MOE_INT8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1" or (
                os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_INT8", "1")
    result = smoke.configure_variant(FakeKernel, smoke._variant_map()["v1_moe_v2"])

    assert "MINISGL_DSV4_SM80_MOE_INT8" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"] == {
        "MINISGL_DSV4_SM80_MOE_V2": "1",
        "MINISGL_DSV4_SM80_V1_MOE": "1",
    }
    assert result["active_dsv4_toggles"] == [
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_MOE_V2",
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V1_MOE",
    ]


def test_configure_variant_sets_vllm_runner(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
            "MINISGL_DSV4_SM80_MOE_INT8",
            "MINISGL_DSV4_SM80_KV_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = (
                os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_VLLM_RUNNER") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_INT8", "1")
    monkeypatch.setenv("MINISGL_DSV4_SM80_KV_FP8", "1")
    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()[
            "v1_moe_vllm_runner_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ],
    )

    assert "MINISGL_DSV4_SM80_MOE_INT8" in result["cleared_dsv4_sm80_env"]
    assert "MINISGL_DSV4_SM80_KV_FP8" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_VLLM_RUNNER"] == "1"
    active = set(result["active_dsv4_toggles"])
    assert {
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_MOE_V2",
        "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_V1_MOE",
    } <= active
    assert "MINISGL_DSV4_SM80_MOE_INT8" not in active
    assert "MINISGL_DSV4_SM80_KV_FP8" not in active


def test_configure_variant_sets_marlin_wna16_backend(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_SWIGLU",
            "MINISGL_DSV4_SM80_MOE_ROUTE",
            "MINISGL_DSV4_SM80_MOE_INT8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if os.environ.get(name) in {"1", "true"}:
                return True
            moe_bundle = (
                os.environ.get("MINISGL_DSV4_SM80_V1_MOE") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_V2") == "1"
                or os.environ.get("MINISGL_DSV4_SM80_MOE_VLLM_RUNNER") == "1"
            )
            return (
                name in {"MINISGL_DSV4_SM80_SWIGLU", "MINISGL_DSV4_SM80_MOE_ROUTE"} and moe_bundle
            )

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_INT8", "1")
    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_"
            "qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert "MINISGL_DSV4_SM80_MOE_INT8" in result["cleared_dsv4_sm80_env"]
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_VLLM_RUNNER"] == "1"
    assert "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND" not in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_MOE_INT8" not in result["active_dsv4_toggles"]


def test_configure_variant_sets_marlin_wna16_globaltopk(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_INDEXER_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if name == "MINISGL_DSV4_SM80_INDEXER_FP8":
                return False
            return os.environ.get(name) in {"1", "true"}

    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS"] == "1"
    assert "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_INDEXER_FP8" not in result["raw_dsv4_sm80_env"]


def test_configure_variant_sets_marlin_wna16_globaltopk_splitk(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16",
            "MINISGL_DSV4_SM80_INDEXER_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if name == "MINISGL_DSV4_SM80_INDEXER_FP8":
                return False
            return os.environ.get(name) in {"1", "true"}

    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_"
            "fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache"
        ],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16"] == "1"
    assert "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_INDEXER_FP8" not in result["raw_dsv4_sm80_env"]


def test_configure_variant_sets_marlin_wna16_indexer_fp8_cache(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16",
            "MINISGL_DSV4_SM80_REPLAY_METADATA_COPY",
            "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE",
            "MINISGL_DSV4_SM80_INDEXER_FP8",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if name == "MINISGL_DSV4_SM80_INDEXER_FP8":
                return False
            return os.environ.get(name) in {"1", "true"}

    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()[
            "v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_metacopy_"
            "idxfp8cache_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_"
            "idxwqb_gatecache_idxstorecache"
        ],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"] == "marlin_wna16"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_INDEXER_FP8_CACHE"] == "1"
    assert "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE" in result["active_dsv4_toggles"]
    assert "MINISGL_DSV4_SM80_INDEXER_FP8" not in result["raw_dsv4_sm80_env"]


def test_configure_variant_sets_wo_a_bf16_bmm_cache(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_V1_MOE",
            "MINISGL_DSV4_SM80_MOE_V2",
            "MINISGL_DSV4_SM80_MOE_VLLM_RUNNER",
            "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND",
            "MINISGL_DSV4_SM80_GLOBAL_TOPK_LENS",
            "MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16",
            "MINISGL_DSV4_SM80_REPLAY_METADATA_COPY",
            "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE",
            "MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON",
            "MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE",
            "MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE",
            "MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE",
            "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            if name == "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND":
                return False
            if (
                name == "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE"
                and os.environ.get("MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE") == "1"
            ):
                return True
            return os.environ.get(name) in {"1", "true"}

    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()["target0762_woabf16bmmcache"],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE" not in result["raw_dsv4_sm80_env"]
    assert "MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE" in result["active_dsv4_toggles"]


def test_configure_variant_sets_shared_expert_bf16_cache(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            return os.environ.get(name) in {"1", "true"}

    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()["dsv4_sm80_a100_victory_sharedbf16"],
    )

    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE"]
        == "1"
    )
    assert (
        "MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE"
        in result["active_dsv4_toggles"]
    )


def test_configure_variant_sets_bf16_small_gemm_pretranspose(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            return os.environ.get(name) in {"1", "true"}

    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()["dsv4_sm80_a100_victory_bf16smallgemm"],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert (
        result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE"]
        == "1"
    )
    assert (
        "MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE"
        in result["active_dsv4_toggles"]
    )


def test_graph_init_variant_prefers_bf16_small_gemm_pretranspose():
    smoke = _load_module()
    variants = [
        smoke._variant_map()["dsv4_sm80_a100_victory"],
        smoke._variant_map()["dsv4_sm80_a100_victory_bf16smallgemm"],
    ]

    assert (
        smoke._graph_init_variant(variants).name
        == "dsv4_sm80_a100_victory_bf16smallgemm"
    )
    assert (
        smoke._graph_init_variant([smoke._variant_map()["dsv4_sm80_a100_victory"]]).name
        == "dsv4_sm80_a100_victory"
    )


def test_configure_variant_sets_hc_graph_cleanup(monkeypatch):
    smoke = _load_module()

    class FakeKernel:
        DSV4_SM80_KNOWN_TOGGLES = (
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE",
            "MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP",
        )

        @staticmethod
        def dsv4_env_flag(name: str) -> bool:
            return os.environ.get(name) in {"1", "true"}

    result = smoke.configure_variant(
        FakeKernel,
        smoke._variant_map()["dsv4_sm80_a100_victory_hccleanup"],
    )

    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE"] == "1"
    assert result["raw_dsv4_sm80_env"]["MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP"] == "1"
    assert "MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP" in result["active_dsv4_toggles"]


def test_tp_rank_size_defaults_to_tp8_under_torchrun_env(monkeypatch):
    smoke = _load_module()
    args = smoke.parse_args([])

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    assert smoke._tp_rank_size(args) == (2, 8, 8)

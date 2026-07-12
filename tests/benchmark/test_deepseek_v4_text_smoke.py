from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmark/offline/deepseek_v4_text_smoke.py"
SPEC = importlib.util.spec_from_file_location("deepseek_v4_text_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_defaults_are_canonical_optimized_tp8():
    args = smoke.parse_args([])
    assert args.dsv4_runtime == "optimized"
    assert args.tensor_parallel_size == 8
    assert args.page_size == 256
    assert args.max_tokens == 32


def test_explicit_fallback_is_typed_and_preconstruction():
    args = smoke.parse_args(["--dsv4-runtime", "fallback"])
    assert args.dsv4_runtime == "fallback"


def test_semantic_sanity_requires_expected_substring():
    report = smoke.response_sanity(
        "This is printable but wrong.",
        prompt=smoke.DEFAULT_PROMPTS[1],
        token_ids=[1, 2, 3, 4],
        expected_substrings=("blue",),
    )
    assert report["looks_sane"] is False
    assert "missing_expected_substring" in report["issues"]


@pytest.mark.parametrize(
    "token_ids,issue",
    [
        ([7, 7, 7, 7], "repeated_token_loop"),
        ([1, 2, 1, 2, 1, 2], "repeated_short_token_pattern"),
    ],
)
def test_semantic_sanity_rejects_token_loops(token_ids, issue):
    report = smoke.response_sanity(
        "The sky is blue.",
        prompt=smoke.DEFAULT_PROMPTS[1],
        token_ids=token_ids,
        expected_substrings=("blue",),
    )
    assert report["looks_sane"] is False
    assert issue in report["issues"]


def test_semantic_sanity_accepts_correct_nonloop_response():
    report = smoke.response_sanity(
        "The sky is blue on a clear day.",
        prompt=smoke.DEFAULT_PROMPTS[1],
        token_ids=[671, 12709, 344, 8295, 377, 260, 4521, 2173, 16],
        expected_substrings=("blue",),
    )
    assert report["looks_sane"] is True
    assert report["issues"] == []


def test_invalid_runtime_is_rejected():
    with pytest.raises(SystemExit):
        smoke.parse_args(["--dsv4-runtime", "research"])

from __future__ import annotations

from minisgl.server.args import parse_args


def test_server_args_mark_explicit_max_extend_tokens():
    base = ["--model-path", "/tmp/nonexistent-model", "--dtype", "bfloat16"]

    default_config, _ = parse_args(base)
    assert default_config.max_extend_tokens == 8192
    assert default_config.max_extend_tokens_explicit is False

    spaced_config, _ = parse_args(base + ["--max-prefill-length", "8192"])
    assert spaced_config.max_extend_tokens == 8192
    assert spaced_config.max_extend_tokens_explicit is True

    equals_config, _ = parse_args(base + ["--max-extend-length=8192"])
    assert equals_config.max_extend_tokens == 8192
    assert equals_config.max_extend_tokens_explicit is True

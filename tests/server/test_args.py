from __future__ import annotations

import pytest
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


def test_server_args_expose_typed_dsv4_runtime_mode():
    base = ["--model-path", "/tmp/nonexistent-model", "--dtype", "bfloat16"]

    default_config, _ = parse_args(base)
    fallback_config, _ = parse_args(base + ["--dsv4-runtime", "fallback"])

    assert default_config.dsv4_runtime_mode == "optimized"
    assert fallback_config.dsv4_runtime_mode == "fallback"


def test_server_args_resolve_public_served_model_name():
    local_config, _ = parse_args(
        ["--model-path", "/models/DeepSeek-V4-Flash", "--dtype", "bfloat16"]
    )
    trailing_config, _ = parse_args(
        ["--model-path", "/models/DeepSeek-V4-Flash/", "--dtype", "bfloat16"]
    )
    repo_config, _ = parse_args(
        ["--model-path", "deepseek-ai/DeepSeek-V4-Flash", "--dtype", "bfloat16"]
    )
    explicit_config, _ = parse_args(
        [
            "--model-path",
            "/models/DeepSeek-V4-Flash",
            "--dtype",
            "bfloat16",
            "--served-model-name",
            "deepseek-v4-flash",
        ]
    )

    assert local_config.resolved_served_model_name == "DeepSeek-V4-Flash"
    assert trailing_config.resolved_served_model_name == "DeepSeek-V4-Flash"
    assert repo_config.resolved_served_model_name == "deepseek-ai/DeepSeek-V4-Flash"
    assert explicit_config.resolved_served_model_name == "deepseek-v4-flash"


def test_server_args_expose_periodic_stats_controls():
    base = ["--model-path", "/tmp/nonexistent-model", "--dtype", "bfloat16"]

    default_config, _ = parse_args(base)
    custom_config, _ = parse_args(base + ["--stats-log-interval", "3.5", "--disable-log-stats"])

    assert default_config.stats_log_interval == 10.0
    assert default_config.disable_log_stats is False
    assert custom_config.stats_log_interval == 3.5
    assert custom_config.disable_log_stats is True

    with pytest.raises(SystemExit):
        parse_args(base + ["--stats-log-interval", "0"])


@pytest.mark.parametrize("backend", ["fa", "fi", "trtllm", "fa,fi"])
def test_server_args_reject_removed_attention_backends(backend):
    base = ["--model-path", "/tmp/nonexistent-model", "--dtype", "bfloat16"]

    with pytest.raises(SystemExit):
        parse_args(base + ["--attention-backend", backend])


def test_server_help_has_no_removed_backend_or_model_source_options(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--attention-backend" in help_text
    assert "--moe-backend" not in help_text
    assert "--model-source" not in help_text

from __future__ import annotations

from minisgl.server.api_server import OpenAICompletionRequest


def test_vllm_openai_chat_benchmark_payload_is_supported():
    request = OpenAICompletionRequest.model_validate(
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "benchmark prompt"}],
                }
            ],
            "temperature": 0.0,
            "max_completion_tokens": 1024,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    )

    assert request.output_token_limit == 1024
    assert request.messages is not None
    assert request.messages[0].to_prompt_message() == {
        "role": "user",
        "content": "benchmark prompt",
    }


def test_legacy_string_content_and_max_tokens_are_preserved():
    request = OpenAICompletionRequest(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=32,
    )

    assert request.output_token_limit == 32
    assert request.messages is not None
    assert request.messages[0].to_prompt_message()["content"] == "hello"

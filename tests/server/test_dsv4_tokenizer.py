from __future__ import annotations

from types import SimpleNamespace

from minisgl.tokenizer.tokenize import TokenizeManager, load_dsv4_chat_formatter


class FakeTokenizer:
    chat_template = None

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("generic chat template must not be used for DSV4")

    def encode(self, prompt, return_tensors):
        assert prompt == "DSV4:hello"
        assert return_tensors == "pt"
        import torch

        return torch.tensor([[1, 2]], dtype=torch.int64)


def test_dsv4_formatter_is_used_when_hf_chat_template_is_absent():
    manager = TokenizeManager(
        FakeTokenizer(),
        dsv4_chat_formatter=lambda messages, reasoning_effort: f"DSV4:{messages[0]['content']}",
    )
    msg = SimpleNamespace(
        text=[{"role": "user", "content": "hello"}],
        reasoning_effort=None,
    )
    assert manager.tokenize([msg])[0].tolist() == [1, 2]


def test_real_dsv4_formatter_loads_from_local_model():
    formatter = load_dsv4_chat_formatter("/models/DeepSeek-V4-Flash")
    if formatter is None:
        return
    prompt = formatter([{"role": "user", "content": "hello"}], None)
    assert "hello" in prompt
    assert "<｜Assistant｜>" in prompt
    assert prompt.endswith("</think>")

    thinking_prompt = formatter([{"role": "user", "content": "hello"}], "high")
    assert thinking_prompt.endswith("<think>")

    max_prompt = formatter([{"role": "user", "content": "hello"}], "max")
    assert "Reasoning Effort: Absolute maximum" in max_prompt
    assert max_prompt.endswith("<think>")

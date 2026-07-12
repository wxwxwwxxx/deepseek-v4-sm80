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
        dsv4_chat_formatter=lambda messages: f"DSV4:{messages[0]['content']}",
    )
    msg = SimpleNamespace(text=[{"role": "user", "content": "hello"}])
    assert manager.tokenize([msg])[0].tolist() == [1, 2]


def test_real_dsv4_formatter_loads_from_local_model():
    formatter = load_dsv4_chat_formatter("/models/DeepSeek-V4-Flash")
    if formatter is None:
        return
    prompt = formatter([{"role": "user", "content": "hello"}])
    assert "hello" in prompt
    assert "<｜Assistant｜>" in prompt

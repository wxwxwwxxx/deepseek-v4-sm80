from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dsv4_chat_formatter: Callable[[list[dict]], str] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.dsv4_chat_formatter = dsv4_chat_formatter

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                if self.dsv4_chat_formatter is not None:
                    prompt = self.dsv4_chat_formatter(msg.text)
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        msg.text,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results


def load_dsv4_chat_formatter(model_path: str) -> Callable[[list[dict]], str] | None:
    encoding_path = Path(model_path) / "encoding" / "encoding_dsv4.py"
    if not encoding_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("minisgl_dsv4_encoding", encoding_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load DeepSeek V4 encoding from {encoding_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda messages: module.encode_messages(messages, thinking_mode="chat")

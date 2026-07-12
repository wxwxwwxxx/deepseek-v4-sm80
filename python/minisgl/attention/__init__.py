from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

from .base import BaseAttnBackend, BaseAttnMetadata

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

class BackendCreator(Protocol):
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...


SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


@SUPPORTED_ATTENTION_BACKENDS.register("dsv4")
def create_dsv4_backend(config: ModelConfig):
    from .deepseek_v4 import DSV4AttentionBackend

    return DSV4AttentionBackend(config)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    if backend != "auto":
        required_backends = backend.split(",") if "," in backend else [backend]
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(
    backend: str,
    config: ModelConfig,
) -> BaseAttnBackend:
    validate_attn_backend(backend, allow_auto=False)
    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_attn_backend",
]

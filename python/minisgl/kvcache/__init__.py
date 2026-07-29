from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    device: torch.device,
    max_running_req: int | None = None,
    dsv4_swa_num_pages: int | None = None,
    dsv4_dummy_token_start: int | None = None,
) -> BaseKVCachePool:
    if not model_config.is_deepseek_v4:
        raise ValueError("This release supports the DeepSeek V4 cache pool only.")
    from .deepseek_v4_pool import DeepSeekV4KVCache

    return DeepSeekV4KVCache(
        model_config=model_config,
        num_pages=num_pages,
        page_size=page_size,
        device=device,
        max_running_req=max_running_req,
        swa_num_pages=dsv4_swa_num_pages,
        dummy_token_start=dsv4_dummy_token_start,
    )


def estimate_kvcache_bytes_per_page(
    model_config: ModelConfig,
    page_size: int,
    tp_size: int,
) -> int:
    if not model_config.is_deepseek_v4:
        raise ValueError("This release supports the DeepSeek V4 cache pool only.")
    from .deepseek_v4_pool import estimate_deepseek_v4_kvcache_bytes_per_page

    return estimate_deepseek_v4_kvcache_bytes_per_page(model_config, page_size)


def estimate_c128_sequence_state_bytes(
    model_config: ModelConfig,
    max_running_req: int,
) -> int:
    if not model_config.is_deepseek_v4:
        return 0
    from .deepseek_v4_pool import estimate_deepseek_v4_c128_sequence_state_bytes

    return estimate_deepseek_v4_c128_sequence_state_bytes(
        model_config,
        max_running_req,
    )


def estimate_c4_sequence_state_bytes(
    model_config: ModelConfig,
    max_running_req: int,
) -> int:
    if not model_config.is_deepseek_v4:
        return 0
    from .deepseek_v4_pool import estimate_deepseek_v4_c4_sequence_state_bytes

    return estimate_deepseek_v4_c4_sequence_state_bytes(
        model_config,
        max_running_req,
    )


def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device) -> BasePrefixCache:
    return create_radix_cache(device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "estimate_kvcache_bytes_per_page",
    "estimate_c4_sequence_state_bytes",
    "estimate_c128_sequence_state_bytes",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
]

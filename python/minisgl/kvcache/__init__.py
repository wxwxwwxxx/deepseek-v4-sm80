from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

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


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    device: torch.device,
    enable_dsv4_component_loc_ownership: bool = False,
    enable_dsv4_swa_independent_lifecycle: bool = False,
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
        enable_component_loc_ownership=enable_dsv4_component_loc_ownership,
        enable_swa_independent_lifecycle=enable_dsv4_swa_independent_lifecycle,
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


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "estimate_kvcache_bytes_per_page",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]

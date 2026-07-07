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
    dtype: torch.dtype,
    device: torch.device,
    enable_dsv4_component_loc_ownership: bool = False,
    enable_dsv4_swa_independent_lifecycle: bool = False,
    max_running_req: int | None = None,
    dsv4_swa_num_pages: int | None = None,
    dsv4_dummy_token_start: int | None = None,
    dsv4_online_c128_mtp_max_draft_tokens: int = 0,
) -> BaseKVCachePool:
    if model_config.is_deepseek_v4:
        from .deepseek_v4_pool import DeepSeekV4KVCache

        return DeepSeekV4KVCache(
            model_config=model_config,
            num_pages=num_pages,
            page_size=page_size,
            device=device,
            dtype=dtype,
            enable_component_loc_ownership=enable_dsv4_component_loc_ownership,
            enable_swa_independent_lifecycle=enable_dsv4_swa_independent_lifecycle,
            max_running_req=max_running_req,
            swa_num_pages=dsv4_swa_num_pages,
            dummy_token_start=dsv4_dummy_token_start,
            online_c128_mtp_max_draft_tokens=dsv4_online_c128_mtp_max_draft_tokens,
        )

    from .mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


def estimate_kvcache_bytes_per_page(
    model_config: ModelConfig,
    page_size: int,
    dtype: torch.dtype,
    tp_size: int,
) -> int:
    if model_config.is_deepseek_v4:
        from .deepseek_v4_pool import estimate_deepseek_v4_kvcache_bytes_per_page

        return estimate_deepseek_v4_kvcache_bytes_per_page(model_config, page_size)

    from minisgl.utils import div_even

    return (
        2
        * model_config.head_dim
        * div_even(model_config.num_kv_heads, tp_size, allow_replicate=True)
        * page_size
        * dtype.itemsize
        * model_config.num_layers
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


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

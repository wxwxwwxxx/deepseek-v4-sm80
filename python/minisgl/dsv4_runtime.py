from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


DSV4RuntimeMode: TypeAlias = Literal["optimized", "fallback"]


@dataclass(frozen=True)
class DSV4RuntimeConfig:
    mode: DSV4RuntimeMode
    moe_expert_backend: Literal["marlin_wna16", "grouped_fp4"]
    direct_graph_metadata_groups: frozenset[str]
    marlin_release_timing: Literal["before_kv_alloc"] | None
    clear_allocated_page_scope: Literal["component"] | None
    pynccl_max_buffer_bytes: int | None
    release_raw_expert_weights: bool
    marlin_prebuild: bool
    marlin_capacity_credit: bool

    @property
    def optimized(self) -> bool:
        return self.mode == "optimized"


_OPTIMIZED = DSV4RuntimeConfig(
    mode="optimized",
    moe_expert_backend="marlin_wna16",
    direct_graph_metadata_groups=frozenset({"swa", "c4"}),
    marlin_release_timing="before_kv_alloc",
    clear_allocated_page_scope="component",
    pynccl_max_buffer_bytes=32 * 1024 * 1024,
    release_raw_expert_weights=True,
    marlin_prebuild=True,
    marlin_capacity_credit=True,
)

_FALLBACK = DSV4RuntimeConfig(
    mode="fallback",
    moe_expert_backend="grouped_fp4",
    direct_graph_metadata_groups=frozenset(),
    marlin_release_timing=None,
    clear_allocated_page_scope=None,
    pynccl_max_buffer_bytes=None,
    release_raw_expert_weights=False,
    marlin_prebuild=False,
    marlin_capacity_credit=False,
)

_runtime_config = _OPTIMIZED


def resolve_dsv4_runtime_config(mode: DSV4RuntimeMode) -> DSV4RuntimeConfig:
    if mode == "optimized":
        return _OPTIMIZED
    if mode == "fallback":
        return _FALLBACK
    raise ValueError(
        f"Unknown dsv4_runtime_mode={mode!r}; expected 'optimized' or 'fallback'."
    )


def configure_dsv4_runtime(mode: DSV4RuntimeMode) -> DSV4RuntimeConfig:
    global _runtime_config
    _runtime_config = resolve_dsv4_runtime_config(mode)
    return _runtime_config


def get_dsv4_runtime_config() -> DSV4RuntimeConfig:
    return _runtime_config


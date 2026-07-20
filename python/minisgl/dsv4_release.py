"""Immutable DeepSeek V4 SM80 release policy.

This module intentionally has no runtime selector.  Operator-level Torch
references live at their explicit wrapper boundaries; the supported product
runtime is always the qualified optimized SM80 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DSV4ReleaseConfig:
    direct_graph_metadata_groups: frozenset[str]
    marlin_release_timing: Literal["before_kv_alloc"]
    clear_allocated_page_scope: Literal["component"]
    pynccl_max_buffer_bytes: int
    release_raw_expert_weights: bool
    marlin_prebuild: bool
    marlin_capacity_credit: bool


DSV4_RELEASE = DSV4ReleaseConfig(
    direct_graph_metadata_groups=frozenset({"swa", "c4"}),
    marlin_release_timing="before_kv_alloc",
    clear_allocated_page_scope="component",
    pynccl_max_buffer_bytes=32 * 1024 * 1024,
    release_raw_expert_weights=True,
    marlin_prebuild=True,
    marlin_capacity_credit=True,
)


__all__ = ["DSV4_RELEASE", "DSV4ReleaseConfig"]

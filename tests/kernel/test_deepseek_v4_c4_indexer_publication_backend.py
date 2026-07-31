from __future__ import annotations

import math

import pytest
import torch
from minisgl.kernel import deepseek_v4 as dsv4_kernel

from debug.dsv4.kernel import deepseek_v4_reference as dsv4_reference


def _has_sm80_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


pytestmark = pytest.mark.skipif(not _has_sm80_cuda(), reason="requires SM80 CUDA")

PRODUCTION_ROPE = {
    "rotary_dim": 64,
    "base": 160000.0,
    "original_seq_len": 65536,
    "factor": 16.0,
    "beta_fast": 32,
    "beta_slow": 1,
}
PRODUCTION_MAX_POSITION = 1048576


class _FakeIndexerCache:
    def __init__(self, slots: int, *, canary: int = 0xA5) -> None:
        self._page_size = 64
        self._packed = torch.full(
            (
                max(math.ceil(max(slots, 1) / self._page_size), 1),
                self._page_size * (128 + 4),
            ),
            canary,
            dtype=torch.uint8,
            device="cuda",
        )

    def has_indexer_fp8_cache(self) -> bool:
        return True

    def has_indexer_fp8_paged_cache(self) -> bool:
        return True

    def indexer_fp8_paged_cache(self, layer_id: int) -> torch.Tensor:
        assert layer_id == 0
        return self._packed

    @property
    def indexer_fp8_page_size(self) -> int:
        return self._page_size


def _reference_publication(
    cache: _FakeIndexerCache,
    source: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    values = source.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    source.copy_((values * norm_weight.float()).to(source.dtype))
    dsv4_reference.apply_rotary_tail(
        source,
        positions,
        **PRODUCTION_ROPE,
    )
    dsv4_reference.indexer_kv_hadamard_fallback(source)
    dsv4_reference.store_indexer_fp8_cache_fallback(cache, 0, source, loc)


def _candidate_publication(
    cache: _FakeIndexerCache,
    source: torch.Tensor,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    dsv4_reference.compress_norm_rope_store_fallback(
        cache,
        0,
        source,
        loc,
        positions=positions,
        norm_weight=norm_weight,
        rms_norm_eps=1e-6,
        **PRODUCTION_ROPE,
        cache_type="indexer",
        apply_hadamard=True,
    )


@pytest.mark.parametrize("rows", [1, 4, 16, 64, 128, 1024])
@pytest.mark.parametrize("pattern", ["none", "mixed", "all"])
def test_c4_indexer_publication_is_bitwise_exact_and_canary_safe(
    rows: int,
    pattern: str,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17300 + rows)
    original = torch.randn(
        rows,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(rows, dtype=torch.int64, device="cuda") - 3
    if pattern == "none":
        valid = torch.zeros(rows, dtype=torch.bool, device="cuda")
    elif pattern == "all":
        valid = torch.ones(rows, dtype=torch.bool, device="cuda")
    else:
        valid = torch.arange(rows, device="cuda") % 4 == 3
    loc = torch.where(
        valid,
        torch.arange(rows, device="cuda"),
        torch.full((rows,), -1, device="cuda"),
    ).to(torch.int64)

    reference_source = original.clone()
    candidate_source = original.clone()
    reference_cache = _FakeIndexerCache(rows)
    candidate_cache = _FakeIndexerCache(rows)
    _reference_publication(
        reference_cache,
        reference_source,
        positions,
        norm_weight,
        loc,
    )
    _candidate_publication(
        candidate_cache,
        candidate_source,
        positions,
        norm_weight,
        loc,
    )
    torch.cuda.synchronize()

    assert torch.equal(reference_cache._packed, candidate_cache._packed)
    assert torch.equal(reference_source[valid], candidate_source[valid])
    assert torch.equal(original[~valid], candidate_source[~valid])


def test_c4_indexer_publication_crosses_fp8_page_boundary_exactly() -> None:
    rows = 16
    generator = torch.Generator(device="cuda").manual_seed(17400)
    original = torch.randn(
        rows,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    positions = torch.arange(249, 249 + rows, dtype=torch.int64, device="cuda")
    valid = torch.arange(rows, device="cuda") % 4 == 3
    loc = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    loc[valid] = torch.arange(62, 62 + int(valid.sum()), device="cuda")

    reference_source = original.clone()
    candidate_source = original.clone()
    reference_cache = _FakeIndexerCache(128)
    candidate_cache = _FakeIndexerCache(128)
    _reference_publication(
        reference_cache,
        reference_source,
        positions,
        norm_weight,
        loc,
    )
    _candidate_publication(
        candidate_cache,
        candidate_source,
        positions,
        norm_weight,
        loc,
    )
    torch.cuda.synchronize()

    assert torch.equal(reference_cache._packed, candidate_cache._packed)
    assert torch.equal(reference_source[valid], candidate_source[valid])
    assert torch.equal(original[~valid], candidate_source[~valid])


@pytest.mark.parametrize(
    ("scenario", "publication_positions", "valid_mask", "valid_locs"),
    [
        (
            "no_valid_rows",
            [0, 4, 65532, 65536],
            [False, False, False, False],
            [],
        ),
        (
            "one_valid_row",
            [65540, 131068, 131072, 524284],
            [False, True, False, False],
            [7],
        ),
        (
            "natural_mixed_rows",
            [0, 4, 65532, 65536, 65540, 131068, 131072, 524284],
            [False, True, False, True, False, True, False, True],
            [0, 1, 63, 64],
        ),
        (
            "all_valid_rows",
            [524288, 1048568, 1048572],
            [True, True, True],
            [61, 62, 63],
        ),
        (
            "fp8_page_boundary_locations",
            [65532, 131068, 524284, 1048568],
            [True, True, True, True],
            [62, 63, 64, 65],
        ),
        (
            "ragged_independent_high_starts",
            [65536, 131072, 524288, 1048572, 65540, 131068, 524284, 1048568],
            [True, False, True, False, False, True, False, True],
            [3, 11, 64, 79],
        ),
    ],
)
def test_c4_indexer_publication_production_high_positions_are_bitwise_exact(
    scenario: str,
    publication_positions: list[int],
    valid_mask: list[bool],
    valid_locs: list[int],
) -> None:
    assert max(publication_positions) < PRODUCTION_MAX_POSITION, scenario
    rows = len(publication_positions)
    generator = torch.Generator(device="cuda").manual_seed(17500 + sum(publication_positions) % 997)
    original = torch.randn(
        rows,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    norm_weight = torch.randn(
        128,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    positions = torch.tensor(
        publication_positions,
        dtype=torch.int64,
        device="cuda",
    )
    valid = torch.tensor(valid_mask, dtype=torch.bool, device="cuda")
    loc = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    if valid_locs:
        loc[valid] = torch.tensor(valid_locs, dtype=torch.int64, device="cuda")

    reference_source = original.clone()
    candidate_source = original.clone()
    slots = max(valid_locs, default=0) + 1
    reference_cache = _FakeIndexerCache(slots)
    candidate_cache = _FakeIndexerCache(slots)
    _reference_publication(
        reference_cache,
        reference_source,
        positions,
        norm_weight,
        loc,
    )
    _candidate_publication(
        candidate_cache,
        candidate_source,
        positions,
        norm_weight,
        loc,
    )
    torch.cuda.synchronize()

    assert torch.equal(reference_cache._packed, candidate_cache._packed)
    assert torch.equal(reference_source[valid], candidate_source[valid])
    assert torch.equal(original[~valid], candidate_source[~valid])
    reference_dequant = dsv4_reference.dequantize_indexer_fp8_paged_cache_ref(
        reference_cache._packed,
        page_size=reference_cache.indexer_fp8_page_size,
        dim=128,
    )
    candidate_dequant = dsv4_reference.dequantize_indexer_fp8_paged_cache_ref(
        candidate_cache._packed,
        page_size=candidate_cache.indexer_fp8_page_size,
        dim=128,
    )
    assert torch.equal(reference_dequant, candidate_dequant)


def test_c4_indexer_publication_graph_replay_is_stable_and_allocation_free() -> None:
    dsv4_kernel.warmup_indexer_fp8_backend(
        torch.device("cuda"),
        base=PRODUCTION_ROPE["base"],
        original_seq_len=PRODUCTION_ROPE["original_seq_len"],
        factor=PRODUCTION_ROPE["factor"],
        beta_fast=PRODUCTION_ROPE["beta_fast"],
        beta_slow=PRODUCTION_ROPE["beta_slow"],
        page_size=64,
    )
    rows = 128
    original = torch.randn(rows, 128, dtype=torch.bfloat16, device="cuda")
    source = torch.empty_like(original)
    high_positions = torch.tensor(
        [0, 4, 65532, 65536, 65540, 131068, 131072, 524284, 524288, 1048568, 1048572],
        dtype=torch.int64,
        device="cuda",
    )
    positions = high_positions[
        torch.arange(rows, dtype=torch.int64, device="cuda") % high_positions.numel()
    ].contiguous()
    norm_weight = torch.randn(128, dtype=torch.float32, device="cuda")
    valid = torch.arange(rows, device="cuda") % 4 == 3
    loc = torch.where(
        valid,
        torch.arange(rows, device="cuda"),
        torch.full((rows,), -1, device="cuda"),
    ).to(torch.int64)
    cache = _FakeIndexerCache(rows)
    eager_source = original.clone()
    eager_cache = _FakeIndexerCache(rows)
    _candidate_publication(
        eager_cache,
        eager_source,
        positions,
        norm_weight,
        loc,
    )
    torch.cuda.synchronize()

    def invoke() -> None:
        source.copy_(original)
        _candidate_publication(cache, source, positions, norm_weight, loc)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    pointers = (
        source.data_ptr(),
        original.data_ptr(),
        positions.data_ptr(),
        norm_weight.data_ptr(),
        loc.data_ptr(),
        cache._packed.data_ptr(),
    )
    first = torch.empty_like(cache._packed)
    repeat = torch.empty_like(cache._packed)
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    graph.replay()
    first.copy_(cache._packed)
    graph.replay()
    repeat.copy_(cache._packed)
    torch.cuda.synchronize()
    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()

    assert pointers == (
        source.data_ptr(),
        original.data_ptr(),
        positions.data_ptr(),
        norm_weight.data_ptr(),
        loc.data_ptr(),
        cache._packed.data_ptr(),
    )
    assert torch.equal(first, repeat)
    assert torch.equal(first, eager_cache._packed)
    assert torch.equal(source, eager_source)
    assert after_allocated == before_allocated
    assert after_reserved == before_reserved

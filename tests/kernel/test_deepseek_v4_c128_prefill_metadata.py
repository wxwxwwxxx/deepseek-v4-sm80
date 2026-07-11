from __future__ import annotations

import pytest
import torch
from minisgl.kernel import deepseek_v4 as dsv4_kernel


def _has_sm80_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _oracle(
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    width: int,
    component_page_size: int,
) -> torch.Tensor:
    rows = lengths.numel()
    cols = torch.arange(width, dtype=torch.long, device=page_table.device)
    logical_pages = cols.div(component_page_size, rounding_mode="floor")
    page_in_table = logical_pages < page_table.shape[1]
    safe_pages = logical_pages.clamp(max=max(page_table.shape[1] - 1, 0))
    if page_table.shape[1] == 0:
        pages = torch.full((rows, width), -1, dtype=torch.int32, device=page_table.device)
    else:
        pages = torch.gather(page_table, 1, safe_pages[None, :].expand(rows, -1))
    locs = pages * component_page_size + (cols % component_page_size).to(torch.int32)
    valid = (cols[None, :] < lengths[:, None]) & page_in_table[None, :] & (pages >= 0)
    return torch.where(valid, locs, torch.full_like(locs, -1))


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
@pytest.mark.parametrize("width", [1, 2, 3, 127, 128, 129, 512])
def test_c128_prefill_one_surface_matches_exact_oracle(width: int):
    # A full-cache page_size of 256 gives two C128 slots per component page.
    component_page_size = 256 // 128
    rows = 8
    table_width = max((width + component_page_size - 1) // component_page_size, 1)
    page_table = torch.arange(
        rows * table_width,
        dtype=torch.int32,
        device="cuda",
    ).reshape(rows, table_width)
    page_table[1, 0] = -1
    page_table[3, -1] = -1
    boundary_lengths = torch.tensor(
        [0, 1, 2, 3, 127, 128, 129, width],
        dtype=torch.int32,
        device="cuda",
    ).clamp(max=width)

    backend: list[str] = []
    output = dsv4_kernel.c128_prefill_page_indices_one_surface(
        page_table,
        boundary_lengths,
        width=width,
        component_page_size=component_page_size,
        _backend=backend,
    )

    assert output is not None
    assert output.dtype == torch.int32
    assert output.shape == (rows, width)
    assert backend == ["triton_c128_prefill_one_surface"]
    assert torch.equal(
        output,
        _oracle(
            page_table,
            boundary_lengths,
            width=width,
            component_page_size=component_page_size,
        ),
    )


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_c128_prefill_one_surface_full_token_boundaries_127_128_129():
    seq_lengths = torch.tensor([127, 128, 129], dtype=torch.int32, device="cuda")
    c128_lengths = seq_lengths.div(128, rounding_mode="floor")
    page_table = torch.tensor([[7], [11], [13]], dtype=torch.int32, device="cuda")

    output = dsv4_kernel.c128_prefill_page_indices_one_surface(
        page_table,
        c128_lengths,
        width=2,
        component_page_size=2,
    )

    assert output is not None
    assert c128_lengths.tolist() == [0, 1, 1]
    assert output.tolist() == [[-1, -1], [22, -1], [26, -1]]


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_c128_prefill_one_surface_supports_preallocated_output_and_missing_pages():
    rows, width = 16, 2048
    component_page_size = 2
    # Deliberately make the table too short for the last four logical pages.
    table_width = width // component_page_size - 4
    page_table = torch.arange(
        rows * table_width,
        dtype=torch.int32,
        device="cuda",
    ).reshape(rows, table_width)
    page_table[:, 5::37] = -1
    lengths = (torch.arange(rows, dtype=torch.int32, device="cuda") * 137) % (width + 1)
    output = torch.full((rows, width), 123, dtype=torch.int32, device="cuda")

    result = dsv4_kernel.c128_prefill_page_indices_one_surface(
        page_table,
        lengths,
        width=width,
        component_page_size=component_page_size,
        out=output,
    )

    assert result is output
    assert torch.equal(
        output,
        _oracle(
            page_table,
            lengths,
            width=width,
            component_page_size=component_page_size,
        ),
    )


def test_c128_prefill_one_surface_rejects_non_cuda_inputs():
    page_table = torch.zeros((2, 4), dtype=torch.int32)
    lengths = torch.ones(2, dtype=torch.int32)
    assert (
        dsv4_kernel.c128_prefill_page_indices_one_surface(
            page_table,
            lengths,
            width=8,
            component_page_size=2,
        )
        is None
    )

#!/usr/bin/env python3
"""Executable TARGET 16.15 C128 write-before-read poison proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minisgl.kernel.deepseek_v4 import c128_online_pool_and_update_fallback


RATIO = 128
HEAD_DIM = 8
SLOTS = 3
ALIGNED_STARTS = (0, 256, 512)
PARTITIONS = {
    "all_at_once": [128],
    "one_token_at_a_time": [1] * 128,
    "127_1": [127, 1],
    "1_127": [1, 127],
    "17_31_80": [17, 31, 80],
    "64_63_1": [64, 63, 1],
}
PRIOR_CASES = {
    "raw_poison": ("poison", 0),
    "abort_phase_1": ("abort", 1),
    "abort_phase_63": ("abort", 63),
    "abort_phase_127": ("abort", 127),
    "completed_128": ("completion", 128),
    "completed_256": ("completion", 256),
}


def _inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1615)
    projected = torch.randn(
        640,
        2 * HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    ape = torch.randn(
        RATIO,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    return projected.to(device), ape.to(device)


def _poison_slot(state: torch.Tensor, slot: int, family: str) -> None:
    rows = state[slot * RATIO : (slot + 1) * RATIO]
    row_index = torch.arange(RATIO, device=state.device).unsqueeze(1)
    if family == "finite_extreme":
        sign = torch.where(row_index.remainder(2) == 0, 1.0, -1.0)
        rows[:, :HEAD_DIM] = sign * 1.0e20
        rows[:, HEAD_DIM:] = -sign * 1.0e20
        return
    if family == "nan_inf":
        rows.fill_(float("nan"))
        rows[1::3, :HEAD_DIM] = float("inf")
        rows[2::3, :HEAD_DIM] = float("-inf")
        rows[1::3, HEAD_DIM:] = float("-inf")
        rows[2::3, HEAD_DIM:] = float("inf")
        return
    raise ValueError(f"unknown poison family: {family}")


def _expected_group(
    projected: torch.Tensor,
    ape: torch.Tensor,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = projected[start : start + RATIO].clone()
    rows[:, HEAD_DIM:] += ape
    weights = rows[:, HEAD_DIM:].softmax(dim=0)
    output = (rows[:, :HEAD_DIM] * weights).sum(dim=0).to(torch.bfloat16)
    return rows, output


def _assert_exact(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True):
        return
    finite = bool(torch.isfinite(actual.float()).all().item())
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.nan_to_num(nan=float("inf")).max().item())
    raise AssertionError(
        f"{label}: mismatch, finite={finite}, max_abs={max_abs}, "
        f"actual={actual.detach().cpu()}, expected={expected.detach().cpu()}"
    )


def _run_rows(
    state: torch.Tensor,
    projected_rows: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    slot: int,
) -> torch.Tensor:
    table_indices = torch.full_like(positions, slot)
    return c128_online_pool_and_update_fallback(
        projected_rows,
        state,
        ape,
        positions,
        table_indices,
    )


def _write_prior_owner(
    state: torch.Tensor,
    projected: torch.Tensor,
    ape: torch.Tensor,
    slot: int,
    kind: str,
    length: int,
) -> None:
    if kind == "poison":
        return
    positions = torch.arange(length, device=state.device, dtype=torch.int32)
    _run_rows(state, projected[:length], ape, positions, slot)


def _run_partitioned_group(
    *,
    label: str,
    state: torch.Tensor,
    projected: torch.Tensor,
    ape: torch.Tensor,
    slot: int,
    start: int,
    chunks: list[int],
) -> float:
    expected_state, expected_output = _expected_group(projected, ape, start)
    offset = 0
    completed = None
    max_abs = 0.0
    for chunk in chunks:
        end = offset + chunk
        positions = torch.arange(
            start + offset,
            start + end,
            device=state.device,
            dtype=torch.int32,
        )
        produced = _run_rows(
            state,
            projected[start + offset : start + end],
            ape,
            positions,
            slot,
        )
        phase = (start + end) % RATIO
        valid_rows = RATIO if phase == 0 else phase
        actual_valid = state[
            slot * RATIO : slot * RATIO + valid_rows
        ]
        expected_valid = expected_state[:valid_rows]
        _assert_exact(f"{label}/valid_partial_rows_at_{start + end}", actual_valid, expected_valid)
        if phase == 0:
            completed = produced[-1]
        offset = end
    if offset != RATIO or completed is None:
        raise AssertionError(f"{label}: partition did not close exactly one C128 group")
    if not bool(torch.isfinite(completed.float()).all().item()):
        raise AssertionError(f"{label}: completed output contains poison/non-finite values")
    _assert_exact(f"{label}/completed_output", completed, expected_output)
    if completed.numel():
        max_abs = float((completed.float() - expected_output.float()).abs().max().item())
    return max_abs


def _run_matrix(
    state: torch.Tensor,
    projected: torch.Tensor,
    ape: torch.Tensor,
) -> dict[str, object]:
    cases = 0
    max_abs = 0.0
    by_family: dict[str, int] = {}
    for family in ("finite_extreme", "nan_inf"):
        family_cases = 0
        for start in ALIGNED_STARTS:
            for partition_name, chunks in PARTITIONS.items():
                for prior_name, (prior_kind, prior_length) in PRIOR_CASES.items():
                    state.zero_()
                    _poison_slot(state, 0, family)
                    _write_prior_owner(
                        state,
                        projected,
                        ape,
                        0,
                        prior_kind,
                        prior_length,
                    )
                    label = (
                        f"family={family}/start={start}/partition={partition_name}/"
                        f"prior={prior_name}"
                    )
                    max_abs = max(
                        max_abs,
                        _run_partitioned_group(
                            label=label,
                            state=state,
                            projected=projected,
                            ape=ape,
                            slot=0,
                            start=start,
                            chunks=chunks,
                        ),
                    )
                    cases += 1
                    family_cases += 1
        by_family[family] = family_cases
    return {
        "cases": cases,
        "cases_by_poison_family": by_family,
        "max_completed_output_abs_error": max_abs,
    }


def _run_concurrent_slots_and_dummy(
    state: torch.Tensor,
    projected: torch.Tensor,
    ape: torch.Tensor,
) -> dict[str, object]:
    state.zero_()
    _poison_slot(state, 0, "finite_extreme")
    _poison_slot(state, 1, "nan_inf")
    _poison_slot(state, 2, "nan_inf")
    dummy_before = state[2 * RATIO :].clone()

    _run_rows(
        state,
        projected[:17],
        ape,
        torch.arange(0, 17, device=state.device, dtype=torch.int32),
        0,
    )
    _run_rows(
        state,
        projected[256:319],
        ape,
        torch.arange(256, 319, device=state.device, dtype=torch.int32),
        1,
    )
    expected0, output0 = _expected_group(projected, ape, 0)
    expected1, output1 = _expected_group(projected, ape, 256)
    _assert_exact("concurrent/slot0_partial", state[:17], expected0[:17])
    _assert_exact(
        "concurrent/slot1_partial",
        state[RATIO : RATIO + 63],
        expected1[:63],
    )
    _assert_exact("concurrent/dummy_untouched", state[2 * RATIO :], dummy_before)

    produced0 = _run_rows(
        state,
        projected[17:128],
        ape,
        torch.arange(17, 128, device=state.device, dtype=torch.int32),
        0,
    )
    produced1 = _run_rows(
        state,
        projected[319:384],
        ape,
        torch.arange(319, 384, device=state.device, dtype=torch.int32),
        1,
    )
    _assert_exact("concurrent/slot0_completed", produced0[-1], output0)
    _assert_exact("concurrent/slot1_completed", produced1[-1], output1)

    # Captured decode pads the stable table-index vector with the dedicated
    # physical dummy slot. Complete a live group beside one independently
    # poisoned dummy row and prove the dummy cannot affect the live output.
    state.zero_()
    _poison_slot(state, 0, "finite_extreme")
    _poison_slot(state, 2, "nan_inf")
    _run_rows(
        state,
        projected[:127],
        ape,
        torch.arange(0, 127, device=state.device, dtype=torch.int32),
        0,
    )
    live_before_dummy = state[:RATIO].clone()
    mixed_projected = torch.stack([projected[127], projected[512]])
    mixed_positions = torch.tensor([127, 0], device=state.device, dtype=torch.int32)
    mixed_tables = torch.tensor([0, 2], device=state.device, dtype=torch.int32)
    mixed_output = c128_online_pool_and_update_fallback(
        mixed_projected,
        state,
        ape,
        mixed_positions,
        mixed_tables,
    )
    _, expected_live = _expected_group(projected, ape, 0)
    _assert_exact("graph_dummy/live_completed_output", mixed_output[0], expected_live)
    _assert_exact(
        "graph_dummy/live_prior_rows",
        state[: RATIO - 1],
        live_before_dummy[: RATIO - 1],
    )
    if not bool(torch.isfinite(mixed_output[0].float()).all().item()):
        raise AssertionError("graph_dummy/live output is non-finite")
    return {
        "two_live_slots_different_poison_and_phases": "pass",
        "graph_dummy_independent_poison": "pass",
        "graph_dummy_live_output_max_abs_error": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("C128 poison proof requires CUDA/Triton")

    device = torch.device("cuda")
    projected, ape = _inputs(device)
    state = torch.empty(
        SLOTS * RATIO,
        2 * HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    report = {
        "status": "pass",
        "device": torch.cuda.get_device_name(device),
        "contract": (
            "every C128 row contributing to a completed output is written by "
            "the current owner before read"
        ),
        "aligned_starts": list(ALIGNED_STARTS),
        "partitions": list(PARTITIONS),
        "prior_owner_cases": list(PRIOR_CASES),
        "matrix": _run_matrix(state, projected, ape),
        "concurrent_and_graph": _run_concurrent_slots_and_dummy(
            state,
            projected,
            ape,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

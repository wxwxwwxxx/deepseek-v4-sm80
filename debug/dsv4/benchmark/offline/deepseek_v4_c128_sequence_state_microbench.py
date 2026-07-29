#!/usr/bin/env python3
"""No-weight TARGET 16.1 C128 boundary/partition contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from minisgl.kernel.deepseek_v4 import c128_online_pool_and_update

POSITIONS = (0, 1, 2, 126, 127, 128, 129, 254, 255, 256, 257, 510, 511, 512, 513)
TOTAL = max(POSITIONS) + 1
RATIO = 128
HEAD_DIM = 8
PAGE_SIZE = 256


def _partitions(total: int) -> dict[str, list[int]]:
    candidates = {
        "all_at_once": [total],
        "one_token_at_a_time": [1] * total,
        "127_1": [127, 1, total - 128],
        "128": [128, total - 128],
        "255_1": [255, 1, total - 256],
        "256": [256, total - 256],
        "ragged": [3, 61, 63, 1, 17, 111, 5, 122, 2, 127, 1, 1],
        "release_256": [256, 256, total - 512],
    }
    result: dict[str, list[int]] = {}
    for name, chunks in candidates.items():
        clean: list[int] = []
        remaining = total
        for chunk in chunks:
            if remaining <= 0:
                break
            take = min(max(int(chunk), 0), remaining)
            if take:
                clean.append(take)
                remaining -= take
        if remaining:
            clean.append(remaining)
        result[name] = clean
    return result


def _inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1601)
    projected = torch.randn(TOTAL, 2 * HEAD_DIM, generator=generator, dtype=torch.float32)
    ape = torch.randn(RATIO, HEAD_DIM, generator=generator, dtype=torch.float32)
    return projected.to(device), ape.to(device)


def _official(
    projected: torch.Tensor, ape: torch.Tensor
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    outputs: dict[int, torch.Tensor] = {}
    active: dict[int, torch.Tensor] = {}
    state = torch.empty(RATIO, 2 * HEAD_DIM, dtype=torch.float32, device=projected.device)
    state[:, :HEAD_DIM].zero_()
    state[:, HEAD_DIM:].fill_(float("-inf"))
    for pos in range(TOTAL):
        slot = pos % RATIO
        state[slot, :HEAD_DIM] = projected[pos, :HEAD_DIM]
        state[slot, HEAD_DIM:] = projected[pos, HEAD_DIM:] + ape[slot]
        if (pos + 1) % RATIO == 0:
            weights = state[:, HEAD_DIM:].softmax(dim=0)
            outputs[pos] = (state[:, :HEAD_DIM] * weights).sum(dim=0).to(torch.bfloat16)
        if pos in POSITIONS:
            phase = (pos + 1) % RATIO
            active[pos] = state[:phase].clone() if phase else state.new_empty((0, 2 * HEAD_DIM))
    return outputs, active


def _sequence_run(
    projected: torch.Tensor,
    ape: torch.Tensor,
    chunks: list[int],
    total: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    device = projected.device
    state = torch.empty(RATIO, 2 * HEAD_DIM, dtype=torch.float32, device=device)
    state[:, :HEAD_DIM].zero_()
    state[:, HEAD_DIM:].fill_(float("-inf"))
    outputs: dict[int, torch.Tensor] = {}
    active: dict[int, torch.Tensor] = {}
    start = 0
    for chunk in chunks:
        end = start + chunk
        positions = torch.arange(start, end, dtype=torch.int32, device=device)
        table_indices = torch.zeros(chunk, dtype=torch.int32, device=device)
        produced = c128_online_pool_and_update(
            projected[start:end],
            state,
            ape,
            positions,
            table_indices,
        )
        for local, pos in enumerate(range(start, end)):
            if (pos + 1) % RATIO == 0:
                outputs[pos] = produced[local].clone()
            if pos in POSITIONS:
                phase = (pos + 1) % RATIO
                active[pos] = state[:phase].clone() if phase else state.new_empty((0, 2 * HEAD_DIM))
        start = end
    assert start == total
    return outputs, active


def _error(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    if lhs.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0}
    diff = (lhs.float() - rhs.float()).abs()
    return {"max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())}


def _hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The production C128 fixture requires CUDA/Triton")
    device = torch.device("cuda")
    saved_fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    projected = saved_fixture["projected"].to(device)
    ape = saved_fixture["ape"].to(device)
    official_outputs, official_active = _official(projected, ape)
    report: dict[str, object] = {
        "mode": "sequence_owned_vs_official_and_pre_edit_fixture",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "positions": list(POSITIONS),
        "partitions": {},
    }
    for name in _partitions(TOTAL):
        endpoint_cases: dict[str, object] = {}
        for endpoint in POSITIONS:
            total = endpoint + 1
            chunks = _partitions(total)[name]
            outputs, active = _sequence_run(projected, ape, chunks, total)
            final_active = active[endpoint]
            legacy = saved_fixture["cases"][name][endpoint]
            endpoint_cases[str(endpoint)] = {
                "chunks": chunks if name != "one_token_at_a_time" else f"{total}x1",
                "completed_output_errors": {
                    str(pos): _error(value, official_outputs[pos]) for pos, value in outputs.items()
                },
                "active_state_error": _error(final_active, official_active[endpoint]),
                "pre_edit_active_state_error": _error(final_active, legacy["active"].to(device)),
                "completed_output_hashes": {
                    str(pos): _hash(value) for pos, value in outputs.items()
                },
                "active_state_hash": _hash(final_active),
                "pre_edit_completed_output_errors": {
                    str(pos): _error(value, legacy["outputs"][pos].to(device))
                    for pos, value in outputs.items()
                },
            }
        report["partitions"][name] = endpoint_cases
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

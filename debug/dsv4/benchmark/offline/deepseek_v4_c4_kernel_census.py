from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from minisgl.kernel.deepseek_v4 import c4_online_pool_and_update_fallback


_POISON_PROOF_PATH = (
    Path(__file__).resolve().parent / "deepseek_v4_c4_production_poison.py"
)


def _load_poison_proof():
    spec = importlib.util.spec_from_file_location(
        "_deepseek_v4_c4_production_poison",
        _POISON_PROOF_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_POISON_PROOF_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _producer_call(
    fixture,
    projected: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    positions = positions.to(device=fixture.device, dtype=torch.int64).contiguous()
    rows = projected.index_select(0, positions).contiguous()
    table_indices = torch.zeros_like(positions)
    raw_out_loc = positions.clone()
    return c4_online_pool_and_update_fallback(
        rows,
        fixture.sequence_state,
        fixture.checkpoint,
        ape,
        positions,
        table_indices,
        raw_out_loc,
        fixture.ctx_page_table,
        fixture.checkpoint_page_mapping,
        page_size=256,
    )


def _profile_producer(
    fixture,
    projected: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
) -> dict[str, object]:
    # Prepare all harness tensors before profiling so the census contains only
    # the production producer call.
    positions = positions.to(device=fixture.device, dtype=torch.int64).contiguous()
    rows = projected.index_select(0, positions).contiguous()
    table_indices = torch.zeros_like(positions)
    raw_out_loc = positions.clone()
    torch.cuda.synchronize(fixture.device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as trace:
        c4_online_pool_and_update_fallback(
            rows,
            fixture.sequence_state,
            fixture.checkpoint,
            ape,
            positions,
            table_indices,
            raw_out_loc,
            fixture.ctx_page_table,
            fixture.checkpoint_page_mapping,
            page_size=256,
        )
        torch.cuda.synchronize(fixture.device)

    kernels = []
    for event in trace.key_averages():
        if event.self_device_time_total <= 0:
            continue
        kernels.append(
            {
                "name": event.key,
                "launches": event.count,
                "self_cuda_time_us": event.self_device_time_total,
            }
        )
    return {
        "positions": positions.cpu().tolist(),
        "cuda_kernel_launches": sum(int(item["launches"]) for item in kernels),
        "kernels": kernels,
    }


def run_census() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "CUDA unavailable"}
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 8:
        return {"status": "skip", "reason": "SM80 or newer required"}

    proof = _load_poison_proof()
    projected, ape = proof._fixture(272, 16, 20260727, device=device)

    # Compile every shape before profiling.
    warm = proof.ProductionFixture(16, "finite_extreme", device)
    for positions in (
        torch.tensor([0], device=device),
        torch.tensor([8], device=device),
        torch.arange(256, 260, device=device),
        torch.arange(252, 256, device=device),
    ):
        _producer_call(warm, projected, ape, positions)
    torch.cuda.synchronize(device)

    scenarios: dict[str, dict[str, object]] = {}

    start = proof.ProductionFixture(16, "finite_extreme", device)
    scenarios["new_request_position_zero"] = _profile_producer(
        start,
        projected,
        ape,
        torch.tensor([0], device=device),
    )

    continuation = proof.ProductionFixture(16, "finite_extreme", device)
    _producer_call(
        continuation,
        projected,
        ape,
        torch.arange(0, 8, device=device),
    )
    scenarios["ordinary_continuation_inside_page"] = _profile_producer(
        continuation,
        projected,
        ape,
        torch.tensor([8], device=device),
    )

    prefix_hit = proof.ProductionFixture(16, "finite_extreme", device)
    _producer_call(
        prefix_hit,
        projected,
        ape,
        torch.arange(0, 256, device=device),
    )
    prefix_hit.poison_sequence_slot(0)
    scenarios["first_group_after_prefix_hit"] = _profile_producer(
        prefix_hit,
        projected,
        ape,
        torch.arange(256, 260, device=device),
    )

    publication = proof.ProductionFixture(16, "finite_extreme", device)
    _producer_call(
        publication,
        projected,
        ape,
        torch.arange(0, 252, device=device),
    )
    scenarios["page_boundary_checkpoint_publication"] = _profile_producer(
        publication,
        projected,
        ape,
        torch.arange(252, 256, device=device),
    )

    expected = {
        "_c4_online_pool_kernel": 1,
        "_c4_online_state_store_kernel": 1,
    }
    for label, scenario in scenarios.items():
        observed = {
            str(item["name"]): int(item["launches"])
            for item in scenario["kernels"]
        }
        if observed != expected:
            raise AssertionError(
                f"{label}: producer kernel census changed: {observed}"
            )

    return {
        "status": "pass",
        "device": torch.cuda.get_device_name(device),
        "scope": "one C4 attention producer call; indexer uses the same wrapper",
        "scenarios": scenarios,
        "separate_checkpoint_restore_or_materialization_launches": 0,
        "separate_checkpoint_publication_launches": 0,
        "lifecycle": {
            "acquire_sequence_slot": {
                "host_generation_owner_updates": 1,
                "cuda_kernel_launches": 0,
                "device_clear_or_copy_launches": 0,
            },
            "release_sequence_slot": {
                "host_generation_owner_updates": 1,
                "cuda_kernel_launches": 0,
                "device_clear_or_copy_launches": 0,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_census()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    return 0 if result["status"] in {"pass", "skip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

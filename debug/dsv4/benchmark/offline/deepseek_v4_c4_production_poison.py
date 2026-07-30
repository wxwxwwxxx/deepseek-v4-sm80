from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from minisgl.kernel.deepseek_v4 import c4_online_pool_and_update

PAGE_SIZE = 256
RATIO = 4
RING_SIZE = 8
LIVE_SLOTS = 2
SEQUENCE_SLOTS = LIVE_SLOTS + 1
PHYSICAL_PAGES = 4
LOGICAL_PAGES = 4
MAX_LOGICAL_TOKENS = LOGICAL_PAGES * PAGE_SIZE


def _poison(
    shape: tuple[int, ...],
    family: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    size = math.prod(shape)
    if family == "finite_extreme":
        flat = torch.full((size,), 1.0e20, dtype=torch.float32, device=device)
        flat[1::2] = -1.0e20
    elif family == "nan_inf":
        values = torch.tensor(
            [float("nan"), float("inf"), float("-inf")],
            dtype=torch.float32,
            device=device,
        )
        flat = values[torch.arange(size, device=device) % values.numel()]
    else:
        raise ValueError(f"unknown poison family: {family}")
    return flat.view(shape).clone()


def _fixture(
    length: int,
    head_dim: int,
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    projected = torch.randn(
        (length, 4 * head_dim),
        generator=generator,
        dtype=torch.float32,
    )
    ape = torch.randn(
        (RATIO, 2 * head_dim),
        generator=generator,
        dtype=torch.float32,
    )
    projected[:, 2 * head_dim :] *= 2.0
    ape *= 1.5
    return projected.to(device), ape.to(device)


def _official_output(
    projected: torch.Tensor,
    ape: torch.Tensor,
    end_pos: int,
) -> torch.Tensor:
    head_dim = projected.shape[-1] // 4
    kv_rows: list[torch.Tensor] = []
    score_rows: list[torch.Tensor] = []
    for source_slot in range(RING_SIZE):
        pos = end_pos - (RING_SIZE - 1 - source_slot)
        if pos < 0:
            kv_rows.append(torch.zeros(head_dim, dtype=torch.float32, device=projected.device))
            score_rows.append(
                torch.full(
                    (head_dim,),
                    float("-inf"),
                    dtype=torch.float32,
                    device=projected.device,
                )
            )
        elif source_slot < RATIO:
            kv_rows.append(projected[pos, :head_dim])
            score_rows.append(projected[pos, 2 * head_dim : 3 * head_dim])
        else:
            kv_rows.append(projected[pos, head_dim : 2 * head_dim])
            score_rows.append(projected[pos, 3 * head_dim :])
    kv = torch.stack(kv_rows)
    score = torch.stack(score_rows)
    ape_rows = torch.cat([ape[:, :head_dim], ape[:, head_dim:]], dim=0)
    return (kv * (score + ape_rows).softmax(dim=0)).sum(dim=0).to(torch.bfloat16)


def _equal_with_nan(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))


def _partitions(total: int, pattern: tuple[int, ...]) -> list[int]:
    result: list[int] = []
    remaining = total
    index = 0
    while remaining:
        size = min(pattern[index % len(pattern)], remaining)
        result.append(size)
        remaining -= size
        index += 1
    return result


@dataclass
class ProductionFixture:
    head_dim: int
    poison_family: str
    device: torch.device

    def __post_init__(self) -> None:
        self.sequence_state = _poison(
            (SEQUENCE_SLOTS * RING_SIZE, 4 * self.head_dim),
            self.poison_family,
            device=self.device,
        )
        self.checkpoint = _poison(
            (PHYSICAL_PAGES * RATIO, 2 * self.head_dim),
            self.poison_family,
            device=self.device,
        )
        # Test-only raw locations make the tombstoned precondition explicit.
        # They are intentionally absent from the production producer ABI.
        self.raw_page_table = torch.full(
            (SEQUENCE_SLOTS, MAX_LOGICAL_TOKENS),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.raw_page_table[0].copy_(
            torch.arange(
                MAX_LOGICAL_TOKENS,
                dtype=torch.int64,
                device=self.device,
            )
        )
        self.component_page_table = torch.full(
            (SEQUENCE_SLOTS, LOGICAL_PAGES),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.component_page_table[0].copy_(
            torch.arange(
                LOGICAL_PAGES,
                dtype=torch.int64,
                device=self.device,
            )
        )
        self.component_page_table[1].copy_(
            torch.arange(
                LOGICAL_PAGES,
                dtype=torch.int64,
                device=self.device,
            )
        )

    def configure_fork(self, fork: int, *, tombstone_raw: bool) -> None:
        shared_pages = fork // PAGE_SIZE
        self.component_page_table[1].fill_(-1)
        self.component_page_table[1, :shared_pages].copy_(
            self.component_page_table[0, :shared_pages]
        )
        self.component_page_table[1, shared_pages] = PHYSICAL_PAGES - 1
        self.raw_page_table[1].copy_(self.raw_page_table[0])
        if tombstone_raw:
            self.raw_page_table[1, :fork].fill_(-1)
        if not bool(torch.all(self.component_page_table[1, :shared_pages] >= 0)):
            raise AssertionError("fork lost a retained C4 component page")

    def retained_component_pages(self, fork: int) -> list[int]:
        return self.component_page_table[1, : fork // PAGE_SIZE].cpu().tolist()

    def raw_prefix_is_tombstoned(self, fork: int) -> bool:
        return bool(torch.all(self.raw_page_table[1, :fork] == -1))

    def poison_sequence_slot(self, slot: int) -> None:
        start = slot * RING_SIZE
        self.sequence_state[start : start + RING_SIZE].copy_(
            _poison(
                (RING_SIZE, 4 * self.head_dim),
                self.poison_family,
                device=self.device,
            )
        )

    def run(
        self,
        projected: torch.Tensor,
        ape: torch.Tensor,
        positions: torch.Tensor,
        *,
        table_idx: int,
    ) -> torch.Tensor:
        positions = positions.to(device=self.device, dtype=torch.int64)
        rows = projected.index_select(0, positions)
        table_indices = torch.full_like(positions, table_idx)
        component_page_table = self.component_page_table.index_select(
            0,
            table_indices.to(torch.long),
        ).contiguous()
        return c4_online_pool_and_update(
            rows.contiguous(),
            self.sequence_state,
            self.checkpoint,
            ape,
            positions.contiguous(),
            table_indices,
            component_page_table,
            page_size=PAGE_SIZE,
        )


def _check_outputs(
    projected: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    *,
    label: str,
) -> list[int]:
    checked: list[int] = []
    for row, pos in enumerate(positions.tolist()):
        if (pos + 1) % RATIO:
            continue
        expected = _official_output(projected, ape, pos)
        if not torch.equal(output[row], expected):
            difference = (output[row].float() - expected.float()).abs().max().item()
            raise AssertionError(
                f"{label}: output {pos} differs from official FP32 reference "
                f"after BF16 publication (max_abs={difference})"
            )
        checked.append(pos)
    return checked


def _run_partitioned(
    fixture: ProductionFixture,
    projected: torch.Tensor,
    ape: torch.Tensor,
    start: int,
    stop: int,
    pattern: tuple[int, ...],
    *,
    table_idx: int = 0,
    label: str,
) -> list[int]:
    checked: list[int] = []
    cursor = start
    for size in _partitions(stop - start, pattern):
        positions = torch.arange(
            cursor,
            cursor + size,
            dtype=torch.int64,
            device=fixture.device,
        )
        output = fixture.run(
            projected,
            ape,
            positions,
            table_idx=table_idx,
        )
        checked.extend(
            _check_outputs(
                projected,
                ape,
                positions,
                output,
                label=label,
            )
        )
        cursor += size
    return checked


def _run_component(
    *,
    component: str,
    head_dim: int,
    poison_family: str,
    device: torch.device,
) -> dict[str, object]:
    checked: list[int] = []

    # Clean start: neither sequence poison nor checkpoint poison may leak into
    # the first three groups.
    clean, ape = _fixture(12, head_dim, 100 + head_dim, device=device)
    fixture = ProductionFixture(head_dim, poison_family, device)
    checked.extend(
        _run_partitioned(
            fixture,
            clean,
            ape,
            0,
            12,
            (1, 2, 1, 3),
            label=f"{component}/{poison_family}/clean",
        )
    )

    # Partial abort/reuse: a reused slot contains arbitrary rows from the old
    # generation and is not device-cleared.
    for abort_after in (1, 2, 3, 4):
        stale, _ = _fixture(12, head_dim, 200 + abort_after, device=device)
        reused, reused_ape = _fixture(
            12,
            head_dim,
            300 + abort_after,
            device=device,
        )
        fixture = ProductionFixture(head_dim, poison_family, device)
        _run_partitioned(
            fixture,
            stale,
            reused_ape,
            0,
            abort_after,
            (1,),
            label=f"{component}/{poison_family}/aborted",
        )
        checked.extend(
            _run_partitioned(
                fixture,
                reused,
                reused_ape,
                0,
                12,
                (1,),
                label=f"{component}/{poison_family}/reuse-{abort_after}",
            )
        )

    # Partial-prefix branches retain component pages while historical raw
    # locations are tombstoned. Prefix publication and recovery both use the
    # production fused kernels.
    contract_cases: list[dict[str, object]] = []
    for cached_len, fork, outputs in (
        (512, 256, [259, 263]),
        (768, 512, [515, 519]),
    ):
        projected, prefix_ape = _fixture(
            cached_len + 8,
            head_dim,
            400 + cached_len + head_dim,
            device=device,
        )
        branch_outputs: dict[str, torch.Tensor] = {}
        checkpoint_before: torch.Tensor | None = None
        retained_pages: list[int] = []
        for raw_mode in ("live", "tombstoned"):
            fixture = ProductionFixture(head_dim, poison_family, device)
            publish_positions = torch.arange(
                cached_len,
                dtype=torch.int64,
                device=device,
            )
            fixture.run(
                projected,
                prefix_ape,
                publish_positions,
                table_idx=0,
            )
            fixture.configure_fork(
                fork,
                tombstone_raw=raw_mode == "tombstoned",
            )
            checkpoint_page = int(fixture.component_page_table[1, fork // PAGE_SIZE - 1])
            checkpoint_slice = slice(
                checkpoint_page * RATIO,
                (checkpoint_page + 1) * RATIO,
            )
            published = fixture.checkpoint[checkpoint_slice].clone()
            expected_checkpoint = torch.cat(
                (
                    projected[fork - RATIO : fork, :head_dim],
                    projected[fork - RATIO : fork, 2 * head_dim : 3 * head_dim],
                ),
                dim=1,
            )
            if not torch.equal(published, expected_checkpoint):
                raise AssertionError(
                    f"{component}/{poison_family}: cached={cached_len}, fork={fork} "
                    "did not publish the exact component-indexed checkpoint"
                )
            fixture.poison_sequence_slot(1)
            positions = torch.arange(
                fork,
                fork + 8,
                dtype=torch.int64,
                device=device,
            )
            output = fixture.run(
                projected,
                prefix_ape,
                positions,
                table_idx=1,
            )
            prefix_checked = _check_outputs(
                projected,
                prefix_ape,
                positions,
                output,
                label=(
                    f"{component}/{poison_family}/cached-{cached_len}/"
                    f"fork-{fork}/{raw_mode}"
                ),
            )
            if not torch.equal(fixture.checkpoint[checkpoint_slice], published):
                raise AssertionError(
                    f"{component}/{poison_family}: immutable prefix checkpoint changed"
                )
            if raw_mode == "tombstoned" and not fixture.raw_prefix_is_tombstoned(fork):
                raise AssertionError("historical raw prefix was not tombstoned")
            branch_outputs[raw_mode] = output
            checkpoint_before = published
            retained_pages = fixture.retained_component_pages(fork)
        if prefix_checked != outputs:
            raise AssertionError(
                f"unexpected checked outputs for cached={cached_len}, fork={fork}: "
                f"{prefix_checked}"
            )
        if not torch.equal(branch_outputs["live"], branch_outputs["tombstoned"]):
            raise AssertionError(
                f"{component}/{poison_family}: live and tombstoned raw pages diverged"
            )
        assert checkpoint_before is not None
        checked.extend(prefix_checked)
        contract_cases.append(
            {
                "cached_prefix": cached_len,
                "fork": fork,
                "checked_outputs": prefix_checked,
                "publisher_component_pages": list(range(cached_len // PAGE_SIZE)),
                "retained_component_pages": retained_pages,
                "branch_component_pages": (
                    retained_pages + [PHYSICAL_PAGES - 1]
                ),
                "checkpoint_component_page": retained_pages[-1],
                "raw_prefix_tombstoned": True,
                "live_tombstoned_bitwise_identical": True,
                "checkpoint_immutable": True,
            }
        )

    # Two live slots share one immutable prefix checkpoint while their suffix
    # sequence rings and physical output pages diverge.
    common, concurrent_ape = _fixture(
        PAGE_SIZE + 8,
        head_dim,
        900 + head_dim,
        device=device,
    )
    branch_b, _ = _fixture(
        PAGE_SIZE + 8,
        head_dim,
        1000 + head_dim,
        device=device,
    )
    branch_b[:PAGE_SIZE].copy_(common[:PAGE_SIZE])
    fixture = ProductionFixture(head_dim, poison_family, device)
    _run_partitioned(
        fixture,
        common,
        concurrent_ape,
        0,
        PAGE_SIZE,
        (PAGE_SIZE,),
        label=f"{component}/{poison_family}/shared-publish",
    )
    fixture.poison_sequence_slot(0)
    fixture.poison_sequence_slot(1)
    for pos in range(PAGE_SIZE, PAGE_SIZE + 8):
        position = torch.tensor([pos], dtype=torch.int64, device=device)
        output_a = fixture.run(
            common,
            concurrent_ape,
            position,
            table_idx=0,
        )
        checked.extend(
            _check_outputs(
                common,
                concurrent_ape,
                position,
                output_a,
                label=f"{component}/{poison_family}/concurrent-a",
            )
        )
        output_b = fixture.run(
            branch_b,
            concurrent_ape,
            position,
            table_idx=1,
        )
        checked.extend(
            _check_outputs(
                branch_b,
                concurrent_ape,
                position,
                output_b,
                label=f"{component}/{poison_family}/concurrent-b",
            )
        )

    # The graph dummy owns the final stable sequence slot and cannot mutate
    # either live slot or any compact checkpoint.
    live_before = fixture.sequence_state[: LIVE_SLOTS * RING_SIZE].clone()
    checkpoint_before = fixture.checkpoint.clone()
    dummy, dummy_ape = _fixture(4, head_dim, 1200 + head_dim, device=device)
    dummy_positions = torch.arange(4, dtype=torch.int64, device=device)
    fixture.run(
        dummy,
        dummy_ape,
        dummy_positions,
        table_idx=LIVE_SLOTS,
    )
    if not _equal_with_nan(
        fixture.sequence_state[: LIVE_SLOTS * RING_SIZE],
        live_before,
    ):
        raise AssertionError(f"{component}/{poison_family}: dummy changed live state")
    if not _equal_with_nan(fixture.checkpoint, checkpoint_before):
        raise AssertionError(f"{component}/{poison_family}: dummy changed checkpoint")

    return {
        "component": component,
        "head_dim": head_dim,
        "poison": poison_family,
        "checked_boundary_outputs": len(checked),
        "required_prefix_outputs": [pos for pos in checked if pos in (259, 263, 515, 519)],
        "component_page_contract_cases": contract_cases,
        "status": "pass",
    }


def run_production_poison() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"status": "skip", "reason": "CUDA unavailable"}
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 8:
        return {"status": "skip", "reason": "SM80 or newer required"}

    cases = []
    for component, head_dim in (("attention", 16), ("indexer", 8)):
        for poison_family in ("finite_extreme", "nan_inf"):
            cases.append(
                _run_component(
                    component=component,
                    head_dim=head_dim,
                    poison_family=poison_family,
                    device=device,
                )
            )
    torch.cuda.synchronize(device)
    return {
        "status": "pass",
        "device": torch.cuda.get_device_name(device),
        "page_size": PAGE_SIZE,
        "sequence_slots": {
            "live": LIVE_SLOTS,
            "graph_dummy": 1,
        },
        "checkpoint_layout": "[C4 component page, four tail rows, kv_left + score_left]",
        "checkpoint_addressing": "component_page_table[row, logical_full_page]",
        "raw_page_inputs_to_producer": 0,
        "phase_metadata_required": False,
        "candidate": "A",
        "producer_kernel_launches_per_call": 2,
        "separate_restore_or_materialization_launches": 0,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_production_poison()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    return 0 if result["status"] in {"pass", "skip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

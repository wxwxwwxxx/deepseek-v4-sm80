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
MAX_LOGICAL_TOKENS = 528


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
        self.ctx_page_table = torch.full(
            (SEQUENCE_SLOTS, MAX_LOGICAL_TOKENS),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        positions = torch.arange(
            MAX_LOGICAL_TOKENS,
            dtype=torch.int64,
            device=self.device,
        )
        self.ctx_page_table[0].copy_(positions)
        self.ctx_page_table[1, :PAGE_SIZE].copy_(positions[:PAGE_SIZE])
        self.ctx_page_table[1, PAGE_SIZE:].copy_(2 * PAGE_SIZE + positions[PAGE_SIZE:] - PAGE_SIZE)
        self.checkpoint_page_mapping = torch.arange(
            PHYSICAL_PAGES,
            dtype=torch.int64,
            device=self.device,
        )

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
        raw_out_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = positions.to(device=self.device, dtype=torch.int64)
        rows = projected.index_select(0, positions)
        table_indices = torch.full_like(positions, table_idx)
        if raw_out_loc is None:
            raw_out_loc = positions
        return c4_online_pool_and_update(
            rows.contiguous(),
            self.sequence_state,
            self.checkpoint,
            ape,
            positions.contiguous(),
            table_indices,
            raw_out_loc.to(device=self.device, dtype=torch.int64).contiguous(),
            self.ctx_page_table,
            self.checkpoint_page_mapping,
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
    physical_page: int | None = None,
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
        raw_out_loc = None
        if physical_page is not None:
            raw_out_loc = physical_page * PAGE_SIZE + positions - start
        output = fixture.run(
            projected,
            ape,
            positions,
            table_idx=table_idx,
            raw_out_loc=raw_out_loc,
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

    # Page-aligned prefix hits. Prefix publication uses the production fused
    # store path; the new sequence generation starts from poison.
    for prefix_len, outputs in ((256, [259, 263]), (512, [515, 519])):
        projected, prefix_ape = _fixture(
            prefix_len + 8,
            head_dim,
            400 + prefix_len + head_dim,
            device=device,
        )
        fixture = ProductionFixture(head_dim, poison_family, device)
        _run_partitioned(
            fixture,
            projected,
            prefix_ape,
            0,
            prefix_len,
            (prefix_len,),
            label=f"{component}/{poison_family}/publish-{prefix_len}",
        )
        checkpoint_page = prefix_len // PAGE_SIZE - 1
        checkpoint_tail = fixture.checkpoint[
            checkpoint_page * RATIO : (checkpoint_page + 1) * RATIO
        ]
        if not bool(torch.isfinite(checkpoint_tail).all()):
            raise AssertionError(
                f"{component}/{poison_family}: prefix {prefix_len} did not "
                "publish four finite compact rows"
            )
        fixture.poison_sequence_slot(0)
        prefix_checked = _run_partitioned(
            fixture,
            projected,
            prefix_ape,
            prefix_len,
            prefix_len + 8,
            (2, 1, 5),
            label=f"{component}/{poison_family}/hit-{prefix_len}",
        )
        if prefix_checked != outputs:
            raise AssertionError(
                f"unexpected checked outputs for prefix {prefix_len}: {prefix_checked}"
            )
        checked.extend(prefix_checked)

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
            raw_out_loc=position,
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
            raw_out_loc=2 * PAGE_SIZE + position - PAGE_SIZE,
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
        raw_out_loc=torch.full_like(dummy_positions, -1),
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
        "checkpoint_layout": "[page, four tail rows, kv_left + score_left]",
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

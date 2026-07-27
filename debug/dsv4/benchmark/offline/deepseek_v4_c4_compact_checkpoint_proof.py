from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch


PAGE_SIZE = 256
RATIO = 4
RING_SIZE = 8


def _poison(shape: tuple[int, ...], family: str) -> torch.Tensor:
    size = math.prod(shape)
    if family == "finite_extreme":
        flat = torch.full((size,), 1.0e20, dtype=torch.float32)
        flat[1::2] = -1.0e20
    elif family == "nan_inf":
        values = torch.tensor(
            [float("nan"), float("inf"), float("-inf")],
            dtype=torch.float32,
        )
        flat = values[torch.arange(size) % values.numel()]
    else:
        raise ValueError(f"unknown poison family: {family}")
    return flat.view(shape).clone()


def _fixture(length: int, head_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    # Keep the executable reference finite while making the softmax rows
    # meaningfully non-uniform.
    projected[:, 2 * head_dim :] *= 3.0
    ape *= 2.0
    return projected, ape


def _select_half(row: torch.Tensor, source_slot: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if source_slot < RATIO:
        return row[:head_dim], row[2 * head_dim : 3 * head_dim]
    return row[head_dim : 2 * head_dim], row[3 * head_dim :]


def _pool(
    rows: list[tuple[torch.Tensor, torch.Tensor]],
    ape: torch.Tensor,
) -> torch.Tensor:
    kv = torch.stack([row[0] for row in rows])
    score = torch.stack([row[1] for row in rows])
    head_dim = kv.shape[-1]
    ape_rows = torch.cat([ape[:, :head_dim], ape[:, head_dim:]], dim=0)
    return (kv * (score + ape_rows).softmax(dim=0)).sum(dim=0)


def _official_output(
    projected: torch.Tensor,
    ape: torch.Tensor,
    end_pos: int,
) -> torch.Tensor:
    head_dim = projected.shape[-1] // 4
    rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    for source_slot in range(RING_SIZE):
        pos = end_pos - (RING_SIZE - 1 - source_slot)
        if pos < 0:
            rows.append(
                (
                    torch.zeros(head_dim, dtype=torch.float32),
                    torch.full((head_dim,), float("-inf"), dtype=torch.float32),
                )
            )
        else:
            rows.append(_select_half(projected[pos], source_slot, head_dim))
    return _pool(rows, ape)


def _partition(total: int, pattern: list[int]) -> list[int]:
    out: list[int] = []
    remaining = total
    index = 0
    while remaining:
        size = min(pattern[index % len(pattern)], remaining)
        out.append(size)
        remaining -= size
        index += 1
    return out


def _tensor_equal_with_nan(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))


@dataclass
class ReplayResult:
    outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    source_counts: Counter[str] = field(default_factory=Counter)
    read_before_write: list[str] = field(default_factory=list)


class FullPageOracle:
    """Bounded Layer-A oracle matching Mini's legacy 8-row page state."""

    def __init__(self, *, pages: int, head_dim: int, poison_family: str) -> None:
        self.head_dim = head_dim
        self.state = _poison((pages, RING_SIZE, 4 * head_dim), poison_family)
        self.tags = torch.full((pages, RING_SIZE), -1, dtype=torch.int64)
        self.poison_family = poison_family

    def publish_prefix(self, projected: torch.Tensor, prefix_len: int) -> None:
        for pos in range(prefix_len):
            page = pos // PAGE_SIZE
            slot = pos % RING_SIZE
            self.state[page, slot].copy_(projected[pos])
            self.tags[page, slot] = pos

    def retain_only_minimal_tail(self, prefix_len: int) -> None:
        if prefix_len == 0:
            return
        if prefix_len % PAGE_SIZE:
            raise ValueError("prefix must be page aligned")
        page = prefix_len // PAGE_SIZE - 1
        saved = self.state[page].clone()
        saved_tags = self.tags[page].clone()
        self.state[page].copy_(
            _poison(tuple(self.state[page].shape), self.poison_family)
        )
        self.tags[page].fill_(-1)
        for pos in range(prefix_len - RATIO, prefix_len):
            slot = pos % RING_SIZE
            self.state[page, slot, : self.head_dim].copy_(
                saved[slot, : self.head_dim]
            )
            self.state[
                page,
                slot,
                2 * self.head_dim : 3 * self.head_dim,
            ].copy_(saved[slot, 2 * self.head_dim : 3 * self.head_dim])
            self.tags[page, slot] = saved_tags[slot]

    def process(
        self,
        positions: torch.Tensor,
        projected_rows: torch.Tensor,
        ape: torch.Tensor,
    ) -> ReplayResult:
        result = ReplayResult()
        current = {int(pos): row for row, pos in enumerate(positions.tolist())}
        for pos in positions.tolist():
            if (pos + 1) % RATIO:
                continue
            rows: list[tuple[torch.Tensor, torch.Tensor]] = []
            for source_slot in range(RING_SIZE):
                logical_pos = pos - (RING_SIZE - 1 - source_slot)
                if logical_pos < 0:
                    rows.append(
                        (
                            torch.zeros(self.head_dim),
                            torch.full((self.head_dim,), float("-inf")),
                        )
                    )
                    result.source_counts["masked"] += 1
                    continue
                if logical_pos in current:
                    row = projected_rows[current[logical_pos]]
                    result.source_counts["current"] += 1
                else:
                    page = logical_pos // PAGE_SIZE
                    slot = logical_pos % RING_SIZE
                    row = self.state[page, slot]
                    if int(self.tags[page, slot]) != logical_pos:
                        result.read_before_write.append(
                            f"full page={page} slot={slot} expected={logical_pos} "
                            f"tag={int(self.tags[page, slot])}"
                        )
                    result.source_counts["full_checkpoint"] += 1
                rows.append(_select_half(row, source_slot, self.head_dim))
            result.outputs[pos] = _pool(rows, ape)

        # Match the production kernel's last-writer-wins ring update.
        for row, pos in enumerate(positions.tolist()):
            page = pos // PAGE_SIZE
            slot = pos % RING_SIZE
            self.state[page, slot].copy_(projected_rows[row])
            self.tags[page, slot] = pos
        return result


class CompactReplay:
    """Candidate-A replay: compact cross-page tail plus sequence working ring."""

    def __init__(
        self,
        *,
        pages: int,
        sequence_slots: int,
        live_slots: int,
        head_dim: int,
        poison_family: str,
    ) -> None:
        self.head_dim = head_dim
        self.live_slots = live_slots
        self.sequence = _poison(
            (sequence_slots, RING_SIZE, 4 * head_dim),
            poison_family,
        )
        self.sequence_tags = torch.full(
            (sequence_slots, RING_SIZE),
            -1,
            dtype=torch.int64,
        )
        self.sequence_tag_owners = torch.full_like(self.sequence_tags, -1)
        self.checkpoint = _poison(
            (pages, RATIO, 2 * head_dim),
            poison_family,
        )
        self.checkpoint_tags = torch.full(
            (pages, RATIO),
            -1,
            dtype=torch.int64,
        )
        self.owners: list[int | None] = [None] * live_slots

    def acquire(self, slot: int, owner: int) -> None:
        if slot < 0 or slot >= self.live_slots:
            raise ValueError("graph dummy is not a live request slot")
        if self.owners[slot] is not None:
            raise RuntimeError("slot already owned")
        self.owners[slot] = owner
        # Logical invalidation only. Device bytes intentionally remain poison.
        self.sequence_tags[slot].fill_(-1)
        self.sequence_tag_owners[slot].fill_(-1)

    def release(self, slot: int, owner: int) -> None:
        if self.owners[slot] != owner:
            raise RuntimeError("owner mismatch")
        self.owners[slot] = None

    def publish_prefix(self, projected: torch.Tensor, prefix_len: int) -> None:
        if prefix_len % PAGE_SIZE:
            raise ValueError("prefix must be page aligned")
        for pos in range(prefix_len):
            if pos % PAGE_SIZE < PAGE_SIZE - RATIO:
                continue
            self._publish_checkpoint_row(pos, projected[pos])

    def _publish_checkpoint_row(self, pos: int, projected_row: torch.Tensor) -> None:
        page = pos // PAGE_SIZE
        slot = pos % RATIO
        compact = torch.cat(
            [
                projected_row[: self.head_dim],
                projected_row[2 * self.head_dim : 3 * self.head_dim],
            ]
        )
        self.checkpoint[page, slot].copy_(compact)
        self.checkpoint_tags[page, slot] = pos

    def process(
        self,
        *,
        slot: int,
        owner: int,
        positions: torch.Tensor,
        projected_rows: torch.Tensor,
        ape: torch.Tensor,
        graph_dummy: bool = False,
    ) -> ReplayResult:
        if graph_dummy:
            if slot < self.live_slots:
                raise ValueError("dummy must use the reserved physical slot")
        elif self.owners[slot] != owner:
            raise RuntimeError("slot owner mismatch")

        result = ReplayResult()
        current = {int(pos): row for row, pos in enumerate(positions.tolist())}
        for pos in positions.tolist():
            if (pos + 1) % RATIO:
                continue
            rows: list[tuple[torch.Tensor, torch.Tensor]] = []
            for source_slot in range(RING_SIZE):
                logical_pos = pos - (RING_SIZE - 1 - source_slot)
                if logical_pos < 0:
                    rows.append(
                        (
                            torch.zeros(self.head_dim),
                            torch.full((self.head_dim,), float("-inf")),
                        )
                    )
                    result.source_counts["masked"] += 1
                    continue
                if logical_pos in current:
                    row = projected_rows[current[logical_pos]]
                    rows.append(_select_half(row, source_slot, self.head_dim))
                    result.source_counts["current"] += 1
                    continue

                cross_page_left = (
                    source_slot < RATIO
                    and logical_pos // PAGE_SIZE != pos // PAGE_SIZE
                )
                if cross_page_left:
                    page = logical_pos // PAGE_SIZE
                    checkpoint_slot = logical_pos % RATIO
                    compact = self.checkpoint[page, checkpoint_slot]
                    if int(self.checkpoint_tags[page, checkpoint_slot]) != logical_pos:
                        result.read_before_write.append(
                            f"checkpoint page={page} slot={checkpoint_slot} "
                            f"expected={logical_pos} "
                            f"tag={int(self.checkpoint_tags[page, checkpoint_slot])}"
                        )
                    rows.append(
                        (
                            compact[: self.head_dim],
                            compact[self.head_dim :],
                        )
                    )
                    result.source_counts["compact_checkpoint"] += 1
                    continue

                sequence_slot = logical_pos % RING_SIZE
                row = self.sequence[slot, sequence_slot]
                tag = int(self.sequence_tags[slot, sequence_slot])
                tag_owner = int(self.sequence_tag_owners[slot, sequence_slot])
                if tag != logical_pos or tag_owner != owner:
                    result.read_before_write.append(
                        f"sequence slot={slot}:{sequence_slot} expected={logical_pos}/"
                        f"{owner} tag={tag}/{tag_owner}"
                    )
                rows.append(_select_half(row, source_slot, self.head_dim))
                result.source_counts["sequence"] += 1
            result.outputs[pos] = _pool(rows, ape)

        # Persist the full official row in the stable sequence ring. Aliases
        # intentionally resolve to the final row in this call.
        for row, pos in enumerate(positions.tolist()):
            sequence_slot = pos % RING_SIZE
            self.sequence[slot, sequence_slot].copy_(projected_rows[row])
            self.sequence_tags[slot, sequence_slot] = pos
            self.sequence_tag_owners[slot, sequence_slot] = owner
            if pos % PAGE_SIZE >= PAGE_SIZE - RATIO:
                self._publish_checkpoint_row(pos, projected_rows[row])
        return result


def _max_errors(
    expected: dict[int, torch.Tensor],
    actual: dict[int, torch.Tensor],
) -> tuple[float, float, bool]:
    if expected.keys() != actual.keys():
        return float("inf"), float("inf"), False
    errors = [
        (expected[pos] - actual[pos]).abs()
        for pos in expected
    ]
    if not errors:
        return 0.0, 0.0, True
    flat = torch.cat([error.flatten() for error in errors])
    return float(flat.max()), float(flat.mean()), bool(torch.equal(
        torch.stack([expected[pos] for pos in expected]),
        torch.stack([actual[pos] for pos in actual]),
    ))


def _run_path(
    *,
    component: str,
    head_dim: int,
    poison_family: str,
    prefix_len: int,
    end_pos: int,
    partitions: list[int],
    seed: int,
) -> dict[str, object]:
    projected, ape = _fixture(end_pos + 1, head_dim, seed)
    pages = math.ceil((end_pos + 1) / PAGE_SIZE) + 1
    full = FullPageOracle(
        pages=pages,
        head_dim=head_dim,
        poison_family=poison_family,
    )
    compact = CompactReplay(
        pages=pages,
        sequence_slots=2,
        live_slots=1,
        head_dim=head_dim,
        poison_family=poison_family,
    )
    full.publish_prefix(projected, prefix_len)
    compact.publish_prefix(projected, prefix_len)
    if prefix_len:
        # Poison every field except the hypothesized four left-half KV/score
        # rows in the full legacy checkpoint.
        full.retain_only_minimal_tail(prefix_len)
    compact.acquire(0, 17)

    full_outputs: dict[int, torch.Tensor] = {}
    compact_outputs: dict[int, torch.Tensor] = {}
    source_counts: Counter[str] = Counter()
    read_before_write: list[str] = []
    cursor = prefix_len
    for size in partitions:
        positions = torch.arange(cursor, cursor + size, dtype=torch.int64)
        rows = projected[cursor : cursor + size]
        full_result = full.process(positions, rows, ape)
        compact_result = compact.process(
            slot=0,
            owner=17,
            positions=positions,
            projected_rows=rows,
            ape=ape,
        )
        full_outputs.update(full_result.outputs)
        compact_outputs.update(compact_result.outputs)
        source_counts.update(full_result.source_counts)
        source_counts.update(compact_result.source_counts)
        read_before_write.extend(full_result.read_before_write)
        read_before_write.extend(compact_result.read_before_write)
        cursor += size
    compact.release(0, 17)
    if cursor != end_pos + 1:
        raise AssertionError((cursor, end_pos))

    expected = {
        pos: _official_output(projected, ape, pos)
        for pos in range(prefix_len, end_pos + 1)
        if (pos + 1) % RATIO == 0
    }
    full_max, full_mean, full_exact = _max_errors(expected, full_outputs)
    compact_max, compact_mean, compact_exact = _max_errors(expected, compact_outputs)
    downstream_exact = bool(
        torch.equal(
            torch.stack([expected[pos].to(torch.bfloat16) for pos in expected]),
            torch.stack([compact_outputs[pos].to(torch.bfloat16) for pos in expected]),
        )
    )
    return {
        "component": component,
        "poison": poison_family,
        "prefix_len": prefix_len,
        "end_pos": end_pos,
        "partitions": partitions,
        "outputs": sorted(expected),
        "full_checkpoint_max_abs_error": full_max,
        "full_checkpoint_mean_abs_error": full_mean,
        "full_checkpoint_exact": full_exact,
        "compact_checkpoint_max_abs_error": compact_max,
        "compact_checkpoint_mean_abs_error": compact_mean,
        "compact_checkpoint_exact": compact_exact,
        "downstream_bf16_exact": downstream_exact,
        "read_before_write": read_before_write,
        "source_counts": dict(sorted(source_counts.items())),
        "passed": (
            full_exact
            and compact_exact
            and downstream_exact
            and not read_before_write
        ),
    }


def _run_abort_reuse(
    *,
    component: str,
    head_dim: int,
    poison_family: str,
    abort_len: int,
    prefix_len: int,
    seed: int,
) -> dict[str, object]:
    end_pos = prefix_len + 7 if prefix_len else 11
    projected_old, ape = _fixture(max(end_pos + 1, 16), head_dim, seed)
    projected_new, _ = _fixture(end_pos + 1, head_dim, seed + 1000)
    replay = CompactReplay(
        pages=math.ceil((end_pos + 1) / PAGE_SIZE) + 1,
        sequence_slots=2,
        live_slots=1,
        head_dim=head_dim,
        poison_family=poison_family,
    )
    replay.acquire(0, 1)
    old_positions = torch.arange(abort_len, dtype=torch.int64)
    replay.process(
        slot=0,
        owner=1,
        positions=old_positions,
        projected_rows=projected_old[:abort_len],
        ape=ape,
    )
    stale_bytes = replay.sequence[0].clone()
    replay.release(0, 1)
    replay.publish_prefix(projected_new, prefix_len)
    replay.acquire(0, 2)
    if not _tensor_equal_with_nan(replay.sequence[0], stale_bytes):
        raise AssertionError("acquire performed a physical clear")

    outputs: dict[int, torch.Tensor] = {}
    read_before_write: list[str] = []
    cursor = prefix_len
    for size in _partition(end_pos + 1 - prefix_len, [1, 2, 1]):
        positions = torch.arange(cursor, cursor + size, dtype=torch.int64)
        result = replay.process(
            slot=0,
            owner=2,
            positions=positions,
            projected_rows=projected_new[cursor : cursor + size],
            ape=ape,
        )
        outputs.update(result.outputs)
        read_before_write.extend(result.read_before_write)
        cursor += size
    expected = {
        pos: _official_output(projected_new, ape, pos)
        for pos in range(prefix_len, end_pos + 1)
        if (pos + 1) % RATIO == 0
    }
    max_error, mean_error, exact = _max_errors(expected, outputs)
    replay.release(0, 2)
    return {
        "component": component,
        "poison": poison_family,
        "abort_len": abort_len,
        "prefix_len": prefix_len,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "exact": exact,
        "physical_clear_elided": True,
        "read_before_write": read_before_write,
        "passed": exact and not read_before_write,
    }


def _run_concurrent_and_dummy(
    *,
    component: str,
    head_dim: int,
    poison_family: str,
    seed: int,
) -> dict[str, object]:
    prefix_len = PAGE_SIZE
    end_pos = prefix_len + 7
    shared, ape = _fixture(end_pos + 1, head_dim, seed)
    left = shared.clone()
    right = shared.clone()
    suffix_right, _ = _fixture(8, head_dim, seed + 2000)
    right[prefix_len:] = suffix_right
    replay = CompactReplay(
        pages=3,
        sequence_slots=3,
        live_slots=2,
        head_dim=head_dim,
        poison_family=poison_family,
    )
    replay.publish_prefix(shared, prefix_len)
    checkpoint_before = replay.checkpoint.clone()
    replay.acquire(0, 10)
    replay.acquire(1, 11)
    outputs = [{}, {}]
    errors: list[str] = []
    cursor = [prefix_len, prefix_len]
    for size in [2, 2, 4]:
        for slot, owner, projected in (
            (0, 10, left),
            (1, 11, right),
        ):
            positions = torch.arange(cursor[slot], cursor[slot] + size)
            result = replay.process(
                slot=slot,
                owner=owner,
                positions=positions,
                projected_rows=projected[cursor[slot] : cursor[slot] + size],
                ape=ape,
            )
            outputs[slot].update(result.outputs)
            errors.extend(result.read_before_write)
            cursor[slot] += size
    exact = []
    for projected, actual in ((left, outputs[0]), (right, outputs[1])):
        expected = {
            pos: _official_output(projected, ape, pos)
            for pos in (prefix_len + 3, prefix_len + 7)
        }
        exact.append(_max_errors(expected, actual)[2])

    # The divergent suffix does not mutate the shared prior-page checkpoint.
    checkpoint_immutable = _tensor_equal_with_nan(
        checkpoint_before[0],
        replay.checkpoint[0],
    )
    live_before = replay.sequence[:2].clone()
    try:
        replay.acquire(2, 99)
        dummy_rejected_as_live = False
    except ValueError:
        dummy_rejected_as_live = True
    dummy_projected, _ = _fixture(4, head_dim, seed + 3000)
    replay.process(
        slot=2,
        owner=-999,
        positions=torch.arange(4),
        projected_rows=dummy_projected,
        ape=ape,
        graph_dummy=True,
    )
    live_unchanged_by_dummy = _tensor_equal_with_nan(
        live_before,
        replay.sequence[:2],
    )
    replay.release(0, 10)
    replay.release(1, 11)
    return {
        "component": component,
        "poison": poison_family,
        "concurrent_exact": all(exact),
        "shared_checkpoint_immutable": checkpoint_immutable,
        "dummy_rejected_as_live": dummy_rejected_as_live,
        "live_slots_unchanged_by_dummy": live_unchanged_by_dummy,
        "read_before_write": errors,
        "passed": (
            all(exact)
            and checkpoint_immutable
            and dummy_rejected_as_live
            and live_unchanged_by_dummy
            and not errors
        ),
    }


def run_proof() -> dict[str, object]:
    components = {"attention": 7, "indexer": 5}
    poison_families = ("finite_extreme", "nan_inf")
    cases: list[dict[str, object]] = []
    seed = 160200
    for component, head_dim in components.items():
        for poison_family in poison_families:
            path_specs = [
                (0, 11, [12]),
                (0, 11, [1] * 12),
                (0, 11, _partition(12, [2, 1, 5])),
                (0, 135, [136]),
                (0, 135, _partition(136, [17, 31, 7, 19])),
                (256, 263, [8]),
                (256, 263, [1] * 8),
                (256, 263, [3, 1, 2, 2]),
                (512, 519, [8]),
                (512, 519, [1] * 8),
                (512, 519, [2, 3, 1, 2]),
            ]
            for prefix_len, end_pos, partitions in path_specs:
                cases.append(
                    _run_path(
                        component=component,
                        head_dim=head_dim,
                        poison_family=poison_family,
                        prefix_len=prefix_len,
                        end_pos=end_pos,
                        partitions=partitions,
                        seed=seed,
                    )
                )
                seed += 1
            for abort_len in (1, 2, 3, 7):
                for prefix_len in (0, 256, 512):
                    cases.append(
                        _run_abort_reuse(
                            component=component,
                            head_dim=head_dim,
                            poison_family=poison_family,
                            abort_len=abort_len,
                            prefix_len=prefix_len,
                            seed=seed,
                        )
                    )
                    seed += 1
            cases.append(
                _run_concurrent_and_dummy(
                    component=component,
                    head_dim=head_dim,
                    poison_family=poison_family,
                    seed=seed,
                )
            )
            seed += 1

    old_floats = RING_SIZE * 4
    compact_floats = RATIO * 2
    failures = [
        index
        for index, case in enumerate(cases)
        if not bool(case["passed"])
    ]
    return {
        "status": "pass" if not failures else "fail",
        "page_size": PAGE_SIZE,
        "page_size_mod_ratio": PAGE_SIZE % RATIO,
        "phase_metadata_required": False,
        "old_checkpoint_floats_per_head_dim": old_floats,
        "compact_checkpoint_floats_per_head_dim": compact_floats,
        "compact_to_old_ratio": compact_floats / old_floats,
        "required_checkpoint": {
            "rows": 4,
            "kv_half": "left",
            "score_half": "left",
            "dtype": "torch.float32",
            "ordering": "absolute positions page_end-4..page_end-1",
        },
        "cases_total": len(cases),
        "cases_passed": len(cases) - len(failures),
        "failure_indices": failures,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_proof()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

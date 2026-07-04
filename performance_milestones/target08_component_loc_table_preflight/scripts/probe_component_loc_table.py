from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
SUMMARIES = ROOT / "summaries"

PAGE_SIZE = 256
WINDOW_SIZE = 128
INDEX_TOPK = 512
ALIGNMENT = 64


@dataclass(frozen=True)
class ReqLayout:
    table_idx: int
    device_len: int
    cached_len: int

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


@dataclass(frozen=True)
class Scenario:
    name: str
    phase: Literal["prefill", "decode"]
    hit_style: str
    reqs: tuple[ReqLayout, ...]
    note: str


@dataclass(frozen=True)
class Phase1Metadata:
    raw_out_loc: torch.Tensor
    positions: torch.Tensor
    table_indices: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    swa_page_indices: torch.Tensor
    c4_sparse_page_indices: torch.Tensor
    c128_page_indices: torch.Tensor
    c4_out_loc: torch.Tensor
    c128_out_loc: torch.Tensor
    c4_indexer_out_loc: torch.Tensor
    c4_topk_lengths_raw: torch.Tensor
    indexer_page_table: torch.Tensor
    indexer_loc_table_gather: torch.Tensor
    c4_state_loc_from_swa: torch.Tensor
    c128_state_loc_from_swa: torch.Tensor
    c4_indexer_state_loc_from_swa: torch.Tensor


@dataclass(frozen=True)
class DirectLocTables:
    swa_loc_table: torch.Tensor
    c4_loc_table: torch.Tensor
    c128_loc_table: torch.Tensor
    c4_indexer_loc_table: torch.Tensor
    swa_page_table: torch.Tensor
    c4_page_table: torch.Tensor
    c128_page_table: torch.Tensor
    c4_indexer_page_table: torch.Tensor
    state_loc_placeholder: dict[str, str]


def _div_ceil(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _pad_last_dim(x: torch.Tensor, *, multiple: int = ALIGNMENT, value: int = -1) -> torch.Tensor:
    size = x.shape[-1]
    target = _div_ceil(size, multiple) * multiple
    if target == size:
        return x
    out = torch.full((*x.shape[:-1], target), value, dtype=x.dtype)
    out[..., :size] = x
    return out


def _state_loc_from_swa(swa_loc: torch.Tensor, *, ring_size: int) -> torch.Tensor:
    state_loc = (swa_loc // PAGE_SIZE) * ring_size + (swa_loc % ring_size)
    return torch.where(swa_loc < 0, torch.full_like(swa_loc, -1), state_loc)


def _active_positions(scenario: Scenario) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    positions: list[torch.Tensor] = []
    table_indices: list[torch.Tensor] = []
    extend_lens: list[int] = []
    req_seq_lens: list[int] = []
    for req in scenario.reqs:
        if scenario.phase == "decode":
            pos = torch.tensor([req.device_len - 1], dtype=torch.int32)
        else:
            pos = torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
        positions.append(pos)
        table_indices.append(torch.full((pos.numel(),), req.table_idx, dtype=torch.int32))
        extend_lens.append(int(pos.numel()))
        req_seq_lens.append(req.device_len)
    return (
        torch.cat(positions) if positions else torch.empty(0, dtype=torch.int32),
        torch.cat(table_indices) if table_indices else torch.empty(0, dtype=torch.int32),
        extend_lens,
        req_seq_lens,
    )


def _make_full_loc_table(scenario: Scenario) -> torch.Tensor:
    max_table_idx = max(req.table_idx for req in scenario.reqs)
    max_len = max(req.device_len for req in scenario.reqs)
    page_count = _div_ceil(max(max_len, 1), PAGE_SIZE)
    width = page_count * PAGE_SIZE
    table = torch.full((max_table_idx + 1, width), -1, dtype=torch.int32)
    base_page = 11
    for req_id, req in enumerate(scenario.reqs):
        # Non-zero, row-distinct physical pages catch accidental logical-position
        # reuse while preserving the phase-1 page-aligned full-token layout.
        first_page = base_page + req_id * 17
        for page in range(page_count):
            page_id = first_page + page
            start = page * PAGE_SIZE
            stop = start + PAGE_SIZE
            table[req.table_idx, start:stop] = torch.arange(
                page_id * PAGE_SIZE,
                page_id * PAGE_SIZE + PAGE_SIZE,
                dtype=torch.int32,
            )
    return table


def _gather_locs(
    loc_table: torch.Tensor,
    table_indices: torch.Tensor,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    valid = logical_positions >= 0
    clamped = logical_positions.clamp_min(0).to(torch.long)
    rows = table_indices.to(torch.long)[:, None].expand_as(clamped)
    gathered = loc_table[rows, clamped].to(torch.int32)
    return torch.where(valid, gathered, torch.full_like(gathered, -1))


def _make_phase_page_table(
    full_loc_table: torch.Tensor,
    table_indices: torch.Tensor,
    max_seqlen_k: int,
) -> torch.Tensor:
    page_width = _div_ceil(max(max_seqlen_k, 1), PAGE_SIZE)
    offsets = torch.arange(page_width, dtype=torch.long) * PAGE_SIZE
    rows = table_indices.to(torch.long)
    full_locs = full_loc_table[rows[:, None], offsets[None, :]]
    page_table = torch.where(full_locs >= 0, full_locs // PAGE_SIZE, full_locs)
    return page_table.to(torch.int32)


def _make_direct_tables(phase_page_table: torch.Tensor) -> DirectLocTables:
    rows, page_width = phase_page_table.shape

    def expand_by_page(slots_per_page: int) -> torch.Tensor:
        offsets = torch.arange(slots_per_page, dtype=torch.int32)
        locs = phase_page_table[:, :, None] * slots_per_page + offsets[None, None, :]
        locs = torch.where(
            phase_page_table[:, :, None] >= 0,
            locs,
            torch.full_like(locs, -1),
        )
        return locs.reshape(rows, page_width * slots_per_page).to(torch.int32)

    swa_loc_table = expand_by_page(PAGE_SIZE)
    c4_loc_table = expand_by_page(PAGE_SIZE // 4)
    c128_loc_table = expand_by_page(PAGE_SIZE // 128)
    return DirectLocTables(
        swa_loc_table=swa_loc_table,
        c4_loc_table=c4_loc_table,
        c128_loc_table=c128_loc_table,
        c4_indexer_loc_table=c4_loc_table.clone(),
        swa_page_table=phase_page_table.clone(),
        c4_page_table=phase_page_table.clone(),
        c128_page_table=phase_page_table.clone(),
        c4_indexer_page_table=phase_page_table.clone(),
        state_loc_placeholder={
            "c4": "derived from direct SWA loc with ring_size=8; ownership deferred to B2",
            "c128": "derived from direct SWA loc with ring_size=128; ownership deferred to B2",
            "c4_indexer": "derived from direct SWA loc with ring_size=8; ownership deferred to B2",
        },
    )


def _make_sparse_raw(lengths: torch.Tensor) -> torch.Tensor:
    width = max(INDEX_TOPK, 1)
    raw = torch.full((lengths.numel(), width), -1, dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        start = max(0, int(length) - INDEX_TOPK)
        values = torch.arange(start, int(length), dtype=torch.int32)
        raw[row, : values.numel()] = values
    return _pad_last_dim(raw, value=-1)


def _make_all_raw(lengths: torch.Tensor) -> torch.Tensor:
    width = max(int(lengths.max().item()) if lengths.numel() else 0, 1)
    raw = torch.full((lengths.numel(), width), -1, dtype=torch.int32)
    for row, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        values = torch.arange(int(length), dtype=torch.int32)
        raw[row, : values.numel()] = values
    return _pad_last_dim(raw, value=-1)


def _compressed_locs_from_full(
    raw_out_loc: torch.Tensor,
    positions: torch.Tensor,
    ratio: Literal[4, 128],
) -> torch.Tensor:
    mask = (positions + 1) % ratio == 0
    values = raw_out_loc.to(torch.long)[mask]
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.int32)
    return torch.unique_consecutive(values // ratio).to(torch.int32)


def _direct_out_locs(
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    loc_table: torch.Tensor,
    ratio: Literal[4, 128],
) -> torch.Tensor:
    mask = (positions + 1) % ratio == 0
    if not bool(torch.any(mask)):
        return torch.empty(0, dtype=torch.int32)
    rows = table_indices[mask].to(torch.long)
    component_positions = (positions[mask] // ratio).to(torch.long)
    values = loc_table[rows, component_positions]
    return torch.unique_consecutive(values.to(torch.int32))


def _build_metadata_pair(scenario: Scenario) -> tuple[Phase1Metadata, Phase1Metadata, dict[str, object]]:
    full_loc_table = _make_full_loc_table(scenario)
    positions, table_indices, extend_lens, req_seq_lens = _active_positions(scenario)
    max_seqlen_k = max(req_seq_lens)
    raw_out_loc = full_loc_table[table_indices.to(torch.long), positions.to(torch.long)]
    seq_lens = positions + 1
    phase_page_table = _make_phase_page_table(full_loc_table, table_indices, max_seqlen_k)
    direct_tables = _make_direct_tables(phase_page_table)

    swa_offsets = positions[:, None] - torch.arange(WINDOW_SIZE, dtype=torch.int32)[None, :]
    phase_swa = _gather_locs(full_loc_table, table_indices, swa_offsets)
    direct_swa = _gather_locs(direct_tables.swa_loc_table, torch.arange(positions.numel()), swa_offsets)

    c4_lengths = torch.div(seq_lens, 4, rounding_mode="floor")
    c4_raw = _make_sparse_raw(c4_lengths)
    c4_raw_positions = c4_raw * 4 + 3
    phase_c4_full = _gather_locs(full_loc_table, table_indices, c4_raw_positions)
    phase_c4_locs = torch.where(phase_c4_full >= 0, phase_c4_full // 4, phase_c4_full)
    direct_c4_locs = _gather_locs(
        direct_tables.c4_loc_table,
        torch.arange(positions.numel()),
        c4_raw.to(torch.int32),
    )
    direct_indexer_locs = _gather_locs(
        direct_tables.c4_indexer_loc_table,
        torch.arange(positions.numel()),
        c4_raw.to(torch.int32),
    )

    c128_lengths = torch.div(seq_lens, 128, rounding_mode="floor")
    c128_raw = _make_all_raw(c128_lengths)
    c128_raw_positions = c128_raw * 128 + 127
    phase_c128_full = _gather_locs(full_loc_table, table_indices, c128_raw_positions)
    phase_c128_locs = torch.where(phase_c128_full >= 0, phase_c128_full // 128, phase_c128_full)
    direct_c128_locs = _gather_locs(
        direct_tables.c128_loc_table,
        torch.arange(positions.numel()),
        c128_raw.to(torch.int32),
    )

    phase_c4_out = _compressed_locs_from_full(raw_out_loc, positions, 4)
    phase_c128_out = _compressed_locs_from_full(raw_out_loc, positions, 128)
    direct_c4_out = _direct_out_locs(
        torch.arange(positions.numel(), dtype=torch.int32),
        positions,
        direct_tables.c4_loc_table,
        4,
    )
    direct_c128_out = _direct_out_locs(
        torch.arange(positions.numel(), dtype=torch.int32),
        positions,
        direct_tables.c128_loc_table,
        128,
    )

    phase = Phase1Metadata(
        raw_out_loc=raw_out_loc,
        positions=positions,
        table_indices=table_indices,
        page_table=phase_page_table,
        seq_lens=seq_lens,
        swa_page_indices=phase_swa,
        c4_sparse_page_indices=phase_c4_locs,
        c128_page_indices=phase_c128_locs,
        c4_out_loc=phase_c4_out,
        c128_out_loc=phase_c128_out,
        c4_indexer_out_loc=phase_c4_out,
        c4_topk_lengths_raw=c4_lengths,
        indexer_page_table=phase_page_table,
        indexer_loc_table_gather=phase_c4_locs,
        c4_state_loc_from_swa=_state_loc_from_swa(phase_swa, ring_size=8),
        c128_state_loc_from_swa=_state_loc_from_swa(phase_swa, ring_size=128),
        c4_indexer_state_loc_from_swa=_state_loc_from_swa(phase_swa, ring_size=8),
    )
    direct = Phase1Metadata(
        raw_out_loc=raw_out_loc,
        positions=positions,
        table_indices=table_indices,
        page_table=phase_page_table,
        seq_lens=seq_lens,
        swa_page_indices=direct_swa,
        c4_sparse_page_indices=direct_c4_locs,
        c128_page_indices=direct_c128_locs,
        c4_out_loc=direct_c4_out,
        c128_out_loc=direct_c128_out,
        c4_indexer_out_loc=direct_c4_out,
        c4_topk_lengths_raw=c4_lengths,
        indexer_page_table=direct_tables.c4_indexer_page_table,
        indexer_loc_table_gather=direct_indexer_locs,
        c4_state_loc_from_swa=_state_loc_from_swa(direct_swa, ring_size=8),
        c128_state_loc_from_swa=_state_loc_from_swa(direct_swa, ring_size=128),
        c4_indexer_state_loc_from_swa=_state_loc_from_swa(direct_swa, ring_size=8),
    )
    schema = {
        "page_size": PAGE_SIZE,
        "swa_loc_table": {
            "shape": list(direct_tables.swa_loc_table.shape),
            "dtype": str(direct_tables.swa_loc_table.dtype),
            "meaning": "per active metadata row, logical token position -> SWA/full loc; B0 identity while full pages live",
        },
        "c4_loc_table": {
            "shape": list(direct_tables.c4_loc_table.shape),
            "dtype": str(direct_tables.c4_loc_table.dtype),
            "slots_per_full_page": PAGE_SIZE // 4,
            "meaning": "per active row, logical C4 slot -> physical C4 loc",
        },
        "c128_loc_table": {
            "shape": list(direct_tables.c128_loc_table.shape),
            "dtype": str(direct_tables.c128_loc_table.dtype),
            "slots_per_full_page": PAGE_SIZE // 128,
            "meaning": "per active row, logical C128 slot -> physical C128 loc",
        },
        "c4_indexer_loc_table": {
            "shape": list(direct_tables.c4_indexer_loc_table.shape),
            "dtype": str(direct_tables.c4_indexer_loc_table.dtype),
            "slots_per_full_page": PAGE_SIZE // 4,
            "meaning": "per active row, logical indexer/C4 slot -> physical indexer loc",
        },
        "component_page_tables": {
            "swa_page_table": list(direct_tables.swa_page_table.shape),
            "c4_page_table": list(direct_tables.c4_page_table.shape),
            "c128_page_table": list(direct_tables.c128_page_table.shape),
            "c4_indexer_page_table": list(direct_tables.c4_indexer_page_table.shape),
            "meaning": "logical full-page ordinal -> component physical page; B0 equals full page_table",
        },
        "state_loc_placeholder": direct_tables.state_loc_placeholder,
    }
    metadata = {
        "extend_lens": extend_lens,
        "req_seq_lens": req_seq_lens,
        "active_rows": int(positions.numel()),
        "max_seqlen_k": max_seqlen_k,
        "direct_schema": schema,
    }
    return phase, direct, metadata


COMPARE_FIELDS = (
    "swa_page_indices",
    "c4_sparse_page_indices",
    "c128_page_indices",
    "c4_out_loc",
    "c128_out_loc",
    "c4_indexer_out_loc",
    "indexer_page_table",
    "indexer_loc_table_gather",
    "c4_state_loc_from_swa",
    "c128_state_loc_from_swa",
    "c4_indexer_state_loc_from_swa",
)


def _compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    equal = bool(torch.equal(left, right))
    mismatch = None
    if not equal:
        diff = left != right
        if left.shape != right.shape:
            mismatch = {"reason": "shape", "phase1_shape": list(left.shape), "direct_shape": list(right.shape)}
        elif bool(torch.any(diff)):
            index = torch.nonzero(diff, as_tuple=False)[0].tolist()
            mismatch = {
                "index": index,
                "phase1": int(left[tuple(index)].item()),
                "direct": int(right[tuple(index)].item()),
            }
    return {
        "equal": equal,
        "phase1_shape": list(left.shape),
        "direct_shape": list(right.shape),
        "mismatch": mismatch,
    }


def _scenario_report(scenario: Scenario) -> dict[str, object]:
    phase, direct, metadata = _build_metadata_pair(scenario)
    comparisons = {
        field: _compare_tensor(getattr(phase, field), getattr(direct, field))
        for field in COMPARE_FIELDS
    }
    all_equal = all(bool(item["equal"]) for item in comparisons.values())
    return {
        "name": scenario.name,
        "phase": scenario.phase,
        "hit_style": scenario.hit_style,
        "note": scenario.note,
        "reqs": [
            {
                "table_idx": req.table_idx,
                "device_len": req.device_len,
                "cached_len": req.cached_len,
                "extend_len": req.extend_len,
            }
            for req in scenario.reqs
        ],
        **metadata,
        "comparisons": comparisons,
        "all_equal": all_equal,
    }


def _write_equality_table(reports: list[dict[str, object]]) -> None:
    columns = [
        "scenario",
        "phase",
        "hit_style",
        "active_rows",
        *COMPARE_FIELDS,
        "all_equal",
    ]
    csv_rows = []
    for report in reports:
        row = {
            "scenario": report["name"],
            "phase": report["phase"],
            "hit_style": report["hit_style"],
            "active_rows": report["active_rows"],
            "all_equal": "pass" if report["all_equal"] else "fail",
        }
        comparisons = report["comparisons"]
        assert isinstance(comparisons, dict)
        for field in COMPARE_FIELDS:
            item = comparisons[field]
            assert isinstance(item, dict)
            row[field] = "pass" if item["equal"] else "fail"
        csv_rows.append(row)

    with (SUMMARIES / "equality_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "| scenario | phase | hit style | rows | swa | c4 sparse | c128 | c4 out | c128 out | indexer out | indexer page | indexer loc | state c4 | state c128 | state indexer | result |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in csv_rows:
        lines.append(
            "| {scenario} | {phase} | {hit_style} | {active_rows} | "
            "{swa_page_indices} | {c4_sparse_page_indices} | {c128_page_indices} | "
            "{c4_out_loc} | {c128_out_loc} | {c4_indexer_out_loc} | "
            "{indexer_page_table} | {indexer_loc_table_gather} | "
            "{c4_state_loc_from_swa} | {c128_state_loc_from_swa} | "
            "{c4_indexer_state_loc_from_swa} | {all_equal} |".format(**row)
        )
    (SUMMARIES / "equality_table.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARIES.mkdir(parents=True, exist_ok=True)

    scenarios = [
        Scenario(
            name="full_hit_decode_page257",
            phase="decode",
            hit_style="full hit",
            reqs=(ReqLayout(table_idx=0, device_len=257, cached_len=256),),
            note="page-aligned prefix is live; metadata row is the first post-hit decode token",
        ),
        Scenario(
            name="partial_hit_suffix_prefill_256_to_258",
            phase="prefill",
            hit_style="partial hit",
            reqs=(ReqLayout(table_idx=0, device_len=258, cached_len=256),),
            note="one cached full page plus two suffix prefill tokens",
        ),
        Scenario(
            name="miss_style_prefill_258",
            phase="prefill",
            hit_style="miss style",
            reqs=(ReqLayout(table_idx=0, device_len=258, cached_len=0),),
            note="no cached prefix; all prompt tokens build metadata from fresh full pages",
        ),
        Scenario(
            name="page_boundaries_255_256_257_258_decode",
            phase="decode",
            hit_style="boundary",
            reqs=(
                ReqLayout(table_idx=0, device_len=255, cached_len=254),
                ReqLayout(table_idx=1, device_len=256, cached_len=255),
                ReqLayout(table_idx=2, device_len=257, cached_len=256),
                ReqLayout(table_idx=3, device_len=258, cached_len=257),
            ),
            note="full-page boundary rows around 256 tokens",
        ),
        Scenario(
            name="c4_boundaries_decode",
            phase="decode",
            hit_style="C4 boundary",
            reqs=(
                ReqLayout(table_idx=0, device_len=3, cached_len=2),
                ReqLayout(table_idx=1, device_len=4, cached_len=3),
                ReqLayout(table_idx=2, device_len=5, cached_len=4),
                ReqLayout(table_idx=3, device_len=8, cached_len=7),
                ReqLayout(table_idx=4, device_len=9, cached_len=8),
            ),
            note="positions before/on/after C4 compression endpoints",
        ),
        Scenario(
            name="c128_boundaries_decode",
            phase="decode",
            hit_style="C128 boundary",
            reqs=(
                ReqLayout(table_idx=0, device_len=127, cached_len=126),
                ReqLayout(table_idx=1, device_len=128, cached_len=127),
                ReqLayout(table_idx=2, device_len=129, cached_len=128),
                ReqLayout(table_idx=3, device_len=256, cached_len=255),
                ReqLayout(table_idx=4, device_len=257, cached_len=256),
            ),
            note="positions before/on/after C128 endpoints, including page boundary",
        ),
        Scenario(
            name="swa_127_128_129_decode",
            phase="decode",
            hit_style="SWA boundary",
            reqs=(
                ReqLayout(table_idx=0, device_len=127, cached_len=126),
                ReqLayout(table_idx=1, device_len=128, cached_len=127),
                ReqLayout(table_idx=2, device_len=129, cached_len=128),
            ),
            note="sliding-window length below/equal/above 128",
        ),
        Scenario(
            name="batched_same_layout_rows",
            phase="prefill",
            hit_style="batched same-layout",
            reqs=(
                ReqLayout(table_idx=0, device_len=258, cached_len=256),
                ReqLayout(table_idx=1, device_len=258, cached_len=256),
            ),
            note="two requests with the same logical layout but distinct physical pages",
        ),
    ]

    reports = [_scenario_report(scenario) for scenario in scenarios]
    all_equal = all(bool(report["all_equal"]) for report in reports)
    payload = {
        "page_size": PAGE_SIZE,
        "window_size": WINDOW_SIZE,
        "index_topk": INDEX_TOPK,
        "alignment": ALIGNMENT,
        "state_policy": (
            "state loc is a B0 placeholder: current mini does not pass state loc through "
            "attention metadata; this probe validates the existing full/SWA-derived formula "
            "against the direct SWA loc table and defers independent state ownership to B2"
        ),
        "reports": reports,
        "summary": {
            "all_equal": all_equal,
            "scenario_count": len(reports),
            "decision": "proceed_to_TARGET_08.21.2" if all_equal else "revise_B0_or_block",
        },
    }
    (RAW / "component_loc_table_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_equality_table(reports)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = ROOT / "summaries"
RAW = ROOT / "raw"
PAGE_SIZE = 256
FIXED_PAGES = 128
BYTES_PER_PHASE1_PAGE = 19_313_920

CASES = (
    ("short shared prefix", 1),
    ("1024-token prefix", 4),
    ("4096-token prefix", 16),
    ("multi-prefix mixed", 40),
    ("08.10 sustained workload", 56),
    ("eviction pressure", 112),
)


def _gib(value: int) -> float:
    return value / float(1 << 30)


def _row(name: str, pages: int) -> dict[str, int | float | str]:
    phase1_tokens = pages * PAGE_SIZE
    phase1_bytes = pages * BYTES_PER_PHASE1_PAGE
    c4_slots = pages * (PAGE_SIZE // 4)
    c128_slots = pages * (PAGE_SIZE // 128)
    c4_state_slots = pages * 8
    c128_state_slots = pages * 128
    return {
        "case": name,
        "phase1_full_swa_pages": pages,
        "phase1_tokens": phase1_tokens,
        "phase1_gib_per_rank": round(_gib(phase1_bytes), 3),
        "phase1_c4_slots": c4_slots,
        "phase1_c128_slots": c128_slots,
        "phase1_indexer_slots": c4_slots,
        "phase1_c4_state_slots": c4_state_slots,
        "phase1_c128_state_slots": c128_state_slots,
        "phase1_indexer_state_slots": c4_state_slots,
        "v1_fail_closed_full_swa_pages": pages,
        "v1_fail_closed_c4_slots": c4_slots,
        "v1_fail_closed_c128_slots": c128_slots,
        "v1_fail_closed_indexer_slots": c4_slots,
        "v1_fail_closed_c4_state_slots": c4_state_slots,
        "v1_fail_closed_c128_state_slots": c128_state_slots,
        "v1_fail_closed_indexer_state_slots": c4_state_slots,
        "v1_recovered_pages": 0,
        "v1_recovered_tokens": 0,
        "v1_recovered_gib_per_rank": 0.0,
        "decision": "reject_runtime_retention",
    }


def main() -> int:
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    rows = [_row(name, pages) for name, pages in CASES]

    raw_payload = {
        "page_size": PAGE_SIZE,
        "fixed_pages": FIXED_PAGES,
        "bytes_per_phase1_page": BYTES_PER_PHASE1_PAGE,
        "v1_runtime_status": "fail_closed",
        "reason": (
            "Current mini derives C4/C128/indexer/compression-state locations from "
            "full-token pages; releasing those pages would risk component reuse "
            "collisions or double free."
        ),
        "rows": rows,
    }
    (RAW / "capacity_summary.json").write_text(
        json.dumps(raw_payload, indent=2, sort_keys=True) + "\n"
    )

    with (SUMMARIES / "recovered_capacity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| case | phase-1 full/SWA pages | V1 full/SWA pages | phase-1 C4/C128/indexer slots | phase-1 state slots C4/C128/indexer | V1 C4/C128/indexer slots | V1 state slots C4/C128/indexer | recovered pages | recovered tokens | recovered GiB/rank | decision |",
        "| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {phase1_full_swa_pages} | {v1_fail_closed_full_swa_pages} | "
            "{phase1_c4_slots}/{phase1_c128_slots}/{phase1_indexer_slots} | "
            "{phase1_c4_state_slots}/{phase1_c128_state_slots}/{phase1_indexer_state_slots} | "
            "{v1_fail_closed_c4_slots}/{v1_fail_closed_c128_slots}/{v1_fail_closed_indexer_slots} | "
            "{v1_fail_closed_c4_state_slots}/{v1_fail_closed_c128_state_slots}/{v1_fail_closed_indexer_state_slots} | "
            "{v1_recovered_pages} | {v1_recovered_tokens} | "
            "{v1_recovered_gib_per_rank:.3f} | {decision} |".format(**row)
        )
    (SUMMARIES / "recovered_capacity.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

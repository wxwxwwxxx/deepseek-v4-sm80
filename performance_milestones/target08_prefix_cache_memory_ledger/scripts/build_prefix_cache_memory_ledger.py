#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


GIB = 1024**3
PAGE_SIZE = 256
NUM_PAGES = 128
KV_POOL_BYTES = 2_491_495_680
GRAPH_PRIVATE_POOL_BYTES = 20_440_940_544
BF16_CACHE_BASELINE_BYTES = round(1.588 * GIB)
DEVICE_FREE_AFTER_GRAPH_BYTES = 39_135_936_512

MODEL = {
    "num_layers": 43,
    "head_dim": 512,
    "index_head_dim": 128,
    "c4_layers": 21,
    "c128_layers": 20,
    "dtype_size": 2,
    "state_dtype_size": 2,
    "c4_state_ring_size": 8,
    "c128_state_ring_size": 128,
    "indexer_fp8_extra_enabled": True,
}

CASES = [
    {
        "case": "short shared prefix",
        "source": "TARGET 08.10 prefix_full_hit_257_bs4",
        "interpretation": "257-token prompt; retained hit length is page-aligned to 256 tokens.",
        "retained_pages": 1,
        "measurement": "measured",
    },
    {
        "case": "1024-token prefix",
        "source": "TARGET 08 phase-1 shared_prompt_reuse_bs8",
        "interpretation": "One 1024-token shared prefix retained for seven reuse requests.",
        "retained_pages": 4,
        "measurement": "measured",
    },
    {
        "case": "4096-token prefix",
        "source": "Formula estimate, matching the TARGET 08.10 16-page retained row",
        "interpretation": "Canonical 4096-token page-aligned retained prefix.",
        "retained_pages": 16,
        "measurement": "estimated",
    },
    {
        "case": "multi-prefix mixed",
        "source": "TARGET 08.10 prefix_mixed_hit_miss_bs16 final retained state",
        "interpretation": "Mixed full-hit and unrelated-miss workload after prior same-process cases.",
        "retained_pages": 40,
        "measurement": "measured",
    },
    {
        "case": "08.10 sustained workload",
        "source": "TARGET 08.10 prefix_multi_112req_wave16 final retained state",
        "interpretation": "112 requests in waves of 16 cycling across eight 512-token shared prefixes.",
        "retained_pages": 56,
        "measurement": "measured",
    },
    {
        "case": "eviction pressure",
        "source": "TARGET 08.10 prefix_eviction_pressure_96req_wave16 final retained state",
        "interpretation": "96 distinct 512-token prefixes under --num-pages 128.",
        "retained_pages": 112,
        "measurement": "measured",
    },
]


def gib(value: float | int) -> float:
    return float(value) / GIB


def retention_for_pages(pages: int) -> dict[str, int]:
    tokens = pages * PAGE_SIZE
    c4_slots = pages * (PAGE_SIZE // 4)
    c128_slots = pages * (PAGE_SIZE // 128)
    c4_state_slots = pages * MODEL["c4_state_ring_size"]
    c128_state_slots = pages * MODEL["c128_state_ring_size"]
    c4_indexer_state_slots = c4_state_slots

    swa_bytes = (
        MODEL["num_layers"]
        * tokens
        * MODEL["head_dim"]
        * MODEL["dtype_size"]
    )
    c4_bytes = (
        MODEL["c4_layers"]
        * c4_slots
        * MODEL["head_dim"]
        * MODEL["dtype_size"]
    )
    c128_bytes = (
        MODEL["c128_layers"]
        * c128_slots
        * MODEL["head_dim"]
        * MODEL["dtype_size"]
    )
    c4_indexer_bytes = (
        MODEL["c4_layers"]
        * c4_slots
        * MODEL["index_head_dim"]
        * MODEL["dtype_size"]
    )
    c4_indexer_fp8_bytes = (
        MODEL["c4_layers"] * c4_slots * (MODEL["index_head_dim"] + 4)
        if MODEL["indexer_fp8_extra_enabled"]
        else 0
    )
    c4_state_bytes = (
        MODEL["c4_layers"]
        * pages
        * MODEL["c4_state_ring_size"]
        * 4
        * MODEL["head_dim"]
        * MODEL["state_dtype_size"]
    )
    c4_indexer_state_bytes = (
        MODEL["c4_layers"]
        * pages
        * MODEL["c4_state_ring_size"]
        * 4
        * MODEL["index_head_dim"]
        * MODEL["state_dtype_size"]
    )
    c128_state_bytes = (
        MODEL["c128_layers"]
        * pages
        * MODEL["c128_state_ring_size"]
        * 2
        * MODEL["head_dim"]
        * MODEL["state_dtype_size"]
    )
    compress_state_bytes = c4_state_bytes + c4_indexer_state_bytes + c128_state_bytes
    total_bytes = (
        swa_bytes
        + c4_bytes
        + c128_bytes
        + c4_indexer_bytes
        + c4_indexer_fp8_bytes
        + compress_state_bytes
    )
    return {
        "retained_pages": pages,
        "retained_tokens": tokens,
        "full_slots": tokens,
        "swa_slots": tokens,
        "c4_slots": c4_slots,
        "c128_slots": c128_slots,
        "c4_indexer_slots": c4_slots,
        "c4_state_slots": c4_state_slots,
        "c128_state_slots": c128_state_slots,
        "c4_indexer_state_slots": c4_indexer_state_slots,
        "swa_bytes": swa_bytes,
        "c4_bytes": c4_bytes,
        "c128_bytes": c128_bytes,
        "c4_indexer_bytes": c4_indexer_bytes,
        "c4_indexer_fp8_bytes": c4_indexer_fp8_bytes,
        "compress_state_bytes": compress_state_bytes,
        "c4_state_bytes": c4_state_bytes,
        "c128_state_bytes": c128_state_bytes,
        "c4_indexer_state_bytes": c4_indexer_state_bytes,
        "retained_memory_bytes": total_bytes,
    }


def enrich_case(case: dict[str, Any]) -> dict[str, Any]:
    pages = int(case["retained_pages"])
    retention = retention_for_pages(pages)
    bytes_per_kv_page = KV_POOL_BYTES / NUM_PAGES
    retained_bytes = retention["retained_memory_bytes"]
    equivalent_kv_pages_by_bytes = retained_bytes / bytes_per_kv_page
    free_pages = NUM_PAGES - pages
    fixed_resident_bytes = GRAPH_PRIVATE_POOL_BYTES + BF16_CACHE_BASELINE_BYTES + KV_POOL_BYTES
    recoverable_full_pages_upper = max(pages - 1, 0)
    one_page = retention_for_pages(1)
    swa_only_saved_bytes = recoverable_full_pages_upper * one_page["swa_bytes"]
    swa_state_saved_bytes = recoverable_full_pages_upper * (
        one_page["swa_bytes"] + one_page["compress_state_bytes"]
    )
    compressed_component_bytes = (
        retention["c4_bytes"]
        + retention["c128_bytes"]
        + retention["c4_indexer_bytes"]
        + retention["c4_indexer_fp8_bytes"]
    )
    return {
        **case,
        **retention,
        "retained_memory_gib": gib(retained_bytes),
        "kv_pool_fraction_by_pages": pages / NUM_PAGES,
        "kv_pool_fraction_by_retained_bytes": retained_bytes / KV_POOL_BYTES,
        "equivalent_kv_pages_by_bytes": equivalent_kv_pages_by_bytes,
        "equivalent_kv_tokens_by_bytes": equivalent_kv_pages_by_bytes * PAGE_SIZE,
        "equivalent_4096_prompts_by_pages": pages / 16,
        "equivalent_4096_1024_requests_by_pages": pages / 20,
        "remaining_kv_pages_before_active_requests": free_pages,
        "remaining_kv_tokens_before_active_requests": free_pages * PAGE_SIZE,
        "remaining_4096_prompts": free_pages / 16,
        "remaining_4096_1024_requests": free_pages / 20,
        "graph_private_pool_bytes": GRAPH_PRIVATE_POOL_BYTES,
        "bf16_cache_baseline_bytes": BF16_CACHE_BASELINE_BYTES,
        "fixed_kv_pool_bytes": KV_POOL_BYTES,
        "fixed_resident_bytes": fixed_resident_bytes,
        "fixed_resident_gib": gib(fixed_resident_bytes),
        "device_free_after_graph_fixed_bytes": DEVICE_FREE_AFTER_GRAPH_BYTES,
        "device_free_after_graph_fixed_gib": gib(DEVICE_FREE_AFTER_GRAPH_BYTES),
        "conservative_device_free_if_prefix_were_extra_bytes": (
            DEVICE_FREE_AFTER_GRAPH_BYTES - retained_bytes
        ),
        "conservative_device_free_if_prefix_were_extra_gib": gib(
            DEVICE_FREE_AFTER_GRAPH_BYTES - retained_bytes
        ),
        "recoverable_full_pages_upper_bound": recoverable_full_pages_upper,
        "recoverable_full_tokens_upper_bound": recoverable_full_pages_upper * PAGE_SIZE,
        "sglang_swa_only_saved_bytes_upper": swa_only_saved_bytes,
        "sglang_swa_only_saved_gib_upper": gib(swa_only_saved_bytes),
        "sglang_swa_state_saved_bytes_upper": swa_state_saved_bytes,
        "sglang_swa_state_saved_gib_upper": gib(swa_state_saved_bytes),
        "sglang_swa_state_saved_equiv_kv_pages": swa_state_saved_bytes / bytes_per_kv_page,
        "compressed_component_bytes_kept": compressed_component_bytes,
        "compressed_component_gib_kept": gib(compressed_component_bytes),
    }


def read_source_measurements(repo_root: Path) -> dict[str, Any]:
    stability_path = (
        repo_root
        / "performance_milestones/target08_prefix_cache_serving_stability/summaries/"
        / "prefix_cache_serving_stability_summary.json"
    )
    phase1_path = (
        repo_root
        / "performance_milestones/target08_radix_prefix_dsv4/perf_prefix_on/reports/"
        / "000_shared_prompt_reuse_bs8__dsv4_sm80_a100_victory.json"
    )
    out: dict[str, Any] = {}
    if stability_path.exists():
        stability = json.loads(stability_path.read_text())
        out["target08_10_decision_inputs"] = stability.get("decision_inputs", {})
        out["target08_10_prefix_on_rows"] = [
            row
            for row in stability.get("case_rows", [])
            if row.get("mode") == "prefix_on"
        ]
        out["target08_10_text_smoke_long_prefix_on"] = (
            stability.get("text_smoke_long_prefix", {}).get("prefix_on", {})
        )
    if phase1_path.exists():
        phase1 = json.loads(phase1_path.read_text())
        out["target08_phase1_1024_prefix"] = (
            phase1.get("metrics", {}).get("prefix_cache", {}).get("rank0_final", {})
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def fmt_gib(value: float | int) -> str:
    return f"{gib(value):.3f}"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_summary_markdown(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    ledger_rows = []
    for row in rows:
        ledger_rows.append(
            [
                row["case"],
                str(row["retained_pages"]),
                str(row["retained_tokens"]),
                str(row["full_slots"]),
                str(row["c4_slots"]),
                str(row["c128_slots"]),
                str(row["c4_indexer_slots"]),
                str(row["c4_state_slots"]),
                str(row["c128_state_slots"]),
                str(row["c4_indexer_state_slots"]),
                fmt_gib(row["retained_memory_bytes"]),
            ]
        )
    (out_dir / "ledger_table.md").write_text(
        markdown_table(
            [
                "case",
                "pages",
                "tokens",
                "full/SWA slots",
                "C4 slots",
                "C128 slots",
                "indexer slots",
                "C4 state",
                "C128 state",
                "idx state",
                "GiB/rank",
            ],
            ledger_rows,
        )
    )

    component_rows = []
    for row in rows:
        component_rows.append(
            [
                row["case"],
                fmt_gib(row["swa_bytes"]),
                fmt_gib(row["c4_bytes"]),
                fmt_gib(row["c128_bytes"]),
                fmt_gib(row["c4_indexer_bytes"]),
                fmt_gib(row["c4_indexer_fp8_bytes"]),
                fmt_gib(row["compress_state_bytes"]),
                fmt_gib(row["retained_memory_bytes"]),
            ]
        )
    (out_dir / "component_bytes_table.md").write_text(
        markdown_table(
            [
                "case",
                "SWA/full GiB",
                "C4 GiB",
                "C128 GiB",
                "indexer BF16 GiB",
                "indexer FP8 extra GiB",
                "compress-state GiB",
                "total GiB",
            ],
            component_rows,
        )
    )

    capacity_rows = []
    for row in rows:
        capacity_rows.append(
            [
                row["case"],
                fmt_float(row["kv_pool_fraction_by_pages"] * 100, 1) + "%",
                fmt_float(row["equivalent_4096_prompts_by_pages"], 2),
                fmt_float(row["equivalent_4096_1024_requests_by_pages"], 2),
                str(row["remaining_kv_pages_before_active_requests"]),
                fmt_float(row["remaining_4096_1024_requests"], 2),
                fmt_float(row["conservative_device_free_if_prefix_were_extra_gib"], 3),
            ]
        )
    (out_dir / "capacity_table.md").write_text(
        markdown_table(
            [
                "case",
                "KV pool pages used",
                "4096 prompts eq",
                "4096+1024 req eq",
                "remaining KV pages",
                "remaining 4096+1024 reqs",
                "free GiB if extra",
            ],
            capacity_rows,
        )
    )

    savings_rows = []
    for row in rows:
        savings_rows.append(
            [
                row["case"],
                str(row["recoverable_full_pages_upper_bound"]),
                str(row["recoverable_full_tokens_upper_bound"]),
                fmt_gib(row["sglang_swa_only_saved_bytes_upper"]),
                fmt_gib(row["sglang_swa_state_saved_bytes_upper"]),
                fmt_float(row["sglang_swa_state_saved_equiv_kv_pages"], 1),
                fmt_gib(row["compressed_component_bytes_kept"]),
            ]
        )
    (out_dir / "sglang_savings_table.md").write_text(
        markdown_table(
            [
                "case",
                "recoverable full pages upper",
                "recoverable tokens upper",
                "SWA-only saved GiB",
                "SWA+state saved GiB",
                "SWA+state saved eq KV pages",
                "compressed kept GiB",
            ],
            savings_rows,
        )
    )


def main() -> None:
    script_path = Path(__file__).resolve()
    milestone_dir = script_path.parents[1]
    repo_root = script_path.parents[3]
    raw_dir = milestone_dir / "raw"
    summaries_dir = milestone_dir / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    rows = [enrich_case(case) for case in CASES]
    fixed_inputs = {
        "page_size": PAGE_SIZE,
        "num_pages": NUM_PAGES,
        "kv_pool_bytes": KV_POOL_BYTES,
        "kv_pool_gib": gib(KV_POOL_BYTES),
        "kv_pool_bytes_per_accounting_page": KV_POOL_BYTES / NUM_PAGES,
        "marginal_retained_bytes_per_page": retention_for_pages(1)["retained_memory_bytes"],
        "graph_private_pool_bytes": GRAPH_PRIVATE_POOL_BYTES,
        "graph_private_pool_gib": gib(GRAPH_PRIVATE_POOL_BYTES),
        "bf16_cache_baseline_bytes": BF16_CACHE_BASELINE_BYTES,
        "bf16_cache_baseline_gib": gib(BF16_CACHE_BASELINE_BYTES),
        "fixed_resident_bytes": GRAPH_PRIVATE_POOL_BYTES
        + BF16_CACHE_BASELINE_BYTES
        + KV_POOL_BYTES,
        "fixed_resident_gib": gib(
            GRAPH_PRIVATE_POOL_BYTES + BF16_CACHE_BASELINE_BYTES + KV_POOL_BYTES
        ),
        "device_free_after_graph_fixed_bytes": DEVICE_FREE_AFTER_GRAPH_BYTES,
        "device_free_after_graph_fixed_gib": gib(DEVICE_FREE_AFTER_GRAPH_BYTES),
        "model": MODEL,
    }
    (raw_dir / "fixed_inputs.json").write_text(json.dumps(fixed_inputs, indent=2) + "\n")
    (raw_dir / "source_measurements.json").write_text(
        json.dumps(read_source_measurements(repo_root), indent=2) + "\n"
    )
    (raw_dir / "ledger_cases.json").write_text(json.dumps(rows, indent=2) + "\n")

    write_csv(
        summaries_dir / "ledger_cases.csv",
        rows,
        [
            "case",
            "measurement",
            "retained_pages",
            "retained_tokens",
            "full_slots",
            "swa_slots",
            "c4_slots",
            "c128_slots",
            "c4_indexer_slots",
            "c4_state_slots",
            "c128_state_slots",
            "c4_indexer_state_slots",
            "swa_bytes",
            "c4_bytes",
            "c128_bytes",
            "c4_indexer_bytes",
            "c4_indexer_fp8_bytes",
            "compress_state_bytes",
            "retained_memory_bytes",
            "retained_memory_gib",
            "kv_pool_fraction_by_pages",
            "equivalent_4096_prompts_by_pages",
            "equivalent_4096_1024_requests_by_pages",
            "remaining_kv_pages_before_active_requests",
        ],
    )
    write_csv(
        summaries_dir / "sglang_savings.csv",
        rows,
        [
            "case",
            "retained_pages",
            "recoverable_full_pages_upper_bound",
            "recoverable_full_tokens_upper_bound",
            "sglang_swa_only_saved_bytes_upper",
            "sglang_swa_only_saved_gib_upper",
            "sglang_swa_state_saved_bytes_upper",
            "sglang_swa_state_saved_gib_upper",
            "sglang_swa_state_saved_equiv_kv_pages",
            "compressed_component_bytes_kept",
            "compressed_component_gib_kept",
        ],
    )
    write_summary_markdown(summaries_dir, rows)

    decision = {
        "target_08_20_decision": "GO_WITH_GUARDRAILS",
        "reason": (
            "Phase-1 full-page retention crosses the 20-30% KV-pool threshold in "
            "08.10 sustained traffic and reaches 112/128 pages under eviction pressure."
        ),
        "guardrails": [
            "Do not promote prefix cache by default before the 08.10 synthetic token mismatch is resolved.",
            "Keep 08.20 scoped to independent SWA/component retention; do not add low-precision cache or graph allocator work.",
            "Require component-level metrics and correctness tests for full/SWA/C4/C128/indexer/compress-state ownership.",
        ],
    }
    (summaries_dir / "go_no_go.json").write_text(json.dumps(decision, indent=2) + "\n")
    (summaries_dir / "go_no_go.md").write_text(
        "# TARGET 08.20 Go/No-Go\n\n"
        "**Decision: GO with guardrails.**\n\n"
        "The sustained 08.10 retained state uses 56/128 pages, and eviction pressure "
        "reaches 112/128 pages.  That is above the 20%-30% investigation threshold "
        "and removes multiple 4096+1024 request equivalents from the fixed page pool.\n\n"
        "Guardrails: resolve the generated-token/logit correctness follow-up before "
        "any default promotion, and keep 08.20 focused on independent SWA/component "
        "retention rather than new low-precision, graph, or global allocator work.\n"
    )


if __name__ == "__main__":
    main()

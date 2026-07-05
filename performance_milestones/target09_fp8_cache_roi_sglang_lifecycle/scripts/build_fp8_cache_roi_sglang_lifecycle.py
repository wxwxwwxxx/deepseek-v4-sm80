#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SUMMARY_DIR = ROOT / "summaries"

PREFLIGHT_LEDGER = (
    REPO / "performance_milestones/target09_low_precision_preflight/summaries/memory_ledger.json"
)
FP8_SLICE_SUMMARY = (
    REPO
    / "performance_milestones/target09_minimal_fp8_kv_cache_slice/summaries/swa_packed_mla_slice_harness.json"
)

GIB = 1024**3
MIB = 1024**2

NUM_LAYERS = 43
C4_LAYERS = 21
C128_LAYERS = 20
HEAD_DIM = 512
INDEX_HEAD_DIM = 128
PAGE_SIZE = 256
NUM_PAGES = 128
SLIDING_WINDOW = 128
DTYPE_SIZE = 2
STATE_DTYPE_SIZE = 2
C4_STATE_RING = 8
C128_STATE_RING = 128

MLA_NOPE_FP8_BYTES = 448
MLA_ROPE_BF16_BYTES = 64 * 2
MLA_SCALE_PAD_BYTES = 8
MLA_SLOT_BYTES = 576
MLA_TOKEN_BYTES = MLA_NOPE_FP8_BYTES + MLA_ROPE_BF16_BYTES + MLA_SCALE_PAD_BYTES
INDEXER_FP8_SLOT_BYTES = INDEX_HEAD_DIM + 4


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def gib(value: int | float) -> str:
    return f"{float(value) / GIB:.3f}"


def mib(value: int | float) -> str:
    return f"{float(value) / MIB:.2f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def current_components() -> dict[str, dict[str, Any]]:
    swa_bf16 = NUM_LAYERS * PAGE_SIZE * HEAD_DIM * DTYPE_SIZE
    c4_bf16 = C4_LAYERS * (PAGE_SIZE // 4) * HEAD_DIM * DTYPE_SIZE
    c128_bf16 = C128_LAYERS * (PAGE_SIZE // 128) * HEAD_DIM * DTYPE_SIZE
    indexer_bf16 = C4_LAYERS * (PAGE_SIZE // 4) * INDEX_HEAD_DIM * DTYPE_SIZE
    indexer_fp8_side = C4_LAYERS * (PAGE_SIZE // 4) * INDEXER_FP8_SLOT_BYTES
    c4_state = C4_LAYERS * C4_STATE_RING * 4 * HEAD_DIM * STATE_DTYPE_SIZE
    c4_indexer_state = (
        C4_LAYERS * C4_STATE_RING * 4 * INDEX_HEAD_DIM * STATE_DTYPE_SIZE
    )
    c128_state = C128_LAYERS * C128_STATE_RING * 2 * HEAD_DIM * STATE_DTYPE_SIZE
    state = c4_state + c4_indexer_state + c128_state
    no_side = swa_bf16 + c4_bf16 + c128_bf16 + indexer_bf16 + state
    with_side = no_side + indexer_fp8_side
    return {
        "swa_bf16": {"bytes_per_page": swa_bf16, "evidence": "source-derived"},
        "c4_bf16": {"bytes_per_page": c4_bf16, "evidence": "source-derived"},
        "c128_bf16": {"bytes_per_page": c128_bf16, "evidence": "source-derived"},
        "c4_indexer_bf16": {
            "bytes_per_page": indexer_bf16,
            "evidence": "source-derived",
        },
        "c4_indexer_fp8_side_cache": {
            "bytes_per_page": indexer_fp8_side,
            "evidence": "runtime-active in promoted bundle, source-derived size",
        },
        "c4_state_bf16": {"bytes_per_page": c4_state, "evidence": "source-derived"},
        "c4_indexer_state_bf16": {
            "bytes_per_page": c4_indexer_state,
            "evidence": "source-derived",
        },
        "c128_state_bf16": {
            "bytes_per_page": c128_state,
            "evidence": "source-derived",
        },
        "state_total": {"bytes_per_page": state, "evidence": "source-derived"},
        "total_without_fp8_side": {
            "bytes_per_page": no_side,
            "bytes_at_128_pages": no_side * NUM_PAGES,
            "evidence": "source-derived formula",
        },
        "total_with_fp8_side": {
            "bytes_per_page": with_side,
            "bytes_at_128_pages": with_side * NUM_PAGES,
            "evidence": "source-derived formula; matches promoted indexer FP8 side mode",
        },
    }


def fp8_components(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    swa_fp8_per_layer_page = align_up(PAGE_SIZE * MLA_TOKEN_BYTES, MLA_SLOT_BYTES)
    c4_fp8_per_layer_page = align_up((PAGE_SIZE // 4) * MLA_TOKEN_BYTES, MLA_SLOT_BYTES)
    c128_fp8_per_layer_page = align_up(
        (PAGE_SIZE // 128) * MLA_TOKEN_BYTES, MLA_SLOT_BYTES
    )
    swa_fp8 = NUM_LAYERS * swa_fp8_per_layer_page
    c4_fp8 = C4_LAYERS * c4_fp8_per_layer_page
    c128_fp8 = C128_LAYERS * c128_fp8_per_layer_page
    indexer_fp8 = C4_LAYERS * (PAGE_SIZE // 4) * INDEXER_FP8_SLOT_BYTES
    state = current["state_total"]["bytes_per_page"]
    current_with_side = current["total_with_fp8_side"]["bytes_per_page"]
    current_no_side = current["total_without_fp8_side"]["bytes_per_page"]
    return {
        "layout": {
            "mla_token_bytes_unpadded": MLA_TOKEN_BYTES,
            "mla_slot_bytes": MLA_SLOT_BYTES,
            "swa_page_bytes_per_layer": swa_fp8_per_layer_page,
            "c4_page_bytes_per_layer": c4_fp8_per_layer_page,
            "c128_page_bytes_per_layer": c128_fp8_per_layer_page,
            "indexer_fp8_slot_bytes": INDEXER_FP8_SLOT_BYTES,
        },
        "swa_fp8_page": swa_fp8,
        "c4_fp8_page": c4_fp8,
        "c128_fp8_page": c128_fp8,
        "indexer_fp8_replacement_page": indexer_fp8,
        "swa_only_with_existing_side_page": current_with_side
        - current["swa_bf16"]["bytes_per_page"]
        + swa_fp8,
        "swa_only_without_side_page": current_no_side
        - current["swa_bf16"]["bytes_per_page"]
        + swa_fp8,
        "mla_only_with_indexer_bf16_page": (
            swa_fp8
            + c4_fp8
            + c128_fp8
            + current["c4_indexer_bf16"]["bytes_per_page"]
            + state
        ),
        "full_source_aligned_replacement_page": (
            swa_fp8 + c4_fp8 + c128_fp8 + indexer_fp8 + state
        ),
    }


def load_speed(slice_summary: dict[str, Any]) -> dict[str, Any]:
    microbench = slice_summary["microbench"]
    graph = slice_summary["graph_safety"]
    deltas = [
        row["fp8_combined_store_gather_dequant_ms"]
        - row["bf16_combined_store_gather_ms"]
        for row in microbench
    ]
    store_deltas = [
        row["fp8_packed_store_quant_ms"] - row["bf16_store_ms"] for row in microbench
    ]
    gather_deltas = [
        row["fp8_selected_gather_dequant_ms"] - row["bf16_selected_gather_ms"]
        for row in microbench
    ]
    return {
        "microbench": microbench,
        "graph_safety": graph,
        "delta_min_ms": min(deltas),
        "delta_max_ms": max(deltas),
        "delta_mean_ms": sum(deltas) / len(deltas),
        "store_delta_mean_ms": sum(store_deltas) / len(store_deltas),
        "gather_delta_mean_ms": sum(gather_deltas) / len(gather_deltas),
        "worst_43_layer_separated_delta_ms": max(deltas) * NUM_LAYERS,
    }


def lifecycle_scenarios(current: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    preflight = load_json(PREFLIGHT_LEDGER)
    by_name = {r["scenario"]: r for r in preflight["prefix_cache_by_scenario"]}

    # These are scenario estimates because mini does not yet runtime-prove the
    # independent SWA component.  The low/high values bound active-tail and
    # retained-short-branch interpretations where the raw reports do not expose
    # a branch-count ledger.
    specs = [
        (
            "historical_4096_1024_bs4",
            4,
            4,
            "bs4 long prompts; one page-aligned SWA tail per active/retained branch",
        ),
        (
            "serving_mixed_112req_wave16",
            16,
            by_name["serving_mixed_112req_wave16"]["retained_prefix_pages"],
            "wave16 active-tail lower bound; short no-share branch retention upper bound",
        ),
        (
            "prefix_multi_112req_wave16",
            8,
            16,
            "eight shared 512-token prefixes lower bound; wave16 active tails upper bound",
        ),
        (
            "serving_mixed_256req_wave64_est",
            64,
            64,
            "synthetic higher-concurrency wave64 serving estimate",
        ),
    ]
    non_swa_current = (
        current["total_with_fp8_side"]["bytes_per_page"]
        - current["swa_bf16"]["bytes_per_page"]
    ) * NUM_PAGES
    rows: list[dict[str, Any]] = []
    for name, low, high, assumption in specs:
        source = by_name.get(name, {})
        for label, tail_pages in (("low", low), ("high", high)):
            bf16_bytes = non_swa_current + tail_pages * current["swa_bf16"]["bytes_per_page"]
            fp8_value = tail_pages * (
                current["swa_bf16"]["bytes_per_page"]
                - fp8_components(current)["swa_fp8_page"]
            )
            rows.append(
                {
                    "scenario": name,
                    "bound": label,
                    "retained_pages_current": source.get("retained_prefix_pages", "estimated"),
                    "tail_swa_pages": tail_pages,
                    "tail_swa_tokens": tail_pages * PAGE_SIZE,
                    "bf16_lifecycle_bytes": bf16_bytes,
                    "swa_only_fp8_value_bytes": fp8_value,
                    "assumption": assumption,
                    "evidence": "estimated",
                }
            )
    return rows


def roi_rows(
    current: dict[str, dict[str, Any]],
    fp8: dict[str, Any],
    canonical_tail_pages: int = 16,
) -> list[dict[str, Any]]:
    current_page = current["total_with_fp8_side"]["bytes_per_page"]
    current_total = current_page * NUM_PAGES
    state = current["state_total"]["bytes_per_page"]
    non_swa_current_page = current_page - current["swa_bf16"]["bytes_per_page"]
    non_swa_broader_fp8_page = (
        fp8["c4_fp8_page"]
        + fp8["c128_fp8_page"]
        + fp8["indexer_fp8_replacement_page"]
        + state
    )
    candidates = [
        {
            "name": "current mini BF16 + additive indexer FP8 side",
            "bytes": current_total,
            "latency": "baseline",
            "risk": "low; runtime-proven promoted path",
            "scope": "none",
        },
        {
            "name": "current mini + SWA-only FP8",
            "bytes": fp8["swa_only_with_existing_side_page"] * NUM_PAGES,
            "latency": "+0.016 ms/boundary if separated; needs fusion",
            "risk": "medium quality/latency; correctness slice passed",
            "scope": "replace SWA cache only; keep C4/C128/indexer/state",
        },
        {
            "name": "current mini + full source-aligned MLA/indexer FP8",
            "bytes": fp8["full_source_aligned_replacement_page"] * NUM_PAGES,
            "latency": "unknown; likely worse until fused/integrated",
            "risk": "high; C4/C128/indexer/prefix ownership not integrated",
            "scope": "SWA+C4+C128 MLA FP8 and indexer replacement",
        },
        {
            "name": "SGLang lifecycle + BF16",
            "bytes": non_swa_current_page * NUM_PAGES
            + canonical_tail_pages * current["swa_bf16"]["bytes_per_page"],
            "latency": "near baseline or slight metadata cost",
            "risk": "medium correctness; lifecycle not runtime-proven in mini",
            "scope": f"independent SWA pool, {canonical_tail_pages} tail pages, BF16 cache dtype",
        },
        {
            "name": "SGLang lifecycle + SWA-only FP8",
            "bytes": non_swa_current_page * NUM_PAGES
            + canonical_tail_pages * fp8["swa_fp8_page"],
            "latency": "estimated +0.006-0.012 ms/boundary if fused store + selected gather",
            "risk": "medium-high; combines lifecycle and FP8",
            "scope": f"lifecycle plus FP8 SWA tail pool ({canonical_tail_pages} pages)",
        },
        {
            "name": "SGLang lifecycle + broader MLA/indexer FP8",
            "bytes": non_swa_broader_fp8_page * NUM_PAGES
            + canonical_tail_pages * fp8["swa_fp8_page"],
            "latency": "unknown; attention-integrated dequant may be needed",
            "risk": "highest; broad source layout plus ownership rewrite",
            "scope": "lifecycle plus SWA/C4/C128/indexer FP8 replacement",
        },
    ]
    rows = []
    for row in candidates:
        saved = current_total - row["bytes"]
        rows.append(
            {
                **row,
                "persistent_gib": row["bytes"] / GIB,
                "headroom_delta_gib": saved / GIB,
                "current_page_equiv": saved / current_page,
                "current_token_equiv": saved / current_page * PAGE_SIZE,
                "evidence": (
                    "runtime-proven"
                    if row["name"].startswith("current mini BF16")
                    else "estimated/source-derived"
                ),
            }
        )
    return rows


def render_current_md(
    current: dict[str, dict[str, Any]], preflight: dict[str, Any]
) -> str:
    rows = []
    for name in [
        "swa_bf16",
        "c4_bf16",
        "c128_bf16",
        "c4_indexer_bf16",
        "c4_indexer_fp8_side_cache",
        "c4_state_bf16",
        "c4_indexer_state_bf16",
        "c128_state_bf16",
    ]:
        item = current[name]
        rows.append(
            [
                name,
                f"{item['bytes_per_page']:,}",
                mib(item["bytes_per_page"]),
                gib(item["bytes_per_page"] * NUM_PAGES),
                item["evidence"],
            ]
        )
    total = current["total_with_fp8_side"]
    rows.append(
        [
            "total_with_existing_fp8_side",
            f"{total['bytes_per_page']:,}",
            mib(total["bytes_per_page"]),
            gib(total["bytes_at_128_pages"]),
            total["evidence"],
        ]
    )
    no_side = current["total_without_fp8_side"]
    rows.append(
        [
            "total_without_fp8_side_reference",
            f"{no_side['bytes_per_page']:,}",
            mib(no_side["bytes_per_page"]),
            gib(no_side["bytes_at_128_pages"]),
            no_side["evidence"],
        ]
    )
    runtime = preflight["kv_cache_bytes_per_rank"]
    runtime_pages = runtime / total["bytes_per_page"]
    text = md_table(
        ["component", "bytes/page", "MiB/page", "GiB at 128 pages", "evidence"],
        rows,
    )
    text += (
        "\n\nRuntime report from TARGET 09.0: "
        f"{runtime:,} B / {gib(runtime)} GiB per rank. "
        f"That equals {runtime_pages:.1f} pages at the promoted component formula, "
        "so the README uses the user-requested 128-page formula for ROI and keeps "
        "the 129-page runtime pool as separate runtime-proven context."
    )
    return text


def render_lifecycle_md(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["scenario"],
                row["bound"],
                row["retained_pages_current"],
                row["tail_swa_pages"],
                row["tail_swa_tokens"],
                gib(row["bf16_lifecycle_bytes"]),
                gib(row["swa_only_fp8_value_bytes"]),
                row["assumption"],
            ]
        )
    return md_table(
        [
            "scenario",
            "bound",
            "current retained pages",
            "tail SWA pages",
            "tail SWA tokens",
            "BF16 lifecycle GiB",
            "remaining SWA-only FP8 value GiB",
            "assumption",
        ],
        table_rows,
    )


def render_speed_md(speed: dict[str, Any]) -> str:
    bench_rows = []
    for row in speed["microbench"]:
        bench_rows.append(
            [
                row["bucket_bs"],
                row["selected_rows"],
                f"{row['bf16_combined_store_gather_ms']:.6f}",
                f"{row['fp8_combined_store_gather_dequant_ms']:.6f}",
                f"{row['fp8_combined_store_gather_dequant_ms'] - row['bf16_combined_store_gather_ms']:+.6f}",
                mib(row["workspace_bytes"]),
            ]
        )
    design_rows = [
        [
            "BF16 baseline",
            "0",
            "runtime-proven promoted path",
            "none",
        ],
        [
            "separated FP8 store + selected gather/dequant",
            f"+{speed['delta_min_ms']:.3f} to +{speed['delta_max_ms']:.3f} ms/boundary; worst {speed['worst_43_layer_separated_delta_ms']:.2f} ms if paid by all layers",
            "runtime-proven slice",
            "too slow as production shape",
        ],
        [
            "SGLang-aligned fused store + selected-row gather/dequant",
            "+0.006 to +0.012 ms/boundary estimate",
            "estimated from removing store launch and keeping selected-row dequant",
            "acceptable only as capacity opt-in until macro-proven",
        ],
        [
            "attention-integrated dequant",
            "0 to +0.006 ms/boundary estimate",
            "source-derived plausible, not mini-proven",
            "highest kernel and correctness risk",
        ],
    ]
    return (
        md_table(
            [
                "bs",
                "selected rows",
                "BF16 combined ms",
                "FP8 separated combined ms",
                "delta ms",
                "graph workspace",
            ],
            bench_rows,
        )
        + "\n\n"
        + md_table(
            ["design", "expected latency delta", "evidence", "risk note"],
            design_rows,
        )
    )


def render_roi_md(rows: list[dict[str, Any]]) -> str:
    return md_table(
        [
            "row",
            "persistent GiB/rank",
            "graph headroom delta",
            "equiv current pages/tokens",
            "expected latency delta",
            "quality/correctness risk",
            "implementation scope",
        ],
        [
            [
                row["name"],
                f"{row['persistent_gib']:.3f}",
                f"{row['headroom_delta_gib']:+.3f} GiB",
                f"{row['current_page_equiv']:.1f} pages / {row['current_token_equiv']:.0f} tokens",
                row["latency"],
                row["risk"],
                row["scope"],
            ]
            for row in rows
        ],
    )


def main() -> None:
    preflight = load_json(PREFLIGHT_LEDGER)
    fp8_slice = load_json(FP8_SLICE_SUMMARY)
    current = current_components()
    fp8 = fp8_components(current)
    speed = load_speed(fp8_slice)
    lifecycle = lifecycle_scenarios(current)
    roi = roi_rows(current, fp8)
    data = {
        "constants": {
            "num_layers": NUM_LAYERS,
            "c4_layers": C4_LAYERS,
            "c128_layers": C128_LAYERS,
            "page_size": PAGE_SIZE,
            "num_pages": NUM_PAGES,
            "sliding_window": SLIDING_WINDOW,
        },
        "current": current,
        "fp8": fp8,
        "speed": speed,
        "lifecycle": lifecycle,
        "roi": roi,
    }
    write_json(SUMMARY_DIR / "ledger.json", data)
    write_text(SUMMARY_DIR / "current_mini_memory_ledger.md", render_current_md(current, preflight))
    write_text(SUMMARY_DIR / "sglang_lifecycle_ledger.md", render_lifecycle_md(lifecycle))
    write_text(SUMMARY_DIR / "speed_ledger.md", render_speed_md(speed))
    write_text(SUMMARY_DIR / "roi_matrix.md", render_roi_md(roi))


if __name__ == "__main__":
    main()

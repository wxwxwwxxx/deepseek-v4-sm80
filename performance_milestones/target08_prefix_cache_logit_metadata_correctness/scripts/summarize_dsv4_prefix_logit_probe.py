#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_run(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "run.json")


def _load_batches(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "debug_trace" / "batches.rank0.jsonl"
    batches = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                batches.append(json.loads(line))
    return batches


def _scenario_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in run.get("scenarios", [])}


def _find_batch(
    batches: list[dict[str, Any]],
    *,
    scenario: str,
    stage: str,
    phase: str,
) -> dict[str, Any] | None:
    for batch in batches:
        if (
            batch.get("scenario") == scenario
            and batch.get("stage") == stage
            and batch.get("phase") == phase
        ):
            return batch
    return None


def _load_metadata(run_dir: Path, batch: dict[str, Any] | None) -> dict[str, torch.Tensor]:
    if batch is None:
        return {}
    return torch.load(run_dir / "debug_trace" / batch["metadata_path"], map_location="cpu")


def _load_logits(run_dir: Path, batch: dict[str, Any] | None) -> torch.Tensor | None:
    if batch is None or not batch.get("logits_path"):
        return None
    payload = torch.load(run_dir / "debug_trace" / batch["logits_path"], map_location="cpu")
    return payload["logits"].float()


def _int_list(values: Any) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.reshape(-1).tolist()]
    return [int(x) for x in values]


def _req_cached_lens(batch: dict[str, Any] | None) -> list[int]:
    if batch is None:
        return []
    return [int(req["cached_len"]) for req in batch.get("reqs", [])]


def _row_tail_select(
    tensor: torch.Tensor,
    source_extend_lens: Sequence[int],
    target_extend_lens: Sequence[int],
) -> torch.Tensor:
    pieces = []
    offset = 0
    for source_len, target_len in zip(source_extend_lens, target_extend_lens, strict=False):
        source_len = int(source_len)
        target_len = int(target_len)
        start = offset + max(source_len - target_len, 0)
        end = offset + source_len
        pieces.append(tensor[start:end])
        offset += source_len
    if not pieces:
        return tensor[:0]
    return torch.cat(pieces, dim=0)


def _values_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a, b))


def _rows_equal_ignore_trailing_negative(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        return False
    for row in range(a.shape[0]):
        aa = a[row][a[row] >= 0].tolist()
        bb = b[row][b[row] >= 0].tolist()
        if aa != bb:
            return False
    return True


def _valid_counts_match(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.ndim < 2 or b.ndim < 2 or a.shape[0] != b.shape[0]:
        return False
    return bool(torch.equal((a >= 0).sum(dim=-1), (b >= 0).sum(dim=-1)))


def _split_by_lens(tensor: torch.Tensor, lens: Sequence[int]) -> list[torch.Tensor]:
    out = []
    offset = 0
    for length in lens:
        length = int(length)
        out.append(tensor[offset : offset + length])
        offset += length
    return out


def _metadata_comparison(
    *,
    scenario: dict[str, Any],
    off_batch: dict[str, Any] | None,
    on_batch: dict[str, Any] | None,
    on_warm_batch: dict[str, Any] | None,
    off_meta: dict[str, torch.Tensor],
    on_meta: dict[str, torch.Tensor],
    on_warm_meta: dict[str, torch.Tensor],
) -> dict[str, Any]:
    expected_cached = [int(x) for x in scenario.get("expected_cached_lens", [])]
    actual_cached = _req_cached_lens(on_batch)
    cached_len_match = actual_cached == expected_cached
    off_cached_zero = all(value == 0 for value in _req_cached_lens(off_batch))

    off_extend = _int_list(off_meta.get("core.extend_lens", torch.empty(0, dtype=torch.int32)))
    on_extend = _int_list(on_meta.get("core.extend_lens", torch.empty(0, dtype=torch.int32)))
    suffix_range_match = True
    suffix_ranges = []
    if on_batch is not None and "core.positions" in on_meta:
        positions_by_req = _split_by_lens(on_meta["core.positions"], on_extend)
        for req, positions in zip(on_batch.get("reqs", []), positions_by_req, strict=False):
            cached_len = int(req["cached_len"])
            device_len = int(req["device_len"])
            values = _int_list(positions)
            expected = list(range(cached_len, device_len))
            ok = values == expected
            suffix_range_match &= ok
            suffix_ranges.append(
                {
                    "uid": int(req["uid"]),
                    "expected": [cached_len, device_len],
                    "actual_first_last": (
                        [values[0], values[-1] + 1] if values else [None, None]
                    ),
                    "ok": ok,
                }
            )

    exact_row_fields = [
        "core.positions",
        "core.seq_lens",
        "core.swa_topk_lengths",
        "core.c4_topk_lengths_raw",
        "core.c4_topk_lengths_clamp1",
        "core.c4_sparse_topk_lengths",
        "core.c128_topk_lengths_clamp1",
        "indexer.c4_seq_lens",
    ]
    exact_checks = {}
    for field in exact_row_fields:
        if field not in off_meta or field not in on_meta:
            continue
        off_tail = _row_tail_select(off_meta[field], off_extend, on_extend)
        exact_checks[field] = _values_equal(off_tail, on_meta[field])

    raw_index_fields = ["core.c4_sparse_raw_indices", "core.c128_raw_indices"]
    raw_checks = {}
    for field in raw_index_fields:
        if field not in off_meta or field not in on_meta:
            continue
        off_tail = _row_tail_select(off_meta[field], off_extend, on_extend)
        raw_checks[field] = _rows_equal_ignore_trailing_negative(off_tail, on_meta[field])

    physical_count_fields = [
        "core.page_table",
        "core.swa_page_indices",
        "core.c4_sparse_page_indices",
        "core.c4_sparse_full_indices",
        "core.c128_page_indices",
        "core.c128_full_indices",
    ]
    physical_count_checks = {}
    for field in physical_count_fields:
        if field not in off_meta or field not in on_meta:
            continue
        off_tail = _row_tail_select(off_meta[field], off_extend, on_extend)
        physical_count_checks[field] = _valid_counts_match(off_tail, on_meta[field])

    c4_expected = int(((on_meta.get("core.positions", torch.empty(0)) + 1) % 4 == 0).sum().item())
    c128_expected = int(
        ((on_meta.get("core.positions", torch.empty(0)) + 1) % 128 == 0).sum().item()
    )
    c4_out_loc_count = int(on_meta.get("core.c4_out_loc", torch.empty(0)).numel())
    c128_out_loc_count = int(on_meta.get("core.c128_out_loc", torch.empty(0)).numel())
    output_loc_checks = {
        "c4_out_loc_count": c4_out_loc_count == c4_expected,
        "c128_out_loc_count": c128_out_loc_count == c128_expected,
        "c4_out_loc_count_actual": c4_out_loc_count,
        "c4_out_loc_count_expected": c4_expected,
        "c128_out_loc_count_actual": c128_out_loc_count,
        "c128_out_loc_count_expected": c128_expected,
    }

    page_prefix_reused = True
    page_prefix_reused_detail = []
    if on_warm_batch is not None and "batch.global_page_table_rows" in on_warm_meta:
        warm_row = on_warm_meta["batch.global_page_table_rows"][0]
        probe_rows = on_meta.get("batch.global_page_table_rows")
        if probe_rows is not None:
            for row, cached_len in enumerate(actual_cached):
                if cached_len <= 0:
                    continue
                reused = bool(torch.equal(probe_rows[row, :cached_len], warm_row[:cached_len]))
                page_prefix_reused &= reused
                page_prefix_reused_detail.append(
                    {"row": row, "cached_len": cached_len, "reused_warm_prefix_pages": reused}
                )

    semantic_lengths_match = (
        suffix_range_match
        and all(exact_checks.values())
        and all(raw_checks.values())
        and output_loc_checks["c4_out_loc_count"]
        and output_loc_checks["c128_out_loc_count"]
    )
    physical_valid_counts_match = all(physical_count_checks.values())

    earliest = "none"
    if not off_cached_zero:
        earliest = "prefix-disabled cached_len"
    elif not cached_len_match:
        earliest = "cached_len"
    elif not suffix_range_match:
        earliest = "suffix prefill token range"
    elif not semantic_lengths_match:
        earliest = "SWA/C4/C128/indexer semantic metadata"
    elif not physical_valid_counts_match:
        earliest = "page/full physical-index valid counts"
    elif not page_prefix_reused:
        earliest = "page table prefix reuse"

    return {
        "scenario": scenario["name"],
        "expected_cached_lens": expected_cached,
        "actual_cached_lens": actual_cached,
        "off_cached_zero": off_cached_zero,
        "cached_len_match": cached_len_match,
        "suffix_range_match": suffix_range_match,
        "suffix_ranges": suffix_ranges,
        "exact_row_checks": exact_checks,
        "raw_index_checks": raw_checks,
        "physical_valid_count_checks": physical_count_checks,
        "output_loc_checks": output_loc_checks,
        "page_prefix_reused": page_prefix_reused,
        "page_prefix_reused_detail": page_prefix_reused_detail,
        "semantic_lengths_match": semantic_lengths_match,
        "physical_valid_counts_match": physical_valid_counts_match,
        "earliest_metadata_mismatch": earliest,
    }


def _logits_stats(a: torch.Tensor | None, b: torch.Tensor | None, *, atol: float, rtol: float) -> dict[str, Any]:
    if a is None or b is None:
        return {"available": False, "allclose": False, "reason": "missing logits"}
    if a.shape != b.shape:
        return {
            "available": True,
            "allclose": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "reason": "shape mismatch",
        }
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    row_max = [float(x) for x in diff.max(dim=-1).values.tolist()] if diff.ndim == 2 else []
    allclose = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    topk = min(10, a.shape[-1])
    a_top = torch.topk(a, k=topk, dim=-1).indices
    b_top = torch.topk(b, k=topk, dim=-1).indices
    topk_exact = bool(torch.equal(a_top, b_top))
    argmax_equal = bool(torch.equal(torch.argmax(a, dim=-1), torch.argmax(b, dim=-1)))
    return {
        "available": True,
        "allclose": allclose,
        "atol": atol,
        "rtol": rtol,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "row_max_abs": row_max,
        "top10_ids_equal": topk_exact,
        "argmax_equal": argmax_equal,
        "shape": list(a.shape),
    }


def _outputs(run: dict[str, Any], scenario_name: str) -> list[list[int]]:
    scenario = _scenario_map(run)[scenario_name]
    return [list(item.get("token_ids", [])) for item in scenario.get("probe_outputs", [])]


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def summarize(args: argparse.Namespace) -> int:
    off_dir = Path(args.prefix_off)
    on_dir = Path(args.prefix_on)
    eager_dir = Path(args.prefix_on_eager) if args.prefix_on_eager else None
    out_dir = Path(args.output_dir)

    off_run = _load_run(off_dir)
    on_run = _load_run(on_dir)
    eager_run = _load_run(eager_dir) if eager_dir else None
    off_batches = _load_batches(off_dir)
    on_batches = _load_batches(on_dir)
    eager_batches = _load_batches(eager_dir) if eager_dir else []

    scenarios = list(_scenario_map(on_run).values())
    metadata_rows = []
    logits_rows = []
    generated_rows = []
    scenario_rows = []
    earliest_rows = []
    summary: dict[str, Any] = {
        "prefix_off": str(off_dir),
        "prefix_on": str(on_dir),
        "prefix_on_eager": str(eager_dir) if eager_dir else None,
        "metadata": [],
        "logits": [],
        "generated_tokens": [],
        "earliest_mismatch": [],
    }

    for scenario in scenarios:
        name = scenario["name"]
        off_prefill = _find_batch(off_batches, scenario=name, stage="probe", phase="prefill")
        on_prefill = _find_batch(on_batches, scenario=name, stage="probe", phase="prefill")
        on_warm_prefill = _find_batch(on_batches, scenario=name, stage="warm", phase="prefill")
        off_decode = _find_batch(off_batches, scenario=name, stage="probe", phase="decode")
        on_decode = _find_batch(on_batches, scenario=name, stage="probe", phase="decode")
        eager_prefill = (
            _find_batch(eager_batches, scenario=name, stage="probe", phase="prefill")
            if eager_dir
            else None
        )
        eager_decode = (
            _find_batch(eager_batches, scenario=name, stage="probe", phase="decode")
            if eager_dir
            else None
        )

        off_prefill_meta = _load_metadata(off_dir, off_prefill)
        on_prefill_meta = _load_metadata(on_dir, on_prefill)
        on_warm_meta = _load_metadata(on_dir, on_warm_prefill)
        metadata = _metadata_comparison(
            scenario=scenario,
            off_batch=off_prefill,
            on_batch=on_prefill,
            on_warm_batch=on_warm_prefill,
            off_meta=off_prefill_meta,
            on_meta=on_prefill_meta,
            on_warm_meta=on_warm_meta,
        )

        prefill_stats = _logits_stats(
            _load_logits(off_dir, off_prefill),
            _load_logits(on_dir, on_prefill),
            atol=args.atol,
            rtol=args.rtol,
        )
        decode_stats = _logits_stats(
            _load_logits(off_dir, off_decode),
            _load_logits(on_dir, on_decode),
            atol=args.atol,
            rtol=args.rtol,
        )
        graph_prefill_stats = (
            _logits_stats(
                _load_logits(on_dir, on_prefill),
                _load_logits(eager_dir, eager_prefill),
                atol=args.atol,
                rtol=args.rtol,
            )
            if eager_dir
            else {"available": False}
        )
        graph_decode_stats = (
            _logits_stats(
                _load_logits(on_dir, on_decode),
                _load_logits(eager_dir, eager_decode),
                atol=args.atol,
                rtol=args.rtol,
            )
            if eager_dir
            else {"available": False}
        )

        off_tokens = _outputs(off_run, name)
        on_tokens = _outputs(on_run, name)
        eager_tokens = _outputs(eager_run, name) if eager_run else []
        generated = {
            "scenario": name,
            "prefix_off": off_tokens,
            "prefix_on": on_tokens,
            "prefix_on_eager": eager_tokens,
            "off_on_match": off_tokens == on_tokens,
            "on_graph_eager_match": (not eager_run) or on_tokens == eager_tokens,
        }

        earliest = "none"
        if metadata["earliest_metadata_mismatch"] != "none":
            earliest = "metadata: " + metadata["earliest_metadata_mismatch"]
        elif not prefill_stats.get("allclose", False):
            earliest = "suffix prefill logits"
        elif not decode_stats.get("allclose", False):
            earliest = "decode logits"
        elif not generated["off_on_match"]:
            earliest = "sampled token"
        elif graph_decode_stats.get("available") and not graph_decode_stats.get("allclose", False):
            earliest = "graph replay decode logits"

        logit_summary = {
            "scenario": name,
            "prefill_off_on": prefill_stats,
            "decode_off_on": decode_stats,
            "prefill_on_graph_vs_eager": graph_prefill_stats,
            "decode_on_graph_vs_eager": graph_decode_stats,
        }
        summary["metadata"].append(metadata)
        summary["logits"].append(logit_summary)
        summary["generated_tokens"].append(generated)
        summary["earliest_mismatch"].append({"scenario": name, "earliest": earliest})

        scenario_rows.append(
            [
                name,
                ", ".join(scenario.get("coverage", [])),
                scenario.get("warm_prompt_lens", []),
                scenario.get("probe_prompt_lens", []),
                scenario.get("expected_cached_lens", []),
                on_run["config"].get("allow_dsv4_cuda_graph"),
            ]
        )
        metadata_rows.append(
            [
                name,
                metadata["expected_cached_lens"],
                metadata["actual_cached_lens"],
                _fmt_bool(metadata["cached_len_match"]),
                _fmt_bool(metadata["suffix_range_match"]),
                _fmt_bool(metadata["semantic_lengths_match"]),
                _fmt_bool(metadata["physical_valid_counts_match"]),
                _fmt_bool(metadata["page_prefix_reused"]),
                metadata["earliest_metadata_mismatch"],
            ]
        )
        logits_rows.append(
            [
                name,
                _fmt_stat(prefill_stats),
                _fmt_stat(decode_stats),
                _fmt_stat(graph_decode_stats),
                _fmt_bool(bool(prefill_stats.get("top10_ids_equal", False))),
                _fmt_bool(bool(decode_stats.get("top10_ids_equal", False))),
            ]
        )
        generated_rows.append(
            [
                name,
                _fmt_bool(generated["off_on_match"]),
                _fmt_bool(generated["on_graph_eager_match"]),
                _short_tokens(off_tokens),
                _short_tokens(on_tokens),
                _short_tokens(eager_tokens),
            ]
        )
        earliest_rows.append([name, earliest])

    continue_0820 = all(item["earliest"] == "none" for item in summary["earliest_mismatch"])
    summary["decision"] = {
        "allow_continue_target_08_20": continue_0820,
        "default_promotion_allowed": False,
        "reason": (
            "logits and metadata matched for all tested boundaries"
            if continue_0820
            else "one or more deterministic metadata/logit comparisons mismatched"
        ),
    }

    _write_json(out_dir / "comparison_summary.json", summary)
    _write_text(
        out_dir / "scenario_table.md",
        _markdown_table(
            ["Scenario", "Coverage", "Warm lens", "Probe lens", "Expected cached_len", "Graph"],
            scenario_rows,
        ),
    )
    _write_text(
        out_dir / "metadata_comparison.md",
        _markdown_table(
            [
                "Scenario",
                "Expected cached",
                "Actual cached",
                "cached_len",
                "suffix range",
                "semantic metadata",
                "physical counts",
                "prefix page reuse",
                "earliest metadata mismatch",
            ],
            metadata_rows,
        ),
    )
    _write_text(
        out_dir / "logits_comparison.md",
        _markdown_table(
            [
                "Scenario",
                "suffix prefill off/on",
                "decode off/on",
                "decode graph/eager",
                "prefill top10",
                "decode top10",
            ],
            logits_rows,
        ),
    )
    _write_text(
        out_dir / "generated_tokens.md",
        _markdown_table(
            [
                "Scenario",
                "off/on match",
                "on graph/eager match",
                "off tokens",
                "on tokens",
                "on eager tokens",
            ],
            generated_rows,
        ),
    )
    _write_text(
        out_dir / "earliest_mismatch.md",
        _markdown_table(["Scenario", "Earliest mismatch"], earliest_rows),
    )
    print(json.dumps({"summary": str(out_dir / "comparison_summary.json"), **summary["decision"]}))
    return 0 if continue_0820 else 1


def _fmt_stat(stats: dict[str, Any]) -> str:
    if not stats.get("available"):
        return "missing"
    if not stats.get("allclose"):
        if stats.get("reason"):
            return f"FAIL {stats['reason']}"
        return f"FAIL max={stats.get('max_abs', math.nan):.6g}"
    return f"pass max={stats.get('max_abs', 0.0):.6g}"


def _short_tokens(tokens: list[list[int]], limit: int = 2) -> str:
    shown = [row[:4] for row in tokens[:limit]]
    suffix = "" if len(tokens) <= limit else f" +{len(tokens) - limit} rows"
    return f"{shown}{suffix}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TARGET 08.19 DSV4 prefix logits probe.")
    parser.add_argument("--prefix-off", required=True)
    parser.add_argument("--prefix-on", required=True)
    parser.add_argument("--prefix-on-eager", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(summarize(parse_args(argv)))


if __name__ == "__main__":
    main()

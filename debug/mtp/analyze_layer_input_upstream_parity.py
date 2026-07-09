from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rank_path(path: str, rank: int) -> Path:
    candidate = Path(path)
    if candidate.exists() and f".rank{rank}." in candidate.name:
        return candidate
    ranked = candidate.with_suffix(candidate.suffix + f".rank{rank}.json")
    return ranked if ranked.exists() else candidate


def _read(path: str, rank: int = 0) -> dict[str, Any]:
    return json.loads(_rank_path(path, rank).read_text(encoding="utf-8"))


def _write(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_by_bs(payload: dict[str, Any], batch_size: int) -> dict[str, Any]:
    for run in payload.get("runs", []) or []:
        if int(run.get("batch_size", -1)) == int(batch_size):
            return run
    raise KeyError(f"batch_size {batch_size} not found")


def _stats(run: dict[str, Any]) -> dict[str, Any]:
    return dict(run.get("stats_delta") or run.get("stats_after") or {})


def _token_matrix(base: dict[str, Any] | None, mtp: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mtp_run in mtp.get("runs", []) or []:
        bs = int(mtp_run.get("batch_size", -1))
        mtp_tokens = mtp_run.get("token_ids", [])
        base_tokens = []
        exact = None
        diffs: list[dict[str, Any]] = []
        if base is not None:
            base_run = _run_by_bs(base, bs)
            base_tokens = base_run.get("token_ids", [])
            exact = base_tokens == mtp_tokens
            diffs = [
                {
                    "request_index": idx,
                    "baseline": base_row,
                    "mtp": mtp_row,
                }
                for idx, (base_row, mtp_row) in enumerate(zip(base_tokens, mtp_tokens))
                if base_row != mtp_row
            ]
        rows.append(
            {
                "batch_size": bs,
                "exact": exact,
                "diff_count": len(diffs),
                "diffs": diffs,
                "baseline": base_tokens,
                "mtp": mtp_tokens,
            }
        )
    return rows


def _commit_stats(mtp: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for run in mtp.get("runs", []) or []:
        stats = _stats(run)
        out.append(
            {
                "batch_size": int(run.get("batch_size", -1)),
                "target_verify_calls": int(stats.get("target_verify_calls", 0)),
                "target_commit_kv_copies": int(stats.get("target_commit_kv_copies", 0)),
                "accepted_kv_copied_tokens": int(
                    stats.get("accepted_kv_copied_tokens", 0)
                ),
                "draft_tokens_accepted": int(stats.get("draft_tokens_accepted", 0)),
                "target_correction_tokens": int(stats.get("target_correction_tokens", 0)),
                "target_bonus_tokens": int(stats.get("target_bonus_tokens", 0)),
                "target_verify_moe_original_calls": int(
                    stats.get("target_verify_moe_original_calls", 0)
                ),
                "target_verify_moe_microbatch_calls": int(
                    stats.get("target_verify_moe_microbatch_calls", 0)
                ),
            }
        )
    return out


def _debug_env(payload: dict[str, Any]) -> dict[str, Any]:
    env = payload.get("env", {})
    return dict(env.get("debug_env") or {})


def _matches(value: Any, expected: int | None) -> bool:
    if expected is None:
        return True
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        return False


def _find_anchor_row(
    stats: dict[str, Any],
    *,
    uid: int | None,
    position: int | None,
    full_loc: int | None,
    depth: int | None,
) -> dict[str, Any] | None:
    fallback: dict[str, Any] | None = None
    for event in stats.get("row_depth_oracle_debug", []) or []:
        event_index = int(event.get("trace_index", -1))
        for entry in event.get("entries", []) or []:
            if not _matches(entry.get("uid"), uid):
                continue
            for row in entry.get("rows", []) or []:
                metadata = row.get("target_metadata") or {}
                row_depth = row.get("depth")
                row_position = row.get("position", metadata.get("position"))
                row_full_loc = metadata.get("out_cache_loc")
                candidate = {
                    "event_index": event_index,
                    "event": event,
                    "entry": entry,
                    "row": row,
                }
                if fallback is None:
                    parity = row.get("producer_trace_parity") or {}
                    if parity.get("first_mismatch"):
                        fallback = candidate
                if (
                    _matches(row_depth, depth)
                    and _matches(row_position, position)
                    and _matches(row_full_loc, full_loc)
                ):
                    return candidate
    return fallback


def _contract_event(stats: dict[str, Any], event_index: int) -> dict[str, Any]:
    events = list(stats.get("target_verify_contract_trace") or [])
    if 0 <= event_index < len(events):
        return dict(events[event_index])
    return {}


def _debug_entry_for_row(
    stats: dict[str, Any],
    *,
    uid: int | None,
    row_start: int | None,
    depth: int | None,
    position: int | None,
) -> dict[str, Any]:
    for item in stats.get("debug_trace", []) or []:
        if uid is not None and not _matches(item.get("uid"), uid):
            continue
        if row_start is not None and not _matches(item.get("row_start"), row_start):
            continue
        if depth is None:
            return dict(item)
        for row in item.get("row_depths", []) or []:
            if not _matches(row.get("depth"), depth):
                continue
            if position is not None:
                item_position = int(item.get("cached_len", -1)) + int(depth)
                if item_position != int(position):
                    continue
            out = dict(item)
            out["matched_row_depth"] = row
            return out
    return {}


def _sha(comparison: dict[str, Any], label: str) -> Any:
    return comparison.get(f"{label}_raw_sha256")


def _compact_comparison(comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        return {"available": False}
    return {
        "available": True,
        "name": comparison.get("name"),
        "layer_id": comparison.get("layer_id"),
        "boundary": comparison.get("boundary"),
        "exact": not bool(comparison.get("is_mismatch", False)),
        "bit_exact": comparison.get("bit_exact_result"),
        "allclose": comparison.get("allclose_result"),
        "normal_sha": _sha(comparison, "normal"),
        "target_verify_sha": _sha(comparison, "target_verify"),
        "max_delta": comparison.get("max_delta"),
        "mean_delta": comparison.get("mean_delta"),
        "first_differing_index": comparison.get("first_differing_index"),
        "normal_sample": comparison.get("normal_sample"),
        "target_verify_sample": comparison.get("target_verify_sample"),
    }


def _comparison_by_name(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        name = comparison.get("name")
        if isinstance(name, str) and name not in by_name:
            by_name[name] = comparison
    return by_name


def _coarse_layer_table(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = _comparison_by_name(comparisons)
    names = ["embedding"]
    for layer_id in range(0, 11):
        names.append(f"layer{layer_id}.input")
        names.append(f"layer{layer_id}.post_moe_residual")
    names.append("hidden_before_final_norm")
    rows = []
    for name in names:
        record = _compact_comparison(by_name.get(name))
        record["boundary"] = name
        rows.append(record)
    return rows


def _first_mismatch_layer(first_mismatch: dict[str, Any] | None) -> int | None:
    if not isinstance(first_mismatch, dict):
        return None
    name = str(first_mismatch.get("name", ""))
    if not name.startswith("layer"):
        return None
    digits = []
    for char in name[len("layer") :]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def _layer_subboundary_table(
    comparisons: list[dict[str, Any]],
    layer_id: int | None,
) -> list[dict[str, Any]]:
    if layer_id is None:
        return []
    prefix = f"layer{layer_id}."
    return [
        _compact_comparison(comparison)
        for comparison in comparisons
        if str(comparison.get("name", "")).startswith(prefix)
    ]


def _row_category(debug_entry: dict[str, Any], depth: int | None) -> str | None:
    if depth is None:
        return None
    accepted_prefix = debug_entry.get("accepted_prefix")
    mismatch_depth = debug_entry.get("mismatch_depth")
    active_verify_len = debug_entry.get("active_verify_len")
    draft_tokens = debug_entry.get("draft_tokens") or []
    try:
        depth_i = int(depth)
        active_i = int(active_verify_len)
        accepted_i = int(accepted_prefix)
    except (TypeError, ValueError):
        return None
    if depth_i >= active_i:
        return "padded_inactive"
    if depth_i < accepted_i:
        return "accepted"
    if mismatch_depth is not None and int(mismatch_depth) == depth_i:
        return "correction"
    if mismatch_depth is None and depth_i == len(draft_tokens) and active_i > len(draft_tokens):
        return "bonus"
    return "rejected_tail"


def _row_identity_summary(
    *,
    row: dict[str, Any],
    entry: dict[str, Any],
    contract: dict[str, Any],
    debug_entry: dict[str, Any],
) -> dict[str, Any]:
    metadata = row.get("target_metadata") or {}
    oracle_metadata = row.get("oracle_metadata") or {}
    depth = row.get("depth")
    flat_row = row.get("flattened_row")
    row_to_batch = None
    row_to_parent = None
    active_row = None
    padded_row = None
    if isinstance(contract, dict) and flat_row is not None:
        idx = int(flat_row)
        for source, assign in (
            ("row_to_batch_index", "row_to_batch_index"),
            ("row_to_parent_batch_index", "row_to_parent_batch_index"),
            ("active_row_mask", "active_row"),
            ("padded_row_mask", "padded_row"),
        ):
            values = contract.get(source)
            if isinstance(values, list) and 0 <= idx < len(values):
                if assign == "row_to_batch_index":
                    row_to_batch = values[idx]
                elif assign == "row_to_parent_batch_index":
                    row_to_parent = values[idx]
                elif assign == "active_row":
                    active_row = values[idx]
                elif assign == "padded_row":
                    padded_row = values[idx]
    return {
        "uid": entry.get("uid"),
        "batch_index": entry.get("batch_index"),
        "input_token": row.get("input_token"),
        "draft_token": row.get("draft_token"),
        "target_token": row.get("target_token"),
        "oracle_token": row.get("oracle_token"),
        "accepted_flag": row.get("accepted"),
        "category": _row_category(debug_entry, int(depth) if depth is not None else None),
        "position": row.get("position", metadata.get("position")),
        "depth": depth,
        "flattened_row": flat_row,
        "full_loc": metadata.get("out_cache_loc"),
        "swa_loc": metadata.get("out_cache_loc"),
        "row_to_batch_index": (
            metadata.get("row_to_batch_index")
            if metadata.get("row_to_batch_index") is not None
            else row_to_batch
        ),
        "row_to_parent_batch_index": (
            metadata.get("row_to_parent_batch_index")
            if metadata.get("row_to_parent_batch_index") is not None
            else row_to_parent
        ),
        "parent_batch_size": metadata.get(
            "parent_batch_size",
            contract.get("parent_batch_size") if isinstance(contract, dict) else None,
        ),
        "active_row": (
            metadata.get("active_row")
            if metadata.get("active_row") is not None
            else active_row
        ),
        "padded_row": (
            metadata.get("padded_row")
            if metadata.get("padded_row") is not None
            else padded_row
        ),
        "request_table_slot": metadata.get("req_table_index"),
        "seq_len": metadata.get("seq_len"),
        "cached_len": debug_entry.get("cached_len"),
        "device_len": debug_entry.get("device_len"),
        "committed_seq_len": entry.get("committed_seq_len"),
        "active_verify_len": entry.get("active_verify_len"),
        "padded_verify_len": entry.get("padded_verify_len"),
        "accepted_prefix": debug_entry.get("accepted_prefix"),
        "mismatch_depth": debug_entry.get("mismatch_depth"),
        "copy_rows": debug_entry.get("copy_rows"),
        "candidate_copy_rows": debug_entry.get("candidate_copy_rows"),
        "contract_parent_batch_indices": contract.get("parent_batch_indices", []),
        "contract_active_row_mask": contract.get("active_row_mask", []),
        "contract_positions": contract.get("positions", []),
        "contract_out_cache_loc": contract.get("out_cache_loc", []),
        "target_metadata": metadata,
        "oracle_metadata": oracle_metadata,
    }


def _classification_from_first_layer(
    comparisons: list[dict[str, Any]],
    first_mismatch: dict[str, Any] | None,
    first_layer: int | None,
) -> str:
    if not comparisons:
        return "layer10_input_instrumentation_no_go"
    if first_mismatch is None:
        return "layer10_input_analyzer_owner"
    if first_layer == 0:
        return "layer10_input_layer0_owner"
    if first_layer is not None and 1 <= first_layer <= 9:
        return "layer10_input_layerN_owner"
    if first_layer == 10:
        return "layer10_input_hidden_publication_owner"
    return "layer10_input_instrumentation_no_go"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--mtp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--uid", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--full-loc", type=int)
    parser.add_argument("--depth", type=int)
    args = parser.parse_args()

    baseline = _read(args.baseline, args.rank) if args.baseline else None
    mtp = _read(args.mtp, args.rank)
    mtp_run = _run_by_bs(mtp, args.batch_size)
    mtp_stats = _stats(mtp_run)

    anchor = _find_anchor_row(
        mtp_stats,
        uid=args.uid,
        position=args.position,
        full_loc=args.full_loc,
        depth=args.depth,
    )
    if anchor is None:
        result = {
            "classification": "layer10_input_instrumentation_no_go",
            "reason": "anchor row not found in row_depth_oracle_debug",
            "exactness_matrix": _token_matrix(baseline, mtp),
            "accepted_commit_stats": _commit_stats(mtp),
            "env": {
                "baseline_debug_env": _debug_env(baseline) if baseline else {},
                "mtp_debug_env": _debug_env(mtp),
            },
        }
        _write(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    row = anchor["row"]
    entry = anchor["entry"]
    event = anchor["event"]
    event_index = int(anchor["event_index"])
    contract = _contract_event(mtp_stats, event_index)
    debug_entry = _debug_entry_for_row(
        mtp_stats,
        uid=entry.get("uid"),
        row_start=entry.get("row_start"),
        depth=row.get("depth"),
        position=row.get("position"),
    )
    parity = row.get("producer_trace_parity") or {}
    comparisons = list(parity.get("comparisons") or [])
    first_mismatch = parity.get("first_mismatch")
    first_layer = _first_mismatch_layer(first_mismatch)
    result = {
        "classification": _classification_from_first_layer(
            comparisons,
            first_mismatch,
            first_layer,
        ),
        "anchor_query": {
            "uid": args.uid,
            "position": args.position,
            "full_loc": args.full_loc,
            "depth": args.depth,
            "batch_size": args.batch_size,
            "rank": args.rank,
        },
        "row_identity": _row_identity_summary(
            row=row,
            entry=entry,
            contract=contract,
            debug_entry=debug_entry,
        ),
        "producer_trace_summary": {
            "num_lhs_entries": parity.get("num_lhs_entries"),
            "num_rhs_entries": parity.get("num_rhs_entries"),
            "num_compared": parity.get("num_compared"),
            "missing_from_rhs": parity.get("missing_from_rhs", []),
            "missing_from_lhs": parity.get("missing_from_lhs", []),
            "first_mismatch": _compact_comparison(first_mismatch),
            "first_mismatch_layer": first_layer,
        },
        "coarse_upstream_layer_bisection": _coarse_layer_table(comparisons),
        "first_mismatch_layer_subboundaries": _layer_subboundary_table(
            comparisons,
            first_layer,
        ),
        "hidden_deltas": {
            "hidden_delta": row.get("hidden_delta"),
            "hidden_before_norm_delta": row.get("hidden_before_norm_delta"),
            "logit_delta": row.get("logit_delta"),
        },
        "event_summary": {
            "row_depth_event_trace_index": event.get("trace_index"),
            "contract_event": {
                key: contract.get(key)
                for key in (
                    "batch_size",
                    "parent_batch_size",
                    "num_tokens",
                    "verify_lens",
                    "active_verify_lens",
                    "padded_verify_lens",
                    "committed_seq_lens",
                    "input_tokens",
                    "positions",
                    "out_cache_loc",
                    "active_row_mask",
                    "padded_row_mask",
                    "row_depths",
                    "row_to_batch_index",
                    "row_to_parent_batch_index",
                    "parent_batch_indices",
                    "seq_lens",
                    "req_seq_lens",
                )
            },
        },
        "exactness_matrix": _token_matrix(baseline, mtp),
        "accepted_commit_stats": _commit_stats(mtp),
        "env": {
            "baseline_debug_env": _debug_env(baseline) if baseline else {},
            "mtp_debug_env": _debug_env(mtp),
        },
    }
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

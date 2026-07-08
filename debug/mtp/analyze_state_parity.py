from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_by_bs(payload: dict[str, Any], batch_size: int) -> dict[str, Any]:
    for run in payload.get("runs", []):
        if int(run.get("batch_size", -1)) == int(batch_size):
            return run
    raise KeyError(f"batch_size {batch_size} not found")


def _stats(run: dict[str, Any]) -> dict[str, Any]:
    return dict(run.get("stats_delta") or run.get("stats_after") or {})


def _token_matrix(base: dict[str, Any], mtp: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in mtp.get("runs", []):
        bs = int(run.get("batch_size", -1))
        base_run = _run_by_bs(base, bs)
        base_tokens = base_run.get("token_ids", [])
        mtp_tokens = run.get("token_ids", [])
        rows.append(
            {
                "batch_size": bs,
                "exact": base_tokens == mtp_tokens,
                "baseline": base_tokens,
                "mtp": mtp_tokens,
                "diffs": [
                    {
                        "request_index": idx,
                        "baseline": base_row,
                        "mtp": mtp_row,
                    }
                    for idx, (base_row, mtp_row) in enumerate(zip(base_tokens, mtp_tokens))
                    if base_row != mtp_row
                ],
            }
        )
    return rows


def _commit_stats(mtp: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for run in mtp.get("runs", []):
        stats = _stats(run)
        out.append(
            {
                "batch_size": int(run.get("batch_size", -1)),
                "target_verify_calls": int(stats.get("target_verify_calls", 0)),
                "target_commit_kv_copies": int(stats.get("target_commit_kv_copies", 0)),
                "accepted_kv_copied_tokens": int(stats.get("accepted_kv_copied_tokens", 0)),
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


def _contract_events(stats: dict[str, Any]) -> list[dict[str, Any]]:
    return list(stats.get("target_verify_contract_trace") or [])


def _debug_entries_by_event(stats: dict[str, Any], events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    trace = list(stats.get("debug_trace") or [])
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cursor = 0
    for event_id, event in enumerate(events):
        count = int(event.get("batch_size", 0))
        for item in trace[cursor : cursor + count]:
            grouped[event_id].append(item)
        cursor += count
    if cursor < len(trace):
        grouped[-1].extend(trace[cursor:])
    return grouped


def _event_ledger(stats: dict[str, Any]) -> list[dict[str, Any]]:
    events = _contract_events(stats)
    grouped = _debug_entries_by_event(stats, events)
    ledger = []
    for event_id, event in enumerate(events):
        entries = grouped.get(event_id, [])
        ledger.append(
            {
                "event_id": event_id,
                "batch_size": event.get("batch_size"),
                "input_tokens": event.get("input_tokens", []),
                "positions": event.get("positions", []),
                "out_cache_loc": event.get("out_cache_loc", []),
                "row_depths": event.get("row_depths", []),
                "active_row_mask": event.get("active_row_mask", []),
                "padded_row_mask": event.get("padded_row_mask", []),
                "committed_seq_lens": event.get("committed_seq_lens", []),
                "seq_lens": event.get("seq_lens", []),
                "req_seq_lens": event.get("req_seq_lens", []),
                "c4_out_loc": event.get("c4_out_loc", []),
                "c128_out_loc": event.get("c128_out_loc", []),
                "c128_lifecycle": event.get("c128_lifecycle", {}),
                "entries": [
                    {
                        "uid": item.get("uid"),
                        "batch_index": item.get("batch_index"),
                        "cached_len": item.get("cached_len"),
                        "device_len": item.get("device_len"),
                        "input_tokens": item.get("input_tokens", []),
                        "draft_tokens": item.get("draft_tokens", []),
                        "target_tokens": item.get("target_tokens", []),
                        "accepted_prefix": item.get("accepted_prefix"),
                        "mismatch_depth": item.get("mismatch_depth"),
                        "candidate_copy_rows": item.get("candidate_copy_rows"),
                        "copy_rows": item.get("copy_rows"),
                        "emitted_tail": item.get("emitted_tail", []),
                        "row_depths": item.get("row_depths", []),
                    }
                    for item in entries
                ],
            }
        )
    return ledger


def _snapshot_row_map(event: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    mapping = event.get("mapping", {})
    positions = list(mapping.get("positions", []))
    full_locs = list(mapping.get("full_locs", []))
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for item in event.get("snapshot", {}).get("items", []):
        label = str(item.get("label", ""))
        locs = list(item.get("locs", item.get("offsets", [])))
        row_checksums = list(item.get("row_checksums", []))
        if not row_checksums:
            checksum = item.get("checksum") or item.get("data_checksum")
            if checksum:
                for idx, pos in enumerate(positions[: max(1, len(locs))]):
                    by_key[(label, int(pos))] = {
                        "position": int(pos),
                        "full_loc": full_locs[idx] if idx < len(full_locs) else None,
                        "loc": locs[idx] if idx < len(locs) else None,
                        "checksum": checksum,
                        "checksum_source": "tensor",
                        "shape": item.get("shape"),
                    }
            continue
        for row in row_checksums:
            row_idx = int(row.get("row", -1))
            if row_idx < 0 or row_idx >= len(positions):
                continue
            checksum = row.get("checksum", {})
            by_key[(label, int(positions[row_idx]))] = {
                "position": int(positions[row_idx]),
                "full_loc": full_locs[row_idx] if row_idx < len(full_locs) else None,
                "loc": locs[row_idx] if row_idx < len(locs) else None,
                "checksum": checksum,
                "checksum_source": "row",
                "shape": item.get("shape"),
            }
    return by_key


def _layer_id_from_label(label: str) -> int | None:
    marker = ".layer"
    if marker not in label:
        return None
    tail = label.split(marker, 1)[1]
    digits = []
    for ch in tail:
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return None
    return int("".join(digits))


def _checksum_numel(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    checksum = row.get("checksum")
    if not isinstance(checksum, dict):
        return None
    numel = checksum.get("numel")
    if numel is None:
        return None
    return int(numel)


def _checksums_comparable(base_row: dict[str, Any], mtp_row: dict[str, Any]) -> bool:
    base_numel = _checksum_numel(base_row)
    mtp_numel = _checksum_numel(mtp_row)
    return base_numel is not None and base_numel == mtp_numel


def _checksum_field(row: dict[str, Any], name: str) -> Any:
    checksum = row.get("checksum")
    if not isinstance(checksum, dict):
        return None
    return checksum.get(name)


def _checksum_uninitialized_like(row: dict[str, Any]) -> bool:
    """Detect NaN/huge aggregate patterns from torch.empty-style state rows."""
    checksum = row.get("checksum")
    if not isinstance(checksum, dict):
        return False
    for name in ("abs_sum", "max_abs", "sum"):
        if checksum.get(name) is None:
            return True
    for name in ("abs_sum", "max_abs"):
        value = checksum.get(name)
        if isinstance(value, (int, float)) and abs(float(value)) >= 1.0e30:
            return True
    for value in checksum.get("sample") or []:
        if isinstance(value, (int, float)) and abs(float(value)) >= 1.0e30:
            return True
    return False


def _c4_indexer_state_skip(
    label: str,
    base_row: dict[str, Any],
    mtp_row: dict[str, Any],
) -> dict[str, Any] | None:
    if not label.startswith("c4_indexer_state."):
        return None
    baseline_uninitialized_like = _checksum_uninitialized_like(base_row)
    mtp_uninitialized_like = _checksum_uninitialized_like(mtp_row)
    return {
        "classification": "c4_indexer_state_uninitialized_skip",
        "kind": "uninitialized_or_unconsumed_state",
        "component": label,
        "reason": "c4_indexer_state_has_no_analyzer_visible_write_or_consume_surface",
        "full_loc": base_row.get("full_loc"),
        "state_loc": base_row.get("loc"),
        "baseline_uninitialized_like": baseline_uninitialized_like,
        "baseline_sha256": _checksum_field(base_row, "sha256"),
        "baseline_abs_sum": _checksum_field(base_row, "abs_sum"),
        "baseline_max_abs": _checksum_field(base_row, "max_abs"),
        "mtp_uninitialized_like": mtp_uninitialized_like,
        "mtp_sha256": _checksum_field(mtp_row, "sha256"),
        "mtp_abs_sum": _checksum_field(mtp_row, "abs_sum"),
        "mtp_max_abs": _checksum_field(mtp_row, "max_abs"),
        "trace_write_fields": False,
        "trace_consume_fields": False,
    }


def _online_c128_bank0(
    event: dict[str, Any],
    label: str,
    state_loc: Any,
) -> dict[str, Any] | None:
    summary = event.get("online_c128_mtp") or {}
    if not summary.get("available"):
        return None
    if summary.get("storage") != "main_kv_score_buffer":
        return None
    layer_id = _layer_id_from_label(label)
    if layer_id is None:
        return None
    try:
        expected_state_loc = int(state_loc)
    except (TypeError, ValueError):
        return None
    for layer in summary.get("layers", []) or []:
        try:
            if int(layer.get("layer_id", -1)) != layer_id:
                continue
        except (TypeError, ValueError):
            continue
        for bank in layer.get("banks", []) or []:
            try:
                bank_id = int(bank.get("bank", -1))
                bank_state_loc = int(bank.get("state_loc", -1))
            except (TypeError, ValueError):
                continue
            if bank_id == 0 and bank_state_loc == expected_state_loc:
                return bank
    return None


def _c128_raw_loc_mapping_expected(
    event: dict[str, Any],
    label: str,
    base_row: dict[str, Any],
    mtp_row: dict[str, Any],
) -> dict[str, Any] | None:
    if not label.startswith("c128_attention_state."):
        return None
    base_full_loc = base_row.get("full_loc")
    mtp_full_loc = mtp_row.get("full_loc")
    if base_full_loc is None or mtp_full_loc is None or int(base_full_loc) != int(mtp_full_loc):
        return None
    mtp_loc = mtp_row.get("loc")
    if mtp_loc is None:
        return None
    full_loc = int(mtp_full_loc)
    expected_chunk_loc = full_loc // 128
    if int(mtp_loc) != expected_chunk_loc:
        return None
    bank0 = _online_c128_bank0(event, label, mtp_loc)
    if bank0 is None:
        return None
    comparable = _checksums_comparable(base_row, mtp_row)
    return {
        "kind": "raw_loc_mapping_expected",
        "component": label,
        "full_loc": full_loc,
        "chunk_id": expected_chunk_loc,
        "baseline_raw_loc": base_row.get("loc"),
        "mtp_bank0_loc": mtp_loc,
        "mtp_storage": (event.get("online_c128_mtp") or {}).get("storage"),
        "mtp_bank": 0,
        "baseline_checksum_numel": _checksum_numel(base_row),
        "mtp_checksum_numel": _checksum_numel(mtp_row),
        "checksum_comparable": comparable,
    }


def _baseline_index(stats: dict[str, Any]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in stats.get("state_parity_trace", []) or []:
        if event.get("event") != "baseline_after_normal_decode":
            continue
        uid = int(event.get("uid", -1))
        for pos in event.get("mapping", {}).get("positions", []):
            by_key[(uid, int(pos))].append(event)
    return by_key


def _event_kind_rank(kind: str) -> int:
    order = {
        "mtp_after_normal_before_verify": 0,
        "mtp_after_accepted_commit": 1,
        "mtp_after_normal_target_decode": 2,
    }
    return order.get(kind, 99)


def _checksum_owner(event_name: str, component: str) -> str:
    component_lower = component.lower()
    if (
        "attention_state" in component_lower
        or "indexer_state" in component_lower
        or component_lower.startswith("c4_")
        or component_lower.startswith("c128_")
    ):
        return "component_state_owner"
    if event_name == "mtp_after_accepted_commit":
        return "commit_row_value_owner"
    if event_name in {
        "mtp_after_normal_before_verify",
        "mtp_after_normal_target_decode",
    }:
        return "attention_state_owner"
    return "earlier_event_owner"


def _state_bisection(base_stats: dict[str, Any], mtp_stats: dict[str, Any]) -> dict[str, Any]:
    baseline = _baseline_index(base_stats)
    comparisons = []
    first = None
    c128_reclassifications = []
    c4_indexer_state_skips = []
    mtp_events = sorted(
        list(mtp_stats.get("state_parity_trace", []) or []),
        key=lambda e: (int(e.get("trace_index", 0)), _event_kind_rank(str(e.get("event", "")))),
    )
    for event in mtp_events:
        uid = int(event.get("uid", -1))
        row_map = _snapshot_row_map(event)
        positions = sorted({key[1] for key in row_map})
        for position in positions:
            candidates = baseline.get((uid, int(position)), [])
            base_event = candidates[0] if candidates else None
            record = {
                "mtp_trace_index": event.get("trace_index"),
                "mtp_event": event.get("event"),
                "uid": uid,
                "position": int(position),
                "cached_len": event.get("cached_len"),
                "baseline_trace_index": None if base_event is None else base_event.get("trace_index"),
                "mapping_status": "aligned" if base_event is not None else "missing_baseline",
                "first_mismatch": None,
                "planner_notes": [],
            }
            if base_event is None:
                record["first_mismatch"] = {
                    "owner": "instrumentation_no_go",
                    "component": "baseline_state_trace",
                }
                comparisons.append(record)
                if first is None:
                    first = record
                continue
            base_map = _snapshot_row_map(base_event)
            labels = sorted({label for label, pos in row_map if pos == position})
            for label in labels:
                mtp_row = row_map.get((label, position))
                base_row = base_map.get((label, position))
                if base_row is None:
                    record["first_mismatch"] = {
                        "owner": "component_state_owner",
                        "component": label,
                        "reason": "missing_baseline_component_row",
                    }
                    break
                mapping_fields = ("full_loc", "loc")
                mapping_equal = all(mtp_row.get(name) == base_row.get(name) for name in mapping_fields)
                checksum_equal = mtp_row.get("checksum") == base_row.get("checksum")
                if not mapping_equal:
                    c128_mapping = _c128_raw_loc_mapping_expected(event, label, base_row, mtp_row)
                    if c128_mapping is not None:
                        record["planner_notes"].append(c128_mapping)
                        c128_reclassifications.append(
                            {
                                "mtp_trace_index": event.get("trace_index"),
                                "mtp_event": event.get("event"),
                                "baseline_trace_index": base_event.get("trace_index"),
                                "uid": uid,
                                "position": int(position),
                                **c128_mapping,
                            }
                        )
                        if checksum_equal:
                            continue
                        if not c128_mapping["checksum_comparable"]:
                            record["planner_notes"].append(
                                {
                                    "kind": "checksum_not_comparable",
                                    "component": label,
                                    "reason": "legacy_c128_row_vs_online_c128_bank0_storage_shape",
                                    "baseline_checksum_numel": c128_mapping[
                                        "baseline_checksum_numel"
                                    ],
                                    "mtp_checksum_numel": c128_mapping["mtp_checksum_numel"],
                                }
                            )
                            continue
                        record["first_mismatch"] = {
                            "owner": "component_state_owner",
                            "component": label,
                            "reason": "c128_checksum_mismatch_after_logical_mapping",
                            "logical_mapping": c128_mapping,
                            "baseline_checksum": base_row.get("checksum"),
                            "mtp_checksum": mtp_row.get("checksum"),
                        }
                        break
                    record["first_mismatch"] = {
                        "owner": "commit_mapping_owner",
                        "component": label,
                        "baseline": {name: base_row.get(name) for name in mapping_fields},
                        "mtp": {name: mtp_row.get(name) for name in mapping_fields},
                    }
                    break
                if not checksum_equal:
                    c4_skip = _c4_indexer_state_skip(label, base_row, mtp_row)
                    if c4_skip is not None:
                        note = {
                            "mtp_trace_index": event.get("trace_index"),
                            "mtp_event": event.get("event"),
                            "baseline_trace_index": base_event.get("trace_index"),
                            "uid": uid,
                            "position": int(position),
                            **c4_skip,
                        }
                        record["planner_notes"].append(note)
                        c4_indexer_state_skips.append(note)
                        continue
                    record["first_mismatch"] = {
                        "owner": _checksum_owner(str(event.get("event", "")), label),
                        "component": label,
                        "baseline_checksum": base_row.get("checksum"),
                        "mtp_checksum": mtp_row.get("checksum"),
                    }
                    break
            comparisons.append(record)
            if first is None and record.get("first_mismatch"):
                first = record
    return {
        "first_divergence": first,
        "comparisons": comparisons[:256],
        "comparison_count": len(comparisons),
        "c128_raw_loc_reclassifications": c128_reclassifications[:256],
        "c128_raw_loc_reclassification_count": len(c128_reclassifications),
        "c4_indexer_state_skips": c4_indexer_state_skips[:256],
        "c4_indexer_state_skip_count": len(c4_indexer_state_skips),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mtp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    base = _read(args.baseline)
    mtp = _read(args.mtp)
    base_run = _run_by_bs(base, args.batch_size)
    mtp_run = _run_by_bs(mtp, args.batch_size)
    base_stats = _stats(base_run)
    mtp_stats = _stats(mtp_run)
    result = {
        "exactness_matrix": _token_matrix(base, mtp),
        "accepted_commit_stats": _commit_stats(mtp),
        "bs6_event_timeline": _event_ledger(mtp_stats),
        "bs6_state_bisection": _state_bisection(base_stats, mtp_stats),
        "bs6_baseline_state_trace_count": len(base_stats.get("state_parity_trace", []) or []),
        "bs6_mtp_state_trace_count": len(mtp_stats.get("state_parity_trace", []) or []),
        "env": {
            "baseline": base.get("env", {}),
            "mtp": mtp.get("env", {}),
        },
    }
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

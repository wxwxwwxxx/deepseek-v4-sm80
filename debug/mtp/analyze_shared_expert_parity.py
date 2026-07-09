from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BOUNDARIES = (
    "shared_expert_input",
    "shared_expert_forward_input",
    "shared_expert_gate_up",
    "shared_expert_gate_chunk",
    "shared_expert_up_chunk",
    "shared_expert_silu_and_mul_clamp",
    "shared_expert_hidden_for_down",
    "shared_expert_down_proj",
    "shared_expert_output_raw",
    "shared_expert_output",
)


def _rank_path(path: str, rank: int) -> Path:
    candidate = Path(path)
    ranked = candidate.with_suffix(candidate.suffix + f".rank{rank}.json")
    if ranked.exists():
        return ranked
    if candidate.exists() and f".rank{rank}." in candidate.name:
        return candidate
    return candidate


def _read(path: str, rank: int) -> dict[str, Any]:
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


def _find_baseline_event(
    stats: dict[str, Any],
    *,
    input_token: int,
    position: int,
    full_loc: int,
) -> tuple[dict[str, Any], int]:
    for event in stats.get("normal_producer_trace_debug", []) or []:
        positions = list(event.get("positions") or [])
        locs = list(event.get("out_cache_loc") or [])
        input_ids = list(event.get("input_ids") or [])
        for row, (pos, loc, token) in enumerate(zip(positions, locs, input_ids)):
            if int(pos) == position and int(loc) == full_loc and int(token) == input_token:
                return event, row
    raise KeyError("baseline anchor event not found")


def _row_record(record: dict[str, Any], row_index: int) -> dict[str, Any]:
    for row in record.get("row_records", []) or []:
        if int(row.get("row", -1)) == int(row_index):
            return dict(row)
    rows = record.get("row_records", []) or []
    return dict(rows[0]) if rows else {}


def _get_path(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_present(value: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        found = _get_path(value, *path)
        if found is not None:
            return found
    return None


def _baseline_records(event: dict[str, Any], row_index: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in event.get("operator_trace", []) or []:
        name = str(record.get("operator_name", ""))
        if name not in BOUNDARIES or int(record.get("layer_id", -1)) != 1:
            continue
        if name in out:
            continue
        row = _row_record(record, row_index)
        extra = record.get("extra") or {}
        out[name] = {
            "operator_name": name,
            "path": record.get("path"),
            "context": record.get("batch_context"),
            "params": record.get("params"),
            "extra": extra,
            "input_metadata": record.get("input_tensor_metadata"),
            "output_metadata": record.get("output_tensor_metadata"),
            "input_sha": _first_present(
                {"row": row, "extra": extra},
                (
                    ("row", "input", "raw_sha256"),
                    ("extra", "input_census", "row0_checksum"),
                    ("extra", "shared_input", "row0_checksum"),
                ),
            ),
            "output_sha": _first_present(
                {"row": row, "extra": extra},
                (
                    ("row", "output", "raw_sha256"),
                    ("extra", "output_census", "row0_checksum"),
                    ("extra", "raw_output", "row0_checksum"),
                    ("extra", "finalized_output", "row0_checksum"),
                ),
            ),
            "source_row_records": extra.get("source_row_records"),
        }
    return out


def _target_records(
    stats: dict[str, Any],
    *,
    rank: int,
    input_token: int,
    position: int,
    depth: int,
    request_id: int,
) -> dict[str, dict[str, Any]]:
    prefix = f"mtp_target_verify_vs_row0_normal_oracle.rank{rank}."
    suffix = f".req{request_id}.depth{depth}.pos{position}.tok{input_token}"
    out: dict[str, dict[str, Any]] = {}
    for event in stats.get("operator_parity_debug", []) or []:
        for record in event.get("records", []) or []:
            case_id = str(record.get("case_id", ""))
            if not case_id.startswith(prefix) or not case_id.endswith(suffix):
                continue
            name = str(record.get("operator_name", ""))
            if name in BOUNDARIES and int(record.get("layer", -1)) == 1 and name not in out:
                out[name] = record
    return out


def _microbench_record(
    stats: dict[str, Any],
    *,
    rank: int,
    input_token: int,
    position: int,
    depth: int,
    request_id: int,
) -> dict[str, Any] | None:
    for record in stats.get("target_verify_moe_microbatch_runtime_trace", []) or []:
        if record.get("record_type") != "shared_expert_microbench":
            continue
        if int(record.get("layer_id", -1)) != 1:
            continue
        for row in record.get("source_row_records", []) or []:
            if int(row.get("input_token", -1)) != int(input_token):
                continue
            if int(row.get("position", -1)) != int(position):
                continue
            row_depth = row.get("row_depth")
            if row_depth is not None and int(row_depth) != int(depth):
                continue
            return record
    prefix = f"mtp_target_verify_vs_row0_normal_oracle.rank{rank}."
    suffix = f".req{request_id}.depth{depth}.pos{position}.tok{input_token}"
    for event in stats.get("operator_parity_debug", []) or []:
        for record in event.get("records", []) or []:
            if record.get("operator_name") != "shared_expert_microbench":
                continue
            case_id = str(record.get("case_id", ""))
            if case_id.startswith(prefix) and case_id.endswith(suffix):
                return record
    return None


def _target_sha(record: dict[str, Any] | None) -> Any:
    if not isinstance(record, dict):
        return None
    extra = record.get("target_verify_extra") or {}
    return _first_present(
        {"record": record, "extra": extra},
        (
            ("record", "target_verify_raw_sha256"),
            ("extra", "output_census", "row0_checksum"),
            ("extra", "raw_output", "row0_checksum"),
            ("extra", "finalized_output", "row0_checksum"),
            ("extra", "shared_input", "row0_checksum"),
        ),
    )


def _target_input_sha(record: dict[str, Any] | None) -> Any:
    if not isinstance(record, dict):
        return None
    extra = record.get("target_verify_extra") or {}
    return _first_present(
        {"record": record, "extra": extra},
        (
            ("record", "target_verify_input_raw_sha256"),
            ("extra", "input_census", "row0_checksum"),
            ("extra", "shared_input", "row0_checksum"),
        ),
    )


def _boundary_table(
    baseline: dict[str, dict[str, Any]],
    target: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in BOUNDARIES:
        base = baseline.get(name)
        tgt = target.get(name)
        base_sha = None if base is None else base.get("output_sha")
        target_sha = _target_sha(tgt)
        if base_sha is None or target_sha is None:
            exact = None
        else:
            exact = base_sha == target_sha
        rows.append(
            {
                "operator": name,
                "baseline_output_sha": base_sha,
                "target_output_sha": target_sha,
                "true_baseline_exact": exact,
                "same_run_oracle_bit_exact": None if tgt is None else tgt.get("bit_exact_result"),
                "same_run_oracle_max_delta": None if tgt is None else tgt.get("max_delta"),
                "baseline_path": None if base is None else base.get("path"),
                "target_path": None if tgt is None else tgt.get("target_verify_kernel_or_path"),
                "baseline_output_metadata": None if base is None else base.get("output_metadata"),
                "target_output_metadata": None if tgt is None else _get_path(tgt, "output_tensor_metadata", "target_verify"),
            }
        )
    return rows


def _input_equivalence_table(
    baseline: dict[str, dict[str, Any]],
    target: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    names = ("shared_expert_input", "shared_expert_forward_input")
    rows: list[dict[str, Any]] = []
    for name in names:
        base = baseline.get(name)
        tgt = target.get(name)
        base_sha = None if base is None else base.get("output_sha")
        target_sha = _target_sha(tgt)
        rows.append(
            {
                "operator": name,
                "baseline_sha": base_sha,
                "target_sha": target_sha,
                "true_baseline_exact": (
                    None if base_sha is None or target_sha is None else base_sha == target_sha
                ),
                "baseline_metadata": None if base is None else base.get("output_metadata"),
                "target_metadata": None if tgt is None else _get_path(tgt, "output_tensor_metadata", "target_verify"),
                "target_source_row_records": _get_path(
                    tgt, "target_verify_extra", "source_row_records"
                ),
                "target_forward_context": _get_path(
                    tgt, "target_verify_extra", "shared_forward_context"
                ),
            }
        )
    return rows


def _microbench_tables(
    record: dict[str, Any] | None,
    *,
    baseline_raw_sha: Any,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {
            "available": False,
            "row_shape_oracles": [],
            "value_sweep": [],
            "stability": {},
        }
    micro = _get_path(record, "target_verify_extra", "microbench") or record.get("microbench") or {}
    row_shape_rows = []
    for row in micro.get("row_shape_oracles", []) or []:
        output_sha = _get_path(row, "output", "row0_checksum")
        row_shape_rows.append(
            {
                "variant": row.get("variant"),
                "output_sha": output_sha,
                "matches_baseline_raw": (
                    None if output_sha is None or baseline_raw_sha is None else output_sha == baseline_raw_sha
                ),
                "vs_one_row": row.get("vs_one_row"),
            }
        )
    value_rows = []
    for row in micro.get("value_sweep", []) or []:
        value_rows.append(
            {
                "variant": row.get("variant"),
                "one_row_sha": _get_path(row, "one_row", "row0_checksum"),
                "parent_like_4_zero_sha": _get_path(
                    row, "parent_like_4_zero_fill", "row0_checksum"
                ),
                "one_vs_parent_like_4": row.get("one_vs_parent_like_4"),
            }
        )
    return {
        "available": True,
        "stability": micro.get("repeated_same_input_same_shape") or {},
        "row_shape_oracles": row_shape_rows,
        "value_sweep": value_rows,
        "reference_torch": micro.get("reference_torch"),
        "contract": micro.get("contract"),
        "parent_rows": micro.get("parent_rows"),
        "parent_slot": micro.get("parent_slot"),
        "source_row_records": (
            _get_path(record, "target_verify_extra", "source_row_records")
            or record.get("source_row_records")
        ),
        "input_tensor_metadata": (
            _get_path(record, "input_tensor_metadata", "target_verify")
            or record.get("input_tensor_metadata")
        ),
        "output_tensor_metadata": (
            _get_path(record, "output_tensor_metadata", "target_verify")
            or record.get("output_tensor_metadata")
        ),
        "input_census": record.get("input_census"),
        "output_census": record.get("output_census"),
    }


def _classify(
    input_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    microbench: dict[str, Any],
) -> str:
    row_shape_rows = microbench.get("row_shape_oracles", []) if microbench.get("available") else []
    current_matches = None
    parent_matches = False
    for row in row_shape_rows:
        if row.get("variant") == "one_row":
            current_matches = row.get("matches_baseline_raw")
        if str(row.get("variant", "")).startswith("parent_like_4") and row.get("matches_baseline_raw"):
            parent_matches = True
    if current_matches is False and parent_matches:
        return "shared_expert_backend_row_shape_owner"

    forward_input = next(
        (row for row in input_rows if row["operator"] == "shared_expert_forward_input"),
        None,
    )
    shared_input = next(
        (row for row in input_rows if row["operator"] == "shared_expert_input"),
        None,
    )
    input_exact = (
        forward_input.get("true_baseline_exact")
        if isinstance(forward_input, dict) and forward_input.get("true_baseline_exact") is not None
        else (shared_input or {}).get("true_baseline_exact")
    )
    if input_exact is False:
        return "shared_expert_input_owner"

    first_mismatch = None
    for row in boundary_rows:
        if row.get("operator") in {"shared_expert_input", "shared_expert_forward_input"}:
            continue
        if row.get("true_baseline_exact") is False:
            first_mismatch = row.get("operator")
            break
    if first_mismatch in {"shared_expert_gate_up", "shared_expert_gate_chunk", "shared_expert_up_chunk"}:
        return "shared_expert_gate_up_owner"
    if first_mismatch in {"shared_expert_silu_and_mul_clamp", "shared_expert_hidden_for_down"}:
        return "shared_expert_activation_owner"
    if first_mismatch in {"shared_expert_down_proj", "shared_expert_output_raw"}:
        return "shared_expert_down_proj_owner"
    if first_mismatch == "shared_expert_output":
        return "shared_expert_finalize_cast_owner"
    return "shared_expert_no_go"


def _token_matrix(base: dict[str, Any], mtp: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mtp_run in mtp.get("runs", []) or []:
        bs = int(mtp_run.get("batch_size", -1))
        try:
            base_run = _run_by_bs(base, bs)
        except KeyError:
            continue
        base_tokens = base_run.get("token_ids", [])
        mtp_tokens = mtp_run.get("token_ids", [])
        rows.append(
            {
                "batch_size": bs,
                "exact": base_tokens == mtp_tokens,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mtp", required=True)
    parser.add_argument("--microbench-mtp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate-analysis")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--input-token", type=int, default=11111)
    parser.add_argument("--position", type=int, default=5)
    parser.add_argument("--full-loc", type=int, default=3077)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--request-id", type=int, default=0)
    args = parser.parse_args()

    baseline_payload = _read(args.baseline, args.rank)
    mtp_payload = _read(args.mtp, args.rank)
    microbench_payload = (
        _read(args.microbench_mtp, args.rank)
        if args.microbench_mtp
        else mtp_payload
    )
    baseline_run = _run_by_bs(baseline_payload, args.batch_size)
    mtp_run = _run_by_bs(mtp_payload, args.batch_size)
    microbench_run = _run_by_bs(microbench_payload, args.batch_size)
    baseline_event, baseline_row = _find_baseline_event(
        _stats(baseline_run),
        input_token=args.input_token,
        position=args.position,
        full_loc=args.full_loc,
    )
    base_records = _baseline_records(baseline_event, baseline_row)
    target_records = _target_records(
        _stats(mtp_run),
        rank=args.rank,
        input_token=args.input_token,
        position=args.position,
        depth=args.depth,
        request_id=args.request_id,
    )
    input_table = _input_equivalence_table(base_records, target_records)
    boundary_table = _boundary_table(base_records, target_records)
    baseline_raw_sha = (
        (base_records.get("shared_expert_output") or {}).get("output_sha")
        or (base_records.get("shared_expert_down_proj") or {}).get("output_sha")
        or (base_records.get("shared_expert_output_raw") or {}).get("output_sha")
    )
    microbench = _microbench_tables(
        _microbench_record(
            _stats(microbench_run),
            rank=args.rank,
            input_token=args.input_token,
            position=args.position,
            depth=args.depth,
            request_id=args.request_id,
        ),
        baseline_raw_sha=baseline_raw_sha,
    )
    aggregate = None
    if args.aggregate_analysis:
        aggregate = json.loads(Path(args.aggregate_analysis).read_text(encoding="utf-8"))

    payload = {
        "classification": _classify(input_table, boundary_table, microbench),
        "anchor": {
            "rank": int(args.rank),
            "layer": 1,
            "batch_size": int(args.batch_size),
            "request_id": int(args.request_id),
            "input_token": int(args.input_token),
            "position": int(args.position),
            "full_loc": int(args.full_loc),
            "depth": int(args.depth),
            "baseline_row": int(baseline_row),
        },
        "token_matrix": _token_matrix(baseline_payload, mtp_payload),
        "aggregate_analysis": aggregate,
        "rank2_shared_input_equivalence": input_table,
        "shared_expert_internal_boundaries": boundary_table,
        "targeted_microbench": microbench,
        "weight_cache_backend_ledger": _get_path(
            target_records.get("shared_expert_forward_input", {}),
            "target_verify_extra",
            "backend",
        )
        or _get_path(
            target_records.get("shared_expert_gate_up", {}),
            "target_verify_extra",
            "backend",
        ),
        "source_parity_inputs": {
            "baseline": str(_rank_path(args.baseline, args.rank)),
            "mtp": str(_rank_path(args.mtp, args.rank)),
            "microbench_mtp": (
                str(_rank_path(args.microbench_mtp, args.rank))
                if args.microbench_mtp
                else None
            ),
            "aggregate_analysis": args.aggregate_analysis,
        },
    }
    _write(args.output, payload)


if __name__ == "__main__":
    main()

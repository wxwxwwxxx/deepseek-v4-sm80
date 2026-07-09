from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPERATORS = (
    "routed_expert_output",
    "shared_expert_output",
    "expert_aggregate_before_reduce",
    "expert_reduce_output",
    "moe_output",
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


def _token_matrix(base: dict[str, Any], mtp: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mtp_run in mtp.get("runs", []) or []:
        bs = int(mtp_run.get("batch_size", -1))
        base_run = _run_by_bs(base, bs)
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
                "baseline": base_tokens,
                "mtp": mtp_tokens,
            }
        )
    return rows


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


def _checksum(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _baseline_operator_table(event: dict[str, Any], row_index: int) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for record in event.get("operator_trace", []) or []:
        name = record.get("operator_name")
        if name not in OPERATORS or int(record.get("layer_id", -1)) != 1:
            continue
        row = _row_record(record, row_index)
        extra = record.get("extra") or {}
        table[str(name)] = {
            "row_output": _checksum(row, "output", "raw_sha256"),
            "row_input": _checksum(row, "input", "raw_sha256"),
            "routed_input": _checksum(extra, "routed_input", "row0_checksum"),
            "shared_input": _checksum(extra, "shared_input", "row0_checksum"),
            "aggregate_output": _checksum(extra, "aggregate_output", "row0_checksum"),
            "communication_input": _checksum(extra, "communication_input", "row0_checksum"),
            "post_reduce": _checksum(extra, "post_reduce", "row0_checksum"),
            "final_cast_input": _checksum(extra, "final_cast_input", "row0_checksum"),
            "final_cast_output": _checksum(extra, "final_cast_output", "row0_checksum"),
            "stage": extra.get("stage"),
            "aggregate_order": extra.get("aggregate_order"),
        }
    return table


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
            name = record.get("operator_name")
            if name in OPERATORS and int(record.get("layer", -1)) == 1:
                out[str(name)] = record
    return out


def _target_operator_table(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for name, record in records.items():
        extra = record.get("target_verify_extra") or {}
        table[name] = {
            "finalized_output": _checksum(extra, "finalized_output", "row0_checksum"),
            "routed_input": _checksum(extra, "routed_input", "row0_checksum"),
            "shared_input": _checksum(extra, "shared_input", "row0_checksum"),
            "aggregate_output": _checksum(extra, "aggregate_output", "row0_checksum"),
            "communication_input": _checksum(extra, "communication_input", "row0_checksum"),
            "post_reduce": _checksum(extra, "post_reduce", "row0_checksum"),
            "final_cast_input": _checksum(extra, "final_cast_input", "row0_checksum"),
            "final_cast_output": _checksum(extra, "final_cast_output", "row0_checksum"),
            "pre_reduce_snapshot_preserved": extra.get("pre_reduce_snapshot_preserved"),
            "microbatch_contract": extra.get("microbatch_contract"),
            "bit_exact_same_run_normal_oracle": record.get("bit_exact_result"),
            "max_delta_same_run_normal_oracle": record.get("max_delta"),
        }
    return table


def _compare_rank(
    baseline: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    base_routed = _checksum(baseline, "routed_expert_output", "row_output")
    target_routed = _checksum(target, "routed_expert_output", "finalized_output")
    base_shared = _checksum(baseline, "shared_expert_output", "row_output")
    target_shared = _checksum(target, "shared_expert_output", "finalized_output")
    base_aggregate = _checksum(
        baseline, "expert_aggregate_before_reduce", "aggregate_output"
    )
    target_aggregate = _checksum(
        target, "expert_aggregate_before_reduce", "aggregate_output"
    )
    base_reduce_input = _checksum(
        baseline, "expert_reduce_output", "communication_input"
    )
    target_reduce_input = _checksum(
        target, "expert_reduce_output", "communication_input"
    )
    base_post_reduce = _checksum(baseline, "expert_reduce_output", "post_reduce")
    target_post_reduce = _checksum(target, "expert_reduce_output", "post_reduce")
    return {
        "routed_matches": base_routed == target_routed,
        "shared_matches": base_shared == target_shared,
        "aggregate_matches": base_aggregate == target_aggregate,
        "reduce_input_matches": base_reduce_input == target_reduce_input,
        "post_reduce_matches": base_post_reduce == target_post_reduce,
        "baseline_routed": base_routed,
        "target_routed": target_routed,
        "baseline_shared": base_shared,
        "target_shared": target_shared,
        "baseline_aggregate": base_aggregate,
        "target_aggregate": target_aggregate,
        "baseline_reduce_input": base_reduce_input,
        "target_reduce_input": target_reduce_input,
        "baseline_post_reduce": base_post_reduce,
        "target_post_reduce": target_post_reduce,
    }


def _classification(rank_rows: list[dict[str, Any]]) -> str:
    if not rank_rows:
        return "moe_aggregate_instrumentation_no_go"
    if any(
        not row["comparison"].get("routed_matches")
        or not row["comparison"].get("shared_matches")
        for row in rank_rows
    ):
        return "moe_aggregate_instrumentation_owner"
    if all(row["comparison"].get("aggregate_matches") for row in rank_rows):
        if all(row["comparison"].get("reduce_input_matches") for row in rank_rows):
            if all(row["comparison"].get("post_reduce_matches") for row in rank_rows):
                return "moe_aggregate_instrumentation_owner"
            return "moe_aggregate_microbatch_contract_owner"
        return "moe_aggregate_reduce_input_owner"
    return "moe_aggregate_row_index_owner"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mtp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--ranks", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--input-token", type=int, default=11111)
    parser.add_argument("--position", type=int, default=5)
    parser.add_argument("--full-loc", type=int, default=3077)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--request-id", type=int, default=0)
    args = parser.parse_args()

    rank_rows: list[dict[str, Any]] = []
    token_matrix: list[dict[str, Any]] = []
    for rank in args.ranks:
        baseline_payload = _read(args.baseline, rank)
        mtp_payload = _read(args.mtp, rank)
        if int(rank) == 0:
            token_matrix = _token_matrix(baseline_payload, mtp_payload)
        baseline_run = _run_by_bs(baseline_payload, args.batch_size)
        mtp_run = _run_by_bs(mtp_payload, args.batch_size)
        baseline_event, baseline_row = _find_baseline_event(
            _stats(baseline_run),
            input_token=args.input_token,
            position=args.position,
            full_loc=args.full_loc,
        )
        baseline_ops = _baseline_operator_table(baseline_event, baseline_row)
        target_ops = _target_operator_table(
            _target_records(
                _stats(mtp_run),
                rank=rank,
                input_token=args.input_token,
                position=args.position,
                depth=args.depth,
                request_id=args.request_id,
            )
        )
        rank_rows.append(
            {
                "rank": int(rank),
                "baseline_trace_index": baseline_event.get("trace_index"),
                "baseline_row": int(baseline_row),
                "baseline": baseline_ops,
                "target": target_ops,
                "comparison": _compare_rank(baseline_ops, target_ops),
            }
        )

    result = {
        "classification": _classification(rank_rows),
        "anchor": {
            "input_token": int(args.input_token),
            "position": int(args.position),
            "full_loc": int(args.full_loc),
            "depth": int(args.depth),
            "request_id": int(args.request_id),
            "batch_size": int(args.batch_size),
        },
        "exactness_matrix": token_matrix,
        "rank_rows": rank_rows,
    }
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

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


def _read(path: str, rank: int) -> dict[str, Any]:
    return json.loads(_rank_path(path, rank).read_text(encoding="utf-8"))


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
    for mtp_run in mtp.get("runs", []):
        bs = int(mtp_run.get("batch_size", -1))
        base_run = _run_by_bs(base, bs)
        base_tokens = base_run.get("token_ids", [])
        mtp_tokens = mtp_run.get("token_ids", [])
        rows.append(
            {
                "batch_size": bs,
                "exact": base_tokens == mtp_tokens,
                "diff_count": sum(
                    1
                    for base_row, mtp_row in zip(base_tokens, mtp_tokens)
                    if base_row != mtp_row
                ),
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


def _debug_env(payload: dict[str, Any]) -> dict[str, Any]:
    env = payload.get("env", {})
    return dict(env.get("debug_env") or {})


def _find_anchor_row(
    mtp_stats: dict[str, Any],
    *,
    uid: int,
    position: int,
    full_loc: int,
    depth: int,
) -> dict[str, Any] | None:
    for event in mtp_stats.get("row_depth_oracle_debug", []) or []:
        for entry in event.get("entries", []) or []:
            if int(entry.get("uid", -1)) != int(uid):
                continue
            for row in entry.get("rows", []) or []:
                metadata = row.get("target_metadata") or {}
                row_position = row.get("position", metadata.get("position"))
                row_depth = row.get("depth")
                row_full_loc = metadata.get("out_cache_loc")
                if (
                    int(row_position) == int(position)
                    and int(row_depth) == int(depth)
                    and int(row_full_loc) == int(full_loc)
                ):
                    return {
                        "event": event,
                        "entry": entry,
                        "row": row,
                    }
    return None


def _find_baseline_anchor(
    baseline_stats: dict[str, Any],
    *,
    position: int,
    full_loc: int,
) -> dict[str, Any] | None:
    for event in baseline_stats.get("normal_producer_trace_debug", []) or []:
        positions = list(event.get("positions") or [])
        out_locs = list(event.get("out_cache_loc") or [])
        for row, (row_position, row_loc) in enumerate(zip(positions, out_locs)):
            if int(row_position) == int(position) and int(row_loc) == int(full_loc):
                return {
                    "event": event,
                    "row": int(row),
                }
    return None


def _operator_record(
    row: dict[str, Any],
    *,
    operator_name: str,
    layer_id: int,
) -> dict[str, Any] | None:
    parity = row.get("operator_parity") or {}
    for record in parity.get("records", []) or []:
        if (
            str(record.get("operator_name")) == str(operator_name)
            and int(record.get("layer", -1)) == int(layer_id)
        ):
            return record
    return None


def _trace_row_record(entry: dict[str, Any] | None, row: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}
    for record in entry.get("row_records", []) or []:
        if int(record.get("row", -1)) == int(row):
            return record
    return {}


def _trace_entry(
    entries: list[dict[str, Any]],
    *,
    operator_name: str,
    layer_id: int,
) -> dict[str, Any] | None:
    for entry in entries or []:
        if (
            str(entry.get("operator_name")) == str(operator_name)
            and int(entry.get("layer_id", -1)) == int(layer_id)
        ):
            return entry
    return None


def _weight_cache_digest(extra: dict[str, Any] | None) -> dict[str, Any]:
    probe = (extra or {}).get("weight_cache_probe") or {}
    cache = probe.get("cache") or {}
    return {
        "owner_label": probe.get("owner_label"),
        "cache_kind": probe.get("cache_kind"),
        "cache_name": probe.get("cache_name"),
        "source_weight_sampled_raw_sha256": (
            (probe.get("source_weight") or {}).get("sampled_raw_sha256")
        ),
        "source_scale_sampled_raw_sha256": (
            (probe.get("source_scale") or {}).get("sampled_raw_sha256")
        ),
        "cache_sampled_raw_sha256": cache.get("sampled_raw_sha256"),
        "cache_metadata": cache.get("metadata"),
        "source_weight_metadata": (probe.get("source_weight") or {}).get("metadata"),
        "source_scale_metadata": (probe.get("source_scale") or {}).get("metadata"),
    }


def _per_row_digest(extra: dict[str, Any] | None) -> dict[str, Any]:
    probe = (extra or {}).get("per_row_probe") or {}
    compare = probe.get("per_row_vs_batched") or {}
    return {
        "available": probe.get("available"),
        "path": probe.get("path"),
        "rows": probe.get("rows"),
        "allclose": compare.get("allclose"),
        "bit_exact": compare.get("bit_exact"),
        "max_delta": compare.get("max_delta"),
        "mean_delta": compare.get("mean_delta"),
        "reason": probe.get("reason") or probe.get("error"),
    }


def _classification(record: dict[str, Any] | None) -> str:
    if record is None:
        return "q_wqb_instrumentation_no_go"
    if not record.get("input_comparison", {}).get("available", False):
        return "q_wqb_instrumentation_no_go"
    if not bool(record.get("input_bit_exact_result", False)):
        return "q_wqb_input_owner"
    if bool(record.get("bit_exact_result", False)):
        return "q_wqb_not_owner"

    normal_extra = record.get("normal_extra") or {}
    target_extra = record.get("target_verify_extra") or {}
    normal_weight = _weight_cache_digest(normal_extra)
    target_weight = _weight_cache_digest(target_extra)
    comparable_keys = (
        "cache_kind",
        "cache_name",
        "source_weight_sampled_raw_sha256",
        "source_scale_sampled_raw_sha256",
        "cache_sampled_raw_sha256",
    )
    if any(normal_weight.get(key) != target_weight.get(key) for key in comparable_keys):
        return "q_wqb_weight_cache_owner"

    normal_path = record.get("normal_kernel_or_path")
    target_path = record.get("target_verify_kernel_or_path")
    normal_params = record.get("normal_params") or {}
    target_params = record.get("target_verify_params") or {}
    only_row_invariant_diff = (
        normal_path != target_path
        and str(target_path).endswith(".row_invariant_local")
        and str(target_path).startswith(str(normal_path))
    )
    if only_row_invariant_diff or normal_params.get("row_invariant_local") != target_params.get(
        "row_invariant_local"
    ):
        return "q_wqb_row_invariant_local_owner"
    if normal_path != target_path:
        return "q_wqb_dispatch_owner"

    target_per_row = _per_row_digest(target_extra)
    if target_per_row.get("available") is True and target_per_row.get("bit_exact") is False:
        return "q_wqb_row_shape_owner"
    if bool(record.get("allclose_result", False)):
        return "q_wqb_precision_backend_owner"
    return "q_wqb_precision_backend_owner"


def _direct_baseline_target_classification(
    *,
    baseline_wq_b: dict[str, Any] | None,
    baseline_row: int,
    target_wq_b: dict[str, Any] | None,
    target_row: int,
    oracle_record: dict[str, Any] | None,
) -> str:
    if not isinstance(baseline_wq_b, dict) or not isinstance(target_wq_b, dict):
        return "q_wqb_instrumentation_no_go"
    base_row = _trace_row_record(baseline_wq_b, baseline_row)
    target_row_record = _trace_row_record(target_wq_b, target_row)
    base_input = base_row.get("input") or {}
    target_input = target_row_record.get("input") or {}
    base_output = base_row.get("output") or {}
    target_output = target_row_record.get("output") or {}
    if not base_input.get("available") or not target_input.get("available"):
        return "q_wqb_instrumentation_no_go"
    if base_input.get("raw_sha256") != target_input.get("raw_sha256"):
        return "q_wqb_input_owner"
    if base_output.get("raw_sha256") == target_output.get("raw_sha256"):
        return "q_wqb_not_owner"

    base_extra = baseline_wq_b.get("extra") or {}
    target_extra = target_wq_b.get("extra") or {}
    base_weight = _weight_cache_digest(base_extra)
    target_weight = _weight_cache_digest(target_extra)
    comparable_keys = (
        "cache_kind",
        "cache_name",
        "source_weight_sampled_raw_sha256",
        "source_scale_sampled_raw_sha256",
        "cache_sampled_raw_sha256",
    )
    if any(base_weight.get(key) != target_weight.get(key) for key in comparable_keys):
        return "q_wqb_weight_cache_owner"

    base_per_row = _per_row_digest(base_extra)
    oracle_pass = bool(oracle_record and oracle_record.get("bit_exact_result"))
    if base_per_row.get("available") is True and base_per_row.get("bit_exact") is False:
        return "q_wqb_row_shape_owner"
    if oracle_pass and str(target_wq_b.get("path", "")).endswith(".row_invariant_local"):
        return "q_wqb_row_shape_owner"
    if baseline_wq_b.get("path") != target_wq_b.get("path"):
        return "q_wqb_dispatch_owner"
    return "q_wqb_precision_backend_owner"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _read(args.baseline, args.rank)
    mtp = _read(args.mtp, args.rank)
    baseline_run = _run_by_bs(baseline, args.batch_size)
    mtp_run = _run_by_bs(mtp, args.batch_size)
    baseline_stats = _stats(baseline_run)
    mtp_stats = _stats(mtp_run)
    anchor = _find_anchor_row(
        mtp_stats,
        uid=args.uid,
        position=args.position,
        full_loc=args.full_loc,
        depth=args.depth,
    )
    if anchor is None:
        return {
            "classification": "q_wqb_instrumentation_no_go",
            "reason": "anchor row not found in row_depth_oracle_debug",
            "rank": int(args.rank),
            "anchor": {
                "uid": int(args.uid),
                "position": int(args.position),
                "full_loc": int(args.full_loc),
                "depth": int(args.depth),
            },
            "exactness_matrix": _token_matrix(baseline, mtp),
            "env": {
                "baseline_debug": _debug_env(baseline),
                "mtp_debug": _debug_env(mtp),
            },
        }

    row = anchor["row"]
    wq_b_record = _operator_record(row, operator_name="wq_b", layer_id=args.layer)
    q_wqb_output_record = _operator_record(
        row,
        operator_name="q_wqb_output",
        layer_id=args.layer,
    )
    target_trace = anchor["event"].get("target_operator_trace") or []
    oracle_trace = row.get("oracle_operator_trace") or []
    target_wq_b = _trace_entry(target_trace, operator_name="wq_b", layer_id=args.layer)
    oracle_wq_b = _trace_entry(oracle_trace, operator_name="wq_b", layer_id=args.layer)
    baseline_anchor = _find_baseline_anchor(
        baseline_stats,
        position=args.position,
        full_loc=args.full_loc,
    )
    baseline_wq_b = None
    baseline_row = None
    if baseline_anchor is not None:
        baseline_row = int(baseline_anchor["row"])
        baseline_wq_b = _trace_entry(
            baseline_anchor["event"].get("operator_trace") or [],
            operator_name="wq_b",
            layer_id=args.layer,
        )
    direct_classification = _direct_baseline_target_classification(
        baseline_wq_b=baseline_wq_b,
        baseline_row=-1 if baseline_row is None else baseline_row,
        target_wq_b=target_wq_b,
        target_row=int(row.get("flattened_row", 0)),
        oracle_record=wq_b_record,
    )

    return {
        "classification": direct_classification,
        "mtp_oracle_classification": _classification(wq_b_record),
        "rank": int(args.rank),
        "batch_size": int(args.batch_size),
        "schedule": [1, 2, 4, 5, 6],
        "anchor": {
            "uid": int(args.uid),
            "position": int(args.position),
            "full_loc": int(args.full_loc),
            "depth": int(args.depth),
            "event_trace_index": anchor["event"].get("trace_index"),
            "flattened_row": row.get("flattened_row"),
            "row_start": anchor["entry"].get("row_start"),
            "input_token": row.get("input_token"),
            "target_metadata": row.get("target_metadata"),
            "oracle_metadata": row.get("oracle_metadata"),
        },
        "exactness_matrix": _token_matrix(baseline, mtp),
        "env": {
            "baseline_debug": _debug_env(baseline),
            "mtp_debug": _debug_env(mtp),
            "baseline_applied": baseline.get("env", {}).get("applied_env", {}),
            "mtp_applied": mtp.get("env", {}).get("applied_env", {}),
        },
        "model_prepare_report_rank": {
            "baseline": baseline.get("rank"),
            "mtp": mtp.get("rank"),
            "baseline_q_wqb": (
                baseline.get("model_prepare_report", {})
                .get("q_wqb_bf16_weight_cache", {})
            ),
            "mtp_q_wqb": (
                mtp.get("model_prepare_report", {})
                .get("q_wqb_bf16_weight_cache", {})
            ),
            "baseline_marlin_q_wqb": (
                baseline.get("model_prepare_report", {})
                .get("dense_fp8_marlin_projection_cache", {})
                .get("q_wqb", {})
            ),
            "mtp_marlin_q_wqb": (
                mtp.get("model_prepare_report", {})
                .get("dense_fp8_marlin_projection_cache", {})
                .get("q_wqb", {})
            ),
        },
        "wq_b_operator_parity": wq_b_record,
        "q_wqb_output_operator_parity": q_wqb_output_record,
        "baseline_vs_target_wq_b": {
            "baseline_anchor": (
                {
                    "trace_index": baseline_anchor["event"].get("trace_index"),
                    "row": baseline_row,
                    "input_ids": baseline_anchor["event"].get("input_ids"),
                    "positions": baseline_anchor["event"].get("positions"),
                    "out_cache_loc": baseline_anchor["event"].get("out_cache_loc"),
                }
                if baseline_anchor is not None
                else None
            ),
            "baseline_path": baseline_wq_b.get("path") if isinstance(baseline_wq_b, dict) else None,
            "target_path": target_wq_b.get("path") if isinstance(target_wq_b, dict) else None,
            "baseline_params": (
                baseline_wq_b.get("params") if isinstance(baseline_wq_b, dict) else None
            ),
            "target_params": target_wq_b.get("params") if isinstance(target_wq_b, dict) else None,
            "baseline_row_record": _trace_row_record(
                baseline_wq_b,
                -1 if baseline_row is None else baseline_row,
            ),
            "target_row_record": _trace_row_record(
                target_wq_b,
                int(row.get("flattened_row", 0)),
            ),
            "baseline_weight_cache": _weight_cache_digest(
                baseline_wq_b.get("extra") if isinstance(baseline_wq_b, dict) else None
            ),
            "target_weight_cache": _weight_cache_digest(
                target_wq_b.get("extra") if isinstance(target_wq_b, dict) else None
            ),
            "baseline_per_row_probe": _per_row_digest(
                baseline_wq_b.get("extra") if isinstance(baseline_wq_b, dict) else None
            ),
            "target_per_row_probe": _per_row_digest(
                target_wq_b.get("extra") if isinstance(target_wq_b, dict) else None
            ),
        },
        "wq_b_input_summary": {
            "normal_metadata": (
                wq_b_record.get("input_tensor_metadata", {}).get("normal")
                if isinstance(wq_b_record, dict)
                else None
            ),
            "target_verify_metadata": (
                wq_b_record.get("input_tensor_metadata", {}).get("target_verify")
                if isinstance(wq_b_record, dict)
                else None
            ),
            "input_comparison": (
                wq_b_record.get("input_comparison") if isinstance(wq_b_record, dict) else None
            ),
        },
        "wq_b_output_summary": {
            "normal_metadata": (
                wq_b_record.get("output_tensor_metadata", {}).get("normal")
                if isinstance(wq_b_record, dict)
                else None
            ),
            "target_verify_metadata": (
                wq_b_record.get("output_tensor_metadata", {}).get("target_verify")
                if isinstance(wq_b_record, dict)
                else None
            ),
            "output_comparison": (
                wq_b_record.get("output_comparison") if isinstance(wq_b_record, dict) else None
            ),
            "normal_raw_sha256": (
                wq_b_record.get("normal_raw_sha256") if isinstance(wq_b_record, dict) else None
            ),
            "target_verify_raw_sha256": (
                wq_b_record.get("target_verify_raw_sha256")
                if isinstance(wq_b_record, dict)
                else None
            ),
        },
        "dispatch_and_cache": {
            "normal_path": (
                wq_b_record.get("normal_kernel_or_path") if isinstance(wq_b_record, dict) else None
            ),
            "target_verify_path": (
                wq_b_record.get("target_verify_kernel_or_path")
                if isinstance(wq_b_record, dict)
                else None
            ),
            "normal_params": (
                wq_b_record.get("normal_params") if isinstance(wq_b_record, dict) else None
            ),
            "target_verify_params": (
                wq_b_record.get("target_verify_params")
                if isinstance(wq_b_record, dict)
                else None
            ),
            "normal_weight_cache": _weight_cache_digest(
                wq_b_record.get("normal_extra") if isinstance(wq_b_record, dict) else None
            ),
            "target_verify_weight_cache": _weight_cache_digest(
                wq_b_record.get("target_verify_extra")
                if isinstance(wq_b_record, dict)
                else None
            ),
            "normal_per_row_probe": _per_row_digest(
                wq_b_record.get("normal_extra") if isinstance(wq_b_record, dict) else None
            ),
            "target_verify_per_row_probe": _per_row_digest(
                wq_b_record.get("target_verify_extra")
                if isinstance(wq_b_record, dict)
                else None
            ),
        },
        "trace_entries": {
            "oracle_wq_b": oracle_wq_b,
            "target_wq_b": target_wq_b,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--mtp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--uid", type=int, default=0)
    parser.add_argument("--position", type=int, default=5)
    parser.add_argument("--full-loc", type=int, default=3077)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()

    result = analyze(args)
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

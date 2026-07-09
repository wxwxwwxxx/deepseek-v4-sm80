from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _runs_by_batch(artifact: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not artifact:
        return {}
    return {int(run["batch_size"]): run for run in artifact.get("runs", [])}


def _first_token_mismatch(lhs: list[int], rhs: list[int]) -> dict[str, Any] | None:
    limit = min(len(lhs), len(rhs))
    for index in range(limit):
        if int(lhs[index]) != int(rhs[index]):
            return {
                "token_index": int(index),
                "baseline": int(lhs[index]),
                "candidate": int(rhs[index]),
            }
    if len(lhs) != len(rhs):
        return {
            "token_index": int(limit),
            "baseline": int(lhs[limit]) if limit < len(lhs) else None,
            "candidate": int(rhs[limit]) if limit < len(rhs) else None,
            "length_mismatch": True,
        }
    return None


def _exactness_matrix(
    baseline: dict[str, Any] | None,
    artifacts: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    baseline_runs = _runs_by_batch(baseline)
    matrix: dict[str, Any] = {}
    for name, artifact in artifacts.items():
        if artifact is None:
            continue
        rows = []
        for batch_size, run in sorted(_runs_by_batch(artifact).items()):
            baseline_run = baseline_runs.get(batch_size)
            diff_reqs = []
            if baseline_run is None:
                rows.append(
                    {
                        "batch_size": int(batch_size),
                        "status": "missing_baseline_batch",
                    }
                )
                continue
            baseline_tokens = baseline_run.get("token_ids", [])
            candidate_tokens = run.get("token_ids", [])
            for req_index, (lhs, rhs) in enumerate(zip(baseline_tokens, candidate_tokens)):
                mismatch = _first_token_mismatch(lhs, rhs)
                if mismatch is not None:
                    diff_reqs.append({"request_id": int(req_index), **mismatch})
            if len(candidate_tokens) != len(baseline_tokens):
                diff_reqs.append(
                    {
                        "request_id": "batch_width",
                        "baseline": len(baseline_tokens),
                        "candidate": len(candidate_tokens),
                    }
                )
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "exact": not diff_reqs,
                    "diff_reqs": diff_reqs,
                }
            )
        matrix[name] = rows
    return matrix


def _iter_stats_lists(artifact: dict[str, Any] | None, key: str):
    if not artifact:
        return
    for run in artifact.get("runs", []):
        batch_size = int(run.get("batch_size", -1))
        stats = run.get("stats_delta", {})
        values = stats.get(key, [])
        if isinstance(values, list):
            for value in values:
                yield batch_size, value


def _summarize_contract_trace(artifact: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    for batch_size, trace in _iter_stats_lists(artifact, "target_verify_contract_trace"):
        positions = [int(x) for x in trace.get("positions", [])]
        full_locs = [int(x) for x in trace.get("out_cache_loc", [])]
        rows.append(
            {
                "batch_size": int(batch_size),
                "runtime": trace.get("runtime", ""),
                "attention_mode": trace.get("attention_mode", ""),
                "kv_store_mode": trace.get("kv_store_mode", ""),
                "verify_batch_size": int(trace.get("batch_size", -1)),
                "parent_batch_size": int(trace.get("parent_batch_size", -1)),
                "speculative_num_draft_tokens": int(
                    trace.get("speculative_num_draft_tokens", -1)
                ),
                "num_tokens": int(trace.get("num_tokens", -1)),
                "positions": positions,
                "full_locs": full_locs,
                "page_offsets_256": [int(loc) % 256 for loc in full_locs],
                "position_page_offset_mismatch": [
                    {
                        "row": int(index),
                        "position": int(position),
                        "full_loc": int(full_locs[index]),
                        "page_offset": int(full_locs[index]) % 256,
                    }
                    for index, position in enumerate(positions[: len(full_locs)])
                    if int(position) != int(full_locs[index]) % 256
                ],
                "row_depths": trace.get("row_depths", []),
                "row_to_batch_index": trace.get("row_to_batch_index", []),
                "row_to_parent_batch_index": trace.get(
                    "row_to_parent_batch_index", []
                ),
                "active_row_mask": trace.get("active_row_mask", []),
                "padded_row_mask": trace.get("padded_row_mask", []),
                "seq_lens": trace.get("seq_lens", []),
                "extend_lens": trace.get("extend_lens", []),
                "c4_out_loc": trace.get("c4_out_loc", []),
                "c128_out_loc": trace.get("c128_out_loc", []),
                "c128_pending_write_commit": trace.get(
                    "c128_pending_write_commit", ""
                ),
            }
        )
    return rows


def _summarize_teacher_forced_trace(
    artifact: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []
    for batch_size, event in _iter_stats_lists(artifact, "teacher_forced_trace"):
        for entry in event.get("entries", []):
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "trace_index": int(event.get("trace_index", -1)),
                    "uid": entry.get("uid"),
                    "request_id": entry.get("batch_index"),
                    "generated_prefix_len": entry.get("generated_prefix_len"),
                    "current_target_token": entry.get("current_target_token"),
                    "expected_current_baseline_token": entry.get(
                        "expected_current_baseline_token"
                    ),
                    "current_matches_baseline": entry.get(
                        "current_matches_baseline"
                    ),
                    "forced_draft_tokens": entry.get("forced_draft_tokens", []),
                    "status": entry.get("status", ""),
                }
            )
    return rows


def _summarize_row_depth_oracle(artifact: dict[str, Any] | None) -> dict[str, Any]:
    rows = []
    failures = []
    for batch_size, event in _iter_stats_lists(artifact, "row_depth_oracle_debug"):
        for entry in event.get("entries", []):
            for row in entry.get("rows", []):
                logit_delta = row.get("logit_delta", {})
                hidden_delta = row.get("hidden_delta", {})
                record = {
                    "batch_size": int(batch_size),
                    "trace_index": int(event.get("trace_index", -1)),
                    "uid": entry.get("uid"),
                    "request_id": entry.get("batch_index"),
                    "depth": row.get("depth"),
                    "flattened_row": row.get("flattened_row"),
                    "input_token": row.get("input_token"),
                    "position": row.get("position"),
                    "target_token": row.get("target_token"),
                    "oracle_token": row.get("oracle_token"),
                    "draft_token": row.get("draft_token"),
                    "accepted": row.get("accepted"),
                    "target_top1": (
                        row.get("target_topk", {}).get("token_ids", [None])[0]
                        if row.get("target_topk", {}).get("token_ids")
                        else None
                    ),
                    "oracle_top1": (
                        row.get("oracle_topk", {}).get("token_ids", [None])[0]
                        if row.get("oracle_topk", {}).get("token_ids")
                        else None
                    ),
                    "oracle_margin": row.get("oracle_topk", {}).get(
                        "top1_top2_margin"
                    ),
                    "max_abs_logit_delta": logit_delta.get("max_abs"),
                    "mean_abs_logit_delta": logit_delta.get("mean_abs"),
                    "max_abs_hidden_delta": hidden_delta.get("max_abs"),
                    "target_metadata": row.get("target_metadata", {}),
                    "oracle_metadata": row.get("oracle_metadata", {}),
                }
                if record["target_token"] != record["oracle_token"]:
                    failures.append(record)
                rows.append(record)
    return {
        "row_count": len(rows),
        "token_mismatch_count": len(failures),
        "first_token_mismatch": failures[0] if failures else None,
        "rows": rows[:64],
    }


def _summarize_debug_trace(artifact: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    for batch_size, trace in _iter_stats_lists(artifact, "debug_trace"):
        for row in trace.get("row_depths", []):
            full_loc = row.get("full_loc")
            position = row.get("position")
            rows.append(
                {
                    "batch_size": int(batch_size),
                    "uid": trace.get("uid"),
                    "request_id": trace.get("batch_index"),
                    "committed_seq_len": trace.get("cached_len"),
                    "row_start": trace.get("row_start"),
                    "depth": row.get("depth"),
                    "flattened_row": row.get("flattened_row"),
                    "row_type": (
                        "accepted_candidate"
                        if row.get("draft_token") is not None
                        and row.get("target_token") == row.get("draft_token")
                        else (
                            "correction"
                            if trace.get("mismatch_depth") == row.get("depth")
                            else "bonus_or_candidate"
                        )
                    ),
                    "input_token": row.get("input_token"),
                    "token_scored": row.get("target_token"),
                    "draft_token": row.get("draft_token"),
                    "position": position,
                    "full_loc": full_loc,
                    "page_id": int(full_loc) // 256 if full_loc is not None else None,
                    "page_offset": int(full_loc) % 256 if full_loc is not None else None,
                    "accepted_prefix": trace.get("accepted_prefix"),
                    "mismatch_depth": trace.get("mismatch_depth"),
                    "candidate_copy_rows": trace.get("candidate_copy_rows"),
                    "copy_rows": trace.get("copy_rows"),
                    "accepted_commit_blocker": trace.get("accepted_commit_blocker"),
                }
            )
    return rows[:128]


def _summarize_canonical_replay(
    artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = []
    for batch_size, event in _iter_stats_lists(
        artifact, "canonical_replay_commit_debug"
    ):
        for replay in event.get("rows", []):
            rows.append({"batch_size": int(batch_size), **replay})
    return {
        "row_count": len(rows),
        "rows": rows[:64],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--teacher-forced")
    parser.add_argument("--canonical")
    parser.add_argument("--real-mtp")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = _load_json(args.baseline)
    teacher_forced = _load_json(args.teacher_forced)
    canonical = _load_json(args.canonical)
    real_mtp = _load_json(args.real_mtp)

    compared = {
        "teacher_forced": teacher_forced,
        "canonical_replay_commit": canonical,
        "real_mtp": real_mtp,
    }
    summary = {
        "inputs": {
            "baseline": args.baseline,
            "teacher_forced": args.teacher_forced,
            "canonical": args.canonical,
            "real_mtp": args.real_mtp,
        },
        "exactness_matrix": _exactness_matrix(baseline, compared),
        "teacher_forced_trace": _summarize_teacher_forced_trace(teacher_forced),
        "teacher_forced_row_depth_oracle": _summarize_row_depth_oracle(
            teacher_forced
        ),
        "teacher_forced_contract_trace": _summarize_contract_trace(teacher_forced),
        "teacher_forced_semantic_rows": _summarize_debug_trace(teacher_forced),
        "canonical_contract_trace": _summarize_contract_trace(canonical),
        "canonical_semantic_rows": _summarize_debug_trace(canonical),
        "canonical_replay_commit": _summarize_canonical_replay(canonical),
    }
    position_controls = []
    for key in ("teacher_forced_contract_trace", "canonical_contract_trace"):
        for trace in summary.get(key, []):
            position_controls.extend(trace.get("position_page_offset_mismatch", []))
    summary["position_page_offset_control"] = {
        "covered": bool(position_controls),
        "mismatch_examples": position_controls[:16],
    }
    _write_json(args.output, summary)
    print(json.dumps({"ok": True, "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build compact TARGET 15.1 reports from no-weight and TP8 rank payloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mixed_fairness_harness import build_report


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p / 100.0
    lo, hi = int(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def max_runs(sequence: str) -> dict[str, int]:
    result = {"prefill": 0, "decode": 0}
    last = ""
    count = 0
    for phase in sequence:
        count = count + 1 if phase == last else 1
        last = phase
        key = "prefill" if phase == "P" else "decode"
        result[key] = max(result[key], count)
    return result


def summarize_rank(path: Path) -> dict:
    payload = json.loads(path.read_text())
    repeat = payload["repeats"][0]
    trace = repeat["schedule_trace"]
    requests = repeat["requests"]
    sequence = "".join(row["phase"][0].upper() for row in trace)
    long_req = max(requests, key=lambda row: row["input_tokens"])
    decode_reqs = [row for row in requests if row["input_tokens"] < 1000]
    gaps = [
        right - left
        for req in decode_reqs
        for left, right in zip(req["token_times_s"], req["token_times_s"][1:])
    ]
    long_uid = long_req["uid"]
    long_prefill_indices = [
        i
        for i, row in enumerate(trace)
        if row["phase"] == "prefill"
        and any(req["uid"] == long_uid for req in row["reqs"])
    ]
    mixed_sequence = ""
    if long_prefill_indices and decode_reqs:
        mixed_sequence = sequence[long_prefill_indices[0] : long_prefill_indices[-1] + 1]
    long_chunks = [
        req["extend_len"]
        for row in trace
        if row["phase"] == "prefill"
        for req in row["reqs"]
        if req["uid"] == long_uid
    ]
    graph_replay = sum(int(row["graph_replay_delta"]) for row in trace)
    graph_eager = sum(int(row["graph_eager_delta"]) for row in trace)
    return {
        "artifact": str(path),
        "phase_sequence": sequence,
        "mixed_phase_sequence": mixed_sequence,
        "phase_counts": {
            "prefill": sequence.count("P"),
            "decode": sequence.count("D"),
        },
        "max_consecutive_all": max_runs(sequence),
        "max_consecutive_mixed_span": max_runs(mixed_sequence),
        "long_prefill_chunk_sizes": long_chunks,
        "long_prefill_progress_tokens": sum(long_chunks),
        "long_prefill_ttft_s": long_req["ttft_s"],
        "long_request_latency_s": long_req["latency_s"],
        "long_request_output_tokens": long_req["output_tokens"],
        "long_request_output_token_ids": long_req["output_token_ids"],
        "decode_inter_token_gap_s": {
            "count": len(gaps),
            "p50": percentile(gaps, 50),
            "p95": percentile(gaps, 95),
            "p99": percentile(gaps, 99),
            "max": percentile(gaps, 100),
        },
        "graph": {"replay": graph_replay, "eager": graph_eager},
        "scheduler_prepare_s": sum(float(row["prepare_s"]) for row in trace),
        "forward_s": sum(float(row["forward_s"]) for row in trace),
        "prefix_cache_integrity_snapshot": payload["prefix_cache_metrics"],
        "memory_after_case": payload["memory_after_case"],
        "rank_error": payload.get("error"),
    }


def rank0_files(root: Path) -> list[Path]:
    return sorted((root / "reports").glob("*.rank0.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-b", type=Path, required=True)
    parser.add_argument("--selected-long", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    harness = build_report()
    baseline = {
        "harness": harness["harness"],
        "release_prefill_budget": harness["release_prefill_budget"],
        "production_prefill_first": harness["policies"]["production_prefill_first"],
        "rejected_permanent_decode_first": harness["policies"][
            "rejected_permanent_decode_first"
        ],
    }
    candidates = {
        "bounded_candidate_limit": 2,
        "candidate_a": {
            "contract": {"max_decode": 4, "mixed_prefill_budget": 2048},
            "no_weight": harness["policies"]["candidate_a_d4_p2048"],
        },
        "candidate_b": {
            "contract": {"max_decode": 8, "mixed_prefill_budget": 4096},
            "no_weight": harness["policies"]["candidate_b_d8_p4096"],
        },
        "tp8_short": {
            "legacy": [summarize_rank(path) for path in rank0_files(args.legacy)],
            "candidate_a": [summarize_rank(path) for path in rank0_files(args.candidate_a)],
            "candidate_b": [summarize_rank(path) for path in rank0_files(args.candidate_b)],
        },
        "selected": "candidate_a",
        "selection_reason": (
            "Both candidates satisfy finite progress; candidate A has materially lower "
            "M=1 and M=4 worst decode gaps than candidate B."
        ),
    }
    mixed = {
        "selected_policy": "candidate_a_d4_p2048",
        "short_probe": candidates["tp8_short"],
        "selected_128k_smoke_and_same_process_controls": [
            summarize_rank(path) for path in rank0_files(args.selected_long)
        ],
        "fresh_isolated_controls": [
            summarize_rank(path) for path in rank0_files(args.controls)
        ],
    }
    for name, data in (
        ("baseline_starvation.json", baseline),
        ("candidate_results.json", candidates),
        ("mixed_load_results.json", mixed),
    ):
        (args.output_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

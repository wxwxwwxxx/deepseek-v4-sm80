from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "summaries" / "target0777_summary.json"


REPORTS = {
    "repeat2_4096x1024_baseline": RAW
    / "repeat2_4096x1024_baseline_np128/reports/000_decode_throughput_bs8__dsv4_sm80_a100_victory.json",
    "repeat2_4096x1024_candidate": RAW
    / "repeat2_4096x1024_candidate_np128/reports/000_decode_throughput_bs8__dsv4_sm80_a100_victory_densefp8marlinproj.json",
    "timing_4096x128_baseline": RAW
    / "timing_4096x128_baseline_np128/reports/000_decode_throughput_bs8__dsv4_sm80_a100_victory.json",
    "timing_4096x128_candidate": RAW
    / "timing_4096x128_candidate_np128/reports/000_decode_throughput_bs8__dsv4_sm80_a100_victory_densefp8marlinproj.json",
    "timing_4096x1024_baseline": RAW
    / "timing_4096x1024_baseline_np128/reports/000_decode_throughput_bs8__dsv4_sm80_a100_victory.json",
    "timing_4096x1024_candidate": RAW
    / "timing_4096x1024_candidate_np128/reports/000_decode_throughput_bs8__dsv4_sm80_a100_victory_densefp8marlinproj.json",
}


OWNER_LABELS = {
    "q_wqb": {
        "baseline": "dsv4.owner.attn.q_wqb.bf16_cache_local_total",
        "candidate": "dsv4.owner.attn.q_wqb.dense_fp8_marlin_local_total",
        "baseline_gemm": "dsv4.owner.attn.q_wqb.bf16_cache_linear",
        "candidate_gemm": "dsv4.owner.attn.q_wqb.dense_fp8_marlin_apply",
    },
    "wo_b": {
        "baseline": "dsv4.owner.attn.wo_b.bf16_cache_local_total",
        "candidate": "dsv4.owner.attn.wo_b.dense_fp8_marlin_local_total",
        "baseline_gemm": "dsv4.owner.attn.wo_b.bf16_cache_linear",
        "candidate_gemm": "dsv4.owner.attn.wo_b.dense_fp8_marlin_apply",
    },
    "shared_down": {
        "baseline": "dsv4.owner.shared_down.bf16_cache_local_total",
        "candidate": "dsv4.owner.shared_down.dense_fp8_marlin_local_total",
        "baseline_gemm": "dsv4.owner.shared_down.bf16_cache_linear",
        "candidate_gemm": "dsv4.owner.shared_down.dense_fp8_marlin_apply",
    },
    "wo_b_all_reduce": {
        "baseline": "dsv4.owner.attn.wo_b.row_parallel_all_reduce",
        "candidate": "dsv4.owner.attn.wo_b.row_parallel_all_reduce",
    },
    "moe_reduce_once": {
        "baseline": "dsv4.owner.moe.reduce_once_all_reduce",
        "candidate": "dsv4.owner.moe.reduce_once_all_reduce",
    },
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    m = report["metrics"]
    p = m["phase_totals"]
    return {
        "output_tok_s": m["end_to_end_output_tokens_per_s"],
        "decode_tok_s": m["decode_tokens_per_s"],
        "ttft_s": m["ttft_s_mean"],
        "prefill_tok_s": m["prefill_tokens_per_s"],
        "elapsed_s": m["elapsed_s"],
        "prefill_forward_s": p["prefill_forward_s"],
        "decode_forward_s": p["decode_forward_s"],
        "prefill_prepare_s": p["prefill_prepare_s"],
        "decode_prepare_s": p["decode_prepare_s"],
        "prepare_s": p["prefill_prepare_s"] + p["decode_prepare_s"],
        "load_init_s": report["load_init"]["seconds_max"],
        "replay_count": report["config"]["graph_runner"]["replay_count"],
        "eager_decode": report["config"]["graph_runner"]["eager_decode_count"],
        "peak_allocated_bytes": m["peak_gpu_memory_allocated_bytes"],
    }


def repeat_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for repeat in report["repeats"]:
        p = repeat["phase_totals"]
        ttft = [req["ttft_s"] for req in repeat["requests"] if req["ttft_s"] is not None]
        rows.append(
            {
                "repeat_index": repeat["repeat_index"],
                "output_tok_s": repeat["actual_output_tokens"] / repeat["elapsed_s"],
                "elapsed_s": repeat["elapsed_s"],
                "ttft_s": sum(ttft) / len(ttft),
                "prefill_forward_s": p["prefill_forward_s"],
                "decode_forward_s": p["decode_forward_s"],
                "prefill_prepare_s": p["prefill_prepare_s"],
                "decode_prepare_s": p["decode_prepare_s"],
                "prepare_s": p["prefill_prepare_s"] + p["decode_prepare_s"],
            }
        )
    return rows


def delta(candidate: float, baseline: float) -> dict[str, float]:
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": candidate - baseline,
        "delta_pct": 100.0 * (candidate / baseline - 1.0),
    }


def owner_timing(report: dict[str, Any], label: str) -> float | None:
    shapes = report["owner_timing"]["cuda"]["by_label_shape"]
    values = []
    for key, stats in shapes.items():
        if not key.startswith(label + "|captured=1|shape=[4, "):
            continue
        value = stats.get("max_rank_total_ms")
        if value:
            values.append(float(value))
    return sum(values) if values else None


def owner_table(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    replay = int(cand["config"]["graph_runner"]["replay_count"])
    rows = {}
    for owner, labels in OWNER_LABELS.items():
        b = owner_timing(base, labels["baseline"])
        c = owner_timing(cand, labels["candidate"])
        row = {
            "baseline_ms_per_replay": b,
            "candidate_ms_per_replay": c,
            "delta_ms_per_replay": None if b is None or c is None else c - b,
            "estimated_delta_s_for_replay_count": None
            if b is None or c is None
            else (c - b) * replay / 1000.0,
        }
        if "baseline_gemm" in labels:
            bg = owner_timing(base, labels["baseline_gemm"])
            cg = owner_timing(cand, labels["candidate_gemm"])
            row.update(
                {
                    "baseline_gemm_ms_per_replay": bg,
                    "candidate_gemm_ms_per_replay": cg,
                    "gemm_delta_ms_per_replay": None if bg is None or cg is None else cg - bg,
                    "estimated_gemm_delta_s_for_replay_count": None
                    if bg is None or cg is None
                    else (cg - bg) * replay / 1000.0,
                }
            )
        rows[owner] = row
    return {"replay_count": replay, "rows": rows}


def host_prepare(report: dict[str, Any]) -> dict[str, float]:
    host = report["owner_timing"]["host"]["by_label"]
    return {
        key: float(stats["max_rank_total_ms"])
        for key, stats in host.items()
        if key.startswith("dsv4.prepare.prefill.")
        or key.startswith("dsv4.prepare.decode.")
        or key == "dsv4.prepare.dense_fp8_marlin.total"
    }


def dense_prepare_cuda(report: dict[str, Any]) -> dict[str, float]:
    cuda = report["owner_timing"]["cuda"]["by_label"]
    return {
        key: float(stats["max_rank_total_ms"])
        for key, stats in cuda.items()
        if key.startswith("dsv4.prepare.dense_fp8_marlin.")
    }


def layout_counters(report: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in report["per_rank"]:
        for counter in payload["owner_timing"].get("counters", []):
            label = counter["label"]
            if "dense_fp8_marlin" not in label:
                continue
            if "reshape" not in label and "contiguous" not in label:
                continue
            counts[label] = counts.get(label, 0) + int(counter["count"])
    return dict(sorted(counts.items()))


def first_owner_order(report: dict[str, Any]) -> list[str]:
    order: list[str] = []
    for sample in report["owner_timing"]["rank0"]["cuda_samples"]:
        if not sample.get("captured") or sample.get("elapsed_ms") is None:
            continue
        label = sample["label"]
        metadata = sample.get("metadata", {})
        shape = (metadata.get("input") or metadata.get("tensor") or {}).get("shape")
        if not shape or shape[0] != 4:
            continue
        if not any(
            token in label
            for token in ("attn.q_wqb", "attn.wo_b", "shared_down", "reduce_once_all_reduce")
        ):
            continue
        if label not in order:
            order.append(label)
        if len(order) >= 16:
            break
    return order


def communication(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_count": report["communication_counters"]["total_count"],
        "total_bytes": report["communication_counters"]["total_bytes"],
        "by_label": report["communication_counters"]["by_label"],
    }


def main() -> None:
    reports = {name: load(path) for name, path in REPORTS.items()}
    rep_base = reports["repeat2_4096x1024_baseline"]
    rep_cand = reports["repeat2_4096x1024_candidate"]
    long_base = reports["timing_4096x1024_baseline"]
    long_cand = reports["timing_4096x1024_candidate"]
    short_base = reports["timing_4096x128_baseline"]
    short_cand = reports["timing_4096x128_candidate"]

    payload = {
        "macro_repeat2_4096x1024": {
            "baseline": metrics(rep_base),
            "candidate": metrics(rep_cand),
            "deltas": {
                "output_tok_s": delta(
                    metrics(rep_cand)["output_tok_s"], metrics(rep_base)["output_tok_s"]
                ),
                "decode_tok_s": delta(
                    metrics(rep_cand)["decode_tok_s"], metrics(rep_base)["decode_tok_s"]
                ),
                "elapsed_s": delta(metrics(rep_cand)["elapsed_s"], metrics(rep_base)["elapsed_s"]),
                "ttft_s": delta(metrics(rep_cand)["ttft_s"], metrics(rep_base)["ttft_s"]),
                "decode_forward_s": delta(
                    metrics(rep_cand)["decode_forward_s"], metrics(rep_base)["decode_forward_s"]
                ),
                "prefill_forward_s": delta(
                    metrics(rep_cand)["prefill_forward_s"], metrics(rep_base)["prefill_forward_s"]
                ),
                "prepare_s": delta(metrics(rep_cand)["prepare_s"], metrics(rep_base)["prepare_s"]),
            },
            "baseline_repeats": repeat_rows(rep_base),
            "candidate_repeats": repeat_rows(rep_cand),
        },
        "owner_timing_4096x1024": owner_table(long_base, long_cand),
        "owner_timing_4096x128": owner_table(short_base, short_cand),
        "prepare_breakdown_4096x1024": {
            "baseline_host_ms": host_prepare(long_base),
            "candidate_host_ms": host_prepare(long_cand),
            "candidate_dense_marlin_cuda_ms": dense_prepare_cuda(long_cand),
        },
        "layout_counters_4096x1024_candidate": layout_counters(long_cand),
        "owner_order_rank0_4096x1024": {
            "baseline": first_owner_order(long_base),
            "candidate": first_owner_order(long_cand),
        },
        "communication_4096x1024_timing_runs": {
            "baseline": communication(long_base),
            "candidate": communication(long_cand),
        },
        "raw_reports": {name: str(path) for name, path in REPORTS.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

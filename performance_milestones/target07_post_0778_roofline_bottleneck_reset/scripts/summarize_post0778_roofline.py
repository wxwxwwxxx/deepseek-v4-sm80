#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


VARIANT = "dsv4_sm80_a100_victory"
SCENARIO = "decode_throughput_bs8"
TP_SIZE = 8
A100_BF16_TFLOPS_PER_GPU = 312.0
A100_HBM_TBPS_PER_GPU = 2.039
TP8_BF16_TFLOPS = A100_BF16_TFLOPS_PER_GPU * TP_SIZE
TP8_HBM_TBPS = A100_HBM_TBPS_PER_GPU * TP_SIZE


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return None if not filtered else float(statistics.mean(filtered))


def stdev(values: list[float]) -> float:
    return 0.0 if len(values) <= 1 else float(statistics.stdev(values))


def stats(values: list[float], *, higher_is_better: bool) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "best": None, "worst": None, "std": None, "cv": None}
    avg = float(statistics.mean(values))
    sd = stdev(values)
    return {
        "mean": avg,
        "median": float(statistics.median(values)),
        "best": float(max(values) if higher_is_better else min(values)),
        "worst": float(min(values) if higher_is_better else max(values)),
        "std": sd,
        "cv": None if avg == 0 else float(sd / abs(avg)),
    }


def report_path(raw_dir: Path, shape: str) -> Path:
    matches = sorted((raw_dir / f"{shape}_victory" / "reports").glob(f"*{SCENARIO}__{VARIANT}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one report for {shape}, found {matches}")
    return matches[0]


def timing_report_path(raw_dir: Path) -> Path | None:
    matches = sorted((raw_dir / "timing_4096x128_owner_victory" / "reports").glob(f"*{SCENARIO}__{VARIANT}.json"))
    return matches[0] if matches else None


def repeat_row(repeat: dict[str, Any]) -> dict[str, Any]:
    phase = repeat.get("phase_totals", {})
    elapsed_s = float(repeat.get("elapsed_s") or 0.0)
    output_tokens = int(repeat.get("actual_output_tokens") or 0)
    decode_tokens = int(phase.get("decode_tokens") or 0)
    decode_forward_s = float(phase.get("decode_forward_s") or 0.0)
    requests = repeat.get("requests", [])
    return {
        "repeat_index": int(repeat.get("repeat_index", 0)),
        "elapsed_s": elapsed_s,
        "output_tokens_per_s": None if elapsed_s <= 0 else output_tokens / elapsed_s,
        "decode_tokens_per_s": None if decode_forward_s <= 0 else decode_tokens / decode_forward_s,
        "ttft_s_mean": mean(request.get("ttft_s") for request in requests),
        "prefill_forward_s": float(phase.get("prefill_forward_s") or 0.0),
        "decode_forward_s": decode_forward_s,
        "prefill_prepare_s": float(phase.get("prefill_prepare_s") or 0.0),
        "decode_prepare_s": float(phase.get("decode_prepare_s") or 0.0),
        "actual_output_tokens": output_tokens,
        "target_output_tokens": int(repeat.get("target_output_tokens") or 0),
    }


def shape_summary(report: dict[str, Any], shape: str, path: Path) -> dict[str, Any]:
    repeats = [repeat_row(repeat) for repeat in report.get("repeats", [])]
    metric_keys = (
        "output_tokens_per_s",
        "decode_tokens_per_s",
        "ttft_s_mean",
        "prefill_forward_s",
        "decode_forward_s",
        "elapsed_s",
    )
    higher = {
        "output_tokens_per_s": True,
        "decode_tokens_per_s": True,
        "ttft_s_mean": False,
        "prefill_forward_s": False,
        "decode_forward_s": False,
        "elapsed_s": False,
    }
    graph = report.get("config", {}).get("graph_runner", {})
    metrics = report.get("metrics", {})
    phase = metrics.get("phase_totals", {})
    return {
        "report_path": str(path),
        "status": report.get("status"),
        "shape": shape,
        "warmup": report.get("per_rank", [{}])[0].get("warmup", {}),
        "repeats": repeats,
        "stats": {
            key: stats(
                [float(row[key]) for row in repeats if row.get(key) is not None],
                higher_is_better=higher[key],
            )
            for key in metric_keys
        },
        "metrics": metrics,
        "phase_totals": phase,
        "graph": {
            "enabled": bool(graph.get("enabled")),
            "captured_bs": graph.get("captured_bs", []),
            "replay_count": int(graph.get("replay_count") or 0),
            "greedy_sample_replay_count": int(graph.get("greedy_sample_replay_count") or 0),
            "eager_decode_count": int(graph.get("eager_decode_count") or 0),
            "replay_input_copy_bytes": int(graph.get("replay_input_copy_bytes") or 0),
        },
        "communication": report.get("communication_counters", {}),
        "kernel_counters": report.get("kernel_counters", {}),
        "model_prepare_report_rank0": report.get("config", {}).get("model_prepare_report_rank0", {}),
        "schedule_summary": report.get("schedule_summary", {}),
    }


def active_flops_bounds(report: dict[str, Any], *, split: str) -> dict[str, float]:
    cfg = {
        "layers": 43,
        "hidden": 4096,
        "q_lora": 1024,
        "heads": 64,
        "head_dim": 512,
        "o_groups": 8,
        "o_lora": 1024,
        "moe_intermediate": 2048,
        "experts_per_token": 6,
        "shared_experts": 1,
        "vocab": 129280,
        "window": 128,
        "index_topk": 512,
    }
    metrics = report["metrics"]
    phase = metrics["phase_totals"]
    if split == "prefill":
        layer_tokens = int(phase.get("prefill_input_tokens") or 0)
        lm_head_tokens = max(0, int(report["scenario"]["batch_size"]) * int(report["scenario"]["repeats"]))
    elif split == "decode":
        layer_tokens = int(phase.get("decode_tokens") or 0)
        lm_head_tokens = max(0, int(metrics.get("actual_output_tokens") or 0) - int(report["scenario"]["batch_size"]) * int(report["scenario"]["repeats"]))
    else:
        layer_tokens = int(phase.get("prefill_input_tokens") or 0) + int(phase.get("decode_tokens") or 0)
        lm_head_tokens = int(metrics.get("actual_output_tokens") or 0)

    h = cfg["hidden"]
    q = cfg["q_lora"]
    heads = cfg["heads"]
    hd = cfg["head_dim"]
    o_groups = cfg["o_groups"]
    o_rank = cfg["o_lora"]
    inter = cfg["moe_intermediate"]
    attention_projection_per_layer = (
        2 * h * q
        + 2 * q * heads * hd
        + 2 * h * hd
        + 2 * (heads * hd // o_groups) * (o_groups * o_rank)
        + 2 * (o_groups * o_rank) * h
    )
    shared_per_layer = 2 * h * (2 * inter * cfg["shared_experts"]) + 2 * (inter * cfg["shared_experts"]) * h
    routed_per_layer = cfg["experts_per_token"] * (2 * h * (2 * inter) + 2 * inter * h)
    elementwise_per_layer = 0.02 * (attention_projection_per_layer + shared_per_layer + routed_per_layer)
    lower_per_token = cfg["layers"] * (
        attention_projection_per_layer + shared_per_layer + routed_per_layer + elementwise_per_layer
    )
    sparse_window = cfg["window"] + cfg["index_topk"]
    sparse_attention_per_layer = 4 * sparse_window * heads * hd
    upper_per_token = lower_per_token + cfg["layers"] * sparse_attention_per_layer
    lm_head = 2 * h * cfg["vocab"] * lm_head_tokens
    return {
        "lower": float(layer_tokens * lower_per_token + lm_head),
        "upper": float(layer_tokens * upper_per_token + lm_head),
        "layer_tokens": float(layer_tokens),
        "lm_head_tokens": float(lm_head_tokens),
    }


def efficiency_table(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    phase = metrics["phase_totals"]
    rows = {}
    for name, seconds in (
        ("prefill", float(phase.get("prefill_forward_s") or 0.0)),
        ("decode", float(phase.get("decode_forward_s") or 0.0)),
        ("whole", float(metrics.get("elapsed_s") or 0.0)),
    ):
        flops = active_flops_bounds(report, split=name)
        lower_tflops_s = None if seconds <= 0 else flops["lower"] / seconds / 1e12
        upper_tflops_s = None if seconds <= 0 else flops["upper"] / seconds / 1e12
        rows[name] = {
            "seconds": seconds,
            "active_flops_lower": flops["lower"],
            "active_flops_upper": flops["upper"],
            "active_tflops_s_lower": lower_tflops_s,
            "active_tflops_s_upper": upper_tflops_s,
            "mfu_like_pct_lower": None if lower_tflops_s is None else 100.0 * lower_tflops_s / TP8_BF16_TFLOPS,
            "mfu_like_pct_upper": None if upper_tflops_s is None else 100.0 * upper_tflops_s / TP8_BF16_TFLOPS,
            "layer_tokens": flops["layer_tokens"],
            "lm_head_tokens": flops["lm_head_tokens"],
        }
    rows["throughput"] = {
        "output_tokens_per_s": metrics.get("end_to_end_output_tokens_per_s"),
        "total_tokens_per_s": metrics.get("end_to_end_total_tokens_per_s"),
        "prefill_tokens_per_s": metrics.get("prefill_tokens_per_s"),
        "decode_tokens_per_s": metrics.get("decode_tokens_per_s"),
    }
    return rows


def owner_timing_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {"available": False}
    by_label = report.get("owner_timing", {}).get("cuda", {}).get("by_label", {})
    top = []
    for label, values in sorted(
        by_label.items(),
        key=lambda item: float(item[1].get("max_rank_total_ms") or 0.0),
        reverse=True,
    )[:20]:
        top.append(
            {
                "label": label,
                "max_rank_total_ms": float(values.get("max_rank_total_ms") or 0.0),
                "sum_rank_total_ms": float(values.get("sum_rank_total_ms") or 0.0),
                "count": int(values.get("count") or 0),
                "captured_count": int(values.get("captured_count") or 0),
            }
        )
    return {
        "available": True,
        "report_path": report.get("report_path"),
        "metrics": report.get("metrics", {}),
        "graph": report.get("config", {}).get("graph_runner", {}),
        "top_cuda_labels": top,
    }


def fixed_capacity(shape: dict[str, Any]) -> dict[str, Any]:
    metrics = shape["metrics"]
    page_size = 256
    logical_pages = 128
    logical_tokens = logical_pages * page_size
    scenario = {"prompt_len": 4096, "decode_len": 1024, "batch_size": 4}
    live_tokens = scenario["batch_size"] * (scenario["prompt_len"] + scenario["decode_len"])
    kv_bytes_rank = int(metrics.get("kv_cache_memory_bytes_per_rank_max") or 0)
    return {
        "page_size": page_size,
        "num_pages": logical_pages,
        "logical_kv_token_capacity_per_rank": logical_tokens,
        "kv_cache_bytes_per_rank_including_dummy": kv_bytes_rank,
        "kv_bytes_per_logical_token_rank": None if logical_tokens == 0 else kv_bytes_rank / logical_tokens,
        "long_shape_live_tokens": live_tokens,
        "long_shape_capacity_utilization": live_tokens / logical_tokens,
        "max_equivalent_4096_prompts": logical_tokens // 4096,
        "max_equivalent_4096_plus_1024_requests": logical_tokens // (4096 + 1024),
    }


def capacity_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    data = read_json(path)
    rank0 = data.get("rank0", {})
    cap = rank0.get("capacity", {})
    graph = rank0.get("graph_runner", {})
    records = rank0.get("memory_sync_records", [])
    dense_savings_bytes = 806_961_152
    bytes_per_page = int(cap.get("kv_cache_bytes_per_page_per_rank") or 0)
    extra_pages = None if bytes_per_page <= 0 else dense_savings_bytes / bytes_per_page
    return {
        "available": True,
        "path": str(path),
        "status": data.get("status"),
        "rank0_capacity": cap,
        "memory_sync_records": records,
        "memory_after_init": rank0.get("memory_after_init", {}),
        "graph_capture_error": graph.get("error"),
        "dense_fp8_memory_savings_bytes_per_rank": dense_savings_bytes,
        "dense_fp8_equivalent_extra_pages": extra_pages,
        "dense_fp8_equivalent_extra_tokens": None if extra_pages is None else extra_pages * int(cap.get("page_size") or 256),
        "dense_fp8_equivalent_4096_prompts": None if extra_pages is None else (extra_pages * int(cap.get("page_size") or 256)) / 4096,
        "dense_fp8_equivalent_4096_plus_1024_requests": None if extra_pages is None else (extra_pages * int(cap.get("page_size") or 256)) / 5120,
    }


def communication_summary(shape: dict[str, Any]) -> dict[str, Any]:
    comm = shape.get("communication", {})
    metrics = shape.get("metrics", {})
    total_bytes = int(comm.get("total_bytes") or 0)
    output_tokens = int(metrics.get("actual_output_tokens") or 0)
    elapsed = float(metrics.get("elapsed_s") or 0.0)
    decode = float(metrics.get("phase_totals", {}).get("decode_forward_s") or 0.0)
    return {
        "total_count": int(comm.get("total_count") or 0),
        "total_bytes": total_bytes,
        "bytes_per_output_token": None if output_tokens <= 0 else total_bytes / output_tokens,
        "aggregate_gb_s_elapsed": None if elapsed <= 0 else total_bytes / elapsed / 1e9,
        "aggregate_gb_s_decode_forward": None if decode <= 0 else total_bytes / decode / 1e9,
        "by_label": comm.get("by_label", {}),
        "entries": comm.get("entries", []),
    }


def memory_bandwidth_summary(shape: dict[str, Any]) -> dict[str, Any]:
    metrics = shape["metrics"]
    phase = metrics["phase_totals"]
    prepare = shape.get("model_prepare_report_rank0", {})
    cache_bytes_rank = int(prepare.get("projection_bf16_weight_cache_total", {}).get("total_bytes") or 0)
    decode_replays = int(shape["graph"].get("replay_count") or 0)
    decode_s = float(phase.get("decode_forward_s") or 0.0)
    aggregate_cache_read_bytes = cache_bytes_rank * decode_replays * TP_SIZE
    hbm_bytes_s = TP8_HBM_TBPS * 1e12
    kv_bytes_rank = int(metrics.get("kv_cache_memory_bytes_per_rank_max") or 0)
    token_count = int(phase.get("prefill_input_tokens") or 0) + int(phase.get("decode_tokens") or 0)
    kv_min_bytes = kv_bytes_rank / (129 * 256) * token_count * TP_SIZE if kv_bytes_rank else 0
    return {
        "projection_cache_decode_min_read_bytes": aggregate_cache_read_bytes,
        "projection_cache_decode_estimated_tb_s": None if decode_s <= 0 else aggregate_cache_read_bytes / decode_s / 1e12,
        "projection_cache_decode_mbu_pct": None if decode_s <= 0 else 100.0 * aggregate_cache_read_bytes / (decode_s * hbm_bytes_s),
        "kv_min_stream_bytes": kv_min_bytes,
        "kv_min_stream_mbu_pct_whole": None
        if float(metrics.get("elapsed_s") or 0.0) <= 0
        else 100.0 * kv_min_bytes / (float(metrics.get("elapsed_s")) * hbm_bytes_s),
        "replay_metadata_copy_bytes": int(shape["graph"].get("replay_input_copy_bytes") or 0),
    }


def opportunity_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    long_shape = summary["shapes"]["4096x1024"]
    metrics = long_shape["metrics"]
    elapsed = float(metrics["elapsed_s"])
    phase = metrics["phase_totals"]
    prefill = float(phase["prefill_forward_s"])
    decode_prepare = float(phase["decode_prepare_s"])
    prefill_prepare = float(phase["prefill_prepare_s"])
    comm = summary["communication"]["4096x1024"]
    return [
        {
            "target_surface": "TARGET 08 radix prefix cache",
            "current_time_or_percent": f"prefill_forward={prefill:.4f}s aggregate ({100.0 * prefill / elapsed:.2f}% elapsed); TTFT mean={metrics['ttft_s_mean']:.4f}s",
            "estimated_max_possible_speedup": "workload dependent; shared 4096-token prefixes can skip most prefill for cache hits",
            "likely_bound_type": "feature/cache reuse, not kernel roofline",
            "confidence": "high for shared-prefix workloads",
            "next_step": "TARGET 08 radix prefix cache",
        },
        {
            "target_surface": "decode/prefill prepare metadata",
            "current_time_or_percent": f"prepare={prefill_prepare + decode_prepare:.4f}s aggregate ({100.0 * (prefill_prepare + decode_prepare) / elapsed:.2f}% elapsed)",
            "estimated_max_possible_speedup": "likely below 2% E2E without a broader graph/metadata redesign",
            "likely_bound_type": "latency/launch/CPU-GPU staging",
            "confidence": "medium",
            "next_step": "no exact TARGET 07 action",
        },
        {
            "target_surface": "TP communication",
            "current_time_or_percent": f"{comm['total_count']} collectives, {comm['total_bytes'] / 1e9:.1f} GB counter bytes",
            "estimated_max_possible_speedup": "unknown; needs NCCL timeline before a separate target",
            "likely_bound_type": "communication-bound / latency-bound",
            "confidence": "medium",
            "next_step": "no action",
        },
        {
            "target_surface": "dense FP8 Marlin projection memory mode",
            "current_time_or_percent": "807 MB/rank saved in TARGET 07.78, speed neutral",
            "estimated_max_possible_speedup": "none for default speed path",
            "likely_bound_type": "memory/capacity",
            "confidence": "high",
            "next_step": "memory/capacity target only if max context becomes the limiter",
        },
    ]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# TARGET 07.79 Summary Snippets", ""]
    for shape, item in summary["shapes"].items():
        lines.append(f"## {shape} repeats")
        lines.append("| Repeat | Output tok/s | Decode tok/s | TTFT s | Prefill fwd s | Decode fwd s | Elapsed s |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in item["repeats"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["repeat_index"]),
                        fmt(row["output_tokens_per_s"]),
                        fmt(row["decode_tokens_per_s"]),
                        fmt(row["ttft_s_mean"]),
                        fmt(row["prefill_forward_s"]),
                        fmt(row["decode_forward_s"]),
                        fmt(row["elapsed_s"]),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("| Metric | Mean | Median | Best | Worst | Std | CV |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for metric, values in item["stats"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{metric}`",
                        fmt(values["mean"]),
                        fmt(values["median"]),
                        fmt(values["best"]),
                        fmt(values["worst"]),
                        fmt(values["std"]),
                        fmt(values["cv"], 6),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.append("## Whole-run MFU-like")
    lines.append("| Shape | Split | Seconds | Active FLOPs lower | Active FLOPs upper | TFLOP/s lower | TFLOP/s upper | MFU lower | MFU upper |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for shape, eff in summary["efficiency"].items():
        for split in ("prefill", "decode", "whole"):
            row = eff[split]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{shape}`",
                        split,
                        fmt(row["seconds"]),
                        fmt(row["active_flops_lower"] / 1e15, 3) + " PF",
                        fmt(row["active_flops_upper"] / 1e15, 3) + " PF",
                        fmt(row["active_tflops_s_lower"]),
                        fmt(row["active_tflops_s_upper"]),
                        fmt(row["mfu_like_pct_lower"], 3) + "%",
                        fmt(row["mfu_like_pct_upper"], 3) + "%",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def build_summary(milestone_dir: Path) -> dict[str, Any]:
    raw_dir = milestone_dir / "raw"
    reports = {}
    shapes = {}
    for shape in ("4096x1024", "4096x128"):
        path = report_path(raw_dir, shape)
        report = read_json(path)
        reports[shape] = report
        shapes[shape] = shape_summary(report, shape, path)
    timing_path = timing_report_path(raw_dir)
    timing_report = read_json(timing_path) if timing_path else None
    smoke_path = raw_dir / f"smoke_{VARIANT}" / "text_smoke.json"
    smoke = read_json(smoke_path) if smoke_path.exists() else {}
    summary = {
        "variant": VARIANT,
        "roofline_assumptions": {
            "gpu": "A100-SXM4-80GB",
            "bf16_tensor_core_peak_tflops_per_gpu": A100_BF16_TFLOPS_PER_GPU,
            "hbm_tbps_per_gpu": A100_HBM_TBPS_PER_GPU,
            "tp8_bf16_tensor_core_peak_tflops": TP8_BF16_TFLOPS,
            "tp8_hbm_tbps": TP8_HBM_TBPS,
            "bf16_tc_hbm_crossover_flops_per_byte": A100_BF16_TFLOPS_PER_GPU / A100_HBM_TBPS_PER_GPU,
        },
        "smoke": {
            "path": str(smoke_path),
            "status": smoke.get("status"),
            "variant_status": (smoke.get("variants") or [{}])[0].get("status") if smoke else None,
            "graph": smoke.get("config", {}).get("graph_runner", {}),
            "prompts": smoke.get("prompts", []),
        },
        "shapes": shapes,
        "efficiency": {shape: efficiency_table(report) for shape, report in reports.items()},
        "communication": {shape: communication_summary(item) for shape, item in shapes.items()},
        "memory_bandwidth": {shape: memory_bandwidth_summary(item) for shape, item in shapes.items()},
        "fixed_capacity": fixed_capacity(shapes["4096x1024"]),
        "automatic_capacity": capacity_summary(raw_dir / "capacity_auto_victory" / "capacity_probe.json"),
        "owner_timing": owner_timing_summary(timing_report),
    }
    summary["opportunities"] = opportunity_table(summary)
    summary["recommendation"] = {
        "next_action": "start TARGET 08 radix prefix cache",
        "reason": (
            "No remaining exact-route bucket has fresh evidence for a clean >=2% E2E win without "
            "broader precision, communication, or graph-risk work; shared-prefix reuse can skip "
            "the measured multi-second prefill surface."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = build_summary(args.milestone_dir)
    write_json(args.milestone_dir / "summaries" / "post0778_roofline_summary.json", summary)
    md = render_markdown(summary)
    (args.milestone_dir / "summaries" / "post0778_roofline_summary.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

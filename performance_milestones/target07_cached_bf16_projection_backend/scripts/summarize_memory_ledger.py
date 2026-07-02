#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _q_wqb_prepare(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    prepare = config.get("model_prepare_report_rank0") or {}
    if not prepare:
        load_init = report.get("load_init", {})
        per_rank = load_init.get("seconds_per_rank") or []
        if per_rank:
            prepare = per_rank[0].get("model_prepare_report") or {}
    return prepare.get("q_wqb_bf16_weight_cache") or {}


def _graph_status(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("config", {}).get("graph_runner", {}) or {}


def _memory_row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    return {
        "output_tok_s": metrics.get("end_to_end_output_tokens_per_s"),
        "decode_tok_s": metrics.get("decode_tokens_per_s"),
        "peak_allocated": metrics.get("peak_gpu_memory_allocated_bytes"),
        "peak_reserved": metrics.get("peak_gpu_memory_reserved_bytes"),
        "kv_cache_memory": metrics.get("kv_cache_memory_bytes_per_rank_max"),
    }


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _compute(qwqb: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    prepare = _q_wqb_prepare(qwqb)
    extra_bytes = int(prepare.get("total_bytes") or 0)
    config = qwqb.get("config", {})
    num_pages = int(config.get("num_pages") or 0)
    page_size = int(config.get("page_size") or 0)
    metrics = qwqb.get("metrics") if isinstance(qwqb.get("metrics"), dict) else qwqb
    kv_memory = int(metrics.get("kv_cache_memory_bytes_per_rank_max") or 0)
    bytes_per_kv_token = None
    kv_tokens_lost = None
    kv_pages_lost = None
    if num_pages > 0 and page_size > 0 and kv_memory > 0:
        bytes_per_kv_token = kv_memory / float(num_pages * page_size)
        kv_tokens_lost = extra_bytes / bytes_per_kv_token if bytes_per_kv_token else None
        kv_pages_lost = kv_tokens_lost / page_size if kv_tokens_lost is not None else None

    peak_alloc_delta = None
    peak_reserved_delta = None
    if baseline is not None:
        q_mem = _memory_row(qwqb)
        b_mem = _memory_row(baseline)
        if q_mem["peak_allocated"] is not None and b_mem["peak_allocated"] is not None:
            peak_alloc_delta = int(q_mem["peak_allocated"]) - int(b_mem["peak_allocated"])
        if q_mem["peak_reserved"] is not None and b_mem["peak_reserved"] is not None:
            peak_reserved_delta = int(q_mem["peak_reserved"]) - int(b_mem["peak_reserved"])

    return {
        "prepare": prepare,
        "extra_cached_weight_bytes_per_rank": extra_bytes,
        "extra_cached_weight_gib_per_rank": extra_bytes / float(1 << 30),
        "num_pages": num_pages,
        "page_size": page_size,
        "kv_cache_memory_bytes_per_rank_max": kv_memory,
        "bytes_per_kv_token_per_rank": bytes_per_kv_token,
        "kv_tokens_lost_per_rank": kv_tokens_lost,
        "kv_pages_lost_per_rank": kv_pages_lost,
        "peak_allocated_delta_vs_baseline": peak_alloc_delta,
        "peak_reserved_delta_vs_baseline": peak_reserved_delta,
        "qwqb_metrics": _memory_row(qwqb),
        "baseline_metrics": _memory_row(baseline) if baseline is not None else None,
        "graph_status": _graph_status(qwqb),
    }


def _render_md(summary: dict[str, Any], *, qwqb_path: Path, baseline_path: Path | None) -> str:
    prepare = summary["prepare"]
    lines: list[str] = []
    lines.append("# q_wqb Cached BF16 Memory Ledger")
    lines.append("")
    lines.append(f"- q_wqb report: `{qwqb_path}`")
    if baseline_path is not None:
        lines.append(f"- baseline report: `{baseline_path}`")
    lines.append("")
    lines.append("| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    entries = prepare.get("entries") or []
    shape = entries[0].get("shape") if entries else None
    lines.append(
        "| `attn.q_wqb` | {layers} | `{shape}` | {bytes} | `{gib:.4f}` | `{tokens}` | `{pages}` |".format(
            layers=int(prepare.get("layers_cached") or 0),
            shape=shape or "n/a",
            bytes=_fmt_int(summary["extra_cached_weight_bytes_per_rank"]),
            gib=float(summary["extra_cached_weight_gib_per_rank"]),
            tokens=_fmt_float(summary["kv_tokens_lost_per_rank"], 2),
            pages=_fmt_float(summary["kv_pages_lost_per_rank"], 2),
        )
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| bytes/token/rank | `{_fmt_float(summary['bytes_per_kv_token_per_rank'], 2)}` |")
    lines.append(f"| page size | `{summary['page_size']}` |")
    lines.append(f"| num pages | `{summary['num_pages']}` |")
    lines.append(f"| KV cache bytes/rank max | `{_fmt_int(summary['kv_cache_memory_bytes_per_rank_max'])}` |")
    lines.append(f"| peak allocated delta vs baseline | `{_fmt_int(summary['peak_allocated_delta_vs_baseline'])}` |")
    lines.append(f"| peak reserved delta vs baseline | `{_fmt_int(summary['peak_reserved_delta_vs_baseline'])}` |")
    graph = summary["graph_status"]
    lines.append(f"| graph replay count | `{graph.get('replay_count', 'n/a')}` |")
    lines.append(f"| eager decode count | `{graph.get('eager_decode_count', 'n/a')}` |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwqb-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_cached_bf16_projection_backend/summaries/qwqb_memory_ledger.json"
        ),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_cached_bf16_projection_backend/summaries/qwqb_memory_ledger.md"
        ),
    )
    args = parser.parse_args()
    qwqb = _load_json(args.qwqb_report)
    assert qwqb is not None
    baseline = _load_json(args.baseline_report)
    summary = _compute(qwqb, baseline)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.md_out.write_text(
        _render_md(summary, qwqb_path=args.qwqb_report, baseline_path=args.baseline_report) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()

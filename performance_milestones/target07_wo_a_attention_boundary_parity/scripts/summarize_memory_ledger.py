#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CACHE_KEYS: tuple[tuple[str, str], ...] = (
    ("attn.q_wqb", "q_wqb_bf16_weight_cache"),
    ("attn.wo_b", "wo_b_bf16_weight_cache"),
    ("indexer.wq_b", "indexer_wq_b_bf16_weight_cache"),
    ("attn.wo_a", "wo_a_bf16_bmm_cache"),
)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    prepare = config.get("model_prepare_report_rank0") or {}
    if not prepare:
        load_init = report.get("load_init", {})
        per_rank = load_init.get("seconds_per_rank") or []
        if per_rank:
            prepare = per_rank[0].get("model_prepare_report") or {}
    return prepare


def _metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    return {
        "output_tok_s": metrics.get("end_to_end_output_tokens_per_s"),
        "decode_tok_s": metrics.get("decode_tokens_per_s"),
        "peak_allocated": metrics.get("peak_gpu_memory_allocated_bytes"),
        "peak_reserved": metrics.get("peak_gpu_memory_reserved_bytes"),
        "kv_cache_memory": metrics.get("kv_cache_memory_bytes_per_rank_max"),
    }


def _graph_status(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("config", {}).get("graph_runner", {}) or {}


def _shape(prepare: dict[str, Any]) -> Any:
    entries = prepare.get("entries") or []
    return entries[0].get("shape") if entries else None


def _source_weight_shape(prepare: dict[str, Any]) -> Any:
    entries = prepare.get("entries") or []
    return entries[0].get("source_weight_shape") if entries else None


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _memory_delta(report: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, int | None]:
    if baseline is None:
        return {"peak_allocated_delta": None, "peak_reserved_delta": None}
    current = _metrics(report)
    base = _metrics(baseline)
    alloc_delta = None
    reserved_delta = None
    if current.get("peak_allocated") is not None and base.get("peak_allocated") is not None:
        alloc_delta = int(current["peak_allocated"]) - int(base["peak_allocated"])
    if current.get("peak_reserved") is not None and base.get("peak_reserved") is not None:
        reserved_delta = int(current["peak_reserved"]) - int(base["peak_reserved"])
    return {"peak_allocated_delta": alloc_delta, "peak_reserved_delta": reserved_delta}


def _owner_row(
    owner: str,
    prepare: dict[str, Any],
    *,
    bytes_per_kv_token: float | None,
    page_size: int,
) -> dict[str, Any]:
    bytes_per_rank = int(prepare.get("total_bytes") or 0)
    kv_tokens = None
    kv_pages = None
    if bytes_per_kv_token:
        kv_tokens = bytes_per_rank / bytes_per_kv_token
        kv_pages = kv_tokens / page_size if page_size > 0 else None
    return {
        "owner": owner,
        "enabled": bool(prepare.get("enabled")),
        "toggle": prepare.get("toggle"),
        "layers_cached": int(prepare.get("layers_cached") or 0),
        "shape_per_local_rank": _shape(prepare),
        "source_weight_shape_per_local_rank": _source_weight_shape(prepare),
        "extra_bytes_per_rank": bytes_per_rank,
        "extra_gib_per_rank": bytes_per_rank / float(1 << 30),
        "kv_tokens_lost_per_rank": kv_tokens,
        "kv_pages_lost_per_rank": kv_pages,
    }


def _compute(report: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    config = report.get("config", {})
    num_pages = int(config.get("num_pages") or 0)
    page_size = int(config.get("page_size") or 0)
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    kv_memory = int(metrics.get("kv_cache_memory_bytes_per_rank_max") or 0)
    bytes_per_kv_token = None
    if num_pages > 0 and page_size > 0 and kv_memory > 0:
        bytes_per_kv_token = kv_memory / float(num_pages * page_size)

    prepare = _prepare(report)
    rows = [
        _owner_row(
            owner,
            prepare.get(key) or {},
            bytes_per_kv_token=bytes_per_kv_token,
            page_size=page_size,
        )
        for owner, key in CACHE_KEYS
    ]
    total_bytes = sum(row["extra_bytes_per_rank"] for row in rows)
    wo_a_bytes = next(
        (row["extra_bytes_per_rank"] for row in rows if row["owner"] == "attn.wo_a"),
        0,
    )
    total_tokens = total_bytes / bytes_per_kv_token if bytes_per_kv_token else None
    total_pages = total_tokens / page_size if total_tokens is not None and page_size > 0 else None
    wo_a_tokens = wo_a_bytes / bytes_per_kv_token if bytes_per_kv_token else None
    wo_a_pages = wo_a_tokens / page_size if wo_a_tokens is not None and page_size > 0 else None
    return {
        "owners": rows,
        "total": {
            "owner": "total_cached_bf16_projection",
            "extra_bytes_per_rank": total_bytes,
            "extra_gib_per_rank": total_bytes / float(1 << 30),
            "kv_tokens_lost_per_rank": total_tokens,
            "kv_pages_lost_per_rank": total_pages,
        },
        "wo_a_incremental": {
            "owner": "attn.wo_a",
            "extra_bytes_per_rank": wo_a_bytes,
            "extra_gib_per_rank": wo_a_bytes / float(1 << 30),
            "kv_tokens_lost_per_rank": wo_a_tokens,
            "kv_pages_lost_per_rank": wo_a_pages,
        },
        "num_pages": num_pages,
        "page_size": page_size,
        "kv_cache_memory_bytes_per_rank_max": kv_memory,
        "bytes_per_kv_token_per_rank": bytes_per_kv_token,
        "memory_delta_vs_baseline": _memory_delta(report, baseline),
        "metrics": _metrics(report),
        "baseline_metrics": _metrics(baseline),
        "graph_status": _graph_status(report),
        "prepare_report": prepare,
    }


def _render_md(summary: dict[str, Any], *, report_path: Path, baseline_path: Path | None) -> str:
    lines: list[str] = []
    lines.append("# wo_a BF16 BMM Cache Memory Ledger")
    lines.append("")
    lines.append(f"- wo_a report: `{report_path}`")
    if baseline_path is not None:
        lines.append(f"- baseline report: `{baseline_path}`")
    lines.append("")
    lines.append("| Cached owner | Enabled | Layers | Cache shape/rank | Source shape/rank | Extra bytes/rank | Extra GiB/rank | KV tokens/rank | KV pages/rank |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for row in [*summary["owners"], summary["total"], summary["wo_a_incremental"]]:
        lines.append(
            "| `{owner}` | `{enabled}` | {layers} | `{shape}` | `{source}` | {bytes} | `{gib:.4f}` | `{tokens}` | `{pages}` |".format(
                owner=row["owner"],
                enabled=row.get("enabled", "n/a"),
                layers=int(row.get("layers_cached") or 0),
                shape=row.get("shape_per_local_rank") or "mixed",
                source=row.get("source_weight_shape_per_local_rank") or "mixed",
                bytes=_fmt_int(row["extra_bytes_per_rank"]),
                gib=float(row["extra_gib_per_rank"]),
                tokens=_fmt_float(row["kv_tokens_lost_per_rank"], 2),
                pages=_fmt_float(row["kv_pages_lost_per_rank"], 2),
            )
        )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| bytes/token/rank | `{_fmt_float(summary['bytes_per_kv_token_per_rank'], 2)}` |")
    lines.append(f"| page size | `{summary['page_size']}` |")
    lines.append(f"| num pages | `{summary['num_pages']}` |")
    lines.append(f"| KV cache bytes/rank max | `{_fmt_int(summary['kv_cache_memory_bytes_per_rank_max'])}` |")
    delta = summary["memory_delta_vs_baseline"]
    lines.append(f"| peak allocated delta vs baseline | `{_fmt_int(delta['peak_allocated_delta'])}` |")
    lines.append(f"| peak reserved delta vs baseline | `{_fmt_int(delta['peak_reserved_delta'])}` |")
    graph = summary["graph_status"]
    lines.append(f"| graph replay count | `{graph.get('replay_count', 'n/a')}` |")
    lines.append(f"| eager decode count | `{graph.get('eager_decode_count', 'n/a')}` |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_wo_a_attention_boundary_parity/summaries/wo_a_memory_ledger.json"
        ),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_wo_a_attention_boundary_parity/summaries/wo_a_memory_ledger.md"
        ),
    )
    args = parser.parse_args()
    report = _load_json(args.report)
    assert report is not None
    baseline = _load_json(args.baseline_report)
    summary = _compute(report, baseline)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.md_out.write_text(
        _render_md(summary, report_path=args.report, baseline_path=args.baseline_report) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()

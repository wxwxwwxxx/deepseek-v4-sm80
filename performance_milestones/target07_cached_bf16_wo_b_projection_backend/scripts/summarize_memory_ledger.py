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


def _prepare(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    prepare = config.get("model_prepare_report_rank0") or {}
    if not prepare:
        load_init = report.get("load_init", {})
        per_rank = load_init.get("seconds_per_rank") or []
        if per_rank:
            prepare = per_rank[0].get("model_prepare_report") or {}
    return prepare


def _cache_prepare(report: dict[str, Any], key: str) -> dict[str, Any]:
    return _prepare(report).get(key) or {}


def _graph_status(report: dict[str, Any]) -> dict[str, Any]:
    return report.get("config", {}).get("graph_runner", {}) or {}


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


def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _shape(prepare: dict[str, Any]) -> Any:
    entries = prepare.get("entries") or []
    return entries[0].get("shape") if entries else None


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
    kv_tokens_lost = None
    kv_pages_lost = None
    if bytes_per_kv_token:
        kv_tokens_lost = bytes_per_rank / bytes_per_kv_token
        kv_pages_lost = kv_tokens_lost / page_size if page_size > 0 else None
    return {
        "owner": owner,
        "layers_cached": int(prepare.get("layers_cached") or 0),
        "shape_per_local_rank": _shape(prepare),
        "extra_bytes_per_rank": bytes_per_rank,
        "extra_gib_per_rank": bytes_per_rank / float(1 << 30),
        "kv_tokens_lost_per_rank": kv_tokens_lost,
        "kv_pages_lost_per_rank": kv_pages_lost,
    }


def _compute(
    qwqb_wob: dict[str, Any],
    qwqb_baseline: dict[str, Any] | None,
    exact_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    config = qwqb_wob.get("config", {})
    num_pages = int(config.get("num_pages") or 0)
    page_size = int(config.get("page_size") or 0)
    metrics = qwqb_wob.get("metrics") if isinstance(qwqb_wob.get("metrics"), dict) else qwqb_wob
    kv_memory = int(metrics.get("kv_cache_memory_bytes_per_rank_max") or 0)
    bytes_per_kv_token = None
    if num_pages > 0 and page_size > 0 and kv_memory > 0:
        bytes_per_kv_token = kv_memory / float(num_pages * page_size)

    q_prepare = _cache_prepare(qwqb_wob, "q_wqb_bf16_weight_cache")
    wo_prepare = _cache_prepare(qwqb_wob, "wo_b_bf16_weight_cache")
    q_row = _owner_row(
        "attn.q_wqb",
        q_prepare,
        bytes_per_kv_token=bytes_per_kv_token,
        page_size=page_size,
    )
    wo_row = _owner_row(
        "attn.wo_b",
        wo_prepare,
        bytes_per_kv_token=bytes_per_kv_token,
        page_size=page_size,
    )
    total_bytes = q_row["extra_bytes_per_rank"] + wo_row["extra_bytes_per_rank"]
    total_tokens = total_bytes / bytes_per_kv_token if bytes_per_kv_token else None
    total_pages = total_tokens / page_size if total_tokens is not None and page_size > 0 else None

    return {
        "owners": [q_row, wo_row],
        "total": {
            "owner": "total",
            "layers_cached": q_row["layers_cached"] + wo_row["layers_cached"],
            "shape_per_local_rank": "mixed",
            "extra_bytes_per_rank": total_bytes,
            "extra_gib_per_rank": total_bytes / float(1 << 30),
            "kv_tokens_lost_per_rank": total_tokens,
            "kv_pages_lost_per_rank": total_pages,
        },
        "num_pages": num_pages,
        "page_size": page_size,
        "kv_cache_memory_bytes_per_rank_max": kv_memory,
        "bytes_per_kv_token_per_rank": bytes_per_kv_token,
        "memory_delta_vs_qwqb_baseline": _memory_delta(qwqb_wob, qwqb_baseline),
        "memory_delta_vs_exact_baseline": _memory_delta(qwqb_wob, exact_baseline),
        "qwqb_wob_metrics": _metrics(qwqb_wob),
        "qwqb_baseline_metrics": _metrics(qwqb_baseline),
        "exact_baseline_metrics": _metrics(exact_baseline),
        "graph_status": _graph_status(qwqb_wob),
        "prepare_report": _prepare(qwqb_wob),
    }


def _render_md(
    summary: dict[str, Any],
    *,
    qwqb_wob_path: Path,
    qwqb_baseline_path: Path | None,
    exact_baseline_path: Path | None,
) -> str:
    lines: list[str] = []
    lines.append("# q_wqb + wo_b Cached BF16 Memory Ledger")
    lines.append("")
    lines.append(f"- q_wqb + wo_b report: `{qwqb_wob_path}`")
    if qwqb_baseline_path is not None:
        lines.append(f"- q_wqb-only baseline report: `{qwqb_baseline_path}`")
    if exact_baseline_path is not None:
        lines.append(f"- exact baseline report: `{exact_baseline_path}`")
    lines.append("")
    lines.append("| Cached owner | Layers cached | Shape per local rank | Extra bytes/rank | Extra GiB/rank | KV tokens lost/rank | KV pages lost/rank |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for row in [*summary["owners"], summary["total"]]:
        lines.append(
            "| `{owner}` | {layers} | `{shape}` | {bytes} | `{gib:.4f}` | `{tokens}` | `{pages}` |".format(
                owner=row["owner"],
                layers=int(row["layers_cached"]),
                shape=row["shape_per_local_rank"] or "n/a",
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
    q_delta = summary["memory_delta_vs_qwqb_baseline"]
    e_delta = summary["memory_delta_vs_exact_baseline"]
    lines.append(f"| peak allocated delta vs q_wqb-only baseline | `{_fmt_int(q_delta['peak_allocated_delta'])}` |")
    lines.append(f"| peak reserved delta vs q_wqb-only baseline | `{_fmt_int(q_delta['peak_reserved_delta'])}` |")
    lines.append(f"| peak allocated delta vs exact baseline | `{_fmt_int(e_delta['peak_allocated_delta'])}` |")
    lines.append(f"| peak reserved delta vs exact baseline | `{_fmt_int(e_delta['peak_reserved_delta'])}` |")
    graph = summary["graph_status"]
    lines.append(f"| graph replay count | `{graph.get('replay_count', 'n/a')}` |")
    lines.append(f"| eager decode count | `{graph.get('eager_decode_count', 'n/a')}` |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwqb-wob-report", type=Path, required=True)
    parser.add_argument("--qwqb-baseline-report", type=Path)
    parser.add_argument("--exact-baseline-report", type=Path)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_cached_bf16_wo_b_projection_backend/summaries/qwqb_wob_memory_ledger.json"
        ),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path(
            "performance_milestones/target07_cached_bf16_wo_b_projection_backend/summaries/qwqb_wob_memory_ledger.md"
        ),
    )
    args = parser.parse_args()
    qwqb_wob = _load_json(args.qwqb_wob_report)
    assert qwqb_wob is not None
    qwqb_baseline = _load_json(args.qwqb_baseline_report)
    exact_baseline = _load_json(args.exact_baseline_report)
    summary = _compute(qwqb_wob, qwqb_baseline, exact_baseline)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.md_out.write_text(
        _render_md(
            summary,
            qwqb_wob_path=args.qwqb_wob_report,
            qwqb_baseline_path=args.qwqb_baseline_report,
            exact_baseline_path=args.exact_baseline_report,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()

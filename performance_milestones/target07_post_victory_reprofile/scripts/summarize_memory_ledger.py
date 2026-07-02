#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

CACHE_KEYS: tuple[tuple[str, str, str], ...] = (
    ("attn.q_wqb", "q_wqb_bf16_weight_cache", "prebuilt before graph capture"),
    ("attn.wo_b", "wo_b_bf16_weight_cache", "prebuilt before graph capture"),
    ("indexer.wq_b", "indexer_wq_b_bf16_weight_cache", "prebuilt before graph capture"),
    ("attn.wo_a", "wo_a_bf16_bmm_cache", "prebuilt before graph capture"),
)


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def prepare_report(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config", {})
    prepare = config.get("model_prepare_report_rank0") or {}
    if not prepare:
        load_init = report.get("load_init", {})
        per_rank = load_init.get("seconds_per_rank") or []
        if per_rank:
            prepare = per_rank[0].get("model_prepare_report") or {}
    return prepare


def metrics(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    values = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    return {
        "output_tok_s": values.get("end_to_end_output_tokens_per_s"),
        "decode_tok_s": values.get("decode_tokens_per_s"),
        "peak_allocated": values.get("peak_gpu_memory_allocated_bytes"),
        "peak_reserved": values.get("peak_gpu_memory_reserved_bytes"),
        "kv_cache_memory": values.get("kv_cache_memory_bytes_per_rank_max"),
    }


def shape_of(prepare: dict[str, Any]) -> Any:
    entries = prepare.get("entries") or []
    return entries[0].get("shape") if entries else None


def source_shape_of(prepare: dict[str, Any]) -> Any:
    entries = prepare.get("entries") or []
    return entries[0].get("source_weight_shape") if entries else None


def fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def memory_delta(report: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, int | None]:
    if baseline is None:
        return {"peak_allocated_delta": None, "peak_reserved_delta": None}
    current = metrics(report)
    base = metrics(baseline)
    alloc_delta = None
    reserved_delta = None
    if current.get("peak_allocated") is not None and base.get("peak_allocated") is not None:
        alloc_delta = int(current["peak_allocated"]) - int(base["peak_allocated"])
    if current.get("peak_reserved") is not None and base.get("peak_reserved") is not None:
        reserved_delta = int(current["peak_reserved"]) - int(base["peak_reserved"])
    return {"peak_allocated_delta": alloc_delta, "peak_reserved_delta": reserved_delta}


def kv_equiv(bytes_per_rank: int, bytes_per_kv_token: float | None, page_size: int) -> tuple[float | None, float | None]:
    if not bytes_per_kv_token:
        return None, None
    tokens = bytes_per_rank / bytes_per_kv_token
    pages = tokens / page_size if page_size > 0 else None
    return tokens, pages


def owner_row(
    owner: str,
    prepare: dict[str, Any],
    *,
    bytes_per_kv_token: float | None,
    page_size: int,
    lifecycle: str,
) -> dict[str, Any]:
    bytes_per_rank = int(prepare.get("total_bytes") or 0)
    tokens, pages = kv_equiv(bytes_per_rank, bytes_per_kv_token, page_size)
    entries = prepare.get("entries") or []
    dtype = entries[0].get("dtype") if entries else "n/a"
    return {
        "owner": owner,
        "enabled": bool(prepare.get("enabled")),
        "toggle": prepare.get("toggle"),
        "layers_cached": int(prepare.get("layers_cached") or 0),
        "shape_per_local_rank": shape_of(prepare),
        "source_weight_shape_per_local_rank": source_shape_of(prepare),
        "dtype": dtype,
        "extra_bytes_per_rank": bytes_per_rank,
        "extra_gib_per_rank": bytes_per_rank / float(1 << 30),
        "kv_tokens_lost_per_rank": tokens,
        "kv_pages_lost_per_rank": pages,
        "lifecycle": lifecycle,
        "decode_allocation": False,
        "ownership_note": "owned by model module cache attribute",
    }


def fp8_indexer_cache_row(
    report: dict[str, Any],
    *,
    bytes_per_kv_token: float | None,
) -> dict[str, Any]:
    config = report.get("config", {})
    page_size = int(config.get("page_size") or 0)
    num_pages = int(config.get("num_pages") or 0)
    model_path = report.get("model_path") or "/models/DeepSeek-V4-Flash"
    config_path = Path(model_path) / "config.json"
    hf_config = json.loads(config_path.read_text())
    num_layers = int(hf_config.get("num_hidden_layers") or 0)
    ratios = hf_config.get("compress_ratios") or [0] * num_layers
    if len(ratios) < num_layers:
        ratios = ratios + [0] * (num_layers - len(ratios))
    c4_layers = sum(r == 4 for r in ratios[:num_layers])
    index_head_dim = int(hf_config.get("index_head_dim") or hf_config.get("head_dim") or 0)
    c4_slots = math.ceil((num_pages * page_size) / 4)
    c4_page_size = max(page_size // 4, 1)
    c4_fp8_pages = math.ceil(c4_slots / c4_page_size)
    bytes_per_rank = int(c4_layers * c4_fp8_pages * c4_page_size * (index_head_dim + 4))
    tokens, pages = kv_equiv(bytes_per_rank, bytes_per_kv_token, page_size)
    return {
        "owner": "indexer.fp8_paged_cache",
        "enabled": "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE"
        in report.get("variant", {}).get("active_dsv4_toggles", []),
        "toggle": "MINISGL_DSV4_SM80_INDEXER_FP8_CACHE",
        "layers_cached": c4_layers,
        "shape_per_local_rank": [c4_layers, c4_fp8_pages, c4_page_size * (index_head_dim + 4)],
        "source_weight_shape_per_local_rank": "runtime C4 indexer activations",
        "dtype": "torch.uint8",
        "extra_bytes_per_rank": bytes_per_rank,
        "extra_gib_per_rank": bytes_per_rank / float(1 << 30),
        "kv_tokens_lost_per_rank": tokens,
        "kv_pages_lost_per_rank": pages,
        "lifecycle": "allocated with KV cache pool before decode; populated during prefill/decode store",
        "decode_allocation": False,
        "ownership_note": "currently owned by DeepSeekV4KVCache; included in kv_cache_memory_bytes_per_rank_max",
    }


def compute(report: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    config = report.get("config", {})
    num_pages = int(config.get("num_pages") or 0)
    page_size = int(config.get("page_size") or 0)
    values = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    kv_memory = int(values.get("kv_cache_memory_bytes_per_rank_max") or 0)
    bytes_per_kv_token = None
    if num_pages > 0 and page_size > 0 and kv_memory > 0:
        bytes_per_kv_token = kv_memory / float(num_pages * page_size)

    prepare = prepare_report(report)
    rows = [
        owner_row(
            owner,
            prepare.get(key) or {},
            bytes_per_kv_token=bytes_per_kv_token,
            page_size=page_size,
            lifecycle=lifecycle,
        )
        for owner, key, lifecycle in CACHE_KEYS
    ]
    rows.append(fp8_indexer_cache_row(report, bytes_per_kv_token=bytes_per_kv_token))
    rows.append(
        {
            "owner": "moe_v2_workspace",
            "enabled": False,
            "toggle": "MINISGL_DSV4_SM80_MOE_V2",
            "layers_cached": 0,
            "shape_per_local_rank": "lazy reusable buffers",
            "source_weight_shape_per_local_rank": "n/a",
            "dtype": "mixed",
            "extra_bytes_per_rank": 0,
            "extra_gib_per_rank": 0.0,
            "kv_tokens_lost_per_rank": 0.0,
            "kv_pages_lost_per_rank": 0.0,
            "lifecycle": "not materialized in current Marlin WNA16 MoE backend",
            "decode_allocation": False,
            "ownership_note": "ad hoc DSV4MoEWorkspace exists for grouped backend; inactive here",
        }
    )

    bf16_total_bytes = int((prepare.get("projection_bf16_weight_cache_total") or {}).get("total_bytes") or 0)
    bf16_tokens, bf16_pages = kv_equiv(bf16_total_bytes, bytes_per_kv_token, page_size)
    all_extra_bytes = sum(int(row["extra_bytes_per_rank"]) for row in rows)
    all_tokens, all_pages = kv_equiv(all_extra_bytes, bytes_per_kv_token, page_size)
    return {
        "owners": rows,
        "totals": {
            "cached_bf16_projection": {
                "extra_bytes_per_rank": bf16_total_bytes,
                "extra_gib_per_rank": bf16_total_bytes / float(1 << 30),
                "kv_tokens_lost_per_rank": bf16_tokens,
                "kv_pages_lost_per_rank": bf16_pages,
            },
            "listed_extra_cache_and_workspace": {
                "extra_bytes_per_rank": all_extra_bytes,
                "extra_gib_per_rank": all_extra_bytes / float(1 << 30),
                "kv_tokens_lost_per_rank": all_tokens,
                "kv_pages_lost_per_rank": all_pages,
            },
        },
        "num_pages": num_pages,
        "page_size": page_size,
        "kv_cache_memory_bytes_per_rank_max": kv_memory,
        "bytes_per_kv_token_per_rank": bytes_per_kv_token,
        "memory_delta_vs_baseline": memory_delta(report, baseline),
        "metrics": metrics(report),
        "baseline_metrics": metrics(baseline),
        "graph_status": report.get("config", {}).get("graph_runner", {}) or {},
        "prepare_report": prepare,
    }


def render_md(summary: dict[str, Any], *, report_path: Path, baseline_path: Path | None) -> str:
    lines: list[str] = []
    lines.append("# Post-Victory Cache/Workspace Memory Ledger")
    lines.append("")
    lines.append(f"- report: `{report_path}`")
    if baseline_path is not None:
        lines.append(f"- baseline report: `{baseline_path}`")
    lines.append("")
    lines.append("| Owner | Enabled | Layers | Shape/rank | Dtype | Bytes/rank | GiB/rank | KV tokens/rank | KV pages/rank | Lifecycle | Ownership |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in summary["owners"]:
        lines.append(
            "| `{owner}` | `{enabled}` | {layers} | `{shape}` | `{dtype}` | {bytes} | `{gib:.4f}` | `{tokens}` | `{pages}` | {lifecycle} | {ownership} |".format(
                owner=row["owner"],
                enabled=row.get("enabled", "n/a"),
                layers=int(row.get("layers_cached") or 0),
                shape=row.get("shape_per_local_rank") or "mixed",
                dtype=row.get("dtype") or "n/a",
                bytes=fmt_int(row["extra_bytes_per_rank"]),
                gib=float(row["extra_gib_per_rank"]),
                tokens=fmt_float(row["kv_tokens_lost_per_rank"], 2),
                pages=fmt_float(row["kv_pages_lost_per_rank"], 2),
                lifecycle=row["lifecycle"],
                ownership=row["ownership_note"],
            )
        )
    lines.append("")
    lines.append("| Total | Bytes/rank | GiB/rank | KV tokens/rank | KV pages/rank |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, total in summary["totals"].items():
        lines.append(
            "| `{}` | {} | `{:.4f}` | `{}` | `{}` |".format(
                name,
                fmt_int(total["extra_bytes_per_rank"]),
                float(total["extra_gib_per_rank"]),
                fmt_float(total["kv_tokens_lost_per_rank"], 2),
                fmt_float(total["kv_pages_lost_per_rank"], 2),
            )
        )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| bytes/token/rank | `{fmt_float(summary['bytes_per_kv_token_per_rank'], 2)}` |")
    lines.append(f"| page size | `{summary['page_size']}` |")
    lines.append(f"| num pages | `{summary['num_pages']}` |")
    lines.append(f"| KV cache bytes/rank max | `{fmt_int(summary['kv_cache_memory_bytes_per_rank_max'])}` |")
    delta = summary["memory_delta_vs_baseline"]
    lines.append(f"| peak allocated delta vs baseline | `{fmt_int(delta['peak_allocated_delta'])}` |")
    lines.append(f"| peak reserved delta vs baseline | `{fmt_int(delta['peak_reserved_delta'])}` |")
    graph = summary["graph_status"]
    lines.append(f"| graph replay count | `{graph.get('replay_count', 'n/a')}` |")
    lines.append(f"| eager decode count | `{graph.get('eager_decode_count', 'n/a')}` |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    report = load_json(args.report)
    assert report is not None
    baseline = load_json(args.baseline_report)
    summary = compute(report, baseline)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(
        render_md(summary, report_path=args.report, baseline_path=args.baseline_report) + "\n"
    )


if __name__ == "__main__":
    main()

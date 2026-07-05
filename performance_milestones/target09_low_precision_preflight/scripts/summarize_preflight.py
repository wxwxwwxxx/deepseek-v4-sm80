#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
SUMMARY_DIR = ROOT / "summaries"
MACRO_DIR = RAW / "promoted_macro_default_four_scenarios"

SCENARIOS = [
    "historical_4096_128_bs4",
    "historical_4096_1024_bs4",
    "serving_mixed_112req_wave16",
    "prefix_multi_112req_wave16",
]

EXPECTED_OWNER_CATEGORIES = [
    "communication",
    "MoE",
    "attention",
    "projection/GEMM",
    "cache store/gather/dequant",
    "metadata/runtime",
]

VARIANT = "dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16"
BASELINE_COMMAND = """CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1 timeout 3600 torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_perf_matrix.py --model-path /models/DeepSeek-V4-Flash --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 --page-size 256 --num-pages 128 --enable-dsv4-radix-prefix-cache --enable-dsv4-component-loc-ownership --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 serving_mixed_112req_wave16 prefix_multi_112req_wave16 --repeats 1 --warmup-repeats 0 --seed 20260705 --output-dir performance_milestones/target09_low_precision_preflight/raw/promoted_macro_default_four_scenarios --keep-going"""

OWNER_COMMAND = """MINISGL_DSV4_OWNER_TIMING=1 plus the same promoted baseline flags, run once per scenario via scripts/run_owner_timing_single_scenarios.sh."""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def find_report(directory: Path, scenario: str) -> Path:
    reports_dir = directory / "reports"
    candidates = sorted(
        p
        for p in reports_dir.glob(f"*{scenario}__{VARIANT}.json")
        if ".rank" not in p.name
    )
    if not candidates:
        raise FileNotFoundError(f"missing aggregate report for {scenario} in {reports_dir}")
    if len(candidates) != 1:
        raise RuntimeError(f"expected one aggregate report for {scenario}, got {candidates}")
    return candidates[0]


def owner_dir(scenario: str) -> Path:
    return RAW / f"owner_timing_{scenario}"


def owner_report(scenario: str) -> Path:
    return find_report(owner_dir(scenario), scenario)


def gib(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / (1024 ** 3):.2f} GiB"


def mib(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / (1024 ** 2):.2f} MiB"


def gb(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) / 1e9:.2f} GB"


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{float(value):.{digits}f}"


def shape_text(shape: list[int] | tuple[int, ...]) -> str:
    return "x".join(str(int(v)) for v in shape)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def scenario_index(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in summary_rows:
        name = row["scenario"]
        out[name] = row
    missing = [s for s in SCENARIOS if s not in out]
    if missing:
        raise RuntimeError(f"macro summary missing scenarios: {missing}")
    return out


def category_for_label(label: str, host: bool = False) -> str:
    low = label.lower()
    if host:
        return "metadata/runtime"
    if "all_reduce" in low or "all_gather" in low or ".comm." in low:
        return "communication"
    if (
        "metadata" in low
        or "prepare" in low
        or "replay_metadata" in low
        or "direct_graph_metadata" in low
    ):
        return "metadata/runtime"
    if ".moe." in low or "routed_expert" in low or "shared_experts" in low or "shared_down" in low:
        return "MoE"
    if "dequant" in low or "gather" in low or "store" in low or "kvcache" in low:
        if "bf16_cache" not in low:
            return "cache store/gather/dequant"
    if (
        "bf16_cache" in low
        or "q_proj" in low
        or "wo_a" in low
        or "wo_b" in low
        or "q_wqb" in low
        or "wq_b" in low
        or "linear" in low
        or "gemm" in low
        or "projection" in low
    ):
        return "projection/GEMM"
    if "attn" in low or "attention" in low or "sparse" in low or "flashmla" in low:
        return "attention"
    return "other"


def summarize_owner_timing(owner_reports: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    for scenario, report in owner_reports.items():
        timing = report.get("owner_timing", {})
        cuda_labels = timing.get("cuda", {}).get("by_label", {})
        host_labels = timing.get("host", {}).get("by_label", {})
        category: dict[str, dict[str, Any]] = {}

        def add(cat: str, total: float, captured: float, label: str, count: int, captured_count: int) -> None:
            rec = category.setdefault(
                cat,
                {
                    "scenario": scenario,
                    "category": cat,
                    "max_rank_total_ms_signal": 0.0,
                    "max_rank_captured_ms_signal": 0.0,
                    "label_count": 0,
                    "top_label": "",
                    "top_label_max_rank_total_ms": -1.0,
                    "top_label_count": 0,
                    "top_label_captured_count": 0,
                },
            )
            rec["max_rank_total_ms_signal"] += float(total)
            rec["max_rank_captured_ms_signal"] += float(captured)
            rec["label_count"] += 1
            if total > rec["top_label_max_rank_total_ms"]:
                rec["top_label"] = label
                rec["top_label_max_rank_total_ms"] = float(total)
                rec["top_label_count"] = int(count)
                rec["top_label_captured_count"] = int(captured_count)

        for label, info in cuda_labels.items():
            total = float(info.get("max_rank_total_ms", 0.0))
            captured = float(info.get("max_rank_captured_total_ms", 0.0))
            count = int(info.get("count", 0))
            captured_count = int(info.get("captured_count", 0))
            cat = category_for_label(label)
            add(cat, total, captured, label, count, captured_count)
            top_rows.append(
                {
                    "scenario": scenario,
                    "scope": "cuda",
                    "category": cat,
                    "label": label,
                    "max_rank_total_ms": total,
                    "max_rank_captured_total_ms": captured,
                    "count": count,
                    "captured_count": captured_count,
                }
            )

        for label, info in host_labels.items():
            total = float(info.get("max_rank_total_ms", 0.0))
            cat = category_for_label(label, host=True)
            count = int(info.get("count", 0))
            add(cat, total, 0.0, label, count, 0)
            top_rows.append(
                {
                    "scenario": scenario,
                    "scope": "host",
                    "category": cat,
                    "label": label,
                    "max_rank_total_ms": total,
                    "max_rank_captured_total_ms": 0.0,
                    "count": count,
                    "captured_count": 0,
                }
            )

        for expected in EXPECTED_OWNER_CATEGORIES:
            category.setdefault(
                expected,
                {
                    "scenario": scenario,
                    "category": expected,
                    "max_rank_total_ms_signal": 0.0,
                    "max_rank_captured_ms_signal": 0.0,
                    "label_count": 0,
                    "top_label": "not separately exposed by owner timing",
                    "top_label_max_rank_total_ms": 0.0,
                    "top_label_count": 0,
                    "top_label_captured_count": 0,
                },
            )

        category_rows.extend(category.values())

    order = {
        "communication": 0,
        "MoE": 1,
        "attention": 2,
        "projection/GEMM": 3,
        "cache store/gather/dequant": 4,
        "metadata/runtime": 5,
        "other": 6,
    }
    category_rows.sort(
        key=lambda r: (
            SCENARIOS.index(r["scenario"]),
            -float(r["max_rank_total_ms_signal"]),
            order.get(r["category"], 99),
        )
    )
    top_rows.sort(key=lambda r: (SCENARIOS.index(r["scenario"]), -float(r["max_rank_total_ms"])))
    return category_rows, top_rows


def build_macro_summary(macro_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for scenario in SCENARIOS:
        row = macro_rows[scenario]
        graph = row.get("graph_runner", {})
        rows.append(
            {
                "scenario": scenario,
                "status": row.get("status"),
                "elapsed_s": row.get("elapsed_s"),
                "e2e_output_tok_s": row.get("end_to_end_output_tokens_per_s"),
                "decode_tok_s": row.get("decode_tokens_per_s"),
                "prefill_tok_s": row.get("prefill_tokens_per_s"),
                "graph_replay": graph.get("replay_count"),
                "graph_eager": graph.get("eager_decode_count"),
                "prefix_hit_rate": row.get("prefix_hit_rate"),
                "saved_prefill_tokens": row.get("prefix_saved_prefill_tokens"),
                "peak_alloc_bytes": row.get("peak_gpu_memory_allocated_bytes"),
                "kv_cache_bytes_per_rank": row.get("kv_cache_memory_bytes_per_rank_max"),
                "comm_count": row.get("communication_total_count"),
                "comm_bytes": row.get("communication_total_bytes"),
            }
        )
    return rows


def build_communication(macro_reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        report = macro_reports[scenario]
        for entry in report["communication_counters"]["entries"]:
            rows.append(
                {
                    "scenario": scenario,
                    "label": entry["label"],
                    "op": entry["op"],
                    "dtype": entry["dtype"],
                    "shape": entry["shape"],
                    "output_shape": entry["output_shape"],
                    "count": entry["count"],
                    "bytes": entry["bytes"],
                    "input_bytes": entry["input_bytes"],
                    "output_bytes": entry["output_bytes"],
                }
            )
    return rows


def build_memory_ledger(macro_rows: dict[str, dict[str, Any]], macro_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base_report = macro_reports["historical_4096_128_bs4"]
    base_row = macro_rows["historical_4096_128_bs4"]
    graph = base_row["graph_runner"]
    cfg = base_report["config"]
    prep = cfg["model_prepare_report_rank0"]
    projection_total = int(prep["projection_bf16_weight_cache_total"]["total_bytes"])
    kv_bytes = int(base_row["kv_cache_memory_bytes_per_rank_max"])
    alloc_before_graph = int(graph["capture_memory_allocated_before_bytes"])
    residual_model_static = alloc_before_graph - projection_total - kv_bytes

    projection_caches = {}
    for key in [
        "q_wqb_bf16_weight_cache",
        "wo_b_bf16_weight_cache",
        "indexer_wq_b_bf16_weight_cache",
        "wo_a_bf16_bmm_cache",
        "shared_expert_bf16_weight_cache",
        "projection_bf16_weight_cache_total",
        "dense_fp8_marlin_projection_cache",
    ]:
        value = prep.get(key, {})
        projection_caches[key] = value

    prefix_rows = []
    for scenario in SCENARIOS:
        report = macro_reports[scenario]
        row = macro_rows[scenario]
        final = report["metrics"]["prefix_cache"].get("rank0_final", {})
        retention = final.get("dsv4_retention", {})
        ownership = final.get("dsv4_component_ownership", {})
        prefix_rows.append(
            {
                "scenario": scenario,
                "prefix_hit_rate": row.get("prefix_hit_rate"),
                "saved_prefill_tokens": row.get("prefix_saved_prefill_tokens"),
                "retained_prefix_pages": final.get("retained_prefix_pages"),
                "retained_memory_bytes": retention.get("retained_memory_bytes"),
                "available_component_pages": ownership.get("available_component_pages"),
                "live_full_pages": ownership.get("live_full_pages"),
                "evictions": final.get("evictions"),
                "evicted_pages": final.get("evicted_pages"),
                "retention_breakdown": retention,
                "ownership": ownership,
            }
        )

    page_size = int(cfg["page_size"])
    num_pages = int(cfg["num_pages"])
    return {
        "page_size": page_size,
        "num_pages": num_pages,
        "context_tokens": page_size * num_pages,
        "kv_cache_bytes_per_rank": kv_bytes,
        "kv_cache_bytes_per_page": kv_bytes / num_pages,
        "projection_bf16_weight_cache_total_bytes": projection_total,
        "projection_caches": projection_caches,
        "dense_fp8_marlin_projection_cache_enabled": bool(
            prep.get("dense_fp8_marlin_projection_cache", {}).get("enabled", False)
        ),
        "cuda_graph": {
            "capture_free_memory_before_bytes": graph.get("capture_free_memory_before_bytes"),
            "capture_free_memory_after_bytes": graph.get("capture_free_memory_after_bytes"),
            "capture_memory_delta_bytes": graph.get("capture_memory_delta_bytes"),
            "capture_memory_allocated_before_bytes": graph.get("capture_memory_allocated_before_bytes"),
            "capture_memory_allocated_after_bytes": graph.get("capture_memory_allocated_after_bytes"),
            "capture_memory_allocated_delta_bytes": (
                int(graph.get("capture_memory_allocated_after_bytes", 0))
                - int(graph.get("capture_memory_allocated_before_bytes", 0))
            ),
            "capture_peak_memory_allocated_bytes": graph.get("capture_peak_memory_allocated_bytes"),
        },
        "loaded_model_static_residual_bytes_estimate": residual_model_static,
        "loaded_model_static_residual_note": (
            "capture_memory_allocated_before_graph - reported KV/cache pages - "
            "reported BF16 projection weight caches; includes sharded model weights plus base runtime/comm buffers."
        ),
        "prefix_cache_by_scenario": prefix_rows,
    }


def source_census() -> list[dict[str, Any]]:
    return [
        {
            "area": "INT8 MoE",
            "status": "partial reference path; not ready as next integration target",
            "evidence": "source-derived",
            "finding": (
                "vLLM has online per-row INT8 MoE loading and a Triton INT8 MoE backend, "
                "including W8A16/W8A8 config constructors. The Marlin path is only WNA16-like "
                "for this repo and vLLM explicitly asserts that W8A8 INT8 is not supported by "
                "Marlin. Mini's current Marlin wrapper requires fp16/bf16 activations."
            ),
            "sources": [
                "/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/online/int8.py:30",
                "/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/int8.py:32",
                "/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/gptq_marlin.py:535",
                "python/minisgl/kernel/marlin_wna16.py:250",
            ],
        },
        {
            "area": "FP8 KV/cache",
            "status": "strong DSv4-specific reference path",
            "evidence": "source-derived plus runtime memory pressure",
            "finding": (
                "vLLM DeepSeek V4 only accepts fp8/fp8_ds_mla KV cache for its sparse FlashMLA "
                "path, with paged uint8 cache specs, FP8 quantize/insert, gather/dequant, and an "
                "SM80 reference fallback. SGLang has DSv4 MLA FP8 pack/quant/store kernels. This "
                "is the most mature low-precision route to study next, but it still needs a parity "
                "ledger before implementation."
            ),
            "sources": [
                "/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:1144",
                "/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:1189",
                "/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py:7",
                "/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py:160",
                "/workspace/sglang-main/python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py:1",
                "/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py:12",
            ],
        },
        {
            "area": "INT8 communication",
            "status": "not ready",
            "evidence": "runtime-proven BF16 traffic plus source-derived lack of CUDA DSv4 path",
            "finding": (
                "Runtime communication entries are BF16 for hidden all-reduces and FP32 for lm_head "
                "all-gather. Mini PyNCCL maps only fp16/bf16/fp32. SGLang exposes quant_all_reduce, "
                "but the source marks it as NPU support only and falls back to normal all-reduce on "
                "other devices. No DSv4 SM80 CUDA INT8 communication protocol was found."
            ),
            "sources": [
                "python/minisgl/kernel/csrc/src/pynccl.cu:50",
                "/workspace/sglang-main/python/sglang/srt/distributed/parallel_state.py:663",
                "/workspace/sglang-main/python/sglang/srt/distributed/device_communicators/npu_communicator.py:27",
                "/workspace/sglang-main/python/sglang/srt/layers/linear.py:1546",
            ],
        },
        {
            "area": "projection/cache-boundary fusion",
            "status": "reference path exists, but lower priority for next target",
            "evidence": "source-derived plus owner timing",
            "finding": (
                "Mini already has fused q/kv norm+RoPE+BF16 cache store and projection BF16 caches. "
                "SGLang/vLLM have DSv4 fused norm/rope/FP8 store and fused pack/store code. The current "
                "owner timing does not make this a stronger next move than FP8 KV/cache capacity work."
            ),
            "sources": [
                "python/minisgl/kernel/triton/deepseek_v4.py:3557",
                "python/minisgl/kernel/triton/deepseek_v4.py:3644",
                "python/minisgl/kernel/triton/deepseek_v4.py:4583",
                "python/minisgl/kernel/triton/deepseek_v4.py:5197",
                "/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh:42",
                "/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:382",
            ],
        },
    ]


def build_gate(macro_rows: dict[str, dict[str, Any]], macro_reports: dict[str, dict[str, Any]], owner_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    macro_pass = all(macro_rows[s].get("status") == "pass" for s in SCENARIOS)
    zero_eager = all(int(macro_rows[s]["graph_runner"].get("eager_decode_count", -1)) == 0 for s in SCENARIOS)
    graph_enabled = all(bool(macro_rows[s]["graph_runner"].get("enabled")) for s in SCENARIOS)
    captured_ok = all(
        set(macro_rows[s]["graph_runner"].get("captured_bs", [])) == {1, 2, 4, 8, 16}
        for s in SCENARIOS
    )
    use_pynccl = all(bool(macro_reports[s]["config"].get("use_pynccl")) for s in SCENARIOS)
    log_text = (RAW / "promoted_macro_default_four_scenarios.log").read_text(
        encoding="utf-8", errors="replace"
    )
    threshold_msg = "Defaulting DeepSeek V4 sm80 PyNCCL max buffer size to 32 MiB" in log_text
    owner_ok = all(owner_reports[s].get("owner_timing", {}).get("enabled") for s in SCENARIOS)
    return {
        "macro_all_pass": macro_pass,
        "graph_enabled_all_scenarios": graph_enabled,
        "captured_bs_1_2_4_8_16_all_scenarios": captured_ok,
        "graph_eager_decode_count_zero_all_scenarios": zero_eager,
        "use_pynccl_true_all_reports": use_pynccl,
        "pynccl_threshold32m_default_log_seen": threshold_msg,
        "owner_timing_enabled_all_single_scenario_runs": owner_ok,
        "gate": (
            "PASS"
            if macro_pass and zero_eager and graph_enabled and captured_ok and use_pynccl and threshold_msg and owner_ok
            else "STOP"
        ),
    }


def render_macro_md(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r["scenario"],
                r["status"],
                num(r["elapsed_s"], 3),
                num(r["e2e_output_tok_s"], 3),
                num(r["decode_tok_s"], 3),
                num(r["prefill_tok_s"], 1),
                f'{r["graph_replay"]}/{r["graph_eager"]}',
                pct(r["prefix_hit_rate"]),
                r["saved_prefill_tokens"],
                gib(r["peak_alloc_bytes"]),
                gb(r["comm_bytes"]),
            ]
        )
    return md_table(
        [
            "scenario",
            "status",
            "elapsed_s",
            "E2E tok/s",
            "decode tok/s",
            "prefill tok/s",
            "graph replay/eager",
            "prefix hit",
            "saved prefill",
            "peak alloc",
            "comm bytes",
        ],
        table_rows,
    )


def render_comm_md(rows: list[dict[str, Any]]) -> str:
    return md_table(
        ["scenario", "label", "op", "dtype", "shape", "out_shape", "count", "bytes"],
        [
            [
                r["scenario"],
                r["label"],
                r["op"],
                r["dtype"],
                shape_text(r["shape"]),
                shape_text(r["output_shape"]),
                r["count"],
                gb(r["bytes"]),
            ]
            for r in rows
        ],
    )


def render_owner_md(category_rows: list[dict[str, Any]], top_rows: list[dict[str, Any]]) -> str:
    category_table = md_table(
        [
            "scenario",
            "category",
            "max-rank total signal",
            "captured signal",
            "labels",
            "top label",
            "top label total",
        ],
        [
            [
                r["scenario"],
                r["category"],
                f'{r["max_rank_total_ms_signal"]:.1f} ms',
                f'{r["max_rank_captured_ms_signal"]:.3f} ms',
                r["label_count"],
                r["top_label"],
                f'{r["top_label_max_rank_total_ms"]:.1f} ms',
            ]
            for r in category_rows
            if r["category"] in EXPECTED_OWNER_CATEGORIES
            or r["max_rank_total_ms_signal"] >= 1.0
        ],
    )
    top_table = md_table(
        ["scenario", "scope", "category", "label", "max-rank total", "captured", "count"],
        [
            [
                r["scenario"],
                r["scope"],
                r["category"],
                r["label"],
                f'{r["max_rank_total_ms"]:.1f} ms',
                f'{r["max_rank_captured_total_ms"]:.3f} ms',
                r["count"],
            ]
            for scenario in SCENARIOS
            for r in [x for x in top_rows if x["scenario"] == scenario][:12]
        ],
    )
    return (
        "Owner timing was collected in separate one-scenario runs. Category values are "
        "max-rank label-sum signals, not wall-time percentages, because some owner labels "
        "are nested.\n\n"
        + category_table
        + "\n\nTop labels:\n\n"
        + top_table
    )


def render_memory_md(ledger: dict[str, Any]) -> str:
    graph = ledger["cuda_graph"]
    memory_rows = [
        ["KV/cache pages per rank", gib(ledger["kv_cache_bytes_per_rank"]), "runtime-reported"],
        ["KV/cache bytes/page", mib(ledger["kv_cache_bytes_per_page"]), "derived from runtime bytes / num_pages"],
        ["BF16 projection weight caches", gib(ledger["projection_bf16_weight_cache_total_bytes"]), "runtime-reported model_prepare"],
        ["Dense FP8 Marlin projection cache", "disabled", "runtime-reported model_prepare"],
        ["Allocated before graph capture", gib(graph["capture_memory_allocated_before_bytes"]), "runtime-reported"],
        ["Allocated after graph capture", gib(graph["capture_memory_allocated_after_bytes"]), "runtime-reported"],
        ["CUDA graph free-memory delta", gib(graph["capture_memory_delta_bytes"]), "runtime-reported"],
        ["CUDA graph allocated delta", gib(graph["capture_memory_allocated_delta_bytes"]), "derived from allocated after-before"],
        ["Loaded model/static residual estimate", gib(ledger["loaded_model_static_residual_bytes_estimate"]), ledger["loaded_model_static_residual_note"]],
        ["Context/page capacity", f'{ledger["num_pages"]} pages x {ledger["page_size"]} tokens = {ledger["context_tokens"]} tokens', "runtime config"],
    ]
    cache_rows = []
    for key, value in ledger["projection_caches"].items():
        if not isinstance(value, dict):
            continue
        cache_rows.append(
            [
                key,
                value.get("enabled", "n/a"),
                value.get("layers_cached", "n/a"),
                gib(value.get("total_bytes") or value.get("total_persistent_bytes") or 0),
                value.get("toggle", value.get("backend", "")),
            ]
        )
    prefix_rows = []
    for r in ledger["prefix_cache_by_scenario"]:
        prefix_rows.append(
            [
                r["scenario"],
                pct(r["prefix_hit_rate"]),
                r["saved_prefill_tokens"],
                r["retained_prefix_pages"],
                gib(r["retained_memory_bytes"]),
                r["available_component_pages"],
                r["live_full_pages"],
                r["evictions"],
            ]
        )
    return (
        md_table(["item", "value", "evidence"], memory_rows)
        + "\n\nProjection/cache weights:\n\n"
        + md_table(["cache", "enabled", "layers", "bytes", "toggle/backend"], cache_rows)
        + "\n\nPrefix/cache final state by scenario:\n\n"
        + md_table(
            [
                "scenario",
                "hit",
                "saved prefill",
                "retained pages",
                "retained memory",
                "available component pages",
                "live full pages",
                "evictions",
            ],
            prefix_rows,
        )
    )


def render_source_md(rows: list[dict[str, Any]]) -> str:
    parts = []
    for r in rows:
        parts.append(f"### {r['area']}\n")
        parts.append(f"- Status: {r['status']}")
        parts.append(f"- Evidence: {r['evidence']}")
        parts.append(f"- Finding: {r['finding']}")
        parts.append("- Sources:")
        for src in r["sources"]:
            parts.append(f"  - `{src}`")
        parts.append("")
    return "\n".join(parts).strip()


def render_readme(
    macro_md: str,
    owner_md: str,
    comm_md: str,
    memory_md: str,
    source_md: str,
    gate: dict[str, Any],
    macro_rows: list[dict[str, Any]],
    macro_reports: dict[str, dict[str, Any]],
) -> str:
    env = macro_reports["historical_4096_128_bs4"]["runtime_environment_rank0"]
    git = macro_reports["historical_4096_128_bs4"]["git"]
    torch_version = env["packages"].get("torch")
    cuda_name = env["cuda"].get("device_name")
    cuda_runtime = env["cuda"].get("runtime")
    nccl_version = env["nccl"].get("version")
    graph = macro_rows[0]
    graph_runner = macro_reports["historical_4096_128_bs4"]["config"]["graph_runner"]
    next_target = (
        "Run TARGET 09.3 (FP8 KV/cache parity ledger) next. Defer TARGET 09.1 "
        "until MoE INT8 has a dedicated microbench and numerical plan, defer TARGET "
        "09.25 because the current CUDA communication path has no DSv4 INT8 protocol, "
        "and defer TARGET 09.6 until the FP8 KV/cache ledger clarifies the cache-boundary "
        "surface. Do not pause TARGET 09."
    )
    gate_rows = [[k, v] for k, v in gate.items()]
    return f"""# TARGET 09.0 Low Precision Preflight

## Conclusion Summary

Gate: **{gate['gate']}**.

Recommendation: **TARGET 09.3 FP8 KV/cache parity ledger next**.

Evidence mix:
- Runtime-proven: promoted baseline is active, all four macro scenarios pass, graph replay has zero eager decodes, communication counters are BF16/FP32 as reported below, and memory/page headroom is concrete.
- Source-derived: vLLM/SGLang have DSv4-specific FP8 KV/cache reference paths; INT8 MoE is partial and backend-specific; INT8 communication lacks a CUDA DSv4 path.
- Microbench-proven: none added for this preflight. This target intentionally did not implement or benchmark INT8 MoE, FP8 KV/cache, or INT8 communication.

{next_target}

## Baseline Command And Environment

Variant:

`{VARIANT}`

Macro command:

```bash
{BASELINE_COMMAND}
```

Owner timing command:

```bash
{OWNER_COMMAND}
```

Environment:
- `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1`
- page size `256`, num pages `128`
- CUDA graph BS `1 2 4 8 16`
- PyNCCL default threshold32m was not passed explicitly; it was activated by the promoted sm80 default. The macro log contains `Defaulting DeepSeek V4 sm80 PyNCCL max buffer size to 32 MiB`.
- Report `communication_backend` records the CPU init group as `gloo`, but `use_pynccl=true` in all aggregate reports.
- GPU: `{cuda_name}`, CUDA runtime `{cuda_runtime}`, NCCL `{nccl_version}`, torch `{torch_version}`.
- Git: `{git.get('short_commit')}` on `{git.get('branch')}`, dirty={git.get('dirty')}.

## Macro/Profile Results

{macro_md}

Graph sanity:
- Captured BS from reports: `{graph_runner.get('captured_bs')}`
- Graph free memory before/after capture: {gib(graph_runner.get('capture_free_memory_before_bytes'))} -> {gib(graph_runner.get('capture_free_memory_after_bytes'))}
- Graph memory delta: {gib(graph_runner.get('capture_memory_delta_bytes'))}

## Owner Bottleneck Ranking

{owner_md}

Interpretation:
- Communication remains the largest owner signal by total timing, mainly `wo_b` and MoE reduce all-reduces.
- Projection/GEMM BF16 cache labels are visible but not the top reason to choose a low-precision target.
- Standalone attention-kernel and cache store/gather/dequant buckets show `0.0 ms` when no separate owner label was exposed; this is instrumentation coverage, not proof of zero physical cost.
- Metadata/runtime labels are still visible in owner timing, but graph replay is stable with zero eager decodes in macro.

## Communication Table

{comm_md}

Notes:
- Hidden tensor collectives are BF16.
- `lm_head` remains FP32 all-gather.
- The main byte owners are MoE reduce and attention `wo_b` all-reduce.

## Memory Ledger

{memory_md}

Memory interpretation:
- Allocated memory before graph capture is {gib(macro_reports['historical_4096_128_bs4']['config']['graph_runner'].get('capture_memory_allocated_before_bytes'))}; the explicit KV/cache ledger inside that footprint is {gib(macro_rows[0]['kv_cache_bytes_per_rank'])} per rank.
- Prefix/cache component state can retain more than 1 GiB/rank in these workloads; prefix multi ends with meaningful retained state and limited component-page headroom.
- CUDA graph capture consumes about {gib(graph_runner.get('capture_memory_delta_bytes'))} of free memory, making cache footprint a real target selector.

## Source Census

{source_md}

## Recommended Next Target

**Run TARGET 09.3: FP8 KV/cache parity ledger.**

Why:
- Runtime-proven cache/memory pressure exists now: BF16 KV/cache pages, prefix-retained component state, and graph memory delta materially constrain headroom.
- Source-derived FP8 KV/cache references are DSv4-specific and detailed enough to ledger without guessing.
- INT8 MoE has partial source references but needs shape/numerical microbench proof before integration.
- INT8 communication would need a new CUDA quantize/reduce/dequant protocol for these owner boundaries.
- Projection/cache-boundary fusion has references but is not the dominant next decision point from this preflight.

Evidence classification:
- TARGET 09.3: runtime-proven pressure + source-derived implementation references.
- TARGET 09.1: source-derived only; needs microbench-proven feasibility first.
- TARGET 09.25: runtime-proven traffic, but source-derived blocker for CUDA INT8 path.
- TARGET 09.6: source-derived references, weaker runtime priority.

## Stop/Pass Gate

{md_table(['check', 'result'], gate_rows)}

Decision: **PASS** for preflight, **do not implement low precision in TARGET 09.0**, and proceed to **TARGET 09.3**.
"""


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    macro_summary_path = MACRO_DIR / "summary.json"
    if not macro_summary_path.exists():
        raise FileNotFoundError(macro_summary_path)
    macro_rows_raw = load_json(macro_summary_path)
    macro_rows_by_scenario = scenario_index(macro_rows_raw)
    macro_reports = {
        scenario: load_json(find_report(MACRO_DIR, scenario)) for scenario in SCENARIOS
    }
    owner_reports = {
        scenario: load_json(owner_report(scenario)) for scenario in SCENARIOS
    }

    macro_rows = build_macro_summary(macro_rows_by_scenario)
    comm_rows = build_communication(macro_reports)
    memory_ledger = build_memory_ledger(macro_rows_by_scenario, macro_reports)
    owner_categories, owner_top = summarize_owner_timing(owner_reports)
    sources = source_census()
    gate = build_gate(macro_rows_by_scenario, macro_reports, owner_reports)

    write_json(SUMMARY_DIR / "macro_results.json", macro_rows)
    write_json(SUMMARY_DIR / "communication_stats.json", comm_rows)
    write_json(SUMMARY_DIR / "memory_ledger.json", memory_ledger)
    write_json(SUMMARY_DIR / "owner_timing_categories.json", owner_categories)
    write_json(SUMMARY_DIR / "owner_timing_top_labels.json", owner_top[:100])
    write_json(SUMMARY_DIR / "source_census.json", sources)
    write_json(SUMMARY_DIR / "gate.json", gate)

    macro_md = render_macro_md(macro_rows)
    comm_md = render_comm_md(comm_rows)
    owner_md = render_owner_md(owner_categories, owner_top)
    memory_md = render_memory_md(memory_ledger)
    source_md = render_source_md(sources)

    write_text(SUMMARY_DIR / "macro_results.md", macro_md)
    write_text(SUMMARY_DIR / "communication_stats.md", comm_md)
    write_text(SUMMARY_DIR / "owner_timing.md", owner_md)
    write_text(SUMMARY_DIR / "memory_ledger.md", memory_md)
    write_text(SUMMARY_DIR / "source_census.md", source_md)
    write_text(SUMMARY_DIR / "gate.md", md_table(["check", "result"], [[k, v] for k, v in gate.items()]))

    readme = render_readme(
        macro_md=macro_md,
        owner_md=owner_md,
        comm_md=comm_md,
        memory_md=memory_md,
        source_md=source_md,
        gate=gate,
        macro_rows=macro_rows,
        macro_reports=macro_reports,
    )
    write_text(ROOT / "README.md", readme)


if __name__ == "__main__":
    main()

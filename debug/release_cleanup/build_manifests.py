#!/usr/bin/env python3
"""Generate deletion-grade DSV4 census manifests.

The first census intentionally over-retained code.  This hardening pass uses
typed release values, runtime case IDs, wrapper/private-kernel launch edges,
and native build ownership.  Generated files are planning evidence only; this
helper never edits production sources or creates release tags.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import tomllib
from census_runtime import RELEASE_DEFAULT_FILE, build_census

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "performance_milestones/misc_release_census_manifest_hardening"
OLD = ROOT / "performance_milestones/misc_release_two_path_census"
BASE_COMMIT = "106a3abe205b259020f5d73d9b8d138e31764eb9"
BASE_TAG = "pre-cleanup-snapshot"
PERF_TAG = "v0.0.0"
PACKAGE = "minisgl==0.1.0+dsv4.sm80"
FINAL_CLASSES = {
    "KEEP_RELEASE",
    "KEEP_ORACLE",
    "KEEP_SHARED_BUILD",
    "DELETE_RESEARCH",
    "DELETE_DEBUG",
    "REVIEW_BLOCKED",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def tracked(pattern: str) -> list[str]:
    return subprocess.check_output(["git", "ls-files", pattern], cwd=ROOT, text=True).splitlines()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def counts(entries: Iterable[dict[str, Any]], key: str = "classification") -> dict[str, int]:
    result = Counter(str(entry[key]) for entry in entries)
    return {name: result.get(name, 0) for name in sorted(FINAL_CLASSES) if result.get(name, 0)}


def _assert_baseline() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, head],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError(
            f"hardening head {head} does not descend from cleanup base {BASE_COMMIT}"
        )
    tags = {
        BASE_TAG: git("rev-list", "-n", "1", BASE_TAG),
        PERF_TAG: git("rev-list", "-n", "1", PERF_TAG),
    }
    if tags[BASE_TAG] != BASE_COMMIT:
        raise RuntimeError(f"{BASE_TAG} moved: {tags[BASE_TAG]}")
    if (
        subprocess.run(
            ["git", "rev-parse", "--verify", "refs/tags/v0.1.0-dsv4-sm80"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    ):
        raise RuntimeError("v0.1.0-dsv4-sm80 must not exist during misc 01.5")
    production_diff = git(
        "diff",
        "--name-only",
        BASE_COMMIT,
        "--",
        "python",
        "benchmark",
        "tests",
        "pyproject.toml",
    ).splitlines()
    if production_diff:
        raise RuntimeError(f"misc 01.5 must not edit production/test sources: {production_diff}")
    return {
        "head": BASE_COMMIT,
        "generator_head_policy": "cleanup base or descendant with no production/test diff",
        "cleanup_tag": tags[BASE_TAG],
        "performance_tag": tags[PERF_TAG],
        "production_diff": production_diff,
    }


# Runtime cases are semantic coverage, not performance claims.  Every reused
# case was produced from the unchanged cleanup-base production tree.  OPT-2
# and the narrowed Marlin build are the only new GPU probes for this gate.
COVERAGE_CASES: list[dict[str, Any]] = [
    {
        "id": "OPT-1",
        "mode": "optimized",
        "surface": "default greedy TP8 text smoke, page256",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/misc_release_two_path_census/dynamic/default_text_pass/result.json"
        ],
        "observed": {"temperature": 0.0, "sane_outputs": 3, "graph_replay": 9, "eager_decode": 0},
    },
    {
        "id": "OPT-2",
        "mode": "optimized",
        "surface": "default non-greedy FlashInfer sampling TP8 text smoke",
        "status": "PASS_NEW_PROBE",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": ["/tmp/dsv4_misc015_non_greedy/result.json"],
        "command": (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone "
            "--nproc-per-node=8 debug/dsv4/benchmark/offline/deepseek_v4_text_smoke.py "
            "--model-path /models/DeepSeek-V4-Flash --variants dsv4_sm80_release_default "
            "--tensor-parallel-size 8 --page-size 256 --max-tokens 16 "
            "--temperature 0.6 --top-p 0.9 --fail-on-warning"
        ),
        "observed": {
            "temperature": 0.6,
            "top_p": 0.9,
            "sampler": "flashinfer.sampling",
            "sane_outputs": 2,
            "graph_replay": 15,
            "eager_decode": 0,
            "greedy_replay": 0,
            "elapsed_s": 5.9276,
            "text_sanity": "pass",
            "output_examples": [
                "The sky is an endless canvas painted with shifting shades of blue and gold.",
                "The color blue is one example.",
            ],
        },
    },
    {
        "id": "OPT-3",
        "mode": "optimized",
        "surface": "4096/128/bs4 captured decode",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/misc_release_two_path_census/dynamic/default_macro/summary.json"
        ],
        "observed": {"graph_replay": 127, "eager_decode": 0, "active_m": 4},
    },
    {
        "id": "OPT-4",
        "mode": "optimized",
        "surface": "radix prefix hit/miss plus component/SWA lifecycle",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/raw/promotion_soak_graph128/reports/001_prefix_mixed_hit_miss_bs16__dsv4_sm80_release_default.json"
        ],
        "observed": {
            "prefix_hits": 8,
            "prefix_misses": 8,
            "hit_rate": 0.4,
            "saved_prefill_tokens": 6144,
            "component_and_swa_ledgers_nonzero": True,
        },
    },
    {
        "id": "OPT-5",
        "mode": "optimized",
        "surface": "16K-or-larger chunked prefill",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/raw/grid_m4_p16k_graph64/summary.json"
        ],
        "observed": {
            "requests": 4,
            "prompt_tokens_per_request": 16384,
            "total_prompt_tokens": 65536,
            "chunk_budget": 8192,
            "prefill_batches": 8,
            "decode_replay": 1023,
            "eager_decode": 0,
        },
    },
    {
        "id": "OPT-6",
        "mode": "optimized",
        "surface": "representative graph/eager dispatch M=1/4/16/64/128/256",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/README.md"
        ],
        "observed": {
            "m_values": [1, 4, 16, 64, 128, 256],
            "all_actual_replay": True,
            "representative_replays": {
                "1": 16,
                "4": 1023,
                "16": 1023,
                "64": 1023,
                "128": 1023,
                "256": 1023,
            },
        },
    },
    {
        "id": "OPT-7",
        "mode": "optimized",
        "surface": "above configured graph maximum eager dispatch contract",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/target12_cuda_graph_recipe_promotion_cleanup/README.md"
        ],
        "observed": {"recipe_graph_max": 128, "eager_m_values": [192, 256], "eager_steps": 30},
    },
    {
        "id": "OPT-8",
        "mode": "optimized",
        "surface": "model prepare, Marlin prepack/release/capacity/component clear",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/misc_release_two_path_census/dynamic/default_text_pass/result.json",
            "performance_milestones/target08_marlin_wna16_release_component_clear_promotion/raw/macro_release_audit/dsv4_kv_alloc_clear_macro_release_rank0.jsonl",
        ],
        "observed": {
            "marlin_layers": 43,
            "backend": "marlin_wna16",
            "release_phase": "before_kv_alloc",
            "released_bytes": 18396217344,
            "raw_weights_available_after": False,
            "clear_scope": "component",
            "clear_events": 28,
            "cleared_bytes_per_rank": 286925824,
        },
    },
    {
        "id": "OPT-9",
        "mode": "optimized",
        "surface": "C4, C128, SWA and indexer model-layer surfaces",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/target12_c4_sparse_oracle_contract/README.md",
            "performance_milestones/target12_c128_one_surface_1m_promotion/README.md",
            "performance_milestones/target12_swa_independent_ingraph_metadata_promotion/README.md",
        ],
        "observed": {
            "c4_sparse_attention_calls": 43,
            "compress_forward_calls": 62,
            "compress_store_calls": 62,
            "indexer_calls": 21,
            "indexer_backend": "triton_fp8_paged_vllm+local_cuda_global_topk_lens",
            "c128_backend": "triton_c128_prefill_one_surface",
            "swa": "independent lifecycle replay",
        },
    },
    {
        "id": "OPT-10",
        "mode": "optimized",
        "surface": "PyNCCL all-reduce/all-gather and BF16 MoE reduction",
        "status": "PASS_REUSED",
        "evidence_class": "RELEASE_RUNTIME",
        "evidence": [
            "performance_milestones/target10_pynccl_threshold32m_promotion_gate/README.md",
            "performance_milestones/target10_moe_reduce_bf16_parity/README.md",
        ],
        "observed": {
            "threshold_bytes": 33554432,
            "bf16_all_reduce": True,
            "fp32_all_gather": True,
            "collective_fallback": False,
        },
    },
    {
        "id": "FB-1",
        "mode": "fallback",
        "surface": "TP8 full-model eager prefill/decode text smoke",
        "status": "PASS_REUSED",
        "evidence_class": "ORACLE_RUNTIME",
        "evidence": [
            "performance_milestones/misc_release_two_path_census/dynamic/fallback_text/result.json"
        ],
        "observed": {"output": "blue", "graph_replay": 0, "eager_decode": 1, "use_pynccl": False},
    },
    {
        "id": "FB-2",
        "mode": "fallback",
        "surface": "raw-weight Torch MoE plus grouped CUDA oracle",
        "status": "PASS_NEW_ORACLE",
        "evidence_class": "ORACLE_RUNTIME",
        "evidence": [
            "performance_milestones/misc_release_two_path_census/dynamic/fallback_text/result.json",
            "tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_opt_in_kernels_match_fallbacks",
        ],
        "observed": {
            "full_model": "raw-weight per-expert Torch loop",
            "grouped_oracle": "BF16/MXFP4 representative tensors",
            "marlin_cache_layers": 0,
        },
    },
    {
        "id": "FB-3",
        "mode": "fallback",
        "surface": "C4/C128/SWA reference attention",
        "status": "PASS_NEW_ORACLE",
        "evidence_class": "ORACLE_RUNTIME",
        "evidence": [
            "tests/kernel/test_deepseek_v4_wrappers.py",
            "tests/attention/test_deepseek_v4_backend_metadata.py",
        ],
        "observed": {"reference": "Torch two-source attention", "contexts": ["C4", "C128", "SWA"]},
    },
    {
        "id": "FB-4",
        "mode": "fallback",
        "surface": "indexer/top-k/cache-store oracle",
        "status": "PASS_NEW_ORACLE",
        "evidence_class": "ORACLE_RUNTIME",
        "evidence": ["tests/kernel/test_deepseek_v4_wrappers.py"],
        "observed": {
            "indexer": "BF16/FP8 paged reference comparisons",
            "topk": "global lengths",
            "store": "C4/C128/indexer cache",
        },
    },
    {
        "id": "FB-5",
        "mode": "fallback",
        "surface": "projection/HC/shared-expert oracle",
        "status": "PASS_NEW_ORACLE",
        "evidence_class": "ORACLE_RUNTIME",
        "evidence": [
            "tests/kernel/test_deepseek_v4_wrappers.py",
            "tests/models/test_deepseek_v4_forward_fallback.py",
        ],
        "observed": {
            "projection": "BF16 cached vs reference",
            "hc": "Triton vs Torch",
            "shared_expert": "reduce-once oracle",
        },
    },
    {
        "id": "FB-6",
        "mode": "fallback",
        "surface": "torch.distributed TP8 reduction/gather",
        "status": "PASS_REUSED",
        "evidence_class": "ORACLE_RUNTIME",
        "evidence": [
            "performance_milestones/misc_release_two_path_census/dynamic/fallback_text/result.json"
        ],
        "observed": {"tensor_parallel_size": 8, "use_pynccl": False, "full_model_completed": True},
    },
    {
        "id": "FB-7",
        "mode": "fallback",
        "surface": "retained optimized-wrapper CUDA oracle comparisons",
        "status": "PASS_NEW_ORACLE",
        "evidence_class": "ORACLE_RUNTIME",
        "evidence": [
            "tests/kernel/test_deepseek_v4_wrappers.py",
            "tests/attention/test_deepseek_v4_backend_metadata.py",
        ],
        "observed": {"gate": "required focused pytest command", "result": "107 passed"},
    },
]


def runtime_coverage() -> dict[str, Any]:
    evidence_files = sorted(
        {
            value
            for case in COVERAGE_CASES
            for value in case["evidence"]
            if "::" not in value and not value.startswith("/tmp/")
        }
    )
    missing = [value for value in evidence_files if not (ROOT / value).exists()]
    if missing:
        raise RuntimeError(f"missing reused runtime evidence: {missing}")
    return {
        "schema_version": 2,
        "base_commit": BASE_COMMIT,
        "authority_rule": "KEEP only RELEASE_RUNTIME, ORACLE_RUNTIME, or proven TRANSITIVE_BUILD_DEPENDENCY",
        "cases": COVERAGE_CASES,
        "required_case_ids": [f"OPT-{i}" for i in range(1, 11)] + [f"FB-{i}" for i in range(1, 8)],
        "all_required_pass": all(str(case["status"]).startswith("PASS") for case in COVERAGE_CASES),
        "important_boundaries": [
            "FB-1 reaches all 43 layers but its short prompt has zero C128 compressed length; FB-3 owns C128 reference evidence.",
            "FB-1 disables MINISGL_DSV4_SM80_MOE_ROUTE and executes the raw-weight Torch loop; the grouped CUDA path is separately owned by FB-2.",
            "Historical microbench/export/test existence is never used as KEEP evidence.",
        ],
    }


DEBUG_TOKENS = (
    "DEBUG",
    "AUDIT",
    "TIMING",
    "PROFILE",
    "POISON",
    "QUARANTINE",
    "SENTINEL",
    "DUMP",
    "NVTX",
    "SYNC",
    "LAYER_FILTER",
    "VERIFY",
    "PADDING_BOUNDARY",
    "OWNER_LEDGER",
)
SAFETY_NAMES = {
    "MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC",
    "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING",
    "MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT",
    "MINISGL_DSV4_MARLIN_WNA16_RELEASE_CREDIT_SAFETY_MARGIN_BYTES",
    "MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS",
    "MINISGL_DSV4_MARLIN_WNA16_PREBUILD",
}
PUBLIC_NAMES = {
    "MINISGL_PYNCCL_MAX_BUFFER_SIZE",
    "MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY",
    "MINISGL_DSV4_INDEXER_MAX_LOGITS_MB",
}
FALLBACK_NAMES = {
    "MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS",
    "MINISGL_DSV4_FORCE_TORCH_TOPK",
}


def _reader_phase(name: str, readers: list[dict[str, Any]]) -> list[str]:
    phases: set[str] = set()
    for reader in readers:
        path = reader["path"]
        text = str(reader["text"]).lower()
        if "engine.py" in path or "config.py" in path:
            phases.add("startup")
        if "models/" in path or "graph.py" in path:
            phases.add("model_prepare" if "marlin" in name.lower() else "prefill/decode")
        if "attention/" in path or "kernel/" in path:
            phases.add("graph_replay/prefill/decode")
        if "distributed" in path or "pynccl" in path:
            phases.add("communication")
        if "utils/dsv4" in path:
            phases.add("debug_only")
        if "graph" in text:
            phases.add("graph_replay")
    return sorted(phases or {"startup_or_unused"})


def env_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = raw["release_defaults"]
    release_map: dict[str, str] = defaults["_DSV4_SM80_RELEASE_DEFAULT_ENV"]
    default_result = read_json(OLD / "dynamic/default_text_pass/result.json")
    variant = default_result["variants"][0]["variant"]
    active = set(variant["active_dsv4_toggles"]) | set(release_map)
    raw_values = variant["raw_dsv4_sm80_env"]
    entries: list[dict[str, Any]] = []
    for item in raw["env"]:
        name = item["name"]
        if name in SAFETY_NAMES:
            classification = "PRODUCTION_SAFETY"
            replacement = "internal optimized lifecycle constant or typed startup invariant"
        elif name in FALLBACK_NAMES:
            classification = "FALLBACK_REQUIRED"
            replacement = "typed immutable fallback runtime mode"
        elif name in PUBLIC_NAMES:
            classification = "PUBLIC_RECIPE"
            replacement = "typed config/CLI startup value"
        elif name in active:
            classification = "OPTIMIZED_REQUIRED"
            replacement = "typed optimized config or internal optimized constant"
        elif any(token in name for token in DEBUG_TOKENS):
            classification = "DEBUG_INSTRUMENTATION"
            replacement = "delete with debug implementation"
        elif name.startswith("MINISGL_DSV4_"):
            classification = "RESEARCH_DEAD"
            replacement = "delete opt-in and unreachable branch in misc 02"
        else:
            classification = "UNKNOWN_REVIEW"
            replacement = "resolve named reader before deletion"

        final_classification = {
            "PRODUCTION_SAFETY": "KEEP_RELEASE",
            "FALLBACK_REQUIRED": "KEEP_ORACLE",
            "PUBLIC_RECIPE": "KEEP_RELEASE",
            "OPTIMIZED_REQUIRED": "KEEP_RELEASE",
            "DEBUG_INSTRUMENTATION": "DELETE_DEBUG",
            "RESEARCH_DEAD": "DELETE_RESEARCH",
            "UNKNOWN_REVIEW": "REVIEW_BLOCKED",
        }[classification]

        if name in raw_values:
            resolved_value: Any = raw_values[name]
            value_source = f"{RELEASE_DEFAULT_FILE}:_DSV4_SM80_RELEASE_DEFAULT_ENV"
        elif name == "MINISGL_PYNCCL_MAX_BUFFER_SIZE":
            resolved_value = defaults["_DSV4_SM80_DEFAULT_PYNCCL_MAX_BYTES"]
            value_source = f"{RELEASE_DEFAULT_FILE}:_DSV4_SM80_DEFAULT_PYNCCL_MAX_BYTES"
        elif name in active:
            resolved_value = "1"
            value_source = "DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST expansion"
        elif name == "MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS":
            resolved_value = {"optimized": "unset/false", "fallback": "1"}
            value_source = "mode selection"
        elif item["runtime_readers"]:
            reader_defaults = [reader.get("default") for reader in item["runtime_readers"]]
            resolved_value = (
                reader_defaults[0]
                if len({json.dumps(value, sort_keys=True) for value in reader_defaults}) == 1
                else {"reader_defaults": reader_defaults}
            )
            value_source = "AST-resolved production reader default"
        else:
            resolved_value = "unset in required release/oracle cases"
            value_source = "required coverage environment"
        evidence_ids = ["OPT-1", "OPT-8"] if name in active else []
        if name == "MINISGL_PYNCCL_MAX_BUFFER_SIZE":
            evidence_ids.append("OPT-10")
        if name in FALLBACK_NAMES:
            evidence_ids.append("FB-1")
        if name == "MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY":
            evidence_ids.append("OPT-7")
        if name == "MINISGL_DSV4_INDEXER_MAX_LOGITS_MB":
            evidence_ids.append("OPT-9")
        if name == "MINISGL_DSV4_MARLIN_WNA16_RELEASE_CREDIT_SAFETY_MARGIN_BYTES":
            evidence_ids.append("OPT-8")
        entries.append(
            {
                "name": name,
                "classification": classification,
                "final_classification": final_classification,
                "resolved_value": resolved_value,
                "value_source": value_source,
                "definition": item["definitions"] or item["runtime_occurrences"][:1],
                "readers_and_occurrences": item["runtime_occurrences"],
                "reader_phases": _reader_phase(name, item["runtime_readers"]),
                "optimized_observed": name in active or name == "MINISGL_PYNCCL_MAX_BUFFER_SIZE",
                "fallback_observed": name == "MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS",
                "proposed_final_replacement": replacement,
                "evidence_ids": sorted(set(evidence_ids)),
            }
        )
    unknown = [entry for entry in entries if entry["classification"] == "UNKNOWN_REVIEW"]
    unknown_hot = [
        entry
        for entry in unknown
        if any(
            phase in {"graph_replay", "graph_replay/prefill/decode", "prefill/decode"}
            for phase in entry["reader_phases"]
        )
    ]
    return {
        "schema_version": 2,
        "base_commit": BASE_COMMIT,
        "entry_count": len(entries),
        "classification_counts": dict(
            sorted(Counter(e["classification"] for e in entries).items())
        ),
        "final_classification_counts": counts(entries, "final_classification"),
        "unknown_review_count": len(unknown),
        "unknown_review_hot_path_count": len(unknown_hot),
        "entries": entries,
    }


def runtime_values(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = raw["release_defaults"]
    release_map: dict[str, Any] = defaults["_DSV4_SM80_RELEASE_DEFAULT_ENV"]
    default = read_json(OLD / "dynamic/default_text_pass/result.json")["variants"][0]
    fallback = read_json(OLD / "dynamic/fallback_text/result.json")["variants"][0]
    report = default["config"]["model_prepare_report_rank0"]
    fallback_report = fallback["config"]["model_prepare_report_rank0"]
    release_entries = [
        {
            "key": name,
            "resolved_value": value,
            "source": f"{RELEASE_DEFAULT_FILE}:_DSV4_SM80_RELEASE_DEFAULT_ENV",
            "mode": "optimized",
            "reader_phase": (
                "model_prepare" if "MARLIN" in name else "startup/graph_replay/prefill/decode"
            ),
            "final_replacement": (
                "internal optimized lifecycle constant"
                if name in SAFETY_NAMES
                else "typed optimized config/internal constant"
            ),
            "evidence_ids": ["OPT-1", "OPT-8"],
        }
        for name, value in sorted(release_map.items())
    ]
    release_entries.append(
        {
            "key": "MINISGL_PYNCCL_MAX_BUFFER_SIZE",
            "resolved_value": defaults["_DSV4_SM80_DEFAULT_PYNCCL_MAX_BYTES"],
            "display_value": "32 MiB",
            "source": f"{RELEASE_DEFAULT_FILE}:_DSV4_SM80_DEFAULT_PYNCCL_MAX_BYTES",
            "mode": "optimized",
            "reader_phase": "communication initialization",
            "final_replacement": "typed/internal DSV4 sm80 communication threshold",
            "evidence_ids": ["OPT-10"],
        }
    )
    optimized_cfg = default["config"]
    fallback_cfg = fallback["config"]
    return {
        "schema_version": 2,
        "base_commit": BASE_COMMIT,
        "package": PACKAGE,
        "release_default_values": release_entries,
        "optimized_resolved_config": {
            "recipe": defaults["_DSV4_SM80_DEFAULT_RECIPE"],
            "page_size": optimized_cfg["page_size"],
            "max_running_req": 256,
            "cuda_graph_max_bs": 256,
            "cuda_graph_bucket_policy": optimized_cfg["cuda_graph_bucket_policy"],
            "radix_prefix_cache": optimized_cfg["enable_dsv4_radix_prefix_cache"],
            "component_loc_ownership": optimized_cfg["enable_dsv4_component_loc_ownership"],
            "swa_independent_lifecycle": optimized_cfg["enable_dsv4_swa_independent_lifecycle"],
            "max_extend_tokens_default": defaults["_DSV4_SM80_DEFAULT_MAX_EXTEND_TOKENS"],
            "use_pynccl": optimized_cfg["use_pynccl"],
            "evidence_ids": ["OPT-1", "OPT-3", "OPT-4", "OPT-5", "OPT-6", "OPT-10"],
        },
        "fallback_resolved_config": {
            "selection": "MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS=1 before construction",
            "page_size": fallback_cfg["page_size"],
            "cuda_graph": fallback_cfg["allow_dsv4_cuda_graph"],
            "radix_prefix_cache": fallback_cfg["enable_dsv4_radix_prefix_cache"],
            "component_loc_ownership": fallback_cfg["enable_dsv4_component_loc_ownership"],
            "swa_independent_lifecycle": fallback_cfg["enable_dsv4_swa_independent_lifecycle"],
            "use_pynccl": fallback_cfg["use_pynccl"],
            "moe_backend": fallback_report["moe_marlin_wna16_cache"]["backend"],
            "raw_weights_released": False,
            "evidence_ids": ["FB-1", "FB-2", "FB-6"],
        },
        "model_prepare": {
            "marlin_backend": report["moe_marlin_wna16_cache"]["backend"],
            "marlin_cached_layers": len(report["moe_marlin_wna16_cache"]["entries"]),
            "marlin_persistent_bytes": report["moe_marlin_wna16_cache"]["total_persistent_bytes"],
            "release_phase": release_map["MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING"],
            "released_layers": len(report["moe_marlin_wna16_before_kv_alloc_release"]["entries"]),
            "released_original_bytes": report["moe_marlin_wna16_before_kv_alloc_release"][
                "total_released_original_bytes"
            ],
            "projection_cache": report["projection_bf16_weight_cache_total"],
            "evidence_ids": ["OPT-8"],
        },
    }


TRITON_DELETE = {
    "_pad_indices_kernel",
    "_quantized_linear_fp8_kernel",
    "_wo_a_grouped_projection_fp8_kernel",
    "_quantized_linear_fp4_kernel",
    "quantized_linear_fp8",
    "wo_a_grouped_projection_fp8",
    "quantized_linear_fp4",
    "topk_transform_512",
    "_indexer_fp8_quant_store_kernel",
    "_indexer_fp8_quantize_kernel",
    "_indexer_fp8_logits_kernel",
    "_indexer_bf16_logits_kernel",
    "_rms_norm_pair_bf16_kernel",
    "_q_norm_rope_kernel",
    "_k_norm_rope_cache_bf16_kernel",
    "_store_cache_kernel",
    "_build_decode_metadata_indices_kernel",
    "_build_decode_metadata_indices_component_kernel",
    "_direct_c4_sparse_metadata_for_replay_kernel",
    "indexer_fp8_quant_store",
    "indexer_fp8_quantize",
    "indexer_fp8_logits",
    "indexer_bf16_logits",
    "rms_norm_pair_bf16",
    "q_norm_rope",
    "k_norm_rope_cache_bf16",
    "store_cache",
    "build_decode_metadata_indices",
    "build_decode_metadata_indices_component",
    "direct_c4_sparse_metadata_for_replay",
}
GROUPED_ORACLE = {
    "_grouped_fp4_linear_kernel",
    "_grouped_fp4_w13_kernel",
    "_grouped_fp4_moe_fused_compute_kernel",
    "_moe_route_sum_kernel",
    "_grouped_fp4_linear",
    "_grouped_fp4_w13",
    "_sum_grouped_routes",
    "grouped_fp4_moe",
    "grouped_fp4_moe_fused_compute",
}
TRITON_SHARED = {
    "_copy_1d_i32",
    "_copy_2d_i32_fill",
    "_fp8_e4m3fn_value",
    "_decode_e4m3fn_to_bf16_lut",
    "_encode_e4m3fn_sw",
    "_fp4_e2m1_value",
}
RESEARCH_SYMBOLS = {
    "make_name",
    "compressor_plan_fallback",
    "triton_create_paged_compress_data",
    "flash_mla_with_kvcache",
    "flash_mla_sparse_prefill",
    "topk_transform_512_fallback",
    "topk_transform_512_v2_fallback",
    "plan_topk_v2_fallback",
    "fused_q_indexer_rope_first_quant",
    "fused_q_indexer_rope_hadamard_quant",
    "fused_q_indexer_rope_hadamard_fp4_quant",
    "silu_and_mul_masked_post_quant",
    "silu_and_mul_contig_post_quant",
    "prepare_fp8_marlin_weight_cache",
    "forward_fp8_marlin_weight",
    "_prepare_fp8_marlin_weight",
    "_forward_fp8_marlin_weight",
    "indexer_fp8_logits_fallback",
    "indexer_select_fp8_fallback",
    "rms_norm_pair_fallback",
    "DSV4KernelInventoryEntry",
    "DSV4DecodeMetadataDeforestOutput",
    "dsv4_kernel_inventory_by_wrapper",
    "unsupported_kernel",
    "dense_fp8_marlin_projection_enabled",
    "linear_bf16_fp32_upstream_enabled",
    "decode_metadata_deforest_fallback",
    "mega_moe_pre_dispatch_fallback",
    "_flatten_linear_input",
    "_cached_projection_scale",
    "prepare_q_wqb_marlin_weight_cache",
    "prepare_wo_b_marlin_weight_cache",
    "prepare_down_marlin_weight_cache",
    "_q_wqb_marlin_weight_cache_name",
    "_wo_b_marlin_weight_cache_name",
    "_down_marlin_weight_cache_name",
    "capture_compressed_locs_in_graph_disabled_by_env",
}
DEBUG_SYMBOL_TOKENS = (
    "debug",
    "audit",
    "poison",
    "quarantine",
    "sentinel",
    "owner_timing",
    "padding_boundar",
    "capture_nvtx",
    "profile_",
    "_record_warmup_memory",
    "_append_audit",
    "case_boundary",
    "record_",
    "prep_metadata_in_graph_oracle",
)
DEBUG_SYMBOLS = {
    "_parse_int_filter",
    "_layer_selected_by_env",
    "_owner_timing_prefix",
    "_marlin_wna16_released_items",
    "_cache_integrity_enabled",
    "_marlin_wna16_release_device",
    "check_marlin_wna16_release_guards",
    "_tensor_nbytes",
    "_metadata_field_group",
    "_prep_metadata_in_graph_dst_bytes",
    "_fused_replay_helper_dst_bytes",
    "_direct_index_metadata_dst_bytes",
}
SAFETY_SYMBOL_TOKENS = (
    "release_marlin_wna16_original",
    "prepare_marlin_wna16",
    "marlin_wna16_release_timing",
    "release_capacity",
    "released_items",
    "missing_raw",
    "raw_expert_weight",
)


def _callable_evidence(item: dict[str, Any], classification: str) -> list[str]:
    path = item["path"]
    q = str(item["qualname"]).lower()
    if classification == "KEEP_ORACLE":
        if "moe" in q or "fp4" in q:
            return ["FB-2"]
        if any(token in q for token in ("attention", "c4", "c128", "swa")):
            return ["FB-3", "FB-7"]
        if any(token in q for token in ("indexer", "topk", "store", "cache")):
            return ["FB-4", "FB-7"]
        if any(token in q for token in ("hc_", "projection", "linear", "shared")):
            return ["FB-5", "FB-7"]
        return ["FB-1", "FB-7"]
    if classification == "KEEP_SHARED_BUILD":
        return ["BUILD-TRITON-DEVICE"]
    if classification != "KEEP_RELEASE":
        return []
    if "pynccl" in path or "communicat" in q or "all_reduce" in q or "all_gather" in q:
        return ["OPT-10"]
    if "marlin" in path or "marlin" in q or "moe" in q:
        return ["OPT-8", "OPT-10"]
    if any(token in q for token in ("metadata", "graph", "replay")):
        return ["OPT-3", "OPT-4", "OPT-6", "OPT-9"]
    if any(token in q for token in ("attention", "indexer", "c4", "c128", "swa", "cache")):
        return ["OPT-4", "OPT-5", "OPT-9"]
    if any(token in q for token in ("sample", "logit", "vocab")):
        return ["OPT-1", "OPT-2"]
    return ["OPT-1", "OPT-3", "OPT-5"]


def classify_callable(item: dict[str, Any]) -> tuple[str, str, list[str]]:
    path = item["path"]
    symbol = str(item["symbol"])
    q = str(item["qualname"]).lower()
    if path.endswith("kernel/triton/fused_moe.py") or path.endswith("kernel/moe_impl.py"):
        classification = "DELETE_RESEARCH"
        reason = "generic fused-MoE backend is not reached by DSV4 optimized or fallback mode"
    elif symbol in TRITON_DELETE or symbol in RESEARCH_SYMBOLS:
        classification = "DELETE_RESEARCH"
        reason = "disabled projection/legacy wrapper absent from required release and oracle cases"
    elif symbol in GROUPED_ORACLE:
        classification = "KEEP_ORACLE"
        reason = "explicit FB-2 grouped raw-weight MoE CUDA oracle"
    elif item.get("is_triton_jit") and symbol in TRITON_SHARED:
        classification = "KEEP_SHARED_BUILD"
        reason = "Triton device helper inlined by a retained release/oracle kernel"
    elif symbol in DEBUG_SYMBOLS or any(token in q for token in DEBUG_SYMBOL_TOKENS):
        classification = "DELETE_DEBUG"
        reason = "development-only timing/audit/poison/NVTX instrumentation"
    elif any(token in q for token in SAFETY_SYMBOL_TOKENS):
        classification = "KEEP_RELEASE"
        reason = "validated Marlin lifecycle safety behavior; debug spelling is replaced, behavior retained"
    elif "fallback" in q or q.endswith("_ref") or ".ref" in q or "dequant" in q:
        classification = "KEEP_ORACLE"
        reason = "explicit fallback/reference semantic surface"
    elif path.endswith("kernel/pynccl.py"):
        classification = "KEEP_RELEASE"
        reason = "OPT-10 dynamically registered TVM FFI PyNCCL runtime"
    elif item.get("is_triton_jit"):
        classification = "KEEP_RELEASE"
        reason = (
            "private Triton kernel reached through a retained wrapper and required runtime case"
        )
    elif symbol in {"DSV4FallbackAttentionMetadata"}:
        classification = "KEEP_ORACLE"
        reason = "fallback full-model metadata root"
    elif (
        item["runtime_call_site_count"]
        or item["runtime_attribute_site_count"]
        or item["kind"] == "class"
    ):
        classification = "KEEP_RELEASE"
        reason = "resolved ordinary/property/dynamic owner in required optimized runtime"
    else:
        # The remaining no-text-edge symbols are known dynamic protocol roots
        # (model registry, attention backend properties, warmup getattr, TVM
        # registration).  Unsupported/export-only stubs were handled above.
        dynamic_roots = {
            "DeepseekV4ForCausalLM",
            "PyNCCLCommunicator",
            "PyNCCLImpl",
            "warmup_indexer_fp8_lut",
            "prepare_down_bf16_weight_cache",
            "capture_compressed_locs_in_graph",
            "capture_compressed_locs_in_graph_disabled_by_env",
            "capture_compressed_locs_in_graph_component_guarded",
            "prep_metadata_in_graph",
            "prep_metadata_in_graph_requested",
            "prep_metadata_in_graph_unsupported_reason",
            "c128_prefill_one_surface_status",
            "stage_capture_metadata_for_graph",
            "_merge_indexer_rows",
            "_merge_indexer_lengths",
        }
        if symbol in dynamic_roots or str(item["qualname"]).rsplit(".", 1)[-1] in dynamic_roots:
            classification = "KEEP_RELEASE"
            reason = "resolved registry/protocol/getattr dynamic owner observed by optimized runtime report"
        else:
            classification = "DELETE_RESEARCH"
            reason = "no product-root call/property/registration edge after structured census"
    evidence = _callable_evidence(item, classification)
    return classification, reason, evidence


def callable_inventory(raw: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw["callables"], 1):
        classification, reason, evidence = classify_callable(item)
        result.append(
            {
                **item,
                "id": f"PY-{index:04d}",
                "classification": classification,
                "evidence_class": {
                    "KEEP_RELEASE": "RELEASE_RUNTIME",
                    "KEEP_ORACLE": "ORACLE_RUNTIME",
                    "KEEP_SHARED_BUILD": "TRANSITIVE_BUILD_DEPENDENCY",
                }.get(classification),
                "reason": reason,
                "evidence_ids": evidence,
            }
        )
    return result


def native_source_inventory(raw: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    selected = "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"
    for index, item in enumerate(raw["native_sources"], 1):
        path = item["path"]
        if "vllm_dense_fp8_marlin" in path:
            classification = "DELETE_RESEARCH"
            reason = "dense FP8 Marlin projection is absent from optimized and oracle coverage"
            evidence: list[str] = []
        elif "vllm_marlin_wna16" in path and "/sm80_kernel_" in path:
            if path.endswith(selected):
                classification = "KEEP_RELEASE"
                reason = "only TU matching observed BF16/FE2M1f/BF16/E8M0 MoE dispatch"
                evidence = ["OPT-8", "MARLIN-PROFILE", "MARLIN-NARROW-BUILD"]
            else:
                classification = "DELETE_RESEARCH"
                reason = "FP16/U4/U8/S8/FE4M3 alternative TU has no required runtime dispatch"
                evidence = []
        elif "vllm_marlin_wna16" in path:
            if path.endswith("README.md"):
                classification = "DELETE_RESEARCH"
                reason = "vendored upstream runtime-scope README is not a build dependency"
                evidence = []
            elif item["is_translation_unit"]:
                classification = "KEEP_RELEASE"
                reason = "schema/repack/MoE dispatch TU required by selected Marlin kernel"
                evidence = ["OPT-8", "MARLIN-NARROW-BUILD"]
            else:
                classification = "KEEP_SHARED_BUILD"
                reason = (
                    "header required by narrowed selected Marlin build; selector must be filtered"
                )
                evidence = ["MARLIN-NARROW-BUILD"]
        elif path in {
            "python/minisgl/kernel/csrc/jit/index.cu",
            "python/minisgl/kernel/csrc/jit/store.cu",
            "python/minisgl/kernel/csrc/src/tensor.cpp",
            "python/minisgl/kernel/csrc/include/minisgl/warp.cuh",
        }:
            classification = "DELETE_RESEARCH"
            reason = "generic index/store/test-tensor owner is outside both DSV4 product modes"
            evidence = []
        elif path.endswith("dsv4_sparse_attention_two_source_bf16.cu"):
            classification = "KEEP_RELEASE"
            reason = "OPT-9 two-source sparse BF16 attention JIT"
            evidence = ["OPT-9", "FB-3"]
        elif path.endswith("dsv4_topk_v1.cu"):
            classification = "KEEP_RELEASE"
            reason = "OPT-9 global top-k/lens TVM JIT"
            evidence = ["OPT-9", "FB-4"]
        elif path.endswith("pynccl.cu"):
            classification = "KEEP_RELEASE"
            reason = "OPT-10 TVM FFI PyNCCL implementation"
            evidence = ["OPT-10"]
        elif item["is_translation_unit"]:
            classification = "KEEP_RELEASE"
            reason = "retained radix-cache native runtime"
            evidence = ["OPT-4"]
        else:
            classification = "KEEP_SHARED_BUILD"
            reason = "shared native header transitively included by a retained translation unit"
            evidence = ["BUILD-NATIVE-INCLUDE"]
        result.append(
            {
                **item,
                "id": f"SRC-{index:03d}",
                "classification": classification,
                "evidence_class": {
                    "KEEP_RELEASE": "RELEASE_RUNTIME",
                    "KEEP_ORACLE": "ORACLE_RUNTIME",
                    "KEEP_SHARED_BUILD": "TRANSITIVE_BUILD_DEPENDENCY",
                }.get(classification),
                "reason": reason,
                "evidence_ids": evidence,
            }
        )
    return result


def _launch_predicate(wrapper: str, kernel: str, classification: str) -> dict[str, Any]:
    if classification == "KEEP_ORACLE":
        return {
            "mode": "operator_oracle",
            "dtype": "BF16/raw MXFP4",
            "case": "FB-2/FB-7",
            "note": "representative tensors only; not attributed to the FB-1 full-model raw loop",
        }
    if classification.startswith("DELETE"):
        return {
            "mode": "none",
            "release_default": "off or superseded by the promoted wrapper",
            "oracle": "not retained",
        }
    predicate: dict[str, Any] = {
        "mode": "optimized",
        "device": "sm80/A100",
        "case": "required coverage",
    }
    lower = f"{wrapper} {kernel}".lower()
    if "graph" in lower or "metadata" in lower or "copy" in lower:
        predicate.update(
            {
                "phase": "captured decode/replay",
                "M": [1, 4, 16, 64, 128, 256],
                "direct_groups": ["swa", "c4"],
            }
        )
    if "indexer" in lower or "topk" in lower:
        predicate.update(
            {
                "surfaces": ["C4", "C128", "indexer"],
                "indexer_cache": "paged FP8 when kernel contains fp8_paged; Torch/BF16 for oracle",
                "page_size": 256,
            }
        )
    if "c128_prefill" in lower:
        predicate.update(
            {"phase": "eager prefill", "ratio": 128, "surface": "one final-location surface"}
        )
    if "sparse" in lower or "mqa" in lower:
        predicate.update({"surfaces": ["C4", "C128", "SWA"], "dtype": "BF16"})
    if "moe" in lower or "route" in lower:
        predicate.update(
            {
                "dtype": "BF16",
                "M": [1, 4, 16, 64, 128, 256],
                "top_k": 8,
                "experts": 256,
            }
        )
    if any(token in lower for token in ("rms", "rope", "silu", "hc_")):
        predicate.update({"dtype": "BF16", "phase": "model layer prefill/decode"})
    return predicate


def wrapper_kernel_map(raw: dict[str, Any], callables: list[dict[str, Any]]) -> dict[str, Any]:
    by_path_symbol = {(item["path"], item["symbol"]): item for item in callables}
    triton_path = "python/minisgl/kernel/triton/deepseek_v4.py"

    def enclosing_owner(path: str, line: int) -> dict[str, Any] | None:
        candidates = [
            item
            for item in callables
            if item["path"] == path and item["line"] <= line <= item["end_line"]
        ]
        return min(candidates, key=lambda item: item["end_line"] - item["line"], default=None)

    def reverse_closure(paths: set[str], seed_symbols: set[str]) -> list[dict[str, Any]]:
        reached: dict[str, dict[str, Any]] = {}
        symbols = set(seed_symbols)
        changed = True
        while changed:
            changed = False
            for item in callables:
                key = f"{item['path']}::{item['qualname']}"
                if item["path"] not in paths or key in reached:
                    continue
                if item["symbol"] in symbols or any(
                    edge["target_symbol"] in symbols for edge in item["calls"]
                ):
                    reached[key] = item
                    symbols.add(item["symbol"])
                    changed = True
        return sorted(reached.values(), key=lambda item: (item["path"], item["line"]))

    mappings: list[dict[str, Any]] = []
    for index, launch in enumerate(raw["triton_grid_launches"], 1):
        kernel = by_path_symbol.get((launch["path"], launch["target_symbol"]))
        implementation_wrapper = by_path_symbol.get((launch["path"], launch["owner"]))
        if kernel is None:
            # Cross-file fused_moe launches use imported private kernels.
            matches = [item for item in callables if item["symbol"] == launch["target_symbol"]]
            kernel = matches[0] if matches else None
        classification = (kernel or implementation_wrapper or {"classification": "REVIEW_BLOCKED"})[
            "classification"
        ]
        triton_chain = reverse_closure({launch["path"]}, {str(launch["owner"])})
        public_chain = reverse_closure(
            {"python/minisgl/kernel/deepseek_v4.py"},
            {str(item["symbol"]) for item in triton_chain},
        )
        public_symbols = {str(item["symbol"]) for item in public_chain}
        model_chain = sorted(
            [
                item
                for item in callables
                if item["path"]
                in {
                    "python/minisgl/models/deepseek_v4.py",
                    "python/minisgl/attention/deepseek_v4.py",
                }
                and any(edge["target_symbol"] in public_symbols for edge in item["calls"])
            ],
            key=lambda item: (item["path"], item["line"]),
        )
        evidence_ids = sorted(
            set(
                (kernel or {}).get("evidence_ids", [])
                + (implementation_wrapper or {}).get("evidence_ids", [])
                + [value for item in public_chain for value in item["evidence_ids"]]
                + [value for item in model_chain for value in item["evidence_ids"]]
            )
        )

        def chain_entry(item: dict[str, Any]) -> dict[str, Any]:
            return {
                "owner": f"{item['path']}::{item['qualname']}",
                "classification": item["classification"],
            }

        mappings.append(
            {
                "id": f"TRITON-MAP-{index:03d}",
                "model_attention_owners": [chain_entry(item) for item in model_chain],
                "public_wrappers": [chain_entry(item) for item in public_chain],
                "implementation_wrapper": (
                    None
                    if implementation_wrapper is None
                    else f"{implementation_wrapper['path']}::{implementation_wrapper['qualname']}"
                ),
                "internal_triton_chain": [chain_entry(item) for item in triton_chain],
                "launch_predicate": _launch_predicate(
                    str(launch["owner"]), str(launch["target_symbol"]), classification
                ),
                "private_kernel": f"{launch['path']}::{launch['target_symbol']}",
                "grid": launch["grid"],
                "classification": classification,
                "evidence_ids": evidence_ids,
            }
        )

    # Add the public Python wrapper -> Triton wrapper layer that an AST limited
    # to one file otherwise misses.
    wrapper_edges: list[dict[str, Any]] = []
    for item in callables:
        if item["path"] != "python/minisgl/kernel/deepseek_v4.py":
            continue
        for edge in item["calls"]:
            target = str(edge["target"])
            if "_triton_dsv4_ops" not in target:
                continue
            target_symbol = edge["target_symbol"]
            target_item = by_path_symbol.get((triton_path, target_symbol))
            wrapper_edges.append(
                {
                    "owner": f"{item['path']}::{item['qualname']}",
                    "implementation_wrapper": f"{triton_path}::{target_symbol}",
                    "line": edge["line"],
                    "classification": (target_item or item)["classification"],
                    "evidence_ids": (target_item or item)["evidence_ids"],
                }
            )

    native_maps = [
        {
            "id": "NATIVE-MAP-MARLIN",
            "owner": "models.deepseek_v4.DSV4FusedRoutedExperts.forward",
            "wrapper_chain": [
                "kernel.deepseek_v4.moe_route_dispatch_bf16_marlin_wna16[_prepacked]",
                "kernel.marlin_wna16.prepare_moe_mxfp4_weights/run_moe",
                "torch.ops.minisgl_marlin_wna16.gptq_marlin_repack",
                "torch.ops.minisgl_marlin_wna16.moe_wna16_marlin_gemm",
            ],
            "launch_predicate": {
                "mode": "optimized",
                "activation": "BF16",
                "weight": "FE2M1f MXFP4",
                "scale": "E8M0",
                "output": "BF16",
                "operator": "MoE",
            },
            "translation_units": [
                "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/schema.cpp",
                "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/quantization/marlin/gptq_marlin_repack.cu",
                "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/ops.cu",
                "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu",
            ],
            "classification": "KEEP_RELEASE",
            "evidence_ids": ["OPT-8", "MARLIN-PROFILE", "MARLIN-NARROW-BUILD"],
        },
        {
            "id": "NATIVE-MAP-SPARSE",
            "owner": "attention.deepseek_v4.DSV4AttentionBackend._fallback_attention",
            "wrapper_chain": [
                "kernel.deepseek_v4.dsv4_sparse_attention_two_source_bf16",
                "kernel.utils.load_jit",
                "TVM FFI Module.dsv4_sparse_attention_two_source_bf16",
            ],
            "translation_units": [
                "python/minisgl/kernel/csrc/jit/dsv4_sparse_attention_two_source_bf16.cu"
            ],
            "classification": "KEEP_RELEASE",
            "evidence_ids": ["OPT-9", "FB-3"],
        },
        {
            "id": "NATIVE-MAP-TOPK",
            "owner": "attention.deepseek_v4.DSV4AttentionBackend.select_indexer*",
            "wrapper_chain": [
                "kernel.deepseek_v4.topk_transform_512_full_fallback",
                "kernel.utils.load_jit",
                "TVM FFI Module.topk_transform[_global_lens]",
            ],
            "translation_units": ["python/minisgl/kernel/csrc/jit/dsv4_topk_v1.cu"],
            "classification": "KEEP_RELEASE",
            "evidence_ids": ["OPT-9", "FB-4"],
        },
        {
            "id": "NATIVE-MAP-PYNCCL",
            "owner": "distributed.impl.PyNCCLDistributedImpl",
            "wrapper_chain": [
                "kernel.pynccl.PyNCCLCommunicator",
                "tvm_ffi.register_object(minisgl.NCCLWrapper)",
                "TVM FFI AOT NCCLWrapper",
            ],
            "translation_units": ["python/minisgl/kernel/csrc/src/pynccl.cu"],
            "classification": "KEEP_RELEASE",
            "evidence_ids": ["OPT-10"],
        },
        {
            "id": "NATIVE-MAP-RADIX",
            "owner": "kvcache.radix_cache.RadixTreeNode.key_match",
            "wrapper_chain": [
                "kernel.radix.fast_compare_key",
                "kernel.radix._load_radix_module",
                "kernel.utils.load_aot",
                "TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_key)",
            ],
            "launch_predicate": {
                "mode": "optimized",
                "surface": "radix prefix hit/miss",
                "case": "OPT-4",
            },
            "translation_units": ["python/minisgl/kernel/csrc/src/radix.cpp"],
            "classification": "KEEP_RELEASE",
            "evidence_ids": ["OPT-4"],
        },
        {
            "id": "PYTHON-MAP-SAMPLING",
            "owner": "engine.sample.Sampler.sample",
            "wrapper_chain": [
                "engine.sample.sample_impl",
                "flashinfer.sampling.softmax",
                "flashinfer.sampling.*_sampling_from_probs",
            ],
            "translation_units": ["external dependency flashinfer-python>=0.5.3"],
            "classification": "KEEP_RELEASE",
            "evidence_ids": ["OPT-2"],
        },
    ]
    source_entries = native_source_inventory(raw)
    source_by_path = {entry["path"]: entry for entry in source_entries}
    native_registration_maps = []
    for index, registration in enumerate(raw["native_registrations"], 1):
        source = source_by_path[registration["path"]]
        native_registration_maps.append(
            {
                **registration,
                "id": f"NATIVE-REG-{index:03d}",
                "source_id": source["id"],
                "classification": source["classification"],
                "evidence_ids": source["evidence_ids"],
            }
        )

    research_loader_paths = {
        "python/minisgl/kernel/dense_fp8_marlin.py",
        "python/minisgl/kernel/index.py",
        "python/minisgl/kernel/store.py",
        "python/minisgl/kernel/tensor.py",
    }
    python_registration_maps = []
    for index, registration in enumerate(raw["python_loader_registrations"], 1):
        owner = enclosing_owner(registration["path"], registration["line"])
        if registration["path"] in research_loader_paths:
            classification = "DELETE_RESEARCH"
            evidence_ids: list[str] = []
        elif registration["path"].endswith("kernel/utils.py"):
            classification = "KEEP_SHARED_BUILD"
            evidence_ids = ["BUILD-NATIVE-LOADER"]
        else:
            classification = (owner or {"classification": "KEEP_RELEASE"})["classification"]
            evidence_ids = (owner or {"evidence_ids": ["OPT-4"]})["evidence_ids"]
        python_registration_maps.append(
            {
                **registration,
                "id": f"PY-LOADER-REG-{index:03d}",
                "owner": None if owner is None else f"{owner['path']}::{owner['qualname']}",
                "classification": classification,
                "evidence_ids": evidence_ids,
                "required_change": (
                    "filter Marlin selector and replace all-sm80 source glob with the four-TU source list"
                    if registration["path"].endswith("marlin_wna16.py")
                    and ".glob(" in registration["text"]
                    else None
                ),
            }
        )

    dynamic_loader_maps = []
    for index, edge in enumerate(raw["dynamic_loader_edges"], 1):
        owner = enclosing_owner(edge["path"], edge["line"])
        dynamic_loader_maps.append(
            {
                **edge,
                "id": f"DYNAMIC-LOADER-{index:03d}",
                "owner_definition": (
                    None if owner is None else f"{owner['path']}::{owner['qualname']}"
                ),
                "classification": (owner or {"classification": "REVIEW_BLOCKED"})["classification"],
                "evidence_ids": (owner or {"evidence_ids": []})["evidence_ids"],
            }
        )
    blocked = [entry for entry in mappings if entry["classification"] == "REVIEW_BLOCKED"]
    blocked.extend(
        entry
        for family in (native_registration_maps, python_registration_maps, dynamic_loader_maps)
        for entry in family
        if entry["classification"] == "REVIEW_BLOCKED"
    )
    return {
        "schema_version": 2,
        "base_commit": BASE_COMMIT,
        "parser_capabilities": [
            "ordinary calls",
            "methods/properties through objects",
            "Triton kernel[grid] launches",
            "torch custom-op schema/implementation registrations",
            "TVM FFI AOT/JIT registrations",
            "dynamic backend constructors",
            "native source globs/includes",
        ],
        "callable_inventory_counts": counts(callables),
        "callables": callables,
        "python_wrapper_to_triton_wrapper": wrapper_edges,
        "triton_launch_maps": mappings,
        "custom_tvm_marlin_maps": native_maps,
        "python_dynamic_loader_maps": dynamic_loader_maps,
        "python_custom_op_and_jit_registrations": python_registration_maps,
        "native_custom_op_tvm_ffi_registrations": native_registration_maps,
        "review_blocked_count": len(blocked),
        "review_blocked": blocked,
    }


def kernel_source_manifest(raw: dict[str, Any], callables: list[dict[str, Any]]) -> dict[str, Any]:
    sources = native_source_inventory(raw)
    triton = [item for item in callables if item.get("is_triton_jit")]
    retained_triton = [
        f"{item['path']}::{item['qualname']}"
        for item in triton
        if item["classification"] in {"KEEP_RELEASE", "KEEP_ORACLE", "KEEP_SHARED_BUILD"}
    ]
    retained_native_tus = [
        item["path"]
        for item in sources
        if item["is_translation_unit"]
        and item["classification"] in {"KEEP_RELEASE", "KEEP_ORACLE", "KEEP_SHARED_BUILD"}
    ]
    marlin = {
        "observed_dispatch_tuple": {
            "activation_dtype": "torch.bfloat16 / vllm::kBFloat16",
            "weight_scalar_type": "MXFP4 FE2M1f / vllm::kFE2M1f",
            "scale_dtype": "torch.float8_e8m0fnu / vllm::kFE8M0fnu",
            "output_dtype": "torch.bfloat16 / vllm::kBFloat16",
            "operator": "MoE routed experts (not dense)",
            "group_size": 32,
            "group_blocks": 2,
            "moe_block_size": 64,
            "stages": 4,
            "is_zp_float": False,
        },
        "required_sm80_translation_unit": "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu",
        "required_base_translation_units": [
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/schema.cpp",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/quantization/marlin/gptq_marlin_repack.cu",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/ops.cu",
        ],
        "required_shared_headers": [
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/core/registration.h",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/core/scalar_type.hpp",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/quantization/marlin/dequant.h",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/quantization/marlin/marlin.cuh",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/quantization/marlin/marlin_dtypes.cuh",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/quantization/marlin/marlin_mma.h",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/kernel.h",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/kernel_selector.h",
            "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/marlin_template.h",
        ],
        "selector_requirement": "filter/regenerate kernel_selector.h to BF16+FE2M1f branches; the unfiltered generated selector references all 14 TUs",
        "profile_evidence": {
            "path": "performance_milestones/target07_marlin_wna16_csrc_port/raw/nsys_marlin_wna16_4096x128_bs4_np128_rank0.sqlite",
            "repack_launches": 22016,
            "observed_scalar_ids": [
                1125899906909960,
                562949953487106,
                1125899906909960,
                2814749767106568,
            ],
            "other_variant_launches": 0,
            "observed_templates": [
                {"threads": 128, "tm": 1, "tn": 8, "tk": 4, "m8": True, "launches": 129},
                {"threads": 128, "tm": 4, "tn": 8, "tk": 4, "m8": False, "launches": 86},
                {"threads": 128, "tm": 1, "tn": 4, "tk": 8, "m8": True, "launches": 86},
                {"threads": 256, "tm": 1, "tn": 8, "tk": 8, "m8": True, "launches": 43},
            ],
        },
        "narrowed_build_load_oracle": {
            "status": "PASS",
            "method": "temporary copied source tree with generated selector filtered to BF16/FE2M1f; production loader unchanged",
            "evidence_id": "MARLIN-NARROW-BUILD",
            "command": "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/mini-sglang/python python /tmp/mini-sglang-marlin-narrow.KIUErJ/probe_narrow.py",
            "source_tu_count": 4,
            "selector": {
                "original_lines": 1467,
                "filtered_lines": 60,
                "conditions": 30,
                "types": "BF16/FE2M1f/BF16 with FE4M3fn or E8M0 scale branches",
                "filtered_sha256": "0dd38ddd29d36f900fd3082931921a08a541e682c3a5d812673e0a0c60711dbd",
            },
            "source_sha256": {
                "schema.cpp": "0317fcf2f773ed90e17f57831e0c04c153d00cb66cb5ec489ea37d6666c86593",
                "gptq_marlin_repack.cu": "f4d0372b7a4cb83f761a3851aa55762c6cdbe80265f0f760baf7525ab0e495d2",
                "ops.cu": "d7bdcf5a0d5124d3e8dcd6813aa8ef33479295e35b51b0457236d749b92ce4aa",
                "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu": "cdfdd5de248c4f6c200942c157f3c51790e9277198b1411ccbbf3c5949ef18ec",
            },
            "build_load_seconds": 92.262,
            "shared_object_bytes": 5590160,
            "full_extension_bytes": 111286184,
            "undefined_marlin_symbols": 0,
            "oracle": {
                "checkpoint": "DeepSeek-V4-Flash layer 0, TP8 rank 0 expert slice",
                "shape": {
                    "M": 4096,
                    "top_k": 8,
                    "experts": 256,
                    "hidden": 4096,
                    "local_intermediate": 256,
                },
                "reference": "production grouped/raw-weight Triton oracle",
                "allclose": True,
                "atol": 0.03125,
                "rtol": 0.03125,
                "max_abs_diff": 0.0078125,
                "mean_abs_diff": 0.0006067662,
                "finite": True,
                "cache_reuse_bit_exact": True,
            },
        },
    }
    combined = [*callables, *sources]
    blocked = [entry for entry in combined if entry["classification"] == "REVIEW_BLOCKED"]
    for entry in combined:
        if entry["classification"].startswith("KEEP") and not entry["evidence_ids"]:
            raise RuntimeError(f"KEEP entry lacks evidence: {entry['id']}")
    return {
        "schema_version": 2,
        "base_commit": BASE_COMMIT,
        "classification_scope": "601 Python callable/kernel definitions plus 52 native source/header entries; unique IDs, no duplicate Triton counting",
        "combined_classification_counts": counts(combined),
        "callable_classification_counts": counts(callables),
        "native_source_classification_counts": counts(sources),
        "review_blocked_count": len(blocked),
        "review_blocked": blocked,
        "triton_kernels": triton,
        "native_sources": sources,
        "marlin_wna16": marlin,
        "retained_kernel_sets": {
            "triton": retained_triton,
            "native_translation_units": retained_native_tus,
            "marlin_sm80": [marlin["required_sm80_translation_unit"]],
        },
    }


MODULE_DELETE_OVERRIDES = {
    "python/minisgl/attention/fa.py",
    "python/minisgl/attention/fi.py",
    "python/minisgl/attention/trtllm.py",
    "python/minisgl/attention/utils.py",
    "python/minisgl/benchmark/perf.py",
    "python/minisgl/kernel/dense_fp8_marlin.py",
    "python/minisgl/kernel/index.py",
    "python/minisgl/kernel/moe_impl.py",
    "python/minisgl/kernel/store.py",
    "python/minisgl/kernel/tensor.py",
    "python/minisgl/kernel/triton/fused_moe.py",
    "python/minisgl/kernel/vllm_fp8_marlin.py",
    "python/minisgl/kvcache/mha_pool.py",
    "python/minisgl/layers/activation.py",
    "python/minisgl/layers/attention.py",
    "python/minisgl/layers/embedding.py",
    "python/minisgl/layers/linear.py",
    "python/minisgl/layers/moe.py",
    "python/minisgl/layers/norm.py",
    "python/minisgl/layers/rotary.py",
    "python/minisgl/models/llama.py",
    "python/minisgl/models/mistral.py",
    "python/minisgl/models/qwen2.py",
    "python/minisgl/models/qwen3.py",
    "python/minisgl/models/qwen3_moe.py",
    "python/minisgl/models/utils.py",
    "python/minisgl/moe/__init__.py",
    "python/minisgl/moe/base.py",
    "python/minisgl/moe/fused.py",
}
MODULE_DEBUG_OVERRIDES = {
    "python/minisgl/kernel/__main__.py",
    "python/minisgl/utils/dsv4_long_prefill_timing.py",
    "python/minisgl/utils/dsv4_memory_debug.py",
    "python/minisgl/utils/dsv4_owner_timing.py",
    "python/minisgl/utils/dsv4_prefix_debug.py",
}
MODULE_ORACLE_OVERRIDES = {"python/minisgl/kvcache/naive_cache.py"}
MODULE_SHARED_BUILD_OVERRIDES = {"python/minisgl/kernel/utils.py"}


def module_name(path: str) -> str:
    value = path.removeprefix("python/").removesuffix(".py").replace("/", ".")
    return value.removesuffix(".__init__")


def _resolve_import(current: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    parts = current.split(".")
    if not current.endswith(".__init__"):
        parts = parts[:-1]
    keep = max(0, len(parts) - node.level + 1)
    prefix = parts[:keep]
    suffix = [] if node.module is None else node.module.split(".")
    return ".".join([*prefix, *suffix])


def internal_import_graph(paths: list[str]) -> dict[str, set[str]]:
    modules = {module_name(path): path for path in paths}
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for name, path in modules.items():
        tree = ast.parse((ROOT / path).read_text(), filename=path)
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_import(name, node)
                if base:
                    candidates = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            for candidate in candidates:
                probe = candidate
                while probe and probe not in modules:
                    probe = probe.rpartition(".")[0]
                if probe in modules:
                    graph[name].add(probe)
    return graph


TEST_DELETE_RESEARCH_FILES = {
    "tests/benchmark/test_deepseek_v4_perf_matrix.py",
    "tests/benchmark/test_deepseek_v4_text_smoke.py",
    "tests/core/test_scheduler.py",
    "tests/kernel/test_index.py",
    "tests/kernel/test_store.py",
    "tests/kernel/test_tensor.py",
}
TEST_DELETE_DEBUG_FILES = {"tests/utils/test_dsv4_long_prefill_timing.py"}
TEST_ORACLE_FILES = {
    "tests/kernel/test_comm.py",
    "tests/kernel/test_deepseek_v4_c128_prefill_metadata.py",
}
TEST_DELETE_DEBUG_CASES = {
    "test_dsv4_prep_metadata_oracle_splits_pre_and_post_forward_c4_sparse_boundary",
    "test_dsv4_independent_swa_metadata_rejects_tombstone_inside_active_length",
    "test_dsv4_independent_swa_metadata_rejects_dummy_page_for_real_row",
    "test_dsv4_independent_swa_metadata_rejects_zero_refcount_page",
    "test_dsv4_independent_swa_metadata_rejects_free_page_inside_active_length",
    "test_cuda_graph_replay_timing_records_batch_and_padded_buckets",
    "test_deepseek_v4_release_guard_integrity_reports_mutation",
}
TEST_DELETE_RESEARCH_CASES = {
    "test_dsv4_capture_compressed_locs_graph_hook_can_be_disabled",
    "test_deepseek_v4_release_defaults_honor_explicit_sm80_env",
    "test_explicit_env_research_path_retains_legacy_graph_fallback",
    "test_marlin_wna16_release_credit_rejects_after_kv_timing",
    "test_dsv4_kernel_inventory_covers_sglang_main_exports",
    "test_dsv4_sm80_v0_bf16_bundle_env_policy",
    "test_dsv4_sm80_v1_moe_bundle_env_policy",
    "test_dsv4_sm80_moe_v2_bundle_env_policy",
    "test_dsv4_sm80_moe_vllm_runner_bundle_env_policy",
    "test_decode_metadata_deforest_component_tables_match_oracle",
    "test_direct_c4_sparse_metadata_for_replay_component_tables_match_oracle",
    "test_dsv4_unsupported_sm80_paths_fail_clearly",
    "test_rms_norm_pair_sm80_triton_opt_in_matches_fallback",
    "test_quantized_linear_fp8_per_call_gemm_matches_fallback",
    "test_static_projection_scale_cache_preserves_projection_outputs",
    "test_deepseek_v4_v1_moe_sums_routed_and_shared_before_one_all_reduce",
    "test_deepseek_v4_v1_moe_reduce_once_bf16_opt_in",
    "test_deepseek_v4_moe_v2_builds_execution_plan_before_reduce_once",
    "test_shared_experts_marlin_down_skips_bf16_down_cache_and_releases_original",
}


def test_case_classification(path: str, name: str) -> tuple[str, str, list[str], str]:
    if path in TEST_DELETE_RESEARCH_FILES or name in TEST_DELETE_RESEARCH_CASES:
        return (
            "DELETE_RESEARCH",
            "audited research/export/alternate-policy case absent from both product modes",
            [],
            "delete in misc 02",
        )
    if path in TEST_DELETE_DEBUG_FILES or name in TEST_DELETE_DEBUG_CASES:
        return (
            "DELETE_DEBUG",
            "development timing/integrity/bounds instrumentation leaves with its implementation",
            [],
            "delete or rewrite with the debug implementation in misc 02",
        )
    if path in TEST_ORACLE_FILES:
        return (
            "KEEP_ORACLE",
            "focused retained operator/communication oracle",
            ["FB-3", "FB-4", "FB-6", "FB-7"],
            "retain/narrow to correctness",
        )
    if path.endswith("test_deepseek_v4_forward_fallback.py"):
        if "vllm_runner" in name or "prepare_" in name or "marlin_release" in name:
            return (
                "KEEP_RELEASE",
                "promoted optimized runner or Marlin lifecycle contract",
                ["OPT-8", "OPT-10"],
                "retain",
            )
        return (
            "KEEP_ORACLE",
            "explicit full-model/fallback model semantic oracle",
            ["FB-1", "FB-2", "FB-5"],
            "retain",
        )
    if name == "test_dsv4_sm80_opt_in_kernels_match_fallbacks":
        return (
            "KEEP_ORACLE",
            "mixed retained CUDA oracle; file existence does not retain its research subcases",
            ["FB-2", "FB-3", "FB-4", "FB-5", "FB-7"],
            "split; retain promoted subcases and remove standalone FP4/FP8 research subcases",
        )
    if name == "test_dsv4_sm80_v0_bf16_bundle_kernels_match_fallbacks":
        return (
            "KEEP_ORACLE",
            "mixed historical bundle oracle; retain only kernels reached by the hardened product matrix",
            ["FB-3", "FB-4", "FB-5", "FB-7"],
            "split; remove standalone RMS-pair/Q-norm/K-norm/store/BF16-indexer subcases",
        )
    if name == "test_indexer_bf16_query_logits_and_topk_are_fallback_clean":
        return (
            "KEEP_ORACLE",
            "Torch BF16 reference for fallback indexer/top-k semantics",
            ["FB-4"],
            "retain Torch reference only; it grants no authority to the unused Triton BF16 indexer kernel",
        )
    value = name.lower()
    if any(
        token in value for token in ("fallback", "oracle", "matches_reference", "matches_torch")
    ):
        return (
            "KEEP_ORACLE",
            "explicit reference comparison for a retained semantic surface",
            ["FB-3", "FB-4", "FB-5", "FB-7"],
            "retain",
        )
    action = "retain"
    if name == "test_dsv4_route_b_component_page_table_lifetime_cache_invalidates_lifecycle":
        action = "retain lifecycle assertion; remove development verifier dependency"
    if name == "test_marlin_wna16_before_kv_release_credit_adds_net_pages":
        action = "retain before-KV typed invariant; remove selectable debug timing spelling"
    return (
        "KEEP_RELEASE",
        "audited retained DSV4 serving/runtime contract",
        ["OPT-1", "OPT-3", "OPT-4", "OPT-8", "OPT-9"],
        action,
    )


def test_nodes(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    result: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            result.append((node.name, node))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            result.extend(
                (f"{node.name}.{child.name}", child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
            )
    return result


def dependency_manifest() -> dict[str, Any]:
    module_paths = sorted(set(tracked("python/minisgl/*.py") + tracked("python/minisgl/**/*.py")))
    graph = internal_import_graph(module_paths)
    roots = {
        "minisgl.__main__",
        "minisgl.llm.llm",
        "minisgl.shell",
        "minisgl.server.api_server",
        "minisgl.server.launch",
        "minisgl.benchmark.client",
        "minisgl.models.deepseek_v4",
        "minisgl.models.register",
    }
    modules: list[dict[str, Any]] = []
    for path in module_paths:
        name = module_name(path)
        cleanup_debt: list[str] = []
        if path in MODULE_DEBUG_OVERRIDES:
            classification, reason = "DELETE_DEBUG", "DSV4 runtime diagnostics removed in misc 02"
            evidence_ids: list[str] = []
        elif path in MODULE_DELETE_OVERRIDES:
            classification, reason = (
                "DELETE_RESEARCH",
                "audited unsupported model/generic layer/backend module absent from both product modes",
            )
            evidence_ids = []
        elif path in MODULE_ORACLE_OVERRIDES:
            classification, reason = "KEEP_ORACLE", "explicit fallback cache implementation"
            evidence_ids = ["FB-1"]
        elif path in MODULE_SHARED_BUILD_OVERRIDES:
            classification, reason = (
                "KEEP_SHARED_BUILD",
                "shared AOT/JIT loader needed by retained DSV4 and PyNCCL translation units",
            )
            evidence_ids = ["BUILD-NATIVE-LOADER"]
        else:
            classification = "KEEP_RELEASE"
            reason = "audited DSV4/public-root runtime module after explicit 29-module research cut"
            if path.startswith("python/minisgl/distributed/"):
                evidence_ids = ["OPT-10", "FB-6"]
            elif path.startswith(
                ("python/minisgl/attention/", "python/minisgl/kernel/", "python/minisgl/kvcache/")
            ):
                evidence_ids = ["OPT-4", "OPT-9", "FB-3", "FB-4"]
            elif path.startswith("python/minisgl/models/"):
                evidence_ids = ["OPT-1", "OPT-8", "FB-1"]
            else:
                evidence_ids = ["PUBLIC-ROOT-IMPORT", "OPT-1", "FB-1"]
        if path in {
            "python/minisgl/attention/__init__.py",
            "python/minisgl/kernel/__init__.py",
            "python/minisgl/layers/__init__.py",
            "python/minisgl/models/__init__.py",
        }:
            cleanup_debt.append(
                "narrow aggregate imports/exports with deleted modules during misc 02"
            )
        if path == "python/minisgl/core.py":
            cleanup_debt.append(
                "remove unconsumed Context.moe_backend initialization with generic MoE"
            )
        if path == "python/minisgl/layers/base.py":
            cleanup_debt.append(
                "remove unconsumed set_rope_device edge when generic rotary layer leaves"
            )
        modules.append(
            {
                "path": path,
                "module": name,
                "classification": classification,
                "reason": reason,
                "evidence_ids": evidence_ids,
                "internal_imports": sorted(graph[name]),
                "cleanup_debt": cleanup_debt,
            }
        )

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    dependency_policy = {
        "accelerate": (
            "DELETE_RESEARCH",
            "actual blocked-import/package-build probe passes; no retained minisgl root imports accelerate",
        ),
        "flashinfer-python": (
            "KEEP_RELEASE",
            "engine/sample.py::sample_impl directly imports flashinfer.sampling; OPT-2 executes it",
        ),
        "quack-kernels": (
            "DELETE_RESEARCH",
            "no tracked production or public benchmark import/reference remains",
        ),
    }
    dependencies: list[dict[str, Any]] = []
    for value in declared:
        base = re.split(r"[<>=!~\[]", value, 1)[0]
        classification, reason = dependency_policy.get(
            base,
            ("KEEP_RELEASE", "direct retained runtime/server/CLI/benchmark/native-build import"),
        )
        evidence = (
            ["OPT-2"]
            if base == "flashinfer-python"
            else (
                ["IMPORT-PROBE-ACCELERATE"]
                if base == "accelerate"
                else (
                    ["OPT-9", "OPT-10"]
                    if base == "apache-tvm-ffi"
                    else (
                        ["OPT-1", "OPT-9"]
                        if base == "sgl_kernel"
                        else (
                            ["PUBLIC-ROOT-IMPORT", "OPT-1", "FB-1"]
                            if classification.startswith("KEEP")
                            else []
                        )
                    )
                )
            )
        )
        dependencies.append(
            {
                "name": value,
                "base_name": base,
                "classification": classification,
                "reason": reason,
                "evidence_ids": evidence,
            }
        )

    supplemental_direct_imports = [
        {
            "base_name": name,
            "classification": "KEEP_RELEASE",
            "reason": reason,
            "evidence_ids": evidence,
            "declared_in_project_dependencies": False,
            "packaging_action": action,
        }
        for name, reason, evidence, action in [
            (
                "safetensors",
                "DSV4 checkpoint weight loader imports safetensors.torch",
                ["OPT-1", "IMPORT-PROBE-ACCELERATE"],
                "declare explicitly or document transformers transitive ownership",
            ),
            (
                "huggingface_hub",
                "retained config/model download utility imports it directly",
                ["PUBLIC-ROOT-IMPORT"],
                "declare explicitly or document transformers transitive ownership",
            ),
            (
                "tqdm",
                "retained loading and communication surfaces import tqdm directly",
                ["OPT-1", "OPT-10"],
                "declare explicitly",
            ),
            (
                "numpy",
                "retained engine/model utilities import numpy directly",
                ["OPT-1", "FB-1"],
                "declare explicitly",
            ),
            (
                "psutil",
                "retained process/runtime utilities import psutil directly",
                ["PUBLIC-ROOT-IMPORT"],
                "declare explicitly",
            ),
            (
                "pydantic",
                "retained API/message schemas import pydantic directly",
                ["PUBLIC-ROOT-IMPORT"],
                "declare explicitly or document fastapi transitive ownership",
            ),
            (
                "starlette",
                "retained API server imports starlette directly",
                ["PUBLIC-ROOT-IMPORT"],
                "declare explicitly or document fastapi transitive ownership",
            ),
            (
                "triton",
                "retained DSV4 sm80 kernels import Triton directly",
                ["OPT-3", "OPT-9", "FB-7"],
                "declare supported Triton ownership through torch/runtime constraints",
            ),
            (
                "pyarrow",
                "public bench_wildchat imports pyarrow at module scope",
                ["PUBLIC-BENCHMARK-IMPORT"],
                "promote from dev dependency or make the benchmark import lazy in misc 04",
            ),
        ]
    ]

    test_paths = sorted(set(tracked("tests/*.py") + tracked("tests/**/*.py")))
    tests: list[dict[str, Any]] = []
    for path in test_paths:
        tree = ast.parse((ROOT / path).read_text(), filename=path)
        cases = []
        for qualified_name, node in test_nodes(tree):
            classification, reason, evidence_ids, action = test_case_classification(path, node.name)
            cases.append(
                {
                    "name": qualified_name,
                    "line": node.lineno,
                    "classification": classification,
                    "reason": reason,
                    "evidence_ids": evidence_ids,
                    "misc02_action": action,
                }
            )
        case_counts = Counter(case["classification"] for case in cases)
        if path in TEST_DELETE_RESEARCH_FILES:
            classification = "DELETE_RESEARCH"
            disposition = "delete file"
        elif path in TEST_DELETE_DEBUG_FILES:
            classification = "DELETE_DEBUG"
            disposition = "delete file with debug timing implementation"
        elif path == "tests/kernel/test_comm.py":
            classification = "KEEP_ORACLE"
            disposition = "rewrite manual harness to BF16 all-reduce/all-gather correctness; remove timing loop/unrelated dtypes"
        elif not cases:
            classification = (
                "KEEP_SHARED_BUILD" if path.endswith("conftest.py") else "REVIEW_BLOCKED"
            )
            disposition = (
                "retain fixture"
                if classification.startswith("KEEP")
                else "resolve empty test owner before misc 02"
            )
        elif any(name.startswith("DELETE") for name in case_counts) and any(
            name.startswith("KEEP") for name in case_counts
        ):
            classification = "KEEP_RELEASE" if case_counts["KEEP_RELEASE"] else "KEEP_ORACLE"
            disposition = "narrow file: retain KEEP cases and delete/rewrite DELETE cases"
        elif len(case_counts) == 1:
            classification = next(iter(case_counts))
            disposition = "delete file" if classification.startswith("DELETE") else "retain file"
        else:
            classification = "KEEP_RELEASE" if case_counts["KEEP_RELEASE"] else "KEEP_ORACLE"
            disposition = "retain mixed release/oracle contracts"
        tests.append(
            {
                "path": path,
                "classification": classification,
                "disposition": disposition,
                "case_classification_counts": dict(sorted(case_counts.items())),
                "evidence_ids": (
                    ["FB-6"]
                    if path == "tests/kernel/test_comm.py"
                    else sorted(
                        {
                            evidence
                            for case in cases
                            if case["classification"].startswith("KEEP")
                            for evidence in case["evidence_ids"]
                        }
                    )
                ),
                "cases": cases,
            }
        )

    return {
        "schema_version": 2,
        "base_commit": BASE_COMMIT,
        "public_roots": sorted(roots),
        "module_classification_counts": counts(modules),
        "python_modules": modules,
        "dependency_classification_counts": counts(dependencies),
        "python_dependencies": dependencies,
        "supplemental_direct_imports": supplemental_direct_imports,
        "accelerate_probe": {
            "status": "PASS_WITHOUT_ACCELERATE",
            "method": "meta_path import blocker over retained roots plus git-archive wheel build/install --no-deps",
            "observed_import": False,
            "surfaces": [
                "LLM/shell/server/engine imports",
                "DeepSeek-V4 config and real tokenizer",
                "TP8 meta model construction",
                "real checkpoint first shard: model.embed_tokens.weight [16160,4096] BF16",
                "ModelScope snapshot_download",
            ],
            "wheel_build_install": "PASS",
            "classification": "DELETE_RESEARCH",
            "evidence_id": "IMPORT-PROBE-ACCELERATE",
        },
        "test_file_classification_counts": counts(tests),
        "tests": tests,
        "required_corrections": {
            "flashinfer_sampling": "KEEP_RELEASE",
            "tests/utils/test_dsv4_long_prefill_timing.py": "DELETE_DEBUG",
            "research_cases_in_mixed_wrapper_tests": "narrow/delete per case; file existence is not KEEP authority",
        },
    }


def benchmark_manifest() -> dict[str, Any]:
    files = tracked("benchmark")
    required = {
        "benchmark/offline/bench.py",
        "benchmark/offline/bench_wildchat.py",
        "benchmark/online/bench_qwen.py",
        "benchmark/online/bench_simple.py",
    }
    entries = []
    for path in files:
        if path in required:
            classification = "KEEP_RELEASE"
            benchmark_classification = "PUBLIC_KEEP"
            reason = "one of the four required release benchmark entry files"
        else:
            classification = "DELETE_RESEARCH"
            benchmark_classification = "MOVE_TO_DEBUG"
            reason = "developer smoke/microbenchmark; never kernel KEEP evidence"
        entries.append(
            {
                "path": path,
                "classification": classification,
                "benchmark_classification": benchmark_classification,
                "final_disposition": benchmark_classification,
                "reason": reason,
                "evidence_ids": (
                    ["PUBLIC-BENCHMARK-ROOT"] if classification == "KEEP_RELEASE" else []
                ),
                "kernel_keep_authority": False,
            }
        )
    return {
        "schema_version": 2,
        "base_commit": BASE_COMMIT,
        "entries": entries,
        "classification_counts": counts(entries),
        "benchmark_classification_counts": dict(
            sorted(Counter(e["benchmark_classification"] for e in entries).items())
        ),
        "public_entries": sorted(required),
        "rule": "historical microbench/export/test/source-glob presence never grants kernel KEEP",
    }


def readme(
    coverage: dict[str, Any],
    env: dict[str, Any],
    kernels: dict[str, Any],
    deps: dict[str, Any],
) -> str:
    c = kernels["combined_classification_counts"]
    cases = coverage["cases"]
    table = "\n".join(
        f"| {case['id']} | {case['mode']} | {case['status']} | {case['surface']} |"
        for case in cases
    )
    retained_triton = kernels["retained_kernel_sets"]["triton"]
    retained_native = kernels["retained_kernel_sets"]["native_translation_units"]
    return f"""# DSV4 Census Manifest Hardening

## Verdict

**GO for TARGET misc 02, limited to entries classified `DELETE_RESEARCH` or
`DELETE_DEBUG`.**  The hardened inventory has no `REVIEW_BLOCKED` entry, every
KEEP entry names release/oracle/shared-build evidence, and the narrowed Marlin
source set builds, loads, and passes its oracle.  This milestone does not delete
production code and does not create a release tag.

The old census incorrectly called two string defaults dead and did not count
generic review states honestly.  This pass retains all five required typed
values/behaviors:

- `MOE_EXPERT_BACKEND=marlin_wna16`;
- `DIRECT_GRAPH_METADATA_GROUPS=swa,c4` (C128 remains deliberately excluded);
- Marlin release timing `before_kv_alloc`;
- page-allocation clear scope `component`;
- PyNCCL threshold `33,554,432` bytes (32 MiB).

No required release behavior remains misclassified as dead.

## Immutable baseline

```text
cleanup tag:       {BASE_TAG}
cleanup commit:    {BASE_COMMIT}
performance tag:  {PERF_TAG} -> {git('rev-list', '-n', '1', PERF_TAG)} (unchanged)
package:           {PACKAGE}
release tag:       v0.1.0-dsv4-sm80 absent/not created
production edits:  none
```

The existing prompt edits and initially untracked census helpers were preserved;
only the two helpers and this ignored milestone directory were changed.

## Deletion-authority counts

Counts below cover 601 Python callable/kernel definitions plus 52 native
source/header entries, each with one unique ID:

| Classification | Count |
| --- | ---: |
| `KEEP_RELEASE` | {c.get('KEEP_RELEASE', 0)} |
| `KEEP_ORACLE` | {c.get('KEEP_ORACLE', 0)} |
| `KEEP_SHARED_BUILD` | {c.get('KEEP_SHARED_BUILD', 0)} |
| `DELETE_RESEARCH` | {c.get('DELETE_RESEARCH', 0)} |
| `DELETE_DEBUG` | {c.get('DELETE_DEBUG', 0)} |
| `REVIEW_BLOCKED` | {c.get('REVIEW_BLOCKED', 0)} |

Computed from all {env['entry_count']} env entries: `UNKNOWN_REVIEW` count is
`{env['unknown_review_count']}` and its hot-path count is
`{env['unknown_review_hot_path_count']}`.  Neither value is hard-coded.

## Runtime coverage matrix

| Case | Mode | Result | Surface |
| --- | --- | --- | --- |
{table}

Important attribution boundaries:

- The short fallback full-model smoke covers all layers and raw-weight Torch
  MoE, but does not create a nonzero C128 compressed surface.  FB-3 supplies the
  explicit C128 reference oracle.
- Fallback disables `MOE_ROUTE`; grouped Triton coverage is the separate FB-2
  representative CUDA oracle, not falsely attributed to FB-1.
- OPT-2 is the only new full-model probe.  It executes non-greedy
  `flashinfer.sampling`, which is why `flashinfer-python` is `KEEP_RELEASE`.

## Wrapper and kernel authority

`wrapper_kernel_map.json` contains all 601 definitions plus ordinary/property
edges, {len([x for x in kernels['triton_kernels'] if x.get('is_triton_jit')])}
Triton JIT definitions, every detected `kernel[grid]` launch, torch custom-op
registration/schema line, TVM FFI/JIT loader/registration, dynamic loader edge,
and Marlin/native translation unit.  A
private kernel inherits a case only through its concrete retained wrapper and
launch predicate.

The retained Triton set has {len(retained_triton)} definitions (including
device helpers); the exact names are in `kernel_source_manifest.json`.  The
retained native translation-unit set has {len(retained_native)} files.  Dense
FP8 Marlin, unsupported projection kernels, generic fused-MoE code, and unused
Marlin FP16/U4/U8/S8/FE4M3 variants are DELETE candidates even if an old test,
export, microbench, or glob mentions them.

## Marlin WNA16 narrowing

The actual dispatch is MoE
`BF16 / FE2M1f(MXFP4) / BF16 / E8M0`, group-blocks 2.  Nsight evidence contains
22,016 repack launches and only this scalar tuple.  The sole required variant TU
is:

```text
python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16/moe/marlin_moe_wna16/
sm80_kernel_bfloat16_fe2m1f_bfloat16.cu
```

It is built with `schema.cpp`, `gptq_marlin_repack.cu`, and `ops.cu`.  The
generated `kernel_selector.h` must be filtered/regenerated at the same time;
simply shortening the Python glob leaves undefined references to the other 13
TUs.  A temporary filtered source tree reduced the selector from 1,467 to 60
lines (30 conditions), built four TUs in 92.262 seconds, loaded a 5,590,160-byte
extension with no unresolved Marlin symbol, and passed a real checkpoint
layer-0 TP8-rank0 BF16 MXFP4 oracle (max abs diff 0.0078125) without modifying
the production loader.

## Dependencies, modules, tests, benchmarks

- `flashinfer-python>=0.5.3`: `KEEP_RELEASE`, directly executed by OPT-2
  sampling.
- `accelerate`: `DELETE_RESEARCH`; retained-root imports and metadata build run
  with an explicit import blocker and never import it.
- `quack-kernels`: `DELETE_RESEARCH`; no tracked retained root references it.
- All 104 modules were audited against the import graph and product roots: 70
  are retained, 29 are `DELETE_RESEARCH`, and five are `DELETE_DEBUG`; aggregate
  imports and two stale initialization edges are recorded as misc-02 debt.
- `tests/utils/test_dsv4_long_prefill_timing.py` is `DELETE_DEBUG`.  Mixed test
  files carry per-test dispositions so research/debug cases can be removed or
  rewritten with their implementation.
- Exactly four benchmark files are `KEEP_RELEASE`/`PUBLIC_KEEP`; the other 20
  are `DELETE_RESEARCH`/`MOVE_TO_DEBUG` and have no kernel-KEEP authority.

## Validation

```text
python -m compileall -q debug/release_cleanup                                      PASS
python debug/release_cleanup/build_manifests.py                                   PASS
pytest release-default/wrapper/attention required gate                            PASS (107 passed)
non-greedy TP8 optimized text smoke, temperature=.6 top_p=.9                      PASS
narrow Marlin one-variant build/load/BF16-MXFP4 oracle                            PASS
accelerate blocked-import/build probe                                              PASS
```

Mechanical checks pass: all release key/value pairs are present, all KEEP
entries have evidence, every private Triton/native symbol is mapped or DELETE,
no generic REVIEW reason exists, FlashInfer sampling is retained, and blocked
counts are derived from entries.

## Stop boundary

Stop here.  TARGET misc 02 has not begun, no production source was deleted, and
neither `v0.0.0` nor any release tag was moved/created.  Misc 02 is authorized
to consume these manifests, but only within the classifications above.
"""


def validate_payloads(payloads: dict[str, dict[str, Any]]) -> None:
    env = payloads["env_toggle_manifest.json"]
    runtime = payloads["runtime_values.json"]
    kernel = payloads["kernel_source_manifest.json"]
    wrapper = payloads["wrapper_kernel_map.json"]
    required_values = {
        "MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND": "marlin_wna16",
        "MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS": "swa,c4",
        "MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING": "before_kv_alloc",
        "MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC": "component",
        "MINISGL_PYNCCL_MAX_BUFFER_SIZE": 33554432,
    }
    actual = {entry["key"]: entry["resolved_value"] for entry in runtime["release_default_values"]}
    if any(actual.get(key) != value for key, value in required_values.items()):
        raise RuntimeError(f"release value mismatch: expected {required_values}, got {actual}")
    if env["unknown_review_count"] != sum(
        entry["classification"] == "UNKNOWN_REVIEW" for entry in env["entries"]
    ):
        raise RuntimeError("unknown count is not entry-derived")
    if env["unknown_review_hot_path_count"] != sum(
        entry["classification"] == "UNKNOWN_REVIEW"
        and any(
            phase in {"graph_replay", "graph_replay/prefill/decode", "prefill/decode"}
            for phase in entry["reader_phases"]
        )
        for entry in env["entries"]
    ):
        raise RuntimeError("hot-path unknown count is not entry-derived")
    bad_env_classes = [
        entry["name"]
        for entry in env["entries"]
        if entry["final_classification"] not in FINAL_CLASSES
    ]
    if bad_env_classes:
        raise RuntimeError(f"invalid env final classifications: {bad_env_classes[:5]}")
    combined = [*wrapper["callables"], *kernel["native_sources"]]
    bad_classes = [
        entry["id"] for entry in combined if entry["classification"] not in FINAL_CLASSES
    ]
    if bad_classes:
        raise RuntimeError(f"invalid final classifications: {bad_classes[:5]}")
    missing_evidence = [
        entry["id"]
        for entry in combined
        if entry["classification"].startswith("KEEP") and not entry["evidence_ids"]
    ]
    if missing_evidence:
        raise RuntimeError(f"KEEP entries lack evidence: {missing_evidence[:10]}")
    if wrapper["review_blocked_count"] or kernel["review_blocked_count"]:
        raise RuntimeError("unresolved wrapper/kernel registration mapping remains")
    launched = {entry["private_kernel"] for entry in wrapper["triton_launch_maps"]}
    unmapped_triton = [
        entry["id"]
        for entry in kernel["triton_kernels"]
        if entry["classification"] in {"KEEP_RELEASE", "KEEP_ORACLE"}
        and f"{entry['path']}::{entry['symbol']}" not in launched
    ]
    if unmapped_triton:
        raise RuntimeError(f"retained Triton kernels lack grid mapping: {unmapped_triton}")
    incomplete_launch_chains = [
        entry["id"]
        for entry in wrapper["triton_launch_maps"]
        if entry["classification"] in {"KEEP_RELEASE", "KEEP_ORACLE"}
        and (not entry["public_wrappers"] or not entry["model_attention_owners"])
    ]
    if incomplete_launch_chains:
        raise RuntimeError(
            f"retained Triton launch lacks model/public owner chain: {incomplete_launch_chains}"
        )
    if len(wrapper["native_custom_op_tvm_ffi_registrations"]) != 39:
        raise RuntimeError("native registration census changed without review")
    if len(wrapper["python_custom_op_and_jit_registrations"]) != 16:
        raise RuntimeError("Python loader/registration census changed without review")
    serialized = json.dumps(payloads)
    if (
        '"classification": "REVIEW"' in serialized
        or "dynamic dispatch/export/JIT use cannot be disproved" in serialized
    ):
        raise RuntimeError("generic REVIEW survived hardening")
    flashinfer = next(
        entry
        for entry in payloads["model_dependency_manifest.json"]["python_dependencies"]
        if entry["base_name"] == "flashinfer-python"
    )
    if flashinfer["classification"] != "KEEP_RELEASE":
        raise RuntimeError("flashinfer sampling dependency is not retained")
    deps = payloads["model_dependency_manifest.json"]
    expected_modules = {
        "KEEP_RELEASE": 68,
        "KEEP_ORACLE": 1,
        "KEEP_SHARED_BUILD": 1,
        "DELETE_RESEARCH": 29,
        "DELETE_DEBUG": 5,
    }
    if deps["module_classification_counts"] != expected_modules:
        raise RuntimeError(
            f"module audit drift: expected {expected_modules}, got {deps['module_classification_counts']}"
        )
    timing_test = next(
        entry
        for entry in deps["tests"]
        if entry["path"] == "tests/utils/test_dsv4_long_prefill_timing.py"
    )
    if timing_test["classification"] != "DELETE_DEBUG":
        raise RuntimeError("debug timing test was not scheduled with its implementation")
    benchmarks = payloads["benchmark_manifest.json"]
    if benchmarks["benchmark_classification_counts"] != {
        "MOVE_TO_DEBUG": 20,
        "PUBLIC_KEEP": 4,
    }:
        raise RuntimeError("benchmark public/developer disposition drift")
    marlin = kernel["marlin_wna16"]
    if (
        marlin["narrowed_build_load_oracle"]["status"] != "PASS"
        or len(kernel["retained_kernel_sets"]["marlin_sm80"]) != 1
    ):
        raise RuntimeError("Marlin narrowed build/oracle evidence is incomplete")
    mapped_native = {
        path
        for entry in wrapper["custom_tvm_marlin_maps"]
        for path in entry.get("translation_units", [])
        if path.startswith("python/minisgl/kernel/csrc/")
    }
    missing_native_maps = sorted(
        set(kernel["retained_kernel_sets"]["native_translation_units"]) - mapped_native
    )
    if missing_native_maps:
        raise RuntimeError(f"retained native TUs lack owner maps: {missing_native_maps}")


def main() -> None:
    baseline = _assert_baseline()
    raw = build_census(ROOT)
    coverage = runtime_coverage()
    callables = callable_inventory(raw)
    env = env_manifest(raw)
    deps = dependency_manifest()
    benchmarks = benchmark_manifest()
    wrappers = wrapper_kernel_map(raw, callables)
    kernels = kernel_source_manifest(raw, callables)
    payloads = {
        "runtime_values.json": runtime_values(raw),
        "runtime_coverage.json": coverage,
        "wrapper_kernel_map.json": wrappers,
        "kernel_source_manifest.json": kernels,
        "env_toggle_manifest.json": env,
        "model_dependency_manifest.json": deps,
        "benchmark_manifest.json": benchmarks,
    }
    for payload in payloads.values():
        payload.setdefault("baseline", baseline)
    validate_payloads(payloads)
    for name, payload in payloads.items():
        write(name, payload)
    (OUT / "README.md").write_text(readme(coverage, env, kernels, deps))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import minisgl.core as core  # noqa: E402
from minisgl.core import Req, SamplingParams  # noqa: E402
from minisgl.kvcache import create_kvcache_pool  # noqa: E402
from minisgl.kvcache.deepseek_v4_pool import DeepSeekV4KVCache  # noqa: E402
from minisgl.models.config import ModelConfig, RotaryConfig  # noqa: E402
from minisgl.scheduler.cache import CacheManager  # noqa: E402
from minisgl.scheduler.utils import PendingReq  # noqa: E402
from minisgl.utils import align_down  # noqa: E402

MILESTONE_DIR = REPO_ROOT / "performance_milestones" / "target08_independent_compressed_indexer_ownership"
RAW_DIR = MILESTONE_DIR / "raw"
SUMMARY_DIR = MILESTONE_DIR / "summaries"


def _tiny_dsv4_config(compress_ratios: list[int], *, window_size: int = 128) -> ModelConfig:
    return ModelConfig(
        num_layers=len(compress_ratios),
        num_qo_heads=4,
        num_kv_heads=1,
        head_dim=8,
        hidden_size=16,
        vocab_size=32,
        intermediate_size=0,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(8, 2, 512, 10000.0, None),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        norm_topk_prob=True,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        q_lora_rank=4,
        o_lora_rank=4,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=8,
        window_size=window_size,
        compress_ratios=compress_ratios,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=2,
        n_routed_experts=2,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        expert_dtype="fp4",
        routed_scaling_factor=1.5,
        hc_mult=1,
        hc_sinkhorn_iters=1,
        o_groups=1,
        n_hash_layers=0,
    )


def _make_component_cache(
    *,
    num_pages: int = 8,
    page_size: int = 128,
) -> tuple[DeepSeekV4KVCache, CacheManager]:
    core._GLOBAL_CTX = None
    page_table = torch.empty((16, num_pages * page_size), dtype=torch.int32)
    pool = create_kvcache_pool(
        _tiny_dsv4_config([4, 128, 0], window_size=page_size),
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cpu"),
        enable_dsv4_component_loc_ownership=True,
    )
    assert isinstance(pool, DeepSeekV4KVCache)

    ctx = core.Context(page_size=page_size)
    ctx.page_table = page_table
    ctx.kv_cache = pool
    core.set_global_ctx(ctx)

    cm = CacheManager(num_pages, page_size, page_table, type="radix", kv_cache=pool)
    return pool, cm


def _allocate_req_with_ids(cm: CacheManager, uid: int, input_ids: torch.Tensor) -> Req:
    sampling_params = SamplingParams(max_tokens=1)
    handle = cm.match_req(PendingReq(uid=uid, input_ids=input_ids, sampling_params=sampling_params)).cuda_handle
    cm.lock(handle)
    table_idx = uid % cm.page_table.shape[0]
    if handle.cached_len > 0:
        cm.page_table[table_idx, : handle.cached_len].copy_(handle.get_matched_indices())
    req = Req(
        input_ids=input_ids,
        table_idx=table_idx,
        cached_len=handle.cached_len,
        output_len=1,
        uid=uid,
        sampling_params=sampling_params,
        cache_handle=handle,
    )
    cm.allocate_paged([req])
    return req


def _finish_req(cm: CacheManager, req: Req) -> None:
    req.cached_len = req.device_len
    cm.cache_req(req, finished=True)


def _cache_finished_prompt(cm: CacheManager, uid: int, input_ids: torch.Tensor) -> Req:
    req = _allocate_req_with_ids(cm, uid, input_ids)
    _finish_req(cm, req)
    cm.check_integrity()
    return req


def _evict_all_prefix(cm: CacheManager, pool: DeepSeekV4KVCache) -> None:
    size = cm.prefix_cache.size_info.evictable_size
    if size == 0:
        return
    evicted = cm.prefix_cache.evict(size)
    cm._record_prefix_eviction(evicted)
    valid = cm._valid_full_indices(evicted)
    if valid.numel() > 0:
        pool.on_token_indices_freed(valid, cm.page_size, free_components=False)
        cm.free_slots = torch.cat([cm.free_slots, valid[:: cm.page_size]])
    cm.check_integrity()


def _ids(length: int, *, base: int = 0) -> torch.Tensor:
    return torch.arange(length, dtype=torch.int32) + base


def _snapshot(cm: CacheManager, pool: DeepSeekV4KVCache) -> dict[str, Any]:
    counts = pool.allocation_counts
    metrics = cm.prefix_metrics_snapshot()
    component = metrics.get("dsv4_component_ownership", {})
    return {
        "available_size_tokens": int(cm.available_size),
        "free_full_pages": int(len(cm.free_slots)),
        "retained_prefix_tokens": int(metrics["retained_prefix_tokens"]),
        "evictable_prefix_tokens": int(metrics["evictable_prefix_tokens"]),
        "evictions": int(metrics["evictions"]),
        "full_pages_live": int(counts.full_slots // cm.page_size),
        "full_slots_live": int(counts.full_slots),
        "c4_slots_live": int(counts.c4_slots),
        "c128_slots_live": int(counts.c128_slots),
        "indexer_slots_live": int(counts.c4_indexer_slots),
        "component_available_pages": int(pool.available_component_pages()),
        "evictable_live_full_tokens": int(component.get("evictable_live_full_tokens", 0)),
        "evictable_component_tokens": int(component.get("evictable_component_tokens", 0)),
    }


def _check(checks: dict[str, bool], name: str, condition: bool) -> None:
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


def _finalize(pool: DeepSeekV4KVCache, cm: CacheManager) -> dict[str, Any]:
    _evict_all_prefix(cm, pool)
    pool.assert_no_leak()
    return _snapshot(cm, pool)


def scenario_full_partial_miss() -> dict[str, Any]:
    pool, cm = _make_component_cache(num_pages=8)
    checks: dict[str, bool] = {}
    before = _snapshot(cm, pool)

    base = _ids(260)
    _cache_finished_prompt(cm, 0, base)
    after_first = _snapshot(cm, pool)

    repeat = cm.match_req(PendingReq(1, base, SamplingParams(max_tokens=1))).cuda_handle
    _check(checks, "full_hit_260_to_256", repeat.cached_len == 256)

    branch = torch.cat([base[:256], _ids(130, base=10_000)])
    partial = cm.match_req(PendingReq(2, branch, SamplingParams(max_tokens=1))).cuda_handle
    _check(checks, "partial_hit_shared_two_pages", partial.cached_len == 256)

    miss = cm.match_req(PendingReq(3, _ids(260, base=20_000), SamplingParams(max_tokens=1))).cuda_handle
    _check(checks, "miss_is_zero", miss.cached_len == 0)

    _check(checks, "old_full_head_released", after_first["full_pages_live"] == 1)
    _check(checks, "component_pages_retained", after_first["component_available_pages"] == 6)
    after_cleanup = _finalize(pool, cm)
    return {
        "checks": checks,
        "snapshots": {
            "before": before,
            "after_first_insert": after_first,
            "after_cleanup": after_cleanup,
        },
    }


def scenario_reuse_isolation_and_double_free_guard() -> dict[str, Any]:
    pool, cm = _make_component_cache(num_pages=4)
    checks: dict[str, bool] = {}

    prompt = _ids(260)
    _cache_finished_prompt(cm, 0, prompt)
    after_prefix = _snapshot(cm, pool)

    handle = cm.match_req(PendingReq(1, prompt, SamplingParams(max_tokens=1))).cuda_handle
    matched = handle.get_matched_indices()
    component_pages = handle.get_dsv4_component_pages()
    assert component_pages is not None
    assert component_pages.c4_pages is not None
    retained_c4 = component_pages.c4_pages.clone()

    _check(checks, "released_head_is_tombstoned", bool(torch.all(matched[:128] < 0).item()))
    _check(checks, "live_tail_still_full_owned", bool(torch.all(matched[128:] >= 0).item()))
    _check(checks, "retained_component_pages_survive", retained_c4.tolist() == [0, 1])

    cm.free_slots = torch.sort(cm.free_slots).values
    reuse = _allocate_req_with_ids(cm, 2, _ids(64, base=30_000))
    reused_full_page = int(cm.page_table[reuse.table_idx, 0].item() // cm.page_size)
    reused_components = pool.make_component_page_handles(cm.page_table[reuse.table_idx, : cm.page_size], cm.page_size)
    assert reused_components is not None
    assert reused_components.c4_pages is not None
    _check(checks, "full_page_zero_reused", reused_full_page == 0)
    _check(
        checks,
        "component_page_not_reused_under_retained_prefix",
        int(reused_components.c4_pages[0].item()) not in retained_c4.tolist(),
    )
    _finish_req(cm, reuse)
    cm.check_integrity()
    after_reuse = _snapshot(cm, pool)

    _evict_all_prefix(cm, pool)
    after_evict = _snapshot(cm, pool)
    double_free_guarded = False
    try:
        pool.release_component_page_handles(component_pages)
    except RuntimeError as exc:
        double_free_guarded = "double free" in str(exc)
    _check(checks, "double_free_guarded", double_free_guarded)
    pool.assert_no_leak()
    after_cleanup = _snapshot(cm, pool)

    return {
        "checks": checks,
        "snapshots": {
            "after_prefix": after_prefix,
            "after_reuse": after_reuse,
            "after_evict": after_evict,
            "after_cleanup": after_cleanup,
        },
    }


def scenario_boundaries() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    for length in [255, 256, 257, 258]:
        pool, cm = _make_component_cache(num_pages=4)
        prompt = _ids(length)
        _cache_finished_prompt(cm, 0, prompt)
        handle = cm.match_req(PendingReq(1, prompt, SamplingParams(max_tokens=1))).cuda_handle
        expected_without_state_guard = align_down(max(length - 1, 0), cm.page_size)
        safe_hit = int(handle.cached_len)
        rows.append(
            {
                "prompt_len": length,
                "safe_hit_tokens": safe_hit,
                "full_derived_aligned_hit_tokens": int(expected_without_state_guard),
                "capacity": _snapshot(cm, pool),
            }
        )
        if length in (255, 257, 258):
            _check(checks, f"boundary_{length}_safe_hit_nonzero", safe_hit > 0)
        if length == 256:
            _check(checks, "boundary_256_state_guard_downgrades_to_zero", safe_hit == 0)
        _finalize(pool, cm)

    return {"checks": checks, "rows": rows}


def scenario_c4_c128_indexer_loc_boundaries() -> dict[str, Any]:
    pool, cm = _make_component_cache(num_pages=4)
    checks: dict[str, bool] = {}
    req = _allocate_req_with_ids(cm, 0, _ids(258))
    positions = torch.arange(req.device_len, dtype=torch.int32)
    full_locs = cm.page_table[req.table_idx, : req.device_len]

    c4 = pool.compressed_locs_from_full_locs(full_locs, 4, positions)
    c128 = pool.compressed_locs_from_full_locs(full_locs, 128, positions)
    indexer = pool.indexer_locs_from_full_locs(full_locs, positions)

    _check(checks, "c4_endpoint_count_258", c4.numel() == 64)
    _check(checks, "c128_endpoint_count_258", c128.numel() == 2)
    _check(checks, "indexer_endpoint_count_258", indexer.numel() == c4.numel())
    _check(checks, "c4_cross_page_locs_are_direct", c4[:35].tolist() == list(range(35)))
    _check(checks, "c128_page_boundary_locs_are_direct", c128.tolist() == [0, 1])
    _check(checks, "indexer_locs_are_separate_vector", indexer[:35].tolist() == list(range(35)))

    _finish_req(cm, req)
    after_finish = _snapshot(cm, pool)
    after_cleanup = _finalize(pool, cm)
    return {
        "checks": checks,
        "snapshots": {
            "after_finish": after_finish,
            "after_cleanup": after_cleanup,
        },
        "sample_locs": {
            "c4_first_40": c4[:40].tolist(),
            "c128": c128.tolist(),
            "indexer_first_40": indexer[:40].tolist(),
        },
    }


def scenario_repeated_hit_evict_cycles() -> dict[str, Any]:
    pool, cm = _make_component_cache(num_pages=4)
    checks: dict[str, bool] = {}
    cycle_rows: list[dict[str, Any]] = []
    for cycle in range(3):
        prompt = _ids(260, base=cycle * 10_000)
        _cache_finished_prompt(cm, cycle * 2, prompt)
        hit = _cache_finished_prompt(cm, cycle * 2 + 1, prompt)
        _check(checks, f"cycle_{cycle}_hit_256", hit.cache_handle.cached_len == 256)
        _check(checks, f"cycle_{cycle}_one_full_tail_page", pool.allocation_counts.full_slots == cm.page_size)
        before_evict = _snapshot(cm, pool)
        _evict_all_prefix(cm, pool)
        pool.assert_no_leak()
        cycle_rows.append(
            {
                "cycle": cycle,
                "before_evict": before_evict,
                "after_evict": _snapshot(cm, pool),
            }
        )
    _check(checks, "three_evictions_recorded", cm.prefix_metrics_snapshot()["evictions"] == 3)
    return {"checks": checks, "cycles": cycle_rows}


def scenario_multi_prefix_branching_guard() -> dict[str, Any]:
    pool, cm = _make_component_cache(num_pages=8)
    checks: dict[str, bool] = {}

    base = _ids(390)
    _cache_finished_prompt(cm, 0, base)
    after_base = _snapshot(cm, pool)

    branch = torch.cat([base[:256], _ids(134, base=40_000)])
    branch_handle = cm.match_req(PendingReq(1, branch, SamplingParams(max_tokens=1))).cuda_handle
    _check(checks, "branch_at_released_two_page_boundary_is_guarded", branch_handle.cached_len == 0)
    _cache_finished_prompt(cm, 1, branch)
    after_branch = _snapshot(cm, pool)

    full_base = cm.match_req(PendingReq(2, base, SamplingParams(max_tokens=1))).cuda_handle
    _check(checks, "full_original_path_still_hits_live_tail", full_base.cached_len == 384)
    _check(checks, "component_pages_are_shared_across_branch_prefix", after_branch["c4_slots_live"] == 4 * 32)
    _check(checks, "branching_keeps_two_live_full_tail_pages", after_branch["full_pages_live"] == 2)

    after_cleanup = _finalize(pool, cm)
    return {
        "checks": checks,
        "snapshots": {
            "after_base": after_base,
            "after_branch": after_branch,
            "after_cleanup": after_cleanup,
        },
    }


def scenario_eviction_pressure() -> dict[str, Any]:
    pool, cm = _make_component_cache(num_pages=3)
    checks: dict[str, bool] = {}
    rows: list[dict[str, Any]] = []
    for i in range(4):
        prompt = _ids(260, base=i * 50_000)
        _cache_finished_prompt(cm, i, prompt)
        cm.check_integrity()
        rows.append({"iteration": i, "snapshot": _snapshot(cm, pool)})
    metrics = cm.prefix_metrics_snapshot()
    _check(checks, "eviction_pressure_triggered_evictions", metrics["evictions"] > 0)
    _check(checks, "no_refcount_leak_under_pressure", pool.allocation_counts.full_slots == cm.page_size)
    after_cleanup = _finalize(pool, cm)
    return {"checks": checks, "rows": rows, "after_cleanup": after_cleanup}


SCENARIOS: dict[str, Callable[[], dict[str, Any]]] = {
    "full_partial_miss": scenario_full_partial_miss,
    "reuse_isolation_and_double_free_guard": scenario_reuse_isolation_and_double_free_guard,
    "page_boundaries_255_256_257_258": scenario_boundaries,
    "c4_c128_indexer_loc_boundaries": scenario_c4_c128_indexer_loc_boundaries,
    "repeated_hit_evict_cycles": scenario_repeated_hit_evict_cycles,
    "multi_prefix_branching_guard": scenario_multi_prefix_branching_guard,
    "eviction_pressure": scenario_eviction_pressure,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _write_outputs(results: dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "component_ownership_probe.json"
    summary_json = SUMMARY_DIR / "component_ownership_summary.json"
    summary_md = SUMMARY_DIR / "component_ownership_summary.md"

    raw_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    summary = {
        "all_passed": results["all_passed"],
        "scenario_count": len(results["scenarios"]),
        "passed": [name for name, row in results["scenarios"].items() if row["passed"]],
        "failed": [name for name, row in results["scenarios"].items() if not row["passed"]],
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Component Ownership Probe Summary",
        "",
        f"all_passed: `{str(summary['all_passed']).lower()}`",
        "",
        "| scenario | result | checks |",
        "| --- | --- | ---: |",
    ]
    for name, row in results["scenarios"].items():
        result = "pass" if row["passed"] else "fail"
        checks = row.get("checks", {})
        lines.append(f"| `{name}` | {result} | {sum(bool(v) for v in checks.values())}/{len(checks)} |")
    lines.append("")
    lines.append("Raw JSON: `raw/component_ownership_probe.json`")
    summary_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    torch.set_num_threads(1)
    results: dict[str, Any] = {
        "all_passed": True,
        "scenario_count": len(SCENARIOS),
        "scenarios": {},
    }
    for name, fn in SCENARIOS.items():
        try:
            payload = fn()
            row = {"passed": True, **payload}
        except Exception as exc:  # noqa: BLE001
            row = {
                "passed": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "checks": {},
            }
            results["all_passed"] = False
        results["scenarios"][name] = _jsonable(row)

    _write_outputs(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

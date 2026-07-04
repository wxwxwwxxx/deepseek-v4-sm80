from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import minisgl.core as core
import torch
from minisgl.core import Req, SamplingParams
from minisgl.kvcache import create_kvcache_pool
from minisgl.kvcache.deepseek_v4_pool import DeepSeekV4KVCache
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.utils import PendingReq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
SUMMARIES = ROOT / "summaries"


def _tiny_dsv4_config(compress_ratios: list[int]) -> ModelConfig:
    return ModelConfig(
        num_layers=len(compress_ratios),
        num_qo_heads=4,
        num_kv_heads=1,
        head_dim=8,
        hidden_size=16,
        vocab_size=32,
        intermediate_size=0,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(8, 2, 64, 10000.0, None),
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
        window_size=4,
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


def _make_pool(
    *,
    num_pages: int,
    page_size: int,
    component_ownership: bool,
) -> DeepSeekV4KVCache:
    pool = create_kvcache_pool(
        _tiny_dsv4_config([4, 128, 0]),
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cpu"),
        enable_dsv4_component_loc_ownership=component_ownership,
    )
    assert isinstance(pool, DeepSeekV4KVCache)
    return pool


def _make_cache_manager(
    pool: DeepSeekV4KVCache,
    *,
    num_pages: int,
    page_size: int,
) -> CacheManager:
    page_table = torch.full((8, num_pages * page_size), -1, dtype=torch.int32)
    ctx = core.Context(page_size=page_size)
    ctx.page_table = page_table
    ctx.kv_cache = pool
    core._GLOBAL_CTX = None
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type="radix", kv_cache=pool)


def _allocate_req_with_ids(cm: CacheManager, uid: int, input_ids: torch.Tensor) -> Req:
    sampling_params = SamplingParams(max_tokens=1)
    pending = PendingReq(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
    handle = cm.match_req(pending).cuda_handle
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


def _evict_all_prefix(cm: CacheManager, pool: DeepSeekV4KVCache, page_size: int) -> None:
    size = cm.prefix_cache.size_info.evictable_size
    if size == 0:
        return
    evicted = cm.prefix_cache.evict(size)
    cm._record_prefix_eviction(evicted)
    valid = cm._valid_full_indices(evicted)
    if valid.numel() > 0:
        pool.on_token_indices_freed(
            valid,
            page_size,
            free_components=not pool.component_loc_ownership_enabled,
        )
        cm.free_slots = torch.cat([cm.free_slots, valid[::page_size]])
    cm.check_integrity()


def _counts_dict(pool: DeepSeekV4KVCache) -> dict[str, int]:
    counts = pool.allocation_counts
    return {
        "full_slots": counts.full_slots,
        "c4_slots": counts.c4_slots,
        "c128_slots": counts.c128_slots,
        "c4_indexer_slots": counts.c4_indexer_slots,
        "c4_state_slots": counts.c4_state_slots,
        "c128_state_slots": counts.c128_state_slots,
        "c4_indexer_state_slots": counts.c4_indexer_state_slots,
    }


def _phase1_boundary_hit(prompt: torch.Tensor, page_size: int) -> dict[str, Any]:
    pool = _make_pool(num_pages=4, page_size=page_size, component_ownership=False)
    cm = _make_cache_manager(pool, num_pages=4, page_size=page_size)
    _cache_finished_prompt(cm, 0, prompt)
    hit_256 = cm.match_req(PendingReq(1, prompt[:256], SamplingParams(max_tokens=1)))
    hit_257 = cm.match_req(PendingReq(2, prompt[:257], SamplingParams(max_tokens=1)))
    out = {
        "prompt_len_256_hit": hit_256.cuda_handle.cached_len,
        "prompt_len_257_hit": hit_257.cuda_handle.cached_len,
        "counts_after_insert": _counts_dict(pool),
    }
    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()
    return out


def _route_b_state_ownership(prompt: torch.Tensor, page_size: int) -> dict[str, Any]:
    pool = _make_pool(num_pages=4, page_size=page_size, component_ownership=True)
    cm = _make_cache_manager(pool, num_pages=4, page_size=page_size)
    _cache_finished_prompt(cm, 0, prompt)

    hit_256 = cm.match_req(PendingReq(1, prompt[:256], SamplingParams(max_tokens=1)))
    hit_257 = cm.match_req(PendingReq(2, prompt[:257], SamplingParams(max_tokens=1)))
    hit_257_handles = hit_257.cuda_handle.get_dsv4_component_pages()
    assert hit_257_handles is not None

    retained_c4_state_pages = hit_257_handles.c4_state_pages.clone()
    retained_c128_state_pages = hit_257_handles.c128_state_pages.clone()
    retained_indexer_state_pages = hit_257_handles.c4_indexer_state_pages.clone()

    cm.free_slots = torch.sort(cm.free_slots).values
    reuse = _allocate_req_with_ids(cm, 3, torch.arange(64, dtype=torch.int32) + 10_000)
    reused_handles = pool.make_component_page_handles(
        cm.page_table[reuse.table_idx, :page_size],
        page_size,
    )
    assert reused_handles is not None
    _finish_req(cm, reuse)
    cm.check_integrity()

    out = {
        "prompt_len_256_hit": hit_256.cuda_handle.cached_len,
        "prompt_len_257_hit": hit_257.cuda_handle.cached_len,
        "counts_after_insert": _counts_dict(pool),
        "metrics_after_insert": cm.prefix_metrics_snapshot(),
        "retained_state_pages": {
            "c4": retained_c4_state_pages.tolist(),
            "c128": retained_c128_state_pages.tolist(),
            "c4_indexer": retained_indexer_state_pages.tolist(),
        },
        "reused_state_pages": {
            "c4": reused_handles.c4_state_pages.tolist(),
            "c128": reused_handles.c128_state_pages.tolist(),
            "c4_indexer": reused_handles.c4_indexer_state_pages.tolist(),
        },
        "state_pages_reused_stale": {
            "c4": bool(
                reused_handles.c4_state_pages[0].item()
                in retained_c4_state_pages.tolist()
            ),
            "c128": bool(
                reused_handles.c128_state_pages[0].item()
                in retained_c128_state_pages.tolist()
            ),
            "c4_indexer": bool(
                reused_handles.c4_indexer_state_pages[0].item()
                in retained_indexer_state_pages.tolist()
            ),
        },
    }
    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()
    out["counts_after_eviction"] = _counts_dict(pool)
    return out


def _state_formula_probe(page_size: int) -> dict[str, Any]:
    full_locs = torch.tensor([-1, 0, 7, 8, 127, 128, 255], dtype=torch.int32)
    phase1 = _make_pool(num_pages=4, page_size=page_size, component_ownership=False)
    route_b = _make_pool(num_pages=4, page_size=page_size, component_ownership=True)
    route_b.on_pages_allocated(torch.tensor([0, page_size], dtype=torch.int32), page_size)
    return {
        "full_locs": full_locs.tolist(),
        "phase1_c4_state": phase1.state_locs_from_full_locs(full_locs, 4).tolist(),
        "phase1_c128_state": phase1.state_locs_from_full_locs(full_locs, 128).tolist(),
        "phase1_indexer_state": phase1.state_locs_from_full_locs(
            full_locs,
            4,
            component="indexer",
        ).tolist(),
        "route_b_c4_state": route_b.state_locs_from_full_locs(full_locs, 4).tolist(),
        "route_b_c128_state": route_b.state_locs_from_full_locs(full_locs, 128).tolist(),
        "route_b_indexer_state": route_b.state_locs_from_full_locs(
            full_locs,
            4,
            component="indexer",
        ).tolist(),
    }


def _write_summary(result: dict[str, Any]) -> None:
    rows = [
        ("phase1", result["phase1"]["prompt_len_256_hit"], result["phase1"]["prompt_len_257_hit"]),
        ("route_b", result["route_b"]["prompt_len_256_hit"], result["route_b"]["prompt_len_257_hit"]),
    ]
    csv_lines = ["mode,prompt_len_256_hit,prompt_len_257_hit"]
    md_lines = [
        "| mode | prompt len 256 hit | prompt len 257 hit |",
        "| --- | ---: | ---: |",
    ]
    for mode, hit256, hit257 in rows:
        csv_lines.append(f"{mode},{hit256},{hit257}")
        md_lines.append(f"| {mode} | {hit256} | {hit257} |")

    (SUMMARIES / "hit_boundary_table.csv").write_text("\n".join(csv_lines) + "\n")
    (SUMMARIES / "hit_boundary_table.md").write_text("\n".join(md_lines) + "\n")
    summary = {
        "all_passed": result["all_passed"],
        "decision": result["decision"],
        "route_b_counts_after_insert": result["route_b"]["counts_after_insert"],
        "state_pages_reused_stale": result["route_b"]["state_pages_reused_stale"],
    }
    (SUMMARIES / "state_ownership_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    page_size = 128
    prompt = torch.arange(260, dtype=torch.int32)
    result = {
        "page_size": page_size,
        "state_formula": _state_formula_probe(page_size),
        "phase1": _phase1_boundary_hit(prompt, page_size),
        "route_b": _route_b_state_ownership(prompt, page_size),
    }
    stale = result["route_b"]["state_pages_reused_stale"]
    result["all_passed"] = (
        result["phase1"]["prompt_len_256_hit"] == page_size
        and result["route_b"]["prompt_len_256_hit"] == 0
        and result["route_b"]["prompt_len_257_hit"] == 2 * page_size
        and not any(stale.values())
        and not any(result["route_b"]["counts_after_eviction"].values())
    )
    result["decision"] = (
        "proceed_to_TARGET_08.21.4_with_swa_tail_guard"
        if result["all_passed"]
        else "blocked"
    )
    (RAW / "state_ownership_probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_summary(result)
    print(json.dumps({"all_passed": result["all_passed"], "decision": result["decision"]}))


if __name__ == "__main__":
    main()

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


def _tiny_dsv4_config(page_size: int) -> ModelConfig:
    return ModelConfig(
        num_layers=3,
        num_qo_heads=4,
        num_kv_heads=1,
        head_dim=8,
        hidden_size=16,
        vocab_size=32,
        intermediate_size=0,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(8, 2, 4096, 10000.0, None),
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
        window_size=min(128, page_size),
        compress_ratios=[4, 128, 0],
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


def _make_cache_manager(
    *,
    page_size: int,
    component_ownership: bool,
    num_pages: int = 16,
) -> tuple[CacheManager, DeepSeekV4KVCache]:
    pool = create_kvcache_pool(
        _tiny_dsv4_config(page_size),
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cpu"),
        enable_dsv4_component_loc_ownership=component_ownership,
    )
    assert isinstance(pool, DeepSeekV4KVCache)
    page_table = torch.full((4, num_pages * page_size), -1, dtype=torch.int32)
    ctx = core.Context(page_size=page_size)
    ctx.page_table = page_table
    ctx.kv_cache = pool
    core._GLOBAL_CTX = None
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type="radix", kv_cache=pool), pool


def _cache_finished_prompt(cm: CacheManager, prompt: torch.Tensor) -> None:
    sampling_params = SamplingParams(max_tokens=1)
    pending = PendingReq(uid=0, input_ids=prompt, sampling_params=sampling_params)
    handle = cm.match_req(pending).cuda_handle
    cm.lock(handle)
    req = Req(
        input_ids=prompt,
        table_idx=0,
        cached_len=handle.cached_len,
        output_len=1,
        uid=0,
        sampling_params=sampling_params,
        cache_handle=handle,
    )
    if handle.cached_len > 0:
        cm.page_table[0, : handle.cached_len].copy_(handle.get_matched_indices())
    cm.allocate_paged([req])
    req.cached_len = req.device_len
    cm.cache_req(req, finished=True)
    cm.check_integrity()


def _measure_hit(prompt_len: int, *, page_size: int, component_ownership: bool) -> dict[str, Any]:
    cm, pool = _make_cache_manager(
        page_size=page_size,
        component_ownership=component_ownership,
    )
    prompt = torch.arange(prompt_len, dtype=torch.int32)
    _cache_finished_prompt(cm, prompt)
    result = cm.match_req(
        PendingReq(uid=1, input_ids=prompt, sampling_params=SamplingParams(max_tokens=1))
    )
    metrics = cm.prefix_metrics_snapshot()
    counts = pool.allocation_counts
    return {
        "hit_tokens": int(result.cuda_handle.cached_len),
        "retained_prefix_pages": int(metrics["retained_prefix_pages"]),
        "live_full_pages": int(counts.full_slots // page_size),
        "live_c4_slots": int(counts.c4_slots),
        "live_c128_slots": int(counts.c128_slots),
        "live_c4_indexer_slots": int(counts.c4_indexer_slots),
        "live_c4_state_slots": int(counts.c4_state_slots),
        "live_c128_state_slots": int(counts.c128_state_slots),
        "live_c4_indexer_state_slots": int(counts.c4_indexer_state_slots),
    }


def _measure_page_size(page_size: int) -> list[dict[str, Any]]:
    prompt_lens = [
        page_size - 1,
        page_size,
        page_size + 1,
        2 * page_size - 1,
        2 * page_size,
        2 * page_size + 1,
        3 * page_size - 1,
        3 * page_size,
        3 * page_size + 1,
    ]
    rows: list[dict[str, Any]] = []
    for prompt_len in prompt_lens:
        phase1 = _measure_hit(
            prompt_len,
            page_size=page_size,
            component_ownership=False,
        )
        route_b = _measure_hit(
            prompt_len,
            page_size=page_size,
            component_ownership=True,
        )
        rows.append(
            {
                "page_size": page_size,
                "prompt_len": prompt_len,
                "phase1_hit_tokens": phase1["hit_tokens"],
                "route_b_hit_tokens": route_b["hit_tokens"],
                "shortened_tokens": phase1["hit_tokens"] - route_b["hit_tokens"],
                "route_b_live_full_pages": route_b["live_full_pages"],
                "route_b_live_c4_slots": route_b["live_c4_slots"],
                "route_b_live_c128_slots": route_b["live_c128_slots"],
                "route_b_live_c4_indexer_slots": route_b["live_c4_indexer_slots"],
                "route_b_live_c4_state_slots": route_b["live_c4_state_slots"],
                "route_b_live_c128_state_slots": route_b["live_c128_state_slots"],
                "route_b_live_c4_indexer_state_slots": route_b[
                    "live_c4_indexer_state_slots"
                ],
            }
        )
    return rows


def _write_tables(rows: list[dict[str, Any]]) -> None:
    keys = [
        "page_size",
        "prompt_len",
        "phase1_hit_tokens",
        "route_b_hit_tokens",
        "shortened_tokens",
        "route_b_live_full_pages",
        "route_b_live_c4_slots",
        "route_b_live_c128_slots",
        "route_b_live_c4_indexer_slots",
        "route_b_live_c4_state_slots",
        "route_b_live_c128_state_slots",
        "route_b_live_c4_indexer_state_slots",
    ]
    csv_lines = [",".join(keys)]
    for row in rows:
        csv_lines.append(",".join(str(row[key]) for key in keys))
    (SUMMARIES / "swa_tail_guard_table.csv").write_text("\n".join(csv_lines) + "\n")

    md_lines = [
        "| page | prompt | phase1 hit | Route B hit | shortened | live full | C4 | C128 | indexer | C4 state | C128 state | indexer state |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {page_size} | {prompt_len} | {phase1_hit_tokens} | "
            "{route_b_hit_tokens} | {shortened_tokens} | "
            "{route_b_live_full_pages} | {route_b_live_c4_slots} | "
            "{route_b_live_c128_slots} | {route_b_live_c4_indexer_slots} | "
            "{route_b_live_c4_state_slots} | {route_b_live_c128_state_slots} | "
            "{route_b_live_c4_indexer_state_slots} |".format(**row)
        )
    (SUMMARIES / "swa_tail_guard_table.md").write_text("\n".join(md_lines) + "\n")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    rows = _measure_page_size(128) + _measure_page_size(256)
    shortened = [row for row in rows if row["shortened_tokens"] > 0]
    result = {
        "rows": rows,
        "shortened_case_count": len(shortened),
        "shortened_cases": shortened,
        "decision": (
            "swa_tail_guard_only_shortens_exact_page_multiple_boundaries"
            if shortened
            else "no_guard_shortening_observed"
        ),
    }
    (RAW / "swa_tail_guard_quantification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_tables(rows)
    print(json.dumps({"shortened_case_count": len(shortened), "decision": result["decision"]}))


if __name__ == "__main__":
    main()

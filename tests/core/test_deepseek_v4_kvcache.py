from __future__ import annotations

import pytest
import torch

import minisgl.core as core
from minisgl.core import Req, SamplingParams
from minisgl.kvcache import create_kvcache_pool, estimate_kvcache_bytes_per_page
from minisgl.kvcache.deepseek_v4_pool import DSV4_INDEXER_FP8_CACHE_ENV, DeepSeekV4KVCache
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.scheduler.cache import CacheManager
from minisgl.scheduler.utils import PendingReq


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


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


def _make_dsv4_pool(
    compress_ratios: list[int],
    *,
    num_pages: int = 8,
    page_size: int = 4,
) -> DeepSeekV4KVCache:
    pool = create_kvcache_pool(
        _tiny_dsv4_config(compress_ratios),
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert isinstance(pool, DeepSeekV4KVCache)
    return pool


def _make_cache_manager(
    pool: DeepSeekV4KVCache,
    num_pages: int,
    page_size: int,
    *,
    cache_type: str = "naive",
) -> CacheManager:
    page_table = torch.empty((4, num_pages * page_size), dtype=torch.int32)
    if cache_type == "radix":
        ctx = core.Context(page_size=page_size)
        ctx.page_table = page_table
        ctx.kv_cache = pool
        core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, type=cache_type, kv_cache=pool)


def _allocate_req(cm: CacheManager, uid: int, input_len: int) -> Req:
    input_ids = torch.arange(input_len, dtype=torch.int32) + uid * 100
    return _allocate_req_with_ids(cm, uid, input_ids)


def _allocate_req_with_ids(cm: CacheManager, uid: int, input_ids: torch.Tensor) -> Req:
    sampling_params = SamplingParams(max_tokens=1)
    pending = PendingReq(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
    handle = cm.match_req(pending).cuda_handle
    cm.lock(handle)
    table_idx = uid % 4
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
    pool.on_token_indices_freed(evicted, page_size)
    cm.free_slots = torch.cat([cm.free_slots, evicted[::page_size]])
    cm.check_integrity()


def test_deepseek_v4_pool_factory_defaults_to_bf16_and_maps_layers():
    pool = _make_dsv4_pool([0, 4, 128, 4], num_pages=8, page_size=4)

    assert pool.dtype is torch.bfloat16
    assert pool.policy.layout == "bf16_flat"
    assert [m.compress_ratio for m in pool.layer_mapping] == [0, 4, 128, 4]
    assert pool.layer_mapping[0].normal_layer_id == 0
    assert pool.layer_mapping[1].c4_layer_id == 0
    assert pool.layer_mapping[1].indexer_layer_id == 0
    assert pool.layer_mapping[2].c128_layer_id == 0
    assert pool.layer_mapping[3].c4_layer_id == 1

    assert pool.swa_cache(0).shape == (32, 8)
    assert pool.c4_cache(1).shape == (8, 8)
    assert pool.c128_cache(2).shape == (1, 8)
    assert pool.indexer_cache(1).shape == (8, 4)
    assert pool.attention_compress_state(1).last_dim == 32
    assert pool.indexer_compress_state(1).last_dim == 16
    assert pool.attention_compress_state(2).last_dim == 16

    max_swa_loc = torch.tensor([pool.num_tokens - 1], dtype=torch.int32)
    state_loc = pool.attention_compress_state(1).translate_from_swa_loc_to_state_loc(max_swa_loc)
    state_size = pool.attention_compress_state(1).kv_score_buffer.kv_score.shape[0]
    assert int(state_loc.item()) < state_size


def test_deepseek_v4_memory_estimator_accounts_for_ring_state_pools():
    cfg = _tiny_dsv4_config([4, 128, 0])

    assert (
        estimate_kvcache_bytes_per_page(
            cfg,
            page_size=1,
            dtype=torch.float16,
            tp_size=1,
        )
        == 4919
    )


def test_deepseek_v4_indexer_fp8_side_cache_is_opt_in(monkeypatch):
    monkeypatch.delenv(DSV4_INDEXER_FP8_CACHE_ENV, raising=False)
    default_pool = _make_dsv4_pool([4], num_pages=4, page_size=4)

    assert not default_pool.has_indexer_fp8_cache()
    assert default_pool.indexer_cache(0).dtype is torch.bfloat16

    monkeypatch.setenv(DSV4_INDEXER_FP8_CACHE_ENV, "1")
    fp8_pool = _make_dsv4_pool([4], num_pages=4, page_size=4)
    values, scales = fp8_pool.indexer_fp8_cache(0)

    assert fp8_pool.has_indexer_fp8_cache()
    assert values.shape == (4, 4)
    assert scales.shape == (4, 4)
    assert values.dtype is torch.uint8
    assert scales.dtype is torch.uint8
    assert fp8_pool.indexer_cache(0).dtype is torch.bfloat16


def test_deepseek_v4_pool_can_write_and_read_all_cache_components():
    pool = _make_dsv4_pool([0, 4, 128], num_pages=8, page_size=4)

    loc = torch.tensor([0, 3], dtype=torch.int32)
    swa_kv = torch.arange(16, dtype=torch.float32).view(2, 8).to(torch.bfloat16)
    pool.store_swa(0, swa_kv, loc)
    assert torch.equal(pool.swa_cache(0)[loc.long()], swa_kv)

    c4_loc = torch.tensor([0, 1], dtype=torch.int32)
    c4_kv = torch.full((2, 8), 2.0, dtype=torch.bfloat16)
    pool.store_compressed(1, c4_kv, c4_loc)
    assert torch.equal(pool.c4_cache(1)[c4_loc.long()], c4_kv)

    index_kv = torch.full((2, 4), 3.0, dtype=torch.bfloat16)
    pool.store_indexer(1, index_kv, c4_loc)
    assert torch.equal(pool.indexer_cache(1)[c4_loc.long()], index_kv)

    c128_loc = torch.tensor([0], dtype=torch.int32)
    c128_kv = torch.full((1, 8), 4.0, dtype=torch.bfloat16)
    pool.store_compressed(2, c128_kv, c128_loc)
    assert torch.equal(pool.c128_cache(2)[c128_loc.long()], c128_kv)


def test_deepseek_v4_cache_manager_allocates_and_frees_one_request_without_leak():
    page_size = 4
    num_pages = 4
    pool = _make_dsv4_pool([0, 4, 128], num_pages=num_pages, page_size=page_size)
    cm = _make_cache_manager(pool, num_pages, page_size)

    req = _allocate_req(cm, uid=0, input_len=6)
    counts = pool.allocation_counts
    assert counts.full_slots == 8
    assert counts.c4_slots == 2
    assert counts.c128_slots == 1
    assert counts.c4_indexer_slots == 2

    _finish_req(cm, req)

    cm.check_integrity()
    pool.assert_no_leak()


def test_deepseek_v4_cache_manager_handles_multiple_sequential_requests_without_leak():
    page_size = 4
    num_pages = 8
    pool = _make_dsv4_pool([4, 128, 0], num_pages=num_pages, page_size=page_size)
    cm = _make_cache_manager(pool, num_pages, page_size)

    for uid, input_len in enumerate([5, 9, 3]):
        req = _allocate_req(cm, uid=uid, input_len=input_len)
        assert pool.allocation_counts.full_slots > 0
        _finish_req(cm, req)
        cm.check_integrity()
        pool.assert_no_leak()


def test_deepseek_v4_compressed_location_mapping_uses_full_token_namespace():
    pool = _make_dsv4_pool([4, 128], num_pages=64, page_size=4)
    full_locs = torch.arange(0, 256, dtype=torch.int32)
    positions = torch.arange(0, 256, dtype=torch.int32)

    assert torch.equal(
        pool.compressed_locs_from_full_locs(full_locs[:8], 4, positions[:8]),
        torch.tensor([0, 1]),
    )
    assert torch.equal(
        pool.compressed_locs_from_full_locs(full_locs, 128, positions),
        torch.tensor([0, 1]),
    )


def test_deepseek_v4_radix_prefix_cache_tracks_full_partial_miss_and_components():
    page_size = 128
    num_pages = 8
    pool = _make_dsv4_pool([0, 4, 128], num_pages=num_pages, page_size=page_size)
    cm = _make_cache_manager(pool, num_pages, page_size, cache_type="radix")

    base = torch.arange(450, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, base)
    assert cm.prefix_metrics_snapshot()["retained_prefix_tokens"] == 384

    shared_two_pages = torch.cat(
        [base[: 2 * page_size], torch.arange(450 - 2 * page_size, dtype=torch.int32) + 10_000]
    )
    _cache_finished_prompt(cm, 1, shared_two_pages)

    repeat_base = _cache_finished_prompt(cm, 2, base)
    assert repeat_base.cache_handle.cached_len == 384

    miss_ids = torch.arange(260, dtype=torch.int32) + 20_000
    miss_handle = cm.match_req(
        PendingReq(3, miss_ids, SamplingParams(max_tokens=1))
    ).cuda_handle
    assert miss_handle.cached_len == 0

    snapshot = cm.prefix_metrics_snapshot()
    assert snapshot["hit_requests"] >= 2
    assert snapshot["partial_hit_requests"] >= 1
    assert snapshot["full_hit_requests"] >= 1
    assert snapshot["miss_requests"] >= 2
    assert snapshot["saved_prefill_tokens"] >= 640
    assert snapshot["retained_prefix_pages"] == 4
    assert snapshot["dsv4_retention"]["full_slots"] == 512
    assert snapshot["dsv4_retention"]["c4_slots"] == 128
    assert snapshot["dsv4_retention"]["c128_slots"] == 4
    assert snapshot["dsv4_retention"]["c4_indexer_slots"] == 128
    assert snapshot["dsv4_retention"]["page_size_c128_aligned"]
    assert pool.allocation_counts.full_slots == 512


def test_deepseek_v4_radix_prefix_swa_window_128_boundary_is_page_safe():
    page_size = 128
    num_pages = 4
    pool = _make_dsv4_pool([0, 4, 128], num_pages=num_pages, page_size=page_size)
    cm = _make_cache_manager(pool, num_pages, page_size, cache_type="radix")

    prompt_at_boundary = torch.arange(129, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, prompt_at_boundary)

    below = torch.arange(127, dtype=torch.int32)
    below_handle = cm.match_req(
        PendingReq(1, below, SamplingParams(max_tokens=1))
    ).cuda_handle
    assert below_handle.cached_len == 0

    at_handle = cm.match_req(
        PendingReq(2, prompt_at_boundary, SamplingParams(max_tokens=1))
    ).cuda_handle
    assert at_handle.cached_len == 128
    cm.lock(at_handle)
    cm.unlock(at_handle)

    above = torch.arange(130, dtype=torch.int32)
    above_handle = cm.match_req(
        PendingReq(3, above, SamplingParams(max_tokens=1))
    ).cuda_handle
    assert above_handle.cached_len == 128
    cm.lock(above_handle)
    cm.unlock(above_handle)

    snapshot = cm.prefix_metrics_snapshot()
    assert snapshot["dsv4_retention"]["c4_slots"] == 32
    assert snapshot["dsv4_retention"]["c128_slots"] == 1
    assert snapshot["dsv4_retention"]["c4_state_slots"] == pool.C4_STATE_RING_SIZE
    assert snapshot["dsv4_retention"]["c128_state_slots"] == pool.C128_STATE_RING_SIZE


def test_deepseek_v4_radix_prefix_repeated_hit_evict_cycle_has_no_leak():
    page_size = 128
    num_pages = 4
    pool = _make_dsv4_pool([4, 128, 0], num_pages=num_pages, page_size=page_size)
    cm = _make_cache_manager(pool, num_pages, page_size, cache_type="radix")

    prompt = torch.arange(260, dtype=torch.int32)
    for cycle in range(3):
        _cache_finished_prompt(cm, cycle * 2, prompt + cycle * 10_000)
        hit_req = _cache_finished_prompt(cm, cycle * 2 + 1, prompt + cycle * 10_000)
        assert hit_req.cache_handle.cached_len == 256
        assert pool.allocation_counts.full_slots == 256
        _evict_all_prefix(cm, pool, page_size)
        assert cm.prefix_cache.size_info.total_size == 0
        pool.assert_no_leak()

    snapshot = cm.prefix_metrics_snapshot()
    assert snapshot["evictions"] == 3
    assert snapshot["evicted_tokens"] == 3 * 2 * page_size

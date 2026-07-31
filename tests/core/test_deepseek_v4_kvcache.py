from __future__ import annotations

import minisgl.core as core
import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.kvcache import (
    create_kvcache_pool,
    estimate_c4_sequence_state_bytes,
    estimate_kvcache_bytes_per_page,
)
from minisgl.kvcache.deepseek_v4_pool import (
    DeepSeekV4KVCache,
    DSV4SWAPageHandles,
    _clear_allocated_kv_modes,
)
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
    max_running_req: int | None = None,
    dsv4_dummy_token_start: int | None = None,
) -> DeepSeekV4KVCache:
    pool = create_kvcache_pool(
        _tiny_dsv4_config(compress_ratios),
        num_pages=num_pages,
        page_size=page_size,
        device=torch.device("cpu"),
        max_running_req=max_running_req,
        dsv4_swa_num_pages=num_pages,
        dsv4_dummy_token_start=dsv4_dummy_token_start,
    )
    assert isinstance(pool, DeepSeekV4KVCache)
    return pool


def _make_cache_manager(
    pool: DeepSeekV4KVCache,
    num_pages: int,
    page_size: int,
) -> CacheManager:
    page_table = torch.empty((4, num_pages * page_size), dtype=torch.int32)
    ctx = core.Context(page_size=page_size)
    ctx.page_table = page_table
    ctx.kv_cache = pool
    core.set_global_ctx(ctx)
    return CacheManager(num_pages, page_size, page_table, kv_cache=pool)


def _allocate_req(cm: CacheManager, uid: int, input_len: int) -> Req:
    input_ids = torch.arange(input_len, dtype=torch.int32) + uid * 100
    return _allocate_req_with_ids(cm, uid, input_ids)


def _allocate_req_with_ids(
    cm: CacheManager,
    uid: int,
    input_ids: torch.Tensor,
    *,
    output_len: int = 1,
) -> Req:
    sampling_params = SamplingParams(max_tokens=output_len)
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
        output_len=output_len,
        uid=uid,
        sampling_params=sampling_params,
        cache_handle=handle,
    )
    cm.allocate_paged([req])
    return req


def _finish_req(cm: CacheManager, req: Req) -> None:
    req.cached_len = req.device_len
    cm.cache_req(req, finished=True)


def _complete_one_generated(req: Req, token: int) -> None:
    req.cached_len = req.device_len
    req.device_len += 1
    req.append_host(torch.tensor([token], dtype=req.input_ids.dtype))


def _cache_serving_prompt(cm: CacheManager, uid: int, input_ids: torch.Tensor) -> Req:
    req = _allocate_req_with_ids(cm, uid, input_ids, output_len=2)
    _complete_one_generated(req, 30_000 + uid * 2)
    cm.cache_req(req, finished=False)
    cm.allocate_paged([req])
    _complete_one_generated(req, 30_001 + uid * 2)
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    return req


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
            free_components=False,
            free_swa=False,
        )
        cm.free_slots = torch.cat([cm.free_slots, valid[::page_size]])
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
    assert pool.indexer_fp8_paged_cache(1).shape == (8, 8)
    assert not hasattr(pool, "_c4_indexer_buffer")
    assert not hasattr(pool, "indexer_cache")
    assert not hasattr(pool, "store_indexer")
    assert pool.attention_compress_state(1).last_dim == 32
    assert pool.indexer_compress_state(1).last_dim == 16
    assert pool.attention_compress_state(1).kv_score_buffer.kv_score.shape == (
        2 * pool.C4_STATE_RING_SIZE,
        32,
    )
    assert pool.attention_c4_checkpoint(1).kv_score_buffer.kv_score.shape == (
        8 * pool.C4_CHECKPOINT_ROWS,
        16,
    )
    assert pool.indexer_c4_checkpoint(1).kv_score_buffer.kv_score.shape == (
        8 * pool.C4_CHECKPOINT_ROWS,
        8,
    )
    assert pool.attention_compress_state(2).last_dim == 16
    assert pool.attention_compress_state(2).kv_score_buffer.kv_score.dtype is torch.float32


def test_deepseek_v4_c4_requires_one_component_page_per_full_page():
    with pytest.raises(
        ValueError,
        match="exactly one C4 component page per full page",
    ):
        _make_dsv4_pool([4], num_pages=8, page_size=1)


def test_deepseek_v4_memory_estimator_excludes_fixed_sequence_state():
    cfg = _tiny_dsv4_config([4, 128, 0])

    assert (
        estimate_kvcache_bytes_per_page(
            cfg,
            page_size=1,
            tp_size=1,
        )
        == 439
    )
    assert estimate_c4_sequence_state_bytes(cfg, 2) == 4608


def test_c128_sequence_slot_owner_no_clear_reorder_dummy_abort_completion_and_reuse():
    pool = _make_dsv4_pool(
        [128, 128],
        num_pages=4,
        page_size=256,
        max_running_req=2,
    )
    assert pool.c128_sequence_slots == 3
    assert pool.c128_sequence_state_bytes == 2 * 3 * 128 * 2 * 8 * 4

    state0 = pool.attention_compress_state(0).kv_score_buffer.kv_score
    state1 = pool.attention_compress_state(1).kv_score_buffer.kv_score
    state0[: 2 * 128].fill_(101)
    state1[: 2 * 128].fill_(-101)
    pool.acquire_c128_sequence_slot(0, 10)
    pool.acquire_c128_sequence_slot(1, 11)
    assert torch.all(state0[: 2 * 128] == 101)
    assert torch.all(state1[: 2 * 128] == -101)
    state0[0:7].fill_(10)
    state0[128:133].fill_(11)
    state1[0:7].fill_(20)
    state1[128:133].fill_(21)

    # Reordering batch rows changes only the order of stable table indices.
    reordered = torch.tensor([1, 0], dtype=torch.long)
    assert torch.all(state0[reordered[0] * 128 : reordered[0] * 128 + 5] == 11)
    assert torch.all(state0[reordered[1] * 128 : reordered[1] * 128 + 7] == 10)

    # Graph padding owns the extra slot and cannot alias either live request.
    dummy_start = 2 * 128
    live_before = state0[: 2 * 128].clone()
    state0[dummy_start : dummy_start + 4].fill_(99)
    assert torch.equal(state0[: 2 * 128], live_before)

    # Page allocation/eviction cannot clear or own sequence state.
    pool.on_pages_allocated(torch.tensor([0], dtype=torch.int32), page_size=256)
    assert torch.all(state0[:7] == 10)
    pool.on_token_indices_freed(torch.arange(256, dtype=torch.int32), page_size=256)
    assert torch.all(state0[:7] == 10)

    # Abort and completion use the same owner-checked retirement transition.
    slot0_before_release = state0[:128].clone()
    pool.release_c128_sequence_slot(0, 10)
    assert pool.c128_sequence_owner(0) is None
    assert torch.equal(state0[:128], slot0_before_release)
    assert torch.all(state0[128:133] == 11)
    pool.acquire_c128_sequence_slot(0, 12)
    assert pool.c128_sequence_owner(0) == 12
    assert torch.equal(state0[:128], slot0_before_release)
    pool.release_c128_sequence_slot(1, 11)
    pool.release_c128_sequence_slot(0, 12)

    # Repeated abort/completion-shaped reuse drops only logical ownership.
    previous_payload: dict[int, torch.Tensor] = {}
    for generation in range(13, 21):
        slot = generation % 2
        pool.acquire_c128_sequence_slot(slot, generation)
        start = slot * 128
        if slot in previous_payload:
            assert torch.equal(state0[start : start + 128], previous_payload[slot])
        state0[start : start + 9].fill_(generation)
        payload = state0[start : start + 128].clone()
        pool.release_c128_sequence_slot(slot, generation)
        assert pool.c128_sequence_owner(slot) is None
        assert torch.equal(state0[start : start + 128], payload)
        previous_payload[slot] = payload
    assert pool.c128_sequence_owner_count == 0

    with pytest.raises(RuntimeError, match="owner mismatch"):
        pool.release_c128_sequence_slot(0, 12)
    with pytest.raises(ValueError, match="out of range"):
        pool.acquire_c128_sequence_slot(2, 13)


def test_c4_sequence_slot_owner_no_clear_attention_indexer_dummy_and_reuse():
    pool = _make_dsv4_pool(
        [4, 4],
        num_pages=4,
        page_size=256,
        max_running_req=2,
    )
    assert pool.c4_sequence_slots == 3
    assert pool.c4_sequence_state_bytes == (
        2
        * 3
        * pool.C4_STATE_RING_SIZE
        * 4
        * (pool._head_dim + pool._index_head_dim)
        * torch.float32.itemsize
    )

    attention = pool.attention_compress_state(0).kv_score_buffer.kv_score
    indexer = pool.indexer_compress_state(0).kv_score_buffer.kv_score
    checkpoint = pool.attention_c4_checkpoint(0).kv_score_buffer.kv_score
    attention.fill_(101)
    indexer.fill_(-101)
    checkpoint.fill_(303)

    pool.acquire_sequence_slot(0, 20)
    pool.acquire_sequence_slot(1, 21)
    assert pool.c4_sequence_owner(0) == 20
    assert pool.c4_sequence_owner(1) == 21
    assert torch.all(attention == 101)
    assert torch.all(indexer == -101)

    # Stable request slots do not follow transient batch-row reorder.
    attention[: pool.C4_STATE_RING_SIZE].fill_(20)
    attention[pool.C4_STATE_RING_SIZE : 2 * pool.C4_STATE_RING_SIZE].fill_(21)
    reordered = torch.tensor([1, 0])
    assert torch.all(
        attention[
            reordered[0] * pool.C4_STATE_RING_SIZE : (reordered[0] + 1) * pool.C4_STATE_RING_SIZE
        ]
        == 21
    )

    # The final physical slot is graph-only and cannot become a live owner.
    dummy_start = 2 * pool.C4_STATE_RING_SIZE
    live_before = attention[:dummy_start].clone()
    attention[dummy_start:].fill_(404)
    assert torch.equal(attention[:dummy_start], live_before)
    with pytest.raises(ValueError, match="out of range"):
        pool.acquire_sequence_slot(2, 22)

    pool.release_sequence_slot(0, 20)
    stale_attention = attention[: pool.C4_STATE_RING_SIZE].clone()
    stale_indexer = indexer[: pool.C4_STATE_RING_SIZE].clone()
    pool.acquire_sequence_slot(0, 22)
    assert torch.equal(
        attention[: pool.C4_STATE_RING_SIZE],
        stale_attention,
    )
    assert torch.equal(
        indexer[: pool.C4_STATE_RING_SIZE],
        stale_indexer,
    )
    assert torch.all(checkpoint == 303)
    pool.release_sequence_slot(0, 22)
    pool.release_sequence_slot(1, 21)
    assert pool.c4_sequence_owner_count == 0

    with pytest.raises(RuntimeError, match="owner mismatch"):
        pool.release_sequence_slot(0, 22)


def test_c128_sequence_pool_keeps_graph_dummy_outside_live_owner_namespace():
    pool = _make_dsv4_pool(
        [128],
        num_pages=4,
        page_size=256,
        max_running_req=2,
    )
    state_pool = pool.attention_compress_state(0)
    state = state_pool.kv_score_buffer.kv_score
    dummy_last_row = pool.c128_sequence_slots * 128 - 1
    state[dummy_last_row].fill_(77)
    state[0].fill_(11)
    pool.acquire_c128_sequence_slot(0, 99)
    pool.release_c128_sequence_slot(0, 99)

    assert torch.all(state[0] == 11)
    assert torch.all(state[dummy_last_row] == 77)
    with pytest.raises(ValueError, match="out of range"):
        pool.acquire_c128_sequence_slot(2, 99)


def test_dsv4_radix_all_reusable_boundaries_are_128_aligned_and_do_not_own_c128_active_state():
    page_size = 256
    num_pages = 12
    pool = _make_dsv4_pool(
        [4, 128],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=2,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)
    pool.acquire_c128_sequence_slot(0, 20)
    c128_state = pool.attention_compress_state(1).kv_score_buffer.kv_score
    c128_state[:17].fill_(7)

    base = torch.arange(769, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, base)
    full_hit = cm.match_req(PendingReq(1, base, SamplingParams(max_tokens=1))).cuda_handle
    assert full_hit.cached_len == 3 * page_size
    assert full_hit.cached_len % 128 == 0

    fork = torch.cat([base[:600], torch.arange(169, dtype=torch.int32) + 100_000])
    shared_hit = cm.match_req(PendingReq(2, fork, SamplingParams(max_tokens=1))).cuda_handle
    assert shared_hit.cached_len == 2 * page_size
    assert shared_hit.cached_len % 128 == 0
    _cache_finished_prompt(cm, 1, fork)
    fork_hit = cm.match_req(PendingReq(2, fork, SamplingParams(max_tokens=1))).cuda_handle
    assert fork_hit.cached_len == 3 * page_size
    assert fork_hit.cached_len % 128 == 0

    stack = [cm.prefix_cache.root_node]
    while stack:
        node = stack.pop()
        if not node.is_root():
            assert node.length % page_size == 0
            assert node.length % 128 == 0
        stack.extend(node.children.values())

    _evict_all_prefix(cm, pool, page_size)
    assert torch.all(c128_state[:17] == 7)
    pool.release_c128_sequence_slot(0, 20)


def test_deepseek_v4_release_has_packed_indexer_cache_only():
    fp8_pool = _make_dsv4_pool([4], num_pages=4, page_size=4)
    values, scales = fp8_pool.indexer_fp8_cache(0)

    assert fp8_pool.has_indexer_fp8_cache()
    assert values.shape == (4, 4)
    assert scales.shape == (4, 4)
    assert values.dtype is torch.uint8
    assert scales.dtype is torch.uint8
    assert fp8_pool.indexer_fp8_paged_cache(0).dtype is torch.uint8
    assert not hasattr(fp8_pool, "_c4_indexer_buffer")
    assert not hasattr(fp8_pool, "indexer_cache")
    assert not hasattr(fp8_pool, "store_indexer")
    retained = fp8_pool.estimate_prefix_retention(4)
    assert retained["c4_indexer_packed_bytes"] == 8
    assert "c4_indexer_bytes" not in retained
    assert "c4_indexer_fp8_bytes" not in retained


def test_deepseek_v4_allocated_page_clear_uses_component_release_policy():
    assert _clear_allocated_kv_modes() == {"component"}


def test_deepseek_v4_allocated_page_component_clear_resets_only_new_component_slots():
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=4,
        page_size=256,
    )

    pool.swa_cache(2).fill_(3)
    pool.c4_cache(0).fill_(4)
    pool.c128_cache(1).fill_(5)
    fp8_paged = pool.indexer_fp8_paged_cache(0)
    fp8_values, fp8_scales = pool.indexer_fp8_cache(0)
    fp8_paged.fill_(10)
    fp8_values.fill_(11)
    fp8_scales.fill_(12)
    c4_state = pool.attention_compress_state(0).kv_score_buffer.kv_score
    c128_state = pool.attention_compress_state(1).kv_score_buffer.kv_score
    indexer_state = pool.indexer_compress_state(0).kv_score_buffer.kv_score
    c4_state.fill_(7)
    c128_state.fill_(8)
    indexer_state.fill_(9)

    pool.on_pages_allocated(torch.tensor([0], dtype=torch.int32), page_size=256)

    assert torch.all(pool.swa_cache(2) == 3)
    assert torch.all(pool.c4_cache(0)[:64] == 0)
    assert torch.all(pool.c4_cache(0)[64:] == 4)
    assert torch.all(pool.c128_cache(1)[:2] == 0)
    assert torch.all(pool.c128_cache(1)[2:] == 5)
    assert torch.all(fp8_paged[0] == 0)
    assert torch.all(fp8_paged[1:] == 10)
    assert torch.all(fp8_values[:64] == 0)
    assert torch.all(fp8_values[64:] == 11)
    assert torch.all(fp8_scales[:64] == 0)
    assert torch.all(fp8_scales[64:] == 12)
    assert torch.all(c4_state == 7)
    assert torch.all(indexer_state == 9)
    assert torch.all(c128_state == 8)


def test_deepseek_v4_pool_can_write_and_read_attention_cache_components():
    pool = _make_dsv4_pool([0, 4, 128], num_pages=8, page_size=4)
    pool.on_pages_allocated(torch.tensor([0], dtype=torch.int32), page_size=4)

    loc = torch.tensor([0, 3], dtype=torch.int32)
    swa_kv = torch.arange(16, dtype=torch.float32).view(2, 8).to(torch.bfloat16)
    pool.store_swa(0, swa_kv, loc)
    swa_loc = pool.translate_full_locs_to_swa_locs(loc)
    assert torch.equal(pool.swa_cache(0)[swa_loc.long()], swa_kv)

    c4_loc = torch.tensor([0, 1], dtype=torch.int32)
    c4_kv = torch.full((2, 8), 2.0, dtype=torch.bfloat16)
    pool.store_compressed(1, c4_kv, c4_loc)
    assert torch.equal(pool.c4_cache(1)[c4_loc.long()], c4_kv)

    c128_loc = torch.tensor([0], dtype=torch.int32)
    c128_kv = torch.full((1, 8), 4.0, dtype=torch.bfloat16)
    pool.store_compressed(2, c128_kv, c128_loc)
    assert torch.equal(pool.c128_cache(2)[c128_loc.long()], c128_kv)


def test_deepseek_v4_compressed_location_mapping_uses_full_token_namespace():
    pool = _make_dsv4_pool([4, 128], num_pages=3, page_size=128)
    pool.on_pages_allocated(torch.tensor([0, 128], dtype=torch.int32), page_size=128)
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
    cm = _make_cache_manager(pool, num_pages, page_size)

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
    miss_handle = cm.match_req(PendingReq(3, miss_ids, SamplingParams(max_tokens=1))).cuda_handle
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
    assert pool.allocation_counts.full_slots == 256


def test_deepseek_v4_radix_prefix_swa_window_128_boundary_is_page_safe():
    page_size = 128
    num_pages = 4
    pool = _make_dsv4_pool([0, 4, 128], num_pages=num_pages, page_size=page_size)
    cm = _make_cache_manager(pool, num_pages, page_size)

    prompt_at_boundary = torch.arange(129, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, prompt_at_boundary)

    below = torch.arange(127, dtype=torch.int32)
    below_handle = cm.match_req(PendingReq(1, below, SamplingParams(max_tokens=1))).cuda_handle
    assert below_handle.cached_len == 0

    at_handle = cm.match_req(
        PendingReq(2, prompt_at_boundary, SamplingParams(max_tokens=1))
    ).cuda_handle
    assert at_handle.cached_len == 128
    cm.lock(at_handle)
    cm.unlock(at_handle)

    above = torch.arange(130, dtype=torch.int32)
    above_handle = cm.match_req(PendingReq(3, above, SamplingParams(max_tokens=1))).cuda_handle
    assert above_handle.cached_len == 128
    cm.lock(above_handle)
    cm.unlock(above_handle)

    snapshot = cm.prefix_metrics_snapshot()
    assert snapshot["dsv4_retention"]["c4_slots"] == 32
    assert snapshot["dsv4_retention"]["c128_slots"] == 1
    assert snapshot["dsv4_retention"]["c4_checkpoint_slots"] == pool.C4_CHECKPOINT_ROWS
    assert snapshot["dsv4_retention"]["c4_sequence_state_bytes"] == (pool.c4_sequence_state_bytes)
    assert snapshot["dsv4_retention"]["c128_sequence_state_bytes"] == (
        pool.c128_sequence_state_bytes
    )


def test_deepseek_v4_radix_prefix_repeated_hit_evict_cycle_has_no_leak():
    page_size = 128
    num_pages = 4
    pool = _make_dsv4_pool([4, 128, 0], num_pages=num_pages, page_size=page_size)
    cm = _make_cache_manager(pool, num_pages, page_size)

    prompt = torch.arange(260, dtype=torch.int32)
    for cycle in range(3):
        _cache_finished_prompt(cm, cycle * 2, prompt + cycle * 10_000)
        hit_req = _cache_finished_prompt(cm, cycle * 2 + 1, prompt + cycle * 10_000)
        assert hit_req.cache_handle.cached_len == 256
        assert pool.allocation_counts.full_slots == page_size
        _evict_all_prefix(cm, pool, page_size)
        assert cm.prefix_cache.size_info.total_size == 0
        pool.assert_no_leak()

    snapshot = cm.prefix_metrics_snapshot()
    assert snapshot["evictions"] == 3
    assert snapshot["evicted_tokens"] == 3 * 2 * page_size


def test_dsv4_component_loc_ownership_releases_full_head_without_stale_component_reuse():
    page_size = 128
    num_pages = 4
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    prompt = torch.arange(260, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, prompt)

    counts = pool.allocation_counts
    assert counts.full_slots == page_size
    assert counts.c4_slots == 2 * (page_size // 4)
    assert counts.c128_slots == 2 * (page_size // 128)
    assert counts.c4_indexer_slots == 2 * (page_size // 4)
    assert counts.c4_checkpoint_slots == 2 * pool.C4_CHECKPOINT_ROWS
    assert counts.c4_indexer_checkpoint_slots == 2 * pool.C4_CHECKPOINT_ROWS
    assert cm.prefix_cache.size_info.total_size == 2 * page_size
    assert cm.prefix_metrics_snapshot()["dsv4_component_ownership"]["live_full_pages"] == 1

    handle = cm.match_req(PendingReq(1, prompt, SamplingParams(max_tokens=1))).cuda_handle
    assert handle.cached_len == 2 * page_size
    matched = handle.get_matched_indices()
    assert torch.all(matched[:page_size] < 0)
    assert torch.all(matched[page_size:] >= 0)
    component_pages = handle.get_dsv4_component_pages()
    assert component_pages is not None
    retained_c4_pages = component_pages.c4_pages.clone()
    retained_c4_indexer_pages = component_pages.c4_indexer_pages.clone()
    assert retained_c4_pages.tolist() == [0, 1]
    assert retained_c4_indexer_pages.tolist() == [0, 1]
    assert component_pages.has_required_c4_recovery_pages
    assert not hasattr(component_pages, "c4_checkpoint_pages")
    left = component_pages.slice_tokens(0, page_size)
    right = component_pages.slice_tokens(page_size, 2 * page_size)
    joined = type(component_pages).concat([left, right])
    assert joined is not None
    assert torch.equal(joined.c4_pages, component_pages.c4_pages)
    assert torch.equal(joined.c4_indexer_pages, component_pages.c4_indexer_pages)

    cm.free_slots = torch.sort(cm.free_slots).values
    reuse = _allocate_req_with_ids(cm, 2, torch.arange(64, dtype=torch.int32) + 10_000)
    reused_full_page = cm.page_table[reuse.table_idx, 0].item() // page_size
    assert reused_full_page == 0
    reused_components = pool.make_component_page_handles(
        cm.page_table[reuse.table_idx, :page_size],
        page_size,
    )
    assert reused_components is not None
    assert reused_components.c4_pages[0].item() not in retained_c4_pages.tolist()
    assert (
        reused_components.c4_indexer_pages[0].item()
        not in retained_c4_indexer_pages.tolist()
    )
    _finish_req(cm, reuse)
    cm.check_integrity()

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_component_loc_ownership_implies_checkpoint_and_keeps_swa_safe_boundary():
    page_size = 128
    num_pages = 4
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    prompt = torch.arange(260, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, prompt)

    # A 256-token request can reuse its first page because component checkpoints
    # and the independent SWA handle remain live under the structural contract.
    guarded = cm.match_req(PendingReq(1, prompt[:256], SamplingParams(max_tokens=1))).cuda_handle
    assert guarded.cached_len == page_size

    # A 257-token request matches two full pages. The final page is the retained
    # live SWA tail, while checkpoint pages are owned independently along the path.
    hit = cm.match_req(PendingReq(2, prompt[:257], SamplingParams(max_tokens=1))).cuda_handle
    assert hit.cached_len == 2 * page_size
    handles = hit.get_dsv4_component_pages()
    assert handles is not None
    assert handles.has_required_c4_recovery_pages
    assert handles.c4_pages.tolist() == [0, 1]
    assert not hasattr(handles, "c128_state_pages")
    assert handles.c4_indexer_pages.tolist() == [0, 1]
    assert not hasattr(handles, "c4_indexer_checkpoint_pages")

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_swa_independent_lifecycle_tombstones_swa_without_component_invalidation():
    page_size = 256
    num_pages = 8
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=3,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    prompt = torch.arange(513, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, prompt)

    snapshot = cm.prefix_metrics_snapshot()
    counts = pool.allocation_counts
    assert counts.full_slots == page_size
    assert counts.swa_pages == 2
    assert counts.c4_slots == 2 * (page_size // 4)
    assert counts.c128_slots == 2 * (page_size // 128)
    assert snapshot["dsv4_swa_lifecycle"]["current_swa_tail_pages"] == 2
    assert snapshot["dsv4_swa_lifecycle"]["retained_prefix_swa_pages"] == 2
    assert snapshot["dsv4_swa_lifecycle"]["swa_pages_tombstoned_total"] == 0
    assert snapshot["dsv4_component_ownership"]["live_full_pages"] == 1
    assert snapshot["dsv4_component_ownership"]["available_component_pages"] == num_pages - 2

    handle = cm.match_req(PendingReq(1, prompt, SamplingParams(max_tokens=1))).cuda_handle
    assert handle.cached_len == 2 * page_size
    matched = handle.get_matched_indices()
    assert torch.all(matched[:page_size] < 0)
    assert torch.all(matched[page_size:] >= 0)
    component_pages = handle.get_dsv4_component_pages()
    swa_pages = handle.get_dsv4_swa_pages()
    assert component_pages is not None and component_pages.has_required_c4_recovery_pages
    assert swa_pages is not None and swa_pages.swa_pages is not None
    assert swa_pages.swa_pages.tolist()[0] >= 0
    assert swa_pages.swa_pages.tolist()[1] >= 0

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()
    assert pool.runtime_swa_counters()["current_swa_tail_pages"] == 0


def test_dsv4_swa_capacity_pressure_can_release_prefix_swa_only():
    page_size = 256
    num_pages = 8
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=3,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    prompts = [torch.arange(257, dtype=torch.int32) + i * 1000 for i in range(3)]
    for uid, prompt in enumerate(prompts):
        _cache_finished_prompt(cm, uid, prompt)

    before = pool.allocation_counts
    assert before.full_slots == 3 * page_size
    assert before.swa_pages == 3
    assert before.c4_slots == 3 * (page_size // 4)
    assert before.c128_slots == 3 * (page_size // 128)

    released = cm.prefix_cache.release_dsv4_evictable_swa_pages(2)
    assert released == 2
    after = pool.allocation_counts
    assert after.full_slots == before.full_slots
    assert after.c4_slots == before.c4_slots
    assert after.c128_slots == before.c128_slots
    assert after.swa_pages == 1
    assert pool.runtime_swa_counters()["swa_pages_tombstoned_total"] == 2

    snapshot = cm.prefix_metrics_snapshot()
    assert snapshot["dsv4_swa_lifecycle"]["evictable_prefix_swa_pages"] == 1
    guarded = cm.match_req(PendingReq(99, prompts[0], SamplingParams(max_tokens=1))).cuda_handle
    assert guarded.cached_len == 0

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_active_swa_release_preserves_component_mappings():
    page_size = 256
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=8,
        page_size=page_size,
        max_running_req=3,
    )
    page_starts = torch.tensor([0, page_size, page_size * 2], dtype=torch.int32)
    pool.on_pages_allocated(page_starts, page_size)

    before = pool.allocation_counts
    assert before.full_slots == 3 * page_size
    assert before.swa_pages == 3
    assert before.c4_slots == 3 * (page_size // 4)
    assert before.c128_slots == 3 * (page_size // 128)

    pool.release_swa_for_full_indices(
        torch.arange(0, page_size * 2, dtype=torch.int32),
        page_size,
        tombstone=True,
    )

    after = pool.allocation_counts
    assert after.full_slots == before.full_slots
    assert after.c4_slots == before.c4_slots
    assert after.c128_slots == before.c128_slots
    assert after.c4_indexer_slots == before.c4_indexer_slots
    assert after.swa_pages == 1
    assert pool.runtime_swa_counters()["swa_pages_tombstoned_total"] == 2
    swa_pages = pool.swa_pages_from_full_page_starts(page_starts, page_size)
    assert swa_pages is not None
    assert swa_pages.tolist()[:2] == [-1, -1]
    assert swa_pages.tolist()[2] >= 0
    c4_pages, c128_pages, indexer_pages = pool.component_pages_from_full_page_starts(
        page_starts,
        page_size,
    )
    assert c4_pages is not None and torch.all(c4_pages >= 0)
    assert c128_pages is not None and torch.all(c128_pages >= 0)
    assert indexer_pages is not None and torch.all(indexer_pages >= 0)

    pool.on_token_indices_freed(
        torch.arange(0, page_size * 3, dtype=torch.int32),
        page_size,
    )
    pool.assert_no_leak()


def test_dsv4_active_swa_release_respects_cache_protected_len():
    page_size = 4
    num_pages = 256
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=8,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    allocated = cm._page_to_token(cm._allocate(8))
    cm.page_table[0, : allocated.numel()].copy_(allocated)
    input_ids = torch.arange(allocated.numel() + 1, dtype=torch.int32)

    prefix_tokens = 4 * page_size
    _, prefix_handle = cm.prefix_cache.insert_prefix(
        input_ids[:prefix_tokens],
        allocated[:prefix_tokens],
        dsv4_component_pages_builder=lambda start, end: pool.make_component_page_handles(
            allocated[start:end],
            page_size,
        ),
        dsv4_swa_pages_builder=lambda start, end: pool.make_swa_page_handles(
            allocated[start:end],
            page_size,
        ),
    )
    cm.lock(prefix_handle)

    req = Req(
        input_ids=input_ids,
        table_idx=0,
        cached_len=8 * page_size,
        output_len=2,
        uid=0,
        sampling_params=SamplingParams(max_tokens=2),
        cache_handle=prefix_handle,
    )
    before_handle = prefix_handle.get_dsv4_swa_pages()
    assert before_handle is not None and before_handle.swa_pages is not None
    before_pages = before_handle.swa_pages.clone()

    cm.release_active_dsv4_swa_out_of_window(req)

    after_handle = prefix_handle.get_dsv4_swa_pages()
    assert after_handle is not None and after_handle.swa_pages is not None
    assert after_handle.swa_pages.tolist() == before_pages.tolist()
    assert req.swa_evicted_seqlen == 6 * page_size
    protected_starts = torch.arange(0, prefix_tokens, page_size, dtype=torch.int32)
    protected_swa = pool.swa_pages_from_full_page_starts(protected_starts, page_size)
    assert protected_swa is not None and torch.all(protected_swa >= 0)
    released_starts = torch.arange(prefix_tokens, 6 * page_size, page_size, dtype=torch.int32)
    released_swa = pool.swa_pages_from_full_page_starts(released_starts, page_size)
    assert released_swa is not None and released_swa.tolist() == [-1, -1]

    cm.cache_req(req, finished=True)
    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_active_swa_release_uses_monotonic_frontier():
    page_size = 4
    num_pages = 64
    pool = _make_dsv4_pool(
        [4, 0],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=6,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)
    req = _allocate_req_with_ids(
        cm,
        0,
        torch.arange(10 * page_size, dtype=torch.int32),
        output_len=1,
    )
    req.device_len = 8 * page_size
    req.cached_len = req.device_len - 1

    cm.release_active_dsv4_swa_out_of_window(req)
    first = pool.runtime_swa_counters()
    assert req.swa_evicted_seqlen == 6 * page_size
    assert first["swa_pages_freed_total"] == 6

    cm.release_active_dsv4_swa_out_of_window(req)
    assert req.swa_evicted_seqlen == 6 * page_size
    assert pool.runtime_swa_counters()["swa_pages_freed_total"] == 6

    req.device_len = 10 * page_size
    req.cached_len = req.device_len - 1
    cm.release_active_dsv4_swa_out_of_window(req)
    after_active = pool.runtime_swa_counters()
    assert req.swa_evicted_seqlen == 8 * page_size
    assert after_active["swa_pages_freed_total"] == 8

    req.cached_len = req.device_len
    cm.cache_req(req, finished=True)
    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_finished_cache_req_commits_swa_tombstone_from_frontier():
    page_size = 4
    num_pages = 64
    pool = _make_dsv4_pool(
        [4, 0],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=6,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)
    req = _allocate_req_with_ids(
        cm,
        0,
        torch.arange(8 * page_size, dtype=torch.int32),
        output_len=1,
    )

    cm.release_active_dsv4_swa_out_of_window(req)
    after_active = pool.runtime_swa_counters()
    req.cached_len = req.device_len
    cm.cache_req(req, finished=True)
    handle = cm.prefix_cache.match_prefix(req.input_ids).cuda_handle
    swa = handle.get_dsv4_swa_pages()
    assert swa is not None and swa.swa_pages is not None
    assert swa.swa_pages.tolist()[:6] == [-1] * 6
    assert all(page >= 0 for page in swa.swa_pages.tolist()[6:])
    after_finish = pool.runtime_swa_counters()
    assert after_finish["swa_pages_freed_total"] == after_active["swa_pages_freed_total"]
    assert after_finish["swa_pages_tombstoned_total"] == after_active["swa_pages_tombstoned_total"]

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_unfinished_cache_req_commits_swa_tombstone_from_frontier():
    page_size = 4
    num_pages = 64
    pool = _make_dsv4_pool(
        [4, 0],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=6,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)
    req = _allocate_req_with_ids(
        cm,
        0,
        torch.arange(8 * page_size, dtype=torch.int32),
        output_len=1,
    )

    cm.release_active_dsv4_swa_out_of_window(req)
    req.cached_len = req.device_len
    cm.cache_req(req, finished=False)

    swa = req.cache_handle.get_dsv4_swa_pages()
    assert swa is not None and swa.swa_pages is not None
    assert swa.swa_pages.tolist()[:6] == [-1] * 6
    assert all(page >= 0 for page in swa.swa_pages.tolist()[6:])
    assert req.cache_handle.cached_len == 8 * page_size

    cm.cache_req(req, finished=True)
    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_swa_pressure_eviction_versions_metadata():
    page_size = 256
    num_pages = 8
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
        max_running_req=3,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)
    prompt = torch.arange(3 * page_size + 1, dtype=torch.int32)
    _cache_finished_prompt(cm, 0, prompt)

    before = pool.swa_ownership_version
    released = cm.prefix_cache.release_dsv4_evictable_swa_pages(1)
    assert released == 1
    assert pool.swa_ownership_version > before

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_swa_double_free_guard_still_rejects_duplicate_owner():
    page_size = 4
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=8,
        page_size=page_size,
        max_running_req=2,
    )
    page_starts = torch.tensor([0], dtype=torch.int32)
    pool.on_pages_allocated(page_starts, page_size)
    handle = pool.make_swa_page_handles(torch.arange(page_size, dtype=torch.int32), page_size)

    pool.release_swa_page_handles(handle, tombstone=True)
    with pytest.raises(RuntimeError, match="DSV4 KV cache double free detected in SWA page slots"):
        pool.release_swa_page_handles(handle, tombstone=True)

    dummy_handle = DSV4SWAPageHandles(
        length=page_size,
        page_size=page_size,
        swa_pages=torch.tensor([pool._swa_dummy_page], dtype=torch.int32),
    )
    before = pool.runtime_swa_counters()
    pool.release_swa_page_handles(dummy_handle, tombstone=True)
    assert pool.runtime_swa_counters() == before
    assert pool._swa_page_refcount[pool._swa_dummy_page].item() == 1

    pool.on_token_indices_freed(torch.arange(page_size, dtype=torch.int32), page_size)
    pool.assert_no_leak()


def test_dsv4_swa_independent_dummy_full_sentinel_maps_to_swa_dummy_only():
    page_size = 4
    pool = _make_dsv4_pool(
        [0, 4, 128],
        num_pages=8,
        page_size=page_size,
        max_running_req=2,
    )
    dummy_full_loc = torch.tensor([pool.num_tokens], dtype=torch.int32)

    swa_locs = pool.translate_full_locs_to_swa_locs(dummy_full_loc)
    assert swa_locs.tolist() == [
        (pool.runtime_swa_counters()["swa_capacity_pages"] - 1) * page_size
    ]

    swa_pages = pool.swa_pages_from_full_page_starts(dummy_full_loc, page_size)
    assert swa_pages is not None
    assert swa_pages.tolist() == [pool.runtime_swa_counters()["swa_capacity_pages"] - 1]

    c4_pages, c128_pages, indexer_pages = pool.component_pages_from_full_page_starts(
        dummy_full_loc,
        page_size,
    )
    assert c4_pages is not None and c4_pages.tolist() == [-1]
    assert c128_pages is not None and c128_pages.tolist() == [-1]
    assert indexer_pages is not None and indexer_pages.tolist() == [-1]
    pool.assert_no_leak()


def test_dsv4_swa_independent_translate_full_locs_is_vectorized_for_mixed_rows():
    page_size = 4
    pool = _make_dsv4_pool(
        [0, 4],
        num_pages=8,
        page_size=page_size,
        max_running_req=2,
    )
    allocated_page_starts = torch.tensor([4, 12], dtype=torch.int32)
    pool.on_pages_allocated(allocated_page_starts, page_size)

    dummy_loc = torch.tensor([pool.num_tokens], dtype=torch.int32)
    locs = torch.tensor([5, 14, 0, -1, int(dummy_loc.item())], dtype=torch.int32)
    swa_locs = pool.translate_full_locs_to_swa_locs(locs)
    dummy_swa_page = pool.runtime_swa_counters()["swa_capacity_pages"] - 1

    assert swa_locs.tolist() == [1, page_size + 2, -1, -1, dummy_swa_page * page_size]

    freed = torch.cat(
        [
            torch.arange(4, 8, dtype=torch.int32),
            torch.arange(12, 16, dtype=torch.int32),
        ]
    )
    pool.on_token_indices_freed(freed, page_size)
    pool.assert_no_leak()


def test_dsv4_swa_independent_engine_dummy_page_maps_to_swa_dummy_only():
    page_size = 4
    planned_pages = 7
    dummy_full_loc = planned_pages * page_size
    pool = _make_dsv4_pool(
        [0, 4, 128],
        num_pages=planned_pages + 1,
        page_size=page_size,
        max_running_req=2,
        dsv4_dummy_token_start=dummy_full_loc,
    )

    assert pool.num_tokens == (planned_pages + 1) * page_size
    assert pool.dummy_token_start == dummy_full_loc
    assert dummy_full_loc != pool.num_tokens

    swa_locs = pool.translate_full_locs_to_swa_locs(
        torch.tensor([dummy_full_loc], dtype=torch.int32)
    )
    dummy_swa_page = pool.runtime_swa_counters()["swa_capacity_pages"] - 1
    assert swa_locs.tolist() == [dummy_swa_page * page_size]

    swa_pages = pool.swa_pages_from_full_page_starts(
        torch.tensor([dummy_full_loc], dtype=torch.int32),
        page_size,
    )
    assert swa_pages is not None
    assert swa_pages.tolist() == [dummy_swa_page]

    pool.swa_cache(0).zero_()
    kv = torch.ones((1, pool.swa_cache(0).shape[1]), dtype=torch.float32)
    pool.store_swa(0, kv, torch.tensor([dummy_full_loc], dtype=torch.int32))
    assert torch.all(pool.swa_cache(0)[dummy_swa_page * page_size] == 1)
    assert pool.allocation_counts.swa_pages == 0
    pool.assert_no_leak()


@pytest.mark.parametrize(
    ("prompt_len", "expected_hit"),
    [
        (257, 256),
        (512, 256),
        (513, 512),
        (768, 512),
        (769, 768),
    ],
)
def test_dsv4_component_lifecycle_serving_boundaries_do_not_reuse_stale_mappings(
    prompt_len: int,
    expected_hit: int,
):
    page_size = 256
    num_pages = 12
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    prompt = torch.arange(prompt_len, dtype=torch.int32)
    _cache_serving_prompt(cm, 0, prompt)

    handle = cm.match_req(PendingReq(1, prompt, SamplingParams(max_tokens=2))).cuda_handle
    assert handle.cached_len == expected_hit
    cm.lock(handle)
    cm.unlock(handle)

    _cache_serving_prompt(cm, 1, prompt)
    cm.check_integrity()

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_component_lifecycle_does_not_build_handles_for_tombstoned_heads(
    monkeypatch,
):
    page_size = 256
    num_pages = 8
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)
    calls: list[torch.Tensor] = []
    original_make_handles = pool.make_component_page_handles

    def record_make_handles(
        full_indices: torch.Tensor,
        page_size_arg: int,
    ):
        calls.append(full_indices.clone())
        return original_make_handles(full_indices, page_size_arg)

    monkeypatch.setattr(pool, "make_component_page_handles", record_make_handles)
    req = _allocate_req_with_ids(
        cm,
        0,
        torch.arange(512, dtype=torch.int32),
        output_len=2,
    )

    _complete_one_generated(req, 40_000)
    cm.cache_req(req, finished=False)
    assert calls
    assert calls[-1].numel() == 2 * page_size
    assert torch.all(calls[-1] >= 0)
    assert cm.page_table[req.table_idx, 0].item() == -1
    assert cm.page_table[req.table_idx, page_size].item() >= 0

    cm.allocate_paged([req])
    _complete_one_generated(req, 40_001)
    call_count = len(calls)
    cm.cache_req(req, finished=True)
    assert len(calls) == call_count
    cm.check_integrity()

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_component_lifecycle_partial_and_mixed_hit_miss_batches_are_safe():
    page_size = 256
    num_pages = 24
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    warm_prefix = torch.arange(257, dtype=torch.int32)
    _cache_serving_prompt(cm, 0, warm_prefix)
    partial = torch.cat([warm_prefix, torch.arange(512, dtype=torch.int32) + 10_000])
    partial_req = _allocate_req_with_ids(cm, 1, partial, output_len=2)
    assert partial_req.cache_handle.cached_len == page_size
    _complete_one_generated(partial_req, 41_000)
    cm.cache_req(partial_req, finished=False)
    cm.allocate_paged([partial_req])
    _complete_one_generated(partial_req, 41_001)
    cm.cache_req(partial_req, finished=True)

    warm_full = torch.arange(769, dtype=torch.int32) + 20_000
    _cache_serving_prompt(cm, 2, warm_full)
    before = cm.prefix_metrics_snapshot()
    for offset in range(6):
        prompt = (
            warm_full.clone()
            if offset % 2 == 0
            else torch.arange(769, dtype=torch.int32) + 30_000 + offset * 1_000
        )
        _cache_serving_prompt(cm, 3 + offset, prompt)

    after = cm.prefix_metrics_snapshot()
    assert after["hit_requests"] > before["hit_requests"]
    assert after["miss_requests"] > before["miss_requests"]
    cm.check_integrity()

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_component_lifecycle_multi_prefix_sustained_reuse_and_evict():
    page_size = 256
    num_pages = 32
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    prefixes = [torch.arange(512, dtype=torch.int32) + i * 10_000 for i in range(4)]
    uid = 0
    for wave in range(3):
        for prefix_id, prefix in enumerate(prefixes):
            suffix = torch.arange(64, dtype=torch.int32) + 100_000 + wave * 1_000 + prefix_id * 100
            _cache_serving_prompt(cm, uid, torch.cat([prefix, suffix]))
            uid += 1

    snapshot = cm.prefix_metrics_snapshot()
    assert snapshot["hit_requests"] >= 8
    assert snapshot["saved_prefill_tokens"] >= 8 * 512
    cm.check_integrity()

    _evict_all_prefix(cm, pool, page_size)
    pool.assert_no_leak()


def test_dsv4_component_lifecycle_repeated_hit_evict_has_no_double_free_or_leak():
    page_size = 256
    num_pages = 8
    pool = _make_dsv4_pool(
        [4, 128, 0],
        num_pages=num_pages,
        page_size=page_size,
    )
    cm = _make_cache_manager(pool, num_pages, page_size)

    for cycle in range(3):
        prompt = torch.arange(513, dtype=torch.int32) + cycle * 10_000
        _cache_serving_prompt(cm, cycle * 2, prompt)
        hit = cm.match_req(PendingReq(cycle * 2 + 1, prompt, SamplingParams(max_tokens=2)))
        assert hit.cuda_handle.cached_len == 2 * page_size
        _cache_serving_prompt(cm, cycle * 2 + 1, prompt)
        _evict_all_prefix(cm, pool, page_size)
        assert cm.prefix_cache.size_info.total_size == 0
        pool.assert_no_leak()

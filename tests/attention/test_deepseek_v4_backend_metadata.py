from __future__ import annotations

import pytest
import torch

import minisgl.core as core
import minisgl.distributed.info as dist_info
from minisgl.attention import create_attention_backend
from minisgl.attention.deepseek_v4 import DSV4AttentionMetadata
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.models.config import ModelConfig, RotaryConfig


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
        window_size=128,
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


@pytest.fixture(autouse=True)
def reset_globals():
    old_ctx = core._GLOBAL_CTX
    old_tp = dist_info._TP_INFO
    core._GLOBAL_CTX = None
    dist_info._TP_INFO = None
    set_tp_info(0, 1)
    yield
    core._GLOBAL_CTX = old_ctx
    dist_info._TP_INFO = old_tp


def _req(uid: int, table_idx: int, device_len: int, cached_len: int = 0) -> Req:
    return Req(
        input_ids=torch.arange(device_len, dtype=torch.int32) + uid * 1000,
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=1,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=None,  # type: ignore[arg-type]
    )


def _install_context(
    cfg: ModelConfig,
    *,
    page_size: int,
    table_bases: list[int],
    max_len: int,
) -> Context:
    ctx = Context(page_size=page_size)
    ctx.kv_cache = create_kvcache_pool(
        cfg,
        num_pages=512,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    page_table = torch.full((len(table_bases), max_len), -1, dtype=torch.int32)
    for row, base in enumerate(table_bases):
        page_table[row] = torch.arange(base, base + max_len, dtype=torch.int32)
    ctx.page_table = page_table
    core.set_global_ctx(ctx)
    ctx.attn_backend = create_attention_backend("dsv4", cfg)
    return ctx


def _prepare_batch(reqs: list[Req]) -> Batch:
    ctx = core.get_global_ctx()
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    batch.positions = torch.cat(
        [
            torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
            for req in reqs
        ]
    )
    batch.out_loc = ctx.page_table[
        torch.cat(
            [
                torch.full((req.extend_len,), req.table_idx, dtype=torch.long)
                for req in reqs
            ]
        ),
        batch.positions.long(),
    ]
    batch.input_ids = torch.cat([req.input_ids[req.cached_len :] for req in reqs])
    return batch


def _prepare_decode_batch(reqs: list[Req]) -> Batch:
    ctx = core.get_global_ctx()
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.positions = torch.tensor([req.device_len - 1 for req in reqs], dtype=torch.int32)
    batch.out_loc = ctx.page_table[
        torch.tensor([req.table_idx for req in reqs], dtype=torch.long),
        batch.positions.long(),
    ]
    batch.input_ids = torch.tensor([int(req.input_ids[-1]) for req in reqs], dtype=torch.int32)
    return batch


def test_dsv4_metadata_builds_sequence_lengths_positions_and_last_indices():
    cfg = _tiny_dsv4_config([0])
    _install_context(cfg, page_size=1, table_bases=[0], max_len=8)
    batch = _prepare_batch([_req(0, 0, 3)])

    core.get_global_ctx().attn_backend.prepare_metadata(batch)
    assert isinstance(batch.attn_metadata, DSV4AttentionMetadata)
    meta = batch.attn_metadata.core_metadata

    assert meta.seq_lens.tolist() == [1, 2, 3]
    assert meta.positions.tolist() == [0, 1, 2]
    assert meta.raw_out_loc.tolist() == [0, 1, 2]
    assert meta.cu_seqlens_q.tolist() == [0, 3]
    assert batch.attn_metadata.get_last_indices(1).tolist() == [2]


def test_dsv4_swa_window_boundaries_below_equal_and_above_128():
    cfg = _tiny_dsv4_config([0])
    _install_context(cfg, page_size=1, table_bases=[0, 256, 512], max_len=160)
    reqs = [
        _req(0, 0, 127, cached_len=126),
        _req(1, 1, 128, cached_len=127),
        _req(2, 2, 130, cached_len=129),
    ]
    batch = _prepare_decode_batch(reqs)

    core.get_global_ctx().attn_backend.prepare_metadata(batch)
    meta = batch.attn_metadata.core_metadata

    assert meta.swa_topk_lengths.tolist() == [127, 128, 128]
    assert int((meta.swa_page_indices[0] >= 0).sum().item()) == 127
    assert int((meta.swa_page_indices[1] >= 0).sum().item()) == 128
    assert int((meta.swa_page_indices[2] >= 0).sum().item()) == 128
    assert meta.swa_page_indices[2, 0].item() == 512 + 129
    assert meta.swa_page_indices[2, 127].item() == 512 + 2


def test_dsv4_ratio_dispatch_and_fallback_attention_shapes():
    cfg = _tiny_dsv4_config([0, 4, 128])
    _install_context(cfg, page_size=1, table_bases=[0], max_len=260)
    batch = _prepare_batch([_req(0, 0, 256)])
    backend = core.get_global_ctx().attn_backend
    backend.prepare_metadata(batch)
    meta = batch.attn_metadata.core_metadata

    assert [backend.get_layer_compress_ratio(i) for i in range(3)] == [0, 4, 128]
    assert meta.c4_out_loc[:3].tolist() == [3 // 4, 7 // 4, 11 // 4]
    assert meta.c128_out_loc.tolist() == [127 // 128, 255 // 128]

    q = torch.randn(256, 4, 8, dtype=torch.bfloat16)
    kv = torch.randn(256, 8, dtype=torch.bfloat16)
    for layer_id, ratio in enumerate([0, 4, 128]):
        out = backend.forward(
            q,
            kv,
            kv,
            layer_id,
            batch,
            compress_ratio=ratio,
            attn_sink=torch.zeros(4),
        )
        assert out.shape == q.shape
        assert torch.isfinite(out.float()).all()


def test_dsv4_indexer_select_updates_c4_sparse_metadata():
    cfg = _tiny_dsv4_config([4])
    ctx = _install_context(cfg, page_size=4, table_bases=[0], max_len=16)
    batch = _prepare_decode_batch([_req(0, 0, 16, cached_len=15)])
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)
    assert isinstance(batch.attn_metadata, DSV4AttentionMetadata)

    cache = ctx.kv_cache.indexer_cache(0)
    cache.zero_()
    cache[:4] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ],
        dtype=cache.dtype,
    )
    q = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=cache.dtype,
    )
    weights = torch.ones(1, 2, dtype=torch.float32)

    out = backend.select_indexer(0, q, weights, batch)

    assert out is not None
    meta = batch.attn_metadata.core_metadata
    assert sorted(meta.c4_sparse_raw_indices[0, :2].tolist()) == [1, 2]
    assert sorted(meta.c4_sparse_page_indices[0, :2].tolist()) == [1, 2]
    assert sorted(meta.c4_sparse_full_indices[0, :2].tolist()) == [7, 11]
    assert meta.c4_sparse_raw_indices.shape[1] % 64 == 0


def test_dsv4_metadata_repeats_page_table_for_multi_request_batch():
    cfg = _tiny_dsv4_config([4])
    _install_context(cfg, page_size=4, table_bases=[0, 64], max_len=12)
    batch = _prepare_batch([_req(0, 0, 6), _req(1, 1, 9)])

    core.get_global_ctx().attn_backend.prepare_metadata(batch)
    meta = batch.attn_metadata.core_metadata

    assert meta.page_table.shape == (15, 3)
    assert meta.page_table[0].tolist() == [0, 1, 2]
    assert meta.page_table[6].tolist() == [16, 17, 18]
    assert meta.req_table_indices[:6].tolist() == [0] * 6
    assert meta.req_table_indices[6:].tolist() == [1] * 9
    assert meta.c4_sparse_page_indices.shape[1] % 64 == 0

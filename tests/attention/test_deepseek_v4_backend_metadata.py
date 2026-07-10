from __future__ import annotations

from types import SimpleNamespace

import minisgl.core as core
import minisgl.distributed.info as dist_info
import pytest
import torch
from minisgl.attention import create_attention_backend
from minisgl.attention.deepseek_v4 import DSV4AttentionMetadata, DSV4CoreAttentionMetadata
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.kernel import deepseek_v4 as dsv4_kernel
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
    enable_component_loc_ownership: bool = False,
    enable_swa_independent_lifecycle: bool = False,
) -> Context:
    ctx = Context(page_size=page_size)
    ctx.kv_cache = create_kvcache_pool(
        cfg,
        num_pages=512,
        page_size=page_size,
        dtype=torch.float16,
        device=torch.device("cpu"),
        enable_dsv4_component_loc_ownership=enable_component_loc_ownership,
        enable_dsv4_swa_independent_lifecycle=enable_swa_independent_lifecycle,
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
    ctx = _install_context(cfg, page_size=1, table_bases=[0], max_len=260)
    batch = _prepare_batch([_req(0, 0, 256)])
    backend = core.get_global_ctx().attn_backend
    backend.prepare_metadata(batch)
    meta = batch.attn_metadata.core_metadata

    assert [backend.get_layer_compress_ratio(i) for i in range(3)] == [0, 4, 128]
    assert meta.c4_out_loc[:3].tolist() == [3 // 4, 7 // 4, 11 // 4]
    assert meta.c128_out_loc.tolist() == [127 // 128, 255 // 128]

    ctx.kv_cache.swa_cache(0).zero_()
    ctx.kv_cache.swa_cache(1).zero_()
    ctx.kv_cache.swa_cache(2).zero_()
    ctx.kv_cache.c4_cache(1).zero_()
    ctx.kv_cache.c128_cache(2).zero_()

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


def test_dsv4_capture_replay_uses_fixed_masked_compressed_locs():
    cfg = _tiny_dsv4_config([4, 128])
    ctx = _install_context(cfg, page_size=1, table_bases=[0, 512, 1024], max_len=260)
    reqs = [
        _req(0, 0, 4, cached_len=3),
        _req(1, 1, 5, cached_len=4),
        _req(2, 2, 128, cached_len=127),
    ]
    batch = _prepare_decode_batch(reqs)
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)
    meta = batch.attn_metadata.core_metadata

    assert meta.c4_out_loc.tolist() == [3 // 4, (1024 + 127) // 4]
    assert meta.c128_out_loc.tolist() == [(1024 + 127) // 128]

    backend.init_capture_graph(max_seq_len=260, bs_list=[4])
    backend.prepare_for_replay(batch)
    assert backend.capture is not None
    capture = backend.capture.core_metadata

    assert capture.c4_out_loc.tolist() == [3 // 4, -1, (1024 + 127) // 4, -1]
    assert capture.c128_out_loc.tolist() == [-1, -1, (1024 + 127) // 128, -1]
    assert backend.capture.c4_compress_metadata.write_loc is capture.c4_out_loc
    assert backend.capture.c128_compress_metadata.write_loc is capture.c128_out_loc


def test_dsv4_capture_replay_can_bind_graph_out_loc_and_positions():
    cfg = _tiny_dsv4_config([4, 128])
    ctx = _install_context(cfg, page_size=1, table_bases=[0, 512, 1024], max_len=260)
    reqs = [
        _req(0, 0, 4, cached_len=3),
        _req(1, 1, 5, cached_len=4),
        _req(2, 2, 128, cached_len=127),
    ]
    batch = _prepare_decode_batch(reqs)
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)

    backend.init_capture_graph(max_seq_len=260, bs_list=[4])
    graph_out_loc = torch.full((4,), -123, dtype=torch.int32)
    graph_positions = torch.full((4,), -456, dtype=torch.int32)
    backend.bind_capture_graph_inputs(
        input_ids=torch.empty(4, dtype=torch.int32),
        out_loc=graph_out_loc,
        positions=graph_positions,
    )
    assert backend.capture is not None
    capture = backend.capture.core_metadata
    assert capture.raw_out_loc is graph_out_loc
    assert capture.positions is graph_positions
    assert backend.capture.c4_compress_metadata.positions is graph_positions
    assert backend.capture.c128_compress_metadata.positions is graph_positions

    graph_out_loc[: batch.padded_size].copy_(batch.out_loc)
    graph_positions[: batch.padded_size].copy_(batch.positions)
    backend.prepare_for_replay(batch)

    assert capture.raw_out_loc is graph_out_loc
    assert capture.positions is graph_positions
    assert capture.raw_out_loc.tolist() == batch.out_loc.tolist() + [-123]
    assert capture.positions.tolist() == batch.positions.tolist() + [-456]
    assert capture.c4_out_loc.tolist() == [3 // 4, -1, (1024 + 127) // 4, -1]
    assert capture.c128_out_loc.tolist() == [-1, -1, (1024 + 127) // 128, -1]


def test_dsv4_capture_replay_can_defer_compressed_locs_to_graph_hook(monkeypatch):
    cfg = _tiny_dsv4_config([4, 128])
    ctx = _install_context(cfg, page_size=1, table_bases=[0, 512, 1024], max_len=260)
    reqs = [
        _req(0, 0, 4, cached_len=3),
        _req(1, 1, 5, cached_len=4),
        _req(2, 2, 128, cached_len=127),
    ]
    batch = _prepare_decode_batch(reqs)
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)

    def fake_triton_enabled(toggle: str) -> bool:
        return toggle == "MINISGL_DSV4_SM80_COMPRESS_STORE"

    def fake_copy_masked_compressed_locs(
        raw_out_loc: torch.Tensor,
        positions: torch.Tensor,
        c4_out_loc: torch.Tensor | None,
        c128_out_loc: torch.Tensor | None,
        rows: int,
    ) -> None:
        for dst, ratio in ((c4_out_loc, 4), (c128_out_loc, 128)):
            assert dst is not None
            values = torch.where(
                (positions[:rows] + 1) % ratio == 0,
                raw_out_loc[:rows].div(ratio, rounding_mode="floor"),
                torch.full_like(raw_out_loc[:rows], -1),
            )
            dst[:rows].copy_(values)
            dst[rows:].fill_(-1)

    monkeypatch.setattr(dsv4_kernel, "dsv4_sm80_triton_enabled", fake_triton_enabled)
    monkeypatch.setattr(
        dsv4_kernel,
        "copy_masked_compressed_locs",
        fake_copy_masked_compressed_locs,
    )

    backend.init_capture_graph(max_seq_len=260, bs_list=[4])
    graph_out_loc = torch.full((4,), -123, dtype=torch.int32)
    graph_positions = torch.full((4,), -456, dtype=torch.int32)
    backend.bind_capture_graph_inputs(
        input_ids=torch.empty(4, dtype=torch.int32),
        out_loc=graph_out_loc,
        positions=graph_positions,
    )
    assert backend.capture_compressed_locs_in_graph
    assert backend.capture is not None
    capture = backend.capture.core_metadata

    graph_out_loc[: batch.padded_size].copy_(batch.out_loc)
    graph_positions[: batch.padded_size].copy_(batch.positions)
    backend.prepare_for_replay(batch)

    assert capture.c4_out_loc.tolist() == [-1, -1, -1, -1]
    assert capture.c128_out_loc.tolist() == [-1, -1, -1, -1]

    capture_batch = Batch(reqs=[reqs[0]] * 4, phase="decode")
    capture_batch.padded_reqs = capture_batch.reqs
    backend.prepare_for_capture(capture_batch)
    backend.stage_capture_metadata_for_graph(capture_batch)

    assert capture.c4_out_loc.tolist() == [3 // 4, -1, (1024 + 127) // 4, -1]
    assert capture.c128_out_loc.tolist() == [-1, -1, (1024 + 127) // 128, -1]


def test_dsv4_capture_compressed_locs_graph_hook_can_be_disabled(monkeypatch):
    cfg = _tiny_dsv4_config([4, 128])
    ctx = _install_context(cfg, page_size=1, table_bases=[0], max_len=260)
    backend = ctx.attn_backend
    monkeypatch.setenv("MINISGL_DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH", "1")
    monkeypatch.setattr(
        dsv4_kernel,
        "dsv4_sm80_triton_enabled",
        lambda toggle: toggle == "MINISGL_DSV4_SM80_COMPRESS_STORE",
    )

    backend.init_capture_graph(max_seq_len=260, bs_list=[4])
    backend.bind_capture_graph_inputs(
        input_ids=torch.empty(4, dtype=torch.int32),
        out_loc=torch.full((4,), -1, dtype=torch.int32),
        positions=torch.full((4,), -1, dtype=torch.int32),
    )

    assert backend.capture_compressed_locs_in_graph_disabled_by_env
    assert not backend.capture_compressed_locs_in_graph


def test_dsv4_masked_compressed_store_ignores_negative_locs():
    cfg = _tiny_dsv4_config([4])
    ctx = _install_context(cfg, page_size=1, table_bases=[0], max_len=16)
    cache = ctx.kv_cache.c4_cache(0)
    cache.zero_()
    kv = torch.tensor([[11.0] + [0.0] * 7, [22.0] + [0.0] * 7], dtype=cache.dtype)
    loc = torch.tensor([-1, 2], dtype=torch.int32)

    dsv4_kernel.store_compressed_fallback(ctx.kv_cache, 0, kv, loc)

    assert cache[2, 0].item() == pytest.approx(22.0)
    assert cache[-1, 0].item() == pytest.approx(0.0)


def test_dsv4_fallback_attention_reads_compressed_cache_as_separate_source():
    cfg = _tiny_dsv4_config([4])
    ctx = _install_context(cfg, page_size=1, table_bases=[0], max_len=16)
    batch = _prepare_decode_batch([_req(0, 0, 8, cached_len=7)])
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)

    ctx.kv_cache.swa_cache(0).zero_()
    ctx.kv_cache.c4_cache(0).zero_()
    ctx.kv_cache.c4_cache(0)[0, 0] = 10.0
    ctx.kv_cache.c4_cache(0)[1, 0] = 20.0

    q = torch.zeros(1, cfg.num_qo_heads, cfg.head_dim, dtype=torch.bfloat16)
    q[..., 0] = 8.0
    kv = torch.zeros(1, cfg.head_dim, dtype=torch.bfloat16)

    out = backend.forward(
        q,
        kv,
        kv,
        0,
        batch,
        compress_ratio=4,
        attn_sink=None,
        swa_cache_written=True,
    )

    assert out[..., 0].float().mean().item() > 15.0
    assert torch.allclose(out[..., 1:].float(), torch.zeros_like(out[..., 1:].float()))


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
    assert meta.c4_sparse_topk_lengths[0].item() == 2
    assert meta.c4_sparse_raw_indices.shape[1] % 64 == 0


def test_dsv4_indexer_select_preserves_metadata_buffer_identity():
    cfg = _tiny_dsv4_config([4])
    ctx = _install_context(cfg, page_size=4, table_bases=[0], max_len=16)
    backend = ctx.attn_backend
    backend.init_capture_graph(max_seq_len=16, bs_list=[1])
    batch = Batch(reqs=[_req(0, 0, 16, cached_len=15)], phase="decode")
    batch.padded_reqs = batch.reqs
    backend.prepare_for_capture(batch)
    assert isinstance(batch.attn_metadata, DSV4AttentionMetadata)
    meta = batch.attn_metadata.core_metadata

    raw_indices = meta.c4_sparse_raw_indices
    page_indices = meta.c4_sparse_page_indices
    full_indices = meta.c4_sparse_full_indices
    topk_lengths = meta.c4_sparse_topk_lengths

    ctx.kv_cache.indexer_cache(0).zero_()
    q = torch.ones(1, cfg.index_n_heads, cfg.index_head_dim, dtype=torch.bfloat16)
    weights = torch.ones(1, cfg.index_n_heads, dtype=torch.float32)

    out = backend.select_indexer(0, q, weights, batch)

    assert out is not None
    assert meta.c4_sparse_raw_indices is raw_indices
    assert meta.c4_sparse_page_indices is page_indices
    assert meta.c4_sparse_full_indices is full_indices
    assert meta.c4_sparse_topk_lengths is topk_lengths
    assert meta.c4_sparse_topk_lengths[0].item() > 0


def test_dsv4_indexer_select_uses_c4_attention_table_for_attention_rows():
    cfg = _tiny_dsv4_config([4])
    page_size = 128
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[0],
        max_len=page_size,
        enable_component_loc_ownership=True,
    )
    ctx.kv_cache.on_pages_allocated(torch.tensor([0], dtype=torch.int32), page_size)
    batch = _prepare_decode_batch([_req(0, 0, page_size, cached_len=page_size - 1)])
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)
    assert isinstance(batch.attn_metadata, DSV4AttentionMetadata)
    meta = batch.attn_metadata.core_metadata
    assert batch.attn_metadata.indexer_metadata.page_table is meta.c4_indexer_page_table
    assert batch.attn_metadata.indexer_metadata.c4_page_size == page_size // 4

    meta.c4_page_table[0, 0] = 10
    meta.c4_indexer_page_table[0, 0] = 20
    ctx.kv_cache.indexer_cache(0).zero_()
    q = torch.ones(1, cfg.index_n_heads, cfg.index_head_dim, dtype=torch.bfloat16)
    weights = torch.ones(1, cfg.index_n_heads, dtype=torch.float32)

    out = backend.select_indexer(0, q, weights, batch)

    assert out is not None
    active = int(meta.c4_sparse_topk_lengths[0].item())
    assert active == cfg.index_topk
    raw = meta.c4_sparse_raw_indices[0, :active].to(torch.long)
    assert torch.all(raw >= 0)
    component_page_size = ctx.kv_cache.c4_component_page_size
    logical_pages = raw.div(component_page_size, rounding_mode="floor")
    offsets = raw % component_page_size
    expected_attention_locs = (
        meta.c4_page_table[0, logical_pages].to(torch.long) * component_page_size + offsets
    ).tolist()
    expected_indexer_locs = (
        meta.c4_indexer_page_table[0, logical_pages].to(torch.long)
        * component_page_size
        + offsets
    ).tolist()
    expected_full_locs = ctx.page_table[0, raw * 4 + 3].tolist()
    assert meta.c4_sparse_page_indices[0, :active].tolist() == expected_attention_locs
    assert meta.c4_sparse_page_indices[0, :active].tolist() != expected_indexer_locs
    assert meta.c4_sparse_full_indices[0, :active].tolist() == expected_full_locs


def test_dsv4_component_loc_ownership_metadata_uses_direct_component_tables():
    cfg = _tiny_dsv4_config([4, 128])
    page_size = 128
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[0],
        max_len=384,
        enable_component_loc_ownership=True,
    )
    page_starts = torch.tensor([0, page_size, 2 * page_size], dtype=torch.int32)
    ctx.kv_cache.on_pages_allocated(page_starts, page_size)
    prefix_handles = ctx.kv_cache.make_component_page_handles(
        ctx.page_table[0, : 2 * page_size],
        page_size,
    )
    assert prefix_handles is not None
    ctx.kv_cache.on_token_indices_freed(
        ctx.page_table[0, :page_size],
        page_size,
        free_components=False,
    )
    ctx.page_table[0, :page_size] = -1

    req = _req(0, 0, 258, cached_len=256)
    req.cache_handle = SimpleNamespace(get_dsv4_component_pages=lambda: prefix_handles)
    batch = _prepare_batch([req])

    ctx.attn_backend.prepare_metadata(batch)
    meta = batch.attn_metadata.core_metadata

    assert meta.component_loc_ownership
    assert meta.c128_page_indices[0, :2].tolist() == [0, 1]
    assert meta.c128_full_indices[0, :2].tolist() == [-1, 255]
    assert batch.attn_metadata.indexer_metadata.page_table[0, :3].tolist() == [0, 1, 2]


def test_dsv4_component_loc_ownership_capture_replay_copies_direct_component_metadata():
    cfg = _tiny_dsv4_config([4, 128])
    page_size = 128
    base = 1024
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[base],
        max_len=384,
        enable_component_loc_ownership=True,
    )
    page_starts = torch.tensor([base, base + page_size], dtype=torch.int32)
    ctx.kv_cache.on_pages_allocated(page_starts, page_size)
    prefix_handles = ctx.kv_cache.make_component_page_handles(
        ctx.page_table[0, : 2 * page_size],
        page_size,
    )
    assert prefix_handles is not None
    ctx.kv_cache.on_token_indices_freed(
        ctx.page_table[0, :page_size],
        page_size,
        free_components=False,
    )
    ctx.page_table[0, :page_size] = -1

    req = _req(0, 0, 256, cached_len=255)
    req.cache_handle = SimpleNamespace(get_dsv4_component_pages=lambda: prefix_handles)
    batch = _prepare_decode_batch([req])
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)
    src = batch.attn_metadata.core_metadata

    assert src.component_loc_ownership
    assert src.c4_out_loc.tolist() == [63]
    assert src.c128_out_loc.tolist() == [1]
    assert src.c4_indexer_out_loc.tolist() == [63]
    assert int(batch.out_loc[0].item() // 4) == 319
    assert int(batch.out_loc[0].item() // 128) == 9

    backend.init_capture_graph(max_seq_len=384, bs_list=[1])
    backend.prepare_for_replay(batch)

    assert backend.capture is not None
    capture = backend.capture.core_metadata
    assert capture.component_loc_ownership
    assert capture.c4_page_table[0, :3].tolist() == [0, 1, -1]
    assert capture.c128_page_table[0, :3].tolist() == [0, 1, -1]
    assert capture.c4_indexer_page_table[0, :3].tolist() == [0, 1, -1]
    assert capture.c4_out_loc.tolist() == [63]
    assert capture.c128_out_loc.tolist() == [1]
    assert capture.c4_indexer_out_loc.tolist() == [63]
    assert capture.c4_indexer_out_loc is not capture.c4_out_loc
    assert batch.attn_metadata.indexer_metadata.page_table is src.c4_indexer_page_table
    assert backend.capture.indexer_metadata.page_table is capture.c4_indexer_page_table


def test_dsv4_graph_replay_clamps_compressed_reads_to_materialized_prompt():
    cfg = _tiny_dsv4_config([4, 128])
    page_size = 128
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[0],
        max_len=512,
        enable_component_loc_ownership=True,
    )
    page_starts = torch.arange(0, 4 * page_size, page_size, dtype=torch.int32)
    ctx.kv_cache.on_pages_allocated(page_starts, page_size)
    prefix_handles = ctx.kv_cache.make_component_page_handles(
        ctx.page_table[0, : 2 * page_size],
        page_size,
    )
    assert prefix_handles is not None

    req = _req(0, 0, 384, cached_len=383)
    req.output_len = 256
    req.max_device_len = 512
    req.cache_handle = SimpleNamespace(get_dsv4_component_pages=lambda: prefix_handles)
    batch = _prepare_decode_batch([req])
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)
    src = batch.attn_metadata.core_metadata

    assert src.c4_topk_lengths_raw.tolist() == [96]
    assert src.c4_sparse_raw_indices[0, :2].tolist() == [94, 95]
    assert src.c4_sparse_page_indices[0, :2].tolist() == [94, 95]
    assert src.c128_raw_indices[0, :3].tolist() == [0, 1, 2]

    backend.init_capture_graph(max_seq_len=512, bs_list=[1])
    backend.prepare_for_replay(batch)

    assert backend.capture is not None
    capture = backend.capture.core_metadata
    assert src.c4_topk_lengths_raw.tolist() == [64]
    assert src.c4_sparse_raw_indices[0, :2].tolist() == [62, 63]
    assert capture.c4_topk_lengths_raw.tolist() == [64]
    assert capture.c4_sparse_raw_indices[0, :2].tolist() == [62, 63]
    assert capture.c4_sparse_page_indices[0, :2].tolist() == [62, 63]
    assert capture.c128_raw_indices[0, :3].tolist() == [0, 1, -1]


def test_dsv4_prep_metadata_oracle_splits_pre_and_post_forward_c4_sparse_boundary():
    cfg = _tiny_dsv4_config([4, 128])
    ctx = _install_context(cfg, page_size=128, table_bases=[0], max_len=512)
    backend = ctx.attn_backend

    def make_core(
        *,
        c4_sparse_topk_lengths: list[int],
        c4_sparse_raw_indices: list[list[int]],
    ) -> DSV4CoreAttentionMetadata:
        rows = len(c4_sparse_topk_lengths)
        raw_out_loc = torch.arange(rows, dtype=torch.int32)
        page_table = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4)
        cu_seqlens_q = torch.arange(rows + 1, dtype=torch.int32)
        seq_lens = torch.tensor([256 + row for row in range(rows)], dtype=torch.int32)
        positions = seq_lens - 1
        req_seq_lens = seq_lens.clone()
        extend_lens = torch.ones(rows, dtype=torch.int32)
        req_table_indices = torch.zeros(rows, dtype=torch.int32)
        swa_topk_lengths = torch.full((rows,), 4, dtype=torch.int32)
        swa_page_indices = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4)
        c4_topk_lengths_raw = torch.full((rows,), 64, dtype=torch.int32)
        c4_topk_lengths_clamp1 = c4_topk_lengths_raw.clone()
        c4_sparse_lens = torch.tensor(c4_sparse_topk_lengths, dtype=torch.int32)
        c4_raw = torch.tensor(c4_sparse_raw_indices, dtype=torch.int32)
        c4_page = c4_raw.clone()
        c4_full = c4_raw * 4 + 3
        c4_full = torch.where(c4_raw >= 0, c4_full, c4_raw)
        c128_lengths = torch.full((rows,), 2, dtype=torch.int32)
        c128_raw = torch.tensor([[0, 1, -1, -1] for _ in range(rows)], dtype=torch.int32)
        c128_page = c128_raw.clone()
        c128_full = torch.where(c128_raw >= 0, c128_raw * 128 + 127, c128_raw)
        return DSV4CoreAttentionMetadata(
            raw_out_loc=raw_out_loc,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            seq_lens=seq_lens,
            req_seq_lens=req_seq_lens,
            extend_lens=extend_lens,
            positions=positions,
            req_table_indices=req_table_indices,
            max_seqlen_q=1,
            max_seqlen_k=512,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_out_loc=None,
            c128_out_loc=None,
            c4_indexer_out_loc=None,
            c4_topk_lengths_raw=c4_topk_lengths_raw,
            c4_topk_lengths_clamp1=c4_topk_lengths_clamp1,
            c4_sparse_topk_lengths=c4_sparse_lens,
            c4_sparse_raw_indices=c4_raw,
            c4_sparse_page_indices=c4_page,
            c4_sparse_full_indices=c4_full,
            c128_topk_lengths_clamp1=c128_lengths,
            c128_raw_indices=c128_raw,
            c128_page_indices=c128_page,
            c128_full_indices=c128_full,
            materialized_seq_lens=seq_lens.clone(),
        )

    expected = make_core(c4_sparse_topk_lengths=[2], c4_sparse_raw_indices=[[62, 63, -1, -1]])
    pre_forward_mismatch = make_core(
        c4_sparse_topk_lengths=[2],
        c4_sparse_raw_indices=[[10, 11, -1, -1]],
    )
    with pytest.raises(RuntimeError, match="c4_sparse_raw_indices.*pre_forward"):
        backend._compare_prep_metadata_in_graph_oracle(
            pre_forward_mismatch,
            expected,
            1,
            boundary="pre_forward",
        )

    post_forward_indexer_mutated = make_core(
        c4_sparse_topk_lengths=[1],
        c4_sparse_raw_indices=[[10, -1, -1, -1]],
    )
    backend._compare_prep_metadata_in_graph_oracle(
        post_forward_indexer_mutated,
        expected,
        1,
        boundary="post_forward",
    )


def test_dsv4_route_b_component_page_table_lifetime_cache_invalidates_lifecycle(
    monkeypatch,
):
    cfg = _tiny_dsv4_config([4, 128])
    page_size = 128
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[0, 4 * page_size, 8 * page_size],
        max_len=4 * page_size,
        enable_component_loc_ownership=True,
    )
    page_starts = torch.arange(0, 12 * page_size, page_size, dtype=torch.int32)
    ctx.kv_cache.on_pages_allocated(page_starts, page_size)
    backend = ctx.attn_backend

    monkeypatch.setenv("MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE", "1")
    monkeypatch.setenv("MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY", "1")

    def cache_handle(row: int, cached_pages: int, node_uuid: int) -> SimpleNamespace:
        prefix_len = cached_pages * page_size
        handles = ctx.kv_cache.make_component_page_handles(
            ctx.page_table[row, :prefix_len],
            page_size,
        )
        return SimpleNamespace(
            cached_len=prefix_len,
            node=SimpleNamespace(uuid=node_uuid),
            get_dsv4_component_pages=lambda handles=handles: handles,
        )

    def decode_c4_pages(req: Req) -> list[int]:
        batch = _prepare_decode_batch([req])
        backend.prepare_metadata(batch)
        width = (req.device_len + page_size - 1) // page_size
        meta = batch.attn_metadata.core_metadata
        return meta.c4_page_table[0, :width].tolist()

    req = _req(11, 0, 2 * page_size + 1, cached_len=2 * page_size)
    req.cache_handle = cache_handle(0, 2, 100)
    assert decode_c4_pages(req) == [0, 1, 2]
    original_signature = backend._component_page_table_cache_signatures[0]

    assert decode_c4_pages(req) == [0, 1, 2]
    assert backend._component_page_table_cache_signatures[0] == original_signature

    ctx.page_table[0].copy_(ctx.page_table[1])
    reused_uid_and_node = _req(11, 0, 2 * page_size + 1, cached_len=2 * page_size)
    reused_uid_and_node.cache_handle = cache_handle(0, 2, 100)
    assert decode_c4_pages(reused_uid_and_node) == [4, 5, 6]

    ctx.page_table[0].copy_(ctx.page_table[1])
    reused_slot = _req(12, 0, 2 * page_size + 1, cached_len=2 * page_size)
    reused_slot.cache_handle = cache_handle(0, 2, 200)
    assert decode_c4_pages(reused_slot) == [4, 5, 6]

    ctx.page_table[0].copy_(ctx.page_table[2])
    moved_prefix = _req(12, 0, 2 * page_size + 1, cached_len=2 * page_size)
    moved_prefix.cache_handle = cache_handle(0, 2, 201)
    assert decode_c4_pages(moved_prefix) == [8, 9, 10]

    grown_active_page = _req(12, 0, 3 * page_size + 1, cached_len=3 * page_size)
    grown_active_page.cache_handle = cache_handle(0, 2, 201)
    assert decode_c4_pages(grown_active_page) == [8, 9, 10, 11]


def test_dsv4_component_loc_ownership_capture_locs_graph_hook_is_guarded(monkeypatch):
    cfg = _tiny_dsv4_config([4, 128])
    ctx = _install_context(
        cfg,
        page_size=128,
        table_bases=[0],
        max_len=256,
        enable_component_loc_ownership=True,
    )
    backend = ctx.attn_backend
    monkeypatch.setattr(
        dsv4_kernel,
        "dsv4_sm80_triton_enabled",
        lambda toggle: toggle == "MINISGL_DSV4_SM80_COMPRESS_STORE",
    )

    backend.init_capture_graph(max_seq_len=256, bs_list=[4])
    backend.bind_capture_graph_inputs(
        input_ids=torch.empty(4, dtype=torch.int32),
        out_loc=torch.full((4,), -1, dtype=torch.int32),
        positions=torch.full((4,), -1, dtype=torch.int32),
    )

    assert backend.capture_compressed_locs_in_graph_component_guarded
    assert not backend.capture_compressed_locs_in_graph


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


def _install_independent_swa_context(page_size: int = 4, max_len: int = 16) -> Context:
    cfg = _tiny_dsv4_config([4, 128, 0])
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[0],
        max_len=max_len,
        enable_component_loc_ownership=True,
        enable_swa_independent_lifecycle=True,
    )
    page_starts = torch.arange(0, max_len, page_size, dtype=torch.int32)
    ctx.kv_cache.on_pages_allocated(page_starts, page_size)
    return ctx


def test_dsv4_independent_swa_page_table_cache_reuses_and_invalidates(monkeypatch):
    cfg = _tiny_dsv4_config([0])
    page_size = 4
    max_len = 16
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[0, max_len],
        max_len=max_len,
        enable_component_loc_ownership=True,
        enable_swa_independent_lifecycle=True,
    )
    ctx.kv_cache.on_pages_allocated(
        torch.arange(0, 2 * max_len, page_size, dtype=torch.int32),
        page_size,
    )
    backend = ctx.attn_backend
    monkeypatch.setenv("MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE", "1")

    def expected_pages(row: int, logical_pages: int) -> list[int]:
        starts = ctx.page_table[
            row,
            torch.arange(logical_pages, dtype=torch.long) * page_size,
        ]
        pages = ctx.kv_cache.swa_pages_from_full_page_starts(starts, page_size)
        assert pages is not None
        return pages.to(torch.int32).tolist()

    def cache_handle(row: int, cached_pages: int, node_uuid: int) -> SimpleNamespace:
        prefix_len = cached_pages * page_size
        handles = ctx.kv_cache.make_swa_page_handles(
            ctx.page_table[row, :prefix_len],
            page_size,
        )
        return SimpleNamespace(
            cached_len=prefix_len,
            node=SimpleNamespace(uuid=node_uuid),
            get_dsv4_swa_pages=lambda handles=handles: handles,
        )

    def decode_cached_row(req: Req) -> list[int]:
        batch = _prepare_decode_batch([req])
        backend.prepare_metadata(batch)
        logical_pages = (req.device_len + page_size - 1) // page_size
        assert backend._swa_page_table_cache is not None
        return backend._swa_page_table_cache[req.table_idx, :logical_pages].tolist()

    req = _req(11, 0, 2 * page_size + 1, cached_len=2 * page_size)
    req.cache_handle = cache_handle(0, 2, 100)
    assert decode_cached_row(req) == expected_pages(0, 3)
    original_signature = backend._swa_page_table_cache_signatures[0]

    assert decode_cached_row(req) == expected_pages(0, 3)
    assert backend._swa_page_table_cache_signatures[0] == original_signature

    ctx.kv_cache._bump_swa_ownership_version()
    assert decode_cached_row(req) == expected_pages(0, 3)
    assert backend._swa_page_table_cache_signatures[0] == original_signature

    ctx.page_table[0].copy_(ctx.page_table[1])
    reused_slot = _req(12, 0, 2 * page_size + 1, cached_len=2 * page_size)
    reused_slot.cache_handle = cache_handle(0, 2, 200)
    assert decode_cached_row(reused_slot) == expected_pages(0, 3)

    grown_active_page = _req(12, 0, 3 * page_size + 1, cached_len=3 * page_size)
    grown_active_page.cache_handle = cache_handle(0, 2, 200)
    assert decode_cached_row(grown_active_page) == expected_pages(0, 4)


def test_dsv4_independent_swa_direct_token_metadata_matches_page_table_and_wins(
    monkeypatch,
):
    cfg = _tiny_dsv4_config([0])
    page_size = 4
    max_len = 16
    base = 64
    ctx = _install_context(
        cfg,
        page_size=page_size,
        table_bases=[base],
        max_len=max_len,
        enable_component_loc_ownership=True,
        enable_swa_independent_lifecycle=True,
    )
    ctx.kv_cache.on_pages_allocated(
        torch.arange(base, base + max_len, page_size, dtype=torch.int32),
        page_size,
    )
    backend = ctx.attn_backend
    req = _req(21, 0, 2 * page_size + 1, cached_len=2 * page_size)
    batch = _prepare_decode_batch([req])

    table = backend._make_swa_page_tables_uncached(
        [req],
        req.device_len,
        timing_base={},
    )
    expected = backend._make_swa_indices_from_page_table(table, batch.positions)

    monkeypatch.setenv("MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA", "1")
    monkeypatch.setenv("MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE", "1")
    backend.prepare_metadata(batch)
    meta = batch.attn_metadata.core_metadata

    assert torch.equal(meta.swa_page_indices, expected)
    assert backend._swa_page_table_cache is None
    assert meta.swa_page_indices[0, 0].item() == 2 * page_size
    assert meta.swa_page_indices[0, 0].item() != ctx.page_table[0, req.device_len - 1].item()


def test_dsv4_independent_swa_metadata_rejects_tombstone_inside_active_length(
    monkeypatch,
):
    ctx = _install_independent_swa_context()
    backend = ctx.attn_backend
    monkeypatch.setenv("MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG", "1")

    with pytest.raises(RuntimeError, match="out of bounds"):
        backend._debug_check_swa_index_bounds(
            torch.tensor([[-1]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            ctx.kv_cache.swa_cache(0).shape[0],
            layer_id=0,
        )


def test_dsv4_independent_swa_metadata_rejects_dummy_page_for_real_row(
    monkeypatch,
):
    ctx = _install_independent_swa_context()
    backend = ctx.attn_backend
    monkeypatch.setenv("MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG", "1")
    dummy_loc = int(ctx.kv_cache._swa_dummy_page) * ctx.page_size

    with pytest.raises(RuntimeError, match="dummy page"):
        backend._debug_check_swa_index_bounds(
            torch.tensor([[dummy_loc]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            ctx.kv_cache.swa_cache(0).shape[0],
            layer_id=0,
        )


def test_dsv4_independent_swa_metadata_rejects_zero_refcount_page(monkeypatch):
    ctx = _install_independent_swa_context()
    backend = ctx.attn_backend
    monkeypatch.setenv("MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG", "1")
    page = int(ctx.kv_cache._full_to_swa_page[0].item())
    ctx.kv_cache._swa_page_refcount[page] = 0

    with pytest.raises(RuntimeError, match="zero-refcount"):
        backend._debug_check_swa_index_bounds(
            torch.tensor([[page * ctx.page_size]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            ctx.kv_cache.swa_cache(0).shape[0],
            layer_id=0,
        )


def test_dsv4_independent_swa_metadata_rejects_free_page_inside_active_length(
    monkeypatch,
):
    ctx = _install_independent_swa_context()
    backend = ctx.attn_backend
    monkeypatch.setenv("MINISGL_DSV4_SWA_INDEX_BOUNDS_DEBUG", "1")
    page = int(ctx.kv_cache._full_to_swa_page[0].item())
    ctx.kv_cache._free_swa_pages = torch.cat(
        [ctx.kv_cache._free_swa_pages, torch.tensor([page], dtype=torch.int32)]
    )

    with pytest.raises(RuntimeError, match="free list"):
        backend._debug_check_swa_index_bounds(
            torch.tensor([[page * ctx.page_size]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            ctx.kv_cache.swa_cache(0).shape[0],
            layer_id=0,
        )


def test_dsv4_cuda_graph_replay_rejects_stale_swa_metadata_version():
    ctx = _install_independent_swa_context(max_len=32)
    req = _req(0, 0, 8, cached_len=7)
    batch = _prepare_decode_batch([req])
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)
    backend.init_capture_graph(max_seq_len=16, bs_list=[1])
    assert backend.capture is not None

    ctx.kv_cache.release_swa_for_full_indices(
        ctx.page_table[0, 4 * ctx.page_size : 5 * ctx.page_size],
        ctx.page_size,
        tombstone=True,
    )

    with pytest.raises(RuntimeError, match="ownership version is stale"):
        backend._copy_metadata_for_replay(backend.capture, batch.attn_metadata, 1)


def test_dsv4_cuda_graph_replay_rebuilds_stale_swa_metadata_version():
    ctx = _install_independent_swa_context(max_len=32)
    req = _req(0, 0, 16, cached_len=15)
    batch = _prepare_decode_batch([req])
    backend = ctx.attn_backend
    backend.prepare_metadata(batch)
    backend.init_capture_graph(max_seq_len=16, bs_list=[1])
    assert backend.capture is not None

    ctx.kv_cache.release_swa_for_full_indices(
        ctx.page_table[0, 4 * ctx.page_size : 5 * ctx.page_size],
        ctx.page_size,
        tombstone=True,
    )
    backend.prepare_for_replay(batch)

    assert batch.attn_metadata.core_metadata.swa_ownership_version == ctx.kv_cache.swa_ownership_version
    assert backend.capture.core_metadata.swa_ownership_version == ctx.kv_cache.swa_ownership_version


def test_dsv4_cuda_graph_replay_keeps_direct_swa_disabled_under_independent(monkeypatch):
    monkeypatch.setenv("MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS", "swa")
    monkeypatch.setenv("MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS", "1")
    ctx = _install_independent_swa_context()
    req = _req(0, 0, 8, cached_len=7)
    batch = _prepare_decode_batch([req])

    ctx.attn_backend.prepare_metadata(batch)

    meta = batch.attn_metadata.core_metadata
    assert not meta.swa_source_elided_for_graph

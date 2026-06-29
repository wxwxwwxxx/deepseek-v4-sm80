from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kvcache import create_kvcache_pool
from minisgl.kvcache.deepseek_v4_pool import DeepSeekV4KVCache
from minisgl.models.config import ModelConfig, RotaryConfig
import minisgl.attention.deepseek_v4 as dsv4_attention
import minisgl.models.deepseek_v4 as dsv4_model


def _has_sm80_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _tiny_dsv4_cache_config(compress_ratios: list[int]) -> ModelConfig:
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


def test_dsv4_kernel_inventory_covers_sglang_main_exports():
    sources = "\n".join(entry.source_function for entry in dsv4_kernel.DSV4_KERNEL_INVENTORY)
    expected_exports = {
        "CompressorDecodePlan",
        "CompressorPrefillPlan",
        "compress_forward",
        "compress_norm_rope_store",
        "fused_norm_rope_inplace",
        "fused_store_cache",
        "fused_rope_inplace",
        "fused_q_norm_rope",
        "fused_q_indexer_rope_first_quant",
        "fused_q_indexer_rope_hadamard_fp4_quant",
        "fused_q_indexer_rope_hadamard_quant",
        "fused_k_norm_rope_flashmla",
        "make_name",
        "linear_bf16_fp32",
        "get_paged_mqa_logits_metadata",
        "triton_create_paged_compress_data",
        "topk_transform_512",
        "topk_transform_512_v2",
        "plan_topk_v2",
        "hash_topk",
        "mega_moe_pre_dispatch",
        "mask_topk_ids",
        "silu_and_mul_clamp",
        "silu_and_mul_masked_post_quant",
        "silu_and_mul_contig_post_quant",
    }

    missing = {name for name in expected_exports if name not in sources}
    assert not missing
    assert {entry.status for entry in dsv4_kernel.DSV4_KERNEL_INVENTORY} <= {
        "native",
        "fallback",
        "unsupported",
        "todo",
    }


def test_dsv4_capability_detection_keeps_sm80_gates_explicit():
    caps = dsv4_kernel.detect_dsv4_kernel_capabilities()

    if caps.cuda_capability is not None:
        assert caps.is_sm80 is (caps.cuda_capability == (8, 0))
        if caps.cuda_capability[0] < 9:
            assert not caps.deep_gemm_usable
    assert set(caps.sgl_kernel_dsv4_ops) == {
        "deepseek_v4_topk_transform_512",
        "dsv4_fused_q_indexer_rope_hadamard_quant",
        "dsv4_fused_q_indexer_rope_hadamard_fp4_quant",
    }


def test_dsv4_unsupported_sm80_paths_fail_clearly():
    with pytest.raises(NotImplementedError) as exc:
        dsv4_kernel.fused_q_indexer_rope_hadamard_fp4_quant()

    message = str(exc.value)
    assert "fused_q_indexer_rope_hadamard_fp4_quant" in message
    assert "sm" in message or "no CUDA" in message


def test_dsv4_fallback_wrappers_preserve_shape_dtype_and_values():
    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(3, 4, dtype=torch.float32)
    y = dsv4_kernel.quantized_linear_ref(x, weight, None, weight_kind="bf16")
    assert torch.allclose(y, F.linear(x, weight))

    rope_x = torch.randn(3, 2, 4, dtype=torch.float32)
    positions = torch.arange(3, dtype=torch.int64)
    rotated = dsv4_kernel.apply_rotary_tail(
        rope_x.clone(),
        positions,
        rotary_dim=2,
        base=10000.0,
    )
    restored = dsv4_kernel.apply_rotary_tail(
        rotated.clone(),
        positions,
        rotary_dim=2,
        base=10000.0,
        inverse=True,
    )
    assert restored.shape == rope_x.shape
    assert restored.dtype is rope_x.dtype
    assert torch.allclose(restored, rope_x, atol=1e-5)

    q = torch.randn(2, 2, 4, dtype=torch.float32)
    cache = torch.randn(4, 4, dtype=torch.float32)
    out = dsv4_kernel.paged_mqa_attention_fallback(
        q,
        cache,
        [torch.tensor([0, 1], dtype=torch.int32), torch.tensor([1, 2], dtype=torch.int32)],
        softmax_scale=0.5,
        attn_sink=torch.zeros(2),
    )
    metadata = dsv4_kernel.get_paged_mqa_logits_metadata_fallback(
        [torch.tensor([0, 1], dtype=torch.int32), torch.tensor([1, 2], dtype=torch.int32)]
    )
    metadata_out = dsv4_kernel.paged_mqa_attention_fallback(
        q,
        cache,
        metadata,
        softmax_scale=0.5,
        attn_sink=torch.zeros(2),
    )
    assert out.shape == q.shape
    assert out.dtype is q.dtype
    assert torch.isfinite(out).all()
    assert metadata.indptr.tolist() == [0, 2, 4]
    assert torch.allclose(metadata_out, out)

    kv = torch.randn(4, 8, dtype=torch.float32)
    norm_weight = torch.linspace(0.5, 1.25, 8, dtype=torch.float32)
    kv_positions = torch.tensor([0, 2, 4, 7], dtype=torch.int64)
    kv_loc = torch.tensor([5, 2, 9, 0], dtype=torch.int32)
    expected_kv = kv.clone()
    normed = expected_kv.float()
    normed = normed * torch.rsqrt(normed.square().mean(-1, keepdim=True) + 1e-6)
    expected_kv.copy_((normed * norm_weight.float()).to(expected_kv.dtype))
    dsv4_kernel.apply_rotary_tail(
        expected_kv,
        kv_positions,
        rotary_dim=4,
        base=10000.0,
    )
    expected_cache = torch.zeros(12, 8, dtype=torch.bfloat16)
    expected_cache[kv_loc.long()] = expected_kv.to(expected_cache.dtype)
    actual_kv = kv.clone()
    actual_cache = torch.zeros_like(expected_cache)
    returned_kv = dsv4_kernel.k_norm_rope_cache_fallback(
        actual_kv,
        kv_positions,
        norm_weight=norm_weight,
        rms_norm_eps=1e-6,
        cache=actual_cache,
        out_loc=kv_loc,
        rotary_dim=4,
        base=10000.0,
    )
    assert returned_kv.data_ptr() == actual_kv.data_ptr()
    assert torch.allclose(actual_kv, expected_kv, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual_cache, expected_cache)

    class FakeCompressedCache:
        def __init__(self) -> None:
            self.compressed = torch.zeros(12, 8, dtype=torch.bfloat16)
            self.indexer = torch.zeros(12, 4, dtype=torch.bfloat16)

        def component_cache(self, layer_id: int) -> torch.Tensor:
            assert layer_id == 0
            return self.compressed

        def indexer_cache(self, layer_id: int) -> torch.Tensor:
            assert layer_id == 0
            return self.indexer

        def store_compressed(self, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
            assert layer_id == 0
            self.compressed[loc.long()] = kv.reshape(-1, 8).to(self.compressed.dtype)

        def store_indexer(self, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
            assert layer_id == 0
            self.indexer[loc.long()] = kv.reshape(-1, 4).to(self.indexer.dtype)

    compressed_kv = torch.randn(4, 8, dtype=torch.float32)
    compressed_weight = torch.linspace(0.75, 1.5, 8, dtype=torch.float32)
    compressed_positions = torch.tensor([3, 7, 11, 15], dtype=torch.int64)
    compressed_loc = torch.tensor([0, 5, 2, -1], dtype=torch.int32)
    expected_compressed_kv = compressed_kv.clone()
    compressed_norm = expected_compressed_kv.float()
    compressed_norm = compressed_norm * torch.rsqrt(
        compressed_norm.square().mean(-1, keepdim=True) + 1e-6
    )
    expected_compressed_kv.copy_(
        (compressed_norm * compressed_weight.float()).to(expected_compressed_kv.dtype)
    )
    dsv4_kernel.apply_rotary_tail(
        expected_compressed_kv,
        compressed_positions,
        rotary_dim=4,
        base=10000.0,
    )
    expected_compressed_cache = FakeCompressedCache()
    valid_compressed = compressed_loc >= 0
    expected_compressed_cache.compressed[compressed_loc[valid_compressed].long()] = (
        expected_compressed_kv[valid_compressed].to(expected_compressed_cache.compressed.dtype)
    )
    actual_compressed_kv = compressed_kv.clone()
    actual_compressed_cache = FakeCompressedCache()
    dsv4_kernel.compress_norm_rope_store_fallback(
        actual_compressed_cache,
        0,
        actual_compressed_kv,
        compressed_loc,
        positions=compressed_positions,
        norm_weight=compressed_weight,
        rms_norm_eps=1e-6,
        rotary_dim=4,
        base=10000.0,
    )
    assert torch.allclose(actual_compressed_kv, expected_compressed_kv, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual_compressed_cache.compressed, expected_compressed_cache.compressed)

    indexer_kv = torch.randn(2, 4, dtype=torch.float32)
    indexer_positions = torch.tensor([3, 7], dtype=torch.int64)
    indexer_loc = torch.tensor([1, 4], dtype=torch.int32)
    expected_indexer_kv = dsv4_kernel.apply_rotary_tail(
        indexer_kv.clone(),
        indexer_positions,
        rotary_dim=2,
        base=10000.0,
    )
    expected_indexer_cache = FakeCompressedCache()
    expected_indexer_cache.indexer[indexer_loc.long()] = expected_indexer_kv.to(
        expected_indexer_cache.indexer.dtype
    )
    actual_indexer_kv = indexer_kv.clone()
    actual_indexer_cache = FakeCompressedCache()
    dsv4_kernel.compress_norm_rope_store_fallback(
        actual_indexer_cache,
        0,
        actual_indexer_kv,
        indexer_loc,
        positions=indexer_positions,
        rotary_dim=2,
        base=10000.0,
        cache_type="indexer",
    )
    assert torch.allclose(actual_indexer_kv, expected_indexer_kv, atol=1e-5, rtol=1e-5)
    assert torch.equal(actual_indexer_cache.indexer, expected_indexer_cache.indexer)

    padded = dsv4_kernel.topk_transform_512_fallback(torch.tensor([[1, 2]], dtype=torch.int32))
    assert padded.shape == (1, 512)
    assert padded[0, :2].tolist() == [1, 2]
    assert padded[0, 2:].eq(-1).all()

    wo_o = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    wo_weight = torch.randn(10, 8, dtype=torch.float32).clamp(-4, 4).to(dsv4_kernel.fp8_dtype())
    wo_scale = torch.rand(
        dsv4_kernel.scale_dim(10),
        dsv4_kernel.scale_dim(8),
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())
    wo_out = dsv4_kernel.wo_a_grouped_projection_fallback(
        wo_o,
        wo_weight,
        wo_scale,
        num_local_groups=2,
        o_lora_rank=5,
    )
    assert wo_out.shape == (2, 10)
    assert wo_out.dtype is wo_o.dtype


def test_compress_norm_rope_store_writes_real_c4_c128_and_indexer_caches():
    pool = create_kvcache_pool(
        _tiny_dsv4_cache_config([4, 128]),
        num_pages=64,
        page_size=4,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    assert isinstance(pool, DeepSeekV4KVCache)

    c4_kv = torch.randn(2, 8, dtype=torch.float32)
    c4_weight = torch.linspace(0.5, 1.25, 8, dtype=torch.float32)
    c4_positions = torch.tensor([3, 7], dtype=torch.int64)
    c4_loc = torch.tensor([0, 3], dtype=torch.int32)
    expected_c4 = dsv4_kernel.norm_rope_inplace_fallback(
        c4_kv.clone(),
        c4_positions,
        weight=c4_weight,
        eps=1e-6,
        rotary_dim=4,
        base=10000.0,
    )
    dsv4_kernel.compress_norm_rope_store_fallback(
        pool,
        0,
        c4_kv,
        c4_loc,
        positions=c4_positions,
        norm_weight=c4_weight,
        rms_norm_eps=1e-6,
        rotary_dim=4,
        base=10000.0,
    )
    assert torch.allclose(c4_kv, expected_c4, atol=1e-5, rtol=1e-5)
    assert torch.equal(pool.c4_cache(0)[c4_loc.long()], expected_c4.to(pool.dtype))

    c128_kv = torch.randn(2, 8, dtype=torch.float32)
    c128_weight = torch.linspace(0.75, 1.5, 8, dtype=torch.float32)
    c128_positions = torch.tensor([127, 255], dtype=torch.int64)
    c128_loc = torch.tensor([0, 1], dtype=torch.int32)
    expected_c128 = dsv4_kernel.norm_rope_inplace_fallback(
        c128_kv.clone(),
        c128_positions,
        weight=c128_weight,
        eps=1e-6,
        rotary_dim=4,
        base=10000.0,
    )
    dsv4_kernel.compress_norm_rope_store_fallback(
        pool,
        1,
        c128_kv,
        c128_loc,
        positions=c128_positions,
        norm_weight=c128_weight,
        rms_norm_eps=1e-6,
        rotary_dim=4,
        base=10000.0,
    )
    assert torch.allclose(c128_kv, expected_c128, atol=1e-5, rtol=1e-5)
    assert torch.equal(pool.c128_cache(1)[c128_loc.long()], expected_c128.to(pool.dtype))

    indexer_kv = torch.randn(2, 4, dtype=torch.float32)
    indexer_weight = torch.linspace(0.8, 1.1, 4, dtype=torch.float32)
    indexer_positions = torch.tensor([3, 7], dtype=torch.int64)
    indexer_loc = torch.tensor([1, 4], dtype=torch.int32)
    expected_indexer = dsv4_kernel.norm_rope_inplace_fallback(
        indexer_kv.clone(),
        indexer_positions,
        weight=indexer_weight,
        eps=1e-6,
        rotary_dim=2,
        base=10000.0,
    )
    dsv4_kernel.compress_norm_rope_store_fallback(
        pool,
        0,
        indexer_kv,
        indexer_loc,
        positions=indexer_positions,
        norm_weight=indexer_weight,
        rms_norm_eps=1e-6,
        rotary_dim=2,
        base=10000.0,
        cache_type="indexer",
    )
    assert torch.allclose(indexer_kv, expected_indexer, atol=1e-5, rtol=1e-5)
    assert torch.equal(pool.indexer_cache(0)[indexer_loc.long()], expected_indexer.to(pool.dtype))


def test_dsv4_moe_route_plan_groups_and_pads_routes():
    indices = torch.tensor(
        [
            [2, 1],
            [0, -1],
            [2, 0],
        ],
        dtype=torch.int64,
    )

    plan = dsv4_kernel.build_moe_route_plan(indices, num_experts=3, block_size_m=2)

    assert plan.route_count == 6
    assert plan.topk == 2
    assert plan.block_size_m == 2
    assert plan.num_tokens_post_padded.item() == 6
    assert plan.expert_ids.tolist() == [0, 1, 2]
    assert plan.sorted_route_ids.tolist() == [2, 5, 1, 6, 0, 4]

    route_experts = torch.repeat_interleave(plan.expert_ids, plan.block_size_m)
    valid = plan.sorted_route_ids < plan.route_count
    pairs = list(
        zip(
            plan.sorted_route_ids[valid].tolist(),
            route_experts[valid].tolist(),
        )
    )
    assert pairs == [(2, 0), (5, 0), (1, 1), (0, 2), (4, 2)]


def test_dsv4_model_and_attention_do_not_import_optional_kernels_directly():
    model_source = inspect.getsource(dsv4_model)
    attention_source = inspect.getsource(dsv4_attention)

    assert "def _quantized_linear_ref" not in model_source
    assert "def _apply_rotary_tail" not in model_source
    for source in (model_source, attention_source):
        assert "import sgl_kernel" not in source
        assert "import flashinfer" not in source
        assert "import deep_gemm" not in source


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_dsv4_sm80_opt_in_kernels_match_fallbacks(monkeypatch):
    for name in (
        "MINISGL_DSV4_SM80_SWIGLU",
        "MINISGL_DSV4_SM80_ROPE",
        "MINISGL_DSV4_SM80_Q_NORM_ROPE",
        "MINISGL_DSV4_SM80_KV_BF16",
        "MINISGL_DSV4_SM80_STORE_CACHE",
        "MINISGL_DSV4_SM80_COMPRESS_STORE",
        "MINISGL_DSV4_SM80_COMPRESS",
        "MINISGL_DSV4_SM80_TOPK",
        "MINISGL_DSV4_SM80_FP8_GEMM",
        "MINISGL_DSV4_SM80_FP4_GEMM",
        "MINISGL_DSV4_SM80_MOE_ROUTE",
        "MINISGL_DSV4_SM80_WO_A_BF16",
        "MINISGL_DSV4_SM80_PAGED_MQA_BF16",
    ):
        monkeypatch.delenv(name, raising=False)

    device = torch.device("cuda")
    torch.manual_seed(5)

    gate = torch.randn(17, 513, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    weights = torch.rand(17, 1, device=device, dtype=torch.float32)
    expected_swiglu = dsv4_kernel.silu_and_mul_clamp_fallback(
        gate,
        up,
        swiglu_limit=2.0,
        weights=weights,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_SWIGLU", "1")
    actual_swiglu = dsv4_kernel.silu_and_mul_clamp_fallback(
        gate,
        up,
        swiglu_limit=2.0,
        weights=weights,
    )
    torch.cuda.synchronize()
    assert actual_swiglu.dtype is torch.float32
    assert torch.allclose(actual_swiglu, expected_swiglu, atol=2e-2, rtol=2e-2)
    monkeypatch.delenv("MINISGL_DSV4_SM80_SWIGLU", raising=False)

    positions = torch.arange(7, device=device, dtype=torch.int64)
    rope_x = torch.randn(7, 3, 12, device=device, dtype=torch.float32)
    expected_rope = dsv4_kernel.apply_rotary_tail(
        rope_x.clone(),
        positions,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_ROPE", "1")
    actual_rope = dsv4_kernel.apply_rotary_tail(
        rope_x.clone(),
        positions,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_rope, expected_rope, atol=1e-4, rtol=1e-4)
    monkeypatch.delenv("MINISGL_DSV4_SM80_ROPE", raising=False)

    q = torch.randn(7, 2, 16, device=device, dtype=torch.float32)
    expected_q = dsv4_kernel.q_norm_rope_fallback(
        q.clone(),
        positions,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_Q_NORM_ROPE", "1")
    actual_q = dsv4_kernel.q_norm_rope_fallback(
        q.clone(),
        positions,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_q, expected_q, atol=1e-4, rtol=1e-4)
    monkeypatch.delenv("MINISGL_DSV4_SM80_Q_NORM_ROPE", raising=False)

    high_positions = torch.tensor([0, 127, 255, 511], device=device, dtype=torch.int64)
    high_q = torch.randn(4, 1, 128, device=device, dtype=torch.bfloat16)
    expected_high_q = dsv4_kernel.q_norm_rope_fallback(
        high_q.clone(),
        high_positions,
        rms_norm_eps=1e-6,
        rotary_dim=64,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_Q_NORM_ROPE", "1")
    actual_high_q = dsv4_kernel.q_norm_rope_fallback(
        high_q.clone(),
        high_positions,
        rms_norm_eps=1e-6,
        rotary_dim=64,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_high_q, expected_high_q, atol=2e-2, rtol=2e-2)
    monkeypatch.delenv("MINISGL_DSV4_SM80_Q_NORM_ROPE", raising=False)

    kv_for_fused = torch.randn(5, 16, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(16, device=device, dtype=torch.bfloat16)
    k_positions = torch.tensor([0, 127, 255, 511, 777], device=device, dtype=torch.int64)
    k_loc = torch.tensor([3, 7, 11, 13, 19], device=device, dtype=torch.int32)
    expected_k_cache = torch.zeros(32, 16, device=device, dtype=torch.bfloat16)
    expected_k = dsv4_kernel.k_norm_rope_cache_fallback(
        kv_for_fused.clone(),
        k_positions,
        norm_weight=k_weight,
        rms_norm_eps=1e-6,
        cache=expected_k_cache,
        out_loc=k_loc,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    actual_k_cache = torch.zeros_like(expected_k_cache)
    monkeypatch.setenv("MINISGL_DSV4_SM80_KV_BF16", "1")
    actual_k = dsv4_kernel.k_norm_rope_cache_fallback(
        kv_for_fused.clone(),
        k_positions,
        norm_weight=k_weight,
        rms_norm_eps=1e-6,
        cache=actual_k_cache,
        out_loc=k_loc,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_k, expected_k, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_k_cache, expected_k_cache, atol=2e-2, rtol=2e-2)
    monkeypatch.delenv("MINISGL_DSV4_SM80_KV_BF16", raising=False)

    class FakeCache:
        def __init__(self) -> None:
            self.cache = torch.zeros(32, 16, device=device, dtype=torch.bfloat16)

        def swa_cache(self, layer_id: int) -> torch.Tensor:
            assert layer_id == 0
            return self.cache

        def store_swa(self, layer_id: int, kv: torch.Tensor, out_loc: torch.Tensor) -> None:
            assert layer_id == 0
            self.cache[out_loc.long()] = kv.reshape(-1, 16).to(self.cache.dtype)

    kv = torch.randn(5, 16, device=device, dtype=torch.bfloat16)
    loc = torch.tensor([3, 7, 11, 13, 19], device=device, dtype=torch.int32)
    expected_cache = FakeCache()
    dsv4_kernel.store_swa_fallback(expected_cache, 0, kv, loc)
    actual_cache = FakeCache()
    monkeypatch.setenv("MINISGL_DSV4_SM80_STORE_CACHE", "1")
    dsv4_kernel.store_swa_fallback(actual_cache, 0, kv, loc)
    torch.cuda.synchronize()
    assert torch.equal(actual_cache.cache, expected_cache.cache)
    monkeypatch.delenv("MINISGL_DSV4_SM80_STORE_CACHE", raising=False)

    class FakeCompressedCudaCache:
        def __init__(self) -> None:
            self.cache = torch.zeros(32, 16, device=device, dtype=torch.bfloat16)

        def component_cache(self, layer_id: int) -> torch.Tensor:
            assert layer_id == 0
            return self.cache

        def store_compressed(self, layer_id: int, kv: torch.Tensor, out_loc: torch.Tensor) -> None:
            assert layer_id == 0
            valid = out_loc >= 0
            self.cache[out_loc[valid].long()] = kv.reshape(-1, 16)[valid].to(self.cache.dtype)

    compressed_for_fused = torch.randn(5, 16, device=device, dtype=torch.bfloat16)
    compressed_weight = torch.randn(16, device=device, dtype=torch.bfloat16)
    compressed_positions = torch.tensor([3, 7, 11, 15, 19], device=device, dtype=torch.int64)
    compressed_loc = torch.tensor([4, 8, 12, -1, 20], device=device, dtype=torch.int32)
    expected_compressed_cache = FakeCompressedCudaCache()
    expected_compressed_kv = compressed_for_fused.clone()
    dsv4_kernel.compress_norm_rope_store_fallback(
        expected_compressed_cache,
        0,
        expected_compressed_kv,
        compressed_loc,
        positions=compressed_positions,
        norm_weight=compressed_weight,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    actual_compressed_cache = FakeCompressedCudaCache()
    actual_compressed_kv = compressed_for_fused.clone()
    monkeypatch.setenv("MINISGL_DSV4_SM80_COMPRESS_STORE", "1")
    dsv4_kernel.compress_norm_rope_store_fallback(
        actual_compressed_cache,
        0,
        actual_compressed_kv,
        compressed_loc,
        positions=compressed_positions,
        norm_weight=compressed_weight,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_compressed_kv, expected_compressed_kv, atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        actual_compressed_cache.cache,
        expected_compressed_cache.cache,
        atol=2e-2,
        rtol=2e-2,
    )
    monkeypatch.delenv("MINISGL_DSV4_SM80_COMPRESS_STORE", raising=False)

    indices = torch.tensor([[1, 2, 3], [5, 8, 13]], device=device, dtype=torch.int32)
    expected_topk = dsv4_kernel.topk_transform_512_fallback(indices, width=512)
    monkeypatch.setenv("MINISGL_DSV4_SM80_TOPK", "1")
    actual_topk = dsv4_kernel.topk_transform_512_fallback(indices, width=512)
    torch.cuda.synchronize()
    assert torch.equal(actual_topk, expected_topk)
    monkeypatch.delenv("MINISGL_DSV4_SM80_TOPK", raising=False)

    q_attn = torch.randn(4, 3, 64, device=device, dtype=torch.bfloat16)
    cache_attn = torch.randn(192, 64, device=device, dtype=torch.bfloat16)
    attn_contexts = [
        torch.empty(0, device=device, dtype=torch.int32),
        torch.tensor([3], device=device, dtype=torch.int32),
        torch.arange(8, 104, 3, device=device, dtype=torch.int32),
        torch.tensor([2, 4, 4, 6, 8, 16, 32, 64], device=device, dtype=torch.int32),
    ]
    attn_sink = torch.randn(3, device=device, dtype=torch.float32)
    attn_metadata = dsv4_kernel.get_paged_mqa_logits_metadata_fallback(
        attn_contexts,
        device=device,
    )
    assert attn_metadata.indptr.cpu().tolist() == [0, 0, 1, 33, 41]
    expected_attn = dsv4_kernel.paged_mqa_attention_fallback(
        q_attn,
        cache_attn,
        attn_contexts,
        softmax_scale=0.125,
        attn_sink=attn_sink,
    )
    expected_attn_no_sink = dsv4_kernel.paged_mqa_attention_fallback(
        q_attn,
        cache_attn,
        attn_contexts,
        softmax_scale=0.125,
        attn_sink=None,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_PAGED_MQA_BF16", "1")
    actual_attn = dsv4_kernel.paged_mqa_attention_fallback(
        q_attn,
        cache_attn,
        attn_metadata,
        softmax_scale=0.125,
        attn_sink=attn_sink,
    )
    actual_attn_no_sink = dsv4_kernel.paged_mqa_attention_fallback(
        q_attn,
        cache_attn,
        attn_metadata,
        softmax_scale=0.125,
        attn_sink=None,
    )
    torch.cuda.synchronize()
    assert actual_attn.dtype is q_attn.dtype
    assert torch.allclose(actual_attn, expected_attn, atol=3e-2, rtol=3e-2)
    assert torch.allclose(actual_attn_no_sink, expected_attn_no_sink, atol=3e-2, rtol=3e-2)
    assert torch.equal(actual_attn[0], torch.zeros_like(actual_attn[0]))
    monkeypatch.delenv("MINISGL_DSV4_SM80_PAGED_MQA_BF16", raising=False)

    class FakeWkvGate:
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            values = torch.arange(
                x.shape[0] * 16,
                device=x.device,
                dtype=torch.float32,
            ).view(x.shape[0], 16)
            return values.to(x.dtype) / 16

    class IdentityNorm:
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    x = torch.randn(12, 4, device=device, dtype=torch.bfloat16)
    ape = torch.randn(4, 8, device=device, dtype=torch.float32)
    compress_positions = torch.arange(12, device=device, dtype=torch.int64)
    expected_compress = dsv4_kernel.compress_forward_fallback(
        x,
        compress_positions,
        ratio=4,
        head_dim=4,
        overlap=True,
        ape=ape,
        wkv_gate=FakeWkvGate(),
        norm=IdentityNorm(),
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_COMPRESS", "1")
    actual_compress = dsv4_kernel.compress_forward_fallback(
        x,
        compress_positions,
        ratio=4,
        head_dim=4,
        overlap=True,
        ape=ape,
        wkv_gate=FakeWkvGate(),
        norm=IdentityNorm(),
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_compress, expected_compress, atol=1e-4, rtol=1e-4)

    x_linear = torch.randn(5, 128, device=device, dtype=torch.bfloat16)
    fp8_weight = torch.randn(96, 128, device=device, dtype=torch.float32).clamp(-4, 4).to(
        dsv4_kernel.fp8_dtype()
    )
    fp8_scale = torch.rand(
        dsv4_kernel.scale_dim(96),
        dsv4_kernel.scale_dim(128),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())
    expected_fp8 = dsv4_kernel.quantized_linear_ref(
        x_linear,
        fp8_weight,
        fp8_scale,
        weight_kind="fp8",
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_FP8_GEMM", "1")
    actual_fp8 = dsv4_kernel.quantized_linear_ref(
        x_linear,
        fp8_weight,
        fp8_scale,
        weight_kind="fp8",
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_fp8, expected_fp8, atol=3e-2, rtol=3e-2)
    monkeypatch.delenv("MINISGL_DSV4_SM80_FP8_GEMM", raising=False)

    wo_o = torch.randn(5, 2, 64, device=device, dtype=torch.bfloat16)
    wo_rank = 48
    wo_weight = torch.randn(
        2 * wo_rank,
        wo_o.shape[-1],
        device=device,
        dtype=torch.float32,
    ).clamp(-4, 4).to(dsv4_kernel.fp8_dtype())
    wo_scale = torch.rand(
        dsv4_kernel.scale_dim(2 * wo_rank),
        dsv4_kernel.scale_dim(wo_o.shape[-1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())
    expected_wo_a = dsv4_kernel.wo_a_grouped_projection_fallback(
        wo_o,
        wo_weight,
        wo_scale,
        num_local_groups=2,
        o_lora_rank=wo_rank,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_WO_A_BF16", "1")
    actual_wo_a = dsv4_kernel.wo_a_grouped_projection_fallback(
        wo_o,
        wo_weight,
        wo_scale,
        num_local_groups=2,
        o_lora_rank=wo_rank,
    )
    torch.cuda.synchronize()
    assert actual_wo_a.dtype is wo_o.dtype
    assert torch.allclose(actual_wo_a, expected_wo_a, atol=4e-2, rtol=4e-2)
    monkeypatch.delenv("MINISGL_DSV4_SM80_WO_A_BF16", raising=False)

    fp4_weight = torch.randint(-128, 127, (96, 64), device=device, dtype=torch.int8)
    fp4_scale = torch.rand(96, 4, device=device, dtype=torch.float32).to(
        dsv4_kernel.e8m0_dtype()
    )
    expected_fp4 = dsv4_kernel.quantized_linear_ref(
        x_linear,
        fp4_weight,
        fp4_scale,
        weight_kind="fp4",
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_FP4_GEMM", "1")
    actual_fp4 = dsv4_kernel.quantized_linear_ref(
        x_linear,
        fp4_weight,
        fp4_scale,
        weight_kind="fp4",
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_fp4, expected_fp4, atol=3e-2, rtol=3e-2)

    num_tokens = 5
    topk = 2
    num_experts = 4
    hidden = 64
    intermediate = 32
    moe_x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    moe_weights = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
    moe_indices = torch.tensor(
        [[0, 2], [1, 3], [2, 0], [3, 1], [0, 1]],
        device=device,
        dtype=torch.int64,
    )
    w13_weight = torch.randint(
        -128,
        127,
        (num_experts, 2, intermediate, hidden // 2),
        device=device,
        dtype=torch.int8,
    )
    w13_scale = torch.rand(
        num_experts,
        2,
        intermediate,
        dsv4_kernel.scale_dim(hidden, block_size=32),
        device=device,
        dtype=torch.float32,
    )
    w2_weight = torch.randint(
        -128,
        127,
        (num_experts, hidden, intermediate // 2),
        device=device,
        dtype=torch.int8,
    )
    w2_scale = torch.rand(
        num_experts,
        hidden,
        dsv4_kernel.scale_dim(intermediate, block_size=32),
        device=device,
        dtype=torch.float32,
    )
    expected_moe = torch.zeros_like(moe_x, dtype=torch.float32)
    for expert_idx in range(num_experts):
        token_idx, top_idx = torch.where(moe_indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_x = moe_x[token_idx]
        w1 = dsv4_kernel.quantized_linear_ref(
            expert_x,
            w13_weight[expert_idx, 0],
            w13_scale[expert_idx, 0],
            weight_kind="fp4",
        ).float()
        w3 = dsv4_kernel.quantized_linear_ref(
            expert_x,
            w13_weight[expert_idx, 1],
            w13_scale[expert_idx, 1],
            weight_kind="fp4",
        ).float()
        expert_hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            w1,
            w3,
            swiglu_limit=2.5,
            weights=moe_weights[token_idx, top_idx, None],
        )
        expected_moe[token_idx] += dsv4_kernel.quantized_linear_ref(
            expert_hidden.to(moe_x.dtype),
            w2_weight[expert_idx],
            w2_scale[expert_idx],
            weight_kind="fp4",
        ).float()
    expected_moe = expected_moe.to(moe_x.dtype)

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_ROUTE", "1")
    actual_moe = dsv4_kernel.moe_route_dispatch_bf16_grouped(
        moe_x,
        moe_weights,
        moe_indices,
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        swiglu_limit=2.5,
    )
    torch.cuda.synchronize()
    assert actual_moe is not None
    assert actual_moe.dtype is moe_x.dtype
    assert torch.allclose(actual_moe, expected_moe, atol=8e-2, rtol=8e-2)

from __future__ import annotations

import inspect
import os

import minisgl.attention.deepseek_v4 as dsv4_attention
import minisgl.models.deepseek_v4 as dsv4_model
import pytest
import torch
import torch.nn.functional as F
from minisgl.distributed import set_tp_info
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kvcache import create_kvcache_pool
from minisgl.kvcache.deepseek_v4_pool import DeepSeekV4KVCache
from minisgl.models.config import ModelConfig, RotaryConfig


def _has_sm80_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0)


def _clear_dsv4_sm80_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("MINISGL_DSV4_SM80_"):
            monkeypatch.delenv(name, raising=False)
    for name in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES:
        monkeypatch.delenv(name, raising=False)


def _assert_full_topk_transform(
    out: dsv4_kernel.DSV4TopKTransformOutput,
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int,
    ratio: int,
) -> None:
    assert out.raw_indices.shape == (scores.shape[0], width)
    assert out.page_indices.shape == out.raw_indices.shape
    assert out.full_indices.shape == out.raw_indices.shape
    cpu_scores = scores.detach().cpu()
    cpu_lens = seq_lens.detach().cpu().tolist()
    cpu_pages = page_table.detach().cpu()
    raw = out.raw_indices.detach().cpu()
    pages = out.page_indices.detach().cpu()
    full = out.full_indices.detach().cpu()
    for row, seq_len in enumerate(cpu_lens):
        valid_raw = [int(x) for x in raw[row].tolist() if x >= 0]
        if seq_len <= width:
            assert raw[row, :seq_len].tolist() == list(range(seq_len))
            assert raw[row, seq_len:].eq(-1).all()
        else:
            expected = torch.topk(cpu_scores[row, :seq_len], width, sorted=False).indices.tolist()
            assert sorted(valid_raw) == sorted(int(x) for x in expected)
        for raw_idx, page_idx, full_idx in zip(
            raw[row].tolist(),
            pages[row].tolist(),
            full[row].tolist(),
        ):
            if raw_idx < 0:
                assert page_idx == -1
                assert full_idx == -1
                continue
            physical_page = int(cpu_pages[row, raw_idx // page_size].item())
            expected_page = physical_page * page_size + raw_idx % page_size
            assert page_idx == expected_page
            assert full_idx == expected_page * ratio + (ratio - 1)


def _manual_indexer_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    rows = q.shape[0]
    max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
    logits = torch.full((rows, max_seq_len), float("-inf"), dtype=torch.float32, device=q.device)
    for row in range(rows):
        for raw_idx in range(int(seq_lens[row].item())):
            page = int(page_table[row, raw_idx // page_size].item())
            if page < 0:
                continue
            cache_row = page * page_size + raw_idx % page_size
            scores = torch.einsum("hd,d->h", q[row].float(), cache[cache_row].float())
            logits[row, raw_idx] = (torch.relu(scores) * weights[row].float()).sum()
    return logits


def test_indexer_fp8_quantized_logits_and_topk_match_reference():
    if getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is required for FP8 indexer reference checks")

    torch.manual_seed(7)
    rows = 3
    heads = 2
    dim = 8
    page_size = 4
    q = torch.rand(rows, heads, dim, dtype=torch.float32).to(torch.bfloat16)
    weights = torch.rand(rows, heads, dtype=torch.float32)
    positions = torch.arange(rows, dtype=torch.int64)
    query = dsv4_kernel.indexer_q_rope_fp8_fallback(
        q,
        weights,
        positions,
        rotary_dim=0,
        base=10000.0,
        softmax_scale=dim**-0.5,
        head_scale=heads**-0.5,
    )
    assert query.q_values.dtype is torch.uint8
    assert query.weights.dtype is torch.float32

    cache = torch.rand(12, dim, dtype=torch.float32).to(torch.bfloat16)
    cache_values, cache_scales = dsv4_kernel.quantize_indexer_fp8_cache_ref(cache)
    cache_dequant = dsv4_kernel.dequantize_indexer_fp8_cache_ref(
        cache_values,
        cache_scales,
        out_dtype=torch.float32,
    )
    q_dequant = query.q_values.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
    seq_lens = torch.tensor([3, 8, 10], dtype=torch.int32)
    page_table = torch.tensor(
        [[0, 1, 2], [0, 2, 1], [2, 1, 0]],
        dtype=torch.int32,
    )

    expected = _manual_indexer_logits(
        q_dequant,
        cache_dequant,
        query.weights,
        seq_lens,
        page_table,
        page_size=page_size,
    )
    actual = dsv4_kernel.indexer_fp8_logits_fallback(
        query.q_values,
        cache_values,
        cache_scales,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=query.weights,
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    selected = dsv4_kernel.indexer_select_fp8_fallback(
        query.q_values,
        query.weights,
        cache_values,
        cache_scales,
        seq_lens,
        page_table,
        page_size=page_size,
        width=4,
        ratio=4,
    )
    assert "fp8" in selected.backend
    _assert_full_topk_transform(
        selected.topk,
        actual,
        seq_lens,
        page_table,
        page_size=page_size,
        width=4,
        ratio=4,
    )


def test_indexer_fp8_paged_logits_and_topk_match_reference():
    if getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is required for FP8 indexer reference checks")

    torch.manual_seed(752)
    rows = 3
    heads = 2
    dim = 8
    page_size = 4
    q = torch.rand(rows, heads, dim, dtype=torch.float32).to(torch.bfloat16)
    weights = torch.rand(rows, heads, dtype=torch.float32)
    positions = torch.arange(rows, dtype=torch.int64)
    query = dsv4_kernel.indexer_q_rope_fp8_fallback(
        q,
        weights,
        positions,
        rotary_dim=0,
        base=10000.0,
        softmax_scale=dim**-0.5,
        head_scale=heads**-0.5,
    )
    cache = torch.rand(12, dim, dtype=torch.float32).to(torch.bfloat16)
    packed_cache = dsv4_kernel.quantize_indexer_fp8_paged_cache_ref(
        cache,
        page_size=page_size,
    )
    cache_dequant = dsv4_kernel.dequantize_indexer_fp8_paged_cache_ref(
        packed_cache,
        page_size=page_size,
        dim=dim,
        slots=cache.shape[0],
        out_dtype=torch.float32,
    )
    q_dequant = query.q_values.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
    seq_lens = torch.tensor([3, 8, 10], dtype=torch.int32)
    page_table = torch.tensor(
        [[0, 1, 2], [0, 2, 1], [2, 1, 0]],
        dtype=torch.int32,
    )

    expected = _manual_indexer_logits(
        q_dequant,
        cache_dequant,
        query.weights,
        seq_lens,
        page_table,
        page_size=page_size,
    )
    actual = dsv4_kernel.indexer_fp8_paged_logits_fallback(
        query.q_values,
        packed_cache,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=query.weights,
    )
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    selected = dsv4_kernel.indexer_select_fp8_paged_fallback(
        query.q_values,
        query.weights,
        packed_cache,
        seq_lens,
        page_table,
        page_size=page_size,
        width=4,
        ratio=4,
    )
    assert "fp8_paged" in selected.backend
    assert torch.allclose(selected.logits, actual, atol=1e-5, rtol=1e-5)
    _assert_full_topk_transform(
        selected.topk,
        selected.logits,
        seq_lens,
        page_table,
        page_size=page_size,
        width=4,
        ratio=4,
    )


def _manual_two_source_sparse_attention(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    compressed_cache: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    compressed_lengths: torch.Tensor | None,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    out = torch.empty_like(q)
    sink = (
        attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)
        if attn_sink is not None
        else None
    )

    def _source_candidates(
        cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        row: int,
    ) -> torch.Tensor | None:
        row_len = max(0, min(int(lengths[row].item()), indices.shape[-1]))
        if row_len == 0:
            return None
        row_indices = indices[row, :row_len]
        row_indices = row_indices[row_indices >= 0]
        if row_indices.numel() == 0:
            return None
        return cache[row_indices.to(torch.long)].float()

    for row in range(q.shape[0]):
        sources = []
        if (
            compressed_cache is not None
            and compressed_indices is not None
            and compressed_lengths is not None
        ):
            compressed = _source_candidates(
                compressed_cache,
                compressed_indices,
                compressed_lengths,
                row,
            )
            if compressed is not None:
                sources.append(compressed)
        swa = _source_candidates(swa_cache, swa_indices, swa_lengths, row)
        if swa is not None:
            sources.append(swa)
        if not sources:
            out[row].zero_()
            continue

        candidates = torch.cat(sources, dim=0)
        scores = torch.einsum("hd,td->ht", q[row].float(), candidates) * softmax_scale
        if sink is None:
            attn = torch.softmax(scores, dim=-1)
        else:
            max_score = torch.maximum(scores.max(dim=-1).values, sink)
            exp_scores = torch.exp(scores - max_score[:, None])
            denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
            attn = exp_scores / denom[:, None]
        out[row] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
    return out


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


def test_dsv4_sm80_v0_bf16_bundle_env_policy(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)

    assert dsv4_kernel.DSV4_SM80_V0_BF16_TOGGLE in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert (
        dsv4_kernel.DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV
        in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    )
    assert dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert (
        dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
        in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    )
    assert (
        dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE
        in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    )
    assert dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES
    assert (
        dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE
        in dsv4_kernel.DSV4_SM80_EXPERIMENTAL_TOGGLES
    )
    assert (
        dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
        in dsv4_kernel.DSV4_SM80_EXPERIMENTAL_TOGGLES
    )
    assert (
        dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE
        in dsv4_kernel.DSV4_SM80_EXPERIMENTAL_TOGGLES
    )
    assert (
        dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE in dsv4_kernel.DSV4_SM80_EXPERIMENTAL_TOGGLES
    )
    assert (
        dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE
        in dsv4_kernel.DSV4_SM80_EXPERIMENTAL_TOGGLES
    )
    assert (
        dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
        not in dsv4_kernel.DSV4_SM80_BF16_PROJECTION_CACHE_WHITELIST
    )
    assert (
        dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
        in dsv4_kernel.DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST
    )
    assert (
        dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE
        not in dsv4_kernel.DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST
    )
    assert (
        dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE
        not in dsv4_kernel.DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST
    )
    assert (
        dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE
        not in dsv4_kernel.DSV4_SM80_A100_VICTORY_BUNDLE_WHITELIST
    )
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_V0_BF16_TOGGLE)
    assert not any(
        dsv4_kernel.dsv4_env_flag(name) for name in dsv4_kernel.DSV4_SM80_V0_BF16_WHITELIST
    )

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_A100_VICTORY_BUNDLE_TOGGLE, "1")
    assert dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE)
    monkeypatch.setenv(
        dsv4_kernel.DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV,
        "q_wqb,shared_expert,MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE",
    )
    assert dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE in (
        dsv4_kernel.dsv4_env_disabled_toggles()
    )
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag(
        dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
    )
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE)
    assert dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_B_BF16_WEIGHT_CACHE_TOGGLE)
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE, "1")
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE)
    monkeypatch.setenv(
        dsv4_kernel.DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV,
        "projection_bf16_caches",
    )
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_PROJECTION_CACHE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_B_BF16_WEIGHT_CACHE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_WO_A_BF16_BMM_CACHE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE_TOGGLE)
    assert dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE)
    monkeypatch.delenv(dsv4_kernel.DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES_ENV, raising=False)
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE)
    monkeypatch.delenv(dsv4_kernel.DSV4_SM80_A100_VICTORY_BUNDLE_TOGGLE, raising=False)
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_BF16_PROJECTION_CACHE_TOGGLE, "1")
    assert not dsv4_kernel.dsv4_env_flag(
        dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE
    )
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE, "1")
    assert dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE)
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE, "1")
    assert dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE)
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE, "1")
    assert dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE_TOGGLE)
    _clear_dsv4_sm80_env(monkeypatch)

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_V0_BF16_TOGGLE, "1")
    enabled = {
        name for name in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES if dsv4_kernel.dsv4_env_flag(name)
    }
    assert enabled == {
        dsv4_kernel.DSV4_SM80_V0_BF16_TOGGLE,
        *dsv4_kernel.DSV4_SM80_V0_BF16_WHITELIST,
    }
    assert not any(
        dsv4_kernel.dsv4_env_flag(name) for name in dsv4_kernel.DSV4_SM80_EXPERIMENTAL_TOGGLES
    )

    _clear_dsv4_sm80_env(monkeypatch)
    monkeypatch.setenv("MINISGL_DSV4_SM80_SWIGLU", "yes")
    assert dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_SWIGLU")
    assert not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_ROPE")

    monkeypatch.setenv("MINISGL_DSV4_SM80_STORE_CACHE", "true")
    assert dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_STORE_CACHE")
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE, "1")
    assert dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE)


def test_dsv4_sm80_v1_moe_bundle_env_policy(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)

    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_MOE_ROUTE")

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE, "1")
    enabled = {
        name for name in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES if dsv4_kernel.dsv4_env_flag(name)
    }
    assert enabled == {
        dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE,
        *dsv4_kernel.DSV4_SM80_V1_MOE_WHITELIST,
    }
    assert "MINISGL_DSV4_SM80_MOE_ROUTE" in enabled
    assert dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE not in enabled


def test_dsv4_sm80_moe_v2_bundle_env_policy(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)

    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_MOE_ROUTE")

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE, "1")
    enabled = {
        name for name in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES if dsv4_kernel.dsv4_env_flag(name)
    }
    assert enabled == {
        dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE,
        *dsv4_kernel.DSV4_SM80_MOE_V2_WHITELIST,
    }
    assert dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE not in enabled
    assert "MINISGL_DSV4_SM80_MOE_ROUTE" in enabled
    assert dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE not in enabled


def test_dsv4_sm80_moe_vllm_runner_bundle_env_policy(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)

    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE)
    assert not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_MOE_ROUTE")

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE, "1")
    enabled = {
        name for name in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES if dsv4_kernel.dsv4_env_flag(name)
    }
    assert enabled == {
        dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE,
        *dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_WHITELIST,
    }
    assert dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE not in enabled
    assert "MINISGL_DSV4_SM80_MOE_ROUTE" in enabled
    assert dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE not in enabled


def test_dsv4_sm80_moe_expert_backend_selector_blocks_marlin(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)

    assert (
        dsv4_kernel.dsv4_moe_expert_backend()
        == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4
    )
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV)

    monkeypatch.setenv(
        dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV,
        dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_MXFP4_W4A16,
    )
    assert (
        dsv4_kernel.dsv4_moe_expert_backend()
        == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_MXFP4_W4A16
    )
    assert not dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV)
    with pytest.raises(NotImplementedError, match="Marlin MXFP4 W4A16"):
        dsv4_kernel.require_supported_moe_expert_backend()

    monkeypatch.setenv(
        dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV,
        dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_VLLM_MARLIN_BRIDGE,
    )
    assert (
        dsv4_kernel.dsv4_moe_expert_backend()
        == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_VLLM_MARLIN_BRIDGE
    )
    with pytest.raises(NotImplementedError, match="vLLM Marlin bridge"):
        dsv4_kernel.require_supported_moe_expert_backend()

    monkeypatch.setenv(
        dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV,
        dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16,
    )
    assert (
        dsv4_kernel.dsv4_moe_expert_backend()
        == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16
    )
    assert (
        dsv4_kernel.require_supported_moe_expert_backend()
        == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16
    )

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV, "not_a_backend")
    with pytest.raises(ValueError, match="Unsupported MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND"):
        dsv4_kernel.dsv4_moe_expert_backend()


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


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_copy_decode_metadata_for_replay_matches_legacy_copy(monkeypatch):
    device = torch.device("cuda")
    rows = 3
    one_d_names = (
        "raw_out_loc",
        "seq_lens",
        "req_seq_lens",
        "extend_lens",
        "positions",
        "req_table_indices",
        "swa_topk_lengths",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c4_sparse_topk_lengths",
        "c128_topk_lengths_clamp1",
    )
    two_d_specs = {
        "page_table": (2, 5, 0),
        "swa_page_indices": (4, 6, -1),
        "c4_sparse_raw_indices": (3, 8, -1),
        "c4_sparse_page_indices": (5, 8, -1),
        "c4_sparse_full_indices": (4, 8, -1),
        "c128_raw_indices": (2, 4, -1),
        "c128_page_indices": (3, 4, -1),
        "c128_full_indices": (1, 4, -1),
    }

    def make_args(graph_inputs_bound: bool) -> tuple[dict[str, torch.Tensor | int | bool], dict]:
        args: dict[str, torch.Tensor | int | bool] = {}
        expected: dict[str, torch.Tensor] = {}
        counter = 1
        for name in one_d_names:
            src = torch.arange(counter, counter + rows, device=device, dtype=torch.int32)
            dst = torch.full((rows,), -9999, device=device, dtype=torch.int32)
            args[f"src_{name}"] = src
            args[f"dst_{name}"] = dst
            if graph_inputs_bound and name in {"raw_out_loc", "positions"}:
                expected[name] = dst.clone()
            else:
                expected[name] = src.clone()
            counter += 100
        src_cu = torch.arange(7000, 7000 + rows + 1, device=device, dtype=torch.int32)
        dst_cu = torch.full((rows + 1,), -9999, device=device, dtype=torch.int32)
        args["src_cu_seqlens_q"] = src_cu
        args["dst_cu_seqlens_q"] = dst_cu
        expected["cu_seqlens_q"] = src_cu.clone()

        for name, (src_width, dst_width, fill) in two_d_specs.items():
            src = (
                torch.arange(
                    counter,
                    counter + rows * src_width,
                    device=device,
                    dtype=torch.int32,
                )
                .reshape(rows, src_width)
                .contiguous()
            )
            dst = torch.full((rows, dst_width), -9999, device=device, dtype=torch.int32)
            args[f"src_{name}"] = src
            args[f"dst_{name}"] = dst
            exp = torch.full_like(dst, fill)
            exp[:, : min(src_width, dst_width)] = src[:, : min(src_width, dst_width)]
            expected[name] = exp
            counter += 1000
        args["rows"] = rows
        args["graph_inputs_bound"] = graph_inputs_bound
        return args, expected

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE, "1")
    for graph_inputs_bound in (False, True):
        args, expected = make_args(graph_inputs_bound)
        assert dsv4_kernel.copy_decode_metadata_for_replay(**args)
        torch.cuda.synchronize()
        for name in one_d_names:
            assert torch.equal(args[f"dst_{name}"], expected[name])
        assert torch.equal(args["dst_cu_seqlens_q"], expected["cu_seqlens_q"])
        for name in two_d_specs:
            assert torch.equal(args[f"dst_{name}"], expected[name])


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_copy_decode_metadata_for_replay_can_skip_c4_sparse(monkeypatch):
    device = torch.device("cuda")
    rows = 2

    def vec(offset: int) -> torch.Tensor:
        return torch.arange(offset, offset + rows, device=device, dtype=torch.int32)

    args = {
        "dst_raw_out_loc": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_raw_out_loc": vec(10),
        "dst_seq_lens": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_seq_lens": vec(20),
        "dst_req_seq_lens": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_req_seq_lens": vec(30),
        "dst_extend_lens": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_extend_lens": vec(40),
        "dst_positions": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_positions": vec(50),
        "dst_req_table_indices": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_req_table_indices": vec(60),
        "dst_swa_topk_lengths": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_swa_topk_lengths": vec(70),
        "dst_c4_topk_lengths_raw": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_c4_topk_lengths_raw": vec(80),
        "dst_c4_topk_lengths_clamp1": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_c4_topk_lengths_clamp1": vec(90),
        "dst_c4_sparse_topk_lengths": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_c4_sparse_topk_lengths": vec(100),
        "dst_c128_topk_lengths_clamp1": torch.full((rows,), -999, device=device, dtype=torch.int32),
        "src_c128_topk_lengths_clamp1": vec(110),
        "dst_cu_seqlens_q": torch.full((rows + 1,), -999, device=device, dtype=torch.int32),
        "src_cu_seqlens_q": torch.arange(120, 120 + rows + 1, device=device, dtype=torch.int32),
        "dst_page_table": torch.full((rows, 2), -999, device=device, dtype=torch.int32),
        "src_page_table": torch.arange(
            200, 200 + rows * 2, device=device, dtype=torch.int32
        ).reshape(rows, 2),
        "dst_swa_page_indices": torch.full((rows, 3), -999, device=device, dtype=torch.int32),
        "src_swa_page_indices": torch.arange(
            300, 300 + rows * 3, device=device, dtype=torch.int32
        ).reshape(rows, 3),
        "dst_c4_sparse_raw_indices": torch.full((rows, 4), -777, device=device, dtype=torch.int32),
        "src_c4_sparse_raw_indices": torch.arange(
            400, 400 + rows, device=device, dtype=torch.int32
        ).reshape(rows, 1),
        "dst_c4_sparse_page_indices": torch.full((rows, 4), -778, device=device, dtype=torch.int32),
        "src_c4_sparse_page_indices": torch.arange(
            500, 500 + rows, device=device, dtype=torch.int32
        ).reshape(rows, 1),
        "dst_c4_sparse_full_indices": torch.full((rows, 4), -779, device=device, dtype=torch.int32),
        "src_c4_sparse_full_indices": torch.arange(
            600, 600 + rows, device=device, dtype=torch.int32
        ).reshape(rows, 1),
        "dst_c128_raw_indices": torch.full((rows, 2), -999, device=device, dtype=torch.int32),
        "src_c128_raw_indices": torch.arange(
            700, 700 + rows * 2, device=device, dtype=torch.int32
        ).reshape(rows, 2),
        "dst_c128_page_indices": torch.full((rows, 2), -999, device=device, dtype=torch.int32),
        "src_c128_page_indices": torch.arange(
            800, 800 + rows * 2, device=device, dtype=torch.int32
        ).reshape(rows, 2),
        "dst_c128_full_indices": torch.full((rows, 2), -999, device=device, dtype=torch.int32),
        "src_c128_full_indices": torch.arange(
            900, 900 + rows * 2, device=device, dtype=torch.int32
        ).reshape(rows, 2),
        "rows": rows,
        "graph_inputs_bound": False,
        "skip_c4_sparse_indices": True,
    }
    expected_c4 = {
        "dst_c4_sparse_raw_indices": args["dst_c4_sparse_raw_indices"].clone(),
        "dst_c4_sparse_page_indices": args["dst_c4_sparse_page_indices"].clone(),
        "dst_c4_sparse_full_indices": args["dst_c4_sparse_full_indices"].clone(),
    }

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE, "1")
    assert dsv4_kernel.copy_decode_metadata_for_replay(**args)
    torch.cuda.synchronize()

    assert torch.equal(args["dst_seq_lens"], args["src_seq_lens"])
    assert torch.equal(args["dst_page_table"], args["src_page_table"])
    assert torch.equal(args["dst_c128_raw_indices"], args["src_c128_raw_indices"])
    for name, expected in expected_c4.items():
        assert torch.equal(args[name], expected)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_direct_c4_sparse_metadata_for_replay_component_tables_match_oracle(monkeypatch):
    device = torch.device("cuda")
    rows = 3
    page_size = 128
    index_topk = 5
    max_seqlen_k = 384

    ctx_page_table = torch.stack(
        [
            torch.arange(0, max_seqlen_k, dtype=torch.int32),
            torch.arange(1024, 1024 + max_seqlen_k, dtype=torch.int32),
            torch.arange(2048, 2048 + max_seqlen_k, dtype=torch.int32),
        ]
    ).to(device)
    ctx_page_table[0, :page_size] = -1
    table_indices = torch.arange(rows, dtype=torch.int32, device=device)
    positions = torch.tensor([255, 256, 383], dtype=torch.int32, device=device)
    c4_page_table = torch.tensor(
        [[10, 11, 12], [13, -1, 15], [16, 17, 18]],
        dtype=torch.int32,
        device=device,
    )
    dst_raw = torch.full((rows, 8), -99, dtype=torch.int32, device=device)
    dst_page = torch.full_like(dst_raw, -98)
    dst_full = torch.full_like(dst_raw, -97)

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE, "1")
    assert dsv4_kernel.direct_c4_sparse_metadata_for_replay(
        ctx_page_table=ctx_page_table,
        table_indices=table_indices,
        positions=positions,
        c4_page_table=c4_page_table,
        dst_c4_sparse_raw_indices=dst_raw,
        dst_c4_sparse_page_indices=dst_page,
        dst_c4_sparse_full_indices=dst_full,
        rows=rows,
        page_size=page_size,
        index_topk=index_topk,
        component_loc_ownership=True,
    )
    torch.cuda.synchronize()

    c4_page_size = page_size // 4
    cpu_ctx = ctx_page_table.cpu()
    cpu_c4 = c4_page_table.cpu()
    for row, pos in enumerate(positions.cpu().tolist()):
        c4_len = (pos + 1) // 4
        c4_start = max(c4_len - index_topk, 0)
        c4_raw = list(range(c4_start, c4_len))
        expected_raw = c4_raw + [-1] * (dst_raw.shape[1] - len(c4_raw))
        expected_page = []
        expected_full = []
        for raw in c4_raw:
            logical_page = raw // c4_page_size
            offset = raw % c4_page_size
            component_page = int(cpu_c4[row, logical_page].item())
            expected_page.append(
                component_page * c4_page_size + offset if component_page >= 0 else -1
            )
            full = int(cpu_ctx[row, raw * 4 + 3].item())
            expected_full.append(full if full >= 0 else -1)
        expected_page += [-1] * (dst_page.shape[1] - len(expected_page))
        expected_full += [-1] * (dst_full.shape[1] - len(expected_full))

        assert dst_raw[row].cpu().tolist() == expected_raw
        assert dst_page[row].cpu().tolist() == expected_page
        assert dst_full[row].cpu().tolist() == expected_full


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_direct_decode_index_metadata_for_replay_matches_oracle(monkeypatch):
    device = torch.device("cuda")
    rows = 3
    page_size = 128
    window_size = 8
    index_topk = 5
    max_seqlen_k = 384

    ctx_page_table = torch.stack(
        [
            torch.arange(0, max_seqlen_k, dtype=torch.int32),
            torch.arange(1024, 1024 + max_seqlen_k, dtype=torch.int32),
            torch.arange(2048, 2048 + max_seqlen_k, dtype=torch.int32),
        ]
    ).to(device)
    ctx_page_table[0, :page_size] = -1
    table_indices = torch.arange(rows, dtype=torch.int32, device=device)
    positions = torch.tensor([255, 256, 383], dtype=torch.int32, device=device)
    c4_page_table = torch.tensor(
        [[10, 11, 12], [13, -1, 15], [16, 17, 18]],
        dtype=torch.int32,
        device=device,
    )
    c128_page_table = torch.tensor(
        [[20, 21, 22], [-1, 31, 32], [40, 41, 42]],
        dtype=torch.int32,
        device=device,
    )
    dst_swa = torch.full((rows, 8), -91, dtype=torch.int32, device=device)
    dst_c4_raw = torch.full((rows, 8), -92, dtype=torch.int32, device=device)
    dst_c4_page = torch.full_like(dst_c4_raw, -93)
    dst_c4_full = torch.full_like(dst_c4_raw, -94)
    dst_c128_raw = torch.full((rows, 8), -95, dtype=torch.int32, device=device)
    dst_c128_page = torch.full_like(dst_c128_raw, -96)
    dst_c128_full = torch.full_like(dst_c128_raw, -97)

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE, "1")
    assert dsv4_kernel.direct_decode_index_metadata_for_replay(
        ctx_page_table=ctx_page_table,
        table_indices=table_indices,
        positions=positions,
        c4_page_table=c4_page_table,
        c128_page_table=c128_page_table,
        dst_swa_page_indices=dst_swa,
        dst_c4_sparse_raw_indices=dst_c4_raw,
        dst_c4_sparse_page_indices=dst_c4_page,
        dst_c4_sparse_full_indices=dst_c4_full,
        dst_c128_raw_indices=dst_c128_raw,
        dst_c128_page_indices=dst_c128_page,
        dst_c128_full_indices=dst_c128_full,
        rows=rows,
        page_size=page_size,
        window_size=window_size,
        index_topk=index_topk,
        direct_swa=True,
        direct_c4=True,
        direct_c128=True,
    )
    torch.cuda.synchronize()

    cpu_ctx = ctx_page_table.cpu()
    cpu_c4 = c4_page_table.cpu()
    cpu_c128 = c128_page_table.cpu()
    c4_page_size = page_size // 4
    c128_page_size = max(page_size // 128, 1)
    for row, pos in enumerate(positions.cpu().tolist()):
        expected_swa = []
        for offset in range(dst_swa.shape[1]):
            logical = pos - offset
            expected_swa.append(int(cpu_ctx[row, logical].item()) if logical >= 0 else -1)
        assert dst_swa[row].cpu().tolist() == expected_swa

        c4_len = (pos + 1) // 4
        c4_raw = list(range(max(c4_len - index_topk, 0), c4_len))
        expected_c4_raw = c4_raw + [-1] * (dst_c4_raw.shape[1] - len(c4_raw))
        expected_c4_page = []
        expected_c4_full = []
        for raw in c4_raw:
            logical_page = raw // c4_page_size
            offset = raw % c4_page_size
            component_page = int(cpu_c4[row, logical_page].item())
            expected_c4_page.append(
                component_page * c4_page_size + offset if component_page >= 0 else -1
            )
            full = int(cpu_ctx[row, raw * 4 + 3].item())
            expected_c4_full.append(full if full >= 0 else -1)
        expected_c4_page += [-1] * (dst_c4_page.shape[1] - len(expected_c4_page))
        expected_c4_full += [-1] * (dst_c4_full.shape[1] - len(expected_c4_full))
        assert dst_c4_raw[row].cpu().tolist() == expected_c4_raw
        assert dst_c4_page[row].cpu().tolist() == expected_c4_page
        assert dst_c4_full[row].cpu().tolist() == expected_c4_full

        c128_len = (pos + 1) // 128
        expected_c128_raw = list(range(c128_len)) + [-1] * (dst_c128_raw.shape[1] - c128_len)
        expected_c128_page = []
        expected_c128_full = []
        for raw in range(c128_len):
            logical_page = raw // c128_page_size
            offset = raw % c128_page_size
            component_page = int(cpu_c128[row, logical_page].item())
            expected_c128_page.append(
                component_page * c128_page_size + offset if component_page >= 0 else -1
            )
            full = int(cpu_ctx[row, raw * 128 + 127].item())
            expected_c128_full.append(full if full >= 0 else -1)
        expected_c128_page += [-1] * (dst_c128_page.shape[1] - len(expected_c128_page))
        expected_c128_full += [-1] * (dst_c128_full.shape[1] - len(expected_c128_full))
        assert dst_c128_raw[row].cpu().tolist() == expected_c128_raw
        assert dst_c128_page[row].cpu().tolist() == expected_c128_page
        assert dst_c128_full[row].cpu().tolist() == expected_c128_full


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_direct_decode_index_metadata_for_replay_swa_independent_matches_oracle(monkeypatch):
    device = torch.device("cuda")
    rows = 3
    page_size = 128
    window_size = 8
    index_topk = 5
    num_pages = 64
    dummy_token_start = num_pages * page_size
    swa_dummy_page = num_pages - 1

    ctx_page_table = torch.stack(
        [
            torch.arange(0, 512, dtype=torch.int32),
            torch.arange(1024, 1536, dtype=torch.int32),
            torch.arange(2048, 2560, dtype=torch.int32),
        ]
    ).to(device)
    ctx_page_table[1, 125] = -1
    ctx_page_table[2, 376] = dummy_token_start
    table_indices = torch.arange(rows, dtype=torch.int32, device=device)
    positions = torch.tensor([127, 130, 383], dtype=torch.int32, device=device)
    full_to_swa_page = torch.remainder(
        torch.arange(num_pages, dtype=torch.int32, device=device) * 7 + 5,
        num_pages - 1,
    )
    full_to_swa_page[::9] = -1
    dst_swa = torch.full((rows, window_size), -91, dtype=torch.int32, device=device)
    dummy2d = torch.empty((rows, 1), dtype=torch.int32, device=device)

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS_TOGGLE, "1")
    assert dsv4_kernel.direct_decode_index_metadata_for_replay(
        ctx_page_table=ctx_page_table,
        table_indices=table_indices,
        positions=positions,
        c4_page_table=None,
        c128_page_table=None,
        dst_swa_page_indices=dst_swa,
        dst_c4_sparse_raw_indices=dummy2d,
        dst_c4_sparse_page_indices=dummy2d,
        dst_c4_sparse_full_indices=dummy2d,
        dst_c128_raw_indices=dummy2d,
        dst_c128_page_indices=dummy2d,
        dst_c128_full_indices=dummy2d,
        rows=rows,
        page_size=page_size,
        window_size=window_size,
        index_topk=index_topk,
        direct_swa=True,
        direct_c4=False,
        direct_c128=False,
        swa_full_to_swa_page=full_to_swa_page,
        swa_dummy_token_start=dummy_token_start,
        swa_dummy_page=swa_dummy_page,
        swa_independent=True,
    )
    torch.cuda.synchronize()

    cpu_ctx = ctx_page_table.cpu()
    cpu_map = full_to_swa_page.cpu()
    for row, pos in enumerate(positions.cpu().tolist()):
        expected = []
        for offset in range(window_size):
            logical = pos - offset
            if logical < 0:
                expected.append(-1)
                continue
            full_loc = int(cpu_ctx[row, logical].item())
            if full_loc == dummy_token_start:
                expected.append(swa_dummy_page * page_size)
                continue
            full_page = full_loc // page_size
            page_offset = full_loc % page_size
            if full_loc < 0 or full_page < 0 or full_page >= num_pages:
                expected.append(-1)
                continue
            swa_page = int(cpu_map[full_page].item())
            expected.append(swa_page * page_size + page_offset if swa_page >= 0 else -1)
        assert dst_swa[row].cpu().tolist() == expected


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_decode_metadata_deforest_component_tables_match_oracle(monkeypatch):
    device = torch.device("cuda")
    rows = 3
    page_size = 128
    max_seqlen_k = 384
    window_size = 8
    index_topk = 5
    alignment = 64
    table_len = 3

    ctx_page_table = torch.stack(
        [
            torch.arange(0, max_seqlen_k, dtype=torch.int32),
            torch.arange(1024, 1024 + max_seqlen_k, dtype=torch.int32),
            torch.arange(2048, 2048 + max_seqlen_k, dtype=torch.int32),
        ]
    ).to(device)
    ctx_page_table[0, :page_size] = -1
    table_indices = torch.arange(rows, dtype=torch.int32, device=device)
    positions = torch.tensor([255, 256, 383], dtype=torch.int32, device=device)
    c4_page_table = torch.tensor(
        [[10, 11, 12], [13, -1, 15], [16, 17, 18]],
        dtype=torch.int32,
        device=device,
    )
    c128_page_table = torch.tensor(
        [[20, 21, 22], [-1, 31, 32], [40, 41, 42]],
        dtype=torch.int32,
        device=device,
    )

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_DECODE_METADATA_DEFOREST_TOGGLE, "1")
    out = dsv4_kernel.decode_metadata_deforest_fallback(
        ctx_page_table,
        table_indices,
        positions,
        page_size=page_size,
        max_seqlen_k=max_seqlen_k,
        window_size=window_size,
        index_topk=index_topk,
        alignment=alignment,
        c4_page_table=c4_page_table,
        c128_page_table=c128_page_table,
        component_loc_ownership=True,
    )
    assert out is not None
    torch.cuda.synchronize()

    c4_page_size = page_size // 4
    c128_page_size = max(page_size // 128, 1)
    cpu_ctx = ctx_page_table.cpu()
    cpu_c4 = c4_page_table.cpu()
    cpu_c128 = c128_page_table.cpu()
    cpu_pos = positions.cpu().tolist()

    for row, pos in enumerate(cpu_pos):
        seq_len = pos + 1
        expected_pages = []
        for logical_page in range(table_len):
            full = int(cpu_ctx[row, logical_page * page_size].item())
            expected_pages.append(full // page_size if full >= 0 else -1)
        assert out.page_table[row, :table_len].cpu().tolist() == expected_pages

        c4_len = seq_len // 4
        c4_start = max(c4_len - index_topk, 0)
        c4_raw = list(range(c4_start, c4_len))
        expected_c4_locs = []
        expected_c4_full = []
        for raw in c4_raw:
            logical_page = raw // c4_page_size
            offset = raw % c4_page_size
            component_page = int(cpu_c4[row, logical_page].item())
            expected_c4_locs.append(
                component_page * c4_page_size + offset if component_page >= 0 else -1
            )
            full_pos = raw * 4 + 3
            full = int(cpu_ctx[row, full_pos].item())
            expected_c4_full.append(full if full >= 0 else -1)
        assert out.c4_sparse_raw_indices[row, : len(c4_raw)].cpu().tolist() == c4_raw
        assert out.c4_sparse_page_indices[row, : len(c4_raw)].cpu().tolist() == expected_c4_locs
        assert out.c4_sparse_full_indices[row, : len(c4_raw)].cpu().tolist() == expected_c4_full

        c128_len = seq_len // 128
        expected_c128_locs = []
        expected_c128_full = []
        for raw in range(c128_len):
            logical_page = raw // c128_page_size
            offset = raw % c128_page_size
            component_page = int(cpu_c128[row, logical_page].item())
            expected_c128_locs.append(
                component_page * c128_page_size + offset if component_page >= 0 else -1
            )
            full_pos = raw * 128 + 127
            full = int(cpu_ctx[row, full_pos].item())
            expected_c128_full.append(full if full >= 0 else -1)
        assert out.c128_raw_indices[row, :c128_len].cpu().tolist() == list(range(c128_len))
        assert out.c128_page_indices[row, :c128_len].cpu().tolist() == expected_c128_locs
        assert out.c128_full_indices[row, :c128_len].cpu().tolist() == expected_c128_full


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_copy_component_write_locs_for_replay_from_component_tables(monkeypatch):
    device = torch.device("cuda")
    rows = 5
    page_size = 128
    c4_page_table = torch.tensor(
        [[7, 8], [9, 10], [11, 12], [13, 14], [15, -1]],
        dtype=torch.int32,
        device=device,
    )
    c128_page_table = torch.tensor(
        [[17, 18], [19, 20], [21, 22], [23, 24], [25, -1]],
        dtype=torch.int32,
        device=device,
    )
    c4_indexer_page_table = torch.tensor(
        [[27, 28], [29, 30], [31, 32], [33, 34], [35, -1]],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.tensor([3, 4, 127, 128, 255], dtype=torch.int32, device=device)
    c4_out = torch.full((rows,), -999, dtype=torch.int32, device=device)
    c128_out = torch.full((rows,), -999, dtype=torch.int32, device=device)
    c4_indexer_out = torch.full((rows,), -999, dtype=torch.int32, device=device)

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_REPLAY_METADATA_COPY_TOGGLE, "1")
    assert dsv4_kernel.copy_component_write_locs_for_replay(
        c4_page_table=c4_page_table,
        c128_page_table=c128_page_table,
        c4_indexer_page_table=c4_indexer_page_table,
        positions=positions,
        c4_out_loc=c4_out,
        c128_out_loc=c128_out,
        c4_indexer_out_loc=c4_indexer_out,
        rows=rows,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    assert c4_out.cpu().tolist() == [7 * 32, -1, 11 * 32 + 31, -1, -1]
    assert c128_out.cpu().tolist() == [-1, -1, 21, -1, -1]
    assert c4_indexer_out.cpu().tolist() == [27 * 32, -1, 31 * 32 + 31, -1, -1]


def test_dsv4_unsupported_sm80_paths_fail_clearly():
    with pytest.raises(NotImplementedError) as exc:
        dsv4_kernel.fused_q_indexer_rope_hadamard_fp4_quant()

    message = str(exc.value)
    assert "fused_q_indexer_rope_hadamard_fp4_quant" in message
    assert "sm" in message or "no CUDA" in message


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_linear_bf16_fp32_upstream_opt_in_matches_bf16_mm(monkeypatch):
    device = torch.device("cuda")
    torch.manual_seed(23)
    x = torch.randn(3, 2, 128, device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(5, 128, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv(dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE, "1")
    actual = dsv4_kernel.linear_bf16_fp32_fallback(x, weight_bf16)
    expected = torch.mm(
        x.reshape(-1, x.shape[-1]).contiguous(),
        weight_bf16.contiguous().t(),
        out_dtype=torch.float32,
    ).reshape(3, 2, 5)

    assert actual.dtype is torch.float32
    assert torch.allclose(actual, expected)

    weight_fp32 = weight_bf16.float()
    fp32_weight_actual = dsv4_kernel.linear_bf16_fp32_fallback(x, weight_fp32)
    fp32_weight_expected = F.linear(x.float(), weight_fp32)
    assert torch.allclose(fp32_weight_actual, fp32_weight_expected)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_hc_head_maintains_bf16_linear_weight_cache(monkeypatch):
    device = torch.device("cuda")
    set_tp_info(0, 1)
    cfg = _tiny_dsv4_cache_config([0])
    model = dsv4_model.DeepseekV4Model(cfg)
    torch.manual_seed(29)
    model.hc_head_fn = (torch.randn_like(model.hc_head_fn, device=device) * 0.01).contiguous()
    model.hc_head_base = torch.zeros_like(model.hc_head_base, device=device)
    model.hc_head_scale = torch.full_like(model.hc_head_scale, 0.1, device=device)
    x = torch.randn(7, cfg.hc_mult, cfg.hidden_size, device=device, dtype=torch.bfloat16)

    monkeypatch.delenv(dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE, raising=False)
    expected = model._hc_head(x)

    monkeypatch.setenv(dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE, "1")
    actual = model._hc_head(x)
    assert model._hc_head_fn_bf16 is not None
    assert model._hc_head_fn_bf16.dtype is torch.bfloat16
    assert torch.allclose(actual, expected, atol=5e-3, rtol=5e-3)

    old_cache = model._hc_head_fn_bf16
    with torch.no_grad():
        model.hc_head_fn.add_(0.01)
    _ = model._hc_head(x)
    assert model._hc_head_fn_bf16 is not old_cache


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_hc_sm80_triton_opt_in_matches_torch_fallback(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(31)

    tokens = 5
    hc_mult = 4
    hidden = 64
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(tokens, hc_mult, hidden, device=device, dtype=torch.bfloat16)
    fn = (torch.randn(mix_hc, hc_mult * hidden, device=device) * 0.02).contiguous()
    scale = torch.tensor([0.15, 0.1, 0.08], device=device, dtype=torch.float32)
    base = (torch.randn(mix_hc, device=device) * 0.01).contiguous()

    expected_y, expected_post, expected_comb = dsv4_kernel.hc_pre_fallback(
        x,
        fn,
        scale,
        base,
        hc_mult=hc_mult,
        sinkhorn_iters=3,
        eps=1e-6,
        norm_eps=1e-6,
    )
    post_input = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    expected_post_out = dsv4_kernel.hc_post_fallback(
        post_input,
        x,
        expected_post,
        expected_comb,
    )

    monkeypatch.setenv("MINISGL_DSV4_SM80_HC", "1")
    actual_y, actual_post, actual_comb = dsv4_kernel.hc_pre_fallback(
        x,
        fn,
        scale,
        base,
        hc_mult=hc_mult,
        sinkhorn_iters=3,
        eps=1e-6,
        norm_eps=1e-6,
    )
    actual_post_out = dsv4_kernel.hc_post_fallback(
        post_input,
        x,
        actual_post,
        actual_comb,
    )

    assert torch.allclose(actual_y, expected_y, atol=8e-3, rtol=8e-3)
    assert torch.allclose(actual_post, expected_post, atol=8e-3, rtol=8e-3)
    assert torch.allclose(actual_comb, expected_comb, atol=8e-3, rtol=8e-3)
    assert torch.allclose(actual_post_out, expected_post_out, atol=8e-3, rtol=8e-3)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_hc_graph_cleanup_opt_in_matches_current_hc_path(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(32)

    tokens = 6
    hc_mult = 4
    hidden = 128
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(tokens, hc_mult, hidden, device=device, dtype=torch.bfloat16).contiguous()
    fn = (torch.randn(mix_hc, hc_mult * hidden, device=device) * 0.02).to(torch.bfloat16)
    fn = fn.contiguous()
    scale = torch.tensor([0.15, 0.1, 0.08], device=device, dtype=torch.float32)
    base = (torch.randn(mix_hc, device=device) * 0.01).contiguous()
    post_input = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)

    monkeypatch.setenv("MINISGL_DSV4_SM80_HC", "1")
    expected_y, expected_post, expected_comb = dsv4_kernel.hc_pre_fallback(
        x,
        fn,
        scale,
        base,
        hc_mult=hc_mult,
        sinkhorn_iters=5,
        eps=1e-6,
        norm_eps=1e-6,
    )
    expected_post_out = dsv4_kernel.hc_post_fallback(
        post_input,
        x,
        expected_post,
        expected_comb,
    )

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE, "1")
    actual_y, actual_post, actual_comb = dsv4_kernel.hc_pre_fallback(
        x,
        fn,
        scale,
        base,
        hc_mult=hc_mult,
        sinkhorn_iters=5,
        eps=1e-6,
        norm_eps=1e-6,
    )
    actual_post_out = dsv4_kernel.hc_post_fallback(
        post_input,
        x,
        actual_post,
        actual_comb,
    )

    assert actual_y.shape == expected_y.shape
    assert actual_post.shape == expected_post.shape
    assert actual_comb.shape == expected_comb.shape
    assert actual_y.dtype is torch.bfloat16
    assert actual_post.dtype is torch.bfloat16
    assert actual_comb.dtype is torch.bfloat16
    assert torch.allclose(actual_y, expected_y, atol=1.1e-2, rtol=1.1e-2)
    assert torch.allclose(actual_post, expected_post, atol=1.1e-2, rtol=1.1e-2)
    assert torch.allclose(actual_comb, expected_comb, atol=1.1e-2, rtol=1.1e-2)
    assert torch.allclose(actual_post_out, expected_post_out, atol=1.1e-2, rtol=1.1e-2)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_rms_norm_sm80_triton_opt_in_matches_torch_fallback(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(37)

    x = torch.randn(5, 4, 128, device=device, dtype=torch.bfloat16)
    weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    expected = dsv4_kernel.rms_norm_fallback(x, weight, eps=1e-6)

    monkeypatch.setenv("MINISGL_DSV4_SM80_RMSNORM", "1")
    actual = dsv4_kernel.rms_norm_fallback(x, weight, eps=1e-6)

    assert actual.dtype is torch.bfloat16
    assert torch.allclose(actual, expected, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_rms_norm_pair_sm80_triton_opt_in_matches_fallback(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(39)

    q = torch.randn(5, 4, 96, device=device, dtype=torch.bfloat16)
    kv = torch.randn(5, 4, 64, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(96, device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    expected_q = dsv4_kernel.rms_norm_fallback(q, q_weight, eps=1e-6)
    expected_kv = dsv4_kernel.rms_norm_fallback(kv, kv_weight, eps=1e-6)

    monkeypatch.setenv("MINISGL_DSV4_SM80_FUSED_Q_KV_RMSNORM", "1")
    actual_q, actual_kv = dsv4_kernel.rms_norm_pair_fallback(
        q,
        kv,
        q_weight,
        kv_weight,
        eps=1e-6,
    )

    assert actual_q.dtype is torch.bfloat16
    assert actual_kv.dtype is torch.bfloat16
    assert torch.allclose(actual_q, expected_q, atol=5e-3, rtol=5e-3)
    assert torch.allclose(actual_kv, expected_kv, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_q_kv_norm_rope_cache_sm80_triton_opt_in_matches_fallback(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(40)

    tokens = 5
    heads = 3
    dim = 64
    rotary_dim = 32
    positions = torch.arange(tokens, device=device, dtype=torch.long) + 11
    out_loc = torch.arange(tokens, device=device, dtype=torch.long)
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16).contiguous()
    kv = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16).contiguous()
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)
    expected_q = q.clone()
    expected_kv = kv.clone()
    expected_cache = torch.empty(tokens, dim, device=device, dtype=torch.bfloat16)
    actual_q = q.clone()
    actual_kv = kv.clone()
    actual_cache = torch.empty_like(expected_cache)

    dsv4_kernel.q_norm_rope_fallback(
        expected_q,
        positions,
        rms_norm_eps=1e-6,
        rotary_dim=rotary_dim,
        base=10000.0,
    )
    dsv4_kernel.k_norm_rope_cache_fallback(
        expected_kv,
        positions,
        norm_weight=weight,
        rms_norm_eps=1e-6,
        cache=expected_cache,
        out_loc=out_loc,
        rotary_dim=rotary_dim,
        base=10000.0,
    )

    monkeypatch.setenv("MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE", "1")
    assert dsv4_kernel.q_kv_norm_rope_cache_fallback(
        actual_q,
        actual_kv,
        positions,
        norm_weight=weight,
        rms_norm_eps=1e-6,
        cache=actual_cache,
        out_loc=out_loc,
        rotary_dim=rotary_dim,
        base=10000.0,
    )

    assert torch.allclose(actual_q, expected_q, atol=5e-3, rtol=5e-3)
    assert torch.allclose(actual_kv, expected_kv, atol=5e-3, rtol=5e-3)
    assert torch.allclose(actual_cache, expected_cache, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_q_kv_norm_rope_cache_accepts_strided_kv_view(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(45)

    tokens = 4
    heads = 2
    dim = 64
    rotary_dim = 32
    q_prefix = 96
    positions = torch.arange(tokens, device=device, dtype=torch.long) + 23
    out_loc = torch.arange(tokens, device=device, dtype=torch.long)
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16).contiguous()
    merged = torch.randn(tokens, q_prefix + dim, device=device, dtype=torch.bfloat16)
    kv_view = merged[:, q_prefix:]
    assert kv_view.stride(-1) == 1
    assert not kv_view.is_contiguous()
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)

    expected_q = q.clone()
    expected_kv = kv_view.clone()
    expected_cache = torch.empty(tokens, dim, device=device, dtype=torch.bfloat16)
    actual_q = q.clone()
    actual_cache = torch.empty_like(expected_cache)

    dsv4_kernel.q_norm_rope_fallback(
        expected_q,
        positions,
        rms_norm_eps=1e-6,
        rotary_dim=rotary_dim,
        base=10000.0,
    )
    dsv4_kernel.k_norm_rope_cache_fallback(
        expected_kv,
        positions,
        norm_weight=weight,
        rms_norm_eps=1e-6,
        cache=expected_cache,
        out_loc=out_loc,
        rotary_dim=rotary_dim,
        base=10000.0,
    )

    monkeypatch.setenv("MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE", "1")
    assert dsv4_kernel.q_kv_norm_rope_cache_fallback(
        actual_q,
        kv_view,
        positions,
        norm_weight=weight,
        rms_norm_eps=1e-6,
        cache=actual_cache,
        out_loc=out_loc,
        rotary_dim=rotary_dim,
        base=10000.0,
    )

    assert torch.allclose(actual_q, expected_q, atol=5e-3, rtol=5e-3)
    assert torch.allclose(kv_view, expected_kv, atol=5e-3, rtol=5e-3)
    assert torch.allclose(actual_cache, expected_cache, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_fp8_activation_quant_triton_matches_torch_reference(monkeypatch):
    if getattr(torch, "float8_e4m3fn", None) is None:
        pytest.skip("torch.float8_e4m3fn is required for FP8 activation quantization")
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(754)

    x = torch.randn(5, 256, device=device, dtype=torch.bfloat16)
    x[0, ::17] *= 5
    expected = dsv4_kernel.quantize_fp8_activation_ref(x, block_size=128)

    monkeypatch.setenv("MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON", "1")
    actual = dsv4_kernel.quantize_fp8_activation_ref(x, block_size=128)

    assert dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON")
    assert actual.dtype is torch.bfloat16
    assert torch.allclose(actual, expected, atol=1e-2, rtol=0.0)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_quantized_linear_fp8_per_call_gemm_matches_fallback(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(41)

    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    weight = (
        torch.randn(256, 128, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    scale = torch.rand(
        dsv4_kernel.scale_dim(weight.shape[0]),
        dsv4_kernel.scale_dim(weight.shape[1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())

    expected = dsv4_kernel.quantized_linear_ref(x, weight, scale, weight_kind="fp8")
    actual = dsv4_kernel.quantized_linear_ref(
        x,
        weight,
        scale,
        weight_kind="fp8",
        fp8_gemm=True,
    )

    assert not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_FP8_GEMM")
    assert actual.dtype is torch.bfloat16
    assert torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_static_projection_scale_cache_preserves_projection_outputs(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(756)

    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    fp8_weight = (
        torch.randn(256, 128, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    fp8_scale = torch.rand(
        dsv4_kernel.scale_dim(fp8_weight.shape[0]),
        dsv4_kernel.scale_dim(fp8_weight.shape[1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())

    class Owner:
        pass

    owner = Owner()
    assert dsv4_model._cached_projection_scale(owner, "_test_scale_cache", fp8_scale) is fp8_scale

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_STATIC_SCALE_CACHE_TOGGLE, "1")
    cached_fp8_scale = dsv4_model._cached_projection_scale(
        owner,
        "_test_scale_cache",
        fp8_scale,
    )
    assert cached_fp8_scale is not fp8_scale
    assert cached_fp8_scale.dtype is torch.float32
    assert cached_fp8_scale.is_contiguous()
    assert (
        dsv4_model._cached_projection_scale(owner, "_test_scale_cache", fp8_scale)
        is cached_fp8_scale
    )
    assert (
        dsv4_model._cached_projection_scale(owner, "_test_scale_cache", fp8_scale.clone())
        is not cached_fp8_scale
    )

    monkeypatch.setenv("MINISGL_DSV4_SM80_FP8_GEMM", "1")
    expected_fp8 = dsv4_kernel.quantized_linear_ref(
        x,
        fp8_weight,
        fp8_scale,
        weight_kind="fp8",
        fp8_gemm=True,
    )
    actual_fp8 = dsv4_kernel.quantized_linear_ref(
        x,
        fp8_weight,
        cached_fp8_scale,
        weight_kind="fp8",
        fp8_gemm=True,
    )
    assert torch.allclose(actual_fp8, expected_fp8, atol=3e-2, rtol=3e-2)

    fp4_weight = torch.randint(-128, 127, (96, 64), device=device, dtype=torch.int8)
    fp4_scale = torch.rand(96, 4, device=device, dtype=torch.float32).to(dsv4_kernel.e8m0_dtype())
    fp4_owner = Owner()
    cached_fp4_scale = dsv4_model._cached_projection_scale(
        fp4_owner,
        "_test_scale_cache",
        fp4_scale,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_FP4_GEMM", "1")
    expected_fp4 = dsv4_kernel.quantized_linear_ref(
        x[:, :64],
        fp4_weight,
        fp4_scale,
        weight_kind="fp4",
    )
    actual_fp4 = dsv4_kernel.quantized_linear_ref(
        x[:, :64],
        fp4_weight,
        cached_fp4_scale,
        weight_kind="fp4",
    )
    assert torch.allclose(actual_fp4, expected_fp4, atol=3e-2, rtol=3e-2)

    wo_o = torch.randn(4, 2, 64, device=device, dtype=torch.bfloat16)
    wo_rank = 48
    wo_weight = (
        torch.randn(2 * wo_rank, wo_o.shape[-1], device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    wo_scale = torch.rand(
        dsv4_kernel.scale_dim(2 * wo_rank),
        dsv4_kernel.scale_dim(wo_o.shape[-1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())
    wo_owner = Owner()
    cached_wo_scale = dsv4_model._cached_projection_scale(
        wo_owner,
        "_test_scale_cache",
        wo_scale,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_WO_A_BF16", "1")
    expected_wo = dsv4_kernel.wo_a_grouped_projection_fallback(
        wo_o,
        wo_weight,
        wo_scale,
        num_local_groups=2,
        o_lora_rank=wo_rank,
    )
    actual_wo = dsv4_kernel.wo_a_grouped_projection_fallback(
        wo_o,
        wo_weight,
        cached_wo_scale,
        num_local_groups=2,
        o_lora_rank=wo_rank,
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_wo, expected_wo, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_quantized_linear_fp8_pair_shared_activation_matches_fallback(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(43)

    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    weight_a = (
        torch.randn(96, 128, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    weight_b = (
        torch.randn(64, 128, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    scale_a = torch.rand(
        dsv4_kernel.scale_dim(weight_a.shape[0]),
        dsv4_kernel.scale_dim(weight_a.shape[1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())
    scale_b = torch.rand(
        dsv4_kernel.scale_dim(weight_b.shape[0]),
        dsv4_kernel.scale_dim(weight_b.shape[1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())

    expected_a = dsv4_kernel.quantized_linear_ref(x, weight_a, scale_a, weight_kind="fp8")
    expected_b = dsv4_kernel.quantized_linear_ref(x, weight_b, scale_b, weight_kind="fp8")
    actual_a, actual_b = dsv4_kernel.quantized_linear_fp8_pair_shared_activation_ref(
        x, weight_a, scale_a, weight_b, scale_b
    )

    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_b, expected_b)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_fused_wqa_wkv_cached_weight_matches_shared_activation(monkeypatch):
    device = torch.device("cuda")
    _clear_dsv4_sm80_env(monkeypatch)
    torch.manual_seed(44)

    x = torch.randn(4, 128, device=device, dtype=torch.bfloat16)
    weight_a = (
        torch.randn(96, 128, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    weight_b = (
        torch.randn(64, 128, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    scale_a = torch.rand(
        dsv4_kernel.scale_dim(weight_a.shape[0]),
        dsv4_kernel.scale_dim(weight_a.shape[1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())
    scale_b = torch.rand(
        dsv4_kernel.scale_dim(weight_b.shape[0]),
        dsv4_kernel.scale_dim(weight_b.shape[1]),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())

    expected_a, expected_b = dsv4_kernel.quantized_linear_fp8_pair_shared_activation_ref(
        x, weight_a, scale_a, weight_b, scale_b
    )

    class Owner:
        pass

    owner = Owner()
    monkeypatch.setenv("MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE", "1")
    cached = dsv4_model._cached_fused_wqa_wkv_fp8_weight(
        owner,
        "_cached_test_weight",
        weight_a,
        scale_a,
        weight_b,
        scale_b,
        out_dtype=x.dtype,
    )
    assert cached is not None
    qkv = F.linear(dsv4_kernel.quantize_fp8_activation_ref(x), cached)
    actual_a, actual_b = qkv.split([weight_a.shape[0], weight_b.shape[0]], dim=-1)

    assert torch.equal(actual_a, expected_a)
    assert torch.equal(actual_b, expected_b)
    assert (
        dsv4_model._cached_fused_wqa_wkv_fp8_weight(
            owner,
            "_cached_test_weight",
            weight_a,
            scale_a,
            weight_b,
            scale_b,
            out_dtype=x.dtype,
        )
        is cached
    )


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

    q_rope = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    q_positions = torch.tensor([0, 3, 9], dtype=torch.int64)
    expected_q_rope = q_rope.clone()
    expected_q_fp32 = expected_q_rope.float()
    expected_q_scale = torch.rsqrt(expected_q_fp32.square().mean(-1, keepdim=True) + 1e-6)
    expected_q_rope.copy_((expected_q_fp32 * expected_q_scale).to(expected_q_rope.dtype))
    dsv4_kernel.apply_rotary_tail(
        expected_q_rope,
        q_positions,
        rotary_dim=4,
        base=10000.0,
    )
    q_rope_ptr = q_rope.data_ptr()
    returned_q_rope = dsv4_kernel.q_norm_rope_fallback(
        q_rope,
        q_positions,
        rms_norm_eps=1e-6,
        rotary_dim=4,
        base=10000.0,
    )
    assert returned_q_rope.data_ptr() == q_rope_ptr
    assert torch.allclose(q_rope.float(), expected_q_rope.float(), atol=1e-2, rtol=1e-2)

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

    scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, -1.0, -2.0, -3.0, -4.0, -5.0],
            [0.0, 5.0, 1.0, 7.0, 3.0, 6.0, 2.0, 4.0],
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([3, 7, 0], dtype=torch.int32)
    page_table = torch.tensor(
        [
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        dtype=torch.int32,
    )
    full_topk = dsv4_kernel.topk_transform_512_full_fallback(
        scores,
        seq_lens,
        page_table,
        page_size=4,
        width=4,
        ratio=4,
    )
    assert full_topk.backend == "torch"
    _assert_full_topk_transform(
        full_topk,
        scores,
        seq_lens,
        page_table,
        page_size=4,
        width=4,
        ratio=4,
    )

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


def test_dsv4_rotary_yarn_fallback_matches_configured_ramp_range():
    rotary_dim = 64
    base = 10000.0
    original_seq_len = 65536
    factor = 16.0
    beta_fast = 32
    beta_slow = 1
    positions = torch.tensor([0, 127, 511, 1023], dtype=torch.int64)
    x = torch.randn(positions.numel(), 2, 96, dtype=torch.float32)

    def correction_dim(num_rotations: float) -> float:
        return (
            rotary_dim
            * torch.log(torch.tensor(original_seq_len / (num_rotations * 2 * torch.pi))).item()
            / (2 * torch.log(torch.tensor(base)).item())
        )

    low = max(int(torch.floor(torch.tensor(correction_dim(beta_fast))).item()), 0)
    high = min(
        int(torch.ceil(torch.tensor(correction_dim(beta_slow))).item()),
        rotary_dim // 2 - 1,
    )
    assert high == rotary_dim // 2 - 1

    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    ramp = torch.clamp(
        (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
        0,
        1,
    )
    smooth = 1 - ramp
    inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    cos = freqs.cos().unsqueeze(-2)
    sin = freqs.sin().unsqueeze(-2)
    rope = x[..., -rotary_dim:].float().unflatten(-1, (-1, 2))
    a, b = rope[..., 0], rope[..., 1]
    expected = x.clone()
    expected[..., -rotary_dim:] = torch.stack(
        (a * cos - b * sin, a * sin + b * cos),
        dim=-1,
    ).flatten(-2)

    actual = dsv4_kernel.apply_rotary_tail(
        x.clone(),
        positions,
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_dsv4_compress_forward_keeps_request_contiguous_windows():
    class MarkerWkvGate:
        def __init__(self, width: int):
            self.width = width

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            marker = x[:, :1].float()
            kv = marker.repeat(1, self.width)
            score = torch.zeros_like(kv)
            return torch.cat((kv, score), dim=-1).to(x.dtype)

    class IdentityNorm:
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def run_case(ratio: int, request_len: int, *, overlap: bool) -> None:
        rows = request_len * 2
        head_dim = 1
        coff = 2 if overlap else 1
        x = torch.arange(rows, dtype=torch.float32).view(rows, 1).to(torch.bfloat16)
        positions = torch.tensor(list(range(request_len)) * 2, dtype=torch.int64)
        ape = torch.zeros((ratio, coff * head_dim), dtype=torch.float32)
        expected = torch.tensor(
            [
                [(ratio - 1) / 2],
                [request_len + (ratio - 1) / 2],
            ],
            dtype=torch.bfloat16,
        )

        fallback = dsv4_kernel.compress_forward_fallback(
            x,
            positions,
            ratio=ratio,
            head_dim=head_dim,
            overlap=overlap,
            ape=ape,
            wkv_gate=MarkerWkvGate(coff * head_dim),
            norm=IdentityNorm(),
        )
        vectorized = dsv4_kernel._compress_forward_vectorized(
            x,
            positions,
            ratio=ratio,
            head_dim=head_dim,
            overlap=overlap,
            ape=ape,
            wkv_gate=MarkerWkvGate(coff * head_dim),
            norm=IdentityNorm(),
            apply_norm=True,
        )
        assert torch.equal(fallback, expected)
        assert vectorized is not None
        assert torch.equal(vectorized, expected)

    run_case(4, 5, overlap=True)
    run_case(128, 129, overlap=False)


def test_indexer_bf16_query_logits_and_topk_are_fallback_clean():
    q = torch.randn(3, 2, 4, dtype=torch.float32)
    positions = torch.tensor([0, 3, 7], dtype=torch.int64)
    expected_q = dsv4_kernel.apply_rotary_tail(
        q.clone(),
        positions,
        rotary_dim=2,
        base=10000.0,
    )
    expected_q = dsv4_kernel.hadamard_transform_ref(expected_q)
    actual_q = dsv4_kernel.indexer_q_rope_hadamard_bf16_fallback(
        q.clone(),
        positions,
        rotary_dim=2,
        base=10000.0,
    )
    assert torch.allclose(actual_q, expected_q, atol=1e-5, rtol=1e-5)

    q_logits = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=torch.bfloat16,
    )
    cache = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor([[1.0, 1.0], [0.5, 1.0]], dtype=torch.float32)
    seq_lens = torch.tensor([5, 4], dtype=torch.int32)
    page_table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    logits = dsv4_kernel.indexer_bf16_logits_fallback(
        q_logits,
        cache,
        seq_lens,
        page_table,
        page_size=4,
        weights=weights,
    )
    expected_logits = _manual_indexer_logits(
        q_logits,
        cache,
        weights,
        seq_lens,
        page_table,
        page_size=4,
    )
    assert torch.allclose(logits, expected_logits)

    selected = dsv4_kernel.indexer_select_bf16_fallback(
        q_logits,
        weights,
        cache,
        seq_lens,
        page_table,
        page_size=4,
        width=2,
        ratio=4,
    )
    for row in range(seq_lens.numel()):
        expected_raw = torch.topk(expected_logits[row, : seq_lens[row]], 2, sorted=False).indices
        assert sorted(selected.topk.raw_indices[row].tolist()) == sorted(expected_raw.tolist())
    assert selected.topk.topk_lens is not None
    assert selected.topk.topk_lens.tolist() == [2, 2]


def test_topk_transform_full_reports_lens_in_torch_fallback():
    scores = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([0, 2, 4], dtype=torch.int32)
    page_table = torch.tensor(
        [
            [0],
            [2],
            [4],
        ],
        dtype=torch.int32,
    )

    topk = dsv4_kernel.topk_transform_512_full_fallback(
        scores,
        seq_lens,
        page_table,
        page_size=4,
        width=3,
        ratio=4,
    )

    assert topk.backend == "torch"
    assert topk.topk_lens is not None
    assert topk.topk_lens.tolist() == [0, 2, 3]
    _assert_full_topk_transform(
        topk,
        scores,
        seq_lens,
        page_table,
        page_size=4,
        width=3,
        ratio=4,
    )


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


def test_dsv4_moe_v2_execution_plan_and_workspace_reuse_cpu():
    hidden = torch.zeros(3, 8, dtype=torch.bfloat16)
    weights = torch.tensor(
        [
            [0.2, 0.8],
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=torch.bfloat16,
    )
    indices = torch.tensor(
        [
            [2, 1],
            [0, -1],
            [2, 0],
        ],
        dtype=torch.int64,
    )

    plan = dsv4_kernel.build_moe_v2_execution_plan(
        hidden,
        weights,
        indices,
        num_experts=3,
        block_size_m=2,
    )

    assert plan.tokens == 3
    assert plan.hidden == 8
    assert plan.num_experts == 3
    assert plan.reduce_once is True
    assert plan.route_weights.dtype is torch.float32
    assert plan.route_weights.shape == (6,)
    assert torch.allclose(plan.route_weights.cpu(), weights.float().reshape(-1))
    assert plan.route_plan.sorted_route_ids.tolist() == [2, 5, 1, 6, 0, 4]

    workspace = dsv4_kernel.DSV4MoEWorkspace()
    first = workspace.tensor("tmp", (2, 4), torch.float32, torch.device("cpu"), zero=True)
    first.fill_(3.0)
    second = workspace.tensor("tmp", (1, 8), torch.float32, torch.device("cpu"))
    assert second.data_ptr() == first.data_ptr()
    assert second.shape == (1, 8)
    larger = workspace.tensor("tmp", (4, 4), torch.float32, torch.device("cpu"))
    assert larger.numel() == 16


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_dsv4_swiglu_bf16_output_matches_fp32_then_cast():
    from minisgl.kernel.triton import deepseek_v4 as triton_dsv4

    device = torch.device("cuda")
    torch.manual_seed(113)
    gate = torch.randn(17, 129, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    weights = torch.rand(17, 1, device=device, dtype=torch.float32)

    expected = triton_dsv4.silu_and_mul_clamp(
        gate,
        up,
        swiglu_limit=2.5,
        weights=weights,
    )
    actual = triton_dsv4.silu_and_mul_clamp_bf16(
        gate,
        up,
        swiglu_limit=2.5,
        weights=weights,
    )
    torch.cuda.synchronize()

    assert expected is not None
    assert actual is not None
    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected.to(torch.bfloat16))


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_dsv4_moe_route_plan_triton_matches_torch_fallback(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)
    indices = torch.tensor(
        [
            [2, 1],
            [0, -1],
            [2, 0],
            [3, 1],
            [9, 2],
        ],
        dtype=torch.int64,
    )
    expected = dsv4_kernel.build_moe_route_plan(indices, num_experts=4, block_size_m=2)

    monkeypatch.setenv("MINISGL_DSV4_SM80_MOE_ROUTE", "1")
    actual = dsv4_kernel.build_moe_route_plan(
        indices.cuda(),
        num_experts=4,
        block_size_m=2,
    )
    torch.cuda.synchronize()

    actual_padded = int(actual.num_tokens_post_padded.item())
    assert actual.route_count == expected.route_count
    assert actual.topk == expected.topk
    assert actual.block_size_m == expected.block_size_m
    assert actual_padded == int(expected.num_tokens_post_padded.item())
    assert (
        actual.sorted_route_ids[:actual_padded].cpu().tolist() == expected.sorted_route_ids.tolist()
    )
    assert actual.expert_ids[: actual_padded // actual.block_size_m].cpu().tolist() == (
        expected.expert_ids.tolist()
    )


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
def test_dsv4_sparse_attention_two_source_bf16_matches_reference(monkeypatch):
    device = torch.device("cuda")
    torch.manual_seed(17)
    monkeypatch.setenv("MINISGL_DSV4_SM80_SPARSE_ATTN_BF16", "1")

    q = (torch.randn(4, 3, 512, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
    swa_cache = (
        torch.randn(96, 512, device=device, dtype=torch.bfloat16) * 0.25 + 0.5
    ).contiguous()
    compressed_cache = (
        torch.randn(64, 512, device=device, dtype=torch.bfloat16) * 0.25 - 0.75
    ).contiguous()
    compressed_indices = torch.tensor(
        [
            [1, 2, 2, -1, -1],
            [-1, -1, -1, -1, -1],
            [4, 7, 9, 11, 13],
            [3, -1, -1, -1, -1],
        ],
        device=device,
        dtype=torch.int32,
    )
    compressed_lengths = torch.tensor([4, 0, 5, 1], device=device, dtype=torch.int32)
    swa_indices = torch.tensor(
        [
            [5, 6, 7, -1, -1, -1],
            [8, 9, 10, 11, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [3, 3, 4, 5, 6, 7],
        ],
        device=device,
        dtype=torch.int32,
    )
    swa_lengths = torch.tensor([3, 5, 0, 6], device=device, dtype=torch.int32)
    attn_sink = torch.randn(3, device=device, dtype=torch.float32)
    softmax_scale = 512**-0.5

    expected = _manual_two_source_sparse_attention(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=compressed_lengths,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )
    actual = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=compressed_lengths,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )
    assert actual is not None
    torch.cuda.synchronize()
    assert actual.dtype is q.dtype
    assert torch.allclose(actual, expected, atol=6e-2, rtol=6e-2)

    expected_no_sink = _manual_two_source_sparse_attention(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=compressed_lengths,
        softmax_scale=softmax_scale,
        attn_sink=None,
    )
    actual_no_sink = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=compressed_lengths,
        softmax_scale=softmax_scale,
        attn_sink=None,
    )
    assert actual_no_sink is not None
    torch.cuda.synchronize()
    assert torch.allclose(actual_no_sink, expected_no_sink, atol=6e-2, rtol=6e-2)

    expected_swa_only = _manual_two_source_sparse_attention(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=None,
        compressed_indices=None,
        compressed_lengths=None,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )
    actual_swa_only = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )
    assert actual_swa_only is not None
    torch.cuda.synchronize()
    assert torch.allclose(actual_swa_only, expected_swa_only, atol=6e-2, rtol=6e-2)
    assert torch.equal(actual_swa_only[2], torch.zeros_like(actual_swa_only[2]))


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
@pytest.mark.parametrize(
    "case_name,compressed_width,swa_width,compressed_lengths,swa_lengths,attn_sink",
    [
        ("swa_only", 0, 8, [0, 0, 0, 0], [1, 3, 0, 8], True),
        ("c4", 16, 8, [4, 0, 16, 2], [3, 0, 5, 8], True),
        ("c128", 6, 8, [0, 1, 4, 6], [2, 0, 8, 1], False),
        ("empty", 4, 5, [0, 0, 0, 0], [0, 0, 0, 0], True),
        ("short_history", 4, 8, [0, 1, 0, 2], [1, 2, 3, 4], False),
        ("mixed_valid_lengths_split", 96, 96, [0, 7, 64, 96], [0, 9, 41, 96], True),
    ],
)
def test_dsv4_sparse_attention_splitk_bf16_matches_legacy_cases(
    monkeypatch,
    case_name,
    compressed_width,
    swa_width,
    compressed_lengths,
    swa_lengths,
    attn_sink,
):
    del case_name
    device = torch.device("cuda")
    torch.manual_seed(395 + compressed_width + swa_width)
    monkeypatch.setenv("MINISGL_DSV4_SM80_SPARSE_ATTN_BF16", "1")
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_SPARSE_SPLITK_BF16_TOGGLE, "1")

    rows = 4
    heads = 3
    q = (torch.randn(rows, heads, 512, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    swa_cache = (torch.randn(128, 512, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    compressed_cache = (
        torch.randn(160, 512, device=device, dtype=torch.bfloat16) * 0.2 + 0.1
    ).contiguous()

    def make_indices(width: int, stride: int, cache_rows: int) -> torch.Tensor:
        if width <= 0:
            return torch.empty((rows, 0), device=device, dtype=torch.int32)
        base = torch.arange(width, device=device, dtype=torch.int32)
        out = torch.empty((rows, width), device=device, dtype=torch.int32)
        for row in range(rows):
            out[row] = (base + row * stride) % cache_rows
        if width >= 4:
            out[0, min(width - 1, 3) :] = -1
        return out

    swa_indices = make_indices(swa_width, 11, swa_cache.shape[0])
    compressed_indices = make_indices(compressed_width, 17, compressed_cache.shape[0])
    swa_lengths_t = torch.tensor(swa_lengths, device=device, dtype=torch.int32)
    compressed_lengths_t = torch.tensor(compressed_lengths, device=device, dtype=torch.int32)
    sink = torch.randn(heads, device=device, dtype=torch.float32) if attn_sink else None
    scale = 512**-0.5

    compressed_kwargs = {}
    if compressed_width > 0:
        compressed_kwargs = {
            "compressed_cache": compressed_cache,
            "compressed_indices": compressed_indices,
            "compressed_lengths": compressed_lengths_t,
        }

    expected = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
        q,
        swa_cache,
        swa_indices,
        swa_lengths_t,
        **compressed_kwargs,
        softmax_scale=scale,
        attn_sink=sink,
    )
    actual = dsv4_kernel.dsv4_sparse_attention_two_source_splitk_bf16(
        q,
        swa_cache,
        swa_indices,
        swa_lengths_t,
        **compressed_kwargs,
        softmax_scale=scale,
        attn_sink=sink,
    )

    assert expected is not None
    assert actual is not None
    torch.cuda.synchronize()
    assert actual.dtype is q.dtype
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=7e-2, rtol=7e-2)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_dsv4_sparse_attention_backend_reads_compressed_cache(monkeypatch):
    device = torch.device("cuda")
    torch.manual_seed(23)
    monkeypatch.setenv("MINISGL_DSV4_SM80_SPARSE_ATTN_BF16", "1")

    q = (torch.randn(2, 2, 512, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
    swa_cache = (
        torch.randn(16, 512, device=device, dtype=torch.bfloat16) * 0.2 + 0.25
    ).contiguous()
    compressed_cache = (
        torch.randn(16, 512, device=device, dtype=torch.bfloat16) * 0.2 - 1.0
    ).contiguous()
    swa_indices = torch.tensor(
        [[0, 1, -1], [2, 3, 4]],
        device=device,
        dtype=torch.int32,
    )
    swa_lengths = torch.tensor([2, 3], device=device, dtype=torch.int32)
    compressed_indices = torch.tensor(
        [[5, 6, -1, -1], [7, -1, -1, -1]],
        device=device,
        dtype=torch.int32,
    )
    softmax_scale = 512**-0.5
    attn_sink = torch.randn(2, device=device, dtype=torch.float32)

    empty = torch.empty(0, device=device, dtype=torch.int32)
    meta = dsv4_attention.DSV4CoreAttentionMetadata(
        raw_out_loc=empty,
        page_table=empty,
        cu_seqlens_q=empty,
        seq_lens=empty,
        req_seq_lens=empty,
        extend_lens=empty,
        positions=empty,
        req_table_indices=empty,
        max_seqlen_q=0,
        max_seqlen_k=0,
        swa_page_indices=swa_indices,
        swa_topk_lengths=swa_lengths,
        c4_out_loc=None,
        c128_out_loc=None,
        c4_indexer_out_loc=None,
        c4_topk_lengths_raw=empty,
        c4_topk_lengths_clamp1=empty,
        c4_sparse_topk_lengths=empty,
        c4_sparse_raw_indices=empty,
        c4_sparse_page_indices=compressed_indices,
        c4_sparse_full_indices=empty,
        c128_topk_lengths_clamp1=empty,
        c128_raw_indices=empty,
        c128_page_indices=empty.reshape(0, 0),
        c128_full_indices=empty,
    )

    class FakeKVCache:
        def swa_cache(self, layer_id: int) -> torch.Tensor:
            assert layer_id == 0
            return swa_cache

        def c4_cache(self, layer_id: int) -> torch.Tensor:
            assert layer_id == 0
            return compressed_cache

        def c128_cache(self, layer_id: int) -> torch.Tensor:
            raise AssertionError("ratio 4 path must not read c128 cache")

    class FakeBackend:
        pass

    fake_backend = FakeBackend()
    fake_backend.kvcache = FakeKVCache()
    fake_backend.softmax_scale = softmax_scale

    actual = dsv4_attention.DSV4AttentionBackend._sparse_attention_two_source(
        fake_backend,
        q,
        0,
        meta,
        4,
        attn_sink,
    )
    expected = _manual_two_source_sparse_attention(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=(compressed_indices >= 0).sum(dim=-1).to(torch.int32),
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )
    swa_only = _manual_two_source_sparse_attention(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=None,
        compressed_indices=None,
        compressed_lengths=None,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )

    assert actual is not None
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, atol=6e-2, rtol=6e-2)
    assert not torch.allclose(actual, swa_only, atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_dsv4_sm80_v0_bf16_bundle_kernels_match_fallbacks(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)
    device = torch.device("cuda")
    torch.manual_seed(37)

    gate = torch.randn(9, 129, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    weights = torch.rand(9, 1, device=device, dtype=torch.float32)
    expected_swiglu = dsv4_kernel.silu_and_mul_clamp_fallback(
        gate,
        up,
        swiglu_limit=2.0,
        weights=weights,
    )

    positions = torch.arange(6, device=device, dtype=torch.int64)
    rope_x = torch.randn(6, 2, 16, device=device, dtype=torch.float32)
    expected_rope = dsv4_kernel.apply_rotary_tail(
        rope_x.clone(),
        positions,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    q = torch.randn(6, 2, 16, device=device, dtype=torch.bfloat16)
    expected_q = dsv4_kernel.q_norm_rope_fallback(
        q.clone(),
        positions,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )

    kv = torch.randn(5, 16, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(16, device=device, dtype=torch.bfloat16)
    k_positions = torch.tensor([0, 127, 255, 511, 777], device=device, dtype=torch.int64)
    k_loc = torch.tensor([3, 7, 11, 13, 19], device=device, dtype=torch.int32)
    expected_k_cache = torch.zeros(32, 16, device=device, dtype=torch.bfloat16)
    expected_k = dsv4_kernel.k_norm_rope_cache_fallback(
        kv.clone(),
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

    class FakeCompressedCache:
        def __init__(self) -> None:
            self.cache = torch.zeros(32, 16, device=device, dtype=torch.bfloat16)

        def component_cache(self, layer_id: int) -> torch.Tensor:
            assert layer_id == 0
            return self.cache

        def store_compressed(self, layer_id: int, kv: torch.Tensor, out_loc: torch.Tensor) -> None:
            assert layer_id == 0
            valid = out_loc >= 0
            self.cache[out_loc[valid].long()] = kv.reshape(-1, 16)[valid].to(self.cache.dtype)

    compressed = torch.randn(5, 16, device=device, dtype=torch.bfloat16)
    compressed_weight = torch.randn(16, device=device, dtype=torch.bfloat16)
    compressed_positions = torch.tensor([3, 7, 11, 15, 19], device=device, dtype=torch.int64)
    compressed_loc = torch.tensor([4, 8, 12, -1, 20], device=device, dtype=torch.int32)
    expected_compressed_cache = FakeCompressedCache()
    expected_compressed = compressed.clone()
    dsv4_kernel.compress_norm_rope_store_fallback(
        expected_compressed_cache,
        0,
        expected_compressed,
        compressed_loc,
        positions=compressed_positions,
        norm_weight=compressed_weight,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )

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

    compress_x = torch.randn(12, 4, device=device, dtype=torch.bfloat16)
    ape = torch.randn(4, 8, device=device, dtype=torch.float32)
    compress_positions = torch.arange(12, device=device, dtype=torch.int64)
    expected_compress = dsv4_kernel.compress_forward_fallback(
        compress_x,
        compress_positions,
        ratio=4,
        head_dim=4,
        overlap=True,
        ape=ape,
        wkv_gate=FakeWkvGate(),
        norm=IdentityNorm(),
    )

    topk_scores = torch.randn(3, 1024, device=device, dtype=torch.float32)
    topk_seq_lens = torch.tensor([16, 900, 1024], device=device, dtype=torch.int32)
    topk_page_table = torch.arange(3 * 16, device=device, dtype=torch.int32).reshape(3, 16) + 100
    expected_topk = dsv4_kernel.topk_transform_512_full_fallback(
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )

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
    expected_attn = dsv4_kernel.paged_mqa_attention_fallback(
        q_attn,
        cache_attn,
        attn_contexts,
        softmax_scale=0.125,
        attn_sink=attn_sink,
    )

    q_indexer = torch.randn(3, 4, 128, device=device, dtype=torch.bfloat16)
    indexer_cache = torch.randn(192, 128, device=device, dtype=torch.bfloat16)
    indexer_weights = torch.randn(3, 4, device=device, dtype=torch.float32)
    indexer_seq_lens = torch.tensor([16, 64, 97], device=device, dtype=torch.int32)
    indexer_page_table = torch.tensor([[0, 1], [0, 1], [0, 1]], device=device, dtype=torch.int32)
    expected_indexer_logits = dsv4_kernel.indexer_bf16_logits_fallback(
        q_indexer,
        indexer_cache,
        indexer_seq_lens,
        indexer_page_table,
        page_size=64,
        weights=indexer_weights,
    )

    q_sparse = torch.randn(2, 2, 512, device=device, dtype=torch.bfloat16)
    swa_cache = torch.randn(32, 512, device=device, dtype=torch.bfloat16)
    compressed_cache = torch.randn(32, 512, device=device, dtype=torch.bfloat16)
    swa_indices = torch.tensor([[0, 1, 2, -1], [3, 4, 4, 5]], device=device, dtype=torch.int32)
    compressed_indices = torch.tensor(
        [[6, 7, -1, -1], [8, 9, 10, -1]],
        device=device,
        dtype=torch.int32,
    )
    swa_lengths = torch.tensor([3, 4], device=device, dtype=torch.int32)
    compressed_lengths = (compressed_indices >= 0).sum(dim=-1).to(torch.int32)
    sparse_sink = torch.randn(2, device=device, dtype=torch.float32)
    sparse_scale = 512**-0.5
    expected_sparse = _manual_two_source_sparse_attention(
        q_sparse,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=compressed_lengths,
        softmax_scale=sparse_scale,
        attn_sink=sparse_sink,
    )

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_V0_BF16_TOGGLE, "1")
    assert all(dsv4_kernel.dsv4_env_flag(name) for name in dsv4_kernel.DSV4_SM80_V0_BF16_WHITELIST)
    assert not any(
        dsv4_kernel.dsv4_env_flag(name) for name in dsv4_kernel.DSV4_SM80_EXPERIMENTAL_TOGGLES
    )

    actual_swiglu = dsv4_kernel.silu_and_mul_clamp_fallback(
        gate,
        up,
        swiglu_limit=2.0,
        weights=weights,
    )
    actual_rope = dsv4_kernel.apply_rotary_tail(
        rope_x.clone(),
        positions,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    actual_q = dsv4_kernel.q_norm_rope_fallback(
        q.clone(),
        positions,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    actual_k_cache = torch.zeros_like(expected_k_cache)
    actual_k = dsv4_kernel.k_norm_rope_cache_fallback(
        kv.clone(),
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
    actual_compressed_cache = FakeCompressedCache()
    actual_compressed = compressed.clone()
    dsv4_kernel.compress_norm_rope_store_fallback(
        actual_compressed_cache,
        0,
        actual_compressed,
        compressed_loc,
        positions=compressed_positions,
        norm_weight=compressed_weight,
        rms_norm_eps=1e-6,
        rotary_dim=8,
        base=10000.0,
        original_seq_len=4096,
        factor=2.0,
    )
    actual_compress = dsv4_kernel.compress_forward_fallback(
        compress_x,
        compress_positions,
        ratio=4,
        head_dim=4,
        overlap=True,
        ape=ape,
        wkv_gate=FakeWkvGate(),
        norm=IdentityNorm(),
    )
    actual_topk = dsv4_kernel.topk_transform_512_full_fallback(
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )
    actual_attn = dsv4_kernel.paged_mqa_attention_fallback(
        q_attn,
        cache_attn,
        attn_metadata,
        softmax_scale=0.125,
        attn_sink=attn_sink,
    )
    actual_indexer_logits = dsv4_kernel.indexer_bf16_logits_fallback(
        q_indexer,
        indexer_cache,
        indexer_seq_lens,
        indexer_page_table,
        page_size=64,
        weights=indexer_weights,
    )
    actual_sparse = dsv4_kernel.dsv4_sparse_attention_two_source_bf16(
        q_sparse,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=compressed_lengths,
        softmax_scale=sparse_scale,
        attn_sink=sparse_sink,
    )

    torch.cuda.synchronize()
    assert torch.allclose(actual_swiglu, expected_swiglu, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_rope, expected_rope, atol=1e-4, rtol=1e-4)
    assert torch.allclose(actual_q, expected_q, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_k, expected_k, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_k_cache, expected_k_cache, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_compressed, expected_compressed, atol=2e-2, rtol=2e-2)
    assert torch.allclose(
        actual_compressed_cache.cache,
        expected_compressed_cache.cache,
        atol=2e-2,
        rtol=2e-2,
    )
    assert torch.allclose(actual_compress, expected_compress, atol=1e-4, rtol=1e-4)
    _assert_full_topk_transform(
        actual_topk,
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )
    for row in range(topk_scores.shape[0]):
        assert sorted(actual_topk.raw_indices[row].cpu().tolist()) == sorted(
            expected_topk.raw_indices[row].cpu().tolist()
        )
    assert torch.allclose(actual_attn, expected_attn, atol=3e-2, rtol=3e-2)
    assert torch.allclose(actual_indexer_logits, expected_indexer_logits, atol=3e-2, rtol=3e-2)
    assert actual_sparse is not None
    assert torch.allclose(actual_sparse, expected_sparse, atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not _has_sm80_cuda(), reason="requires an sm80 CUDA device")
def test_dsv4_sm80_opt_in_kernels_match_fallbacks(monkeypatch):
    _clear_dsv4_sm80_env(monkeypatch)

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

    topk_scores = torch.randn(3, 1024, device=device, dtype=torch.float32)
    topk_seq_lens = torch.tensor([16, 900, 1024], device=device, dtype=torch.int32)
    topk_page_table = torch.arange(3 * 16, device=device, dtype=torch.int32).reshape(3, 16) + 100
    expected_full_topk = dsv4_kernel.topk_transform_512_full_fallback(
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )
    assert expected_full_topk.backend == "torch"
    monkeypatch.setenv("MINISGL_DSV4_SM80_TOPK", "1")
    actual_full_topk = dsv4_kernel.topk_transform_512_full_fallback(
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )
    torch.cuda.synchronize()
    assert actual_full_topk.backend in {"torch", "local_cuda_v1"}
    _assert_full_topk_transform(
        actual_full_topk,
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )
    for row in range(topk_scores.shape[0]):
        assert sorted(actual_full_topk.raw_indices[row].cpu().tolist()) == sorted(
            expected_full_topk.raw_indices[row].cpu().tolist()
        )
    monkeypatch.delenv("MINISGL_DSV4_SM80_TOPK", raising=False)

    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE, "1")
    actual_global_topk = dsv4_kernel.topk_transform_512_full_fallback(
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )
    torch.cuda.synchronize()
    assert actual_global_topk.backend == "local_cuda_global_topk_lens"
    assert actual_global_topk.topk_lens is not None
    assert torch.equal(
        actual_global_topk.topk_lens.cpu(),
        torch.tensor([16, 512, 512], dtype=torch.int32),
    )
    _assert_full_topk_transform(
        actual_global_topk,
        topk_scores,
        topk_seq_lens,
        topk_page_table,
        page_size=64,
        width=512,
        ratio=4,
    )
    for row in range(topk_scores.shape[0]):
        assert sorted(actual_global_topk.raw_indices[row].cpu().tolist()) == sorted(
            expected_full_topk.raw_indices[row].cpu().tolist()
        )
    monkeypatch.delenv(dsv4_kernel.DSV4_SM80_GLOBAL_TOPK_LENS_TOGGLE, raising=False)

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

    q_indexer = torch.randn(3, 4, 128, device=device, dtype=torch.bfloat16)
    indexer_cache = torch.randn(192, 128, device=device, dtype=torch.bfloat16)
    indexer_weights = torch.randn(3, 4, device=device, dtype=torch.float32)
    indexer_seq_lens = torch.tensor([16, 64, 97], device=device, dtype=torch.int32)
    indexer_page_table = torch.tensor([[0, 1], [0, 1], [0, 1]], device=device, dtype=torch.int32)
    expected_indexer_logits = dsv4_kernel.indexer_bf16_logits_fallback(
        q_indexer,
        indexer_cache,
        indexer_seq_lens,
        indexer_page_table,
        page_size=64,
        weights=indexer_weights,
    )
    monkeypatch.setenv("MINISGL_DSV4_SM80_INDEXER_BF16", "1")
    actual_indexer_logits = dsv4_kernel.indexer_bf16_logits_fallback(
        q_indexer,
        indexer_cache,
        indexer_seq_lens,
        indexer_page_table,
        page_size=64,
        weights=indexer_weights,
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual_indexer_logits, expected_indexer_logits, atol=3e-2, rtol=3e-2)
    monkeypatch.delenv("MINISGL_DSV4_SM80_INDEXER_BF16", raising=False)

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

    linear_x = torch.randn(6, 128, device=device, dtype=torch.bfloat16)
    linear_weight = torch.randn(11, 128, device=device, dtype=torch.bfloat16)
    expected_linear = dsv4_kernel.linear_bf16_fp32_fallback(linear_x, linear_weight)
    monkeypatch.setenv(dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE, "1")
    actual_linear = dsv4_kernel.linear_bf16_fp32_fallback(linear_x, linear_weight)
    torch.cuda.synchronize()
    assert actual_linear.dtype is torch.float32
    assert torch.allclose(actual_linear, expected_linear, atol=2e-2, rtol=2e-2)
    monkeypatch.delenv(dsv4_kernel.DSV4_LINEAR_BF16_FP32_TOGGLE, raising=False)

    x_linear = torch.randn(5, 128, device=device, dtype=torch.bfloat16)
    fp8_weight = (
        torch.randn(96, 128, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
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
    wo_weight = (
        torch.randn(
            2 * wo_rank,
            wo_o.shape[-1],
            device=device,
            dtype=torch.float32,
        )
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
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
    fp4_scale = torch.rand(96, 4, device=device, dtype=torch.float32).to(dsv4_kernel.e8m0_dtype())
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
    from minisgl.kernel.triton import deepseek_v4 as triton_dsv4

    plan = dsv4_kernel.build_moe_route_plan(
        moe_indices,
        num_experts=num_experts,
        block_size_m=16,
    )
    fused_routed = triton_dsv4.grouped_fp4_moe_fused_compute(
        moe_x,
        moe_weights.reshape(-1).contiguous(),
        w13_weight,
        w13_scale,
        w2_weight,
        w2_scale,
        plan.sorted_route_ids,
        plan.expert_ids,
        plan.num_tokens_post_padded,
        route_count=plan.route_count,
        topk=plan.topk,
        block_size_m=plan.block_size_m,
        swiglu_limit=2.5,
    )
    assert fused_routed is not None
    fused_moe = fused_routed.view(num_tokens, topk, hidden).sum(dim=1)
    assert torch.allclose(fused_moe, expected_moe, atol=8e-2, rtol=8e-2)

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

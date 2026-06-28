from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from minisgl.kernel import deepseek_v4 as dsv4_kernel
import minisgl.attention.deepseek_v4 as dsv4_attention
import minisgl.models.deepseek_v4 as dsv4_model


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
    assert out.shape == q.shape
    assert out.dtype is q.dtype
    assert torch.isfinite(out).all()

    padded = dsv4_kernel.topk_transform_512_fallback(torch.tensor([[1, 2]], dtype=torch.int32))
    assert padded.shape == (1, 512)
    assert padded[0, :2].tolist() == [1, 2]
    assert padded[0, 2:].eq(-1).all()


def test_dsv4_model_and_attention_do_not_import_optional_kernels_directly():
    model_source = inspect.getsource(dsv4_model)
    attention_source = inspect.getsource(dsv4_attention)

    assert "def _quantized_linear_ref" not in model_source
    assert "def _apply_rotary_tail" not in model_source
    for source in (model_source, attention_source):
        assert "import sgl_kernel" not in source
        assert "import flashinfer" not in source
        assert "import deep_gemm" not in source

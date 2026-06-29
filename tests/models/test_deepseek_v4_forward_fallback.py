from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

import minisgl.core as core
import minisgl.distributed.info as dist_info
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.models.deepseek_v4 import DSV4MoEGate
from minisgl.models.register import get_model_class


def _tiny_dsv4_config() -> ModelConfig:
    return ModelConfig(
        num_layers=1,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=4,
        hidden_size=8,
        vocab_size=16,
        intermediate_size=0,
        rms_norm_eps=1e-6,
        rotary_config=RotaryConfig(4, 2, 32, 10000.0, None),
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
        norm_topk_prob=True,
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        q_lora_rank=4,
        o_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        window_size=4,
        compress_ratios=[0],
        index_head_dim=2,
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


def _reset_globals() -> None:
    core._GLOBAL_CTX = None
    dist_info._TP_INFO = None
    set_tp_info(0, 1)



def _install_dsv4_context(cfg: ModelConfig, *, max_len: int) -> Context:
    ctx = Context(page_size=1)
    ctx.kv_cache = create_kvcache_pool(
        cfg,
        num_pages=max_len + 8,
        page_size=1,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    ctx.page_table = torch.arange(max_len + 8, dtype=torch.int32).unsqueeze(0)
    core.set_global_ctx(ctx)
    ctx.attn_backend = create_attention_backend("dsv4", cfg)
    return ctx


def _fill_forward_weights(model) -> None:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if tensor.numel() == 0:
                continue
            if "weight_scale_inv" in name:
                tensor.fill_(1.0)
            elif name.endswith("attn_sink") or "e_score_correction_bias" in name:
                tensor.zero_()
            elif name.endswith("_base") or name.endswith("_scale"):
                tensor.zero_()
            elif "norm.weight" in name:
                tensor.fill_(1.0)
            elif "hc_" in name and name.endswith("_fn"):
                tensor.zero_()
            elif tensor.dtype is torch.int8:
                tensor.fill_(0x11)
            elif fp8 is not None and tensor.dtype is fp8:
                tensor.fill_(0.125)
            elif e8m0 is not None and tensor.dtype is e8m0:
                tensor.fill_(1.0)
            elif tensor.is_floating_point():
                values = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
                tensor.copy_(((values % 13) - 6).to(tensor.dtype) / 64)
            else:
                tensor.zero_()


def test_deepseek_v4_small_prefill_forward_fallback_reaches_logits():
    _reset_globals()
    cfg = _tiny_dsv4_config()
    model = get_model_class(cfg.architectures[0], cfg)
    _fill_forward_weights(model)

    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    ctx = _install_dsv4_context(cfg, max_len=input_ids.numel())
    req = Req(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=None,  # type: ignore[arg-type]
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = input_ids
    batch.positions = torch.arange(input_ids.numel(), dtype=torch.int32)
    batch.out_loc = torch.arange(input_ids.numel(), dtype=torch.int32)
    ctx.attn_backend.prepare_metadata(batch)

    with ctx.forward_batch(batch):
        logits = model.forward()

    assert logits.shape == (1, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_deepseek_v4_ratio4_prefill_forward_fallback_reaches_logits():
    _reset_globals()
    cfg = replace(
        _tiny_dsv4_config(),
        compress_ratios=[4],
        compress_rope_theta=10000.0,
        index_head_dim=4,
    )
    model = get_model_class(cfg.architectures[0], cfg)
    _fill_forward_weights(model)

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    ctx = _install_dsv4_context(cfg, max_len=input_ids.numel())
    req = Req(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=None,  # type: ignore[arg-type]
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = input_ids
    batch.positions = torch.arange(input_ids.numel(), dtype=torch.int32)
    batch.out_loc = torch.arange(input_ids.numel(), dtype=torch.int32)
    ctx.attn_backend.prepare_metadata(batch)

    with ctx.forward_batch(batch):
        logits = model.forward()

    assert logits.shape == (1, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_deepseek_v4_ratio4_prefill_forward_with_indexer_bf16_toggle(monkeypatch):
    _reset_globals()
    monkeypatch.setenv("MINISGL_DSV4_SM80_INDEXER_BF16", "1")
    cfg = replace(
        _tiny_dsv4_config(),
        compress_ratios=[4],
        compress_rope_theta=10000.0,
        index_head_dim=4,
    )
    model = get_model_class(cfg.architectures[0], cfg)
    _fill_forward_weights(model)

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    ctx = _install_dsv4_context(cfg, max_len=input_ids.numel())
    req = Req(
        input_ids=input_ids,
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=0,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=None,  # type: ignore[arg-type]
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids = input_ids
    batch.positions = torch.arange(input_ids.numel(), dtype=torch.int32)
    batch.out_loc = torch.arange(input_ids.numel(), dtype=torch.int32)
    ctx.attn_backend.prepare_metadata(batch)

    with ctx.forward_batch(batch):
        logits = model.forward()

    assert logits.shape == (1, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert batch.attn_metadata.core_metadata.c4_sparse_raw_indices[3, 0].item() == 0


def test_deepseek_v4_moe_gate_matches_sqrtsoftplus_oracle():
    _reset_globals()
    cfg = _tiny_dsv4_config()
    gate = DSV4MoEGate(cfg, has_correction_bias=True)
    gate.weight = torch.tensor(
        [
            [0.20, -0.10, 0.00, 0.30, -0.20, 0.10, 0.05, -0.15],
            [-0.05, 0.25, 0.15, -0.20, 0.10, 0.00, -0.10, 0.20],
        ],
        dtype=torch.bfloat16,
    )
    gate.e_score_correction_bias = torch.tensor([0.0, 0.25], dtype=torch.float32)
    hidden = torch.tensor(
        [
            [0.2, -0.3, 0.4, 0.1, -0.2, 0.5, -0.1, 0.3],
            [-0.4, 0.1, 0.3, -0.2, 0.2, -0.1, 0.6, -0.5],
        ],
        dtype=torch.bfloat16,
    )

    weights, indices = gate.forward(
        hidden,
        input_ids=None,
        topk=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
    )

    raw = F.linear(hidden.float(), gate.weight.float())
    original = F.softplus(raw).sqrt()
    expected_indices = (original + gate.e_score_correction_bias).topk(1, dim=-1).indices
    expected_weights = original.gather(1, expected_indices)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * 1.5

    assert torch.equal(indices, expected_indices)
    assert torch.allclose(weights, expected_weights)

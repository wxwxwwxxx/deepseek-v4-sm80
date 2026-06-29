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
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.models.deepseek_v4 import DSV4FusedRoutedExperts, DSV4MoE, DSV4MoEGate
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


def _reset_globals(*, tp_rank: int = 0, tp_size: int = 1) -> None:
    core._GLOBAL_CTX = None
    dist_info._TP_INFO = None
    set_tp_info(tp_rank, tp_size)



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


def test_deepseek_v4_routed_experts_all_reduce_tp_sharded_output(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    cfg = _tiny_dsv4_config()
    experts = DSV4FusedRoutedExperts(cfg)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            self.calls.append(x.clone())
            return x + 10.0

    fake_comm = FakeComm()
    experts._comm = fake_comm

    def fake_expert_forward(local_idx, x, weights):
        del weights
        return torch.full_like(x, float(local_idx + 1), dtype=x.dtype)

    monkeypatch.setattr(experts, "_expert_forward", fake_expert_forward)
    monkeypatch.setattr(dsv4_kernel, "moe_route_dispatch_bf16_grouped", lambda *_, **__: None)

    hidden = torch.zeros(3, cfg.hidden_size, dtype=torch.bfloat16)
    weights = torch.ones(3, 1, dtype=torch.float32)
    indices = torch.tensor([[0], [1], [0]], dtype=torch.long)

    out = experts.forward(hidden, weights, indices)

    expected_local = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32).view(3, 1)
    expected = (expected_local + 10.0).expand_as(hidden).to(torch.bfloat16)
    assert len(fake_comm.calls) == 1
    assert fake_comm.calls[0].dtype is torch.float32
    assert torch.equal(out, expected)


def test_deepseek_v4_grouped_routed_experts_all_reduce_tp_sharded_output(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    cfg = _tiny_dsv4_config()
    experts = DSV4FusedRoutedExperts(cfg)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            self.calls.append(x.clone())
            return x + 20.0

    fake_comm = FakeComm()
    experts._comm = fake_comm

    hidden = torch.zeros(2, cfg.hidden_size, dtype=torch.bfloat16)
    weights = torch.ones(2, 1, dtype=torch.float32)
    indices = torch.tensor([[0], [1]], dtype=torch.long)
    grouped_local = torch.full_like(hidden, 3.0)
    monkeypatch.setattr(
        dsv4_kernel,
        "moe_route_dispatch_bf16_grouped",
        lambda *_, **__: grouped_local,
    )

    out = experts.forward(hidden, weights, indices)

    assert len(fake_comm.calls) == 1
    assert fake_comm.calls[0].dtype is torch.float32
    assert torch.equal(out, torch.full_like(hidden, 23.0))


def test_deepseek_v4_v1_moe_sums_routed_and_shared_before_one_all_reduce(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    cfg = _tiny_dsv4_config()
    moe = DSV4MoE(cfg, layer_id=0)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
            self.calls.append(x.clone())
            return x + 30.0

    fake_comm = FakeComm()
    moe._comm = fake_comm
    calls: list[tuple[str, bool]] = []

    def fake_gate_forward(*args, **kwargs):
        hidden = args[0]
        return (
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden.shape[0], 1, dtype=torch.long),
        )

    def fake_experts_forward(hidden, weights, indices, *, reduce=True):
        del weights, indices
        calls.append(("routed", reduce))
        return torch.full_like(hidden, 1.25)

    def fake_shared_forward(hidden, *, reduce=True):
        calls.append(("shared", reduce))
        return torch.full_like(hidden, 2.75)

    moe.gate.forward = fake_gate_forward
    moe.experts.forward = fake_experts_forward
    moe.shared_experts.forward = fake_shared_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert calls == [("routed", False), ("shared", False)]
    assert len(fake_comm.calls) == 1
    assert fake_comm.calls[0].dtype is torch.float32
    assert torch.equal(out, torch.full_like(hidden, 34.0))

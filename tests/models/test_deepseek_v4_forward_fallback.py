from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import minisgl.core as core
import minisgl.distributed.info as dist_info
import torch
import torch.nn.functional as F
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.kernel import dense_fp8_marlin
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kvcache import create_kvcache_pool
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.models.deepseek_v4 import (
    DSV4FusedRoutedExperts,
    DSV4MoE,
    DSV4MoEGate,
    DSV4SharedExperts,
)
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


def _clear_dsv4_sm80_env(monkeypatch) -> None:
    for name in dsv4_kernel.DSV4_SM80_KNOWN_TOGGLES:
        monkeypatch.delenv(name, raising=False)


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

        def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
            del label
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

        def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
            del label
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


def test_deepseek_v4_moe_v2_workspace_is_decode_sized(monkeypatch):
    _reset_globals()
    cfg = _tiny_dsv4_config()
    experts = DSV4FusedRoutedExperts(cfg)
    seen_workspaces: list[object | None] = []

    def fake_grouped_dispatch(hidden_states, *args, workspace=None, **kwargs):
        del args, kwargs
        seen_workspaces.append(workspace)
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(
        dsv4_kernel,
        "moe_route_dispatch_bf16_grouped",
        fake_grouped_dispatch,
    )

    small_hidden = torch.zeros(4, cfg.hidden_size, dtype=torch.bfloat16)
    small_weights = torch.ones(4, 1, dtype=torch.float32)
    small_indices = torch.zeros(4, 1, dtype=torch.long)
    small_plan = dsv4_kernel.build_moe_v2_execution_plan(
        small_hidden,
        small_weights,
        small_indices,
        num_experts=cfg.n_routed_experts,
    )
    experts.forward(small_hidden, small_weights, small_indices, moe_plan=small_plan)

    large_tokens = dsv4_kernel.DSV4_SM80_MOE_V2_WORKSPACE_MAX_ROUTES + 1
    large_hidden = torch.zeros(large_tokens, cfg.hidden_size, dtype=torch.bfloat16)
    large_weights = torch.ones(large_tokens, 1, dtype=torch.float32)
    large_indices = torch.zeros(large_tokens, 1, dtype=torch.long)
    large_plan = dsv4_kernel.build_moe_v2_execution_plan(
        large_hidden,
        large_weights,
        large_indices,
        num_experts=cfg.n_routed_experts,
    )
    experts.forward(large_hidden, large_weights, large_indices, moe_plan=large_plan)

    assert seen_workspaces[0] is experts._moe_v2_workspace
    assert seen_workspaces[1] is None


def test_deepseek_v4_v1_moe_sums_routed_and_shared_before_one_all_reduce(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    monkeypatch.delenv(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE, raising=False)
    cfg = _tiny_dsv4_config()
    moe = DSV4MoE(cfg, layer_id=0)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
            del label
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


def test_deepseek_v4_v1_moe_reduce_once_bf16_opt_in(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    cfg = _tiny_dsv4_config()
    moe = DSV4MoE(cfg, layer_id=0)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
            del label
            self.calls.append(x.clone())
            return x + 30.0

    fake_comm = FakeComm()
    moe._comm = fake_comm

    def fake_gate_forward(*args, **kwargs):
        hidden = args[0]
        return (
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden.shape[0], 1, dtype=torch.long),
        )

    def fake_experts_forward(hidden, weights, indices, *, reduce=True):
        del weights, indices
        assert reduce is False
        return torch.full_like(hidden, 1.25)

    def fake_shared_forward(hidden, *, reduce=True):
        assert reduce is False
        return torch.full_like(hidden, 2.75)

    moe.gate.forward = fake_gate_forward
    moe.experts.forward = fake_experts_forward
    moe.shared_experts.forward = fake_shared_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE, "1")
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert len(fake_comm.calls) == 1
    assert fake_comm.calls[0].dtype is torch.bfloat16
    assert out.dtype is torch.bfloat16
    assert torch.equal(out, torch.full_like(hidden, 34.0))


def test_deepseek_v4_moe_v2_builds_execution_plan_before_reduce_once(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    monkeypatch.delenv(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE, raising=False)
    cfg = _tiny_dsv4_config()
    moe = DSV4MoE(cfg, layer_id=0)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[torch.Tensor] = []

        def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
            del label
            self.calls.append(x.clone())
            return x + 30.0

    fake_comm = FakeComm()
    moe._comm = fake_comm
    calls: list[tuple[str, bool]] = []
    seen_plans: list[dsv4_kernel.DSV4MoEExecutionPlan] = []

    def fake_gate_forward(*args, **kwargs):
        hidden = args[0]
        return (
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden.shape[0], 1, dtype=torch.long),
        )

    def fake_experts_forward(hidden, weights, indices, *, reduce=True, moe_plan=None):
        del weights, indices
        calls.append(("routed", reduce))
        assert moe_plan is not None
        seen_plans.append(moe_plan)
        return torch.full_like(hidden, 1.25)

    def fake_shared_forward(hidden, *, reduce=True):
        calls.append(("shared", reduce))
        return torch.full_like(hidden, 2.75)

    moe.gate.forward = fake_gate_forward
    moe.experts.forward = fake_experts_forward
    moe.shared_experts.forward = fake_shared_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_V2_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert calls == [("routed", False), ("shared", False)]
    assert len(seen_plans) == 1
    assert seen_plans[0].tokens == 2
    assert seen_plans[0].hidden == cfg.hidden_size
    assert seen_plans[0].route_plan.route_count == 2
    assert len(fake_comm.calls) == 1
    assert fake_comm.calls[0].dtype is torch.float32
    assert torch.equal(out, torch.full_like(hidden, 34.0))


def test_deepseek_v4_vllm_runner_sums_routed_and_shared_before_late_reduce(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    monkeypatch.delenv(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE, raising=False)
    cfg = _tiny_dsv4_config()
    moe = DSV4MoE(cfg, layer_id=0)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, str | None]] = []

        def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
            self.calls.append((x.clone(), label))
            return x + 30.0

    fake_comm = FakeComm()
    moe._comm = fake_comm
    calls: list[tuple[str, bool]] = []
    seen_plans: list[dsv4_kernel.DSV4MoEExecutionPlan] = []

    def fake_gate_forward(*args, **kwargs):
        hidden = args[0]
        return (
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden.shape[0], 1, dtype=torch.long),
        )

    def fake_experts_forward(hidden, weights, indices, *, reduce=True, moe_plan=None):
        del weights, indices
        calls.append(("routed", reduce))
        assert moe_plan is not None
        seen_plans.append(moe_plan)
        return torch.full_like(hidden, 1.25)

    def fake_shared_forward(hidden, *, reduce=True):
        calls.append(("shared", reduce))
        return torch.full_like(hidden, 2.75)

    moe.gate.forward = fake_gate_forward
    moe.experts.forward = fake_experts_forward
    moe.shared_experts.forward = fake_shared_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert calls == [("routed", False), ("shared", False)]
    assert len(seen_plans) == 1
    assert seen_plans[0].tokens == 2
    assert seen_plans[0].route_plan.route_count == 2
    assert len(fake_comm.calls) == 1
    reduced, label = fake_comm.calls[0]
    assert reduced.dtype is torch.float32
    assert label == "dsv4.v1_moe_reduce_once_all_reduce"
    assert torch.equal(out, torch.full_like(hidden, 34.0))


def test_deepseek_v4_vllm_runner_reduce_once_bf16_opt_in(monkeypatch):
    _reset_globals(tp_rank=0, tp_size=2)
    cfg = _tiny_dsv4_config()
    moe = DSV4MoE(cfg, layer_id=0)

    class FakeComm:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, str | None]] = []

        def all_reduce(self, x: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
            self.calls.append((x.clone(), label))
            return x + 30.0

    fake_comm = FakeComm()
    moe._comm = fake_comm

    def fake_gate_forward(*args, **kwargs):
        hidden = args[0]
        return (
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden.shape[0], 1, dtype=torch.long),
        )

    def fake_experts_forward(hidden, weights, indices, *, reduce=True, moe_plan=None):
        del weights, indices
        assert reduce is False
        assert moe_plan is not None
        return torch.full_like(hidden, 1.25)

    def fake_shared_forward(hidden, *, reduce=True):
        assert reduce is False
        return torch.full_like(hidden, 2.75)

    moe.gate.forward = fake_gate_forward
    moe.experts.forward = fake_experts_forward
    moe.shared_experts.forward = fake_shared_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE, "1")
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_REDUCE_BF16_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert len(fake_comm.calls) == 1
    reduced, label = fake_comm.calls[0]
    assert reduced.dtype is torch.bfloat16
    assert label == "dsv4.v1_moe_reduce_once_all_reduce"
    assert out.dtype is torch.bfloat16
    assert torch.equal(out, torch.full_like(hidden, 34.0))


def test_deepseek_v4_vllm_runner_routed_only_path(monkeypatch):
    _reset_globals()
    cfg = replace(_tiny_dsv4_config(), n_shared_experts=0)
    moe = DSV4MoE(cfg, layer_id=0)

    def fake_gate_forward(*args, **kwargs):
        hidden = args[0]
        return (
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden.shape[0], 1, dtype=torch.long),
        )

    def fake_experts_forward(hidden, weights, indices, *, reduce=True, moe_plan=None):
        del weights, indices
        assert reduce is False
        assert moe_plan is not None
        return torch.full_like(hidden, 5.0)

    moe.gate.forward = fake_gate_forward
    moe.experts.forward = fake_experts_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert torch.equal(out, torch.full_like(hidden, 5.0))


def test_deepseek_v4_vllm_runner_shared_only_effect(monkeypatch):
    _reset_globals()
    cfg = _tiny_dsv4_config()
    moe = DSV4MoE(cfg, layer_id=0)

    def fake_gate_forward(*args, **kwargs):
        hidden = args[0]
        return (
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
            torch.zeros(hidden.shape[0], 1, dtype=torch.long),
        )

    def fake_experts_forward(hidden, weights, indices, *, reduce=True, moe_plan=None):
        del weights, indices
        assert reduce is False
        assert moe_plan is not None
        return torch.zeros_like(hidden)

    def fake_shared_forward(hidden, *, reduce=True):
        assert reduce is False
        return torch.full_like(hidden, 7.0)

    moe.gate.forward = fake_gate_forward
    moe.experts.forward = fake_experts_forward
    moe.shared_experts.forward = fake_shared_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert torch.equal(out, torch.full_like(hidden, 7.0))


def test_deepseek_v4_vllm_runner_hash_routing_uses_input_ids(monkeypatch):
    _reset_globals()
    cfg = replace(_tiny_dsv4_config(), n_hash_layers=1, n_shared_experts=0)
    moe = DSV4MoE(cfg, layer_id=0)
    moe.gate.weight.zero_()
    moe.topk.tid2eid.zero_()
    moe.topk.tid2eid[3, 0] = 1
    moe.topk.tid2eid[4, 0] = 0
    seen_indices: list[torch.Tensor] = []

    def fake_experts_forward(hidden, weights, indices, *, reduce=True, moe_plan=None):
        del weights
        assert reduce is False
        assert moe_plan is not None
        seen_indices.append(indices.clone())
        return torch.zeros_like(hidden)

    moe.experts.forward = fake_experts_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.tensor([[3], [4]], dtype=torch.long)
    moe.forward(hidden, input_ids)

    assert len(seen_indices) == 1
    assert seen_indices[0].tolist() == [[1], [0]]


def test_deepseek_v4_vllm_runner_correction_bias_routing(monkeypatch):
    _reset_globals()
    cfg = replace(_tiny_dsv4_config(), n_shared_experts=0)
    moe = DSV4MoE(cfg, layer_id=0)
    moe.gate.weight.zero_()
    moe.gate.e_score_correction_bias.copy_(torch.tensor([0.0, 4.0]))
    seen_indices: list[torch.Tensor] = []

    def fake_experts_forward(hidden, weights, indices, *, reduce=True, moe_plan=None):
        del weights
        assert reduce is False
        assert moe_plan is not None
        seen_indices.append(indices.clone())
        return torch.zeros_like(hidden)

    moe.experts.forward = fake_experts_forward
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_MOE_VLLM_RUNNER_TOGGLE, "1")

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    moe.forward(hidden, input_ids)

    assert len(seen_indices) == 1
    assert seen_indices[0].tolist() == [[1], [1]]


def test_shared_experts_bf16_weight_cache_matches_generic_path(monkeypatch):
    _reset_globals()
    _clear_dsv4_sm80_env(monkeypatch)
    cfg = _tiny_dsv4_config()
    shared = DSV4SharedExperts(cfg, layer_id=0)
    torch.manual_seed(766)

    with torch.no_grad():
        shared.gate_up_proj.weight.copy_(
            torch.randn_like(shared.gate_up_proj.weight.float())
            .clamp(-2, 2)
            .to(dsv4_kernel.fp8_dtype())
        )
        shared.down_proj.weight.copy_(
            torch.randn_like(shared.down_proj.weight.float())
            .clamp(-2, 2)
            .to(dsv4_kernel.fp8_dtype())
        )
        shared.gate_up_proj.weight_scale_inv.copy_(
            torch.ones_like(shared.gate_up_proj.weight_scale_inv.float()).to(
                dsv4_kernel.e8m0_dtype()
            )
        )
        shared.down_proj.weight_scale_inv.copy_(
            torch.ones_like(shared.down_proj.weight_scale_inv.float()).to(dsv4_kernel.e8m0_dtype())
        )

    hidden = torch.randn(3, cfg.hidden_size, dtype=torch.bfloat16)
    expected = shared.forward(hidden, reduce=False)

    assert shared.prepare_bf16_weight_cache() == []
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE, "1")
    reports = shared.prepare_bf16_weight_cache()
    actual = shared.forward(hidden, reduce=False)

    assert {report["owner"] for report in reports} == {
        "layer0.shared_experts.gate_up_proj",
        "layer0.shared_experts.down_proj",
    }
    assert (
        sum(int(report["bytes"]) for report in reports)
        == (shared.gate_up_proj.weight.numel() + shared.down_proj.weight.numel())
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_shared_experts_marlin_down_skips_bf16_down_cache_and_releases_original(
    monkeypatch,
):
    _reset_globals()
    _clear_dsv4_sm80_env(monkeypatch)
    cfg = _tiny_dsv4_config()
    shared = DSV4SharedExperts(cfg, layer_id=0)
    torch.manual_seed(774)

    with torch.no_grad():
        shared.gate_up_proj.weight.copy_(
            torch.randn_like(shared.gate_up_proj.weight.float())
            .clamp(-2, 2)
            .to(dsv4_kernel.fp8_dtype())
        )
        shared.down_proj.weight.copy_(
            torch.randn_like(shared.down_proj.weight.float())
            .clamp(-2, 2)
            .to(dsv4_kernel.fp8_dtype())
        )
        shared.gate_up_proj.weight_scale_inv.copy_(
            torch.ones_like(shared.gate_up_proj.weight_scale_inv.float()).to(
                dsv4_kernel.e8m0_dtype()
            )
        )
        shared.down_proj.weight_scale_inv.copy_(
            torch.ones_like(shared.down_proj.weight_scale_inv.float()).to(dsv4_kernel.e8m0_dtype())
        )

    def fake_prepare(weight, weight_scale_inv, *, owner_label):
        dequant = dsv4_kernel.dequant_fp8_weight(
            weight,
            weight_scale_inv,
            out_dtype=torch.bfloat16,
        ).contiguous()
        return SimpleNamespace(
            weight=dequant,
            weight_scale=torch.empty(0),
            workspace=torch.empty(0),
            size_n=weight.shape[0],
            size_k=weight.shape[1],
            prepared_weight_bytes=dequant.numel() * dequant.element_size(),
            prepared_scale_bytes=0,
            workspace_bytes=0,
            persistent_bytes=dequant.numel() * dequant.element_size(),
            original_weight_bytes=weight.numel() * weight.element_size(),
            original_scale_bytes=weight_scale_inv.numel() * weight_scale_inv.element_size(),
        )

    monkeypatch.setattr(dense_fp8_marlin, "prepare_dense_fp8_marlin_weight", fake_prepare)
    monkeypatch.setattr(
        dense_fp8_marlin,
        "apply_dense_fp8_marlin_linear",
        lambda x, prepared, **_: F.linear(x, prepared.weight),
    )
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE_TOGGLE, "1")
    monkeypatch.setenv(dsv4_kernel.DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION_TOGGLE, "1")

    bf16_reports = shared.prepare_bf16_weight_cache()
    marlin_report = shared.prepare_down_marlin_weight_cache()

    assert {report["owner"] for report in bf16_reports} == {
        "layer0.shared_experts.gate_up_proj",
    }
    assert marlin_report is not None
    assert marlin_report["owner"] == "layer0.shared_experts.down_proj"
    assert marlin_report["released_original"] is True
    assert {entry["attribute"] for entry in marlin_report["released"]} == {
        "weight",
        "weight_scale_inv",
    }
    assert not hasattr(shared.down_proj, "weight")
    assert not hasattr(shared.down_proj, "weight_scale_inv")
    assert not hasattr(shared.down_proj, shared._down_bf16_weight_cache_name)

    hidden = torch.randn(3, cfg.hidden_size, dtype=torch.bfloat16)
    actual = shared.forward(hidden, reduce=False)

    assert actual.shape == (3, cfg.hidden_size)
    assert actual.dtype is torch.bfloat16

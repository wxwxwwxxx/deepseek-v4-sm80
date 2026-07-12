from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import minisgl.core as core
import minisgl.distributed.info as dist_info
import pytest
import torch
import torch.nn.functional as F
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.dsv4_runtime import configure_dsv4_runtime
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kvcache import create_kvcache_pool
from minisgl.models.config import ModelConfig, RotaryConfig
from minisgl.models.deepseek_v4 import (
    DeepseekV4Model,
    DSV4Attention,
    DSV4FusedRoutedExperts,
    DSV4Linear,
    DSV4MoE,
    DSV4MoEGate,
    DSV4SharedExperts,
)
from minisgl.models.register import get_model_class


@pytest.fixture(autouse=True)
def _optimized_runtime_mode():
    configure_dsv4_runtime("optimized")
    yield
    configure_dsv4_runtime("optimized")


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


def test_fallback_fp8_linear_preserves_loaded_scale(monkeypatch):
    configure_dsv4_runtime("fallback")
    _reset_globals()
    linear = DSV4Linear(
        128,
        128,
        weight_dtype=dsv4_kernel.fp8_dtype(),
        scale_dtype=dsv4_kernel.e8m0_dtype(),
    )
    captured = {}

    def fake_quantized_linear(x, weight, scale, **kwargs):
        captured["scale"] = scale
        return torch.zeros((*x.shape[:-1], weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(dsv4_kernel, "quantized_linear_ref", fake_quantized_linear)
    linear.forward(torch.zeros(1, 128, dtype=torch.bfloat16))

    assert captured["scale"] is linear.weight_scale_inv


@pytest.fixture(autouse=True)
def _clear_deepseek_v4_test_globals():
    yield
    core._GLOBAL_CTX = None
    dist_info._TP_INFO = None


def _clear_dsv4_sm80_env(monkeypatch) -> None:
    del monkeypatch


def _install_dsv4_context(cfg: ModelConfig, *, max_len: int) -> Context:
    ctx = Context(page_size=1)
    ctx.kv_cache = create_kvcache_pool(
        cfg,
        num_pages=max_len + 8,
        page_size=1,
        device=torch.device("cpu"),
    )
    ctx.page_table = torch.arange(max_len + 8, dtype=torch.int32).unsqueeze(0)
    core.set_global_ctx(ctx)
    ctx.attn_backend = create_attention_backend("dsv4", cfg)
    return ctx


def test_dsv4_attention_swa_store_out_loc_translates_independent_lifecycle():
    full_locs = torch.tensor([34004, -1], dtype=torch.int32)
    translated_locs = torch.tensor([900, -1], dtype=torch.int64)

    class FakeKVCache:
        swa_independent_lifecycle_enabled = True

        def translate_full_locs_to_swa_locs(self, locs):
            assert locs is full_locs
            return translated_locs

    out = DSV4Attention._swa_store_out_loc(
        SimpleNamespace(kvcache=FakeKVCache()),
        full_locs,
    )

    assert out.dtype is full_locs.dtype
    assert out.device == full_locs.device
    assert out.tolist() == [900, -1]


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
    configure_dsv4_runtime("fallback")
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
    configure_dsv4_runtime("fallback")
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
    configure_dsv4_runtime("fallback")
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
    configure_dsv4_runtime("fallback")
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
    configure_dsv4_runtime("fallback")
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


class _FakeMarlinWNA16Cache:
    def __init__(self, experts: DSV4FusedRoutedExperts) -> None:
        self.w13 = torch.empty(experts.w13_weight.shape[0], 1, dtype=torch.int8)
        self.w2 = torch.empty(experts.w2_weight.shape[0], 1, dtype=torch.int8)
        self.w13_scale = torch.empty(experts.w13_weight.shape[0], 1, dtype=torch.uint8)
        self.w2_scale = torch.empty(experts.w2_weight.shape[0], 1, dtype=torch.uint8)

    def matches(
        self,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> bool:
        del w13_weight, w13_scale, w2_weight, w2_scale
        return True


def test_deepseek_v4_marlin_release_report_is_idempotent():
    _reset_globals()
    cfg = _tiny_dsv4_config()
    experts = DSV4FusedRoutedExperts(cfg)
    expected_source_bytes = sum(
        getattr(experts, name).numel() * getattr(experts, name).element_size()
        for name in experts._raw_expert_weight_names()
    )
    experts._marlin_wna16_weights = _FakeMarlinWNA16Cache(experts)

    report = experts.release_marlin_wna16_original_expert_weights()

    assert report["source_bytes"] == expected_source_bytes
    assert report["released_original_bytes"] == expected_source_bytes
    assert report["released_original_this_call_bytes"] == expected_source_bytes
    assert report["raw_weights_available_after"] is False
    assert report["runtime_policy"] == "marlin_wna16_prepacked_only"
    assert dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR in report["fallback_error"]
    assert all(not hasattr(experts, name) for name in experts._raw_expert_weight_names())
    state = experts.state_dict(prefix="experts")
    assert all(name not in state for name in ("experts.w13_weight", "experts.w2_weight"))

    second = experts.release_marlin_wna16_original_expert_weights()

    assert second["source_bytes"] == expected_source_bytes
    assert second["released_original_bytes"] == expected_source_bytes
    assert second["released_original_this_call_bytes"] == 0
    assert second["raw_weights_available_after"] is False


def test_deepseek_v4_marlin_release_fail_closed_for_grouped_backend():
    _reset_globals()
    cfg = _tiny_dsv4_config()
    experts = DSV4FusedRoutedExperts(cfg)
    experts._marlin_wna16_weights = _FakeMarlinWNA16Cache(experts)
    experts.release_marlin_wna16_original_expert_weights()
    configure_dsv4_runtime("fallback")

    hidden = torch.zeros(2, cfg.hidden_size, dtype=torch.bfloat16)
    weights = torch.ones(2, 1, dtype=torch.float32)
    indices = torch.zeros(2, 1, dtype=torch.long)

    with pytest.raises(
        RuntimeError,
        match="Marlin WNA16 release preset has released raw routed expert weights",
    ):
        experts.forward(hidden, weights, indices)




def test_deepseek_v4_prepare_defers_release_until_before_kv_alloc():
    configure_dsv4_runtime("optimized")
    _reset_globals()
    cfg = replace(_tiny_dsv4_config(), num_layers=3)
    model = DeepseekV4Model(cfg)
    calls: list[tuple[str, int]] = []

    for idx, layer in enumerate(model.layers.op_list):
        def fake_prepare(*, release_original=False, idx=idx):
            assert release_original is False
            calls.append(("prebuild", idx))
            return {
                "persistent_bytes": 11,
                "source_bytes": 22,
                "released_original_bytes": 0,
            }

        def fake_release(idx=idx):
            calls.append(("release", idx))
            return {
                "persistent_bytes": 11,
                "source_bytes": 22,
                "released_original_bytes": 22,
            }

        layer.mlp.experts.prepare_marlin_wna16_weight_cache = fake_prepare
        layer.mlp.experts.release_marlin_wna16_original_expert_weights = fake_release

    report = model.prepare_for_cuda_graph_capture()

    assert calls == [("prebuild", 0), ("prebuild", 1), ("prebuild", 2)]
    moe_report = report["moe_marlin_wna16_cache"]
    assert moe_report["layers_cached"] == 3
    assert moe_report["total_persistent_bytes"] == 33
    assert moe_report["total_source_bytes"] == 66
    assert moe_report["total_released_original_bytes"] == 0
    assert moe_report["release_timing"] == "before_kv_alloc"


def test_deepseek_v4_prepare_does_not_release_after_prebuild_failure():
    configure_dsv4_runtime("optimized")
    _reset_globals()
    cfg = replace(_tiny_dsv4_config(), num_layers=3)
    model = DeepseekV4Model(cfg)
    calls: list[tuple[str, int]] = []

    for idx, layer in enumerate(model.layers.op_list):
        def fake_prepare(*, release_original=False, idx=idx):
            assert release_original is False
            calls.append(("prebuild", idx))
            if idx == 1:
                raise RuntimeError("prebuild failed")
            return {
                "persistent_bytes": 11,
                "source_bytes": 22,
                "released_original_bytes": 0,
            }

        def fake_release(idx=idx):
            calls.append(("release", idx))
            return {
                "persistent_bytes": 11,
                "source_bytes": 22,
                "released_original_bytes": 22,
            }

        layer.mlp.experts.prepare_marlin_wna16_weight_cache = fake_prepare
        layer.mlp.experts.release_marlin_wna16_original_expert_weights = fake_release

    with pytest.raises(RuntimeError, match="prebuild failed"):
        model.prepare_for_cuda_graph_capture()

    assert calls == [("prebuild", 0), ("prebuild", 1)]


def test_deepseek_v4_moe_v2_workspace_is_decode_sized(monkeypatch):
    configure_dsv4_runtime("fallback")
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








def test_deepseek_v4_vllm_runner_sums_routed_and_shared_before_late_reduce(monkeypatch):
    configure_dsv4_runtime("optimized")
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

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    out = moe.forward(hidden, input_ids)

    assert calls == [("routed", False), ("shared", False)]
    assert len(seen_plans) == 1
    assert seen_plans[0].tokens == 2
    assert seen_plans[0].route_plan.route_count == 2
    assert len(fake_comm.calls) == 1
    reduced, label = fake_comm.calls[0]
    assert reduced.dtype is torch.bfloat16
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

    hidden = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    input_ids = torch.zeros(2, 1, dtype=torch.long)
    moe.forward(hidden, input_ids)

    assert len(seen_indices) == 1
    assert seen_indices[0].tolist() == [[1], [1]]


def test_shared_experts_bf16_weight_cache_matches_generic_path(monkeypatch):
    configure_dsv4_runtime("fallback")
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
    configure_dsv4_runtime("optimized")
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

from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors
import torch
from transformers import PretrainedConfig

import minisgl.distributed.info as dist_info
from minisgl.distributed import set_tp_info
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.deepseek_v4 import DeepseekV4ForCausalLM
from minisgl.models.weight import _remap_deepseek_v4_weight_name, _shard_deepseek_v4_tensor
from minisgl.utils import cached_load_hf_config, torch_dtype

MODEL_DIR = Path("/models/DeepSeek-V4-Flash")


@pytest.fixture(autouse=True)
def reset_tp_info():
    old_tp = dist_info._TP_INFO
    dist_info._TP_INFO = None
    yield
    dist_info._TP_INFO = old_tp


def _local_hf_config() -> PretrainedConfig:
    return PretrainedConfig.from_dict(json.loads((MODEL_DIR / "config.json").read_text()))


def _tensor_shape(raw_name: str) -> tuple[int, ...]:
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    with safetensors.safe_open(str(MODEL_DIR / index[raw_name]), framework="pt", device="cpu") as f:
        return tuple(f.get_slice(raw_name).get_shape())


def test_deepseek_v4_config_parse_from_local_json():
    cfg = ModelConfig.from_hf(_local_hf_config())

    assert cfg.is_deepseek_v4
    assert cfg.architectures == ["DeepseekV4ForCausalLM"]
    assert cfg.num_layers == 43
    assert cfg.num_qo_heads == 64
    assert cfg.num_kv_heads == 1
    assert cfg.head_dim == 512
    assert cfg.q_lora_rank == 1024
    assert cfg.o_lora_rank == 1024
    assert cfg.qk_nope_head_dim == 448
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 512
    assert cfg.window_size == 128
    assert cfg.compress_ratios[2:4] == [4, 128]
    assert cfg.index_topk == 512
    assert cfg.index_head_dim == 128
    assert cfg.index_n_heads == 64
    assert cfg.n_routed_experts == 256
    assert cfg.num_experts_per_tok == 6
    assert cfg.n_shared_experts == 1
    assert cfg.scoring_func == "sqrtsoftplus"
    assert cfg.expert_dtype == "fp4"
    assert cfg.routed_scaling_factor == 1.5
    assert cfg.hc_mult == 4
    assert cfg.hc_sinkhorn_iters == 20
    assert cfg.hc_eps == pytest.approx(1e-6)


def test_cached_load_hf_config_falls_back_for_deepseek_v4():
    cfg = cached_load_hf_config(str(MODEL_DIR))

    assert cfg.model_type == "deepseek_v4"
    assert cfg.architectures == ["DeepseekV4ForCausalLM"]
    assert ModelConfig.from_hf(cfg).is_deepseek_v4


def test_deepseek_v4_registry_builds_model_skeleton_on_meta():
    set_tp_info(0, 1)
    cfg = ModelConfig.from_hf(_local_hf_config())

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)

    assert isinstance(model, DeepseekV4ForCausalLM)
    state = model.state_dict()

    assert state["model.embed_tokens.weight"].shape == torch.Size([129280, 4096])
    assert state["lm_head.weight"].shape == torch.Size([129280, 4096])
    assert state["model.hc_head_fn"].shape == torch.Size([4, 16384])
    assert state["model.layers.0.hc_attn_fn"].shape == torch.Size([24, 16384])

    assert state["model.layers.0.self_attn.wq_a.weight"].shape == torch.Size([1024, 4096])
    assert state["model.layers.0.self_attn.wq_a.weight"].dtype is torch.float8_e4m3fn
    assert state["model.layers.0.self_attn.wq_a.weight_scale_inv"].shape == torch.Size([8, 32])
    assert state["model.layers.0.self_attn.wq_a.weight_scale_inv"].dtype is torch.float8_e8m0fnu
    assert state["model.layers.0.self_attn.attn_sink"].shape == torch.Size([64])
    assert state["model.layers.0.self_attn.wkv.weight"].shape == torch.Size([512, 4096])
    assert state["model.layers.0.self_attn.wo_a.weight"].shape == torch.Size([8192, 4096])
    assert state["model.layers.0.self_attn.wo_b.weight"].shape == torch.Size([4096, 8192])

    assert state["model.layers.2.self_attn.compressor.ape"].shape == torch.Size([4, 1024])
    assert state["model.layers.2.self_attn.compressor.wkv_gate.weight"].shape == torch.Size([2048, 4096])
    assert state["model.layers.2.self_attn.indexer.wq_b.weight"].shape == torch.Size([8192, 1024])
    assert state["model.layers.2.self_attn.indexer.weights_proj.weight"].shape == torch.Size([64, 4096])
    assert state["model.layers.2.self_attn.indexer.compressor.ape"].shape == torch.Size([4, 256])
    assert state["model.layers.3.self_attn.compressor.ape"].shape == torch.Size([128, 512])

    assert state["model.layers.0.mlp.gate.weight"].shape == torch.Size([256, 4096])
    assert state["model.layers.0.mlp.topk.tid2eid"].shape == torch.Size([129280, 6])
    assert state["model.layers.3.mlp.gate.e_score_correction_bias"].shape == torch.Size([256])
    assert "model.layers.3.mlp.topk.tid2eid" not in state
    assert state["model.layers.0.mlp.experts.w13_weight"].shape == torch.Size([256, 2, 2048, 2048])
    assert state["model.layers.0.mlp.experts.w13_weight"].dtype is torch.int8
    assert state["model.layers.0.mlp.experts.w13_weight_scale_inv"].shape == torch.Size([256, 2, 2048, 128])
    assert state["model.layers.0.mlp.experts.w2_weight"].shape == torch.Size([256, 4096, 1024])
    assert state["model.layers.0.mlp.experts.w2_weight_scale_inv"].shape == torch.Size([256, 4096, 64])
    assert state["model.layers.0.mlp.shared_experts.gate_up_proj.weight"].shape == torch.Size([4096, 4096])
    assert state["model.layers.0.mlp.shared_experts.down_proj.weight"].shape == torch.Size([4096, 2048])


def test_deepseek_v4_tp8_model_uses_local_attention_sink_shape():
    set_tp_info(3, 8)
    cfg = ModelConfig.from_hf(_local_hf_config())

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        state = create_model(cfg).state_dict()

    assert state["model.layers.0.self_attn.attn_sink"].shape == torch.Size([8])


def test_deepseek_v4_checkpoint_metadata_maps_to_runtime_shapes():
    set_tp_info(0, 1)
    cfg = ModelConfig.from_hf(_local_hf_config())
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        state = create_model(cfg).state_dict()

    direct_pairs = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "layers.0.attn.wq_a.weight": "model.layers.0.self_attn.wq_a.weight",
        "layers.0.attn.wq_a.scale": "model.layers.0.self_attn.wq_a.weight_scale_inv",
        "layers.0.attn.kv_norm.weight": "model.layers.0.self_attn.kv_norm.weight",
        "layers.0.ffn.gate.weight": "model.layers.0.mlp.gate.weight",
        "layers.0.ffn.gate.tid2eid": "model.layers.0.mlp.topk.tid2eid",
    }
    for raw_name, runtime_name in direct_pairs.items():
        assert _remap_deepseek_v4_weight_name(raw_name) == runtime_name
        assert _tensor_shape(raw_name) == tuple(state[runtime_name].shape)

    assert _remap_deepseek_v4_weight_name("layers.2.attn.compressor.wkv.weight") == (
        "model.layers.2.self_attn.compressor.wkv.weight"
    )
    assert _tensor_shape("layers.2.attn.compressor.wkv.weight") == (1024, 4096)
    assert _tensor_shape("layers.2.attn.compressor.wgate.weight") == (1024, 4096)
    assert state["model.layers.2.self_attn.compressor.wkv_gate.weight"].shape == torch.Size([2048, 4096])

    assert _tensor_shape("layers.0.ffn.shared_experts.w1.weight") == (2048, 4096)
    assert _tensor_shape("layers.0.ffn.shared_experts.w3.weight") == (2048, 4096)
    assert state["model.layers.0.mlp.shared_experts.gate_up_proj.weight"].shape == torch.Size([4096, 4096])

    assert _tensor_shape("layers.0.ffn.experts.0.w1.weight") == (2048, 2048)
    assert _tensor_shape("layers.0.ffn.experts.0.w3.weight") == (2048, 2048)
    assert state["model.layers.0.mlp.experts.w13_weight"].shape[1:] == torch.Size([2, 2048, 2048])
    assert _tensor_shape("layers.0.ffn.experts.0.w2.weight") == (4096, 1024)
    assert state["model.layers.0.mlp.experts.w2_weight"].shape[1:] == torch.Size([4096, 1024])


def test_deepseek_v4_weight_sharding_splits_attention_sink_by_local_heads():
    raw = torch.arange(64, dtype=torch.float32)

    shard = _shard_deepseek_v4_tensor("model.layers.0.self_attn.attn_sink", raw, r=3, n=8)

    assert torch.equal(shard, torch.arange(24, 32, dtype=torch.float32))



def test_deepseek_v4_load_weight_packs_synthetic_checkpoint(tmp_path):
    from safetensors.torch import save_file
    from minisgl.models.weight import load_weight

    fp8 = getattr(torch, "float8_e4m3fn", None)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if fp8 is None or e8m0 is None:
        pytest.skip("torch float8 dtypes are required for DSV4 quantized weight tests")

    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 8,
        "vocab_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 2,
        "o_lora_rank": 2,
        "qk_rope_head_dim": 1,
        "max_position_embeddings": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "moe_intermediate_size": 4,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "norm_topk_prob": True,
        "expert_dtype": "fp4",
        "o_groups": 1,
        "index_head_dim": 2,
        "index_n_heads": 2,
        "index_topk": 2,
        "hc_mult": 1,
        "hc_sinkhorn_iters": 1,
        "compress_ratios": [4],
        "rope_theta": 10000,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors = {
        "embed.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
        "layers.0.attn.compressor.wkv.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
        "layers.0.attn.compressor.wgate.weight": torch.ones(8, 8, dtype=torch.bfloat16),
        "layers.0.ffn.shared_experts.w1.weight": torch.zeros(4, 8, dtype=fp8),
        "layers.0.ffn.shared_experts.w3.weight": torch.ones(4, 8, dtype=fp8),
        "layers.0.ffn.shared_experts.w1.scale": torch.ones(1, 1, dtype=e8m0),
        "layers.0.ffn.shared_experts.w3.scale": torch.ones(1, 1, dtype=e8m0),
    }
    for expert in range(2):
        tensors[f"layers.0.ffn.experts.{expert}.w1.weight"] = torch.full(
            (4, 4), expert + 1, dtype=torch.int8
        )
        tensors[f"layers.0.ffn.experts.{expert}.w3.weight"] = torch.full(
            (4, 4), expert + 3, dtype=torch.int8
        )
        tensors[f"layers.0.ffn.experts.{expert}.w2.weight"] = torch.full(
            (8, 2), expert + 5, dtype=torch.int8
        )
        tensors[f"layers.0.ffn.experts.{expert}.w1.scale"] = torch.ones(4, 1, dtype=e8m0)
        tensors[f"layers.0.ffn.experts.{expert}.w3.scale"] = torch.ones(4, 1, dtype=e8m0)
        tensors[f"layers.0.ffn.experts.{expert}.w2.scale"] = torch.ones(8, 1, dtype=e8m0)
    save_file(tensors, tmp_path / "model.safetensors")

    set_tp_info(0, 1)
    loaded = dict(load_weight(str(tmp_path), torch.device("cpu")))

    assert loaded["model.embed_tokens.weight"].shape == torch.Size([16, 8])
    assert loaded["model.layers.0.self_attn.compressor.wkv_gate.weight"].shape == torch.Size([16, 8])
    assert loaded["model.layers.0.mlp.shared_experts.gate_up_proj.weight"].shape == torch.Size([8, 8])
    assert loaded["model.layers.0.mlp.shared_experts.gate_up_proj.weight_scale_inv"].shape == torch.Size([2, 1])
    assert loaded["model.layers.0.mlp.experts.w13_weight"].shape == torch.Size([2, 2, 4, 4])
    assert loaded["model.layers.0.mlp.experts.w13_weight_scale_inv"].shape == torch.Size([2, 2, 4, 1])
    assert loaded["model.layers.0.mlp.experts.w2_weight"].shape == torch.Size([2, 8, 2])
    assert loaded["model.layers.0.mlp.experts.w2_weight_scale_inv"].shape == torch.Size([2, 8, 1])
    assert loaded["model.layers.0.mlp.experts.w13_weight"][1, 0, 0, 0].item() == 2
    assert loaded["model.layers.0.mlp.experts.w13_weight"][1, 1, 0, 0].item() == 4

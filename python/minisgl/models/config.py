from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    q_lora_rank: int = 0
    o_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    window_size: int = 0
    compress_ratios: list[int] = field(default_factory=list)
    compress_rope_theta: float | int | None = None
    index_topk: int = 0
    index_head_dim: int = 0
    index_n_heads: int = 0
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    scoring_func: str = ""
    expert_dtype: str = ""
    routed_scaling_factor: float = 1.0
    hc_mult: int = 1
    hc_sinkhorn_iters: int = 0
    hc_eps: float = 1e-6
    o_groups: int = 1
    n_hash_layers: int = 0
    swiglu_limit: float | None = None
    topk_method: str = ""
    n_group: int = 1
    topk_group: int = 1
    original_seq_len: int = 0
    rope_factor: float = 1.0
    beta_fast: int = 32
    beta_slow: int = 1
    quantization_config: dict[str, Any] = field(default_factory=dict)

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type or self.num_experts > 0 or self.n_routed_experts > 0

    @property
    def is_deepseek_v4(self) -> bool:
        return self.model_type == "deepseek_v4" or any(
            arch == "DeepseekV4ForCausalLM" for arch in self.architectures
        )

    @property
    def rope_head_dim(self) -> int:
        return self.qk_rope_head_dim or self.rotary_config.rotary_dim

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "model_type", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        architectures = getattr(config, "architectures", None)
        if not architectures:
            raise ValueError(
                "Model config is missing architectures; this release supports "
                "DeepSeek V4 Flash only (DeepseekV4ForCausalLM)."
            )
        if not isinstance(architectures, (list, tuple)) or architectures[0] != (
            "DeepseekV4ForCausalLM"
        ):
            raise ValueError(
                f"Model architecture {architectures!r} is not supported; this release "
                "supports DeepSeek V4 Flash only (DeepseekV4ForCausalLM)."
            )

        model_type = getattr(config, "model_type", None)
        if model_type != "deepseek_v4":
            raise ValueError(
                f"Model type {model_type!r} is not supported; this release supports "
                "DeepSeek V4 Flash only (model_type='deepseek_v4')."
            )
        is_deepseek_v4 = True
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        n_routed_experts = getattr(config, "n_routed_experts", 0)
        num_experts = getattr(
            config,
            "num_local_experts",
            getattr(config, "num_experts", n_routed_experts),
        )
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            rope_theta = 10000

        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", head_dim)
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", head_dim - qk_rope_head_dim)
        window_size = getattr(config, "window_size", getattr(config, "sliding_window", 0))
        compress_ratios = list(getattr(config, "compress_ratios", []))
        if is_deepseek_v4 and not compress_ratios:
            compress_ratios = [0] * getattr(config, "num_hidden_layers", 0)

        original_seq_len = 0
        rope_factor = 1.0
        beta_fast = 32
        beta_slow = 1
        if isinstance(rope_scaling, dict):
            original_seq_len = int(rope_scaling.get("original_max_position_embeddings", 0) or 0)
            rope_factor = float(rope_scaling.get("factor", 1.0) or 1.0)
            beta_fast = int(rope_scaling.get("beta_fast", 32) or 32)
            beta_slow = int(rope_scaling.get("beta_slow", 1) or 1)

        quantization_config = getattr(config, "quantization_config", {}) or {}
        if not isinstance(quantization_config, dict):
            quantization_config = {}

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=getattr(config, "intermediate_size", moe_intermediate_size),
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            q_lora_rank=getattr(config, "q_lora_rank", 0) or 0,
            o_lora_rank=getattr(config, "o_lora_rank", 0) or 0,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=getattr(config, "v_head_dim", head_dim),
            window_size=window_size,
            compress_ratios=compress_ratios,
            compress_rope_theta=getattr(config, "compress_rope_theta", None),
            index_topk=getattr(config, "index_topk", 0) or 0,
            index_head_dim=getattr(config, "index_head_dim", 0) or 0,
            index_n_heads=getattr(config, "index_n_heads", 0) or 0,
            n_routed_experts=n_routed_experts,
            n_shared_experts=getattr(config, "n_shared_experts", 0) or 0,
            scoring_func=getattr(config, "scoring_func", ""),
            expert_dtype=getattr(config, "expert_dtype", ""),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            hc_mult=getattr(config, "hc_mult", 1) or 1,
            hc_sinkhorn_iters=getattr(config, "hc_sinkhorn_iters", 0) or 0,
            hc_eps=getattr(config, "hc_eps", 1e-6),
            o_groups=getattr(config, "o_groups", 1) or 1,
            n_hash_layers=getattr(config, "num_hash_layers", getattr(config, "n_hash_layers", 0)) or 0,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            topk_method=getattr(config, "topk_method", ""),
            n_group=getattr(config, "n_group", getattr(config, "num_expert_groups", 1)) or 1,
            topk_group=getattr(config, "topk_group", 1) or 1,
            original_seq_len=original_seq_len,
            rope_factor=rope_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            quantization_config=dict(quantization_config),
        )

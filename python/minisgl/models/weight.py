from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
_DSV4_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale_inv)$"
)


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def _remap_deepseek_v4_weight_name(name: str) -> str:
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    if name == "head.weight":
        return "lm_head.weight"
    if name == "norm.weight":
        return "model.norm.weight"
    if name.startswith("hc_head_"):
        return "model." + name

    if name.startswith("layers."):
        name = "model." + name
    name = name.replace(".attn.", ".self_attn.")
    name = name.replace(".ffn.", ".mlp.")
    name = name.replace(".attn_norm.", ".input_layernorm.")
    name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

    if "self_attn" in name:
        name = name.replace(".scale", ".weight_scale_inv")

    name = name.replace(".gate.tid2eid", ".topk.tid2eid")
    name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
    name = name.replace(".w1.", ".gate_proj.")
    name = name.replace(".w2.", ".down_proj.")
    name = name.replace(".w3.", ".up_proj.")
    if "mlp" in name:
        name = name.replace(".scale", ".weight_scale_inv")

    return name


def _shard_deepseek_v4_tensor(key: str, value: torch.Tensor, r: int, n: int):
    if n == 1:
        return value

    if key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()

    split_dim0 = (
        ".self_attn.wq_b.",
        ".self_attn.wo_a.",
        ".mlp.shared_experts.gate_proj.",
        ".mlp.shared_experts.up_proj.",
    )
    split_dim1 = (
        ".self_attn.wo_b.",
        ".mlp.shared_experts.down_proj.",
    )

    expert_match = _DSV4_EXPERT_PATTERN.match(key)
    if expert_match is not None:
        proj = expert_match.group("proj")
        if proj in ("gate_proj", "up_proj"):
            return value.chunk(n, dim=0)[r].clone()
        if proj == "down_proj":
            return value.chunk(n, dim=1)[r].clone()

    if any(part in key for part in split_dim0):
        return value.chunk(n, dim=0)[r].clone()
    if any(part in key for part in split_dim1):
        return value.chunk(n, dim=1)[r].clone()
    return value


def _get_dsv4_compressor_merge_info(key: str):
    if key.endswith(".compressor.wkv.weight"):
        return key.replace(".wkv.weight", ".wkv_gate.weight"), "wkv", ("wkv", "wgate")
    if key.endswith(".compressor.wgate.weight"):
        return key.replace(".wgate.weight", ".wkv_gate.weight"), "wgate", ("wkv", "wgate")
    return None


def _get_dsv4_shared_merge_info(key: str):
    if ".mlp.shared_experts.gate_proj." in key:
        return (
            key.replace(".gate_proj.", ".gate_up_proj."),
            "gate",
            ("gate", "up"),
        )
    if ".mlp.shared_experts.up_proj." in key:
        return (
            key.replace(".up_proj.", ".gate_up_proj."),
            "up",
            ("gate", "up"),
        )
    return None


def _get_dsv4_expert_pack_info(key: str):
    match = _DSV4_EXPERT_PATTERN.match(key)
    if match is None:
        return None
    prefix = match.group("prefix")
    expert_idx = int(match.group("idx"))
    proj = match.group("proj")
    kind = match.group("kind")
    suffix = "weight_scale_inv" if kind == "weight_scale_inv" else "weight"
    if proj in ("gate_proj", "up_proj"):
        return f"{prefix}.w13_{suffix}", expert_idx, ("gate" if proj == "gate_proj" else "up")
    return f"{prefix}.w2_{suffix}", expert_idx, "down"


def _validate_dsv4_quantized_tensor(name: str, tensor: torch.Tensor) -> None:
    if ".mlp.experts." not in name:
        return
    if name.endswith(".weight") and tensor.dtype != torch.int8:
        raise NotImplementedError(
            f"DeepSeek V4 routed fp4 expert tensor {name} has dtype {tensor.dtype}; "
            "only packed int8 fp4 checkpoint weights are supported in TARGET_01"
        )


def _load_deepseek_v4_weight(
    files: list[str],
    device: torch.device,
    config,
) -> Iterator[Tuple[str, torch.Tensor]]:
    tp_info = get_tp_info()
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor | Dict[str, torch.Tensor]]] = {}

    def maybe_emit_expert(packed_key: str):
        slots_by_expert = expert_buf.get(packed_key)
        if slots_by_expert is None or len(slots_by_expert) != config.n_routed_experts:
            return None
        packed = []
        for expert_idx in range(config.n_routed_experts):
            value = slots_by_expert.get(expert_idx)
            if not isinstance(value, torch.Tensor):
                return None
            packed.append(value)
        del expert_buf[packed_key]
        return packed_key, torch.stack(packed, dim=0)

    for file in tqdm(files, desc="Loading DeepSeek V4 weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if raw_name.startswith("mtp."):
                    continue
                name = _remap_deepseek_v4_weight_name(raw_name)
                if name.startswith("model.layers."):
                    parts = name.split(".", 3)
                    if len(parts) >= 3 and int(parts[2]) >= config.num_layers:
                        continue

                raw = f.get_tensor(raw_name)
                _validate_dsv4_quantized_tensor(name, raw)
                tensor = _shard_deepseek_v4_tensor(name, raw, tp_info.rank, tp_info.size)
                del raw

                if (info := _get_dsv4_compressor_merge_info(name)) is not None:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    yield merged_key, torch.cat(parts, dim=0)
                    continue

                if (info := _get_dsv4_shared_merge_info(name)) is not None:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    yield merged_key, torch.cat(parts, dim=0)
                    continue

                if (info := _get_dsv4_expert_pack_info(name)) is not None:
                    packed_key, expert_idx, slot = info
                    if slot in ("gate", "up"):
                        expert_slots = expert_buf.setdefault(packed_key, {}).setdefault(expert_idx, {})
                        assert isinstance(expert_slots, dict)
                        expert_slots[slot] = tensor
                        if "gate" not in expert_slots or "up" not in expert_slots:
                            continue
                        expert_buf[packed_key][expert_idx] = torch.stack(
                            [expert_slots["gate"], expert_slots["up"]], dim=0
                        )
                    else:
                        expert_buf.setdefault(packed_key, {})[expert_idx] = tensor
                    out = maybe_emit_expert(packed_key)
                    if out is not None:
                        yield out
                    continue

                yield name, tensor

    assert not merge_buf, f"Incomplete DeepSeek V4 merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete DeepSeek V4 expert tensors in checkpoint: {list(expert_buf.keys())}"


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    if config.is_deepseek_v4:
        yield from _load_deepseek_v4_weight(files, device, config)
        return

    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                name = name.removeprefix("language_model.")
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)
                else:
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"

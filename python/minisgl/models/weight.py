from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

_DSV4_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts)\."
    r"(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight|weight_scale_inv)$"
)

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
        ".self_attn.attn_sink",
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
    yield from _load_deepseek_v4_weight(files, device, config)

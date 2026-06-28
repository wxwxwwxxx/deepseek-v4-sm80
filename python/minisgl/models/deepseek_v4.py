from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from minisgl.attention import BaseAttnMetadata
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers import BaseOP, OPList
from minisgl.utils import div_ceil, div_even

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


def _fp8_dtype() -> torch.dtype:
    return getattr(torch, "float8_e4m3fn", torch.uint8)


def _e8m0_dtype() -> torch.dtype:
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def _scale_dim(size: int, block_size: int = 128) -> int:
    return div_ceil(size, block_size)


_FP4_TABLE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _fp4_table(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    table = _FP4_TABLE_CACHE.get(key)
    if table is None:
        table = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )
        _FP4_TABLE_CACHE[key] = table
    return table


def _dequant_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    w = weight.float()
    if scale is None:
        return w.to(out_dtype)
    out_features, in_features = w.shape
    expanded = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expanded = expanded[:out_features, :in_features]
    return (w * expanded).to(out_dtype)


def _dequant_fp4_weight(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    packed = weight.contiguous().view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    table = _fp4_table(weight.device)
    unpacked = torch.stack((table[low.long()], table[high.long()]), dim=-1).flatten(-2)
    if scale is None:
        return unpacked.to(out_dtype)
    expanded = scale.float().repeat_interleave(32, dim=-1)
    expanded = expanded[..., : unpacked.shape[-1]]
    return (unpacked * expanded).to(out_dtype)


def _quantize_fp8_activation_ref(x: torch.Tensor, *, block_size: int = 128) -> torch.Tensor:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or x.numel() == 0 or x.shape[-1] % block_size != 0:
        return x
    dtype = x.dtype
    flat = x.contiguous().view(-1, x.shape[-1]).float()
    groups = flat.view(flat.shape[0], flat.shape[1] // block_size, block_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    y = (groups / scale).clamp(-448.0, 448.0).to(fp8_dtype).float() * scale
    return y.reshape_as(flat).reshape_as(x).to(dtype)


def _quantized_linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    weight_kind: str,
) -> torch.Tensor:
    if weight_kind == "fp4":
        x = _quantize_fp8_activation_ref(x)
        w = _dequant_fp4_weight(weight, scale, out_dtype=x.dtype)
    elif weight_kind == "fp8":
        x = _quantize_fp8_activation_ref(x)
        w = _dequant_fp8_weight(weight, scale, out_dtype=x.dtype)
    else:
        w = weight.to(x.dtype)
    return F.linear(x, w)


def _apply_rotary_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    inverse: bool = False,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    if rotary_dim <= 0:
        return x
    if rotary_dim % 2 != 0:
        raise ValueError(f"DeepSeek V4 rotary_dim must be even, got {rotary_dim}")

    pos = positions.to(device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device) / rotary_dim)
    )
    if original_seq_len > 0:

        def correction_dim(num_rotations: float) -> float:
            return rotary_dim * math.log(
                original_seq_len / (num_rotations * 2 * math.pi)
            ) / (2 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rotary_dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(rotary_dim // 2, dtype=torch.float32, device=x.device) - low)
            / max(high - low, 1),
            0,
            1,
        )
        smooth = 1 - ramp
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

    freqs = torch.outer(pos, inv_freq)
    if inverse:
        freqs = -freqs
    cos = freqs.cos()
    sin = freqs.sin()
    while cos.ndim < x[..., -rotary_dim:].ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    rope = x[..., -rotary_dim:].float().unflatten(-1, (-1, 2))
    a, b = rope[..., 0], rope[..., 1]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    x[..., -rotary_dim:] = rotated.to(x.dtype)
    return x


def _hc_split_sinkhorn_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = mixes.view(-1, mix_hc).float()
    hc_scale = hc_scale.float()
    hc_base = hc_base.float()

    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post_start = hc_mult
    post_end = 2 * hc_mult
    post = 2 * torch.sigmoid(
        mixes[:, post_start:post_end] * hc_scale[1] + hc_base[post_start:post_end]
    )
    comb_raw = mixes[:, post_end:].view(-1, hc_mult, hc_mult)
    comb_base = hc_base[post_end:].view(hc_mult, hc_mult)
    comb = torch.softmax(comb_raw * hc_scale[2] + comb_base, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


@dataclass
class DSV4FallbackAttentionMetadata(BaseAttnMetadata):
    cu_seqlens_q: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class DSV4RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = torch.empty(size, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


class DSV4VocabParallelEmbedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        tp = get_tp_info()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, tp.size)
        start_idx = self.num_embeddings_tp * tp.rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim, dtype=torch.bfloat16)
        self._comm = DistributedCommunicator()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return F.embedding(input_ids.long(), self.weight)
        start, end = self.vocab_range
        local_ids = input_ids.long() - start
        mask = (local_ids < 0) | (local_ids >= end - start)
        local_ids = local_ids.masked_fill(mask, 0)
        y = F.embedding(local_ids, self.weight)
        y = y.masked_fill(mask.unsqueeze(-1), 0)
        return self._comm.all_reduce(y)

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.weight.float())
        if self.tp_size == 1:
            return logits[:, : self.num_embeddings]
        gathered = self._comm.all_gather(logits)
        if x.shape[0] == 1:
            return gathered.view(1, -1)[:, : self.num_embeddings]
        output = gathered.view((self.tp_size,) + tuple(logits.shape))
        output = output.permute(1, 0, 2).contiguous()
        return output.reshape(x.shape[0], self.tp_size * logits.shape[1])[:, : self.num_embeddings]


class DSV4Linear(BaseOP):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        weight_dtype: torch.dtype = torch.bfloat16,
        scale_dtype: torch.dtype | None = None,
        col_parallel: bool = False,
        row_parallel: bool = False,
    ):
        tp = get_tp_info()
        assert not (col_parallel and row_parallel)
        self.row_parallel = row_parallel
        self.col_parallel = col_parallel
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        local_input_size = div_even(input_size, tp.size) if row_parallel else input_size
        local_output_size = div_even(output_size, tp.size) if col_parallel else output_size
        self.weight = torch.empty(local_output_size, local_input_size, dtype=weight_dtype)
        if scale_dtype is not None:
            self.weight_scale_inv = torch.empty(
                _scale_dim(local_output_size),
                _scale_dim(local_input_size),
                dtype=scale_dtype,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = getattr(self, "weight_scale_inv", None)
        if self.weight.dtype is torch.int8:
            y = _quantized_linear_ref(x, self.weight, scale, weight_kind="fp4")
        elif self.weight.dtype is _fp8_dtype():
            y = _quantized_linear_ref(x, self.weight, scale, weight_kind="fp8")
        else:
            y = F.linear(x, self.weight.to(x.dtype))
        if self.row_parallel and self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class DSV4Compressor(BaseOP):
    def __init__(self, config: ModelConfig, ratio: int, head_dim: int):
        self.ratio = ratio
        self.head_dim = head_dim
        self.overlap = ratio == 4
        coff = 2 if ratio == 4 else 1
        self.ape = torch.empty(ratio, coff * head_dim, dtype=torch.float32)
        self.wkv_gate = DSV4Linear(
            config.hidden_size,
            2 * coff * head_dim,
            weight_dtype=torch.bfloat16,
        )
        self.norm = DSV4RMSNorm(head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_empty((0, self.head_dim))
        ratio = self.ratio
        projected = self.wkv_gate.forward(x).float()
        kv, score = projected.chunk(2, dim=-1)
        if positions is None:
            positions = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
        positions = positions.long()

        rows = []
        start = 0
        while start < x.shape[0]:
            end = min(start + ratio, x.shape[0])
            if end - start < ratio:
                break
            slot = (positions[start:end] % ratio).to(torch.long)
            local_score = score[start:end] + self.ape[slot].float()
            local_kv = kv[start:end]
            if self.overlap:
                left = local_kv[:, : self.head_dim]
                right = local_kv[:, self.head_dim :]
                local_score = torch.cat(
                    [local_score[:, : self.head_dim], local_score[:, self.head_dim :]],
                    dim=0,
                )
                local_kv = torch.cat([left, right], dim=0)
            pooled = (local_kv * local_score.softmax(dim=0)).sum(dim=0, keepdim=True)
            rows.append(self.norm.forward(pooled.to(x.dtype)))
            start = end
        if not rows:
            return x.new_empty((0, self.head_dim))
        return torch.cat(rows, dim=0)


class DSV4Indexer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        _ = layer_id
        self.wq_b = DSV4Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
        )
        self.weights_proj = DSV4Linear(
            config.hidden_size,
            config.index_n_heads,
            weight_dtype=torch.bfloat16,
        )
        self.compressor = DSV4Compressor(config, ratio=4, head_dim=config.index_head_dim)

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self.compressor.forward(x, positions)
        self.wq_b.forward(q_lora)
        self.weights_proj.forward(x)
        return torch.empty((x.shape[0], 0), dtype=torch.int64, device=x.device)


class DSV4Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp = get_tp_info()
        self.layer_id = layer_id
        self.num_heads = config.num_qo_heads
        self.num_local_heads = div_even(config.num_qo_heads, tp.size)
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.window_size = config.window_size
        self.softmax_scale = config.head_dim**-0.5
        self.o_groups = config.o_groups
        self.num_local_groups = div_even(config.o_groups, tp.size)
        self.o_lora_rank = config.o_lora_rank
        self.rms_norm_eps = config.rms_norm_eps
        ratio = config.compress_ratios[layer_id] if layer_id < len(config.compress_ratios) else 0
        self.compress_ratio = ratio
        self.rope_base = (
            config.compress_rope_theta
            if ratio and config.compress_rope_theta is not None
            else config.rotary_config.base
        )
        self.original_seq_len = config.original_seq_len if ratio else 0
        self.rope_factor = config.rope_factor
        self.beta_fast = config.beta_fast
        self.beta_slow = config.beta_slow
        self.attn_sink = torch.empty(config.num_qo_heads, dtype=torch.float32)
        self.wq_a = DSV4Linear(
            config.hidden_size,
            config.q_lora_rank,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
        )
        self.wq_b = DSV4Linear(
            config.q_lora_rank,
            config.num_qo_heads * config.head_dim,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
            col_parallel=True,
        )
        self.q_norm = DSV4RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wkv = DSV4Linear(
            config.hidden_size,
            config.head_dim,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
        )
        self.kv_norm = DSV4RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wo_a = DSV4Linear(
            config.num_qo_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
            col_parallel=True,
        )
        self.wo_b = DSV4Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
            row_parallel=True,
        )

        if ratio in (4, 128):
            self.compressor = DSV4Compressor(config, ratio=ratio, head_dim=config.head_dim)
        if ratio == 4:
            self.indexer = DSV4Indexer(config, layer_id)

    def _sequence_spans(self, batch: Batch, total_tokens: int) -> list[tuple[int, int]]:
        reqs = getattr(batch, "padded_reqs", batch.reqs)
        spans = []
        offset = 0
        for req in reqs:
            length = req.extend_len if batch.is_prefill else 1
            spans.append((offset, offset + length))
            offset += length
        if offset != total_tokens:
            return [(0, total_tokens)]
        return spans

    def _fallback_attention(self, q: torch.Tensor, kv: torch.Tensor, batch: Batch) -> torch.Tensor:
        out = torch.empty_like(q)
        sink = self.attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)
        spans = self._sequence_spans(batch, q.shape[0])
        for start, end in spans:
            q_seq = q[start:end].float()
            kv_seq = kv[start:end].float()
            seq_len = end - start
            for local_idx in range(seq_len):
                ctx_start = max(0, local_idx - self.window_size + 1) if self.window_size else 0
                candidates = kv_seq[ctx_start : local_idx + 1]
                scores = torch.einsum("hd,td->ht", q_seq[local_idx], candidates)
                scores = scores * self.softmax_scale
                max_score = torch.maximum(scores.max(dim=-1).values, sink)
                exp_scores = torch.exp(scores - max_score[:, None])
                denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
                attn = exp_scores / denom[:, None]
                out[start + local_idx] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        positions = batch.positions.to(device=x.device, dtype=torch.long)
        q_lora = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(q_lora).view(-1, self.num_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_norm_eps).to(q.dtype)
        _apply_rotary_tail(
            q,
            positions,
            rotary_dim=self.rope_head_dim,
            base=float(self.rope_base),
            original_seq_len=self.original_seq_len,
            factor=self.rope_factor,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
        )

        kv = self.kv_norm.forward(self.wkv.forward(x))
        _apply_rotary_tail(
            kv,
            positions,
            rotary_dim=self.rope_head_dim,
            base=float(self.rope_base),
            original_seq_len=self.original_seq_len,
            factor=self.rope_factor,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
        )
        if self.rope_head_dim < kv.shape[-1]:
            kv[..., : -self.rope_head_dim] = _quantize_fp8_activation_ref(
                kv[..., : -self.rope_head_dim], block_size=64
            )

        if hasattr(self, "indexer"):
            self.indexer.forward(x, q_lora, positions)
        if hasattr(self, "compressor"):
            self.compressor.forward(x, positions)

        o = self._fallback_attention(q, kv, batch)
        _apply_rotary_tail(
            o,
            positions,
            rotary_dim=self.rope_head_dim,
            base=float(self.rope_base),
            inverse=True,
            original_seq_len=self.original_seq_len,
            factor=self.rope_factor,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
        )
        d_per_group = self.num_local_heads * self.head_dim // self.num_local_groups
        o = o.reshape(x.shape[0], self.num_local_groups, d_per_group)
        wo_a = _dequant_fp8_weight(
            self.wo_a.weight,
            getattr(self.wo_a, "weight_scale_inv", None),
            out_dtype=o.dtype,
        )
        wo_a = wo_a.view(self.num_local_groups, self.o_lora_rank, d_per_group)
        o = torch.einsum("tgd,grd->tgr", o, wo_a).reshape(x.shape[0], -1)
        return self.wo_b.forward(o)


class DSV4TopK(BaseOP):
    def __init__(self, config: ModelConfig):
        self.tid2eid = torch.empty(
            config.vocab_size,
            config.num_experts_per_tok,
            dtype=torch.int64,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tid2eid[input_ids.long()]


class DSV4MoEGate(BaseOP):
    def __init__(self, config: ModelConfig, *, has_correction_bias: bool):
        self.weight = torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.bfloat16)
        if has_correction_bias:
            self.e_score_correction_bias = torch.empty(config.n_routed_experts, dtype=torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        topk: int,
        scoring_func: str,
        routed_scaling_factor: float,
        hash_topk: DSV4TopK | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(hidden_states.float(), self.weight.float())
        if scoring_func == "softmax":
            original_scores = scores.softmax(dim=-1)
        elif scoring_func == "sigmoid":
            original_scores = scores.sigmoid()
        else:
            original_scores = F.softplus(scores).sqrt()

        if hash_topk is not None:
            if input_ids is None:
                raise ValueError("DeepSeek V4 hash routing requires input_ids")
            indices = hash_topk.forward(input_ids.view(-1)).long()
        else:
            scores_for_topk = original_scores
            if hasattr(self, "e_score_correction_bias"):
                scores_for_topk = scores_for_topk + self.e_score_correction_bias.float()
            indices = scores_for_topk.topk(topk, dim=-1).indices

        weights = original_scores.gather(1, indices)
        if scoring_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights * routed_scaling_factor, indices


class DSV4FusedRoutedExperts(BaseOP):
    def __init__(self, config: ModelConfig):
        tp = get_tp_info()
        local_intermediate = div_even(config.moe_intermediate_size, tp.size)
        self.swiglu_limit = config.swiglu_limit or 0.0
        self.w13_weight = torch.empty(
            config.n_routed_experts,
            2,
            local_intermediate,
            config.hidden_size // 2,
            dtype=torch.int8,
        )
        self.w13_weight_scale_inv = torch.empty(
            config.n_routed_experts,
            2,
            local_intermediate,
            div_ceil(config.hidden_size, 32),
            dtype=_e8m0_dtype(),
        )
        self.w2_weight = torch.empty(
            config.n_routed_experts,
            config.hidden_size,
            local_intermediate // 2,
            dtype=torch.int8,
        )
        self.w2_weight_scale_inv = torch.empty(
            config.n_routed_experts,
            config.hidden_size,
            div_ceil(local_intermediate, 32),
            dtype=_e8m0_dtype(),
        )

    def _expert_forward(self, local_idx: int, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w1 = _quantized_linear_ref(
            x,
            self.w13_weight[local_idx, 0],
            self.w13_weight_scale_inv[local_idx, 0],
            weight_kind="fp4",
        ).float()
        w3 = _quantized_linear_ref(
            x,
            self.w13_weight[local_idx, 1],
            self.w13_weight_scale_inv[local_idx, 1],
            weight_kind="fp4",
        ).float()
        if self.swiglu_limit > 0:
            w3 = torch.clamp(w3, min=-self.swiglu_limit, max=self.swiglu_limit)
            w1 = torch.clamp(w1, max=self.swiglu_limit)
        hidden = F.silu(w1) * w3 * weights
        return _quantized_linear_ref(
            hidden.to(x.dtype),
            self.w2_weight[local_idx],
            self.w2_weight_scale_inv[local_idx],
            weight_kind="fp4",
        )

    def forward(self, hidden_states: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx in range(self.w13_weight.shape[0]):
            token_idx, top_idx = torch.where(indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            y[token_idx] += self._expert_forward(
                int(expert_idx),
                hidden_states[token_idx],
                weights[token_idx, top_idx, None],
            ).float()
        return y.to(hidden_states.dtype)


class DSV4SharedExperts(BaseOP):
    def __init__(self, config: ModelConfig):
        intermediate = config.moe_intermediate_size * max(config.n_shared_experts, 1)
        self.swiglu_limit = config.swiglu_limit or 0.0
        self.gate_up_proj = DSV4Linear(
            config.hidden_size,
            2 * intermediate,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
            col_parallel=True,
        )
        self.down_proj = DSV4Linear(
            intermediate,
            config.hidden_size,
            weight_dtype=_fp8_dtype(),
            scale_dtype=_e8m0_dtype(),
            row_parallel=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        gate_f = gate.float()
        up_f = up.float()
        if self.swiglu_limit > 0:
            up_f = torch.clamp(up_f, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate_f = torch.clamp(gate_f, max=self.swiglu_limit)
        return self.down_proj.forward((F.silu(gate_f) * up_f).to(up.dtype))


class DSV4MoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        is_hash_layer = layer_id < config.n_hash_layers
        self.topk_count = config.num_experts_per_tok
        self.scoring_func = config.scoring_func or "sqrtsoftplus"
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = DSV4MoEGate(config, has_correction_bias=not is_hash_layer)
        if is_hash_layer:
            self.topk = DSV4TopK(config)
        self.experts = DSV4FusedRoutedExperts(config)
        if config.n_shared_experts > 0:
            self.shared_experts = DSV4SharedExperts(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        weights, indices = self.gate.forward(
            flat,
            input_ids=input_ids.view(-1),
            topk=self.topk_count,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            hash_topk=getattr(self, "topk", None),
        )
        y = self.experts.forward(flat, weights, indices).float()
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts.forward(flat).float()
        return y.to(flat.dtype).view_as(hidden_states)


class DeepseekV4DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.self_attn = DSV4Attention(config, layer_id)
        self.mlp = DSV4MoE(config, layer_id)
        self.input_layernorm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)

        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def _hc_pre(
        self,
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, fn.float()) * rsqrt
        pre, post, comb = _hc_split_sinkhorn_ref(
            mixes,
            scale,
            base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.to(x.dtype).unsqueeze(-1) * x.view(shape), dim=1)
        return y, post.to(x.dtype), comb.to(x.dtype)

    def _hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return post.unsqueeze(-1) * x.unsqueeze(1) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(2), dim=1
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        residual = x
        y, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        y = self.self_attn.forward(self.input_layernorm.forward(y))
        x = self._hc_post(y, residual, post, comb)

        residual = x
        y, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        y = self.mlp.forward(self.post_attention_layernorm.forward(y), input_ids)
        return self._hc_post(y, residual, post, comb)


class DeepseekV4Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.embed_tokens = DSV4VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [DeepseekV4DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = DSV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = torch.empty(config.hc_mult, hc_dim, dtype=torch.float32)
        self.hc_head_base = torch.empty(config.hc_mult, dtype=torch.float32)
        self.hc_head_scale = torch.empty(1, dtype=torch.float32)

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale.float() + self.hc_head_base.float())
        pre = pre + self.hc_eps
        return torch.sum(pre.to(x.dtype).unsqueeze(-1) * x.view(shape), dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers.op_list:
            x = layer.forward(x, input_ids)
        return self.norm.forward(self._hc_head(x))


class DeepseekV4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = DeepseekV4Model(config)
        self.lm_head = DSV4VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        super().__init__()

    def forward(self):
        batch = get_global_ctx().batch
        output = self.model.forward(batch.input_ids)
        if batch.is_prefill:
            output = output[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return self.lm_head.linear(output)


__all__ = ["DeepseekV4ForCausalLM", "DSV4FallbackAttentionMetadata"]

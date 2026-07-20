from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from minisgl.attention import BaseAttnMetadata
from minisgl.attention.deepseek_v4 import DSV4AttentionMetadata
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.dsv4_runtime import get_dsv4_runtime_config
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.layers import BaseOP, OPList
from minisgl.utils import (
    div_ceil,
    div_even,
)

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig






def _marlin_wna16_release_timing() -> str:
    return get_dsv4_runtime_config().marlin_release_timing or "disabled"


def _marlin_wna16_release_deferred_from_model_prepare() -> bool:
    return _marlin_wna16_release_timing() != "model_prepare"






def _cuda_graph_capture_active() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False






















def _cached_hc_bf16_weight(owner: object, cache_name: str, weight: torch.Tensor) -> torch.Tensor:
    if not (
        dsv4_kernel.dsv4_optimized_enabled()
        and weight.is_cuda
    ):
        return weight
    meta_name = f"{cache_name}_meta"
    meta = (
        weight.data_ptr(),
        int(getattr(weight, "_version", 0)),
        weight.device.type,
        weight.device.index,
        tuple(weight.shape),
        tuple(weight.stride()),
    )
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        cached = weight.to(torch.bfloat16).contiguous()
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)
    return cached


def _cached_fp32_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not (
        dsv4_kernel.dsv4_optimized_enabled() and weight.is_cuda and weight.dtype == torch.bfloat16
    ):
        return weight
    meta_name = f"{cache_name}_meta"
    meta = (
        weight.data_ptr(),
        int(getattr(weight, "_version", 0)),
        weight.device.type,
        weight.device.index,
        tuple(weight.shape),
        tuple(weight.stride()),
    )
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        cached = weight.float().contiguous()
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)
    return cached




def _tensor_cache_meta(tensor: torch.Tensor | None) -> tuple | None:
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        int(getattr(tensor, "_version", 0)),
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
    )


def _fp8_bf16_weight_cache_meta(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> tuple:
    return (_tensor_cache_meta(weight), _tensor_cache_meta(scale), out_dtype)


def _cached_fp8_bf16_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    allow_build: bool,
    owner_label: str,
) -> torch.Tensor:
    if weight.dtype != dsv4_kernel.fp8_dtype():
        raise RuntimeError(
            f"{owner_label} cached BF16 weight path requires FP8 weights, got {weight.dtype}."
        )
    if scale is not None and scale.device != weight.device:
        raise RuntimeError(
            f"{owner_label} cached BF16 weight path requires scale on the same device "
            f"as weight, got weight={weight.device} scale={scale.device}."
        )
    if out_dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} cached BF16 weight path requires out_dtype=torch.bfloat16, got {out_dtype}."
        )

    meta_name = f"{cache_name}_meta"
    meta = _fp8_bf16_weight_cache_meta(weight, scale, out_dtype)
    cached = getattr(owner, cache_name, None)
    if cached is not None and getattr(owner, meta_name, None) == meta:
        return cached

    if not allow_build:
        raise RuntimeError(
            f"{owner_label} cached BF16 weight is missing or stale. "
            "Call prepare_for_cuda_graph_capture() after weights are loaded and before "
            "decode CUDA graph capture/replay; rebuilding inside forward is disabled."
        )

    cached = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=out_dtype).contiguous()
    setattr(owner, cache_name, cached)
    setattr(owner, meta_name, meta)
    return cached






def _linear_cached_bf16_weight(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return F.linear(x, weight)


def _wo_a_bf16_bmm_weight_cache_meta(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    num_local_groups: int,
    o_lora_rank: int,
    d_per_group: int,
) -> tuple:
    return (
        _fp8_bf16_weight_cache_meta(weight, scale, out_dtype),
        int(num_local_groups),
        int(o_lora_rank),
        int(d_per_group),
    )


def _cached_wo_a_bf16_bmm_weight(
    owner: object,
    cache_name: str,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    num_local_groups: int,
    o_lora_rank: int,
    d_per_group: int,
    allow_build: bool,
    owner_label: str,
) -> torch.Tensor:
    if num_local_groups <= 0:
        raise RuntimeError(f"{owner_label} BF16 BMM cache requires num_local_groups > 0.")
    if o_lora_rank <= 0:
        raise RuntimeError(f"{owner_label} BF16 BMM cache requires o_lora_rank > 0.")
    if d_per_group <= 0:
        raise RuntimeError(f"{owner_label} BF16 BMM cache requires d_per_group > 0.")
    expected_shape = (num_local_groups * o_lora_rank, d_per_group)
    if tuple(weight.shape) != expected_shape:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache expected FP8 weight shape {expected_shape}, "
            f"got {tuple(weight.shape)}."
        )
    if weight.dtype != dsv4_kernel.fp8_dtype():
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache requires FP8 weights, got {weight.dtype}."
        )
    if scale is not None and scale.device != weight.device:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache requires scale on the same device as weight, "
            f"got weight={weight.device} scale={scale.device}."
        )
    if out_dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache requires out_dtype=torch.bfloat16, got {out_dtype}."
        )

    meta_name = f"{cache_name}_meta"
    meta = _wo_a_bf16_bmm_weight_cache_meta(
        weight,
        scale,
        out_dtype=out_dtype,
        num_local_groups=num_local_groups,
        o_lora_rank=o_lora_rank,
        d_per_group=d_per_group,
    )
    cached = getattr(owner, cache_name, None)
    if cached is not None and getattr(owner, meta_name, None) == meta:
        return cached

    if not allow_build:
        raise RuntimeError(
            f"{owner_label} BF16 BMM cache is missing or stale. "
            "Call prepare_for_cuda_graph_capture() after weights are loaded and before "
            "decode CUDA graph capture/replay; rebuilding inside forward is disabled."
        )

    dequant = dsv4_kernel.dequant_fp8_weight(weight, scale, out_dtype=out_dtype)
    cached = dequant.view(num_local_groups, o_lora_rank, d_per_group).transpose(1, 2).contiguous()
    setattr(owner, cache_name, cached)
    setattr(owner, meta_name, meta)
    return cached


def _wo_a_bf16_bmm_projection(
    o: torch.Tensor,
    cached_weight: torch.Tensor,
    *,
    owner_label: str,
) -> torch.Tensor:
    if o.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection requires bf16 activations, got {o.dtype}."
        )
    if cached_weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection requires bf16 cached weight, "
            f"got {cached_weight.dtype}."
        )
    if o.ndim != 3 or cached_weight.ndim != 3:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection expects o=[tokens, groups, d] and "
            f"weight=[groups, d, rank], got o={tuple(o.shape)} weight={tuple(cached_weight.shape)}."
        )
    tokens, num_local_groups, d_per_group = o.shape
    if cached_weight.shape[0] != num_local_groups or cached_weight.shape[1] != d_per_group:
        raise RuntimeError(
            f"{owner_label} BF16 BMM projection shape mismatch: "
            f"o={tuple(o.shape)} weight={tuple(cached_weight.shape)}."
        )
    x = o.transpose(0, 1).contiguous()
    y = torch.bmm(x, cached_weight)
    return y.transpose(0, 1).reshape(tokens, num_local_groups * cached_weight.shape[2])


def _cached_gate_fp32_weight(owner: object, cache_name: str, weight: torch.Tensor) -> torch.Tensor:
    return _cached_fp32_weight(
        owner,
        cache_name,
        weight,
    )


def _cached_indexer_store_norm_fp32_weight(
    owner: object, cache_name: str, weight: torch.Tensor
) -> torch.Tensor:
    return _cached_fp32_weight(
        owner,
        cache_name,
        weight,
    )


def _cached_fused_wqa_wkv_fp8_weight(
    owner: object,
    cache_name: str,
    weight_q: torch.Tensor,
    scale_q: torch.Tensor | None,
    weight_kv: torch.Tensor,
    scale_kv: torch.Tensor | None,
    *,
    out_dtype: torch.dtype,
    allow_build: bool,
) -> torch.Tensor | None:
    if not (
        dsv4_kernel.dsv4_optimized_enabled()
        and weight_q.is_cuda
        and weight_kv.is_cuda
        and weight_q.dtype is dsv4_kernel.fp8_dtype()
        and weight_kv.dtype is dsv4_kernel.fp8_dtype()
        and out_dtype is torch.bfloat16
        and weight_q.ndim == 2
        and weight_kv.ndim == 2
        and weight_q.shape[-1] == weight_kv.shape[-1]
    ):
        return None
    if scale_q is not None and not scale_q.is_cuda:
        return None
    if scale_kv is not None and not scale_kv.is_cuda:
        return None

    def _tensor_meta(tensor: torch.Tensor | None):
        if tensor is None:
            return None
        return (
            tensor.data_ptr(),
            int(getattr(tensor, "_version", 0)),
            tensor.device.type,
            tensor.device.index,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    meta_name = f"{cache_name}_meta"
    meta = (
        _tensor_meta(weight_q),
        _tensor_meta(scale_q),
        _tensor_meta(weight_kv),
        _tensor_meta(scale_kv),
        out_dtype,
    )
    cached = getattr(owner, cache_name, None)
    if cached is None or getattr(owner, meta_name, None) != meta:
        if not allow_build:
            raise RuntimeError(
                "Fused WQA/WKV cached BF16 weight is missing or stale. Call "
                "prepare_fused_wqa_wkv_bf16_weight_cache() after KV allocation and "
                "before CUDA graph warmup/capture; rebuilding inside forward is disabled."
            )
        q = dsv4_kernel.dequant_fp8_weight(weight_q, scale_q, out_dtype=out_dtype)
        kv = dsv4_kernel.dequant_fp8_weight(weight_kv, scale_kv, out_dtype=out_dtype)
        cached = torch.cat((q, kv), dim=0).contiguous()
        setattr(owner, cache_name, cached)
        setattr(owner, meta_name, meta)
    return cached


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
        return dsv4_kernel.rms_norm_fallback(x, self.weight, eps=self.eps)


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
        return self._comm.all_reduce(y, label="dsv4.embedding_all_reduce")

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.weight.float())
        if self.tp_size == 1:
            return logits[:, : self.num_embeddings]
        gathered = self._comm.all_gather(logits, label="dsv4.lm_head_all_gather")
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
                dsv4_kernel.scale_dim(local_output_size),
                dsv4_kernel.scale_dim(local_input_size),
                dtype=scale_dtype,
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        reduce: bool = True,
        reduce_label: str | None = None,
    ) -> torch.Tensor:
        scale = getattr(self, "weight_scale_inv", None)
        if self.weight.dtype is torch.int8:
            y = dsv4_kernel.quantized_linear_ref(x, self.weight, scale, weight_kind="fp4")
        elif self.weight.dtype is dsv4_kernel.fp8_dtype():
            y = dsv4_kernel.quantized_linear_ref(
                x,
                self.weight,
                scale,
                weight_kind="fp8",
            )
        else:
            y = F.linear(x, self.weight.to(x.dtype))
        if reduce and self.row_parallel and self._tp_size > 1:
            y = self._comm.all_reduce(
                y,
                label=reduce_label or "dsv4.row_parallel_projection_all_reduce",
            )
        return y

    def prepare_fp8_bf16_weight_cache(
        self,
        cache_name: str,
        *,
        owner_label: str,
    ) -> dict[str, object]:
        scale = getattr(self, "weight_scale_inv", None)
        cached = _cached_fp8_bf16_weight(
            self,
            cache_name,
            self.weight,
            scale,
            out_dtype=torch.bfloat16,
            allow_build=True,
            owner_label=owner_label,
        )
        return {
            "owner": owner_label,
            "shape": list(cached.shape),
            "dtype": str(cached.dtype),
            "device": str(cached.device),
            "bytes": int(cached.numel() * cached.element_size()),
        }



    def forward_fp8_cached_bf16_weight(
        self,
        x: torch.Tensor,
        *,
        cache_name: str,
        owner_label: str,
        reduce: bool = False,
        reduce_label: str | None = None,
    ) -> torch.Tensor:
        scale = getattr(self, "weight_scale_inv", None)
        cached_weight = _cached_fp8_bf16_weight(
            self,
            cache_name,
            self.weight,
            scale,
            out_dtype=x.dtype,
            allow_build=False,
            owner_label=owner_label,
        )
        x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
        y = _linear_cached_bf16_weight(
            x_quant,
            cached_weight,
        )
        if reduce and self.row_parallel and self._tp_size > 1:
            y = self._comm.all_reduce(
                y,
                label=reduce_label or "dsv4.row_parallel_projection_all_reduce",
            )
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

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        apply_norm: bool = True,
    ) -> torch.Tensor:
        return dsv4_kernel.compress_forward_fallback(
            x,
            positions,
            ratio=self.ratio,
            head_dim=self.head_dim,
            overlap=self.overlap,
            ape=self.ape,
            wkv_gate=self.wkv_gate,
            norm=self.norm,
            apply_norm=apply_norm,
        )


class DSV4Indexer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.layer_id = layer_id
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.weight_scale = (self.head_dim**-0.5) * (self.n_heads**-0.5)
        self.wq_b = DSV4Linear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.weights_proj = DSV4Linear(
            config.hidden_size,
            self.n_heads,
            weight_dtype=torch.bfloat16,
        )
        self.compressor = DSV4Compressor(config, ratio=4, head_dim=self.head_dim)

    @property
    def _wq_b_bf16_weight_cache_name(self) -> str:
        return "_dsv4_indexer_wq_b_bf16_weight_cache"

    @property
    def _wq_b_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.indexer.wq_b"

    def prepare_wq_b_bf16_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_optimized_enabled():
            return None
        return self.wq_b.prepare_fp8_bf16_weight_cache(
            self._wq_b_bf16_weight_cache_name,
            owner_label=self._wq_b_owner_label,
        )

    def _wq_b_forward(self, q_lora: torch.Tensor) -> torch.Tensor:
        if dsv4_kernel.dsv4_optimized_enabled():
            return self.wq_b.forward_fp8_cached_bf16_weight(
                q_lora,
                cache_name=self._wq_b_bf16_weight_cache_name,
                owner_label=self._wq_b_owner_label,
            )
        return self.wq_b.forward(q_lora)

    def prepare_bf16_query(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        *,
        rotary_dim: int,
        base: float,
        original_seq_len: int,
        factor: float,
        beta_fast: int,
        beta_slow: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._wq_b_forward(q_lora).view(-1, self.n_heads, self.head_dim)
        q = dsv4_kernel.indexer_q_rope_hadamard_bf16_fallback(
            q,
            positions,
            rotary_dim=rotary_dim,
            base=base,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        weights = self.weights_proj.forward(x) * self.weight_scale
        return q, weights

    def prepare_fp8_query(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        *,
        rotary_dim: int,
        base: float,
        original_seq_len: int,
        factor: float,
        beta_fast: int,
        beta_slow: int,
    ) -> dsv4_kernel.DSV4IndexerFP8Query:
        q = self._wq_b_forward(q_lora).view(-1, self.n_heads, self.head_dim)
        weights = self.weights_proj.forward(x)
        return dsv4_kernel.indexer_q_rope_fp8_fallback(
            q,
            weights,
            positions,
            rotary_dim=rotary_dim,
            base=base,
            softmax_scale=self.head_dim**-0.5,
            head_scale=self.n_heads**-0.5,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        *,
        apply_norm: bool = True,
        touch_projections: bool = True,
    ) -> torch.Tensor:
        compressed_kv = self.compressor.forward(x, positions, apply_norm=apply_norm)
        if touch_projections:
            self._wq_b_forward(q_lora)
            self.weights_proj.forward(x)
        return compressed_kv


class DSV4Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp = get_tp_info()
        self.layer_id = layer_id
        self.num_heads = config.num_qo_heads
        self.num_local_heads = div_even(config.num_qo_heads, tp.size)
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self._produce_swa_qat = self.head_dim - self.rope_head_dim == 448
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
        self.attn_sink = torch.empty(self.num_local_heads, dtype=torch.float32)
        self.wq_a = DSV4Linear(
            config.hidden_size,
            config.q_lora_rank,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.wq_b = DSV4Linear(
            config.q_lora_rank,
            config.num_qo_heads * config.head_dim,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            col_parallel=True,
        )
        self.q_norm = DSV4RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.wkv = DSV4Linear(
            config.hidden_size,
            config.head_dim,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
        )
        self.kv_norm = DSV4RMSNorm(config.head_dim, config.rms_norm_eps)
        self.wo_a = DSV4Linear(
            config.num_qo_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            col_parallel=True,
        )
        self.wo_b = DSV4Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            row_parallel=True,
        )

        if ratio in (4, 128):
            self.compressor = DSV4Compressor(config, ratio=ratio, head_dim=config.head_dim)
        if ratio == 4:
            self.indexer = DSV4Indexer(config, layer_id)

    @staticmethod
    def _swa_store_out_loc(attn_backend, batch: Batch | torch.Tensor) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            out_loc = batch
            metadata = None
        else:
            out_loc = batch.out_loc
            metadata = getattr(batch, "attn_metadata", None)
        if isinstance(metadata, DSV4AttentionMetadata):
            cached = getattr(metadata.core_metadata, "swa_out_loc", None)
            rows = int(out_loc.shape[0])
            if cached is not None and int(cached.shape[0]) >= rows:
                return cached[:rows]
        kvcache = getattr(attn_backend, "kvcache", None)
        translate = getattr(kvcache, "translate_full_locs_to_swa_locs", None)
        if callable(translate) and bool(
            getattr(kvcache, "swa_independent_lifecycle_enabled", False)
        ):
            return translate(out_loc).to(device=out_loc.device, dtype=out_loc.dtype)
        return out_loc

    @property
    def _q_wqb_bf16_weight_cache_name(self) -> str:
        return "_dsv4_q_wqb_bf16_weight_cache"

    @property
    def _q_wqb_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.q_wqb"

    @property
    def _wo_b_bf16_weight_cache_name(self) -> str:
        return "_dsv4_wo_b_bf16_weight_cache"

    @property
    def _wo_b_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.wo_b"

    @property
    def _wo_a_bf16_bmm_cache_name(self) -> str:
        return "_dsv4_wo_a_bf16_bmm_weight_cache"

    @property
    def _wo_a_owner_label(self) -> str:
        return f"layer{self.layer_id}.attn.wo_a"

    def _wo_a_d_per_group(self) -> int:
        return self.num_local_heads * self.head_dim // self.num_local_groups

    def prepare_q_wqb_bf16_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_optimized_enabled():
            return None
        return self.wq_b.prepare_fp8_bf16_weight_cache(
            self._q_wqb_bf16_weight_cache_name,
            owner_label=self._q_wqb_owner_label,
        )


    def prepare_wo_b_bf16_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_optimized_enabled():
            return None
        return self.wo_b.prepare_fp8_bf16_weight_cache(
            self._wo_b_bf16_weight_cache_name,
            owner_label=self._wo_b_owner_label,
        )


    def prepare_wo_a_bf16_bmm_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_optimized_enabled():
            return None
        scale = getattr(self.wo_a, "weight_scale_inv", None)
        d_per_group = self._wo_a_d_per_group()
        cached = _cached_wo_a_bf16_bmm_weight(
            self.wo_a,
            self._wo_a_bf16_bmm_cache_name,
            self.wo_a.weight,
            scale,
            out_dtype=torch.bfloat16,
            num_local_groups=self.num_local_groups,
            o_lora_rank=self.o_lora_rank,
            d_per_group=d_per_group,
            allow_build=True,
            owner_label=self._wo_a_owner_label,
        )
        return {
            "owner": self._wo_a_owner_label,
            "shape": list(cached.shape),
            "source_weight_shape": list(self.wo_a.weight.shape),
            "scale_shape": list(scale.shape) if scale is not None else None,
            "dtype": str(cached.dtype),
            "device": str(cached.device),
            "bytes": int(cached.numel() * cached.element_size()),
            "num_local_groups": int(self.num_local_groups),
            "d_per_group": int(d_per_group),
            "o_lora_rank": int(self.o_lora_rank),
        }

    def prepare_indexer_wq_b_bf16_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_optimized_enabled():
            return None
        if not hasattr(self, "indexer"):
            return None
        return self.indexer.prepare_wq_b_bf16_weight_cache()

    def prepare_fused_wqa_wkv_bf16_weight_cache(self) -> dict[str, object] | None:
        if not dsv4_kernel.dsv4_optimized_enabled():
            return None
        cached = _cached_fused_wqa_wkv_fp8_weight(
            self,
            "_cached_fused_wqa_wkv_bf16_weight",
            self.wq_a.weight,
            getattr(self.wq_a, "weight_scale_inv", None),
            self.wkv.weight,
            getattr(self.wkv, "weight_scale_inv", None),
            out_dtype=torch.bfloat16,
            allow_build=True,
        )
        if cached is None:
            return None
        owner_label = f"layer{self.layer_id}.attn.q_proj"
        return {
            "owner": owner_label,
            "shape": list(cached.shape),
            "dtype": str(cached.dtype),
            "device": str(cached.device),
            "bytes": int(cached.numel() * cached.element_size()),
        }

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
        spans = self._sequence_spans(batch, q.shape[0])
        return dsv4_kernel.sequence_mqa_attention_fallback(
            q,
            kv,
            spans,
            window_size=self.window_size,
            softmax_scale=self.softmax_scale,
            attn_sink=self.attn_sink,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        positions = batch.positions.to(device=x.device, dtype=torch.long)
        attn_backend = getattr(get_global_ctx(), "attn_backend", None)
        attn_metadata = getattr(batch, "attn_metadata", None)
        use_dsv4_backend = isinstance(attn_metadata, DSV4AttentionMetadata)
        kv_norm_rope_store_enabled = (
            use_dsv4_backend
            and attn_backend is not None
            and x.is_cuda
            and dsv4_kernel.dsv4_optimized_triton_enabled()
        )
        fused_q_kv_rmsnorm = (
            False
            and not kv_norm_rope_store_enabled
        )
        fused_q_kv_norm_rope_store = (
            kv_norm_rope_store_enabled
            and dsv4_kernel.dsv4_optimized_triton_enabled()
        )
        kv_from_shared_wqa_wkv = None
        kv = None
        fused_wqa_wkv_shared_act = dsv4_kernel.dsv4_optimized_triton_enabled()
        if fused_wqa_wkv_shared_act:
            cached_fused_weight = _cached_fused_wqa_wkv_fp8_weight(
                self,
                "_cached_fused_wqa_wkv_bf16_weight",
                self.wq_a.weight,
                getattr(self.wq_a, "weight_scale_inv", None),
                self.wkv.weight,
                getattr(self.wkv, "weight_scale_inv", None),
                out_dtype=x.dtype,
                allow_build=False,
            )
            if cached_fused_weight is not None:
                x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
                qkv = _linear_cached_bf16_weight(
                    x_quant,
                    cached_fused_weight,
                )
                q_lora_raw, kv_from_shared_wqa_wkv = qkv.split(
                    [self.q_norm.weight.shape[0], self.head_dim],
                    dim=-1,
                )
            else:
                q_lora_raw, kv_from_shared_wqa_wkv = (
                    dsv4_kernel.quantized_linear_fp8_pair_shared_activation_ref(
                        x,
                        self.wq_a.weight,
                        getattr(self.wq_a, "weight_scale_inv", None),
                        self.wkv.weight,
                        getattr(self.wkv, "weight_scale_inv", None),
                    )
                )
        else:
            q_lora_raw = self.wq_a.forward(x)
        if not fused_q_kv_rmsnorm:
            q_lora = self.q_norm.forward(q_lora_raw)
        if fused_q_kv_rmsnorm:
            kv = (
                kv_from_shared_wqa_wkv
                if kv_from_shared_wqa_wkv is not None
                else self.wkv.forward(x)
            )
            q_lora, kv = None
        elif dsv4_kernel.dsv4_optimized_enabled():
            q = self.wq_b.forward_fp8_cached_bf16_weight(
                q_lora,
                cache_name=self._q_wqb_bf16_weight_cache_name,
                owner_label=self._q_wqb_owner_label,
            ).view(-1, self.num_local_heads, self.head_dim)
        else:
            q = self.wq_b.forward(q_lora).view(-1, self.num_local_heads, self.head_dim)
        if fused_q_kv_norm_rope_store and kv is None:
            kv = (
                kv_from_shared_wqa_wkv
                if kv_from_shared_wqa_wkv is not None
                else self.wkv.forward(x)
            )
        q_kv_norm_rope_cache_written = False
        kv_qat_completed = False
        if fused_q_kv_norm_rope_store and kv is not None:
            q_kv_norm_rope_cache_written = dsv4_kernel.q_kv_norm_rope_cache_fallback(
                q,
                kv,
                positions,
                norm_weight=self.kv_norm.weight,
                rms_norm_eps=self.rms_norm_eps,
                cache=attn_backend.kvcache.swa_cache(self.layer_id),
                out_loc=self._swa_store_out_loc(attn_backend, batch),
                rotary_dim=self.rope_head_dim,
                base=float(self.rope_base),
                original_seq_len=self.original_seq_len,
                factor=self.rope_factor,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
                publish_swa_qat=self._produce_swa_qat,
            )
            kv_qat_completed = q_kv_norm_rope_cache_written and self._produce_swa_qat
        if not q_kv_norm_rope_cache_written:
            dsv4_kernel.q_norm_rope_fallback(
                q,
                positions,
                rms_norm_eps=self.rms_norm_eps,
                rotary_dim=self.rope_head_dim,
                base=float(self.rope_base),
                original_seq_len=self.original_seq_len,
                factor=self.rope_factor,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
            )

        if kv is None:
            kv = (
                kv_from_shared_wqa_wkv
                if kv_from_shared_wqa_wkv is not None
                else self.wkv.forward(x)
            )
        kv_cache_written = False
        if kv_norm_rope_store_enabled:
            if not q_kv_norm_rope_cache_written:
                dsv4_kernel.k_norm_rope_cache_fallback(
                    kv,
                    positions,
                    norm_weight=self.kv_norm.weight,
                    rms_norm_eps=self.rms_norm_eps,
                    cache=attn_backend.kvcache.swa_cache(self.layer_id),
                    out_loc=self._swa_store_out_loc(attn_backend, batch),
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                    publish_swa_qat=self._produce_swa_qat,
                )
                kv_qat_completed = self._produce_swa_qat
            kv_cache_written = True
        else:
            if not fused_q_kv_rmsnorm:
                kv = self.kv_norm.forward(kv)
            dsv4_kernel.k_norm_rope_cache_fallback(
                kv,
                positions,
                rotary_dim=self.rope_head_dim,
                base=float(self.rope_base),
                original_seq_len=self.original_seq_len,
                factor=self.rope_factor,
                beta_fast=self.beta_fast,
                beta_slow=self.beta_slow,
            )
        if self.rope_head_dim < kv.shape[-1] and not kv_qat_completed:
            kv[..., : -self.rope_head_dim] = dsv4_kernel.quantize_fp8_activation_ref(
                kv[..., : -self.rope_head_dim], block_size=64
            )

        compress_store_fuses_norm = (
            use_dsv4_backend
            and attn_backend is not None
            and dsv4_kernel.dsv4_optimized_triton_enabled()
        )

        if hasattr(self, "indexer"):
            indexer_select_fp8 = (
                use_dsv4_backend
                and attn_backend is not None
                and dsv4_kernel.dsv4_optimized_enabled()
            )
            indexer_select_bf16 = (
                not indexer_select_fp8
                and use_dsv4_backend
                and attn_backend is not None
                and dsv4_kernel.dsv4_optimized_triton_enabled()
            )
            indexer_q = None
            indexer_weights = None
            indexer_fp8_query = None
            if indexer_select_fp8:
                indexer_fp8_query = self.indexer.prepare_fp8_query(
                    x,
                    q_lora,
                    positions,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )
            if indexer_select_bf16:
                indexer_q, indexer_weights = self.indexer.prepare_bf16_query(
                    x,
                    q_lora,
                    positions,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )
            online_c4 = (
                use_dsv4_backend
                and attn_backend is not None
                and hasattr(attn_backend, "forward_compress")
                and dsv4_kernel.dsv4_optimized_triton_enabled()
            )
            if online_c4:
                indexer_kv = attn_backend.forward_compress(
                    self.layer_id,
                    x,
                    batch,
                    self.indexer.compressor,
                    component="indexer",
                )
                if not (indexer_select_bf16 or indexer_select_fp8):
                    self.indexer._wq_b_forward(q_lora)
                    self.indexer.weights_proj.forward(x)
            else:
                indexer_kv = self.indexer.forward(
                    x,
                    q_lora,
                    positions,
                    apply_norm=not compress_store_fuses_norm,
                    touch_projections=not (indexer_select_bf16 or indexer_select_fp8),
                )
            if use_dsv4_backend and hasattr(attn_backend, "store_indexer"):
                indexer_store_norm_weight = None
                if compress_store_fuses_norm:
                    indexer_store_norm_weight = _cached_indexer_store_norm_fp32_weight(
                        self.indexer.compressor.norm,
                        "_dsv4_indexer_store_norm_fp32_weight",
                        self.indexer.compressor.norm.weight,
                    )
                attn_backend.store_indexer(
                    self.layer_id,
                    indexer_kv,
                    batch,
                    norm_weight=indexer_store_norm_weight,
                    rms_norm_eps=self.rms_norm_eps if compress_store_fuses_norm else None,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                    apply_hadamard=online_c4 or indexer_select_bf16,
                )
            if indexer_select_fp8 and indexer_fp8_query is not None:
                if not hasattr(attn_backend, "select_indexer_fp8"):
                    raise RuntimeError(
                        "Optimized DSV4 FP8 indexer cache requires an attention "
                        "backend with select_indexer_fp8."
                    )
                attn_backend.select_indexer_fp8(
                    self.layer_id,
                    indexer_fp8_query.q_values,
                    indexer_fp8_query.weights,
                    batch,
                )
            if (
                indexer_select_bf16
                and indexer_q is not None
                and indexer_weights is not None
                and hasattr(attn_backend, "select_indexer")
            ):
                attn_backend.select_indexer(
                    self.layer_id,
                    indexer_q,
                    indexer_weights,
                    batch,
                )
        if hasattr(self, "compressor"):
            online_compress = (
                self.compress_ratio in (4, 128)
                and use_dsv4_backend
                and attn_backend is not None
                and hasattr(attn_backend, "forward_compress")
                and dsv4_kernel.dsv4_optimized_triton_enabled()
            )
            if online_compress:
                compressed_kv = attn_backend.forward_compress(
                    self.layer_id,
                    x,
                    batch,
                    self.compressor,
                    component="attention",
                )
            else:
                compressed_kv = self.compressor.forward(
                    x,
                    positions,
                    apply_norm=not compress_store_fuses_norm,
                )
            if use_dsv4_backend and hasattr(attn_backend, "store_compressed"):
                attn_backend.store_compressed(
                    self.layer_id,
                    compressed_kv,
                    batch,
                    self.compress_ratio,
                    norm_weight=(
                        self.compressor.norm.weight if compress_store_fuses_norm else None
                    ),
                    rms_norm_eps=self.rms_norm_eps if compress_store_fuses_norm else None,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )

        if use_dsv4_backend and attn_backend is not None:
            o = attn_backend.forward(
                q,
                kv,
                kv,
                self.layer_id,
                batch,
                compress_ratio=self.compress_ratio,
                attn_sink=self.attn_sink,
                swa_cache_written=kv_cache_written,
            )
        else:
            o = self._fallback_attention(q, kv, batch)
        dsv4_kernel.apply_rotary_tail(
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
        d_per_group = self._wo_a_d_per_group()
        o = o.reshape(x.shape[0], self.num_local_groups, d_per_group)
        if dsv4_kernel.dsv4_optimized_enabled():
            wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
            cached_wo_a = _cached_wo_a_bf16_bmm_weight(
                self.wo_a,
                self._wo_a_bf16_bmm_cache_name,
                self.wo_a.weight,
                wo_a_scale,
                out_dtype=o.dtype,
                num_local_groups=self.num_local_groups,
                o_lora_rank=self.o_lora_rank,
                d_per_group=d_per_group,
                allow_build=False,
                owner_label=self._wo_a_owner_label,
            )
            o = _wo_a_bf16_bmm_projection(
                o,
                cached_wo_a,
                owner_label=self._wo_a_owner_label,
            )
        else:
            wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
            o = dsv4_kernel.wo_a_grouped_projection_fallback(
                o,
                self.wo_a.weight,
                wo_a_scale,
                num_local_groups=self.num_local_groups,
                o_lora_rank=self.o_lora_rank,
            )
        if dsv4_kernel.dsv4_optimized_enabled():
            out = self.wo_b.forward_fp8_cached_bf16_weight(
                o,
                cache_name=self._wo_b_bf16_weight_cache_name,
                owner_label=self._wo_b_owner_label,
                reduce=True,
                reduce_label="dsv4.attn.wo_b.row_parallel_projection_all_reduce",
            )
            return out
        out = self.wo_b.forward(o)
        return out


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
        num_token_non_padded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = _cached_gate_fp32_weight(self, "_cached_gate_weight_fp32", self.weight)
        return dsv4_kernel.moe_gate_fallback(
            hidden_states,
            weight,
            input_ids=input_ids,
            topk=topk,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=getattr(self, "e_score_correction_bias", None),
            hash_topk=hash_topk,
            num_token_non_padded=num_token_non_padded,
        )


class DSV4FusedRoutedExperts(BaseOP):
    def __init__(self, config: ModelConfig, *, layer_id: int | None = None):
        tp = get_tp_info()
        self.layer_id = layer_id
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
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
            dtype=dsv4_kernel.e8m0_dtype(),
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
            dtype=dsv4_kernel.e8m0_dtype(),
        )
        self._moe_v2_workspace = dsv4_kernel.DSV4MoEWorkspace()
        self._marlin_wna16_weights = None
        self._marlin_wna16_released_original_expert_weights = False
        self._marlin_wna16_source_bytes = 0
        self._marlin_wna16_released_original_expert_bytes = 0

    @property
    def _marlin_owner_label(self) -> str:
        if self.layer_id is None:
            return "moe.routed_experts.marlin_wna16"
        return f"layer{self.layer_id}.moe.routed_experts.marlin_wna16"

    def _released_raw_weight_error(self, *, missing: list[str] | None = None) -> str:
        suffix = f" owner={self._marlin_owner_label}."
        if missing:
            suffix = f" owner={self._marlin_owner_label}; missing={missing}."
        return f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR}{suffix}"

    def _raw_expert_weight_names(self) -> tuple[str, str, str, str]:
        return (
            "w13_weight",
            "w13_weight_scale_inv",
            "w2_weight",
            "w2_weight_scale_inv",
        )

    def _missing_raw_expert_weights(self) -> list[str]:
        return [name for name in self._raw_expert_weight_names() if not hasattr(self, name)]

    def _raw_expert_weight_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = self._missing_raw_expert_weights()
        if missing:
            raise RuntimeError(self._released_raw_weight_error(missing=missing))
        return (
            self.w13_weight,
            self.w13_weight_scale_inv,
            self.w2_weight,
            self.w2_weight_scale_inv,
        )

    def _marlin_cache_tensors(self) -> dict[str, torch.Tensor | None]:
        cache = self._marlin_wna16_weights
        return {
            "w13": getattr(cache, "w13", None),
            "w13_scale": getattr(cache, "w13_scale", None),
            "w2": getattr(cache, "w2", None),
            "w2_scale": getattr(cache, "w2_scale", None),
        }



    def _marlin_cache_report(
        self,
        *,
        source_bytes: int,
        released: list[dict[str, object]],
        already_present: bool,
        signature_match_before: bool | None,
        elapsed_ms: float,
    ) -> dict[str, object]:
        cache_tensors = self._marlin_cache_tensors()
        persistent_bytes = int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in cache_tensors.values()
                if tensor is not None
            )
        )
        source_bytes = int(source_bytes)
        if source_bytes:
            self._marlin_wna16_source_bytes = source_bytes
        else:
            source_bytes = int(self._marlin_wna16_source_bytes)
        released_this_call_bytes = int(sum(int(item["bytes"]) for item in released))
        if released_this_call_bytes:
            self._marlin_wna16_released_original_expert_bytes = released_this_call_bytes
        return {
            "owner": self._marlin_owner_label,
            "layer_id": self.layer_id,
            "already_present": bool(already_present),
            "signature_match_before": signature_match_before,
            "elapsed_ms": elapsed_ms,
            "persistent_bytes": persistent_bytes,
            "source_bytes": source_bytes,
            "released_original_bytes": int(self._marlin_wna16_released_original_expert_bytes),
            "released_original_this_call_bytes": released_this_call_bytes,
            "released_original": bool(self._marlin_wna16_released_original_expert_weights),
            "raw_weights_available_after": not self._missing_raw_expert_weights(),
            "runtime_policy": (
                "marlin_wna16_prepacked_only"
                if self._marlin_wna16_released_original_expert_weights
                else "raw_weights_available"
            ),
            "fallback_error": (
                dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR
                if self._marlin_wna16_released_original_expert_weights
                else None
            ),
            "cache_tensors": {
                name: (
                    {
                        "present": True,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        "bytes": tensor.numel() * tensor.element_size(),
                    }
                    if tensor is not None
                    else {"present": False}
                )
                for name, tensor in cache_tensors.items()
            },
            "released": released,
        }

    def prepare_marlin_wna16_weight_cache(
        self,
        *,
        release_original: bool = False,
    ) -> dict[str, object]:
        from minisgl.kernel import marlin_wna16

        existing_cache = self._marlin_wna16_weights
        raw_available = not self._missing_raw_expert_weights()
        if not raw_available:
            if existing_cache is None:
                raise RuntimeError(self._released_raw_weight_error())
            return self._marlin_cache_report(
                source_bytes=0,
                released=[],
                already_present=True,
                signature_match_before=None,
                elapsed_ms=0.0,
            )

        w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
        source_tensors = {
            "w13_weight": w13_weight,
            "w13_weight_scale_inv": w13_scale,
            "w2_weight": w2_weight,
            "w2_weight_scale_inv": w2_scale,
        }
        source_bytes = int(
            sum(tensor.numel() * tensor.element_size() for tensor in source_tensors.values())
        )
        self._marlin_wna16_source_bytes = source_bytes
        signature_match_before = (
            existing_cache.matches(w13_weight, w13_scale, w2_weight, w2_scale)
            if existing_cache is not None
            else False
        )
        start_s = time.perf_counter()
        if existing_cache is None or not signature_match_before:
            self._marlin_wna16_weights = marlin_wna16.prepare_moe_mxfp4_weights(
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                params_dtype=torch.bfloat16,
                owner_label=self._marlin_owner_label,
                cache_was_present=existing_cache is not None,
                cache_signature_match=signature_match_before,
            )
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0

        released: list[dict[str, object]] = []
        if release_original:
            return self.release_marlin_wna16_original_expert_weights(
                already_present=existing_cache is not None,
                signature_match_before=signature_match_before,
                elapsed_ms=elapsed_ms,
            )

        return self._marlin_cache_report(
            source_bytes=source_bytes,
            released=released,
            already_present=existing_cache is not None,
            signature_match_before=signature_match_before,
            elapsed_ms=elapsed_ms,
        )

    def release_marlin_wna16_original_expert_weights(
        self,
        *,
        already_present: bool = True,
        signature_match_before: bool | None = True,
        elapsed_ms: float = 0.0,
    ) -> dict[str, object]:
        if self._marlin_wna16_weights is None:
            raise RuntimeError(
                f"{self._marlin_owner_label} cannot release original expert weights "
                "before Marlin WNA16 cache is built."
            )
        if self._missing_raw_expert_weights():
            self._marlin_wna16_released_original_expert_weights = True
            return self._marlin_cache_report(
                source_bytes=0,
                released=[],
                already_present=already_present,
                signature_match_before=signature_match_before,
                elapsed_ms=elapsed_ms,
            )

        w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
        if not self._marlin_wna16_weights.matches(w13_weight, w13_scale, w2_weight, w2_scale):
            raise RuntimeError(
                f"{self._marlin_owner_label} cannot release original expert weights because "
                "the prebuilt Marlin WNA16 cache signature does not match the live source tensors."
            )
        source_tensors = {
            "w13_weight": w13_weight,
            "w13_weight_scale_inv": w13_scale,
            "w2_weight": w2_weight,
            "w2_weight_scale_inv": w2_scale,
        }
        source_bytes = int(
            sum(tensor.numel() * tensor.element_size() for tensor in source_tensors.values())
        )
        self._marlin_wna16_source_bytes = source_bytes
        if any(tensor.is_cuda for tensor in source_tensors.values()):
            # The release preset immediately makes source storage reusable for KV/cache
            # allocation, so make the post-load repack boundary explicit.
            torch.cuda.synchronize(w13_weight.device)
        released: list[dict[str, object]] = []
        release_names = self._marlin_wna16_release_attribute_names()
        for name, tensor in list(source_tensors.items()):
            if name not in release_names:
                continue
            released.append(
                {
                    "attribute": name,
                    "component": name,
                    "layer_id": self.layer_id,
                    "data_ptr": tensor.data_ptr(),
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "released": True,
                }
            )
            delattr(self, name)
        self._marlin_wna16_released_original_expert_weights = True
        self._marlin_wna16_released_original_expert_bytes = int(
            sum(int(item["bytes"]) for item in released)
        )
        return self._marlin_cache_report(
            source_bytes=source_bytes,
            released=released,
            already_present=already_present,
            signature_match_before=signature_match_before,
            elapsed_ms=elapsed_ms,
        )

    def _marlin_wna16_release_attribute_names(self) -> set[str]:
        return set(self._raw_expert_weight_names())



    def _expert_forward(
        self, local_idx: int, x: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        w1 = dsv4_kernel.quantized_linear_ref(
            x,
            self.w13_weight[local_idx, 0],
            self.w13_weight_scale_inv[local_idx, 0],
            weight_kind="fp4",
        ).float()
        w3 = dsv4_kernel.quantized_linear_ref(
            x,
            self.w13_weight[local_idx, 1],
            self.w13_weight_scale_inv[local_idx, 1],
            weight_kind="fp4",
        ).float()
        hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            w1,
            w3,
            swiglu_limit=self.swiglu_limit,
            weights=weights,
        )
        hidden_for_w2 = hidden.to(x.dtype)
        return dsv4_kernel.quantized_linear_ref(
            hidden_for_w2,
            self.w2_weight[local_idx],
            self.w2_weight_scale_inv[local_idx],
            weight_kind="fp4",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *,
        reduce: bool = True,
        moe_plan: dsv4_kernel.DSV4MoEExecutionPlan | None = None,
    ) -> torch.Tensor:
        backend = dsv4_kernel.require_supported_moe_expert_backend()
        missing_raw_weights = self._missing_raw_expert_weights()
        if missing_raw_weights and backend != dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16:
            raise RuntimeError(
                f"{self._released_raw_weight_error(missing=missing_raw_weights)} "
                f"requested_backend={backend!r}."
            )
        if backend == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16:
            if missing_raw_weights:
                if self._marlin_wna16_weights is None:
                    raise RuntimeError(
                        f"{self._released_raw_weight_error(missing=missing_raw_weights)} "
                        "Marlin WNA16 prebuilt cache is missing."
                    )
                grouped = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16_prepacked(
                    hidden_states,
                    weights,
                    indices,
                    self._marlin_wna16_weights,
                    swiglu_limit=self.swiglu_limit,
                    moe_plan=moe_plan,
                )
            else:
                w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
                grouped, self._marlin_wna16_weights = (
                    dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16(
                        hidden_states,
                        weights,
                        indices,
                        w13_weight,
                        w13_scale,
                        w2_weight,
                        w2_scale,
                        swiglu_limit=self.swiglu_limit,
                        cache=self._marlin_wna16_weights,
                        owner_label=self._marlin_owner_label,
                        moe_plan=moe_plan,
                    )
                )
            if reduce and self._tp_size > 1:
                grouped_for_reduce = grouped.float()
                grouped_reduced = self._comm.all_reduce(
                    grouped_for_reduce,
                    label="dsv4.routed_expert_all_reduce",
                )
                grouped = grouped_reduced.to(grouped.dtype)
            return grouped

        workspace = None
        if (
            moe_plan is not None
            and moe_plan.route_plan.route_count <= dsv4_kernel.DSV4_SM80_MOE_V2_WORKSPACE_MAX_ROUTES
        ):
            workspace = self._moe_v2_workspace
        w13_weight, w13_scale, w2_weight, w2_scale = self._raw_expert_weight_tensors()
        grouped = dsv4_kernel.moe_route_dispatch_bf16_grouped(
            hidden_states,
            weights,
            indices,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            swiglu_limit=self.swiglu_limit,
            moe_plan=moe_plan,
            workspace=workspace,
        )
        if grouped is not None:
            if reduce and self._tp_size > 1:
                grouped_for_reduce = grouped.float()
                grouped_reduced = self._comm.all_reduce(
                    grouped_for_reduce,
                    label="dsv4.routed_expert_all_reduce",
                )
                grouped = grouped_reduced.to(grouped.dtype)
            return grouped

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
        if reduce and self._tp_size > 1:
            y = self._comm.all_reduce(y, label="dsv4.routed_expert_all_reduce")
        return y.to(hidden_states.dtype)


class DSV4SharedExperts(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int | None = None):
        self.layer_id = layer_id
        intermediate = config.moe_intermediate_size * max(config.n_shared_experts, 1)
        self.swiglu_limit = config.swiglu_limit or 0.0
        self.gate_up_proj = DSV4Linear(
            config.hidden_size,
            2 * intermediate,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            col_parallel=True,
        )
        self.down_proj = DSV4Linear(
            intermediate,
            config.hidden_size,
            weight_dtype=dsv4_kernel.fp8_dtype(),
            scale_dtype=dsv4_kernel.e8m0_dtype(),
            row_parallel=True,
        )

    @property
    def _gate_up_bf16_weight_cache_name(self) -> str:
        return "_dsv4_shared_gate_up_bf16_weight_cache"

    @property
    def _down_bf16_weight_cache_name(self) -> str:
        return "_dsv4_shared_down_bf16_weight_cache"

    @property
    def _gate_up_owner_label(self) -> str:
        if self.layer_id is None:
            return "shared_experts.gate_up_proj"
        return f"layer{self.layer_id}.shared_experts.gate_up_proj"

    @property
    def _down_owner_label(self) -> str:
        if self.layer_id is None:
            return "shared_experts.down_proj"
        return f"layer{self.layer_id}.shared_experts.down_proj"

    def prepare_bf16_weight_cache(self) -> list[dict[str, object]]:
        if not dsv4_kernel.dsv4_optimized_enabled():
            return []
        reports = [
            self.gate_up_proj.prepare_fp8_bf16_weight_cache(
                self._gate_up_bf16_weight_cache_name,
                owner_label=self._gate_up_owner_label,
            ),
        ]
        reports.append(
            self.down_proj.prepare_fp8_bf16_weight_cache(
                self._down_bf16_weight_cache_name,
                owner_label=self._down_owner_label,
            )
        )
        return reports


    def prepare_down_bf16_weight_cache(self) -> dict[str, object]:
        return self.down_proj.prepare_fp8_bf16_weight_cache(
            self._down_bf16_weight_cache_name,
            owner_label=self._down_owner_label,
        )

    def forward(self, hidden_states: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        use_bf16_weight_cache = dsv4_kernel.dsv4_optimized_enabled()
        if use_bf16_weight_cache:
            gate_up = self.gate_up_proj.forward_fp8_cached_bf16_weight(
                hidden_states,
                cache_name=self._gate_up_bf16_weight_cache_name,
                owner_label=self._gate_up_owner_label,
            )
        else:
            gate_up = self.gate_up_proj.forward(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            gate,
            up,
            swiglu_limit=self.swiglu_limit,
        )
        hidden_for_down = hidden.to(up.dtype)
        if use_bf16_weight_cache:
            return self.down_proj.forward_fp8_cached_bf16_weight(
                hidden_for_down,
                cache_name=self._down_bf16_weight_cache_name,
                owner_label=self._down_owner_label,
                reduce=reduce,
                reduce_label="dsv4.shared_expert_all_reduce",
            )
        return self.down_proj.forward(
            hidden_for_down,
            reduce=reduce,
            reduce_label="dsv4.shared_expert_all_reduce",
        )


def _dsv4_moe_reduce_once_input(
    output: torch.Tensor,
    *,
    hidden_dtype: torch.dtype,
    layer_id: int,
    path: str,
) -> torch.Tensor:
    if (
        hidden_dtype == torch.bfloat16
        and output.dtype != torch.bfloat16
        and dsv4_kernel.dsv4_optimized_enabled()
    ):
        return output.to(torch.bfloat16)
    return output


def _moe_num_token_non_padded(flat: torch.Tensor) -> torch.Tensor:
    """Return the graph-bound live-row scalar, or an exact eager scalar."""
    try:
        batch = get_global_ctx().batch
        value = getattr(batch, "num_token_non_padded", None)
    except AssertionError:
        value = None
    if value is not None:
        return value
    return torch.tensor([flat.shape[0]], dtype=torch.int32, device=flat.device)


@dataclass(frozen=True)
class DSV4FusedMoERunnerPrepareResult:
    weights: torch.Tensor
    indices: torch.Tensor
    moe_plan: dsv4_kernel.DSV4MoEExecutionPlan


class DSV4FusedMoERunner:
    """Mini-owned exact-path runner shaped after vLLM's standard FusedMoE runner."""

    def __init__(
        self,
        *,
        layer_id: int,
        gate: DSV4MoEGate,
        experts: DSV4FusedRoutedExperts,
        shared_experts: DSV4SharedExperts | None,
        topk_count: int,
        scoring_func: str,
        routed_scaling_factor: float,
        tp_size: int,
    ) -> None:
        self.layer_id = layer_id
        self.gate = gate
        self.experts = experts
        self.shared_experts = shared_experts
        self.topk_count = topk_count
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self._tp_size = tp_size

    def route(
        self,
        flat: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        hash_topk: DSV4TopK | None,
        num_token_non_padded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate.forward(
            flat,
            input_ids=input_ids,
            topk=self.topk_count,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            hash_topk=hash_topk,
            num_token_non_padded=num_token_non_padded,
        )

    def prepare(
        self,
        flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> DSV4FusedMoERunnerPrepareResult:
        if hasattr(self.experts, "w13_weight"):
            num_experts = self.experts.w13_weight.shape[0]
        elif self.experts._marlin_wna16_weights is not None:
            num_experts = self.experts._marlin_wna16_weights.w13.shape[0]
        else:
            raise RuntimeError(
                f"layer{self.layer_id}.moe runner cannot build a route plan because "
                f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR} "
                "Marlin WNA16 cache is missing."
            )
        moe_plan = dsv4_kernel.build_moe_v2_execution_plan(
            flat,
            weights,
            indices,
            num_experts=num_experts,
            block_size_m=dsv4_kernel.moe_execution_block_size(
                tokens=flat.shape[0],
                topk=indices.shape[1],
                num_experts=num_experts,
            ),
            reduce_once=True,
        )
        return DSV4FusedMoERunnerPrepareResult(
            weights=weights,
            indices=indices,
            moe_plan=moe_plan,
        )

    def apply_experts(
        self,
        flat: torch.Tensor,
        prepared: DSV4FusedMoERunnerPrepareResult,
    ) -> torch.Tensor:
        return self.experts.forward(
            flat,
            prepared.weights,
            prepared.indices,
            reduce=False,
            moe_plan=prepared.moe_plan,
        )

    def finalize_routed(self, routed_output: torch.Tensor) -> torch.Tensor:
        # The current grouped FP4 backend already applies top-k weights and
        # sums routes to [tokens, hidden]. Keep the boundary explicit so a
        # future exact backend can return per-route output here.
        return routed_output.float()

    def apply_shared(self, flat: torch.Tensor) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        shared = self.shared_experts.forward(flat, reduce=False)
        return shared.float()

    def maybe_reduce_final(
        self,
        output: torch.Tensor,
        *,
        comm: DistributedCommunicator,
        hidden_dtype: torch.dtype,
        reduce_label: str,
    ) -> torch.Tensor:
        if self._tp_size > 1:
            output = _dsv4_moe_reduce_once_input(
                output,
                hidden_dtype=hidden_dtype,
                layer_id=self.layer_id,
                path="runner_output",
            )
            return comm.all_reduce(output, label=reduce_label)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        comm: DistributedCommunicator,
        hash_topk: DSV4TopK | None,
    ) -> torch.Tensor:
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        flat_input_ids = input_ids.view(-1)
        num_token_non_padded = _moe_num_token_non_padded(flat)
        weights, indices = self.route(
            flat,
            flat_input_ids,
            hash_topk=hash_topk,
            num_token_non_padded=num_token_non_padded,
        )
        prepared = self.prepare(flat, weights, indices)
        routed = self.apply_experts(flat, prepared)
        y = self.finalize_routed(routed)
        shared = self.apply_shared(flat)
        if shared is not None:
            y = y + shared
        y = dsv4_kernel.zero_moe_padded_rows(y, num_token_non_padded)
        y = self.maybe_reduce_final(
            y,
            comm=comm,
            hidden_dtype=flat.dtype,
            reduce_label=prepared.moe_plan.final_reduce_label,
        )
        return y.to(flat.dtype).view_as(hidden_states)


class DSV4MoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        tp = get_tp_info()
        self.layer_id = layer_id
        self._tp_size = tp.size
        self._comm = DistributedCommunicator()
        is_hash_layer = layer_id < config.n_hash_layers
        self.topk_count = config.num_experts_per_tok
        self.scoring_func = config.scoring_func or "sqrtsoftplus"
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = DSV4MoEGate(config, has_correction_bias=not is_hash_layer)
        if is_hash_layer:
            self.topk = DSV4TopK(config)
        self.experts = DSV4FusedRoutedExperts(config, layer_id=layer_id)
        if config.n_shared_experts > 0:
            self.shared_experts = DSV4SharedExperts(config, layer_id=layer_id)
        self._runner = DSV4FusedMoERunner(
            layer_id=layer_id,
            gate=self.gate,
            experts=self.experts,
            shared_experts=getattr(self, "shared_experts", None),
            topk_count=self.topk_count,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            tp_size=self._tp_size,
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if dsv4_kernel.dsv4_optimized_enabled():
            return self._runner.forward(
                hidden_states,
                input_ids,
                comm=self._comm,
                hash_topk=getattr(self, "topk", None),
            )

        flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_token_non_padded = _moe_num_token_non_padded(flat)
        weights, indices = self.gate.forward(
            flat,
            input_ids=input_ids.view(-1),
            topk=self.topk_count,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            hash_topk=getattr(self, "topk", None),
            num_token_non_padded=num_token_non_padded,
        )
        moe_v2 = dsv4_kernel.dsv4_optimized_enabled()
        reduce_once = moe_v2 or dsv4_kernel.dsv4_optimized_enabled()
        moe_plan = None
        if moe_v2:
            if hasattr(self.experts, "w13_weight"):
                num_experts = self.experts.w13_weight.shape[0]
            elif self.experts._marlin_wna16_weights is not None:
                num_experts = self.experts._marlin_wna16_weights.w13.shape[0]
            else:
                raise RuntimeError(
                    f"layer{self.layer_id}.moe cannot build a route plan because "
                    f"{dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR} "
                    "Marlin WNA16 cache is missing."
                )
            moe_plan = dsv4_kernel.build_moe_v2_execution_plan(
                flat,
                weights,
                indices,
                num_experts=num_experts,
                block_size_m=dsv4_kernel.moe_execution_block_size(
                    tokens=flat.shape[0],
                    topk=indices.shape[1],
                    num_experts=num_experts,
                ),
                reduce_once=reduce_once,
            )
        if moe_plan is None:
            y = self.experts.forward(
                flat, weights, indices, reduce=not reduce_once
            ).float()
        else:
            y = self.experts.forward(
                flat,
                weights,
                indices,
                reduce=not reduce_once,
                moe_plan=moe_plan,
            ).float()
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts.forward(
                flat, reduce=not reduce_once
            ).float()
        y = dsv4_kernel.zero_moe_padded_rows(y, num_token_non_padded)
        if reduce_once and self._tp_size > 1:
            y = _dsv4_moe_reduce_once_input(
                y,
                hidden_dtype=flat.dtype,
                layer_id=self.layer_id,
                path="non_runner_output",
            )
            y = self._comm.all_reduce(
                y, label="dsv4.v1_moe_reduce_once_all_reduce"
            )
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
        self._hc_attn_fn_bf16: torch.Tensor | None = None
        self._hc_attn_fn_bf16_meta: tuple | None = None
        self._hc_ffn_fn_bf16: torch.Tensor | None = None
        self._hc_ffn_fn_bf16_meta: tuple | None = None

    def _hc_pre(
        self,
        x: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return dsv4_kernel.hc_pre_fallback(
            x,
            fn,
            scale,
            base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
            norm_eps=self.norm_eps,
        )

    def _hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return dsv4_kernel.hc_post_fallback(x, residual, post, comb)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        residual = x
        attn_fn = _cached_hc_bf16_weight(self, "_hc_attn_fn_bf16", self.hc_attn_fn)
        y, post, comb = self._hc_pre(
            x, attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        y = self.input_layernorm.forward(y)
        y = self.self_attn.forward(y)
        x = self._hc_post(y, residual, post, comb)

        residual = x
        ffn_fn = _cached_hc_bf16_weight(self, "_hc_ffn_fn_bf16", self.hc_ffn_fn)
        y, post, comb = self._hc_pre(
            x, ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        y = self.post_attention_layernorm.forward(y)
        y = self.mlp.forward(y, input_ids)
        output = self._hc_post(y, residual, post, comb)
        return output


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
        self._hc_head_fn_bf16: torch.Tensor | None = None
        self._hc_head_fn_bf16_meta: tuple | None = None

    def prepare_for_cuda_graph_capture(self) -> dict[str, object]:
        q_wqb_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_optimized_enabled():
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_q_wqb_bf16_weight_cache()
                if report is not None:
                    q_wqb_reports.append(report)

        wo_b_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_optimized_enabled():
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_wo_b_bf16_weight_cache()
                if report is not None:
                    wo_b_reports.append(report)

        wo_a_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_optimized_enabled():
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_wo_a_bf16_bmm_cache()
                if report is not None:
                    wo_a_reports.append(report)
        indexer_wq_b_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_optimized_enabled():
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_indexer_wq_b_bf16_weight_cache()
                if report is not None:
                    indexer_wq_b_reports.append(report)
        shared_expert_reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_optimized_enabled():
            for layer in self.layers.op_list:
                shared_experts = getattr(layer.mlp, "shared_experts", None)
                if shared_experts is not None:
                    shared_expert_reports.extend(shared_experts.prepare_bf16_weight_cache())

        moe_marlin_wna16_reports: list[dict[str, object]] = []
        moe_marlin_wna16_prebuild_reports: list[dict[str, object]] = []
        moe_marlin_wna16_release_reports: list[dict[str, object]] = []
        runtime = get_dsv4_runtime_config()
        moe_marlin_backend = dsv4_kernel.dsv4_moe_expert_backend()
        moe_marlin_prebuild_enabled = runtime.marlin_prebuild
        moe_marlin_release_original = runtime.release_raw_expert_weights
        moe_marlin_release_timing = _marlin_wna16_release_timing()
        moe_marlin_release_deferred = _marlin_wna16_release_deferred_from_model_prepare()
        if moe_marlin_release_original and not moe_marlin_prebuild_enabled:
            raise RuntimeError(
                "optimized raw-expert release requires Marlin WNA16 prebuild so the "
                "Marlin WNA16 cache exists before original expert weights are released."
            )
        if (
            moe_marlin_release_original
            and moe_marlin_backend != dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16
        ):
            raise RuntimeError(
                "optimized raw-expert release requires "
                f"backend={dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16!r}, "
                f"got {moe_marlin_backend!r}."
            )
        if (
            moe_marlin_prebuild_enabled
            and moe_marlin_backend == dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16
        ):
            for layer in self.layers.op_list:
                moe_marlin_wna16_prebuild_reports.append(
                    layer.mlp.experts.prepare_marlin_wna16_weight_cache(
                        release_original=False,
                    )
                )
            moe_marlin_wna16_reports = moe_marlin_wna16_prebuild_reports
            if moe_marlin_release_original and not moe_marlin_release_deferred:
                moe_marlin_wna16_release_reports = self.release_marlin_wna16_original_expert_weights(
                    stage_label="model_prepare_release",
                )["entries"]
                moe_marlin_wna16_reports = moe_marlin_wna16_release_reports
        total_q_wqb_bytes = int(sum(int(report["bytes"]) for report in q_wqb_reports))
        total_wo_b_bytes = int(sum(int(report["bytes"]) for report in wo_b_reports))
        total_indexer_wq_b_bytes = int(sum(int(report["bytes"]) for report in indexer_wq_b_reports))
        total_wo_a_bytes = int(sum(int(report["bytes"]) for report in wo_a_reports))
        total_shared_expert_bytes = int(
            sum(int(report["bytes"]) for report in shared_expert_reports)
        )






        total_moe_marlin_wna16_persistent_bytes = int(
            sum(int(report["persistent_bytes"]) for report in moe_marlin_wna16_reports)
        )
        total_moe_marlin_wna16_source_bytes = int(
            sum(int(report["source_bytes"]) for report in moe_marlin_wna16_reports)
        )
        total_moe_marlin_wna16_released_bytes = int(
            sum(int(report["released_original_bytes"]) for report in moe_marlin_wna16_reports)
        )
        projection_cache_owners = []
        if q_wqb_reports:
            projection_cache_owners.append("attn.q_wqb")
        if wo_b_reports:
            projection_cache_owners.append("attn.wo_b")
        if indexer_wq_b_reports:
            projection_cache_owners.append("indexer.wq_b")
        if wo_a_reports:
            projection_cache_owners.append("attn.wo_a")
        if shared_expert_reports:
            if any(
                str(report["owner"]).endswith("gate_up_proj") for report in shared_expert_reports
            ):
                projection_cache_owners.append("shared_experts.gate_up_proj")
            if any(str(report["owner"]).endswith("down_proj") for report in shared_expert_reports):
                projection_cache_owners.append("shared_experts.down_proj")
        return {
            "q_wqb_bf16_weight_cache": {
                "enabled": bool(q_wqb_reports),
                "layers_cached": len(q_wqb_reports),
                "total_bytes": total_q_wqb_bytes,
                "entries": q_wqb_reports,
            },
            "wo_b_bf16_weight_cache": {
                "enabled": bool(wo_b_reports),
                "layers_cached": len(wo_b_reports),
                "total_bytes": total_wo_b_bytes,
                "entries": wo_b_reports,
            },
            "wo_a_bf16_bmm_cache": {
                "enabled": bool(wo_a_reports),
                "layers_cached": len(wo_a_reports),
                "total_bytes": total_wo_a_bytes,
                "entries": wo_a_reports,
            },
            "indexer_wq_b_bf16_weight_cache": {
                "enabled": bool(indexer_wq_b_reports),
                "layers_cached": len(indexer_wq_b_reports),
                "total_bytes": total_indexer_wq_b_bytes,
                "entries": indexer_wq_b_reports,
            },
            "shared_expert_bf16_weight_cache": {
                "enabled": bool(shared_expert_reports),
                "layers_cached": max(
                    sum(
                        1
                        for report in shared_expert_reports
                        if str(report["owner"]).endswith("gate_up_proj")
                    ),
                    sum(
                        1
                        for report in shared_expert_reports
                        if str(report["owner"]).endswith("down_proj")
                    ),
                ),
                "total_bytes": total_shared_expert_bytes,
                "entries": shared_expert_reports,
            },
            "projection_bf16_weight_cache_total": {
                "total_bytes": (
                    total_q_wqb_bytes
                    + total_wo_b_bytes
                    + total_indexer_wq_b_bytes
                    + total_wo_a_bytes
                    + total_shared_expert_bytes
                ),
                "owners": projection_cache_owners,
            },
            "moe_marlin_wna16_cache": {
                "enabled": bool(moe_marlin_wna16_reports),
                "backend": moe_marlin_backend,
                "prebuild_requested": bool(moe_marlin_prebuild_enabled),
                "release_original_requested": bool(moe_marlin_release_original),
                "release_timing": moe_marlin_release_timing,
                "release_deferred_from_model_prepare": bool(moe_marlin_release_deferred),
                "layers_cached": len(moe_marlin_wna16_reports),
                "total_persistent_bytes": total_moe_marlin_wna16_persistent_bytes,
                "total_source_bytes": total_moe_marlin_wna16_source_bytes,
                "total_released_original_bytes": total_moe_marlin_wna16_released_bytes,
                "release_runtime_policy": (
                    "marlin_wna16_prepacked_only"
                    if (
                        moe_marlin_release_original
                        and not moe_marlin_release_deferred
                        and bool(moe_marlin_wna16_reports)
                    )
                    else None
                ),
                "fail_closed_error": (
                    dsv4_kernel.DSV4_MARLIN_WNA16_RELEASE_FALLBACK_ERROR
                    if moe_marlin_release_original
                    else None
                ),
                "prebuild_entries": moe_marlin_wna16_prebuild_reports,
                "release_entries": moe_marlin_wna16_release_reports,
                "entries": moe_marlin_wna16_reports,
            },
        }

    def prepare_fused_wqa_wkv_bf16_weight_cache(self) -> dict[str, object]:
        reports: list[dict[str, object]] = []
        if dsv4_kernel.dsv4_optimized_enabled():
            for layer in self.layers.op_list:
                report = layer.self_attn.prepare_fused_wqa_wkv_bf16_weight_cache()
                if report is not None:
                    reports.append(report)
        return {
            "enabled": bool(reports),
            "layers_cached": len(reports),
            "total_bytes": int(sum(int(report["bytes"]) for report in reports)),
            "entries": reports,
        }

    def release_marlin_wna16_original_expert_weights(
        self,
        *,
        stage_label: str,
    ) -> dict[str, object]:
        release_reports: list[dict[str, object]] = []
        for layer in self.layers.op_list:
            experts = getattr(layer.mlp, "experts", None)
            if experts is None:
                continue
            release_reports.append(experts.release_marlin_wna16_original_expert_weights())

        def _released_this_call(report: dict[str, object]) -> int:
            return int(
                report.get(
                    "released_original_this_call_bytes",
                    report.get("released_original_bytes", 0),
                )
            )

        return {
            "stage_label": stage_label,
            "entries": release_reports,
            "layers_released": len(release_reports),
            "total_released_original_bytes": int(
                sum(int(report["released_original_bytes"]) for report in release_reports)
            ),
            "total_released_original_this_call_bytes": int(
                sum(_released_this_call(report) for report in release_reports)
            ),
        }









    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        hc_head_fn = _cached_hc_bf16_weight(self, "_hc_head_fn_bf16", self.hc_head_fn)
        return dsv4_kernel.hc_head_fallback(
            x,
            hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            eps=self.hc_eps,
            norm_eps=self.norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers.op_list:
            int(getattr(layer.self_attn, "layer_id", -1))
            x = layer.forward(x, input_ids)
        x = self._hc_head(x)
        x = self.norm.forward(x)
        return x


class DeepseekV4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = DeepseekV4Model(config)
        self.lm_head = DSV4VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        super().__init__()

    def prepare_for_cuda_graph_capture(self) -> dict[str, object]:
        return self.model.prepare_for_cuda_graph_capture()

    def prepare_fused_wqa_wkv_bf16_weight_cache(self) -> dict[str, object]:
        return self.model.prepare_fused_wqa_wkv_bf16_weight_cache()

    def release_marlin_wna16_original_expert_weights(
        self,
        *,
        stage_label: str,
    ) -> dict[str, object]:
        return self.model.release_marlin_wna16_original_expert_weights(stage_label=stage_label)





    def forward(self):
        batch = get_global_ctx().batch
        output = self.model.forward(batch.input_ids)
        if batch.is_prefill:
            output = output[
                batch.attn_metadata.get_last_indices(batch.size)
            ].contiguous()
        logits = self.lm_head.linear(output)
        return logits


__all__ = ["DeepseekV4ForCausalLM", "DSV4FallbackAttentionMetadata", "DSV4AttentionMetadata"]

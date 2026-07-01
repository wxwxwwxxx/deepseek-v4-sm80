from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from minisgl.attention import BaseAttnMetadata
from minisgl.attention.deepseek_v4 import DSV4AttentionMetadata
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers import BaseOP, OPList
from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.utils import div_ceil, div_even

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig


def _dsv4_capture_nvtx(name: str):
    if not dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_GRAPH_CAPTURE_NVTX"):
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.nvtx.range(f"dsv4.{name}")


def _cached_hc_bf16_weight(owner: object, cache_name: str, weight: torch.Tensor) -> torch.Tensor:
    if not (dsv4_kernel.linear_bf16_fp32_upstream_enabled() and weight.is_cuda):
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
    *,
    toggle: str,
) -> torch.Tensor:
    if not (
        dsv4_kernel.dsv4_env_flag(toggle)
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
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


def _cached_gate_fp32_weight(owner: object, cache_name: str, weight: torch.Tensor) -> torch.Tensor:
    return _cached_fp32_weight(
        owner,
        cache_name,
        weight,
        toggle="MINISGL_DSV4_SM80_GATE_FP32_WEIGHT_CACHE",
    )


def _cached_indexer_store_norm_fp32_weight(
    owner: object, cache_name: str, weight: torch.Tensor
) -> torch.Tensor:
    return _cached_fp32_weight(
        owner,
        cache_name,
        weight,
        toggle="MINISGL_DSV4_SM80_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE",
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
) -> torch.Tensor | None:
    if not (
        dsv4_kernel.dsv4_env_flag("MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE")
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
        fp8_gemm: bool | None = None,
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
                fp8_gemm=fp8_gemm,
            )
        else:
            y = F.linear(x, self.weight.to(x.dtype))
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
        _ = layer_id
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

    def _wq_b_forward(self, q_lora: torch.Tensor) -> torch.Tensor:
        fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
            "MINISGL_DSV4_SM80_INDEXER_WQB_FP8_GEMM"
        )
        return self.wq_b.forward(q_lora, fp8_gemm=fp8_gemm if fp8_gemm else None)

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
            and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_KV_BF16")
        )
        fused_q_kv_rmsnorm = (
            dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_FUSED_Q_KV_RMSNORM"
            )
            and not kv_norm_rope_store_enabled
        )
        fused_q_kv_norm_rope_store = (
            kv_norm_rope_store_enabled
            and dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE"
            )
        )
        kv_from_shared_wqa_wkv = None
        kv = None
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_proj"):
            q_wqa_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_Q_WQA_FP8_GEMM"
            )
            fused_wqa_wkv_shared_act = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT"
            )
            if fused_wqa_wkv_shared_act:
                cached_fused_weight = _cached_fused_wqa_wkv_fp8_weight(
                    self,
                    "_cached_fused_wqa_wkv_bf16_weight",
                    self.wq_a.weight,
                    getattr(self.wq_a, "weight_scale_inv", None),
                    self.wkv.weight,
                    getattr(self.wkv, "weight_scale_inv", None),
                    out_dtype=x.dtype,
                )
                if cached_fused_weight is not None:
                    x_quant = dsv4_kernel.quantize_fp8_activation_ref(x)
                    qkv = F.linear(x_quant, cached_fused_weight)
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
                q_lora_raw = self.wq_a.forward(
                    x,
                    fp8_gemm=q_wqa_fp8_gemm if q_wqa_fp8_gemm else None,
                )
            if not fused_q_kv_rmsnorm:
                q_lora = self.q_norm.forward(q_lora_raw)
        if fused_q_kv_rmsnorm:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_kv_rmsnorm"):
                q_lora, kv = dsv4_kernel.rms_norm_pair_fallback(
                    q_lora_raw,
                    kv,
                    self.q_norm.weight,
                    self.kv_norm.weight,
                    eps=self.rms_norm_eps,
                )
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_wqb"):
            q_wqb_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_Q_WQB_FP8_GEMM"
            )
            q = self.wq_b.forward(
                q_lora,
                fp8_gemm=q_wqb_fp8_gemm if q_wqb_fp8_gemm else None,
            ).view(-1, self.num_local_heads, self.head_dim)
        if fused_q_kv_norm_rope_store and kv is None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
        q_kv_norm_rope_cache_written = False
        if fused_q_kv_norm_rope_store and kv is not None:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_kv_norm_rope_store"):
                q_kv_norm_rope_cache_written = dsv4_kernel.q_kv_norm_rope_cache_fallback(
                    q,
                    kv,
                    positions,
                    norm_weight=self.kv_norm.weight,
                    rms_norm_eps=self.rms_norm_eps,
                    cache=attn_backend.kvcache.swa_cache(self.layer_id),
                    out_loc=batch.out_loc,
                    rotary_dim=self.rope_head_dim,
                    base=float(self.rope_base),
                    original_seq_len=self.original_seq_len,
                    factor=self.rope_factor,
                    beta_fast=self.beta_fast,
                    beta_slow=self.beta_slow,
                )
        if not q_kv_norm_rope_cache_written:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.q_norm_rope"):
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
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_proj"):
                kv = (
                    kv_from_shared_wqa_wkv
                    if kv_from_shared_wqa_wkv is not None
                    else self.wkv.forward(x)
                )
        kv_cache_written = False
        if kv_norm_rope_store_enabled:
            if not q_kv_norm_rope_cache_written:
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_norm_rope_store"):
                    dsv4_kernel.k_norm_rope_cache_fallback(
                        kv,
                        positions,
                        norm_weight=self.kv_norm.weight,
                        rms_norm_eps=self.rms_norm_eps,
                        cache=attn_backend.kvcache.swa_cache(self.layer_id),
                        out_loc=batch.out_loc,
                        rotary_dim=self.rope_head_dim,
                        base=float(self.rope_base),
                        original_seq_len=self.original_seq_len,
                        factor=self.rope_factor,
                        beta_fast=self.beta_fast,
                        beta_slow=self.beta_slow,
                    )
            kv_cache_written = True
        else:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_norm_rope"):
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
        if self.rope_head_dim < kv.shape[-1]:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.kv_quant"):
                kv[..., : -self.rope_head_dim] = dsv4_kernel.quantize_fp8_activation_ref(
                    kv[..., : -self.rope_head_dim], block_size=64
                )

        compress_store_fuses_norm = (
            use_dsv4_backend
            and attn_backend is not None
            and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_COMPRESS_STORE")
        )

        if hasattr(self, "indexer"):
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.indexer"):
                indexer_select_bf16 = (
                    use_dsv4_backend
                    and attn_backend is not None
                    and dsv4_kernel.dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_INDEXER_BF16")
                )
                indexer_q = None
                indexer_weights = None
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
                indexer_kv = self.indexer.forward(
                    x,
                    q_lora,
                    positions,
                    apply_norm=not compress_store_fuses_norm,
                    touch_projections=not indexer_select_bf16,
                )
            if use_dsv4_backend and hasattr(attn_backend, "store_indexer"):
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.indexer_store"):
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
                        apply_hadamard=indexer_select_bf16,
                    )
            if (
                indexer_select_bf16
                and indexer_q is not None
                and indexer_weights is not None
                and hasattr(attn_backend, "select_indexer")
            ):
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.indexer_select"):
                    attn_backend.select_indexer(
                        self.layer_id,
                        indexer_q,
                        indexer_weights,
                        batch,
                    )
        if hasattr(self, "compressor"):
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.compress"):
                compressed_kv = self.compressor.forward(
                    x,
                    positions,
                    apply_norm=not compress_store_fuses_norm,
                )
            if use_dsv4_backend and hasattr(attn_backend, "store_compressed"):
                with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.compress_store"):
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
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.backend"):
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
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.fallback_backend"):
                o = self._fallback_attention(q, kv, batch)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.o_rope"):
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
        d_per_group = self.num_local_heads * self.head_dim // self.num_local_groups
        o = o.reshape(x.shape[0], self.num_local_groups, d_per_group)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.wo_a"):
            o = dsv4_kernel.wo_a_grouped_projection_fallback(
                o,
                self.wo_a.weight,
                getattr(self.wo_a, "weight_scale_inv", None),
                num_local_groups=self.num_local_groups,
                o_lora_rank=self.o_lora_rank,
            )
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.attn.wo_b"):
            wo_b_fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
                "MINISGL_DSV4_SM80_WO_B_FP8_GEMM"
            )
            return self.wo_b.forward(
                o,
                fp8_gemm=wo_b_fp8_gemm if wo_b_fp8_gemm else None,
            )


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
        )


class DSV4FusedRoutedExperts(BaseOP):
    def __init__(self, config: ModelConfig):
        tp = get_tp_info()
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

    def _expert_forward(self, local_idx: int, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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
        return dsv4_kernel.quantized_linear_ref(
            hidden.to(x.dtype),
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
    ) -> torch.Tensor:
        grouped = dsv4_kernel.moe_route_dispatch_bf16_grouped(
            hidden_states,
            weights,
            indices,
            self.w13_weight,
            self.w13_weight_scale_inv,
            self.w2_weight,
            self.w2_weight_scale_inv,
            swiglu_limit=self.swiglu_limit,
        )
        if grouped is not None:
            if reduce and self._tp_size > 1:
                grouped = self._comm.all_reduce(
                    grouped.float(),
                    label="dsv4.routed_expert_all_reduce",
                ).to(grouped.dtype)
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
    def __init__(self, config: ModelConfig):
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

    def forward(self, hidden_states: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        fp8_gemm = dsv4_kernel.dsv4_sm80_triton_enabled(
            "MINISGL_DSV4_SM80_SHARED_FP8_GEMM"
        )
        gate_up = self.gate_up_proj.forward(
            hidden_states,
            fp8_gemm=fp8_gemm if fp8_gemm else None,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            gate,
            up,
            swiglu_limit=self.swiglu_limit,
        )
        return self.down_proj.forward(
            hidden.to(up.dtype),
            reduce=reduce,
            reduce_label="dsv4.shared_expert_all_reduce",
            fp8_gemm=fp8_gemm if fp8_gemm else None,
        )


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
        self.experts = DSV4FusedRoutedExperts(config)
        if config.n_shared_experts > 0:
            self.shared_experts = DSV4SharedExperts(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.gate"):
            weights, indices = self.gate.forward(
                flat,
                input_ids=input_ids.view(-1),
                topk=self.topk_count,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                hash_topk=getattr(self, "topk", None),
            )
        reduce_once = dsv4_kernel.dsv4_env_flag(dsv4_kernel.DSV4_SM80_V1_MOE_TOGGLE)
        with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.routed"):
            y = self.experts.forward(flat, weights, indices, reduce=not reduce_once).float()
        if hasattr(self, "shared_experts"):
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.shared"):
                y = y + self.shared_experts.forward(flat, reduce=not reduce_once).float()
        if reduce_once and self._tp_size > 1:
            with _dsv4_capture_nvtx(f"layer{self.layer_id}.mlp.reduce_once"):
                y = self._comm.all_reduce(y, label="dsv4.v1_moe_reduce_once_all_reduce")
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
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_attn_pre"):
            y, post, comb = self._hc_pre(x, attn_fn, self.hc_attn_scale, self.hc_attn_base)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.attn_input_norm"):
            y = self.input_layernorm.forward(y)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.attn"):
            y = self.self_attn.forward(y)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_attn_post"):
            x = self._hc_post(y, residual, post, comb)

        residual = x
        ffn_fn = _cached_hc_bf16_weight(self, "_hc_ffn_fn_bf16", self.hc_ffn_fn)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_ffn_pre"):
            y, post, comb = self._hc_pre(x, ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.mlp_input_norm"):
            y = self.post_attention_layernorm.forward(y)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.mlp"):
            y = self.mlp.forward(y, input_ids)
        with _dsv4_capture_nvtx(f"layer{self.self_attn.layer_id}.hc_ffn_post"):
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
        self._hc_head_fn_bf16: torch.Tensor | None = None
        self._hc_head_fn_bf16_meta: tuple | None = None

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
        with _dsv4_capture_nvtx("model.embed"):
            x = self.embed_tokens.forward(input_ids)
        with _dsv4_capture_nvtx("model.hc_expand"):
            x = x.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers.op_list:
            x = layer.forward(x, input_ids)
        with _dsv4_capture_nvtx("model.hc_head"):
            x = self._hc_head(x)
        with _dsv4_capture_nvtx("model.final_norm"):
            return self.norm.forward(x)


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
        with _dsv4_capture_nvtx("lm_head"):
            return self.lm_head.linear(output)


__all__ = ["DeepseekV4ForCausalLM", "DSV4FallbackAttentionMetadata", "DSV4AttentionMetadata"]

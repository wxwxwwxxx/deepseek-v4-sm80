"""DeepSeek V4 fused-kernel wrapper boundary.

The functions in this module are correctness-first fallbacks with the same
semantic boundaries as SGLang's DSV4 fused kernels.  High-performance sm80
ports should replace internals here instead of leaking optional kernel imports
into model or attention code.
"""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F

from minisgl.utils import div_ceil


WeightKind = Literal["bf16", "fp8", "fp4"]
KernelStatus = Literal["native", "fallback", "unsupported", "todo"]
DSV4KernelMode = Literal["fallback", "bf16_direct", "fp8_act", "fp4_act"]


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DSV4KernelInventoryEntry:
    wrapper: str
    source_function: str
    expected_inputs: str
    expected_outputs: str
    mini_call_site: str
    sm80_behavior: str
    optional_dependencies: tuple[str, ...]
    status: KernelStatus
    port_target: str


DSV4_KERNEL_INVENTORY: tuple[DSV4KernelInventoryEntry, ...] = (
    DSV4KernelInventoryEntry(
        "apply_rotary_tail",
        "sglang.jit_kernel.dsv4.fused_rope_inplace",
        "x[..., rotary_dim], positions[tokens]",
        "x updated in-place and returned",
        "models.deepseek_v4.DSV4Attention",
        "torch fallback",
        ("triton", "sglang JIT"),
        "fallback",
        "Triton/TileLang sm80 rope kernel",
    ),
    DSV4KernelInventoryEntry(
        "q_norm_rope_fallback",
        "sglang.jit_kernel.dsv4.fused_q_norm_rope",
        "q[tokens, heads, dim], positions[tokens]",
        "RMS-normalized q with RoPE tail",
        "models.deepseek_v4.DSV4Attention",
        "torch fallback",
        ("triton", "sglang JIT"),
        "fallback",
        "Triton sm80 fused RMSNorm+RoPE",
    ),
    DSV4KernelInventoryEntry(
        "fused_q_indexer_rope_first_quant",
        "sglang.jit_kernel.dsv4.fused_q_indexer_rope_first_quant",
        "indexer q, positions, quant scale buffers",
        "quantized first-stage indexer q",
        "future indexer fast path",
        "explicit NotImplementedError",
        ("sgl_kernel", "triton"),
        "todo",
        "sm80 indexer quant kernel",
    ),
    DSV4KernelInventoryEntry(
        "fused_q_indexer_rope_hadamard_quant",
        "sglang.jit_kernel.dsv4.fused_q_indexer_rope_hadamard_quant",
        "indexer q, positions, hadamard scale buffers",
        "fp8 quantized indexer q",
        "future DSV4Indexer fast path",
        "explicit NotImplementedError",
        ("sgl_kernel", "triton"),
        "unsupported",
        "sm80 fp8 indexer quant kernel",
    ),
    DSV4KernelInventoryEntry(
        "fused_q_indexer_rope_hadamard_fp4_quant",
        "sglang.jit_kernel.dsv4.fused_q_indexer_rope_hadamard_fp4_quant",
        "indexer q, positions, fp4/hadamard scale buffers",
        "fp4 quantized indexer q",
        "future DSV4Indexer fast path",
        "sm100-only path is blocked on sm80",
        ("sgl_kernel", "cutlass", "triton"),
        "unsupported",
        "rewrite fp4 indexer quant for sm80 or keep bf16 indexer fallback",
    ),
    DSV4KernelInventoryEntry(
        "k_norm_rope_cache_fallback",
        "sglang.jit_kernel.dsv4.fused_k_norm_rope_flashmla",
        "kv[tokens, dim], positions[tokens], cache locations",
        "normalized/rotated k and cache writes",
        "models.deepseek_v4.DSV4Attention + attention.deepseek_v4",
        "bf16 torch cache fallback",
        ("sgl_kernel.flash_mla", "flashinfer"),
        "fallback",
        "sm80 FlashMLA-compatible bf16/fp8 cache path",
    ),
    DSV4KernelInventoryEntry(
        "norm_rope_inplace_fallback",
        "sglang.jit_kernel.dsv4.fused_norm_rope_inplace",
        "kv[tokens, dim], positions[tokens]",
        "normalized/rotated compressed kv",
        "models.deepseek_v4.DSV4Compressor",
        "torch fallback inside compressor",
        ("triton", "sglang JIT"),
        "fallback",
        "sm80 fused compressor norm+RoPE",
    ),
    DSV4KernelInventoryEntry(
        "store_swa_fallback",
        "sglang.jit_kernel.dsv4.fused_store_cache",
        "kv[tokens, dim], out_loc[tokens]",
        "KV cache updated",
        "attention.deepseek_v4.DSV4AttentionBackend.forward",
        "bf16 torch cache write",
        ("sgl_kernel", "triton"),
        "fallback",
        "sm80 fused/paged store cache",
    ),
    DSV4KernelInventoryEntry(
        "triton_create_paged_compress_data",
        "sglang.jit_kernel.dsv4.triton_create_paged_compress_data",
        "positions, sequence lengths, compress ratio",
        "paged compressor write metadata",
        "attention.deepseek_v4 metadata builder",
        "torch metadata construction",
        ("triton",),
        "fallback",
        "sm80 Triton metadata kernel if CPU overhead matters",
    ),
    DSV4KernelInventoryEntry(
        "compressor_plan_fallback",
        "sglang.jit_kernel.dsv4.CompressorDecodePlan/CompressorPrefillPlan",
        "compress ratio and seq lengths",
        "compressor plan object",
        "models.deepseek_v4.DSV4Compressor",
        "Python loop fallback",
        ("triton",),
        "fallback",
        "sm80 compressor planning kernels",
    ),
    DSV4KernelInventoryEntry(
        "compress_forward_fallback",
        "sglang.jit_kernel.dsv4.compress_forward",
        "x[tokens, hidden], compressor weights/state",
        "compressed kv[compressed_tokens, dim]",
        "models.deepseek_v4.DSV4Compressor.forward",
        "torch pooled-compress fallback",
        ("triton", "flashinfer"),
        "fallback",
        "sm80 fused compressor forward",
    ),
    DSV4KernelInventoryEntry(
        "compress_norm_rope_store_fallback",
        "sglang.jit_kernel.dsv4.compress_norm_rope_store",
        "compressed kv, positions, cache locations",
        "compressed KV cache updated",
        "attention.deepseek_v4.store_compressed",
        "bf16 torch cache write",
        ("triton", "sgl_kernel"),
        "fallback",
        "sm80 fused compress norm+RoPE+store",
    ),
    DSV4KernelInventoryEntry(
        "topk_transform_512_fallback",
        "sglang.jit_kernel.dsv4.topk_transform_512",
        "topk indices/logits for C4 sparse attention",
        "512-wide transformed topk rows",
        "attention.deepseek_v4 metadata/index fallback",
        "deterministic torch fallback",
        ("sgl_kernel", "triton"),
        "fallback",
        "sm80 topk transform tuned for shared memory",
    ),
    DSV4KernelInventoryEntry(
        "topk_transform_512_v2_fallback",
        "sglang.jit_kernel.dsv4.topk_transform_512_v2",
        "topk indices/logits for C4 sparse attention",
        "512-wide transformed topk rows",
        "future indexer fast path",
        "deterministic torch fallback",
        ("sgl_kernel", "triton"),
        "fallback",
        "sm80 topk v2 without sm90/sm100 assumptions",
    ),
    DSV4KernelInventoryEntry(
        "plan_topk_v2_fallback",
        "sglang.jit_kernel.dsv4.plan_topk_v2",
        "topk lengths and workspace sizing",
        "topk v2 plan metadata",
        "future indexer fast path",
        "Python plan fallback",
        ("sgl_kernel", "triton"),
        "fallback",
        "sm80-safe planner",
    ),
    DSV4KernelInventoryEntry(
        "flash_mla_with_kvcache",
        "sgl_kernel.flash_mla.flash_mla_with_kvcache",
        "q, paged KV caches, topk lengths, FlashMLA metadata",
        "attention output[tokens, heads, dim]",
        "attention.deepseek_v4.DSV4AttentionBackend.forward",
        "blocked on packed cache layout; torch sparse fallback",
        ("sgl_kernel.flash_mla", "flashinfer"),
        "todo",
        "sm80-compatible FlashMLA or FlashInfer MLA backend",
    ),
    DSV4KernelInventoryEntry(
        "flash_mla_sparse_prefill",
        "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
        "q, sparse pages, compressed/SWA cache",
        "prefill attention output",
        "attention.deepseek_v4 prefill fallback",
        "torch sparse fallback",
        ("sgl_kernel.flash_mla",),
        "todo",
        "sm80 sparse prefill attention",
    ),
    DSV4KernelInventoryEntry(
        "paged_mqa_attention_fallback",
        "sglang.jit_kernel.dsv4.get_paged_mqa_logits_metadata",
        "q[tokens, heads, dim], cache[slots, dim], index rows",
        "attention output[tokens, heads, dim]",
        "attention.deepseek_v4._fallback_attention",
        "torch sparse attention fallback",
        ("deep_gemm", "triton"),
        "fallback",
        "sm80 paged-MQA logits + value reduction",
    ),
    DSV4KernelInventoryEntry(
        "linear_bf16_fp32_fallback",
        "sglang.jit_kernel.dsv4.linear_bf16_fp32",
        "x[..., k], weight[n, k]",
        "fp32/bf16 linear output",
        "models.deepseek_v4 HC/router helpers",
        "torch linear fallback",
        ("triton",),
        "fallback",
        "sm80 matmul where torch overhead is visible",
    ),
    DSV4KernelInventoryEntry(
        "quantized_linear_ref",
        "DeepGEMM / fp8-fp4 GEMM call sites in sglang-main",
        "x[..., k], quantized weight[n, k], scale",
        "linear output",
        "models.deepseek_v4.DSV4Linear and MoE experts",
        "dequant + torch linear fallback",
        ("deep_gemm", "flashinfer", "marlin", "tilelang", "triton"),
        "fallback",
        "sm80 fp8/fp4 fused dequant GEMM",
    ),
    DSV4KernelInventoryEntry(
        "wo_a_grouped_projection_fallback",
        "sm100-only fp8 wo_a grouped GEMM/einsum path",
        "o[tokens, groups, d], wo_a[groups, rank, d]",
        "o_lora[tokens, groups*rank]",
        "models.deepseek_v4.DSV4Attention",
        "dequant + torch einsum fallback",
        ("deep_gemm", "sgl_kernel"),
        "fallback",
        "sm80 grouped fp8 GEMM or Triton einsum",
    ),
    DSV4KernelInventoryEntry(
        "hc_pre_fallback/hc_post_fallback/hc_head_fallback",
        "MHC fused helper call sites in sglang-main",
        "HC residual states and learned mixing tensors",
        "mixed residual states",
        "models.deepseek_v4.DeepseekV4DecoderLayer/Model",
        "torch fallback",
        ("tilelang", "triton"),
        "fallback",
        "sm80 HC split/Sinkhorn and post kernels",
    ),
    DSV4KernelInventoryEntry(
        "hash_topk_fallback",
        "sglang.jit_kernel.dsv4.hash_topk",
        "input token ids, token-to-expert table",
        "expert indices[tokens, topk]",
        "models.deepseek_v4.DSV4MoEGate",
        "torch embedding/table fallback",
        ("triton",),
        "fallback",
        "sm80 hash topk kernel if routing overhead matters",
    ),
    DSV4KernelInventoryEntry(
        "moe_gate_fallback",
        "sglang.jit_kernel.dsv4.mask_topk_ids",
        "router scores, correction bias, topk",
        "router weights and expert indices",
        "models.deepseek_v4.DSV4MoEGate",
        "torch topk fallback",
        ("triton",),
        "fallback",
        "sm80 fused router topk/mask",
    ),
    DSV4KernelInventoryEntry(
        "mega_moe_pre_dispatch_fallback",
        "sglang.jit_kernel.dsv4.mega_moe_pre_dispatch",
        "hidden states, router weights/indices",
        "grouped expert dispatch metadata",
        "models.deepseek_v4.DSV4FusedRoutedExperts",
        "Python expert loop fallback",
        ("sgl_kernel", "flashinfer", "triton"),
        "fallback",
        "sm80 grouped MoE dispatch",
    ),
    DSV4KernelInventoryEntry(
        "silu_and_mul_clamp_fallback",
        "sglang.jit_kernel.dsv4.silu_and_mul_clamp",
        "gate/up activations",
        "SiLU(gate) * up",
        "models.deepseek_v4 MoE/shared experts",
        "torch fallback",
        ("sgl_kernel", "triton"),
        "fallback",
        "sm80 fused activation",
    ),
    DSV4KernelInventoryEntry(
        "silu_and_mul_masked_post_quant",
        "sglang.jit_kernel.dsv4.silu_and_mul_masked_post_quant",
        "expert activations and route mask",
        "quantized expert activations",
        "future MoE fast path",
        "explicit NotImplementedError",
        ("sgl_kernel", "triton"),
        "todo",
        "sm80 fused activation+post quant",
    ),
    DSV4KernelInventoryEntry(
        "silu_and_mul_contig_post_quant",
        "sglang.jit_kernel.dsv4.silu_and_mul_contig_post_quant",
        "contiguous expert activations",
        "quantized expert activations",
        "future MoE fast path",
        "explicit NotImplementedError",
        ("sgl_kernel", "triton"),
        "todo",
        "sm80 fused contiguous activation+post quant",
    ),
    DSV4KernelInventoryEntry(
        "make_name",
        "sglang.jit_kernel.dsv4.make_name",
        "kernel name fragments",
        "stable generated kernel name",
        "wrapper/debug utilities",
        "pure Python utility",
        (),
        "native",
        "none; utility export only",
    ),
)


@dataclass(frozen=True)
class DSV4KernelCapability:
    cuda_available: bool
    cuda_capability: tuple[int, int] | None
    is_sm80: bool
    sgl_kernel_available: bool
    sgl_kernel_error: str | None
    sgl_kernel_sm80_common_ops: bool
    sgl_kernel_dsv4_ops: dict[str, bool]
    flash_mla_available: bool
    flash_mla_error: str | None
    flashinfer_available: bool
    flashinfer_error: str | None
    deep_gemm_available: bool
    deep_gemm_error: str | None
    deep_gemm_usable: bool
    tilelang_available: bool
    tilelang_error: str | None
    triton_available: bool
    triton_error: str | None
    marlin_available: bool
    marlin_error: str | None


@dataclass(frozen=True)
class DSV4MoERoutePlan:
    sorted_route_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    route_count: int
    topk: int
    block_size_m: int


def _module_available(name: str) -> tuple[bool, str | None]:
    if importlib.util.find_spec(name) is None:
        return False, "module not installed"
    try:
        __import__(name)
    except Exception as exc:  # pragma: no cover - depends on optional packages.
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _cuda_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - defensive for unusual CUDA setups.
        return None


def _has_sgl_common_ops_for_sm80() -> bool:
    try:
        import sgl_kernel  # type: ignore

        package_paths = list(getattr(sgl_kernel, "__path__", []))
    except Exception:
        return False
    return any(
        os.path.exists(os.path.join(package_path, "sm80", "common_ops.abi3.so"))
        for package_path in package_paths
    )


@lru_cache(maxsize=1)
def detect_dsv4_kernel_capabilities() -> DSV4KernelCapability:
    cap = _cuda_capability()
    is_sm80 = cap == (8, 0)

    sgl_ok, sgl_err = _module_available("sgl_kernel")
    dsv4_ops = {
        name: bool(sgl_ok and hasattr(torch.ops.sgl_kernel, name))
        for name in (
            "deepseek_v4_topk_transform_512",
            "dsv4_fused_q_indexer_rope_hadamard_quant",
            "dsv4_fused_q_indexer_rope_hadamard_fp4_quant",
        )
    }

    flash_mla_ok, flash_mla_err = _module_available("sgl_kernel.flash_mla")
    flashinfer_ok, flashinfer_err = _module_available("flashinfer")
    deep_gemm_ok, deep_gemm_err = _module_available("deep_gemm")
    tilelang_ok, tilelang_err = _module_available("tilelang")
    triton_ok, triton_err = _module_available("triton")
    marlin_ok, marlin_err = _module_available("marlin")

    deep_gemm_usable = bool(deep_gemm_ok and cap is not None and cap[0] >= 9)
    return DSV4KernelCapability(
        cuda_available=torch.cuda.is_available(),
        cuda_capability=cap,
        is_sm80=is_sm80,
        sgl_kernel_available=sgl_ok,
        sgl_kernel_error=sgl_err,
        sgl_kernel_sm80_common_ops=_has_sgl_common_ops_for_sm80(),
        sgl_kernel_dsv4_ops=dsv4_ops,
        flash_mla_available=flash_mla_ok,
        flash_mla_error=flash_mla_err,
        flashinfer_available=flashinfer_ok,
        flashinfer_error=flashinfer_err,
        deep_gemm_available=deep_gemm_ok,
        deep_gemm_error=deep_gemm_err,
        deep_gemm_usable=deep_gemm_usable,
        tilelang_available=tilelang_ok,
        tilelang_error=tilelang_err,
        triton_available=triton_ok,
        triton_error=triton_err,
        marlin_available=marlin_ok,
        marlin_error=marlin_err,
    )


def dsv4_kernel_inventory_by_wrapper() -> dict[str, DSV4KernelInventoryEntry]:
    return {entry.wrapper: entry for entry in DSV4_KERNEL_INVENTORY}


def unsupported_kernel(name: str, detail: str) -> None:
    cap = detect_dsv4_kernel_capabilities()
    sm = "no CUDA" if cap.cuda_capability is None else f"sm{cap.cuda_capability[0]}{cap.cuda_capability[1]}"
    raise NotImplementedError(f"{name} is not available for {sm}: {detail}")


def dsv4_env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def dsv4_sm80_triton_enabled(toggle: str) -> bool:
    if not dsv4_env_flag(toggle):
        return False
    cap = detect_dsv4_kernel_capabilities()
    return bool(cap.is_sm80 and cap.triton_available)


def _triton_dsv4_ops():
    from minisgl.kernel.triton import deepseek_v4 as triton_dsv4

    return triton_dsv4


def fp8_dtype() -> torch.dtype:
    return getattr(torch, "float8_e4m3fn", torch.uint8)


def e8m0_dtype() -> torch.dtype:
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def scale_dim(size: int, block_size: int = 128) -> int:
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


def dequant_fp8_weight(
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


def dequant_fp4_weight(
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


def quantize_fp8_activation_ref(x: torch.Tensor, *, block_size: int = 128) -> torch.Tensor:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or x.numel() == 0 or x.shape[-1] % block_size != 0:
        return x
    dtype = x.dtype
    flat = x.contiguous().view(-1, x.shape[-1]).float()
    groups = flat.view(flat.shape[0], flat.shape[1] // block_size, block_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / 448.0
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    y = (groups / scale).clamp(-448.0, 448.0).to(fp8).float() * scale
    return y.reshape_as(flat).reshape_as(x).to(dtype)


def quantized_linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    weight_kind: WeightKind,
) -> torch.Tensor:
    if weight_kind == "fp4":
        x = quantize_fp8_activation_ref(x)
        if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_FP4_GEMM"):
            y = _triton_dsv4_ops().quantized_linear_fp4(x, weight, scale)
            if y is not None:
                return y
        w = dequant_fp4_weight(weight, scale, out_dtype=x.dtype)
    elif weight_kind == "fp8":
        x = quantize_fp8_activation_ref(x)
        if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_FP8_GEMM"):
            y = _triton_dsv4_ops().quantized_linear_fp8(x, weight, scale)
            if y is not None:
                return y
        w = dequant_fp8_weight(weight, scale, out_dtype=x.dtype)
    else:
        w = weight.to(x.dtype)
    return F.linear(x, w)


def linear_bf16_fp32_fallback(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x.float(), weight.float())


def _compress_forward_vectorized(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    ratio: int,
    head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    wkv_gate,
    norm,
) -> torch.Tensor | None:
    usable_tokens = x.shape[0] // ratio * ratio
    if usable_tokens == 0:
        return x.new_empty((0, head_dim))
    projected = wkv_gate.forward(x[:usable_tokens]).float()
    kv, score = projected.chunk(2, dim=-1)
    slot = (positions[:usable_tokens].long() % ratio).to(torch.long)
    score = score + ape[slot].float()
    kv = kv.view(-1, ratio, kv.shape[-1])
    score = score.view(-1, ratio, score.shape[-1])
    if overlap:
        if kv.shape[-1] != 2 * head_dim or score.shape[-1] != 2 * head_dim:
            return None
        kv = torch.cat([kv[..., :head_dim], kv[..., head_dim:]], dim=1)
        score = torch.cat([score[..., :head_dim], score[..., head_dim:]], dim=1)
    else:
        if kv.shape[-1] != head_dim or score.shape[-1] != head_dim:
            return None
    pooled = (kv * score.softmax(dim=1)).sum(dim=1)
    return norm.forward(pooled.to(x.dtype))


def apply_rotary_tail(
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
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_ROPE"):
        try:
            if _triton_dsv4_ops().apply_rotary_tail(
                x,
                positions,
                rotary_dim=rotary_dim,
                base=base,
                inverse=inverse,
                original_seq_len=original_seq_len,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            ):
                return x
        except Exception:
            pass

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


def q_norm_rope_fallback(
    q: torch.Tensor,
    positions: torch.Tensor,
    *,
    rms_norm_eps: float,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_Q_NORM_ROPE"):
        try:
            if _triton_dsv4_ops().q_norm_rope(
                q,
                positions,
                rms_norm_eps=rms_norm_eps,
                rotary_dim=rotary_dim,
                base=base,
                original_seq_len=original_seq_len,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            ):
                return q
        except Exception:
            pass
    q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + rms_norm_eps).to(q.dtype)
    return apply_rotary_tail(
        q,
        positions,
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )


def norm_rope_inplace_fallback(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    weight: torch.Tensor,
    eps: float,
    rotary_dim: int,
    base: float,
) -> torch.Tensor:
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    x.copy_((y * weight.float()).to(dtype))
    return apply_rotary_tail(x, positions, rotary_dim=rotary_dim, base=base)


def k_norm_rope_cache_fallback(
    kv: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    return apply_rotary_tail(
        kv,
        positions,
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )


def make_name(*parts: object) -> str:
    return "_".join(str(part) for part in parts if str(part))


def compressor_plan_fallback(
    compress_ratio: int,
    seq_lens: torch.Tensor,
    *,
    is_prefill: bool,
) -> dict[str, torch.Tensor | int | bool]:
    return {
        "compress_ratio": int(compress_ratio),
        "seq_lens": seq_lens.to(torch.int32),
        "is_prefill": bool(is_prefill),
    }


def triton_create_paged_compress_data(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "triton_create_paged_compress_data",
        "mini-sglang builds equivalent paged compression metadata in DSV4AttentionBackend._build_metadata",
    )


def compress_forward_fallback(
    x: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    ratio: int,
    head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    wkv_gate,
    norm,
) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_empty((0, head_dim))
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_COMPRESS"):
        if positions is None:
            fast_positions = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
        else:
            fast_positions = positions.to(device=x.device, dtype=torch.long)
        fast = _compress_forward_vectorized(
            x,
            fast_positions,
            ratio=ratio,
            head_dim=head_dim,
            overlap=overlap,
            ape=ape,
            wkv_gate=wkv_gate,
            norm=norm,
        )
        if fast is not None:
            return fast
    projected = wkv_gate.forward(x).float()
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
        local_score = score[start:end] + ape[slot].float()
        local_kv = kv[start:end]
        if overlap:
            left = local_kv[:, :head_dim]
            right = local_kv[:, head_dim:]
            local_score = torch.cat(
                [local_score[:, :head_dim], local_score[:, head_dim:]],
                dim=0,
            )
            local_kv = torch.cat([left, right], dim=0)
        pooled = (local_kv * local_score.softmax(dim=0)).sum(dim=0, keepdim=True)
        rows.append(norm.forward(pooled.to(x.dtype)))
        start = end
    if not rows:
        return x.new_empty((0, head_dim))
    return torch.cat(rows, dim=0)


def flash_mla_with_kvcache(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "flash_mla_with_kvcache",
        "FlashMLA is present only as an optional backend; mini-sglang currently uses bf16 paged-MQA fallback on sm80",
    )


def flash_mla_sparse_prefill(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "flash_mla_sparse_prefill",
        "sparse prefill FlashMLA requires packed cache layouts that are not implemented for sm80 yet",
    )


def get_paged_mqa_logits_metadata_fallback(context_indices: list[torch.Tensor]) -> list[torch.Tensor]:
    return context_indices


def hc_split_sinkhorn_ref(
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


def hc_pre_fallback(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x.shape
    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = linear_bf16_fp32_fallback(flat, fn.float()) * rsqrt
    pre, post, comb = hc_split_sinkhorn_ref(mixes, scale, base, hc_mult, sinkhorn_iters, eps)
    y = torch.sum(pre.to(x.dtype).unsqueeze(-1) * x.view(shape), dim=1)
    return y, post.to(x.dtype), comb.to(x.dtype)


def hc_post_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return post.unsqueeze(-1) * x.unsqueeze(1) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(2), dim=1
    )


def hc_head_fallback(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    eps: float,
    norm_eps: float,
) -> torch.Tensor:
    shape = x.shape
    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = linear_bf16_fp32_fallback(flat, fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + eps
    return torch.sum(pre.to(x.dtype).unsqueeze(-1) * x.view(shape), dim=1)


def sequence_mqa_attention_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    spans: list[tuple[int, int]],
    *,
    window_size: int,
    softmax_scale: float,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q)
    sink = attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)
    for start, end in spans:
        q_seq = q[start:end].float()
        kv_seq = kv[start:end].float()
        seq_len = end - start
        for local_idx in range(seq_len):
            ctx_start = max(0, local_idx - window_size + 1) if window_size else 0
            candidates = kv_seq[ctx_start : local_idx + 1]
            scores = torch.einsum("hd,td->ht", q_seq[local_idx], candidates)
            scores = scores * softmax_scale
            max_score = torch.maximum(scores.max(dim=-1).values, sink)
            exp_scores = torch.exp(scores - max_score[:, None])
            denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
            attn = exp_scores / denom[:, None]
            out[start + local_idx] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
    return out


def paged_mqa_attention_fallback(
    q: torch.Tensor,
    cache: torch.Tensor,
    context_indices: list[torch.Tensor],
    *,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"DSV4 fallback expects q shape [tokens, heads, dim], got {q.shape}")
    out = torch.empty_like(q)
    sink = (
        attn_sink[: q.shape[1]].to(device=q.device, dtype=torch.float32)
        if attn_sink is not None
        else None
    )
    for row, indices in enumerate(context_indices):
        if indices.numel() == 0:
            out[row].zero_()
            continue
        candidates = cache[indices.to(torch.long)].float()
        scores = torch.einsum("hd,td->ht", q[row].float(), candidates) * softmax_scale
        if sink is None:
            attn = torch.softmax(scores, dim=-1)
        else:
            max_score = torch.maximum(scores.max(dim=-1).values, sink)
            exp_scores = torch.exp(scores - max_score[:, None])
            denom = exp_scores.sum(dim=-1) + torch.exp(sink - max_score)
            attn = exp_scores / denom[:, None]
        out[row] = torch.einsum("ht,td->hd", attn, candidates).to(q.dtype)
    return out


def wo_a_grouped_projection_fallback(
    o: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    *,
    num_local_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    d_per_group = o.shape[-1]
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_WO_A_BF16"):
        try:
            y = _triton_dsv4_ops().wo_a_grouped_projection_fp8(
                o,
                weight,
                scale,
                num_local_groups=num_local_groups,
                o_lora_rank=o_lora_rank,
            )
            if y is not None:
                return y
        except Exception:
            pass
    wo_a = dequant_fp8_weight(weight, scale, out_dtype=o.dtype)
    wo_a = wo_a.view(num_local_groups, o_lora_rank, d_per_group)
    return torch.einsum("tgd,grd->tgr", o, wo_a).reshape(o.shape[0], -1)


def hash_topk_fallback(hash_topk, input_ids: torch.Tensor) -> torch.Tensor:
    return hash_topk.forward(input_ids.view(-1)).long()


def build_moe_route_plan(
    indices: torch.Tensor,
    *,
    num_experts: int,
    block_size_m: int = 16,
) -> DSV4MoERoutePlan:
    if indices.ndim != 2:
        raise ValueError(
            f"DSV4 MoE route plan expects indices shape [tokens, topk], got {indices.shape}"
        )
    if num_experts <= 0:
        raise ValueError(f"DSV4 MoE route plan requires num_experts > 0, got {num_experts}")
    if block_size_m <= 0:
        raise ValueError(f"DSV4 MoE route plan requires block_size_m > 0, got {block_size_m}")

    route_count = indices.numel()
    topk = indices.shape[1]
    device = indices.device
    flat_indices = indices.reshape(-1).to(torch.long)
    valid = (flat_indices >= 0) & (flat_indices < num_experts)
    valid_route_ids = torch.arange(route_count, device=device, dtype=torch.long)[valid]
    valid_expert_ids = flat_indices[valid]

    if valid_route_ids.numel() == 0:
        empty_ids = torch.empty((0,), dtype=torch.int32, device=device)
        return DSV4MoERoutePlan(
            sorted_route_ids=empty_ids,
            expert_ids=empty_ids,
            num_tokens_post_padded=torch.zeros((1,), dtype=torch.int32, device=device),
            route_count=route_count,
            topk=topk,
            block_size_m=block_size_m,
        )

    sort_key = valid_expert_ids * max(route_count, 1) + valid_route_ids
    order = torch.argsort(sort_key)
    sorted_valid_routes = valid_route_ids[order]
    sorted_valid_experts = valid_expert_ids[order]

    counts = torch.bincount(valid_expert_ids, minlength=num_experts).to(torch.long)
    padded_counts = ((counts + block_size_m - 1) // block_size_m) * block_size_m
    total_padded = int(padded_counts.sum().item())
    sorted_route_ids = torch.full(
        (total_padded,),
        route_count,
        dtype=torch.int32,
        device=device,
    )

    counts_before = counts.cumsum(0) - counts
    padded_offsets = padded_counts.cumsum(0) - padded_counts
    compact_positions = torch.arange(
        sorted_valid_routes.numel(),
        device=device,
        dtype=torch.long,
    )
    local_ranks = compact_positions - counts_before[sorted_valid_experts]
    padded_positions = padded_offsets[sorted_valid_experts] + local_ranks
    sorted_route_ids[padded_positions] = sorted_valid_routes.to(torch.int32)

    blocks_per_expert = (padded_counts // block_size_m).to(torch.long)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=device),
        blocks_per_expert,
    )
    return DSV4MoERoutePlan(
        sorted_route_ids=sorted_route_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor([total_padded], dtype=torch.int32, device=device),
        route_count=route_count,
        topk=topk,
        block_size_m=block_size_m,
    )


def moe_gate_fallback(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_ids: torch.Tensor | None,
    topk: int,
    scoring_func: str,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
    hash_topk=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(hidden_states.float(), weight.float())
    if scoring_func == "softmax":
        original_scores = scores.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        original_scores = scores.sigmoid()
    else:
        original_scores = F.softplus(scores).sqrt()

    if hash_topk is not None:
        if input_ids is None:
            raise ValueError("DeepSeek V4 hash routing requires input_ids")
        indices = hash_topk_fallback(hash_topk, input_ids)
    else:
        scores_for_topk = original_scores
        if correction_bias is not None:
            scores_for_topk = scores_for_topk + correction_bias.float()
        indices = scores_for_topk.topk(topk, dim=-1).indices

    weights = original_scores.gather(1, indices)
    if scoring_func != "softmax":
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    return weights * routed_scaling_factor, indices


def silu_and_mul_clamp_fallback(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_SWIGLU"):
        try:
            out = _triton_dsv4_ops().silu_and_mul_clamp(
                gate,
                up,
                swiglu_limit=swiglu_limit,
                weights=weights,
            )
            if out is not None:
                return out
        except Exception:
            pass
    gate_f = gate.float()
    up_f = up.float()
    if swiglu_limit > 0:
        up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
        gate_f = torch.clamp(gate_f, max=swiglu_limit)
    out = F.silu(gate_f) * up_f
    if weights is not None:
        out = out * weights
    return out


def mega_moe_pre_dispatch_fallback(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_experts: int | None = None,
    block_size_m: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | DSV4MoERoutePlan:
    if num_experts is not None:
        return build_moe_route_plan(
            indices,
            num_experts=num_experts,
            block_size_m=block_size_m,
        )
    return hidden_states, weights, indices


def moe_route_dispatch_bf16_grouped(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
) -> torch.Tensor | None:
    if not dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_MOE_ROUTE"):
        return None
    if hidden_states.dtype is not torch.bfloat16 or not hidden_states.is_cuda:
        return None
    if hidden_states.ndim != 2 or weights.shape != indices.shape or indices.ndim != 2:
        return None
    if w13_weight.ndim != 4 or w13_weight.shape[1] != 2 or w2_weight.ndim != 3:
        return None
    if w13_weight.shape[0] != w2_weight.shape[0]:
        return None
    if hidden_states.shape[1] != w13_weight.shape[-1] * 2:
        return None

    try:
        plan = build_moe_route_plan(
            indices,
            num_experts=w13_weight.shape[0],
            block_size_m=16,
        )
        return _triton_dsv4_ops().grouped_fp4_moe(
            hidden_states.contiguous(),
            weights.to(device=hidden_states.device, dtype=torch.float32).contiguous(),
            w13_weight.contiguous(),
            w13_scale,
            w2_weight.contiguous(),
            w2_scale,
            plan.sorted_route_ids,
            plan.expert_ids,
            plan.num_tokens_post_padded,
            route_count=plan.route_count,
            topk=plan.topk,
            block_size_m=plan.block_size_m,
            swiglu_limit=swiglu_limit,
        )
    except Exception:
        return None


def topk_transform_512_fallback(indices: torch.Tensor, *, width: int = 512) -> torch.Tensor:
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_TOPK"):
        try:
            out = _triton_dsv4_ops().topk_transform_512(indices, width=width)
            if out is not None:
                return out
        except Exception:
            pass
    if indices.shape[-1] == width:
        return indices
    out = torch.full(
        (*indices.shape[:-1], width),
        -1,
        dtype=indices.dtype,
        device=indices.device,
    )
    n = min(indices.shape[-1], width)
    out[..., :n] = indices[..., :n]
    return out


def topk_transform_512_v2_fallback(indices: torch.Tensor, *, width: int = 512) -> torch.Tensor:
    return topk_transform_512_fallback(indices, width=width)


def plan_topk_v2_fallback(lengths: torch.Tensor, *, width: int = 512) -> dict[str, torch.Tensor | int]:
    return {"lengths": lengths.to(torch.int32), "width": width}


def store_swa_fallback(kvcache, layer_id: int, kv: torch.Tensor, out_loc: torch.Tensor) -> None:
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_STORE_CACHE"):
        try:
            if _triton_dsv4_ops().store_cache(kvcache.swa_cache(layer_id), kv, out_loc):
                return
        except Exception:
            pass
    kvcache.store_swa(layer_id, kv, out_loc)


def store_compressed_fallback(
    kvcache,
    layer_id: int,
    kv: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_COMPRESS_STORE"):
        try:
            if _triton_dsv4_ops().store_cache(kvcache.component_cache(layer_id), kv, loc):
                return
        except Exception:
            pass
    kvcache.store_compressed(layer_id, kv, loc)


def store_indexer_fallback(kvcache, layer_id: int, kv: torch.Tensor, loc: torch.Tensor) -> None:
    if dsv4_sm80_triton_enabled("MINISGL_DSV4_SM80_COMPRESS_STORE"):
        try:
            if _triton_dsv4_ops().store_cache(kvcache.indexer_cache(layer_id), kv, loc):
                return
        except Exception:
            pass
    kvcache.store_indexer(layer_id, kv, loc)


def compress_norm_rope_store_fallback(
    kvcache,
    layer_id: int,
    kv: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    store_compressed_fallback(kvcache, layer_id, kv, loc)


def fused_q_indexer_rope_first_quant(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "fused_q_indexer_rope_first_quant",
        "sm80 indexer first-quant kernel has not been ported; use bf16 indexer fallback",
    )


def fused_q_indexer_rope_hadamard_quant(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "fused_q_indexer_rope_hadamard_quant",
        "sgl_kernel DeepSeek V4 fp8 indexer op is missing for sm80",
    )


def fused_q_indexer_rope_hadamard_fp4_quant(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "fused_q_indexer_rope_hadamard_fp4_quant",
        "the upstream fp4 hadamard quant path is sm100-oriented and must be rewritten for sm80",
    )


def silu_and_mul_masked_post_quant(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "silu_and_mul_masked_post_quant",
        "post-quant MoE activation fusion is not implemented in the sm80 wrapper",
    )


def silu_and_mul_contig_post_quant(*args, **kwargs):
    del args, kwargs
    unsupported_kernel(
        "silu_and_mul_contig_post_quant",
        "contiguous post-quant MoE activation fusion is not implemented in the sm80 wrapper",
    )


__all__ = [
    "DSV4_KERNEL_INVENTORY",
    "DSV4KernelCapability",
    "DSV4KernelInventoryEntry",
    "DSV4KernelMode",
    "DSV4MoERoutePlan",
    "apply_rotary_tail",
    "build_moe_route_plan",
    "compress_norm_rope_store_fallback",
    "compress_forward_fallback",
    "compressor_plan_fallback",
    "flash_mla_sparse_prefill",
    "flash_mla_with_kvcache",
    "get_paged_mqa_logits_metadata_fallback",
    "make_name",
    "triton_create_paged_compress_data",
    "dequant_fp4_weight",
    "dequant_fp8_weight",
    "detect_dsv4_kernel_capabilities",
    "dsv4_env_flag",
    "dsv4_kernel_inventory_by_wrapper",
    "dsv4_sm80_triton_enabled",
    "e8m0_dtype",
    "fp8_dtype",
    "fused_q_indexer_rope_first_quant",
    "fused_q_indexer_rope_hadamard_fp4_quant",
    "fused_q_indexer_rope_hadamard_quant",
    "hash_topk_fallback",
    "hc_head_fallback",
    "hc_post_fallback",
    "hc_pre_fallback",
    "hc_split_sinkhorn_ref",
    "k_norm_rope_cache_fallback",
    "linear_bf16_fp32_fallback",
    "mega_moe_pre_dispatch_fallback",
    "moe_gate_fallback",
    "moe_route_dispatch_bf16_grouped",
    "norm_rope_inplace_fallback",
    "paged_mqa_attention_fallback",
    "plan_topk_v2_fallback",
    "q_norm_rope_fallback",
    "quantize_fp8_activation_ref",
    "quantized_linear_ref",
    "scale_dim",
    "sequence_mqa_attention_fallback",
    "silu_and_mul_clamp_fallback",
    "silu_and_mul_contig_post_quant",
    "silu_and_mul_masked_post_quant",
    "store_compressed_fallback",
    "store_indexer_fallback",
    "store_swa_fallback",
    "topk_transform_512_fallback",
    "topk_transform_512_v2_fallback",
    "unsupported_kernel",
    "wo_a_grouped_projection_fallback",
]

"""DeepSeek V4 Ampere release-kernel wrappers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from minisgl.kernel.utils import load_jit
from minisgl.utils import div_ceil, is_dsv4_ampere_capability

WeightKind = Literal["bf16", "fp8", "fp4"]


DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16 = "marlin_wna16"
DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT = 512
DSV4_MARLIN_WNA16_RELEASE_ERROR = (
    "Marlin WNA16 release preset has released raw routed expert weights; "
    "the required prepacked Marlin cache is unavailable in this Engine."
)
DSV4_INDEXER_CAPTURE_WIDTH_MODES = ("current", "table_width", "seq_len_aligned")


@dataclass(frozen=True)
class DSV4PagedMQAMetadata:
    indptr: torch.Tensor
    indices: torch.Tensor
    lengths: torch.Tensor
    max_length: int

    @property
    def row_count(self) -> int:
        return int(self.lengths.numel())


@dataclass(frozen=True)
class DSV4TwoSourceAttentionMetadata:
    compressed_indices: torch.Tensor
    compressed_lengths: torch.Tensor
    swa_indices: torch.Tensor
    swa_lengths: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.swa_lengths.numel())


@dataclass(frozen=True)
class DSV4TopKTransformOutput:
    raw_indices: torch.Tensor
    page_indices: torch.Tensor
    full_indices: torch.Tensor
    backend: str
    topk_lens: torch.Tensor | None = None


@dataclass(frozen=True)
class DSV4IndexerSelectOutput:
    logits: torch.Tensor
    topk: DSV4TopKTransformOutput
    backend: str


@dataclass(frozen=True)
class DSV4IndexerFP8Query:
    q_values: torch.Tensor
    weights: torch.Tensor


@dataclass(frozen=True)
class DSV4KernelCapability:
    cuda_available: bool
    cuda_capability: tuple[int, int] | None
    is_ampere: bool
    triton_available: bool
    triton_error: str | None


@dataclass(frozen=True)
class DSV4MoERoutePlan:
    sorted_route_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    route_count: int
    topk: int
    block_size_m: int


@dataclass(frozen=True)
class DSV4MoEExecutionPlan:
    route_plan: DSV4MoERoutePlan
    route_weights: torch.Tensor
    tokens: int
    hidden: int
    num_experts: int
    reduce_once: bool
    final_reduce_label: str


def _module_available(name: str) -> tuple[bool, str | None]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:  # pragma: no cover - optional packages can fail in __init__.
        return False, f"{type(exc).__name__}: {exc}"
    if spec is None:
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


@lru_cache(maxsize=1)
def detect_dsv4_kernel_capabilities() -> DSV4KernelCapability:
    cap = _cuda_capability()
    is_ampere = is_dsv4_ampere_capability(cap)

    triton_ok, triton_err = _module_available("triton")
    return DSV4KernelCapability(
        cuda_available=torch.cuda.is_available(),
        cuda_capability=cap,
        is_ampere=is_ampere,
        triton_available=triton_ok,
        triton_error=triton_err,
    )


def dsv4_triton_available() -> bool:
    cap = detect_dsv4_kernel_capabilities()
    return bool(cap.is_ampere and cap.triton_available)


@lru_cache(maxsize=1)
def _local_dsv4_c4_indexer_rmsnorm_module():
    return load_jit(
        "dsv4_c4_indexer_rmsnorm_bf16",
        cuda_files=["dsv4_c4_indexer_rmsnorm_bf16.cu"],
        cuda_wrappers=[
            (
                "rmsnorm_bf16",
                "DSV4C4IndexerRMSNormBF16Kernel::run",
            ),
        ],
        extra_cuda_cflags=[
            "-gencode=arch=compute_80,code=sm_80",
        ],
    )


def _c4_indexer_rmsnorm_bf16_native(
    kv: torch.Tensor,
    norm_weight: torch.Tensor,
    loc: torch.Tensor,
    *,
    rms_norm_eps: float,
) -> bool:
    if (
        not detect_dsv4_kernel_capabilities().is_ampere
        or kv.ndim != 2
        or kv.shape[-1] != 128
        or kv.dtype is not torch.bfloat16
        or norm_weight.shape != (128,)
        or norm_weight.dtype is not torch.float32
        or loc.shape != (kv.shape[0],)
        or loc.dtype is not torch.int64
        or not kv.is_cuda
        or not norm_weight.is_cuda
        or not loc.is_cuda
        or not kv.is_contiguous()
        or not norm_weight.is_contiguous()
        or not loc.is_contiguous()
    ):
        return False
    _local_dsv4_c4_indexer_rmsnorm_module().rmsnorm_bf16(
        kv,
        norm_weight,
        loc,
        float(rms_norm_eps),
    )
    return True


def warmup_indexer_fp8_backend(
    device: torch.device,
    *,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    page_size: int,
) -> None:
    if not dsv4_triton_available():
        return
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        return
    warmup = getattr(_triton_dsv4_ops(), "warmup_indexer_fp8_lut", None)
    if callable(warmup):
        warmup(cuda_device)
    # Compile and launch the exact three-stage C4 publication cluster before
    # any CUDA graph capture. All tensors are disposable warmup fixtures, and
    # the one valid row matches the production RoPE/page specialization.
    kv = torch.zeros((1, 128), dtype=torch.bfloat16, device=cuda_device)
    positions = torch.zeros((1,), dtype=torch.int64, device=cuda_device)
    loc = torch.zeros((1,), dtype=torch.int64, device=cuda_device)
    norm_weight = torch.ones((128,), dtype=torch.float32, device=cuda_device)
    packed_cache = torch.zeros(
        (1, int(page_size) * (128 + 4)),
        dtype=torch.uint8,
        device=cuda_device,
    )
    if not _c4_indexer_rmsnorm_bf16_native(
        kv,
        norm_weight,
        loc,
        rms_norm_eps=1e-6,
    ):
        raise RuntimeError("Failed to warm the native C4 indexer RMSNorm stage.")
    if not _triton_dsv4_ops().indexer_rotary_tail_valid(
        kv,
        positions,
        loc,
        rotary_dim=64,
        base=float(base),
        original_seq_len=int(original_seq_len),
        factor=float(factor),
        beta_fast=int(beta_fast),
        beta_slow=int(beta_slow),
    ):
        raise RuntimeError("Failed to warm the C4 indexer RoPE stage.")
    if not _triton_dsv4_ops().indexer_hadamard_fp8_paged_store(
        kv,
        loc,
        packed_cache,
        page_size=int(page_size),
    ):
        raise RuntimeError("Failed to warm the C4 indexer Hadamard/QAT/store stage.")


def moe_route_dispatch_bf16_marlin_wna16_prepacked(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    cache,
    *,
    swiglu_limit: float = 0.0,
    moe_plan: DSV4MoEExecutionPlan | None = None,
) -> torch.Tensor:
    return _run_moe_bf16_marlin_wna16_prepacked(
        hidden_states,
        weights,
        indices,
        cache,
        swiglu_limit=swiglu_limit,
        moe_plan=moe_plan,
    )


def _run_moe_bf16_marlin_wna16_prepacked(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    cache,
    *,
    swiglu_limit: float,
    moe_plan: DSV4MoEExecutionPlan | None,
) -> torch.Tensor:
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Marlin WNA16 expects fp16/bf16 hidden states, got {hidden_states.dtype}")
    if not hidden_states.is_cuda:
        raise ValueError("Marlin WNA16 requires CUDA hidden states")
    if hidden_states.ndim != 2 or weights.shape != indices.shape or indices.ndim != 2:
        raise ValueError(
            "Marlin WNA16 expects hidden [tokens, hidden] and matching weights/indices [tokens, topk]"
        )
    if cache is None:
        raise RuntimeError("Marlin WNA16 prepacked dispatch requires a prepared weight cache.")
    if not all(hasattr(cache, name) for name in ("w13", "w2", "w13_scale", "w2_scale")):
        raise RuntimeError("Marlin WNA16 prepacked dispatch received an invalid cache object.")

    from minisgl.kernel import marlin_wna16

    experts = cache.w13.shape[0]
    if moe_plan is None:
        block_size_m = marlin_wna16.choose_block_size(
            tokens=hidden_states.shape[0],
            topk=indices.shape[1],
            experts=experts,
            input_dtype=None,
        )
        route_plan = build_moe_route_plan(
            indices,
            num_experts=experts,
            block_size_m=block_size_m,
        )
        topk_weights = weights
    else:
        if (
            moe_plan.tokens != hidden_states.shape[0]
            or moe_plan.hidden != hidden_states.shape[1]
            or moe_plan.num_experts != experts
            or moe_plan.route_plan.route_count != indices.numel()
            or moe_plan.route_plan.topk != indices.shape[1]
        ):
            raise ValueError("Marlin WNA16 received an incompatible DSV4 MoE execution plan")
        route_plan = moe_plan.route_plan
        topk_weights = moe_plan.route_weights.view(hidden_states.shape[0], indices.shape[1])
    output = marlin_wna16.run_moe(
        hidden_states,
        topk_weights,
        cache,
        sorted_token_ids=route_plan.sorted_route_ids,
        expert_ids=route_plan.expert_ids,
        num_tokens_post_padded=route_plan.num_tokens_post_padded,
        block_size_m=route_plan.block_size_m,
        swiglu_limit=swiglu_limit,
    )
    return output


def dsv4_cuda_available() -> bool:
    cap = detect_dsv4_kernel_capabilities()
    return bool(cap.is_ampere and cap.cuda_available)


def linear_bf16_fp32_upstream_enabled() -> bool:
    return dsv4_cuda_available()


def _triton_dsv4_ops():
    from minisgl.kernel.triton import deepseek_v4 as triton_dsv4

    return triton_dsv4


def fp8_dtype() -> torch.dtype:
    return getattr(torch, "float8_e4m3fn", torch.uint8)


def e8m0_dtype() -> torch.dtype:
    return getattr(torch, "float8_e8m0fnu", torch.uint8)


def scale_dim(size: int, block_size: int = 128) -> int:
    return div_ceil(size, block_size)


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


def quantize_fp8_activation(
    x: torch.Tensor,
    *,
    block_size: int = 128,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the release FP8 activation quantizer without backend fallback."""
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 FP8 activation quantization requires the Ampere Triton backend.")
    y = _triton_dsv4_ops().fp8_activation_quantize(
        x,
        block_size=block_size,
        out=out,
    )
    if y is None:
        raise RuntimeError("DSV4 FP8 activation quantization rejected the release tensor contract.")
    return y


def indexer_q_rope_fp8(
    q: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    rotary_dim: int,
    base: float,
    softmax_scale: float,
    head_scale: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> DSV4IndexerFP8Query:
    """Prepare the release FP8 indexer query using the qualified Triton kernels."""
    if q.ndim != 3:
        raise ValueError(f"DSV4 FP8 indexer q expects [tokens, heads, dim], got {q.shape}")
    if weights.shape[:2] != q.shape[:2]:
        raise ValueError(
            "DSV4 FP8 indexer weights must match q [tokens, heads], "
            f"got weights={tuple(weights.shape)} q={tuple(q.shape)}"
        )
    q_work = q.contiguous()
    rotary_tail(
        q_work,
        positions,
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 FP8 indexer query preparation requires the Ampere Triton backend.")
    result = _triton_dsv4_ops().indexer_fp8_quantize_fold(
        q_work,
        weights,
        softmax_scale=float(softmax_scale),
        head_scale=float(head_scale),
    )
    if result is None:
        raise RuntimeError(
            "DSV4 FP8 indexer query preparation rejected the release tensor contract."
        )
    q_values, weights_out = result
    return DSV4IndexerFP8Query(
        q_values=q_values.contiguous(),
        weights=weights_out.contiguous(),
    )


def _linear_bf16_fp32_upstream(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out_shape = tuple(x.shape[:-1]) + (weight.shape[0],)
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    weight_t = weight.contiguous().t()
    return torch.mm(x_2d, weight_t, out_dtype=torch.float32).reshape(out_shape)


def linear_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Run the release BF16-input, FP32-output projection."""
    if not (
        linear_bf16_fp32_upstream_enabled()
        and x.is_cuda
        and weight.is_cuda
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and x.shape[-1] == weight.shape[-1]
    ):
        raise RuntimeError(
            "DSV4 BF16/FP32 projection received tensors outside its release CUDA ABI."
        )
    return _linear_bf16_fp32_upstream(x, weight)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Run the release RMSNorm kernel without a Torch fallback."""
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 RMSNorm requires the Ampere Triton backend.")
    y = _triton_dsv4_ops().rms_norm_bf16(x, weight, eps=eps)
    if y is None:
        raise RuntimeError("DSV4 RMSNorm rejected the release tensor contract.")
    return y


def rotary_tail(
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
    """Run the release in-place RoPE kernel without changing backends."""
    if rotary_dim <= 0:
        return x
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 RoPE requires the Ampere Triton backend.")
    supported = _triton_dsv4_ops().apply_rotary_tail(
        x,
        positions,
        rotary_dim=rotary_dim,
        base=base,
        inverse=inverse,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    if not supported:
        raise RuntimeError("DSV4 RoPE rejected the release tensor contract.")
    return x


def q_kv_norm_rope_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    *,
    norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cache: torch.Tensor,
    out_loc: torch.Tensor,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    publish_swa_qat: bool = False,
) -> None:
    """Publish release Q/KV state with one qualified fused kernel."""
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 fused Q/KV publication requires the Ampere Triton backend.")
    supported = _triton_dsv4_ops().q_kv_norm_rope_cache_bf16(
        q,
        kv,
        positions,
        norm_weight,
        cache,
        out_loc,
        rms_norm_eps=float(rms_norm_eps),
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        publish_swa_qat=publish_swa_qat,
    )
    if not supported:
        raise RuntimeError("DSV4 fused Q/KV publication rejected the release tensor contract.")


def c4_online_pool_and_update(
    projected: torch.Tensor,
    sequence_state: torch.Tensor,
    checkpoint: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    table_indices: torch.Tensor,
    component_page_table: torch.Tensor,
    *,
    page_size: int,
) -> torch.Tensor:
    """Run the fixed-row online C4 producer qualified for the Mini Ampere ABI."""

    if not dsv4_triton_available():
        raise RuntimeError("DSV4 online C4 compression requires the Triton backend")
    output = _triton_dsv4_ops().c4_online_pool_and_update(
        projected,
        sequence_state,
        checkpoint,
        ape,
        positions,
        table_indices,
        component_page_table,
        page_size=int(page_size),
    )
    if output is None:
        raise RuntimeError("DSV4 online C4 Ampere bridge rejected the projection/state ABI")
    return output


def c128_online_pool_and_update(
    projected: torch.Tensor,
    state: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    table_indices: torch.Tensor,
) -> torch.Tensor:
    """Run the fixed-row C128 producer on request/sequence-owned FP32 carry."""

    if not dsv4_triton_available():
        raise RuntimeError("DSV4 online C128 compression requires the Triton backend")
    output = _triton_dsv4_ops().c128_online_pool_and_update(
        projected,
        state,
        ape,
        positions,
        table_indices,
    )
    if output is None:
        raise RuntimeError("DSV4 online C128 Ampere bridge rejected the projection/state ABI")
    return output


@lru_cache(maxsize=1)
def _local_dsv4_sparse_attention_module():
    return load_jit(
        "dsv4_sparse_attention_two_source_bf16",
        cuda_files=["dsv4_sparse_attention_two_source_bf16.cu"],
        cuda_wrappers=[
            (
                "sparse_attention_with_compressed",
                "DSV4SparseAttentionTwoSourceBF16Kernel<true>::run",
            ),
            (
                "sparse_attention_swa_only",
                "DSV4SparseAttentionTwoSourceBF16Kernel<false>::run",
            ),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def dsv4_sparse_attention_two_source_bf16(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    compressed_cache: torch.Tensor | None = None,
    compressed_indices: torch.Tensor | None = None,
    compressed_lengths: torch.Tensor | None = None,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    if not detect_dsv4_kernel_capabilities().is_ampere:
        raise RuntimeError("DSV4 sparse attention requires an Ampere SM80/SM86 device.")
    if (
        q.ndim != 3
        or swa_cache.ndim != 2
        or swa_indices.ndim != 2
        or swa_lengths.ndim != 1
        or q.shape[-1] != 512
        or swa_cache.shape[-1] != q.shape[-1]
        or q.shape[0] != swa_indices.shape[0]
        or q.shape[0] != swa_lengths.numel()
        or not q.is_cuda
        or not swa_cache.is_cuda
        or not swa_indices.is_cuda
        or not swa_lengths.is_cuda
        or q.dtype is not torch.bfloat16
        or swa_cache.dtype is not torch.bfloat16
        or swa_indices.dtype is not torch.int32
        or swa_lengths.dtype is not torch.int32
        or not q.is_contiguous()
        or not swa_cache.is_contiguous()
        or swa_indices.stride(-1) != 1
    ):
        raise RuntimeError("DSV4 sparse attention rejected its release tensor ABI.")

    has_compressed = (
        compressed_cache is not None
        and compressed_indices is not None
        and compressed_lengths is not None
        and compressed_cache.numel() > 0
    )
    if has_compressed:
        if (
            compressed_cache.ndim != 2
            or compressed_indices.ndim != 2
            or compressed_lengths.ndim != 1
            or compressed_cache.shape[-1] != q.shape[-1]
            or compressed_indices.shape[0] != q.shape[0]
            or compressed_lengths.numel() != q.shape[0]
            or not compressed_cache.is_cuda
            or not compressed_indices.is_cuda
            or not compressed_lengths.is_cuda
            or compressed_cache.dtype is not torch.bfloat16
            or compressed_indices.dtype is not torch.int32
            or compressed_lengths.dtype is not torch.int32
            or not compressed_cache.is_contiguous()
            or compressed_indices.stride(-1) != 1
        ):
            raise RuntimeError("DSV4 sparse attention rejected its compressed-cache tensor ABI.")
        compressed_cache_arg = compressed_cache
        compressed_indices_arg = compressed_indices
        compressed_lengths_arg = compressed_lengths
    else:
        compressed_cache_arg = swa_cache[:0]
        compressed_indices_arg = swa_indices[:, :1]
        compressed_lengths_arg = torch.zeros_like(swa_lengths)

    if attn_sink is None:
        sink = q.new_empty((1,), dtype=torch.float32)
    else:
        if (
            not attn_sink.is_cuda
            or attn_sink.dtype is not torch.float32
            or attn_sink.numel() < q.shape[1]
        ):
            raise RuntimeError("DSV4 sparse attention rejected its sink tensor ABI.")
        sink = attn_sink[: q.shape[1]].contiguous()

    out = torch.empty_like(q)
    module = _local_dsv4_sparse_attention_module()
    run = (
        module.sparse_attention_with_compressed
        if has_compressed
        else module.sparse_attention_swa_only
    )
    run(
        q,
        compressed_cache_arg,
        compressed_indices_arg,
        compressed_lengths_arg,
        swa_cache,
        swa_indices,
        swa_lengths,
        sink,
        out,
        float(softmax_scale),
        attn_sink is not None,
    )
    return out


def dsv4_sparse_attention_two_source_splitk_bf16(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    *,
    compressed_cache: torch.Tensor | None = None,
    compressed_indices: torch.Tensor | None = None,
    compressed_lengths: torch.Tensor | None = None,
    softmax_scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor | None:
    out = _triton_dsv4_ops().sparse_attention_splitk_bf16(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        compressed_lengths=compressed_lengths,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )
    if out is None:
        raise RuntimeError(
            "DSV4 exact bf16 sparse split-K decode does not support this "
            "tensor contract; optimized mode does not silently change backends."
        )
    return out


def _cuda_graph_capture_active(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _indexer_capture_static_max_seq_len(
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    capture_active: bool,
) -> tuple[int | None, str, str | None]:
    del seq_lens
    if not capture_active:
        return None, "eager_dynamic_seq_lens", None
    table_width = int(page_table.shape[1])
    return table_width * int(page_size), "current", None


def _indexer_fp8_paged_logits(
    q_values: torch.Tensor,
    packed_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    weights: torch.Tensor,
) -> torch.Tensor:
    capture_active = _cuda_graph_capture_active(q_values.device)
    static_max_seq_len, _, _ = _indexer_capture_static_max_seq_len(
        seq_lens,
        page_table,
        page_size,
        capture_active,
    )
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 paged FP8 indexer requires the Ampere Triton backend.")
    logits = _triton_dsv4_ops().indexer_fp8_paged_logits(
        q_values,
        packed_cache,
        weights,
        seq_lens,
        page_table,
        page_size=page_size,
        max_seq_len=static_max_seq_len,
    )
    if logits is None:
        raise RuntimeError("DSV4 paged FP8 indexer logits rejected the release tensor contract.")
    return logits


def _indexer_topk(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int,
    ratio: int,
) -> DSV4TopKTransformOutput:
    _validate_full_topk_inputs(
        scores,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=ratio,
    )
    if width not in (512, 1024):
        raise RuntimeError(f"DSV4 release top-k supports widths 512/1024, got {width}.")
    raw_indices = torch.empty(
        (scores.shape[0], width),
        dtype=torch.int32,
        device=scores.device,
    )
    page_indices = torch.empty_like(raw_indices)
    full_indices = torch.empty_like(raw_indices)
    topk_lens = torch.empty((scores.shape[0],), dtype=torch.int32, device=scores.device)
    clamped_lens = seq_lens.to(device=scores.device, dtype=torch.int32).clamp(
        min=0,
        max=scores.shape[1],
    )
    module = _local_dsv4_topk_v1_module(width)
    module.topk_transform_global_lens(
        scores.to(torch.float32).contiguous(),
        clamped_lens.contiguous(),
        page_table.to(device=scores.device, dtype=torch.int32).contiguous(),
        page_indices,
        page_size,
        raw_indices,
        full_indices,
        topk_lens,
        ratio,
    )
    return DSV4TopKTransformOutput(
        raw_indices,
        page_indices,
        full_indices,
        "local_cuda_global_topk_lens",
        topk_lens,
    )


def indexer_select_fp8_paged(
    q_values: torch.Tensor,
    weights: torch.Tensor,
    packed_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int = 512,
    ratio: int = 4,
) -> DSV4IndexerSelectOutput:
    """Select C4 index rows with the release packed-FP8 backend."""
    rows = int(q_values.shape[0])
    capture_active = _cuda_graph_capture_active(q_values.device)
    max_seq_len = (
        0 if capture_active else int(seq_lens.clamp_min(0).max().item()) if seq_lens.numel() else 0
    )
    max_logits_bytes = DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT * 1024 * 1024
    full_logits_bytes = rows * max(max_seq_len, 1) * torch.float32.itemsize

    if not capture_active and rows > 0 and max_seq_len > 0 and full_logits_bytes > max_logits_bytes:
        max_chunk_rows = max(
            1,
            max_logits_bytes // (max_seq_len * torch.float32.itemsize),
        )
        raw_indices = torch.empty((rows, width), dtype=torch.int32, device=q_values.device)
        page_indices = torch.empty_like(raw_indices)
        full_indices = torch.empty_like(raw_indices)
        topk_lens = torch.empty((rows,), dtype=torch.int32, device=q_values.device)
        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            logits = _indexer_fp8_paged_logits(
                q_values[start:end],
                packed_cache,
                seq_lens[start:end],
                page_table[start:end],
                page_size=page_size,
                weights=weights[start:end],
            )
            topk = _indexer_topk(
                logits,
                seq_lens[start:end],
                page_table[start:end],
                page_size=page_size,
                width=width,
                ratio=ratio,
            )
            raw_indices[start:end].copy_(topk.raw_indices)
            page_indices[start:end].copy_(topk.page_indices)
            full_indices[start:end].copy_(topk.full_indices)
            topk_lens[start:end].copy_(topk.topk_lens)
        return DSV4IndexerSelectOutput(
            logits=torch.empty((0, 0), dtype=torch.float32, device=q_values.device),
            topk=DSV4TopKTransformOutput(
                raw_indices,
                page_indices,
                full_indices,
                "bounded_query_chunks",
                topk_lens,
            ),
            backend="bounded_query_chunks+triton_fp8_paged_vllm+local_cuda_global_topk_lens",
        )

    logits = _indexer_fp8_paged_logits(
        q_values,
        packed_cache,
        seq_lens,
        page_table,
        page_size=page_size,
        weights=weights,
    )
    topk = _indexer_topk(
        logits,
        seq_lens,
        page_table,
        page_size=page_size,
        width=width,
        ratio=ratio,
    )
    return DSV4IndexerSelectOutput(
        logits=logits,
        topk=topk,
        backend="triton_fp8_paged_vllm+local_cuda_global_topk_lens",
    )


def remap_indexer_topk_locs(
    raw_indices: torch.Tensor,
    component_page_table: torch.Tensor,
    full_page_table: torch.Tensor,
    *,
    component_page_size: int,
    full_page_size: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map compressed raw top-k indices through the release Triton kernel."""
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 indexer top-k remap requires the Ampere Triton backend.")
    result = _triton_dsv4_ops().remap_indexer_topk_locs(
        raw_indices,
        component_page_table,
        full_page_table,
        component_page_size=int(component_page_size),
        full_page_size=int(full_page_size),
        ratio=int(ratio),
    )
    if result is None:
        raise RuntimeError("DSV4 indexer top-k remap rejected the release tensor ABI.")
    return result


def c128_prefill_page_indices_one_surface(
    component_page_table: torch.Tensor,
    c128_lengths: torch.Tensor,
    *,
    width: int,
    component_page_size: int,
    out: torch.Tensor | None = None,
    _backend: list[str] | None = None,
) -> torch.Tensor | None:
    """Build the release eager-prefill C128 final-location surface.

    This native micro boundary consumes only the Route-B component page table
    and raw C128 lengths, writes int32 component locations, and writes ``-1``
    for invalid tails/pages. It deliberately cannot materialize raw/full
    matrices or full-size int64 intermediates. TARGET 12.595 owns integration
    into attention metadata construction; decode graph contracts are unchanged.
    """
    cap = detect_dsv4_kernel_capabilities()
    if not (cap.is_ampere and cap.triton_available):
        return None
    result = _triton_dsv4_ops().c128_prefill_page_indices_one_surface(
        component_page_table,
        c128_lengths,
        width=int(width),
        component_page_size=int(component_page_size),
        out=out,
    )
    if result is not None and _backend is not None:
        _backend.append("triton_c128_prefill_one_surface")
    return result


def hc_pre(
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
    """Run the fused release HC prenorm boundary."""
    mixes = linear_bf16_fp32(x.flatten(1), fn)
    fused = _triton_dsv4_ops().hc_prenorm_split_pre(
        mixes.contiguous(),
        x,
        scale,
        base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        norm_eps=norm_eps,
    )
    if fused is None:
        raise RuntimeError("DSV4 HC prenorm rejected the release tensor contract.")
    return fused


def hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Run the fused release HC post boundary."""
    fused = _triton_dsv4_ops().hc_post(x, residual, post, comb)
    if fused is None:
        raise RuntimeError("DSV4 HC post kernel rejected the release tensor contract.")
    return fused


def hc_head(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    eps: float,
    norm_eps: float,
) -> torch.Tensor:
    """Run the fused release HC output boundary."""
    mixes = linear_bf16_fp32(x.flatten(1), fn)
    fused = _triton_dsv4_ops().hc_prenorm_head(
        mixes.contiguous(),
        x,
        scale,
        base,
        hc_mult=x.shape[1],
        eps=eps,
        norm_eps=norm_eps,
    )
    if fused is None:
        raise RuntimeError("DSV4 HC head kernel rejected the release tensor contract.")
    return fused


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
    if route_count and indices.is_cuda:
        if not dsv4_triton_available():
            raise RuntimeError("DSV4 MoE route planning requires the Ampere Triton backend.")
        route_plan = _triton_dsv4_ops().build_moe_route_plan(
            indices,
            num_experts=num_experts,
            block_size_m=block_size_m,
        )
        if route_plan is None:
            raise RuntimeError("DSV4 MoE route planning rejected the release tensor contract.")
        sorted_route_ids, expert_ids, num_tokens_post_padded = route_plan
        return DSV4MoERoutePlan(
            sorted_route_ids=sorted_route_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            route_count=route_count,
            topk=topk,
            block_size_m=block_size_m,
        )

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


def build_moe_v2_execution_plan(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_experts: int,
    block_size_m: int = 16,
    reduce_once: bool = True,
    final_reduce_label: str = "dsv4.v1_moe_reduce_once_all_reduce",
) -> DSV4MoEExecutionPlan:
    if hidden_states.ndim != 2:
        raise ValueError(
            f"DSV4 MoE V2 execution plan expects hidden_states [tokens, hidden], got {hidden_states.shape}"
        )
    if weights.shape != indices.shape or indices.ndim != 2:
        raise ValueError(
            "DSV4 MoE V2 execution plan expects matching weights/indices [tokens, topk], "
            f"got weights={weights.shape}, indices={indices.shape}"
        )
    route_plan = build_moe_route_plan(
        indices,
        num_experts=num_experts,
        block_size_m=block_size_m,
    )
    route_weights = (
        weights.to(device=hidden_states.device, dtype=torch.float32).reshape(-1).contiguous()
    )
    return DSV4MoEExecutionPlan(
        route_plan=route_plan,
        route_weights=route_weights,
        tokens=int(hidden_states.shape[0]),
        hidden=int(hidden_states.shape[1]),
        num_experts=int(num_experts),
        reduce_once=bool(reduce_once),
        final_reduce_label=final_reduce_label,
    )


def moe_execution_block_size(*, tokens: int, topk: int, num_experts: int) -> int:
    """Use the production backend's route blocking for the authoritative plan."""
    from minisgl.kernel import marlin_wna16

    return marlin_wna16.choose_block_size(
        tokens=tokens,
        topk=topk,
        experts=num_experts,
        input_dtype=None,
    )


def mask_moe_routes_live_rows(
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_token_non_padded: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared live-route contract before any route planning."""
    if num_token_non_padded is None:
        return weights, indices
    if weights.ndim != 2 or indices.shape != weights.shape:
        raise ValueError("DSV4 live-route masking expects matching weights/indices [tokens, topk]")
    if num_token_non_padded.numel() != 1:
        raise ValueError("num_token_non_padded must contain exactly one element")
    if num_token_non_padded.dtype != torch.int32:
        raise TypeError("num_token_non_padded must use torch.int32")
    if num_token_non_padded.device != weights.device or indices.device != weights.device:
        raise ValueError("live count, route weights, and route IDs must share a device")
    if weights.is_cuda:
        if not dsv4_triton_available():
            raise RuntimeError("DSV4 live-route masking requires the Ampere Triton backend.")
        if not _triton_dsv4_ops().mask_moe_routes_live_rows(
            weights,
            indices,
            num_token_non_padded,
        ):
            raise RuntimeError("DSV4 live-route masking rejected the release tensor contract.")
        return weights, indices
    rows = torch.arange(weights.shape[0], device=weights.device)
    live = (rows < num_token_non_padded).unsqueeze(1)
    return (
        torch.where(live, weights, torch.zeros_like(weights)),
        torch.where(live, indices, torch.full_like(indices, -1)),
    )


def zero_moe_padded_rows(
    output: torch.Tensor,
    num_token_non_padded: torch.Tensor | None,
) -> torch.Tensor:
    """Finalize excluded MoE rows without clearing the maximum workspace."""
    if num_token_non_padded is None:
        return output
    if output.ndim != 2:
        raise ValueError("DSV4 MoE padded-row finalize expects [tokens, hidden]")
    if num_token_non_padded.numel() != 1 or num_token_non_padded.dtype != torch.int32:
        raise ValueError("num_token_non_padded must be a one-element int32 tensor")
    if num_token_non_padded.device != output.device:
        raise ValueError("live count and MoE output must share a device")
    if output.is_cuda:
        if not dsv4_triton_available():
            raise RuntimeError("DSV4 padded-row finalization requires the Ampere Triton backend.")
        if not _triton_dsv4_ops().zero_moe_padded_rows(output, num_token_non_padded):
            raise RuntimeError("DSV4 padded-row finalization rejected the release tensor contract.")
        return output
    rows = torch.arange(output.shape[0], device=output.device)
    return torch.where(
        (rows < num_token_non_padded).unsqueeze(1),
        output,
        torch.zeros_like(output),
    )


def moe_gate(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    input_ids: torch.Tensor | None,
    topk: int,
    scoring_func: str,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
    hash_topk=None,
    num_token_non_padded: torch.Tensor | None = None,
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
        indices = hash_topk.forward(input_ids.view(-1)).long()
    else:
        scores_for_topk = original_scores
        if correction_bias is not None:
            scores_for_topk = scores_for_topk + correction_bias.float()
        indices = scores_for_topk.topk(topk, dim=-1).indices

    weights = original_scores.gather(1, indices)
    if scoring_func != "softmax":
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    weights = weights * routed_scaling_factor
    return mask_moe_routes_live_rows(weights, indices, num_token_non_padded)


def silu_and_mul_clamp(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the release fused SwiGLU kernel without a Torch fallback."""
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 fused SwiGLU requires the Ampere Triton backend.")
    out = _triton_dsv4_ops().silu_and_mul_clamp(
        gate,
        up,
        swiglu_limit=swiglu_limit,
        weights=weights,
    )
    if out is None:
        raise RuntimeError("DSV4 fused SwiGLU rejected the release tensor contract.")
    return out


@lru_cache(maxsize=1)
def _local_dsv4_topk_v1_module(width: int):
    return load_jit(
        "dsv4_topk_v1",
        str(int(width)),
        cuda_files=["dsv4_topk_v1.cu"],
        cuda_wrappers=[
            ("topk_transform", f"DSV4TopKTransformKernel<{int(width)}>::run"),
            (
                "topk_transform_global_lens",
                f"DSV4TopKTransformGlobalLensKernel<{int(width)}>::run",
            ),
        ],
    )


def _validate_full_topk_inputs(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    page_size: int,
    width: int,
    ratio: int,
) -> None:
    if scores.dim() != 2:
        raise ValueError(f"scores must have shape [B, max_seq_len], got {tuple(scores.shape)}")
    if seq_lens.dim() != 1 or seq_lens.shape[0] != scores.shape[0]:
        raise ValueError("seq_lens must be int32/int64 with shape [B]")
    if page_table.dim() != 2 or page_table.shape[0] != scores.shape[0]:
        raise ValueError("page_table must have shape [B, num_pages]")
    if scores.shape[1] > 0 and page_table.shape[1] == 0:
        raise ValueError("page_table must contain at least one page when scores are non-empty")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"page_size must be a positive power of two, got {page_size}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")


def prep_decode_metadata_in_graph(
    *,
    ctx_page_table: torch.Tensor,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    materialized_seq_lens: torch.Tensor,
    c4_page_table: torch.Tensor | None,
    c128_page_table: torch.Tensor | None,
    c4_indexer_page_table: torch.Tensor | None,
    dst_seq_lens: torch.Tensor,
    dst_swa_topk_lengths: torch.Tensor,
    dst_c4_topk_lengths_raw: torch.Tensor,
    dst_c4_topk_lengths_clamp1: torch.Tensor,
    dst_c4_sparse_topk_lengths: torch.Tensor,
    dst_c128_topk_lengths_clamp1: torch.Tensor,
    dst_swa_page_indices: torch.Tensor,
    dst_c4_sparse_raw_indices: torch.Tensor,
    dst_c4_sparse_page_indices: torch.Tensor,
    dst_c4_sparse_full_indices: torch.Tensor,
    dst_c128_raw_indices: torch.Tensor,
    dst_c128_page_indices: torch.Tensor,
    dst_c128_full_indices: torch.Tensor,
    dst_c4_out_loc: torch.Tensor | None,
    dst_c128_out_loc: torch.Tensor | None,
    dst_c4_indexer_out_loc: torch.Tensor | None,
    dst_swa_out_loc: torch.Tensor | None = None,
    rows: int,
    page_size: int,
    window_size: int,
    index_topk: int,
    swa_full_to_swa_page: torch.Tensor,
    swa_dummy_token_start: int = -1,
    swa_dummy_page: int = -1,
) -> bool:
    if rows < 0 or page_size <= 0 or window_size <= 0 or index_topk <= 0:
        return False
    if page_size & (page_size - 1):
        return False
    if rows == 0:
        return True
    if c4_page_table is None or c128_page_table is None or c4_indexer_page_table is None:
        return False
    if dst_c4_out_loc is None or dst_c128_out_loc is None or dst_c4_indexer_out_loc is None:
        return False
    tensors = [
        ctx_page_table,
        table_indices,
        positions,
        raw_out_loc,
        materialized_seq_lens,
        c4_page_table,
        c128_page_table,
        c4_indexer_page_table,
        dst_seq_lens,
        dst_swa_topk_lengths,
        dst_c4_topk_lengths_raw,
        dst_c4_topk_lengths_clamp1,
        dst_c4_sparse_topk_lengths,
        dst_c128_topk_lengths_clamp1,
        dst_swa_page_indices,
        dst_c4_sparse_raw_indices,
        dst_c4_sparse_page_indices,
        dst_c4_sparse_full_indices,
        dst_c128_raw_indices,
        dst_c128_page_indices,
        dst_c128_full_indices,
        dst_c4_out_loc,
        dst_c128_out_loc,
        dst_c4_indexer_out_loc,
    ]
    if swa_full_to_swa_page is None or swa_dummy_token_start < 0 or swa_dummy_page < 0:
        return False
    tensors.append(swa_full_to_swa_page)
    if dst_swa_out_loc is not None:
        tensors.append(dst_swa_out_loc)
    if not all(t.is_cuda and t.dtype is torch.int32 and t.is_contiguous() for t in tensors):
        return False
    if (
        ctx_page_table.ndim != 2
        or table_indices.ndim != 1
        or positions.ndim != 1
        or raw_out_loc.ndim != 1
        or materialized_seq_lens.ndim != 1
        or c4_page_table.ndim != 2
        or c128_page_table.ndim != 2
        or c4_indexer_page_table.ndim != 2
        or any(t.ndim != 1 for t in tensors[8:14])
        or any(t.ndim != 2 for t in tensors[14:21])
        or any(t.ndim != 1 for t in tensors[21:])
    ):
        return False
    if (
        table_indices.numel() < rows
        or positions.numel() < rows
        or raw_out_loc.numel() < rows
        or materialized_seq_lens.numel() < rows
        or c4_page_table.shape[0] < rows
        or c128_page_table.shape[0] < rows
        or c4_indexer_page_table.shape[0] < rows
        or any(t.numel() < rows for t in tensors[8:14])
        or any(t.shape[0] < rows for t in tensors[14:21])
        or any(t.numel() < rows for t in tensors[21:24])
    ):
        return False
    if swa_full_to_swa_page.ndim != 1:
        return False
    if dst_swa_out_loc is not None and (
        dst_swa_out_loc.ndim != 1 or dst_swa_out_loc.numel() < rows
    ):
        return False
    dummy_swa_out_loc = dst_swa_out_loc if dst_swa_out_loc is not None else raw_out_loc
    return bool(
        _triton_dsv4_ops().prep_decode_metadata_in_graph(
            ctx_page_table,
            table_indices[:rows],
            positions[:rows],
            raw_out_loc[:rows],
            materialized_seq_lens[:rows],
            c4_page_table[:rows],
            c128_page_table[:rows],
            c4_indexer_page_table[:rows],
            dst_seq_lens,
            dst_swa_topk_lengths,
            dst_c4_topk_lengths_raw,
            dst_c4_topk_lengths_clamp1,
            dst_c4_sparse_topk_lengths,
            dst_c128_topk_lengths_clamp1,
            dst_swa_page_indices,
            dst_c4_sparse_raw_indices,
            dst_c4_sparse_page_indices,
            dst_c4_sparse_full_indices,
            dst_c128_raw_indices,
            dst_c128_page_indices,
            dst_c128_full_indices,
            dst_c4_out_loc,
            dst_c128_out_loc,
            dst_c4_indexer_out_loc,
            swa_full_to_swa_page,
            dummy_swa_out_loc,
            page_size=int(page_size),
            window_size=int(window_size),
            index_topk=int(index_topk),
            swa_dummy_token_start=int(swa_dummy_token_start),
            swa_dummy_page=int(swa_dummy_page),
            write_swa_out_loc=dst_swa_out_loc is not None,
        )
    )


def compress_norm_rope_store(
    kvcache,
    layer_id: int,
    kv: torch.Tensor,
    loc: torch.Tensor,
    *,
    positions: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_norm_eps: float,
    rotary_dim: int,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    cache_type: Literal["compressed", "indexer"] = "compressed",
    apply_hadamard: bool = False,
) -> None:
    """Publish compressed release state without cache-layout or backend fallback."""
    if not dsv4_triton_available():
        raise RuntimeError("DSV4 compressed-state publication requires the Ampere backend.")
    dim = kv.shape[-1]
    flat = kv.reshape(-1, dim)
    loc_flat = loc.to(device=flat.device, dtype=torch.long).reshape(-1)
    positions_flat = positions.to(device=flat.device, dtype=torch.long).reshape(-1)
    if loc_flat.numel() != flat.shape[0] or positions_flat.numel() != flat.shape[0]:
        raise ValueError("DSV4 compressed-state rows, positions, and locations must match.")
    if norm_weight.numel() != dim:
        raise ValueError(
            f"DSV4 compressed norm weight has {norm_weight.numel()} values for dim {dim}."
        )

    if cache_type == "indexer":
        if not apply_hadamard:
            raise RuntimeError("DSV4 release indexer publication requires Hadamard folding.")
        packed_cache = kvcache.indexer_fp8_paged_cache(layer_id)
        norm_weight_fp32 = norm_weight.to(
            device=flat.device,
            dtype=torch.float32,
        ).contiguous()
        if not _c4_indexer_rmsnorm_bf16_native(
            flat,
            norm_weight_fp32,
            loc_flat,
            rms_norm_eps=float(rms_norm_eps),
        ):
            raise RuntimeError("DSV4 C4 indexer RMSNorm rejected the release tensor ABI.")
        if not _triton_dsv4_ops().indexer_rotary_tail_valid(
            flat,
            positions_flat,
            loc_flat,
            rotary_dim=rotary_dim,
            base=base,
            original_seq_len=original_seq_len,
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        ):
            raise RuntimeError("DSV4 C4 indexer RoPE rejected the release tensor ABI.")
        if not _triton_dsv4_ops().indexer_hadamard_fp8_paged_store(
            flat,
            loc_flat,
            packed_cache,
            page_size=int(kvcache.indexer_fp8_page_size),
        ):
            raise RuntimeError("DSV4 C4 indexer store rejected the release tensor ABI.")
        return

    if apply_hadamard:
        raise RuntimeError("Hadamard folding is only valid for DSV4 indexer publication.")
    cache = kvcache.component_cache(layer_id)
    if not _triton_dsv4_ops().compress_norm_rope_store_bf16(
        flat,
        positions_flat,
        norm_weight,
        cache,
        loc_flat,
        rms_norm_eps=float(rms_norm_eps),
        rotary_dim=rotary_dim,
        base=base,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    ):
        raise RuntimeError("DSV4 compressed-state store rejected the release tensor ABI.")


__all__ = [
    "DSV4KernelCapability",
    "DSV4IndexerFP8Query",
    "DSV4IndexerSelectOutput",
    "DSV4_INDEXER_MAX_LOGITS_MB_DEFAULT",
    "DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_WNA16",
    "DSV4_MARLIN_WNA16_RELEASE_ERROR",
    "DSV4MoEExecutionPlan",
    "DSV4MoERoutePlan",
    "DSV4PagedMQAMetadata",
    "DSV4TopKTransformOutput",
    "DSV4TwoSourceAttentionMetadata",
    "rotary_tail",
    "build_moe_route_plan",
    "build_moe_v2_execution_plan",
    "prep_decode_metadata_in_graph",
    "compress_norm_rope_store",
    "c4_online_pool_and_update",
    "c128_online_pool_and_update",
    "dequant_fp8_weight",
    "detect_dsv4_kernel_capabilities",
    "dsv4_cuda_available",
    "dsv4_triton_available",
    "dsv4_sparse_attention_two_source_bf16",
    "dsv4_sparse_attention_two_source_splitk_bf16",
    "e8m0_dtype",
    "fp8_dtype",
    "hc_head",
    "hc_post",
    "hc_pre",
    "indexer_q_rope_fp8",
    "indexer_select_fp8_paged",
    "linear_bf16_fp32",
    "linear_bf16_fp32_upstream_enabled",
    "moe_gate",
    "moe_route_dispatch_bf16_marlin_wna16_prepacked",
    "q_kv_norm_rope_cache",
    "quantize_fp8_activation",
    "remap_indexer_topk_locs",
    "scale_dim",
    "silu_and_mul_clamp",
    "rms_norm",
]

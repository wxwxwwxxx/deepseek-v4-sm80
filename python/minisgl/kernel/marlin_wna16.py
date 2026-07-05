from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from minisgl.utils import dsv4_memory_debug

FLOAT4_E2M1F_ID = 562949953487106
EXTENSION_NAME = "minisgl_marlin_wna16"
SOURCE_ROOT = Path(__file__).resolve().parent / "csrc/vendor/vllm_marlin_wna16"


@dataclass(frozen=True)
class MarlinWNA16Weights:
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    source_signature: tuple[tuple[int, tuple[int, ...], torch.dtype], ...]

    def matches(
        self,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> bool:
        return self.source_signature == _source_signature(
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
        )


def _source_signature(
    *tensors: torch.Tensor,
) -> tuple[tuple[int, tuple[int, ...], torch.dtype], ...]:
    return tuple((tensor.data_ptr(), tuple(tensor.shape), tensor.dtype) for tensor in tensors)


def _build_dir() -> Path:
    configured = os.environ.get("MINISGL_MARLIN_WNA16_BUILD_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache/minisgl/marlin_wna16"


@lru_cache(maxsize=1)
def load_ops() -> Any:
    from torch.utils.cpp_extension import load

    source_root = SOURCE_ROOT
    if not source_root.exists():
        raise FileNotFoundError(f"vendored Marlin WNA16 source root is missing: {source_root}")
    sources = [
        source_root / "schema.cpp",
        source_root / "quantization/marlin/gptq_marlin_repack.cu",
        source_root / "moe/marlin_moe_wna16/ops.cu",
        *sorted((source_root / "moe/marlin_moe_wna16").glob("sm80_kernel_*.cu")),
    ]
    include_dirs = [
        source_root,
        source_root / "moe",
        source_root / "quantization",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"vendored Marlin WNA16 sources are missing: {missing}")

    build_dir = _build_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "8")
    load(
        name=EXTENSION_NAME,
        sources=[str(path) for path in sources],
        extra_include_paths=[str(path) for path in include_dirs],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-static-global-template-stub=false",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_80,code=compute_80",
        ],
        build_directory=str(build_dir),
        verbose=bool(os.environ.get("MINISGL_MARLIN_WNA16_VERBOSE_BUILD")),
        with_cuda=True,
    )
    return getattr(torch.ops, EXTENSION_NAME)


def _get_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _marlin_permute_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    scale_perm, scale_perm_single = _get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        scales = scales.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        scales = scales.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return scales.reshape((-1, size_n)).contiguous()


def _mxfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    *,
    input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if input_dtype is None or input_dtype.itemsize == 2:
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
            marlin_scales.size(0),
            -1,
        )
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        if bool((marlin_scales > 249).any().item()):
            raise ValueError("MXFP4 E8M0 scale exponent overflow for FP8 activation")
        marlin_scales = (marlin_scales + 6).view(torch.float8_e8m0fnu)
    return marlin_scales.contiguous()


def prepare_moe_mxfp4_weights(
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    params_dtype: torch.dtype = torch.bfloat16,
    owner_label: str | None = None,
    cache_was_present: bool | None = None,
    cache_signature_match: bool | None = None,
) -> MarlinWNA16Weights:
    if w13_weight.ndim != 4 or w13_weight.shape[1] != 2:
        raise ValueError(f"expected w13 shape [E,2,N,K/2], got {tuple(w13_weight.shape)}")
    if w2_weight.ndim != 3:
        raise ValueError(f"expected w2 shape [E,K,N/2], got {tuple(w2_weight.shape)}")
    experts, _, intermediate, packed_hidden = w13_weight.shape
    hidden = packed_hidden * 2
    if w2_weight.shape != (experts, hidden, intermediate // 2):
        raise ValueError(
            "expected w2 shape "
            f"{(experts, hidden, intermediate // 2)}, got {tuple(w2_weight.shape)}"
        )

    debug = (
        dsv4_memory_debug.env_flag(dsv4_memory_debug.DSV4_MARLIN_WNA16_CACHE_DEBUG_ENV)
        and w13_weight.is_cuda
    )
    before_memory = (
        dsv4_memory_debug.cuda_memory_snapshot(w13_weight.device, synchronize=True)
        if debug
        else {}
    )
    start_s = time.perf_counter()

    ops = load_ops()
    perm = torch.empty(0, dtype=torch.int, device=w13_weight.device)
    w13_u8 = (
        w13_weight.contiguous()
        .view(torch.uint8)
        .view(
            experts,
            2 * intermediate,
            hidden // 2,
        )
    )
    w2_u8 = w2_weight.contiguous().view(torch.uint8)

    def repack_weight(weight: torch.Tensor, *, size_n: int, size_k: int) -> torch.Tensor:
        pieces = []
        for expert in range(experts):
            qweight = weight[expert].view(torch.int32).T.contiguous()
            pieces.append(
                ops.gptq_marlin_repack(
                    qweight,
                    perm,
                    size_k,
                    size_n,
                    4,
                    False,
                )
            )
        return torch.stack(pieces, dim=0)

    def permute_scales(scales: torch.Tensor, *, size_n: int, size_k: int) -> torch.Tensor:
        typed = scales.contiguous().view(torch.uint8).view(torch.float8_e8m0fnu).to(params_dtype)
        pieces = []
        for expert in range(experts):
            marlin_scales = _marlin_permute_scales(
                typed[expert].T,
                size_k=size_k,
                size_n=size_n,
                group_size=32,
            )
            pieces.append(_mxfp4_marlin_process_scales(marlin_scales))
        return torch.stack(pieces, dim=0)

    marlin_w13 = repack_weight(w13_u8, size_n=2 * intermediate, size_k=hidden)
    marlin_w2 = repack_weight(w2_u8, size_n=hidden, size_k=intermediate)
    marlin_w13_scale = permute_scales(
        w13_scale.contiguous().view(experts, 2 * intermediate, hidden // 32),
        size_n=2 * intermediate,
        size_k=hidden,
    )
    marlin_w2_scale = permute_scales(
        w2_scale.contiguous().view(experts, hidden, intermediate // 32),
        size_n=hidden,
        size_k=intermediate,
    )
    result = MarlinWNA16Weights(
        w13=marlin_w13,
        w2=marlin_w2,
        w13_scale=marlin_w13_scale,
        w2_scale=marlin_w2_scale,
        source_signature=_source_signature(w13_weight, w13_scale, w2_weight, w2_scale),
    )
    if debug:
        after_memory = dsv4_memory_debug.cuda_memory_snapshot(w13_weight.device, synchronize=True)
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        source_bytes = {
            "w13_weight": dsv4_memory_debug.tensor_nbytes(w13_weight),
            "w13_scale": dsv4_memory_debug.tensor_nbytes(w13_scale),
            "w2_weight": dsv4_memory_debug.tensor_nbytes(w2_weight),
            "w2_scale": dsv4_memory_debug.tensor_nbytes(w2_scale),
        }
        repacked_bytes = {
            "w13": dsv4_memory_debug.tensor_nbytes(marlin_w13),
            "w13_scale": dsv4_memory_debug.tensor_nbytes(marlin_w13_scale),
            "w2": dsv4_memory_debug.tensor_nbytes(marlin_w2),
            "w2_scale": dsv4_memory_debug.tensor_nbytes(marlin_w2_scale),
        }
        dsv4_memory_debug.append_jsonl(
            "marlin_wna16_cache",
            {
                "event": "dsv4_marlin_wna16_prepare_moe_mxfp4_weights",
                "owner": owner_label,
                "cache_was_present": cache_was_present,
                "cache_signature_match": cache_signature_match,
                "experts": int(experts),
                "hidden": int(hidden),
                "local_intermediate": int(intermediate),
                "params_dtype": str(params_dtype),
                "elapsed_ms": elapsed_ms,
                "source_tensors": {
                    "w13_weight": dsv4_memory_debug.tensor_summary(w13_weight),
                    "w13_scale": dsv4_memory_debug.tensor_summary(w13_scale),
                    "w2_weight": dsv4_memory_debug.tensor_summary(w2_weight),
                    "w2_scale": dsv4_memory_debug.tensor_summary(w2_scale),
                },
                "repacked_tensors": {
                    "w13": dsv4_memory_debug.tensor_summary(marlin_w13),
                    "w13_scale": dsv4_memory_debug.tensor_summary(marlin_w13_scale),
                    "w2": dsv4_memory_debug.tensor_summary(marlin_w2),
                    "w2_scale": dsv4_memory_debug.tensor_summary(marlin_w2_scale),
                },
                "source_total_bytes": int(sum(source_bytes.values())),
                "repacked_total_bytes": int(sum(repacked_bytes.values())),
                "before_memory": before_memory,
                "after_memory": after_memory,
                "free_delta_bytes": int(
                    before_memory.get("free_memory_bytes", 0)
                    - after_memory.get("free_memory_bytes", 0)
                ),
                "memory_allocated_delta_bytes": int(
                    after_memory.get("memory_allocated_bytes", 0)
                    - before_memory.get("memory_allocated_bytes", 0)
                ),
                "memory_reserved_delta_bytes": int(
                    after_memory.get("memory_reserved_bytes", 0)
                    - before_memory.get("memory_reserved_bytes", 0)
                ),
            },
        )
    return result


def choose_block_size(
    *,
    tokens: int,
    topk: int,
    experts: int,
    input_dtype: torch.dtype | None = None,
) -> int:
    block_size_m = 64
    for candidate in [8, 16, 32, 48, 64]:
        block_size_m = candidate
        if tokens * topk / experts / candidate < 0.9:
            break
    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)
    return block_size_m


def run_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    weights: MarlinWNA16Weights,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_size_m: int,
    swiglu_limit: float,
) -> torch.Tensor:
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Marlin WNA16 expects fp16/bf16 activations, got {hidden_states.dtype}")
    if not hidden_states.is_cuda:
        raise ValueError("Marlin WNA16 requires CUDA tensors")
    hidden_states = hidden_states.contiguous()
    topk_weights = topk_weights.to(device=hidden_states.device, dtype=torch.float32).contiguous()
    tokens, hidden = hidden_states.shape
    topk = topk_weights.shape[1]
    intermediate = weights.w2.shape[1] * 16

    ops = load_ops()
    sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
    workspace = torch.zeros(sms * 4, dtype=torch.int, device=hidden_states.device)
    w13_out = torch.empty(
        (tokens * topk, 2 * intermediate),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    activated = torch.empty(
        (tokens * topk, intermediate),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    route_out = torch.empty(
        (tokens * topk, hidden),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w13_out = ops.moe_wna16_marlin_gemm(
        hidden_states,
        w13_out,
        weights.w13,
        None,
        weights.w13_scale,
        None,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        block_size_m,
        topk,
        False,
        FLOAT4_E2M1F_ID,
        tokens,
        2 * intermediate,
        hidden,
        True,
        False,
        True,
        False,
        -1,
        -1,
        -1,
    )
    gate = w13_out[:, :intermediate].float()
    up = w13_out[:, intermediate:].float()
    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    activated.copy_((F.silu(gate) * up).to(activated.dtype))
    route_out = ops.moe_wna16_marlin_gemm(
        activated,
        route_out,
        weights.w2,
        None,
        weights.w2_scale,
        None,
        None,
        None,
        None,
        None,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        block_size_m,
        1,
        True,
        FLOAT4_E2M1F_ID,
        tokens * topk,
        hidden,
        intermediate,
        True,
        False,
        True,
        False,
        -1,
        -1,
        -1,
    )
    return route_out.view(tokens, topk, hidden).sum(dim=1)

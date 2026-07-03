from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

EXTENSION_NAME = "minisgl_dense_fp8_marlin"
SOURCE_ROOT = Path(__file__).resolve().parent / "csrc/vendor/vllm_dense_fp8_marlin"

# Mirrors vLLM scalar_types.float8_e4m3fn.id without importing vLLM.
FLOAT8_E4M3FN_ID = 2814749767172868
USE_FP32_REDUCE_DEFAULT = True


@dataclass(frozen=True)
class DenseFP8MarlinLinearWeight:
    weight: torch.Tensor
    weight_scale: torch.Tensor
    workspace: torch.Tensor
    size_n: int
    size_k: int
    source_signature: tuple[tuple[int, tuple[int, ...], torch.dtype], ...]
    original_weight_bytes: int
    original_scale_bytes: int
    prepared_weight_bytes: int
    prepared_scale_bytes: int
    workspace_bytes: int
    prepare_ms: float | None = None

    @property
    def persistent_bytes(self) -> int:
        return int(self.prepared_weight_bytes + self.prepared_scale_bytes + self.workspace_bytes)


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def source_signature(
    *tensors: torch.Tensor | None,
) -> tuple[tuple[int, tuple[int, ...], torch.dtype], ...]:
    return tuple(
        (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        for tensor in tensors
        if tensor is not None
    )


def _build_dir() -> Path:
    configured = os.environ.get("MINISGL_DENSE_FP8_MARLIN_BUILD_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache/minisgl/dense_fp8_marlin"


@lru_cache(maxsize=1)
def load_ops() -> Any:
    from torch.utils.cpp_extension import load

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"vendored dense FP8 Marlin source root is missing: {SOURCE_ROOT}")
    marlin_root = SOURCE_ROOT / "quantization/marlin"
    sources = [
        SOURCE_ROOT / "schema.cpp",
        marlin_root / "gptq_marlin_repack.cu",
        marlin_root / "marlin.cu",
        marlin_root / "sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu",
    ]
    include_dirs = [
        SOURCE_ROOT,
        SOURCE_ROOT / "quantization",
        marlin_root,
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"vendored dense FP8 Marlin sources are missing: {missing}")

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
        verbose=bool(os.environ.get("MINISGL_DENSE_FP8_MARLIN_VERBOSE_BUILD")),
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


def marlin_permute_scales(
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


def marlin_permute_bias(bias: torch.Tensor) -> torch.Tensor:
    origin_shape = bias.shape
    _, scale_perm_single = _get_scale_perms()
    bias = bias.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return bias.reshape(*origin_shape).contiguous()


def fp8_fused_exponent_bias_into_scales(scales: torch.Tensor) -> torch.Tensor:
    fp8_exponent = 4
    if scales.dtype == torch.float16:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise ValueError(f"FP8 Marlin scales must be fp16/bf16 before bias fusion, got {scales.dtype}")
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    return scales * (torch.ones_like(scales) * 2) ** exponent_bias


def pack_fp8_to_int32(fp8_tensor: torch.Tensor, *, size_k_first: bool = True) -> torch.Tensor:
    if fp8_tensor.dtype != torch.float8_e4m3fn:
        raise ValueError(f"expected torch.float8_e4m3fn weight, got {fp8_tensor.dtype}")
    if fp8_tensor.ndim != 2:
        raise ValueError(f"expected 2D FP8 weight, got shape {tuple(fp8_tensor.shape)}")

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor


def marlin_make_workspace(device: torch.device, *, max_blocks_per_sm: int = 1) -> torch.Tensor:
    if device.type != "cuda":
        raise ValueError(f"dense FP8 Marlin workspace requires CUDA, got {device}")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False)


def should_use_atomic_add_reduce(
    *,
    m: int,
    n: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False
    enabled = os.environ.get("MINISGL_DENSE_FP8_MARLIN_USE_ATOMIC_ADD")
    if enabled is None:
        enabled = os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD")
    if enabled not in {"1", "true", "TRUE", "on", "ON"}:
        return False
    major, _ = torch.cuda.get_device_capability(device)
    if major < 9 and dtype == torch.bfloat16:
        return False
    return True


def prepare_dense_fp8_marlin_weight(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    owner_label: str,
    params_dtype: torch.dtype = torch.bfloat16,
    weight_block_size: tuple[int, int] = (128, 128),
) -> DenseFP8MarlinLinearWeight:
    if weight_scale_inv is None:
        raise RuntimeError(f"{owner_label} dense FP8 Marlin requires weight_scale_inv.")
    if weight.ndim != 2 or weight_scale_inv.ndim != 2:
        raise RuntimeError(
            f"{owner_label} dense FP8 Marlin expects 2D weight/scale, got "
            f"weight={tuple(weight.shape)} scale={tuple(weight_scale_inv.shape)}."
        )
    if weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"{owner_label} dense FP8 Marlin expects FP8 e4m3fn weight.")
    if not weight.is_cuda or not weight_scale_inv.is_cuda:
        raise RuntimeError(f"{owner_label} dense FP8 Marlin requires CUDA weight and scale.")
    if weight.device != weight_scale_inv.device:
        raise RuntimeError(
            f"{owner_label} dense FP8 Marlin requires scale on the same device as weight, "
            f"got weight={weight.device} scale={weight_scale_inv.device}."
        )

    size_n, size_k = int(weight.shape[0]), int(weight.shape[1])
    block_n, block_k = weight_block_size
    if block_n != 128 or block_k != 128:
        raise RuntimeError(
            f"{owner_label} dense FP8 Marlin currently supports 128x128 block FP8 scales only, "
            f"got {weight_block_size}."
        )
    expected_scale_shape = ((size_n + block_n - 1) // block_n, (size_k + block_k - 1) // block_k)
    if tuple(weight_scale_inv.shape) != expected_scale_shape:
        raise RuntimeError(
            f"{owner_label} dense FP8 Marlin expected scale shape {expected_scale_shape}, "
            f"got {tuple(weight_scale_inv.shape)}."
        )
    if size_k % 128 != 0 or size_n % 64 != 0:
        raise RuntimeError(
            f"{owner_label} dense FP8 Marlin requires size_k % 128 == 0 and "
            f"size_n % 64 == 0, got N={size_n} K={size_k}."
        )

    ops = load_ops()
    original_weight_bytes = tensor_bytes(weight)
    original_scale_bytes = tensor_bytes(weight_scale_inv)
    perm = torch.empty(0, dtype=torch.int, device=weight.device)

    qweight = pack_fp8_to_int32(weight, size_k_first=False).T.contiguous()
    marlin_qweight = ops.gptq_marlin_repack(
        qweight,
        perm,
        size_k,
        size_n,
        8,
        False,
    )

    scales = weight_scale_inv.to(params_dtype)
    scales = scales.T.contiguous()
    scales = scales.repeat_interleave(block_n, dim=1)
    scales = scales[:, :size_n]
    marlin_scales = marlin_permute_scales(
        scales,
        size_k=size_k,
        size_n=size_n,
        group_size=block_k,
    )
    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)

    workspace = marlin_make_workspace(weight.device)
    return DenseFP8MarlinLinearWeight(
        weight=marlin_qweight,
        weight_scale=marlin_scales,
        workspace=workspace,
        size_n=size_n,
        size_k=size_k,
        source_signature=source_signature(weight, weight_scale_inv),
        original_weight_bytes=original_weight_bytes,
        original_scale_bytes=original_scale_bytes,
        prepared_weight_bytes=tensor_bytes(marlin_qweight),
        prepared_scale_bytes=tensor_bytes(marlin_scales),
        workspace_bytes=tensor_bytes(workspace),
    )


def apply_dense_fp8_marlin_linear(
    x: torch.Tensor,
    prepared: DenseFP8MarlinLinearWeight,
    *,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"dense FP8 Marlin W8A16 expects fp16/bf16 activations, got {x.dtype}.")
    if not x.is_cuda:
        raise RuntimeError("dense FP8 Marlin W8A16 requires CUDA activations.")
    if bias is not None:
        bias = marlin_permute_bias(bias.contiguous())

    reshaped_x = x.reshape(-1, x.shape[-1])
    if reshaped_x.stride(-1) != 1 or reshaped_x.stride(0) % 8 != 0:
        reshaped_x = reshaped_x.contiguous()
    out_shape = x.shape[:-1] + (prepared.size_n,)
    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=prepared.size_n,
        k=prepared.size_k,
        device=x.device,
        dtype=x.dtype,
    )

    ops = load_ops()
    output = ops.marlin_gemm(
        reshaped_x,
        None,
        prepared.weight,
        bias,
        prepared.weight_scale,
        None,
        None,
        None,
        None,
        None,
        prepared.workspace,
        FLOAT8_E4M3FN_ID,
        reshaped_x.size(0),
        prepared.size_n,
        prepared.size_k,
        True,
        use_atomic_add,
        use_fp32_reduce,
        False,
    )
    return output.reshape(out_shape)


def prepare_dense_fp8_marlin_report(
    prepared: DenseFP8MarlinLinearWeight,
    *,
    owner_label: str,
) -> dict[str, object]:
    return {
        "owner": owner_label,
        "shape": [int(prepared.size_n), int(prepared.size_k)],
        "dtype": "mini_dense_fp8_marlin_w8a16_block",
        "device": str(prepared.weight.device),
        "prepared_weight_bytes": int(prepared.prepared_weight_bytes),
        "prepared_scale_bytes": int(prepared.prepared_scale_bytes),
        "workspace_bytes": int(prepared.workspace_bytes),
        "persistent_bytes": int(prepared.persistent_bytes),
        "original_weight_bytes": int(prepared.original_weight_bytes),
        "original_scale_bytes": int(prepared.original_scale_bytes),
        "original_needed_after_packing": False,
    }


__all__ = [
    "DenseFP8MarlinLinearWeight",
    "FLOAT8_E4M3FN_ID",
    "apply_dense_fp8_marlin_linear",
    "load_ops",
    "pack_fp8_to_int32",
    "prepare_dense_fp8_marlin_report",
    "prepare_dense_fp8_marlin_weight",
    "tensor_bytes",
]

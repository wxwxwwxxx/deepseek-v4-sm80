from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VllmFP8MarlinLinearWeight:
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


@lru_cache(maxsize=1)
def _helpers() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            process_fp8_weight_block_strategy,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
            prepare_fp8_layer_for_marlin,
        )
    except Exception as exc:  # noqa: BLE001 - turn optional dependency into a clear opt-in error
        raise RuntimeError(
            "MINISGL_DSV4_SM80_VLLM_FP8_MARLIN_PROJECTION=1 requires the vLLM "
            "runtime package with FP8 Marlin custom ops. Run mini from "
            "/workspace/venvs/vllm-dsv4/bin/python or install an equivalent vLLM "
            f"build. Import failed with {type(exc).__name__}: {exc}"
        ) from exc
    return (
        process_fp8_weight_block_strategy,
        prepare_fp8_layer_for_marlin,
        apply_fp8_marlin_linear,
    )


def prepare_linear(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor | None,
    *,
    owner_label: str,
) -> VllmFP8MarlinLinearWeight:
    if weight_scale_inv is None:
        raise RuntimeError(f"{owner_label} vLLM FP8 Marlin requires weight_scale_inv.")
    if weight.ndim != 2 or weight_scale_inv.ndim != 2:
        raise RuntimeError(
            f"{owner_label} vLLM FP8 Marlin expects 2D weight/scale, got "
            f"weight={tuple(weight.shape)} scale={tuple(weight_scale_inv.shape)}."
        )
    if not weight.is_cuda or not weight_scale_inv.is_cuda:
        raise RuntimeError(f"{owner_label} vLLM FP8 Marlin requires CUDA weight and scale.")
    if weight_scale_inv.device != weight.device:
        raise RuntimeError(
            f"{owner_label} vLLM FP8 Marlin requires scale on the same device as weight, "
            f"got weight={weight.device} scale={weight_scale_inv.device}."
        )

    process_fp8_weight_block_strategy, prepare_fp8_layer_for_marlin, _ = _helpers()
    size_n, size_k = weight.shape
    original_weight_bytes = tensor_bytes(weight)
    original_scale_bytes = tensor_bytes(weight_scale_inv)

    layer = nn.Module().to(weight.device)
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.orig_dtype = torch.bfloat16
    layer.logical_widths = [size_n]
    layer.weight_block_size = [128, 128]
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_inv = nn.Parameter(weight_scale_inv, requires_grad=False)
    layer.input_scale = None

    processed_weight, processed_scale = process_fp8_weight_block_strategy(
        layer.weight,
        layer.weight_scale_inv,
    )
    layer.weight = nn.Parameter(processed_weight, requires_grad=False)
    layer.weight_scale_inv = nn.Parameter(processed_scale, requires_grad=False)
    prepare_fp8_layer_for_marlin(layer, size_k_first=False, input_dtype=None)

    return VllmFP8MarlinLinearWeight(
        weight=layer.weight,
        weight_scale=layer.weight_scale_inv,
        workspace=layer.workspace,
        size_n=size_n,
        size_k=size_k,
        source_signature=source_signature(weight, weight_scale_inv),
        original_weight_bytes=original_weight_bytes,
        original_scale_bytes=original_scale_bytes,
        prepared_weight_bytes=tensor_bytes(layer.weight),
        prepared_scale_bytes=tensor_bytes(layer.weight_scale_inv),
        workspace_bytes=tensor_bytes(layer.workspace),
    )


def apply_linear(x: torch.Tensor, prepared: VllmFP8MarlinLinearWeight) -> torch.Tensor:
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"vLLM FP8 Marlin W8A16 expects fp16/bf16 activations, got {x.dtype}.")
    if not x.is_cuda:
        raise RuntimeError("vLLM FP8 Marlin W8A16 requires CUDA activations.")
    _, _, apply_fp8_marlin_linear = _helpers()
    return apply_fp8_marlin_linear(
        input=x,
        weight=prepared.weight,
        weight_scale=prepared.weight_scale,
        workspace=prepared.workspace,
        size_n=prepared.size_n,
        size_k=prepared.size_k,
        input_dtype=None,
        bias=None,
    )

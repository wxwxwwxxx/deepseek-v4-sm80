#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.kernels.linear import init_int8_linear_kernel
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

try:
    from minisgl.kernel.triton import deepseek_v4 as mini_triton_dsv4

    MINI_TRITON_ACT_QUANT_AVAILABLE = True
    MINI_TRITON_ACT_QUANT_ERROR = None
except Exception as exc:  # noqa: BLE001 - benchmark records fallback path
    mini_triton_dsv4 = None
    MINI_TRITON_ACT_QUANT_AVAILABLE = False
    MINI_TRITON_ACT_QUANT_ERROR = f"{type(exc).__name__}: {exc}"


TimedFn = Callable[[], torch.Tensor]

FP8_MAX = 448.0
INT8_MAX = 127.0
TP_SIZE = 8


@dataclass(frozen=True)
class DenseOwner:
    case: str
    owner: str
    weight_keys: tuple[str, ...]
    scale_keys: tuple[str, ...]
    shard_dim: int | None
    input_kind: str
    vllm_analogue: str
    source_boundary: str


@dataclass
class PreparedBackend:
    backend: str
    lane: str
    layer: nn.Module | None
    apply_fn: Callable[[torch.Tensor], torch.Tensor]
    prep_ms: float
    original_weight_bytes: int
    original_scale_bytes: int
    prepared_weight_bytes: int
    prepared_scale_bytes: int
    workspace_bytes: int
    source_conversion_ms: float
    activation_policy: str
    quant_dequant_included: str
    real_dsv4_candidate: bool
    notes: str

    @property
    def persistent_bytes(self) -> int:
        return int(self.prepared_weight_bytes + self.prepared_scale_bytes + self.workspace_bytes)


DENSE_OWNERS: tuple[DenseOwner, ...] = (
    DenseOwner(
        case="attn_qproj_fused_wqa_wkv",
        owner="attention WQA/WKV/compress",
        weight_keys=("layers.{layer}.attn.wq_a.weight", "layers.{layer}.attn.wkv.weight"),
        scale_keys=("layers.{layer}.attn.wq_a.scale", "layers.{layer}.attn.wkv.scale"),
        shard_dim=None,
        input_kind="hidden",
        vllm_analogue="DeepseekV4Attention.fused_wqa_wkv MergedColumnParallelLinear",
        source_boundary="DSV4Attention.forward q_proj fused WQA/WKV",
    ),
    DenseOwner(
        case="attn_q_wqb",
        owner="attention q_wqb",
        weight_keys=("layers.{layer}.attn.wq_b.weight",),
        scale_keys=("layers.{layer}.attn.wq_b.scale",),
        shard_dim=0,
        input_kind="q_lora",
        vllm_analogue="DeepseekV4Attention.wq_b ColumnParallelLinear",
        source_boundary="DSV4Attention.forward q_wqb",
    ),
    DenseOwner(
        case="attn_wo_b_local",
        owner="attention wo_b local",
        weight_keys=("layers.{layer}.attn.wo_b.weight",),
        scale_keys=("layers.{layer}.attn.wo_b.scale",),
        shard_dim=1,
        input_kind="wo_b_local",
        vllm_analogue="DeepseekV4Attention.wo_b RowParallelLinear local GEMM",
        source_boundary="DSV4Attention.forward wo_b local projection before all-reduce",
    ),
    DenseOwner(
        case="shared_experts_gate_up",
        owner="shared experts gate/up",
        weight_keys=(
            "layers.{layer}.ffn.shared_experts.w1.weight",
            "layers.{layer}.ffn.shared_experts.w3.weight",
        ),
        scale_keys=(
            "layers.{layer}.ffn.shared_experts.w1.scale",
            "layers.{layer}.ffn.shared_experts.w3.scale",
        ),
        shard_dim=0,
        input_kind="hidden",
        vllm_analogue="shared experts gate/up quantized ColumnParallelLinear",
        source_boundary="DSV4SharedExperts.forward gate_up_proj",
    ),
    DenseOwner(
        case="shared_experts_down",
        owner="shared experts down",
        weight_keys=("layers.{layer}.ffn.shared_experts.w2.weight",),
        scale_keys=("layers.{layer}.ffn.shared_experts.w2.scale",),
        shard_dim=1,
        input_kind="shared_down",
        vllm_analogue="shared experts down quantized RowParallelLinear local GEMM",
        source_boundary="DSV4SharedExperts.forward down_proj before all-reduce",
    ),
)


def load_index(model_path: Path) -> dict[str, str]:
    with (model_path / "model.safetensors.index.json").open() as handle:
        return json.load(handle)["weight_map"]


def load_tensor(
    model_path: Path,
    index: dict[str, str],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    shard = index.get(key)
    if shard is None:
        raise KeyError(f"tensor not found in safetensors index: {key}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(device=device)


def shard_tensor(tensor: torch.Tensor, *, dim: int | None, tp_size: int = TP_SIZE) -> torch.Tensor:
    if dim is None:
        return tensor.contiguous()
    return tensor.chunk(tp_size, dim=dim)[0].contiguous()


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def sync() -> None:
    torch.cuda.synchronize()


def timed_ms(fn: TimedFn, *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        sync()
        samples.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def wall_cuda_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - start) * 1000.0


def mini_fp8_activation_round(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    if x.numel() == 0 or x.shape[-1] % block_size != 0:
        return x
    if MINI_TRITON_ACT_QUANT_AVAILABLE and mini_triton_dsv4 is not None:
        try:
            y = mini_triton_dsv4.fp8_activation_quantize(x, block_size=block_size)
            if y is not None:
                return y
        except Exception:  # noqa: BLE001 - fall through to reference path
            pass
    dtype = x.dtype
    flat = x.contiguous().view(-1, x.shape[-1]).float()
    groups = flat.view(flat.shape[0], flat.shape[1] // block_size, block_size)
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) / FP8_MAX
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    y = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
    return y.reshape_as(flat).reshape_as(x).to(dtype)


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape
    expanded = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expanded = expanded[:out_features, :in_features]
    return (weight.float() * expanded).to(torch.bfloat16).contiguous()


def quantize_fp8_per_channel(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight_bf16.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / FP8_MAX
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, scale.contiguous()


def quantize_int8_per_channel(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight_bf16.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / INT8_MAX
    q = torch.round(w / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8).contiguous()
    return q, scale.contiguous()


def make_layer(
    *,
    weight: torch.Tensor,
    scale_attr: str,
    scale: torch.Tensor,
    n: int,
    k: int,
    weight_block_size: list[int] | None,
) -> nn.Module:
    layer = nn.Module().to(weight.device)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.orig_dtype = torch.bfloat16
    layer.logical_widths = [n]
    layer.weight_block_size = weight_block_size
    layer.weight = nn.Parameter(weight, requires_grad=False)
    setattr(layer, scale_attr, nn.Parameter(scale, requires_grad=False))
    layer.input_scale = None
    return layer


def prepare_vllm_marlin_block(weight: torch.Tensor, scale: torch.Tensor) -> PreparedBackend:
    n, k = weight.shape
    original_weight_bytes = tensor_bytes(weight)
    original_scale_bytes = tensor_bytes(scale)

    def prepare() -> nn.Module:
        layer = make_layer(
            weight=weight,
            scale_attr="weight_scale_inv",
            scale=scale,
            n=n,
            k=k,
            weight_block_size=[128, 128],
        )
        processed_weight, processed_scale = process_fp8_weight_block_strategy(
            layer.weight,
            layer.weight_scale_inv,
        )
        layer.weight = nn.Parameter(processed_weight, requires_grad=False)
        layer.weight_scale_inv = nn.Parameter(processed_scale, requires_grad=False)
        prepare_fp8_layer_for_marlin(layer, size_k_first=False, input_dtype=None)
        return layer

    layer, prep_ms = wall_cuda_ms(prepare)

    def apply(x: torch.Tensor, layer: nn.Module = layer) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale_inv,
            workspace=layer.workspace,
            size_n=n,
            size_k=k,
            input_dtype=None,
            bias=None,
        )

    return PreparedBackend(
        backend="vllm_fp8_marlin_w8a16_block",
        lane="Lane A",
        layer=layer,
        apply_fn=apply,
        prep_ms=prep_ms,
        original_weight_bytes=original_weight_bytes,
        original_scale_bytes=original_scale_bytes,
        prepared_weight_bytes=tensor_bytes(layer.weight),
        prepared_scale_bytes=tensor_bytes(layer.weight_scale_inv),
        workspace_bytes=tensor_bytes(layer.workspace),
        source_conversion_ms=0.0,
        activation_policy="BF16 activations; no activation quantization",
        quant_dequant_included=(
            "vLLM block-scale processing, Marlin repack, scale expansion/permutation, "
            "and exponent-bias fusion are one-time prep; replay path is weight-only Marlin."
        ),
        real_dsv4_candidate=True,
        notes="Uses DeepSeek V4 block FP8 weight_scale_inv directly.",
    )


def prepare_fbgemm_derived_marlin(weight: torch.Tensor, scale: torch.Tensor) -> PreparedBackend:
    n, k = weight.shape
    original_weight_bytes = tensor_bytes(weight)
    original_scale_bytes = tensor_bytes(scale)
    conversion_ms_holder = {"value": 0.0}

    def prepare() -> nn.Module:
        def convert() -> tuple[torch.Tensor, torch.Tensor]:
            cached = dequant_fp8_block(weight, scale)
            return quantize_fp8_per_channel(cached)

        (qweight, channel_scale), conversion_ms = wall_cuda_ms(convert)
        conversion_ms_holder["value"] = conversion_ms
        layer = make_layer(
            weight=qweight.t().contiguous(),
            scale_attr="weight_scale",
            scale=channel_scale,
            n=n,
            k=k,
            weight_block_size=None,
        )
        prepare_fp8_layer_for_marlin(layer, size_k_first=True, input_dtype=None)
        return layer

    layer, prep_ms = wall_cuda_ms(prepare)

    def apply(x: torch.Tensor, layer: nn.Module = layer) -> torch.Tensor:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=n,
            size_k=k,
            input_dtype=None,
            bias=None,
        )

    return PreparedBackend(
        backend="vllm_fbgemm_fp8_marlin_derived_channel",
        lane="Lane B",
        layer=layer,
        apply_fn=apply,
        prep_ms=prep_ms,
        original_weight_bytes=original_weight_bytes,
        original_scale_bytes=original_scale_bytes,
        prepared_weight_bytes=tensor_bytes(layer.weight),
        prepared_scale_bytes=tensor_bytes(layer.weight_scale),
        workspace_bytes=tensor_bytes(layer.workspace),
        source_conversion_ms=conversion_ms_holder["value"],
        activation_policy="BF16 activations; FBGEMM Marlin path deletes input_scale_ub",
        quant_dequant_included=(
            "Load-time diagnostic converts DSV4 block FP8 to BF16, requantizes to "
            "FBGEMM per-channel FP8, then uses vLLM Marlin repack."
        ),
        real_dsv4_candidate=False,
        notes="Runs on A100 but is not the native DeepSeek V4 block-scale checkpoint contract.",
    )


def prepare_int8_w8a8(weight: torch.Tensor, scale: torch.Tensor) -> PreparedBackend:
    n, k = weight.shape
    original_weight_bytes = tensor_bytes(weight)
    original_scale_bytes = tensor_bytes(scale)
    conversion_ms_holder = {"value": 0.0}

    def prepare() -> tuple[nn.Module, Any]:
        def convert() -> tuple[torch.Tensor, torch.Tensor]:
            cached = dequant_fp8_block(weight, scale)
            return quantize_int8_per_channel(cached)

        (qweight, channel_scale), conversion_ms = wall_cuda_ms(convert)
        conversion_ms_holder["value"] = conversion_ms
        layer = make_layer(
            weight=qweight,
            scale_attr="weight_scale",
            scale=channel_scale,
            n=n,
            k=k,
            weight_block_size=None,
        )
        layer.input_zero_point = None
        layer.azp_adj = None
        kernel = init_int8_linear_kernel(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
            module_name="target073_int8_w8a8_projection_probe",
        )
        kernel.process_weights_after_loading(layer)
        return layer, kernel

    (layer, kernel), prep_ms = wall_cuda_ms(prepare)

    def apply(x: torch.Tensor, layer: nn.Module = layer, kernel: Any = kernel) -> torch.Tensor:
        return kernel.apply_weights(layer, x, bias=None)

    azp_adj = getattr(layer, "azp_adj", None)
    return PreparedBackend(
        backend="vllm_int8_w8a8_cutlass_dynamic",
        lane="Lane D",
        layer=layer,
        apply_fn=apply,
        prep_ms=prep_ms,
        original_weight_bytes=original_weight_bytes,
        original_scale_bytes=original_scale_bytes,
        prepared_weight_bytes=tensor_bytes(layer.weight),
        prepared_scale_bytes=tensor_bytes(layer.weight_scale) + tensor_bytes(azp_adj),
        workspace_bytes=0,
        source_conversion_ms=conversion_ms_holder["value"],
        activation_policy="dynamic INT8 activation quantization inside vLLM apply_weights",
        quant_dequant_included=(
            "Load-time BF16 dequant plus per-channel INT8 weight quantization; replay "
            "latency includes vLLM scaled_int8_quant and cutlass_scaled_mm."
        ),
        real_dsv4_candidate=False,
        notes="Exploratory generic W8A8 projection probe; vLLM DeepSeek V4 INT8 projection is not an existing promoted backend.",
    )


def error_stats(candidate: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    cand = candidate.float()
    base = baseline.float()
    diff = (cand - base).abs()
    denom = base.abs().clamp_min(1e-6)
    rel = diff / denom
    flat = diff.flatten()
    rel_flat = rel.flatten()
    cosine = F.cosine_similarity(cand.flatten().unsqueeze(0), base.flatten().unsqueeze(0))
    return {
        "max_abs_err": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_err": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs_err": float(torch.quantile(flat, 0.99).item()) if flat.numel() else 0.0,
        "max_rel_err": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel_err": float(rel.mean().item()) if rel.numel() else 0.0,
        "p99_rel_err": float(torch.quantile(rel_flat, 0.99).item()) if rel_flat.numel() else 0.0,
        "cosine": float(cosine.item()) if diff.numel() else 1.0,
    }


def load_dense_owner(
    model_path: Path,
    index: dict[str, str],
    layer: int,
    owner: DenseOwner,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = [
        shard_tensor(
            load_tensor(model_path, index, key.format(layer=layer), device),
            dim=owner.shard_dim,
        )
        for key in owner.weight_keys
    ]
    scales = [
        shard_tensor(
            load_tensor(model_path, index, key.format(layer=layer), device),
            dim=owner.shard_dim,
        )
        for key in owner.scale_keys
    ]
    return torch.cat(weights, dim=0).contiguous(), torch.cat(scales, dim=0).contiguous()


def make_input(m: int, k: int, *, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(m, k, device=device, dtype=torch.bfloat16, generator=gen)


def bench_dense_owner(
    *,
    model_path: Path,
    index: dict[str, str],
    layer: int,
    owner: DenseOwner,
    tokens: list[int],
    warmup: int,
    iters: int,
    device: torch.device,
    owner_index: int,
    include_fbgemm: bool,
    include_int8: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weight, scale = load_dense_owner(model_path, index, layer, owner, device)
    n, k = weight.shape
    cached_bf16, cached_prep_ms = wall_cuda_ms(lambda: dequant_fp8_block(weight, scale))
    prepared: list[PreparedBackend] = [prepare_vllm_marlin_block(weight, scale)]
    if include_fbgemm:
        prepared.append(prepare_fbgemm_derived_marlin(weight, scale))
    if include_int8:
        prepared.append(prepare_int8_w8a8(weight, scale))

    prep_rows: list[dict[str, Any]] = [
        {
            "case": owner.case,
            "owner": owner.owner,
            "backend": "promoted_cached_bf16",
            "lane": "baseline",
            "prep_ms": cached_prep_ms,
            "source_conversion_ms": cached_prep_ms,
            "original_weight_bytes": tensor_bytes(weight),
            "original_scale_bytes": tensor_bytes(scale),
            "prepared_weight_bytes": tensor_bytes(cached_bf16),
            "prepared_scale_bytes": 0,
            "workspace_bytes": 0,
            "persistent_bytes": tensor_bytes(cached_bf16),
            "real_dsv4_candidate": True,
            "notes": "Promoted mini path caches BF16 dequantized weights before graph capture.",
        }
    ]
    for backend in prepared:
        prep_rows.append(
            {
                "case": owner.case,
                "owner": owner.owner,
                "backend": backend.backend,
                "lane": backend.lane,
                "prep_ms": backend.prep_ms,
                "source_conversion_ms": backend.source_conversion_ms,
                "original_weight_bytes": backend.original_weight_bytes,
                "original_scale_bytes": backend.original_scale_bytes,
                "prepared_weight_bytes": backend.prepared_weight_bytes,
                "prepared_scale_bytes": backend.prepared_scale_bytes,
                "workspace_bytes": backend.workspace_bytes,
                "persistent_bytes": backend.persistent_bytes,
                "real_dsv4_candidate": backend.real_dsv4_candidate,
                "notes": backend.notes,
            }
        )

    rows: list[dict[str, Any]] = []
    for m in tokens:
        x = make_input(m, k, seed=1000 + owner_index * 100 + m, device=device)
        x_quant = mini_fp8_activation_round(x)
        act_quant_timing = timed_ms(lambda x=x: mini_fp8_activation_round(x), warmup=warmup, iters=iters)
        baseline_gemm_fn = lambda xq=x_quant, w=cached_bf16: F.linear(xq, w)
        baseline_total_fn = lambda x=x, w=cached_bf16: F.linear(mini_fp8_activation_round(x), w)
        exact_bf16_fn = lambda x=x, w=cached_bf16: F.linear(x, w)
        promoted = baseline_total_fn()
        exact_bf16 = exact_bf16_fn()
        baseline_gemm_timing = timed_ms(baseline_gemm_fn, warmup=warmup, iters=iters)
        baseline_total_timing = timed_ms(baseline_total_fn, warmup=warmup, iters=iters)
        rows.append(
            {
                "case": owner.case,
                "owner": owner.owner,
                "tokens_m": m,
                "n": n,
                "k": k,
                "backend": "promoted_cached_bf16_total",
                "lane": "baseline",
                "variant_description": "Promoted mini path: activation FP8 rounding plus cached BF16 F.linear.",
                "source_boundary": owner.source_boundary,
                "vllm_analogue": owner.vllm_analogue,
                "activation_policy": "mini FP8 activation rounding before cached BF16 GEMM",
                "quant_dequant_included": "replay activation rounding included; weight dequant is one-time BF16 cache prep",
                "mean_ms": baseline_total_timing["mean_ms"],
                "median_ms": baseline_total_timing["median_ms"],
                "min_ms": baseline_total_timing["min_ms"],
                "max_ms": baseline_total_timing["max_ms"],
                "baseline_total_ms": baseline_total_timing["mean_ms"],
                "baseline_gemm_only_ms": baseline_gemm_timing["mean_ms"],
                "activation_quant_ms": act_quant_timing["mean_ms"],
                "speedup_vs_promoted_cached_bf16": 0.0,
                "max_abs_err": 0.0,
                "mean_abs_err": 0.0,
                "p99_abs_err": 0.0,
                "cosine": 1.0,
                "max_abs_err_vs_exact_bf16": error_stats(promoted, exact_bf16)["max_abs_err"],
                "mean_abs_err_vs_exact_bf16": error_stats(promoted, exact_bf16)["mean_abs_err"],
                "repack_required_per_decode": False,
                "per_decode_workspace_allocation": False,
                "real_dsv4_candidate": True,
            }
        )
        for backend in prepared:
            try:
                candidate = backend.apply_fn(x)
                sync()
                timing = timed_ms(lambda b=backend, x=x: b.apply_fn(x), warmup=warmup, iters=iters)
                err = error_stats(candidate, promoted)
                exact_err = error_stats(candidate, exact_bf16)
                rows.append(
                    {
                        "case": owner.case,
                        "owner": owner.owner,
                        "tokens_m": m,
                        "n": n,
                        "k": k,
                        "backend": backend.backend,
                        "lane": backend.lane,
                        "variant_description": backend.notes,
                        "source_boundary": owner.source_boundary,
                        "vllm_analogue": owner.vllm_analogue,
                        "activation_policy": backend.activation_policy,
                        "quant_dequant_included": backend.quant_dequant_included,
                        "mean_ms": timing["mean_ms"],
                        "median_ms": timing["median_ms"],
                        "min_ms": timing["min_ms"],
                        "max_ms": timing["max_ms"],
                        "baseline_total_ms": baseline_total_timing["mean_ms"],
                        "baseline_gemm_only_ms": baseline_gemm_timing["mean_ms"],
                        "activation_quant_ms": 0.0
                        if "INT8" not in backend.activation_policy
                        else None,
                        "speedup_vs_promoted_cached_bf16": (
                            (baseline_total_timing["mean_ms"] - timing["mean_ms"])
                            / baseline_total_timing["mean_ms"]
                        ),
                        **err,
                        "max_abs_err_vs_exact_bf16": exact_err["max_abs_err"],
                        "mean_abs_err_vs_exact_bf16": exact_err["mean_abs_err"],
                        "repack_required_per_decode": False,
                        "per_decode_workspace_allocation": False,
                        "real_dsv4_candidate": backend.real_dsv4_candidate,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "case": owner.case,
                        "owner": owner.owner,
                        "tokens_m": m,
                        "n": n,
                        "k": k,
                        "backend": backend.backend,
                        "lane": backend.lane,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=6),
                    }
                )
    return rows, prep_rows


def load_wo_a(
    model_path: Path,
    index: dict[str, str],
    layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wo_a.weight", device),
        dim=0,
    )
    scale = shard_tensor(
        load_tensor(model_path, index, f"layers.{layer}.attn.wo_a.scale", device),
        dim=0,
    )
    return weight.contiguous(), scale.contiguous()


def prepare_grouped_backends(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    include_fbgemm: bool,
    include_int8: bool,
) -> list[PreparedBackend]:
    backends: list[PreparedBackend] = []
    weight_groups = weight.chunk(2, dim=0)
    scale_groups = scale.chunk(2, dim=0)

    def make_grouped(name: str, lane: str, builders: list[PreparedBackend], notes: str) -> PreparedBackend:
        prep_ms = sum(item.prep_ms for item in builders)
        conversion_ms = sum(item.source_conversion_ms for item in builders)

        def apply(x: torch.Tensor, builders: list[PreparedBackend] = builders) -> torch.Tensor:
            outs = [backend.apply_fn(x[:, idx, :]) for idx, backend in enumerate(builders)]
            return torch.stack(outs, dim=1).reshape(x.shape[0], -1)

        return PreparedBackend(
            backend=name,
            lane=lane,
            layer=None,
            apply_fn=apply,
            prep_ms=prep_ms,
            original_weight_bytes=sum(item.original_weight_bytes for item in builders),
            original_scale_bytes=sum(item.original_scale_bytes for item in builders),
            prepared_weight_bytes=sum(item.prepared_weight_bytes for item in builders),
            prepared_scale_bytes=sum(item.prepared_scale_bytes for item in builders),
            workspace_bytes=sum(item.workspace_bytes for item in builders),
            source_conversion_ms=conversion_ms,
            activation_policy=builders[0].activation_policy,
            quant_dequant_included=builders[0].quant_dequant_included,
            real_dsv4_candidate=builders[0].real_dsv4_candidate,
            notes=notes,
        )

    marlin_groups = [
        prepare_vllm_marlin_block(w.contiguous(), s.contiguous())
        for w, s in zip(weight_groups, scale_groups, strict=True)
    ]
    backends.append(
        make_grouped(
            "vllm_fp8_marlin_w8a16_block_grouped_two_launch",
            "Lane A",
            marlin_groups,
            "Grouped wo_a diagnostic using one vLLM Marlin linear per local group.",
        )
    )
    if include_fbgemm:
        fbgemm_groups = [
            prepare_fbgemm_derived_marlin(w.contiguous(), s.contiguous())
            for w, s in zip(weight_groups, scale_groups, strict=True)
        ]
        backends.append(
            make_grouped(
                "vllm_fbgemm_fp8_marlin_derived_channel_grouped_two_launch",
                "Lane B",
                fbgemm_groups,
                "Grouped wo_a diagnostic after per-channel FBGEMM-style conversion.",
            )
        )
    if include_int8:
        int8_groups = [
            prepare_int8_w8a8(w.contiguous(), s.contiguous())
            for w, s in zip(weight_groups, scale_groups, strict=True)
        ]
        backends.append(
            make_grouped(
                "vllm_int8_w8a8_cutlass_dynamic_grouped_two_launch",
                "Lane D",
                int8_groups,
                "Grouped wo_a diagnostic using one dynamic INT8 W8A8 linear per local group.",
            )
        )
    return backends


def bench_wo_a(
    *,
    model_path: Path,
    index: dict[str, str],
    layer: int,
    tokens: list[int],
    warmup: int,
    iters: int,
    device: torch.device,
    include_fbgemm: bool,
    include_int8: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weight, scale = load_wo_a(model_path, index, layer, device)
    cached_bf16, cached_prep_ms = wall_cuda_ms(lambda: dequant_fp8_block(weight, scale))
    groups = 2
    rank = cached_bf16.shape[0] // groups
    k = cached_bf16.shape[1]
    cached_bmm = cached_bf16.view(groups, rank, k).transpose(1, 2).contiguous()
    prepared = prepare_grouped_backends(weight, scale, include_fbgemm=include_fbgemm, include_int8=include_int8)

    prep_rows: list[dict[str, Any]] = [
        {
            "case": "attn_wo_a_grouped_projection",
            "owner": "attention wo_a grouped",
            "backend": "promoted_cached_bf16_grouped_bmm",
            "lane": "baseline",
            "prep_ms": cached_prep_ms,
            "source_conversion_ms": cached_prep_ms,
            "original_weight_bytes": tensor_bytes(weight),
            "original_scale_bytes": tensor_bytes(scale),
            "prepared_weight_bytes": tensor_bytes(cached_bmm),
            "prepared_scale_bytes": 0,
            "workspace_bytes": 0,
            "persistent_bytes": tensor_bytes(cached_bmm),
            "real_dsv4_candidate": True,
            "notes": "Promoted mini wo_a path caches BF16 grouped-BMM weights.",
        }
    ]
    for backend in prepared:
        prep_rows.append(
            {
                "case": "attn_wo_a_grouped_projection",
                "owner": "attention wo_a grouped",
                "backend": backend.backend,
                "lane": backend.lane,
                "prep_ms": backend.prep_ms,
                "source_conversion_ms": backend.source_conversion_ms,
                "original_weight_bytes": backend.original_weight_bytes,
                "original_scale_bytes": backend.original_scale_bytes,
                "prepared_weight_bytes": backend.prepared_weight_bytes,
                "prepared_scale_bytes": backend.prepared_scale_bytes,
                "workspace_bytes": backend.workspace_bytes,
                "persistent_bytes": backend.persistent_bytes,
                "real_dsv4_candidate": backend.real_dsv4_candidate,
                "notes": backend.notes,
            }
        )

    rows: list[dict[str, Any]] = []
    for m in tokens:
        x = make_input(m * groups, k, seed=1700 + m, device=device).reshape(m, groups, k)
        baseline_input = x.transpose(0, 1).contiguous()
        baseline_fn = (
            lambda x=baseline_input, w=cached_bmm, t=m, g=groups: torch.bmm(x, w)
            .transpose(0, 1)
            .reshape(t, g * w.shape[2])
        )
        promoted = baseline_fn()
        baseline_timing = timed_ms(baseline_fn, warmup=warmup, iters=iters)
        rows.append(
            {
                "case": "attn_wo_a_grouped_projection",
                "owner": "attention wo_a grouped",
                "tokens_m": m,
                "n": groups * rank,
                "k": k,
                "backend": "promoted_cached_bf16_grouped_bmm",
                "lane": "baseline",
                "variant_description": "Promoted mini wo_a path: cached BF16 grouped torch.bmm.",
                "source_boundary": "DSV4Attention wo_a grouped projection",
                "vllm_analogue": "DeepSeek V4 wo_a low-precision/einsum boundary",
                "activation_policy": "BF16 activations; no activation quantization",
                "quant_dequant_included": "weight dequant is one-time grouped BF16 cache prep",
                "mean_ms": baseline_timing["mean_ms"],
                "median_ms": baseline_timing["median_ms"],
                "min_ms": baseline_timing["min_ms"],
                "max_ms": baseline_timing["max_ms"],
                "baseline_total_ms": baseline_timing["mean_ms"],
                "baseline_gemm_only_ms": baseline_timing["mean_ms"],
                "activation_quant_ms": 0.0,
                "speedup_vs_promoted_cached_bf16": 0.0,
                "max_abs_err": 0.0,
                "mean_abs_err": 0.0,
                "p99_abs_err": 0.0,
                "cosine": 1.0,
                "max_abs_err_vs_exact_bf16": 0.0,
                "mean_abs_err_vs_exact_bf16": 0.0,
                "repack_required_per_decode": False,
                "per_decode_workspace_allocation": False,
                "real_dsv4_candidate": True,
            }
        )
        for backend in prepared:
            try:
                candidate = backend.apply_fn(x)
                sync()
                timing = timed_ms(lambda b=backend, x=x: b.apply_fn(x), warmup=warmup, iters=iters)
                err = error_stats(candidate, promoted)
                rows.append(
                    {
                        "case": "attn_wo_a_grouped_projection",
                        "owner": "attention wo_a grouped",
                        "tokens_m": m,
                        "n": groups * rank,
                        "k": k,
                        "backend": backend.backend,
                        "lane": backend.lane,
                        "variant_description": backend.notes,
                        "source_boundary": "DSV4Attention wo_a grouped projection",
                        "vllm_analogue": "DeepSeek V4 wo_a low-precision/einsum boundary",
                        "activation_policy": backend.activation_policy,
                        "quant_dequant_included": backend.quant_dequant_included,
                        "mean_ms": timing["mean_ms"],
                        "median_ms": timing["median_ms"],
                        "min_ms": timing["min_ms"],
                        "max_ms": timing["max_ms"],
                        "baseline_total_ms": baseline_timing["mean_ms"],
                        "baseline_gemm_only_ms": baseline_timing["mean_ms"],
                        "activation_quant_ms": 0.0
                        if "INT8" not in backend.activation_policy
                        else None,
                        "speedup_vs_promoted_cached_bf16": (
                            (baseline_timing["mean_ms"] - timing["mean_ms"])
                            / baseline_timing["mean_ms"]
                        ),
                        **err,
                        "max_abs_err_vs_exact_bf16": err["max_abs_err"],
                        "mean_abs_err_vs_exact_bf16": err["mean_abs_err"],
                        "repack_required_per_decode": False,
                        "per_decode_workspace_allocation": False,
                        "real_dsv4_candidate": backend.real_dsv4_candidate,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "case": "attn_wo_a_grouped_projection",
                        "owner": "attention wo_a grouped",
                        "tokens_m": m,
                        "n": groups * rank,
                        "k": k,
                        "backend": backend.backend,
                        "lane": backend.lane,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=6),
                    }
                )
    return rows, prep_rows


def quality_ok(row: dict[str, Any]) -> bool:
    if "error" in row:
        return False
    return float(row.get("cosine", 0.0)) >= 0.995 and float(row.get("mean_abs_err", math.inf)) <= 0.05


def compute_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_backends = sorted(
        {
            row["backend"]
            for row in rows
            if row.get("lane") not in {"baseline", None} and "error" not in row
        }
    )
    backend_results: dict[str, Any] = {}
    for backend in candidate_backends:
        backend_rows = [row for row in rows if row.get("backend") == backend and "error" not in row]
        owners = sorted({row["owner"] for row in backend_rows})
        owners_all_m_pass = []
        owners_any_regress_gt10 = []
        for owner in owners:
            owner_rows = [row for row in backend_rows if row["owner"] == owner]
            speedups = [float(row["speedup_vs_promoted_cached_bf16"]) for row in owner_rows]
            if len(owner_rows) >= 4 and all(speedup >= 0.15 for speedup in speedups) and all(
                quality_ok(row) for row in owner_rows
            ):
                owners_all_m_pass.append(owner)
            if any(speedup < -0.10 for speedup in speedups):
                owners_any_regress_gt10.append(owner)
        backend_results[backend] = {
            "owners_all_m_speedup_ge_15pct_and_quality_ok": owners_all_m_pass,
            "owners_regressing_gt_10pct": owners_any_regress_gt10,
            "standalone_gate_pass": len(owners_all_m_pass) >= 2
            and not owners_any_regress_gt10,
            "covers_m_values": sorted({int(row["tokens_m"]) for row in backend_rows}),
            "covers_owner_count": len(owners),
        }
    return {
        "standalone_backend_gate": {
            "criterion": (
                "candidate covers M=1,4,8,16 and improves >=15% over promoted "
                "cached BF16 on at least two representative owners, with no >10% owner regression"
            ),
            "any_backend_pass": any(
                result["standalone_gate_pass"] for result in backend_results.values()
            ),
            "backend_results": backend_results,
        }
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Focused Quantized-Linear Backend Microbench",
        "",
        f"- model: `{data['model_path']}`",
        f"- layer: `{data['layer']}`",
        f"- device: `{data['device']}` capability `{data['capability']}`",
        f"- torch: `{data['torch_version']}`",
        f"- warmup/iters: `{data['warmup']}` / `{data['iters']}`",
        f"- standalone gate pass: `{data['gates']['standalone_backend_gate']['any_backend_pass']}`",
        "",
        "## M=4 Latency",
        "",
        "| Owner | Backend | Mean ms | Baseline ms | Speedup | Max abs err | Mean abs err | Cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    m4_rows = [
        row for row in data["results"] if int(row.get("tokens_m", -1)) == 4 and "error" not in row
    ]
    for row in m4_rows:
        lines.append(
            "| `{}` | `{}` | `{:.6f}` | `{:.6f}` | `{:.2%}` | `{:.6g}` | `{:.6g}` | `{:.8f}` |".format(
                row["owner"],
                row["backend"],
                float(row["mean_ms"]),
                float(row["baseline_total_ms"]),
                float(row["speedup_vs_promoted_cached_bf16"]),
                float(row.get("max_abs_err", 0.0)),
                float(row.get("mean_abs_err", 0.0)),
                float(row.get("cosine", 1.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Preparation / Workspace",
            "",
            "| Owner | Backend | Prep ms | Conversion ms | Prepared weight bytes | Prepared scale bytes | Workspace bytes | Persistent bytes | Real DSV4 candidate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in data["prep"]:
        lines.append(
            "| `{}` | `{}` | `{:.3f}` | `{:.3f}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["owner"],
                row["backend"],
                float(row["prep_ms"]),
                float(row["source_conversion_ms"]),
                int(row["prepared_weight_bytes"]),
                int(row["prepared_scale_bytes"]),
                int(row["workspace_bytes"]),
                int(row["persistent_bytes"]),
                row["real_dsv4_candidate"],
            )
        )
    lines.extend(["", "## Gate", "", "```json", json.dumps(data["gates"], indent=2), "```"])
    return "\n".join(lines) + "\n"


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    index = load_index(args.model_path)

    results: list[dict[str, Any]] = []
    prep_rows: list[dict[str, Any]] = []
    for owner_index, owner in enumerate(DENSE_OWNERS):
        owner_rows, owner_prep = bench_dense_owner(
            model_path=args.model_path,
            index=index,
            layer=args.layer,
            owner=owner,
            tokens=args.tokens,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            owner_index=owner_index,
            include_fbgemm=not args.skip_fbgemm,
            include_int8=not args.skip_int8,
        )
        results.extend(owner_rows)
        prep_rows.extend(owner_prep)
        del owner_rows, owner_prep
        torch.cuda.empty_cache()

    wo_a_rows, wo_a_prep = bench_wo_a(
        model_path=args.model_path,
        index=index,
        layer=args.layer,
        tokens=args.tokens,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        include_fbgemm=not args.skip_fbgemm,
        include_int8=not args.skip_int8,
    )
    results.extend(wo_a_rows)
    prep_rows.extend(wo_a_prep)

    data = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(args.model_path),
        "layer": args.layer,
        "tokens": args.tokens,
        "tp_size_simulated": TP_SIZE,
        "rank_simulated": 0,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "baseline_contract": (
            "dsv4_sm80_a100_victory promoted cached BF16 projection path; "
            "dense owners include mini activation FP8 rounding plus cached BF16 GEMM."
        ),
        "activation_quant_backend": {
            "requested": "mini Triton fp8_activation_quantize",
            "used": "mini_triton" if MINI_TRITON_ACT_QUANT_AVAILABLE else "torch_reference_fallback",
            "import_error": MINI_TRITON_ACT_QUANT_ERROR,
        },
        "inactive_optins": [
            "MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST=1",
            "MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1",
            "MINISGL_DSV4_SM80_BF16_SMALL_GEMM_PRETRANSPOSE=1",
        ],
        "results": results,
        "prep": prep_rows,
        "gates": compute_gates(results),
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/models/DeepSeek-V4-Flash"))
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-fbgemm", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(data))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe vLLM MXFP4 MoE backend selection for DeepSeek V4 on the local device."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


VLLM_REPO = Path("/workspace/vllm-dsv4-docker")
if VLLM_REPO.exists():
    sys.path.insert(0, str(VLLM_REPO))

import torch

import vllm
from vllm import envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _backend_activation_key,
    _get_priority_backends,
    _get_priority_backends_for_gpt_oss,
    backend_to_kernel_cls,
    make_mxfp4_moe_quant_config,
    mxfp4_round_up_hidden_size_and_intermediate_size,
    select_gpt_oss_mxfp4_moe_backend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static
from vllm.platforms import current_platform


ENV_KEYS = [
    "VLLM_MXFP4_USE_MARLIN",
    "VLLM_MARLIN_INPUT_DTYPE",
    "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
    "VLLM_USE_FLASHINFER_MOE_FP4",
    "VLLM_HAS_FLASHINFER_CUBIN",
    "VLLM_BATCH_INVARIANT",
]

EXPLICIT_BACKENDS = [
    "marlin",
    "triton",
    "triton_unfused",
    "deep_gemm",
    "flashinfer_trtllm",
    "flashinfer_trtllm_afp8",
    "flashinfer_cutlass",
    "flashinfer_cutlass_afp8",
    "aiter",
    "xpu",
    "emulation",
]


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mxfp4MoeBackend):
        return value.value
    if isinstance(value, mk.FusedMoEActivationFormat):
        return value.name
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return str(value)


def _capability() -> dict[str, Any]:
    cap = current_platform.get_device_capability()
    major = getattr(cap, "major", None)
    minor = getattr(cap, "minor", None)
    if major is None and cap is not None:
        try:
            major, minor = cap  # type: ignore[misc]
        except Exception:
            pass
    return {
        "raw": str(cap),
        "major": major,
        "minor": minor,
        "tuple": [major, minor] if major is not None else None,
    }


def _device_name() -> str | None:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception as exc:
            return f"<cuda name error: {type(exc).__name__}: {exc}>"
    return None


def _env_snapshot() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in ENV_KEYS:
        try:
            parsed = getattr(envs, key)
        except Exception as exc:
            parsed = f"<envs error: {type(exc).__name__}: {exc}>"
        try:
            is_set = envs.is_set(key)
        except Exception as exc:
            is_set = f"<is_set error: {type(exc).__name__}: {exc}>"
        out[key] = {
            "raw": os.environ.get(key),
            "is_set": is_set,
            "parsed": parsed,
        }
    return out


def make_config(moe_backend: str = "auto") -> FusedMoEConfig:
    # TP8 / non-EP DeepSeek V4 shape: H=4096, global intermediate=2048,
    # local intermediate per TP rank=256, top-k=6, 256 routed experts.
    parallel = FusedMoEParallelConfig(
        tp_size=8,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=256,
        experts_per_token=6,
        hidden_dim=4096,
        intermediate_size_per_partition=256,
        num_local_experts=256,
        num_logical_experts=256,
        activation=MoEActivation.SILU,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        router_logits_dtype=torch.float32,
        moe_backend=moe_backend,
        max_num_tokens=4096,
        has_bias=False,
        is_act_and_mul=True,
        is_lora_enabled=False,
        disable_inplace=False,
    )


def _kernel_class_name(cls: type[mk.FusedMoEExperts] | None) -> str | None:
    return None if cls is None else f"{cls.__module__}.{cls.__name__}"


def _selected_quant_config(backend: Mxfp4MoeBackend) -> dict[str, Any] | None:
    try:
        dummy_w1_scale = torch.empty(0)
        dummy_w2_scale = torch.empty(0)
        quant_config = make_mxfp4_moe_quant_config(
            backend,
            w1_scale=dummy_w1_scale,
            w2_scale=dummy_w2_scale,
            gemm1_alpha=None,
            gemm1_beta=None,
            swiglu_limit=None,
        )
        if quant_config is None:
            return None
        return {
            "quant_dtype": _jsonify(quant_config.quant_dtype),
            "weight_quant_dtype": _jsonify(quant_config.weight_quant_dtype),
            "is_quantized_activation": bool(quant_config.is_quantized),
            "per_act_token_quant": bool(quant_config.per_act_token_quant),
            "block_shape": _jsonify(quant_config.block_shape),
            "use_mxfp4_w4a16": bool(quant_config.use_mxfp4_w4a16),
            "use_mxfp4_w4a8": bool(quant_config.use_mxfp4_w4a8),
            "use_mxfp4_w4a4": bool(quant_config.use_mxfp4_w4a4),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_selector(name: str, selector: Any, config: FusedMoEConfig) -> dict[str, Any]:
    try:
        backend, cls = selector(config)
        activation_format = (
            mk.FusedMoEActivationFormat.BatchedExperts
            if config.moe_parallel_config.use_batched_activation_format
            else mk.FusedMoEActivationFormat.Standard
        )
        return {
            "name": name,
            "status": "selected",
            "backend": backend.value,
            "experts_class": _kernel_class_name(cls),
            "activation_format": activation_format.name,
            "weight_key": _jsonify(kMxfp4Static),
            "activation_key": _jsonify(_backend_activation_key(backend)),
            "rounded_shape": {
                "hidden_dim": mxfp4_round_up_hidden_size_and_intermediate_size(
                    backend, config.hidden_dim, config.intermediate_size_per_partition
                )[0],
                "intermediate_size_per_partition": mxfp4_round_up_hidden_size_and_intermediate_size(
                    backend, config.hidden_dim, config.intermediate_size_per_partition
                )[1],
            },
            "quant_config": _selected_quant_config(backend),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _support_matrix(config: FusedMoEConfig) -> list[dict[str, Any]]:
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )
    rows: list[dict[str, Any]] = []
    for backend in Mxfp4MoeBackend:
        if backend == Mxfp4MoeBackend.NONE:
            continue

        activation_key = _backend_activation_key(backend)
        row: dict[str, Any] = {
            "backend": backend.value,
            "weight_key": _jsonify(kMxfp4Static),
            "activation_key": _jsonify(activation_key),
            "selector_activation_format": activation_format.name,
            "classes": [],
        }

        try:
            classes = backend_to_kernel_cls(backend)
        except Exception as exc:
            row["classes_import_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        for cls in classes:
            entry: dict[str, Any] = {
                "class": _kernel_class_name(cls),
            }
            try:
                native_format = cls.activation_format()
                entry["native_activation_format"] = native_format.name
            except Exception as exc:
                native_format = None
                entry["native_activation_format_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            try:
                supported, reason = cls.is_supported_config(
                    cls,
                    config,
                    kMxfp4Static,
                    activation_key,
                    activation_format,
                )
                entry["supported_for_selector_format"] = bool(supported)
                entry["selector_reject_reason"] = reason
            except Exception as exc:
                entry["supported_for_selector_format"] = False
                entry["selector_reject_reason"] = f"{type(exc).__name__}: {exc}"

            if native_format is not None:
                try:
                    supported, reason = cls.is_supported_config(
                        cls,
                        config,
                        kMxfp4Static,
                        activation_key,
                        native_format,
                    )
                    entry["supported_for_native_format"] = bool(supported)
                    entry["native_reject_reason"] = reason
                except Exception as exc:
                    entry["supported_for_native_format"] = False
                    entry["native_reject_reason"] = f"{type(exc).__name__}: {exc}"

            row["classes"].append(entry)
        rows.append(row)
    return rows


def _explicit_overrides() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for backend in EXPLICIT_BACKENDS:
        results[backend] = _run_selector(
            f"select_mxfp4_moe_backend(moe_backend={backend})",
            select_mxfp4_moe_backend,
            make_config(moe_backend=backend),
        )
    return results


def _summary(raw: dict[str, Any]) -> dict[str, Any]:
    selector = raw["selectors"]["deepseek_v4_actual_mxfp4"]
    support = raw["support_matrix_for_deepseek_v4_standard_tp8"]

    supported = []
    rejected = []
    for row in support:
        class_entries = row.get("classes", [])
        if any(entry.get("supported_for_selector_format") for entry in class_entries):
            supported.append(row["backend"])
        else:
            reasons = [
                entry.get("selector_reject_reason")
                for entry in class_entries
                if entry.get("selector_reject_reason")
            ]
            rejected.append(
                {
                    "backend": row["backend"],
                    "reason": reasons[0] if reasons else row.get("classes_import_error"),
                }
            )

    return {
        "device": raw["environment"]["device"],
        "actual_deepseek_v4_selector": raw["deepseek_v4_path"],
        "selected_backend": selector.get("backend"),
        "selected_experts_class": selector.get("experts_class"),
        "selected_quant_config": selector.get("quant_config"),
        "activation_format": selector.get("activation_format"),
        "weight_key": selector.get("weight_key"),
        "activation_key": selector.get("activation_key"),
        "classification": {
            "MARLIN": "exact_candidate",
            "BATCHED_MARLIN": "exact_candidate_when_batched_activation_format_is_used; rejected_for_standard_TP8_selector_format",
            "DEEPGEMM_MXFP4": "precision_lane_and_sm100_only",
            "FLASHINFER_TRTLLM_MXFP4_MXFP8": "precision_lane_and_sm100_only",
            "FLASHINFER_TRTLLM_MXFP4_BF16": "exact_candidate_but_sm100_only",
            "FLASHINFER_CUTLASS_MXFP4_BF16": "exact_candidate_but_sm90_or_later_only",
            "FLASHINFER_CUTLASS_MXFP4_MXFP8": "precision_lane_and_sm100_only",
            "TRITON": "defer_or_reject_for_sm80; gpt_oss_triton_kernel_path_requires_sm90_to_sm100_window",
            "TRITON_UNFUSED": "defer_or_reject_for_sm80; gpt_oss_triton_kernel_path_requires_sm90_to_sm100_window",
            "AITER": "defer_or_reject_for_cuda_sm80; ROCm backend",
            "XPU": "defer_or_reject_for_cuda_sm80; XPU backend",
            "EMULATION": "defer_or_reject; not in DeepSeek_V4_actual_priority_and_not_a_performance_backend",
        },
        "supported_for_standard_tp8": supported,
        "rejected_for_standard_tp8": rejected,
        "explicit_overrides": raw["explicit_backend_overrides"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Write raw probe JSON.")
    parser.add_argument("--summary-output", type=Path, help="Write compact summary JSON.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    config = make_config()
    raw = {
        "environment": {
            "python": sys.executable,
            "vllm_version": getattr(vllm, "__version__", None),
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device": {
                "name": _device_name(),
                "current_platform_class": type(current_platform).__name__,
                "device_name": getattr(current_platform, "device_name", None),
                "is_cuda": current_platform.is_cuda(),
                "is_rocm": current_platform.is_rocm(),
                "is_xpu": current_platform.is_xpu(),
                "capability": _capability(),
            },
            "env": _env_snapshot(),
        },
        "deepseek_v4_path": {
            "quant_config_class": "DeepseekV4FP8Config",
            "moe_method": "Mxfp4MoEMethod",
            "selector": "select_mxfp4_moe_backend",
            "priority_backends": [backend.value for backend in _get_priority_backends()],
            "gpt_oss_reference_priority_backends": [
                backend.value for backend in _get_priority_backends_for_gpt_oss()
            ],
        },
        "moe_config": {
            "num_experts": config.num_experts,
            "experts_per_token": config.experts_per_token,
            "hidden_dim": config.hidden_dim,
            "intermediate_size_per_partition": config.intermediate_size_per_partition,
            "num_local_experts": config.num_local_experts,
            "num_logical_experts": config.num_logical_experts,
            "activation": config.activation.value,
            "routing_method": config.routing_method.name,
            "in_dtype": str(config.in_dtype),
            "router_logits_dtype": str(config.router_logits_dtype),
            "moe_backend": config.moe_backend,
            "parallel": {
                "tp_size": config.tp_size,
                "dp_size": config.dp_size,
                "ep_size": config.ep_size,
                "sp_size": config.sp_size,
                "use_ep": config.use_ep,
                "use_batched_activation_format": config.moe_parallel_config.use_batched_activation_format,
                "all2all_backend": config.moe_parallel_config.all2all_backend,
            },
        },
        "selectors": {
            "deepseek_v4_actual_mxfp4": _run_selector(
                "select_mxfp4_moe_backend(auto)", select_mxfp4_moe_backend, config
            ),
            "gpt_oss_reference_mxfp4": _run_selector(
                "select_gpt_oss_mxfp4_moe_backend(auto)",
                select_gpt_oss_mxfp4_moe_backend,
                config,
            ),
        },
        "support_matrix_for_deepseek_v4_standard_tp8": _support_matrix(config),
        "explicit_backend_overrides": _explicit_overrides(),
    }
    summary = _summary(raw)

    indent = 2 if args.pretty else None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(raw, indent=indent, sort_keys=True) + "\n")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=indent, sort_keys=True) + "\n"
        )

    print(json.dumps(summary if args.summary_output else raw, indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()

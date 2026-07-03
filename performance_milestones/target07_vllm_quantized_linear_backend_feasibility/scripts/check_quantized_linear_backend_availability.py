#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


def record_probe(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = fn()
        result.setdefault("ok", True)
        return {"name": name, **result}
    except Exception as exc:  # noqa: BLE001 - availability smoke must record blockers
        return {
            "name": name,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def run_python_probe(python: str, code: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [python, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mini-python",
        default="/usr/bin/python",
        help="Python interpreter used to document mini-env vLLM import availability.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recommended_runner": "/workspace/venvs/vllm-dsv4/bin/python",
        "current_python": sys.executable,
        "mini_python_probe": run_python_probe(
            args.mini_python,
            (
                "import json, sys\n"
                "try:\n"
                " import vllm\n"
                " print(json.dumps({'ok': True, 'version': getattr(vllm, '__version__', None), 'file': vllm.__file__}))\n"
                "except Exception as exc:\n"
                " print(json.dumps({'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)}))\n"
                " sys.exit(1)\n"
            ),
        ),
        "probes": [],
    }

    def torch_env() -> dict[str, Any]:
        import torch

        return {
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
        }

    report["probes"].append(record_probe("torch_cuda_environment", torch_env))

    def vllm_imports() -> dict[str, Any]:
        import vllm
        from vllm import _custom_ops as ops
        from vllm.platforms import current_platform

        op_names = [
            "gptq_marlin_repack",
            "marlin_gemm",
            "cutlass_scaled_mm",
            "cutlass_scaled_mm_azp",
            "scaled_fp8_quant",
            "scaled_int8_quant",
            "cutlass_scaled_mm_supports_fp8",
            "cutlass_scaled_mm_supports_block_fp8",
            "cutlass_group_gemm_supported",
        ]
        return {
            "vllm_version": getattr(vllm, "__version__", None),
            "vllm_file": vllm.__file__,
            "platform": type(current_platform).__name__,
            "device_capability": str(current_platform.get_device_capability()),
            "ops": {name: hasattr(ops, name) for name in op_names},
        }

    report["probes"].append(record_probe("vllm_imports_and_custom_ops", vllm_imports))

    def fp8_kernel_selection() -> dict[str, Any]:
        import torch
        from vllm import _custom_ops as ops
        import vllm.model_executor.kernels.linear as linear_kernels
        from vllm.model_executor.kernels.linear.scaled_mm import (
            FP8ScaledMMLinearLayerConfig,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            is_fp8_marlin_supported,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape,
            create_fp8_quant_key,
            kFp8DynamicTokenSym,
            kFp8StaticTokenSym,
        )

        activation_block = create_fp8_quant_key(
            static=False,
            group_shape=GroupShape(1, 128),
        )
        weight_block = create_fp8_quant_key(
            static=True,
            group_shape=GroupShape(128, 128),
        )
        block_config = FP8ScaledMMLinearLayerConfig(
            activation_quant_key=activation_block,
            weight_quant_key=weight_block,
            weight_shape=(4096, 1024),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        token_config = FP8ScaledMMLinearLayerConfig(
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticTokenSym,
            weight_shape=(4096, 1024),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        block_kernel_type = linear_kernels.choose_scaled_mm_linear_kernel(
            block_config,
            linear_kernels._POSSIBLE_FP8_BLOCK_KERNELS,  # noqa: SLF001
        )
        token_kernel_type = linear_kernels.choose_scaled_mm_linear_kernel(
            token_config,
            linear_kernels._POSSIBLE_FP8_KERNELS,  # noqa: SLF001
        )
        capability = 80
        return {
            "fp8_marlin_supported": bool(is_fp8_marlin_supported()),
            "cutlass_fp8_supported_cap80": bool(
                ops.cutlass_scaled_mm_supports_fp8(capability)
            ),
            "cutlass_block_fp8_supported_cap80": bool(
                ops.cutlass_scaled_mm_supports_block_fp8(capability)
            ),
            "deepseek_v4_block_fp8_selected_kernel": block_kernel_type.__name__,
            "fbgemm_token_fp8_selected_kernel": token_kernel_type.__name__,
            "block_kernel_keeps_activation_bf16": block_kernel_type.__name__
            == "MarlinFP8ScaledMMLinearKernel",
        }

    report["probes"].append(record_probe("fp8_kernel_selection", fp8_kernel_selection))

    def marlin_rejects_w8a8() -> dict[str, Any]:
        import torch
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )

        try:
            prepare_fp8_layer_for_marlin(torch.nn.Module(), input_dtype=torch.float8_e4m3fn)
        except RuntimeError as exc:
            return {
                "ok": True,
                "rejected": True,
                "error": str(exc),
                "policy": "SM80 Marlin FP8 is W8A16 weight-only; do not add activation FP8 quant.",
            }
        return {"ok": False, "rejected": False}

    report["probes"].append(record_probe("marlin_w8a8_rejection", marlin_rejects_w8a8))

    def torch_scaled_mm_fp8_smoke() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"ok": False, "error": "CUDA is unavailable"}
        a = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale = torch.ones(1, device="cuda", dtype=torch.float32)
        try:
            out = torch._scaled_mm(
                a,
                b,
                out_dtype=torch.bfloat16,
                scale_a=scale,
                scale_b=scale,
            )
            if isinstance(out, tuple):
                out = out[0]
            torch.cuda.synchronize()
            return {"ran": True, "output_dtype": str(out.dtype), "output_shape": list(out.shape)}
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "ran": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    report["probes"].append(record_probe("torch_scaled_mm_fp8_smoke", torch_scaled_mm_fp8_smoke))

    def int8_kernel_selection() -> dict[str, Any]:
        import torch
        import vllm.model_executor.kernels.linear as linear_kernels
        from vllm.model_executor.kernels.linear.scaled_mm import (
            Int8ScaledMMLinearLayerConfig,
        )

        config = Int8ScaledMMLinearLayerConfig(
            is_channelwise=True,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        kernel_type = linear_kernels.choose_scaled_mm_linear_kernel(
            config,
            linear_kernels._POSSIBLE_INT8_KERNELS,  # noqa: SLF001
        )
        x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)
        from vllm import _custom_ops as ops

        qx, x_scale, x_zp = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
        torch.cuda.synchronize()
        return {
            "selected_kernel": kernel_type.__name__,
            "scaled_int8_quant_shape": list(qx.shape),
            "scaled_int8_quant_dtype": str(qx.dtype),
            "scaled_int8_scale_shape": list(x_scale.shape),
            "scaled_int8_zero_point": None if x_zp is None else list(x_zp.shape),
            "activation_policy": "dynamic scaled_int8_quant inside apply_weights",
        }

    report["probes"].append(record_probe("int8_w8a8_kernel_selection", int8_kernel_selection))

    report["all_required_vllm_probes_ok"] = all(
        probe.get("ok", False)
        for probe in report["probes"]
        if probe["name"]
        in {
            "torch_cuda_environment",
            "vllm_imports_and_custom_ops",
            "fp8_kernel_selection",
            "marlin_w8a8_rejection",
            "int8_w8a8_kernel_selection",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

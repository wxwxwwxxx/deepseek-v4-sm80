from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


@dataclass(frozen=True)
class BridgeShape:
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int


class _LayerStub:
    def __init__(self, params_dtype: torch.dtype) -> None:
        self.params_dtype = params_dtype


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _time_cuda(fn, *, warmup: int, iters: int) -> tuple[float, Any]:
    last = None
    for _ in range(warmup):
        last = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        last = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1), last


def _choose_marlin_block_size(
    *,
    tokens: int,
    topk: int,
    experts: int,
    input_dtype: torch.dtype | None,
) -> int:
    block_size_m = 64
    for candidate in [8, 16, 32, 48, 64]:
        block_size_m = candidate
        if tokens * topk / experts / candidate < 0.9:
            break
    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)
    return block_size_m


def _make_topk_ids(tokens: int, topk: int, experts: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(tokens, device=device, dtype=torch.long).unsqueeze(1)
    cols = torch.arange(topk, device=device, dtype=torch.long).unsqueeze(0)
    return ((rows * topk + cols) % experts).contiguous()


def _make_case(
    shape: BridgeShape,
    *,
    device: torch.device,
    seed: int,
    scale_byte: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden_states = (
        torch.randn(
            shape.tokens,
            shape.hidden,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    ).to(torch.bfloat16)
    topk_weights = torch.rand(
        shape.tokens,
        shape.topk,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    topk_ids = _make_topk_ids(shape.tokens, shape.topk, shape.experts, device)

    w13_u8 = torch.randint(
        0,
        256,
        (shape.experts, 2 * shape.intermediate, shape.hidden // 2),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w2_u8 = torch.randint(
        0,
        256,
        (shape.experts, shape.hidden, shape.intermediate // 2),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    w13_scale_u8 = torch.full(
        (shape.experts, 2 * shape.intermediate, shape.hidden // 32),
        int(scale_byte),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale_u8 = torch.full(
        (shape.experts, shape.hidden, shape.intermediate // 32),
        int(scale_byte),
        device=device,
        dtype=torch.uint8,
    )
    return {
        "hidden_states": hidden_states.contiguous(),
        "topk_weights": topk_weights.contiguous(),
        "topk_ids": topk_ids,
        "w13_u8": w13_u8,
        "w2_u8": w2_u8,
        "w13_scale_u8": w13_scale_u8,
        "w2_scale_u8": w2_scale_u8,
    }


def _tensor_meta(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _record_error(exc: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(limit=20),
    }


def _prepare_marlin_weights(case: dict[str, torch.Tensor], *, params_dtype: torch.dtype):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )

    return prepare_moe_mxfp4_layer_for_marlin(
        _LayerStub(params_dtype),
        case["w13_u8"],
        case["w2_u8"],
        case["w13_scale_u8"],
        case["w2_scale_u8"],
        None,
        None,
    )


def _run_repack_probe(case: dict[str, torch.Tensor], shape: BridgeShape) -> dict[str, Any]:
    import vllm._custom_ops as ops

    perm = torch.empty(0, dtype=torch.int, device=case["w13_u8"].device)
    qweight = case["w13_u8"][0].view(torch.int32).T.contiguous()
    started = time.perf_counter()
    out = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=shape.hidden,
        size_n=2 * shape.intermediate,
        num_bits=4,
        is_a_8bit=False,
    )
    torch.cuda.synchronize()
    return {
        "status": "pass",
        "elapsed_s": time.perf_counter() - started,
        "input_qweight": _tensor_meta(qweight),
        "output": _tensor_meta(out),
    }


def _run_case(
    shape: BridgeShape,
    *,
    warmup: int,
    iters: int,
    seed: int,
    scale_byte: int,
) -> dict[str, Any]:
    from minisgl.kernel import deepseek_v4 as mini_dsv4
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    device = torch.device("cuda")
    case = _make_case(shape, device=device, seed=seed, scale_byte=scale_byte)
    payload: dict[str, Any] = {
        "shape": shape.__dict__,
        "raw_layout": {
            "mini_w13": "[experts, 2, local_intermediate, hidden/2] int8 byte-compatible MXFP4",
            "mini_w2": "[experts, hidden, local_intermediate/2] int8 byte-compatible MXFP4",
            "vllm_w13_expected": "[experts, 2*local_intermediate, hidden/2] uint8",
            "vllm_w2_expected": "[experts, hidden, local_intermediate/2] uint8",
            "scale_byte": scale_byte,
            "scale_semantic": "uint8 viewed as torch.float8_e8m0fnu; 127 is scale 1.0",
        },
        "inputs": {
            key: _tensor_meta(value) for key, value in case.items() if key not in {"hidden_states"}
        },
        "hidden_states": _tensor_meta(case["hidden_states"]),
    }

    try:
        payload["direct_repack_probe"] = _run_repack_probe(case, shape)
    except BaseException as exc:
        payload["direct_repack_probe"] = _record_error(exc)
        return payload

    try:
        started = time.perf_counter()
        marlin_w13, marlin_w2, marlin_w13_scale, marlin_w2_scale, _, _ = _prepare_marlin_weights(
            case, params_dtype=torch.bfloat16
        )
        torch.cuda.synchronize()
        payload["weight_transform"] = {
            "status": "pass",
            "elapsed_s": time.perf_counter() - started,
            "marlin_w13": _tensor_meta(marlin_w13),
            "marlin_w2": _tensor_meta(marlin_w2),
            "marlin_w13_scale": _tensor_meta(marlin_w13_scale),
            "marlin_w2_scale": _tensor_meta(marlin_w2_scale),
        }
    except BaseException as exc:
        payload["weight_transform"] = _record_error(exc)
        return payload

    block_size_m = _choose_marlin_block_size(
        tokens=shape.tokens,
        topk=shape.topk,
        experts=shape.experts,
        input_dtype=None,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        case["topk_ids"],
        block_size_m,
        shape.experts,
        None,
        ignore_invalid_experts=True,
    )
    mini_plan = mini_dsv4.build_moe_route_plan(
        case["topk_ids"],
        num_experts=shape.experts,
        block_size_m=block_size_m,
    )
    payload["route_metadata"] = {
        "block_size_m": block_size_m,
        "vllm_sorted_token_ids": _tensor_meta(sorted_token_ids),
        "vllm_expert_ids": _tensor_meta(expert_ids),
        "vllm_num_tokens_post_padded": int(num_tokens_post_padded.item()),
        "mini_sorted_route_ids": _tensor_meta(mini_plan.sorted_route_ids),
        "mini_expert_ids": _tensor_meta(mini_plan.expert_ids),
        "mini_num_tokens_post_padded": int(mini_plan.num_tokens_post_padded.item()),
        "same_num_tokens_post_padded": bool(
            int(num_tokens_post_padded.item()) == int(mini_plan.num_tokens_post_padded.item())
        ),
        "same_expert_id_prefix": bool(
            torch.equal(
                expert_ids[: mini_plan.expert_ids.numel()],
                mini_plan.expert_ids[: expert_ids.numel()],
            )
        ),
    }

    quant_type = scalar_types.float4_e2m1f
    workspace = marlin_make_workspace_new(device, 4)
    direct_w13_out = torch.empty(
        (shape.tokens * shape.topk, 2 * shape.intermediate),
        device=device,
        dtype=case["hidden_states"].dtype,
    )

    def run_direct_w13() -> torch.Tensor:
        import vllm._custom_ops as ops

        return ops.moe_wna16_marlin_gemm(
            case["hidden_states"],
            direct_w13_out,
            marlin_w13,
            None,
            marlin_w13_scale,
            None,
            None,
            None,
            None,
            None,
            workspace,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            case["topk_weights"],
            moe_block_size=block_size_m,
            top_k=shape.topk,
            mul_topk_weights=False,
            b_q_type=quant_type,
            size_m=shape.tokens,
            size_n=2 * shape.intermediate,
            size_k=shape.hidden,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    try:
        direct_w13_ms, direct_w13 = _time_cuda(run_direct_w13, warmup=warmup, iters=iters)
        payload["direct_moe_wna16_marlin_gemm_w13"] = {
            "status": "pass",
            "ms": direct_w13_ms,
            "output": _tensor_meta(direct_w13),
            "finite": bool(torch.isfinite(direct_w13.float()).all().item()),
        }
    except BaseException as exc:
        payload["direct_moe_wna16_marlin_gemm_w13"] = _record_error(exc)
        return payload

    def run_fused_marlin() -> torch.Tensor:
        return fused_marlin_moe(
            hidden_states=case["hidden_states"],
            w1=marlin_w13,
            w2=marlin_w2,
            bias1=None,
            bias2=None,
            w1_scale=marlin_w13_scale,
            w2_scale=marlin_w2_scale,
            topk_weights=case["topk_weights"],
            topk_ids=case["topk_ids"],
            quant_type_id=quant_type.id,
            apply_router_weight_on_input=False,
            global_num_experts=shape.experts,
            activation=MoEActivation.SILU,
            input_dtype=None,
            clamp_limit=2.5,
        )

    try:
        marlin_ms, marlin_out = _time_cuda(run_fused_marlin, warmup=warmup, iters=iters)
        payload["fused_marlin_moe"] = {
            "status": "pass",
            "ms": marlin_ms,
            "output": _tensor_meta(marlin_out),
            "finite": bool(torch.isfinite(marlin_out.float()).all().item()),
        }
    except BaseException as exc:
        payload["fused_marlin_moe"] = _record_error(exc)
        return payload

    old_env = {
        "MINISGL_DSV4_SM80_V1_MOE": os.environ.get("MINISGL_DSV4_SM80_V1_MOE"),
        "MINISGL_DSV4_SM80_MOE_V2": os.environ.get("MINISGL_DSV4_SM80_MOE_V2"),
        mini_dsv4.DSV4_SM80_MOE_EXPERT_BACKEND_ENV: os.environ.get(
            mini_dsv4.DSV4_SM80_MOE_EXPERT_BACKEND_ENV
        ),
    }
    _patch_mini_capability_for_vllm_venv(mini_dsv4)
    os.environ["MINISGL_DSV4_SM80_V1_MOE"] = "1"
    os.environ["MINISGL_DSV4_SM80_MOE_V2"] = "1"
    os.environ.pop(mini_dsv4.DSV4_SM80_MOE_EXPERT_BACKEND_ENV, None)
    mini_w13 = (
        case["w13_u8"]
        .view(torch.int8)
        .view(
            shape.experts,
            2,
            shape.intermediate,
            shape.hidden // 2,
        )
    )
    mini_w2 = case["w2_u8"].view(torch.int8)
    e8m0_dtype = mini_dsv4.e8m0_dtype()
    mini_w13_scale = (
        case["w13_scale_u8"]
        .view(e8m0_dtype)
        .view(
            shape.experts,
            2,
            shape.intermediate,
            shape.hidden // 32,
        )
    )
    mini_w2_scale = case["w2_scale_u8"].view(e8m0_dtype)
    mini_workspace = mini_dsv4.DSV4MoEWorkspace()
    mini_exec_plan = mini_dsv4.build_moe_v2_execution_plan(
        case["hidden_states"],
        case["topk_weights"],
        case["topk_ids"],
        num_experts=shape.experts,
        block_size_m=block_size_m,
    )

    def run_grouped() -> torch.Tensor:
        out = mini_dsv4.moe_route_dispatch_bf16_grouped(
            case["hidden_states"],
            case["topk_weights"],
            case["topk_ids"],
            mini_w13,
            mini_w13_scale,
            mini_w2,
            mini_w2_scale,
            swiglu_limit=2.5,
            moe_plan=mini_exec_plan,
            workspace=mini_workspace,
        )
        if out is None:
            raise RuntimeError("mini grouped_fp4 returned None")
        return out

    try:
        grouped_ms, grouped_out = _time_cuda(run_grouped, warmup=warmup, iters=iters)
        diff = (marlin_out.float() - grouped_out.float()).abs()
        denom = grouped_out.float().abs().clamp_min(1e-4)
        payload["mini_grouped_fp4"] = {
            "status": "pass",
            "ms": grouped_ms,
            "output": _tensor_meta(grouped_out),
            "finite": bool(torch.isfinite(grouped_out.float()).all().item()),
        }
        payload["numerics_vs_grouped_fp4"] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "max_rel": float((diff / denom).max().item()),
        }
        payload["speedup_vs_grouped_fp4"] = grouped_ms / marlin_ms if marlin_ms > 0 else None
    except BaseException as exc:
        payload["mini_grouped_fp4"] = _record_error(exc)
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    payload["topk_weight_application"] = {
        "vllm_bridge": "apply_router_weight_on_input=False, so topk weights are multiplied in W2 GEMM",
        "mini_grouped": "route weights are applied before W2 input; mathematically equivalent for per-route scalar weights",
    }
    payload["silent_fallback"] = False
    payload["uses_vllm_runtime_dependency"] = True
    return payload


def _patch_mini_capability_for_vllm_venv(mini_dsv4) -> None:
    """Avoid unrelated sgl_kernel ABI probing while timing mini Triton kernels.

    The vLLM virtualenv used by this bridge probe can contain an sgl_kernel wheel
    that is unrelated to mini's current process.  mini's capability detector
    imports it while deciding whether a Triton-only DSV4 path is enabled.  For
    the grouped-FP4 comparison in this script we only need sm80 + Triton.
    """

    mini_dsv4.detect_dsv4_kernel_capabilities = lambda: mini_dsv4.DSV4KernelCapability(
        cuda_available=True,
        cuda_capability=torch.cuda.get_device_capability(0),
        is_sm80=torch.cuda.get_device_capability(0) == (8, 0),
        sgl_kernel_available=False,
        sgl_kernel_error="skipped in vLLM bridge probe",
        sgl_kernel_sm80_common_ops=False,
        sgl_kernel_dsv4_ops={},
        flash_mla_available=False,
        flash_mla_error="skipped in vLLM bridge probe",
        flashinfer_available=False,
        flashinfer_error="skipped in vLLM bridge probe",
        deep_gemm_available=False,
        deep_gemm_error="skipped in vLLM bridge probe",
        deep_gemm_usable=False,
        tilelang_available=False,
        tilelang_error="skipped in vLLM bridge probe",
        triton_available=True,
        triton_error=None,
        marlin_available=True,
        marlin_error=None,
    )


def _run_import_probe() -> dict[str, Any]:
    import vllm
    import vllm._custom_ops as ops

    return {
        "status": "pass",
        "python": sys.executable,
        "torch": torch.__version__,
        "vllm": getattr(vllm, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        "has_vllm_custom_ops_gptq_marlin_repack": hasattr(ops, "gptq_marlin_repack"),
        "has_vllm_custom_ops_moe_wna16_marlin_gemm": hasattr(ops, "moe_wna16_marlin_gemm"),
        "has_torch_ops_C_gptq_marlin_repack": hasattr(
            getattr(torch.ops, "_C", None), "gptq_marlin_repack"
        ),
        "has_torch_ops_moe_C_moe_wna16_marlin_gemm": hasattr(
            getattr(torch.ops, "_moe_C", None), "moe_wna16_marlin_gemm"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe vLLM Marlin custom-op bridge feasibility for DSV4 MoE."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_marlin_custom_op_bridge/raw/"
        / "vllm_marlin_bridge_probe.json",
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 4096])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scale-byte", type=int, default=127)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "target": "TARGET_07.39_dsv4_sm80_marlin_custom_op_bridge",
        "command": " ".join(sys.argv),
        "bridge_policy": {
            "experimental_only": True,
            "promoted_path": False,
            "allowed_dependency": "/workspace/venvs/vllm-dsv4 compiled ops for probe only",
            "silent_fallback_allowed": False,
        },
    }
    try:
        report["import_probe"] = _run_import_probe()
    except BaseException as exc:
        report["import_probe"] = _record_error(exc)
        report["cases"] = []
    else:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0):
            report["cases"] = []
            report["status"] = "error"
            report["error"] = "This probe requires CUDA sm80."
        else:
            cases = []
            for token_count in args.tokens:
                shape = BridgeShape(
                    tokens=int(token_count),
                    hidden=args.hidden,
                    intermediate=args.intermediate,
                    experts=args.experts,
                    topk=args.topk,
                )
                try:
                    cases.append(
                        _run_case(
                            shape,
                            warmup=args.warmup,
                            iters=args.iters,
                            seed=args.seed + int(token_count),
                            scale_byte=args.scale_byte,
                        )
                    )
                except BaseException as exc:
                    cases.append({"shape": shape.__dict__, **_record_error(exc)})
                    break
            report["cases"] = cases

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(report), indent=2 if args.pretty else None, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()

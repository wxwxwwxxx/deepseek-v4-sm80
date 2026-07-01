from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

FLOAT4_E2M1F_ID = 562949953487106


@dataclass(frozen=True)
class ProbeShape:
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int


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


def _load_extension(build_dir: Path, extension_name: str) -> Any:
    so_path = build_dir / f"{extension_name}.so"
    if not so_path.exists():
        raise FileNotFoundError(
            f"{so_path} does not exist; run probe_minimal_marlin_extension_build.py first"
        )
    spec = importlib.util.spec_from_file_location(extension_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    shape: ProbeShape,
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
        marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(marlin_scales.size(0), -1)
    marlin_scales = marlin_scales.to(torch.float8_e8m0fnu)
    if input_dtype == torch.float8_e4m3fn:
        marlin_scales = marlin_scales.view(torch.uint8)
        if bool((marlin_scales > 249).any().item()):
            raise ValueError("MXFP4 e8m0 scale exponent overflow for FP8 activation")
        marlin_scales = (marlin_scales + 6).view(torch.float8_e8m0fnu)
    return marlin_scales.contiguous()


def _prepare_mxfp4_moe_for_marlin(
    ops: Any,
    case: dict[str, torch.Tensor],
    shape: ProbeShape,
    *,
    param_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = torch.empty(0, dtype=torch.int, device=case["w13_u8"].device)

    def repack_weight(weight: torch.Tensor, *, size_n: int, size_k: int) -> torch.Tensor:
        pieces = []
        for expert in range(shape.experts):
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
        typed = scales.view(torch.float8_e8m0fnu).to(param_dtype)
        pieces = []
        for expert in range(shape.experts):
            marlin_scales = _marlin_permute_scales(
                typed[expert].T,
                size_k=size_k,
                size_n=size_n,
                group_size=32,
            )
            pieces.append(_mxfp4_marlin_process_scales(marlin_scales))
        return torch.stack(pieces, dim=0)

    w13 = repack_weight(
        case["w13_u8"],
        size_n=2 * shape.intermediate,
        size_k=shape.hidden,
    )
    w2 = repack_weight(
        case["w2_u8"],
        size_n=shape.hidden,
        size_k=shape.intermediate,
    )
    w13_scale = permute_scales(
        case["w13_scale_u8"],
        size_n=2 * shape.intermediate,
        size_k=shape.hidden,
    )
    w2_scale = permute_scales(
        case["w2_scale_u8"],
        size_n=shape.hidden,
        size_k=shape.intermediate,
    )
    return w13, w2, w13_scale, w2_scale


def _patch_mini_capability_for_probe(mini_dsv4) -> None:
    mini_dsv4.detect_dsv4_kernel_capabilities = lambda: mini_dsv4.DSV4KernelCapability(
        cuda_available=True,
        cuda_capability=torch.cuda.get_device_capability(0),
        is_sm80=torch.cuda.get_device_capability(0) == (8, 0),
        sgl_kernel_available=False,
        sgl_kernel_error="skipped in Marlin WNA16 csrc probe",
        sgl_kernel_sm80_common_ops=False,
        sgl_kernel_dsv4_ops={},
        flash_mla_available=False,
        flash_mla_error="skipped in Marlin WNA16 csrc probe",
        flashinfer_available=False,
        flashinfer_error="skipped in Marlin WNA16 csrc probe",
        deep_gemm_available=False,
        deep_gemm_error="skipped in Marlin WNA16 csrc probe",
        deep_gemm_usable=False,
        tilelang_available=False,
        tilelang_error="skipped in Marlin WNA16 csrc probe",
        triton_available=True,
        triton_error=None,
        marlin_available=True,
        marlin_error=None,
    )


def _run_case(
    ops: Any,
    shape: ProbeShape,
    *,
    warmup: int,
    iters: int,
    seed: int,
    scale_byte: int,
) -> dict[str, Any]:
    from minisgl.kernel import deepseek_v4 as mini_dsv4

    device = torch.device("cuda")
    case = _make_case(shape, device=device, seed=seed, scale_byte=scale_byte)
    payload: dict[str, Any] = {
        "shape": shape.__dict__,
        "inputs": {
            key: _tensor_meta(value) for key, value in case.items() if key != "hidden_states"
        },
        "hidden_states": _tensor_meta(case["hidden_states"]),
        "runtime_dependency": "mini-owned probe extension namespace; no vLLM import",
    }

    try:
        perm = torch.empty(0, dtype=torch.int, device=device)
        qweight = case["w13_u8"][0].view(torch.int32).T.contiguous()
        started = time.perf_counter()
        repacked = ops.gptq_marlin_repack(
            qweight, perm, shape.hidden, 2 * shape.intermediate, 4, False
        )
        torch.cuda.synchronize()
        payload["direct_repack_probe"] = {
            "status": "pass",
            "elapsed_s": time.perf_counter() - started,
            "output": _tensor_meta(repacked),
        }
    except BaseException as exc:
        payload["direct_repack_probe"] = _record_error(exc)
        return payload

    try:
        started = time.perf_counter()
        marlin_w13, marlin_w2, marlin_w13_scale, marlin_w2_scale = _prepare_mxfp4_moe_for_marlin(
            ops,
            case,
            shape,
            param_dtype=torch.bfloat16,
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
    route_plan = mini_dsv4.build_moe_route_plan(
        case["topk_ids"],
        num_experts=shape.experts,
        block_size_m=block_size_m,
    )
    payload["route_metadata"] = {
        "block_size_m": block_size_m,
        "sorted_route_ids": _tensor_meta(route_plan.sorted_route_ids),
        "expert_ids": _tensor_meta(route_plan.expert_ids),
        "num_tokens_post_padded": int(route_plan.num_tokens_post_padded.item()),
    }

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(sms * 4, dtype=torch.int, device=device)

    direct_w13_out = torch.empty(
        (shape.tokens * shape.topk, 2 * shape.intermediate),
        device=device,
        dtype=case["hidden_states"].dtype,
    )

    def run_direct_w13() -> torch.Tensor:
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
            route_plan.sorted_route_ids,
            route_plan.expert_ids,
            route_plan.num_tokens_post_padded,
            case["topk_weights"],
            block_size_m,
            shape.topk,
            False,
            FLOAT4_E2M1F_ID,
            shape.tokens,
            2 * shape.intermediate,
            shape.hidden,
            True,
            False,
            True,
            False,
            -1,
            -1,
            -1,
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

    w13_cache = torch.empty(
        (shape.tokens * shape.topk, 2 * shape.intermediate),
        device=device,
        dtype=case["hidden_states"].dtype,
    )
    activated = torch.empty(
        (shape.tokens * shape.topk, shape.intermediate),
        device=device,
        dtype=case["hidden_states"].dtype,
    )
    output_routes = torch.empty(
        (shape.tokens * shape.topk, shape.hidden),
        device=device,
        dtype=case["hidden_states"].dtype,
    )

    def run_fused_marlin() -> torch.Tensor:
        w13_out = ops.moe_wna16_marlin_gemm(
            case["hidden_states"],
            w13_cache,
            marlin_w13,
            None,
            marlin_w13_scale,
            None,
            None,
            None,
            None,
            None,
            workspace,
            route_plan.sorted_route_ids,
            route_plan.expert_ids,
            route_plan.num_tokens_post_padded,
            case["topk_weights"],
            block_size_m,
            shape.topk,
            False,
            FLOAT4_E2M1F_ID,
            shape.tokens,
            2 * shape.intermediate,
            shape.hidden,
            True,
            False,
            True,
            False,
            -1,
            -1,
            -1,
        )
        gate = torch.clamp(w13_out[:, : shape.intermediate].float(), max=2.5)
        up = torch.clamp(
            w13_out[:, shape.intermediate :].float(),
            min=-2.5,
            max=2.5,
        )
        activated.copy_((F.silu(gate) * up).to(activated.dtype))
        route_out = ops.moe_wna16_marlin_gemm(
            activated,
            output_routes,
            marlin_w2,
            None,
            marlin_w2_scale,
            None,
            None,
            None,
            None,
            None,
            workspace,
            route_plan.sorted_route_ids,
            route_plan.expert_ids,
            route_plan.num_tokens_post_padded,
            case["topk_weights"],
            block_size_m,
            1,
            True,
            FLOAT4_E2M1F_ID,
            shape.tokens * shape.topk,
            shape.hidden,
            shape.intermediate,
            True,
            False,
            True,
            False,
            -1,
            -1,
            -1,
        )
        return route_out.view(shape.tokens, shape.topk, shape.hidden).sum(dim=1)

    try:
        fused_ms, fused_out = _time_cuda(run_fused_marlin, warmup=warmup, iters=iters)
        payload["fused_marlin_moe_local_ops"] = {
            "status": "pass",
            "ms": fused_ms,
            "output": _tensor_meta(fused_out),
            "finite": bool(torch.isfinite(fused_out.float()).all().item()),
        }
    except BaseException as exc:
        payload["fused_marlin_moe_local_ops"] = _record_error(exc)
        return payload

    old_env = {
        "MINISGL_DSV4_SM80_V1_MOE": os.environ.get("MINISGL_DSV4_SM80_V1_MOE"),
        "MINISGL_DSV4_SM80_MOE_V2": os.environ.get("MINISGL_DSV4_SM80_MOE_V2"),
        mini_dsv4.DSV4_SM80_MOE_EXPERT_BACKEND_ENV: os.environ.get(
            mini_dsv4.DSV4_SM80_MOE_EXPERT_BACKEND_ENV
        ),
    }
    _patch_mini_capability_for_probe(mini_dsv4)
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
        diff = (fused_out.float() - grouped_out.float()).abs()
        denom = grouped_out.float().abs().clamp_min(1e-4)
        payload["mini_grouped_fp4"] = {
            "status": "pass",
            "ms": grouped_ms,
            "output": _tensor_meta(grouped_out),
            "finite": bool(torch.isfinite(grouped_out.float()).all().item()),
        }
        payload["speedup_vs_grouped_fp4"] = grouped_ms / fused_ms if fused_ms > 0 else None
        payload["numerics_vs_grouped_fp4"] = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "max_rel": float((diff / denom).max().item()),
            "interpretation": "random synthetic packed MXFP4 comparison only",
        }
    except BaseException as exc:
        payload["mini_grouped_fp4"] = _record_error(exc)
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    payload["silent_fallback"] = False
    payload["uses_vllm_runtime_dependency"] = False
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe mini-owned Marlin WNA16 custom-op synthetic MoE calls."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_marlin_wna16_csrc_port/raw/"
        / "minisgl_marlin_wna16_ops_probe.json",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_marlin_wna16_csrc_port/raw/"
        / "torch_extension_build",
    )
    parser.add_argument("--extension-name", default="minisgl_marlin_wna16_probe")
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 4096])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scale-byte", type=int, default=127)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "target": "TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port",
        "command": " ".join(sys.argv),
        "extension": {
            "name": args.extension_name,
            "build_dir": args.build_dir,
            "runtime_policy": "mini-owned compiled op namespace; no vLLM import",
        },
    }
    try:
        _load_extension(args.build_dir, args.extension_name)
        namespace = getattr(torch.ops, args.extension_name)
        report["extension"]["import_status"] = "pass"
        report["extension"]["has_gptq_marlin_repack"] = hasattr(
            namespace,
            "gptq_marlin_repack",
        )
        report["extension"]["has_moe_wna16_marlin_gemm"] = hasattr(
            namespace,
            "moe_wna16_marlin_gemm",
        )
    except BaseException as exc:
        report["extension"]["import_status"] = "error"
        report["extension"]["import_error"] = _record_error(exc)
        report["cases"] = []
    else:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0):
            report["status"] = "error"
            report["error"] = "This probe requires CUDA sm80."
            report["cases"] = []
        else:
            cases = []
            for token_count in args.tokens:
                shape = ProbeShape(
                    tokens=int(token_count),
                    hidden=args.hidden,
                    intermediate=args.intermediate,
                    experts=args.experts,
                    topk=args.topk,
                )
                try:
                    cases.append(
                        _run_case(
                            namespace,
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
        json.dumps(_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

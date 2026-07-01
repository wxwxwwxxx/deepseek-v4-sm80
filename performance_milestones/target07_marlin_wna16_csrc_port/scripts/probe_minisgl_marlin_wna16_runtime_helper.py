from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


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


def _make_topk_ids(tokens: int, topk: int, experts: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(tokens, device=device, dtype=torch.long).unsqueeze(1)
    cols = torch.arange(topk, device=device, dtype=torch.long).unsqueeze(0)
    return ((rows * topk + cols) % experts).contiguous()


def _make_case(
    *,
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    topk: int,
    seed: int,
    scale_byte: int,
) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden_states = (
        torch.randn(tokens, hidden, device=device, dtype=torch.float32, generator=generator) * 0.01
    ).to(torch.bfloat16)
    topk_weights = torch.rand(tokens, topk, device=device, dtype=torch.float32, generator=generator)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    w13_weight = torch.randint(
        0,
        256,
        (experts, 2, intermediate, hidden // 2),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.int8)
    w2_weight = torch.randint(
        0,
        256,
        (experts, hidden, intermediate // 2),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.int8)
    w13_scale = torch.full(
        (experts, 2, intermediate, hidden // 32),
        int(scale_byte),
        device=device,
        dtype=torch.uint8,
    ).view(torch.float8_e8m0fnu)
    w2_scale = torch.full(
        (experts, hidden, intermediate // 32),
        int(scale_byte),
        device=device,
        dtype=torch.uint8,
    ).view(torch.float8_e8m0fnu)
    return {
        "hidden_states": hidden_states.contiguous(),
        "topk_weights": topk_weights.contiguous(),
        "topk_ids": _make_topk_ids(tokens, topk, experts, device),
        "w13_weight": w13_weight.contiguous(),
        "w13_scale": w13_scale.contiguous(),
        "w2_weight": w2_weight.contiguous(),
        "w2_scale": w2_scale.contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke the integrated mini Marlin WNA16 runtime helper without importing vLLM."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_marlin_wna16_csrc_port/raw/"
        / "minisgl_marlin_wna16_runtime_helper_smoke.json",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_marlin_wna16_csrc_port/raw/"
        / "runtime_extension_build",
    )
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--scale-byte", type=int, default=127)
    args = parser.parse_args()

    os.environ["MINISGL_MARLIN_WNA16_BUILD_DIR"] = str(args.build_dir)
    report: dict[str, Any] = {
        "target": "TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port",
        "purpose": "integrated runtime-helper smoke; no vLLM runtime dependency",
        "build_dir": args.build_dir,
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "topk": args.topk,
        },
        "uses_vllm_runtime_dependency": False,
        "silent_fallback": False,
        "status": "unknown",
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        from minisgl.kernel import deepseek_v4 as dsv4_kernel
        from minisgl.kernel import marlin_wna16

        report["vllm_imported_before"] = any(
            name == "vllm" or name.startswith("vllm.") for name in sys.modules
        )

        load_started = time.perf_counter()
        ops = marlin_wna16.load_ops()
        torch.cuda.synchronize()
        report["load_ops"] = {
            "status": "pass",
            "elapsed_s": time.perf_counter() - load_started,
            "has_gptq_marlin_repack": hasattr(ops, "gptq_marlin_repack"),
            "has_moe_wna16_marlin_gemm": hasattr(ops, "moe_wna16_marlin_gemm"),
        }

        case = _make_case(
            tokens=args.tokens,
            hidden=args.hidden,
            intermediate=args.intermediate,
            experts=args.experts,
            topk=args.topk,
            seed=args.seed,
            scale_byte=args.scale_byte,
        )
        report["inputs"] = {key: _tensor_meta(value) for key, value in case.items()}

        first_started = time.perf_counter()
        output, cache = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16(
            case["hidden_states"],
            case["topk_weights"],
            case["topk_ids"],
            case["w13_weight"],
            case["w13_scale"],
            case["w2_weight"],
            case["w2_scale"],
            swiglu_limit=2.5,
            cache=None,
        )
        torch.cuda.synchronize()
        report["first_call"] = {
            "status": "pass",
            "elapsed_s": time.perf_counter() - first_started,
            "output": _tensor_meta(output),
            "finite": bool(torch.isfinite(output.float()).all().item()),
            "cache_type": type(cache).__name__,
            "cache_w13": _tensor_meta(cache.w13),
            "cache_w2": _tensor_meta(cache.w2),
            "cache_w13_scale": _tensor_meta(cache.w13_scale),
            "cache_w2_scale": _tensor_meta(cache.w2_scale),
        }

        second_started = time.perf_counter()
        output2, cache2 = dsv4_kernel.moe_route_dispatch_bf16_marlin_wna16(
            case["hidden_states"],
            case["topk_weights"],
            case["topk_ids"],
            case["w13_weight"],
            case["w13_scale"],
            case["w2_weight"],
            case["w2_scale"],
            swiglu_limit=2.5,
            cache=cache,
        )
        torch.cuda.synchronize()
        report["second_call_cache_reuse"] = {
            "status": "pass",
            "elapsed_s": time.perf_counter() - second_started,
            "output": _tensor_meta(output2),
            "finite": bool(torch.isfinite(output2.float()).all().item()),
            "cache_object_reused": cache2 is cache,
            "max_abs_delta_vs_first": float((output2.float() - output.float()).abs().max().item()),
        }

        report["vllm_imported_after"] = any(
            name == "vllm" or name.startswith("vllm.") for name in sys.modules
        )
        report["uses_vllm_runtime_dependency"] = bool(report["vllm_imported_after"])
        report["status"] = "pass"
    except BaseException as exc:
        report.update(_record_error(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from minisgl.kernel import deepseek_v4 as dsv4_kernel


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _fp4_linear_bf16(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    w = dsv4_kernel.dequant_fp4_weight(weight, scale, out_dtype=x.dtype)
    return F.linear(x, w)


def _routed_reference_bf16(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    swiglu_limit: float,
) -> torch.Tensor:
    y = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_idx in range(w13_weight.shape[0]):
        token_idx, top_idx = torch.where(indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        x = hidden_states[token_idx]
        w1 = _fp4_linear_bf16(x, w13_weight[expert_idx, 0], w13_scale[expert_idx, 0]).float()
        w3 = _fp4_linear_bf16(x, w13_weight[expert_idx, 1], w13_scale[expert_idx, 1]).float()
        hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            w1,
            w3,
            swiglu_limit=swiglu_limit,
            weights=weights[token_idx, top_idx, None],
        )
        y[token_idx] += _fp4_linear_bf16(
            hidden.to(hidden_states.dtype),
            w2_weight[expert_idx],
            w2_scale[expert_idx],
        ).float()
    return y.to(hidden_states.dtype)


def _routed_current_fallback(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    w13_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    swiglu_limit: float,
) -> torch.Tensor:
    y = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_idx in range(w13_weight.shape[0]):
        token_idx, top_idx = torch.where(indices == expert_idx)
        if token_idx.numel() == 0:
            continue
        x = hidden_states[token_idx]
        w1 = dsv4_kernel.quantized_linear_ref(
            x,
            w13_weight[expert_idx, 0],
            w13_scale[expert_idx, 0],
            weight_kind="fp4",
        ).float()
        w3 = dsv4_kernel.quantized_linear_ref(
            x,
            w13_weight[expert_idx, 1],
            w13_scale[expert_idx, 1],
            weight_kind="fp4",
        ).float()
        hidden = dsv4_kernel.silu_and_mul_clamp_fallback(
            w1,
            w3,
            swiglu_limit=swiglu_limit,
            weights=weights[token_idx, top_idx, None],
        )
        y[token_idx] += dsv4_kernel.quantized_linear_ref(
            hidden.to(hidden_states.dtype),
            w2_weight[expert_idx],
            w2_scale[expert_idx],
            weight_kind="fp4",
        ).float()
    return y.to(hidden_states.dtype)


def _make_case(
    *,
    tokens: int,
    topk: int,
    experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden_states = torch.randn(
        tokens,
        hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights = torch.rand(tokens, topk, device=device, dtype=torch.float32, generator=generator)
    indices = torch.empty(tokens, topk, device=device, dtype=torch.int64)
    for row in range(tokens):
        indices[row] = torch.randperm(experts, device=device, generator=generator)[:topk]

    w13_weight = torch.randint(
        -128,
        127,
        (experts, 2, intermediate, hidden // 2),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    w13_scale = torch.rand(
        experts,
        2,
        intermediate,
        dsv4_kernel.scale_dim(hidden, block_size=32),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    w2_weight = torch.randint(
        -128,
        127,
        (experts, hidden, intermediate // 2),
        device=device,
        dtype=torch.int8,
        generator=generator,
    )
    w2_scale = torch.rand(
        experts,
        hidden,
        dsv4_kernel.scale_dim(intermediate, block_size=32),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    return {
        "hidden_states": hidden_states,
        "weights": weights,
        "indices": indices,
        "w13_weight": w13_weight,
        "w13_scale": w13_scale,
        "w2_weight": w2_weight,
        "w2_scale": w2_scale,
    }


def _run_case(name: str, spec: dict[str, int], *, warmup: int, iters: int) -> dict[str, object]:
    device = torch.device("cuda")
    tensors = _make_case(device=device, seed=11, **spec)
    swiglu_limit = 2.5

    reference = _routed_reference_bf16(**tensors, swiglu_limit=swiglu_limit)
    current = _routed_current_fallback(**tensors, swiglu_limit=swiglu_limit)
    old_env = os.environ.get("MINISGL_DSV4_SM80_MOE_ROUTE")
    os.environ["MINISGL_DSV4_SM80_MOE_ROUTE"] = "1"
    try:
        grouped = dsv4_kernel.moe_route_dispatch_bf16_grouped(**tensors, swiglu_limit=swiglu_limit)
        if grouped is None:
            raise RuntimeError("grouped MoE route dispatch returned None")
        grouped_ms = _time_cuda(
            lambda: dsv4_kernel.moe_route_dispatch_bf16_grouped(
                **tensors,
                swiglu_limit=swiglu_limit,
            ),
            warmup=warmup,
            iters=iters,
        )
    finally:
        if old_env is None:
            os.environ.pop("MINISGL_DSV4_SM80_MOE_ROUTE", None)
        else:
            os.environ["MINISGL_DSV4_SM80_MOE_ROUTE"] = old_env

    current_ms = _time_cuda(
        lambda: _routed_current_fallback(**tensors, swiglu_limit=swiglu_limit),
        warmup=warmup,
        iters=iters,
    )
    ref_ms = _time_cuda(
        lambda: _routed_reference_bf16(**tensors, swiglu_limit=swiglu_limit),
        warmup=warmup,
        iters=iters,
    )

    max_abs = (grouped.float() - reference.float()).abs().max().item()
    denom = reference.float().abs().clamp_min(1e-4)
    max_rel = ((grouped.float() - reference.float()).abs() / denom).max().item()
    current_vs_bf16 = (current.float() - reference.float()).abs().max().item()
    return {
        "name": name,
        **spec,
        "current_fallback_ms": current_ms,
        "bf16_reference_ms": ref_ms,
        "grouped_ms": grouped_ms,
        "speedup_vs_current_fallback": current_ms / grouped_ms,
        "speedup_vs_bf16_reference": ref_ms / grouped_ms,
        "grouped_vs_bf16_max_abs": max_abs,
        "grouped_vs_bf16_max_rel": max_rel,
        "current_quant_act_vs_bf16_max_abs": current_vs_bf16,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark DSV4 grouped MoE route dispatch.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/dsv4_moe_route_dispatch_bf16_grouped_microbench.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")

    cases = {
        "decode_tiny": dict(tokens=4, topk=2, experts=8, hidden=64, intermediate=32),
        "decode_grouped": dict(tokens=8, topk=6, experts=16, hidden=256, intermediate=128),
        "prefill_grouped": dict(tokens=64, topk=6, experts=16, hidden=256, intermediate=128),
    }
    results = [
        _run_case(name, spec, warmup=args.warmup, iters=args.iters)
        for name, spec in cases.items()
    ]
    payload = {
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

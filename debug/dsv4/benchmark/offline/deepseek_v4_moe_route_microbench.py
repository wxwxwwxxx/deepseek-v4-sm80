from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from minisgl.kernel import deepseek_v4 as dsv4_kernel
from minisgl.kernel.triton import deepseek_v4 as triton_dsv4


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


def _runner_finalize_current_backend(routed_output: torch.Tensor) -> torch.Tensor:
    return routed_output.float()


def _marlin_candidate_report() -> dict[str, object]:
    return {
        "backend": dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_MARLIN_MXFP4_W4A16,
        "status": "unsupported",
        "attempted": False,
        "silent_fallback": False,
        "blocker": dsv4_kernel.DSV4_SM80_MOE_MARLIN_BLOCKER,
        "required_ops": [
            "gptq_marlin_repack",
            "_moe_C::moe_wna16_marlin_gemm",
        ],
    }


def _grouped_stage_breakdown(
    tensors: dict[str, torch.Tensor],
    route_plan: dsv4_kernel.DSV4MoERoutePlan,
    *,
    swiglu_limit: float,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    hidden_states = tensors["hidden_states"]
    route_weights = (
        tensors["weights"]
        .to(
            device=hidden_states.device,
            dtype=torch.float32,
        )
        .reshape(-1)
        .contiguous()
    )
    workspace = dsv4_kernel.DSV4MoEWorkspace()

    def run_w13() -> tuple[torch.Tensor, torch.Tensor]:
        w13 = triton_dsv4._grouped_fp4_w13(
            hidden_states,
            tensors["w13_weight"],
            tensors["w13_scale"],
            route_plan.sorted_route_ids,
            route_plan.expert_ids,
            route_plan.num_tokens_post_padded,
            route_count=route_plan.route_count,
            topk=route_plan.topk,
            block_size_m=route_plan.block_size_m,
            workspace=workspace,
        )
        if w13 is None:
            raise RuntimeError("stage breakdown W13 returned None")
        return w13

    gate, up = run_w13()

    def run_activation() -> torch.Tensor:
        activated = triton_dsv4.silu_and_mul_clamp_bf16(
            gate,
            up,
            swiglu_limit=swiglu_limit,
            weights=route_weights.reshape(-1, 1),
            workspace=workspace,
        )
        if activated is None:
            raise RuntimeError("stage breakdown activation returned None")
        return activated

    activated = run_activation()

    def run_w2() -> torch.Tensor:
        routed = triton_dsv4._grouped_fp4_linear(
            activated,
            tensors["w2_weight"],
            tensors["w2_scale"],
            route_plan.sorted_route_ids,
            route_plan.expert_ids,
            route_plan.num_tokens_post_padded,
            route_count=route_plan.route_count,
            topk=route_plan.topk,
            block_size_m=route_plan.block_size_m,
            slot=None,
            a_rows_are_routes=True,
            workspace=workspace,
        )
        if routed is None:
            raise RuntimeError("stage breakdown W2 returned None")
        return routed

    routed = run_w2()

    def run_route_sum() -> torch.Tensor:
        summed = triton_dsv4._sum_grouped_routes(
            routed,
            tokens=hidden_states.shape[0],
            hidden=hidden_states.shape[1],
            topk=route_plan.topk,
            workspace=workspace,
        )
        if summed is None:
            raise RuntimeError("stage breakdown route_sum returned None")
        return summed

    def run_total() -> torch.Tensor:
        local_gate, local_up = run_w13()
        local_activated = triton_dsv4.silu_and_mul_clamp_bf16(
            local_gate,
            local_up,
            swiglu_limit=swiglu_limit,
            weights=route_weights.reshape(-1, 1),
            workspace=workspace,
        )
        if local_activated is None:
            raise RuntimeError("stage breakdown total activation returned None")
        local_routed = triton_dsv4._grouped_fp4_linear(
            local_activated,
            tensors["w2_weight"],
            tensors["w2_scale"],
            route_plan.sorted_route_ids,
            route_plan.expert_ids,
            route_plan.num_tokens_post_padded,
            route_count=route_plan.route_count,
            topk=route_plan.topk,
            block_size_m=route_plan.block_size_m,
            slot=None,
            a_rows_are_routes=True,
            workspace=workspace,
        )
        if local_routed is None:
            raise RuntimeError("stage breakdown total W2 returned None")
        return triton_dsv4._sum_grouped_routes(
            local_routed,
            tokens=hidden_states.shape[0],
            hidden=hidden_states.shape[1],
            topk=route_plan.topk,
            workspace=workspace,
        )

    route_sum = run_route_sum()
    w13_ms = _time_cuda(run_w13, warmup=warmup, iters=iters)
    activation_ms = _time_cuda(run_activation, warmup=warmup, iters=iters)
    w2_ms = _time_cuda(run_w2, warmup=warmup, iters=iters)
    route_sum_ms = _time_cuda(run_route_sum, warmup=warmup, iters=iters)
    total_ms = _time_cuda(run_total, warmup=warmup, iters=iters)
    return {
        "status": "pass",
        "backend": dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4,
        "w13_ms": w13_ms,
        "activation_ms": activation_ms,
        "w2_ms": w2_ms,
        "route_sum_ms": route_sum_ms,
        "total_ms": total_ms,
        "w13_w2_ms": w13_ms + w2_ms,
        "stage_sum_ms": w13_ms + activation_ms + w2_ms + route_sum_ms,
        "output": route_sum,
    }


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
    old_route_env = os.environ.get("MINISGL_DSV4_SM80_MOE_ROUTE")
    old_v1_env = os.environ.get("MINISGL_DSV4_SM80_V1_MOE")
    old_v2_env = os.environ.get("MINISGL_DSV4_SM80_MOE_V2")
    old_backend_env = os.environ.get(dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV)
    os.environ.pop("MINISGL_DSV4_SM80_MOE_ROUTE", None)
    os.environ.pop(dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV, None)
    os.environ["MINISGL_DSV4_SM80_V1_MOE"] = "1"
    os.environ["MINISGL_DSV4_SM80_MOE_V2"] = "1"
    try:
        route_plan = dsv4_kernel.build_moe_route_plan(
            tensors["indices"],
            num_experts=tensors["w13_weight"].shape[0],
            block_size_m=16,
        )
        route_metadata_ms = _time_cuda(
            lambda: dsv4_kernel.build_moe_route_plan(
                tensors["indices"],
                num_experts=tensors["w13_weight"].shape[0],
                block_size_m=16,
            ),
            warmup=warmup,
            iters=iters,
        )
        grouped = dsv4_kernel.moe_route_dispatch_bf16_grouped(**tensors, swiglu_limit=swiglu_limit)
        if grouped is None:
            raise RuntimeError("grouped MoE route dispatch returned None")
        grouped_stage = _grouped_stage_breakdown(
            tensors,
            route_plan,
            swiglu_limit=swiglu_limit,
            warmup=warmup,
            iters=iters,
        )
        grouped_stage_output = grouped_stage.pop("output")
        grouped_ms = _time_cuda(
            lambda: dsv4_kernel.moe_route_dispatch_bf16_grouped(
                **tensors,
                swiglu_limit=swiglu_limit,
            ),
            warmup=warmup,
            iters=iters,
        )
        moe_v2_plan = dsv4_kernel.build_moe_v2_execution_plan(
            tensors["hidden_states"],
            tensors["weights"],
            tensors["indices"],
            num_experts=tensors["w13_weight"].shape[0],
            block_size_m=16,
        )
        v2_plan_ms = _time_cuda(
            lambda: dsv4_kernel.build_moe_v2_execution_plan(
                tensors["hidden_states"],
                tensors["weights"],
                tensors["indices"],
                num_experts=tensors["w13_weight"].shape[0],
                block_size_m=16,
            ),
            warmup=warmup,
            iters=iters,
        )
        workspace = dsv4_kernel.DSV4MoEWorkspace()
        v2_grouped = dsv4_kernel.moe_route_dispatch_bf16_grouped(
            **tensors,
            swiglu_limit=swiglu_limit,
            moe_plan=moe_v2_plan,
            workspace=workspace,
        )
        if v2_grouped is None:
            raise RuntimeError("V2 grouped MoE route dispatch returned None")

        def run_v2_grouped() -> torch.Tensor | None:
            plan = dsv4_kernel.build_moe_v2_execution_plan(
                tensors["hidden_states"],
                tensors["weights"],
                tensors["indices"],
                num_experts=tensors["w13_weight"].shape[0],
                block_size_m=16,
            )
            return dsv4_kernel.moe_route_dispatch_bf16_grouped(
                **tensors,
                swiglu_limit=swiglu_limit,
                moe_plan=plan,
                workspace=workspace,
            )

        v2_grouped_full_ms = _time_cuda(
            run_v2_grouped,
            warmup=warmup,
            iters=iters,
        )
        v2_grouped_dispatch_ms = _time_cuda(
            lambda: dsv4_kernel.moe_route_dispatch_bf16_grouped(
                **tensors,
                swiglu_limit=swiglu_limit,
                moe_plan=moe_v2_plan,
                workspace=workspace,
            ),
            warmup=warmup,
            iters=iters,
        )
        runner_prepare_ms = v2_plan_ms
        runner_experts_ms = v2_grouped_dispatch_ms
        runner_finalize_ms = _time_cuda(
            lambda: _runner_finalize_current_backend(v2_grouped),
            warmup=warmup,
            iters=iters,
        )
        runner_shared_ms = 0.0

        def run_runner_total() -> torch.Tensor:
            plan = dsv4_kernel.build_moe_v2_execution_plan(
                tensors["hidden_states"],
                tensors["weights"],
                tensors["indices"],
                num_experts=tensors["w13_weight"].shape[0],
                block_size_m=16,
            )
            routed = dsv4_kernel.moe_route_dispatch_bf16_grouped(
                **tensors,
                swiglu_limit=swiglu_limit,
                moe_plan=plan,
                workspace=workspace,
            )
            if routed is None:
                raise RuntimeError("runner grouped MoE route dispatch returned None")
            return _runner_finalize_current_backend(routed)

        runner_total_ms = _time_cuda(
            run_runner_total,
            warmup=warmup,
            iters=iters,
        )
    finally:
        if old_route_env is None:
            os.environ.pop("MINISGL_DSV4_SM80_MOE_ROUTE", None)
        else:
            os.environ["MINISGL_DSV4_SM80_MOE_ROUTE"] = old_route_env
        if old_v1_env is None:
            os.environ.pop("MINISGL_DSV4_SM80_V1_MOE", None)
        else:
            os.environ["MINISGL_DSV4_SM80_V1_MOE"] = old_v1_env
        if old_v2_env is None:
            os.environ.pop("MINISGL_DSV4_SM80_MOE_V2", None)
        else:
            os.environ["MINISGL_DSV4_SM80_MOE_V2"] = old_v2_env
        if old_backend_env is None:
            os.environ.pop(dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV, None)
        else:
            os.environ[dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_ENV] = old_backend_env

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
    v2_max_abs = (v2_grouped.float() - reference.float()).abs().max().item()
    v2_max_rel = ((v2_grouped.float() - reference.float()).abs() / denom).max().item()
    v2_vs_v1_max_abs = (v2_grouped.float() - grouped.float()).abs().max().item()
    grouped_stage_vs_grouped_max_abs = (
        (grouped_stage_output.float() - grouped.float()).abs().max().item()
    )
    current_vs_bf16 = (current.float() - reference.float()).abs().max().item()
    runner_stage_ms = {
        "backend": dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4,
        "prepare_ms": runner_prepare_ms,
        "w13_ms": grouped_stage["w13_ms"],
        "activation_ms": grouped_stage["activation_ms"],
        "w2_ms": grouped_stage["w2_ms"],
        "route_sum_ms": grouped_stage["route_sum_ms"],
        "experts_total_ms": grouped_stage["total_ms"],
        "finalize_ms": runner_finalize_ms,
        "shared_ms": runner_shared_ms,
        "total_ms": runner_total_ms,
        "w13_w2_ms": grouped_stage["w13_w2_ms"],
    }
    return {
        "name": name,
        **spec,
        "backend_comparison": {
            "current_grouped": {
                "backend": dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4,
                "status": "pass",
                "silent_fallback": False,
                "total_ms": grouped_ms,
            },
            "runner_07_36_current_grouped": {
                "backend": dsv4_kernel.DSV4_SM80_MOE_EXPERT_BACKEND_GROUPED_FP4,
                "status": "pass",
                "silent_fallback": False,
                "total_ms": runner_total_ms,
            },
            "marlin_mxfp4_w4a16": _marlin_candidate_report(),
        },
        "current_fallback_ms": current_ms,
        "bf16_reference_ms": ref_ms,
        "route_metadata_ms": route_metadata_ms,
        "v2_plan_ms": v2_plan_ms,
        "route_metadata_padded_tokens": int(route_plan.num_tokens_post_padded.item()),
        "grouped_stage_ms": grouped_stage,
        "runner_stage_ms": runner_stage_ms,
        "grouped_ms": grouped_ms,
        "v1_grouped_full_ms": grouped_ms,
        "v2_grouped_full_ms": v2_grouped_full_ms,
        "v2_grouped_dispatch_ms": v2_grouped_dispatch_ms,
        "runner_prepare_ms": runner_prepare_ms,
        "runner_experts_ms": runner_experts_ms,
        "runner_finalize_ms": runner_finalize_ms,
        "runner_shared_ms": runner_shared_ms,
        "runner_total_ms": runner_total_ms,
        "runner_shared_enabled": False,
        "v2_speedup_vs_v1_grouped_full": grouped_ms / v2_grouped_full_ms,
        "v2_dispatch_speedup_vs_v1_grouped_full": grouped_ms / v2_grouped_dispatch_ms,
        "speedup_vs_current_fallback": current_ms / grouped_ms,
        "speedup_vs_bf16_reference": ref_ms / grouped_ms,
        "grouped_vs_bf16_max_abs": max_abs,
        "grouped_vs_bf16_max_rel": max_rel,
        "v2_grouped_vs_bf16_max_abs": v2_max_abs,
        "v2_grouped_vs_bf16_max_rel": v2_max_rel,
        "v2_vs_v1_grouped_max_abs": v2_vs_v1_max_abs,
        "grouped_stage_vs_grouped_max_abs": grouped_stage_vs_grouped_max_abs,
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
    parser.add_argument(
        "--include-real-shapes",
        action="store_true",
        help="Also run heavier DSV4-like shapes for 07.3 artifact collection.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("This benchmark requires an sm80 CUDA device.")

    cases = {
        "decode_tiny": dict(tokens=4, topk=2, experts=8, hidden=64, intermediate=32),
        "decode_grouped": dict(tokens=8, topk=6, experts=16, hidden=256, intermediate=128),
        "prefill_grouped": dict(tokens=64, topk=6, experts=16, hidden=256, intermediate=128),
    }
    if args.include_real_shapes:
        cases.update(
            {
                "decode_real": dict(tokens=4, topk=6, experts=256, hidden=4096, intermediate=256),
                "prefill_real": dict(
                    tokens=4096, topk=6, experts=256, hidden=4096, intermediate=256
                ),
            }
        )
    results = [
        _run_case(name, spec, warmup=args.warmup, iters=args.iters) for name, spec in cases.items()
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

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


@contextlib.contextmanager
def dsv4_env(env: dict[str, str]):
    saved = {name: os.environ.get(name) for name in tuple(os.environ) if name.startswith("MINISGL_DSV4_SM80_")}
    for name in saved:
        os.environ.pop(name, None)
    os.environ.update(env)
    try:
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith("MINISGL_DSV4_SM80_"):
                os.environ.pop(name, None)
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


def time_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
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
    return float(start.elapsed_time(end) / iters)


def profile_cuda(fn: Callable[[], object]) -> dict[str, object]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        fn()
        torch.cuda.synchronize()
    kernel_names: list[str] = []
    for event in prof.events():
        device_type = str(getattr(event, "device_type", "")).lower()
        if "cuda" not in device_type:
            continue
        name = getattr(event, "name", "")
        if name:
            kernel_names.append(str(name))
    counts: dict[str, int] = {}
    for name in kernel_names:
        counts[name] = counts.get(name, 0) + 1
    return {
        "kernel_count": len(kernel_names),
        "kernel_name_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    rel = diff / denom
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99).item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


def run_variant(
    env: dict[str, str],
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    post_input: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    def pre_call():
        return dsv4_kernel.hc_pre_fallback(
            x,
            fn,
            scale,
            base,
            hc_mult=hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=eps,
            norm_eps=norm_eps,
        )

    with dsv4_env(env):
        y, post, comb = pre_call()

        def post_call():
            return dsv4_kernel.hc_post_fallback(post_input, x, post, comb)

        post_out = post_call()
        pre_ms = time_cuda(pre_call, warmup=warmup, iters=iters)
        post_ms = time_cuda(post_call, warmup=warmup, iters=iters)
        pre_profile = profile_cuda(pre_call)
        post_profile = profile_cuda(post_call)

    return {
        "env": env,
        "hc_pre_ms": pre_ms,
        "hc_post_ms": post_ms,
        "hc_pre_profile": pre_profile,
        "hc_post_profile": post_profile,
        "outputs": {
            "y": tensor_summary(y),
            "post": tensor_summary(post),
            "comb": tensor_summary(comb),
            "post_out": tensor_summary(post_out),
        },
        "tensors": {
            "y": y.detach(),
            "post": post.detach(),
            "comb": comb.detach(),
            "post_out": post_out.detach(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused DSV4 HC pre/post microbenchmark.")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_hc_elementwise_graph_cleanup/raw/focused_hc_microbench.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for focused HC microbench.")
    torch.manual_seed(6807)
    device = torch.device("cuda")
    mix_hc = (2 + args.hc_mult) * args.hc_mult
    x = torch.randn(args.tokens, args.hc_mult, args.hidden, device=device, dtype=torch.bfloat16)
    fn = (torch.randn(mix_hc, args.hc_mult * args.hidden, device=device) * 0.02).to(
        torch.bfloat16
    )
    scale = torch.tensor([0.15, 0.1, 0.08], device=device, dtype=torch.float32)
    base = (torch.randn(mix_hc, device=device) * 0.01).contiguous()
    post_input = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)

    current_env = {"MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE": "1"}
    candidate_env = {
        "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE": "1",
        dsv4_kernel.DSV4_SM80_HC_GRAPH_CLEANUP_TOGGLE: "1",
    }

    start_s = time.time()
    current = run_variant(
        current_env,
        x,
        fn,
        scale,
        base,
        post_input,
        hc_mult=args.hc_mult,
        sinkhorn_iters=args.sinkhorn_iters,
        eps=args.eps,
        norm_eps=args.norm_eps,
        warmup=args.warmup,
        iters=args.iters,
    )
    candidate = run_variant(
        candidate_env,
        x,
        fn,
        scale,
        base,
        post_input,
        hc_mult=args.hc_mult,
        sinkhorn_iters=args.sinkhorn_iters,
        eps=args.eps,
        norm_eps=args.norm_eps,
        warmup=args.warmup,
        iters=args.iters,
    )
    elapsed_s = time.time() - start_s

    errors = {
        name: error_summary(candidate["tensors"][name], current["tensors"][name])
        for name in ("y", "post", "comb", "post_out")
    }
    for section in (current, candidate):
        section.pop("tensors")

    payload = {
        "shape": {
            "tokens": args.tokens,
            "hc_mult": args.hc_mult,
            "hidden": args.hidden,
            "mix_hc": mix_hc,
            "sinkhorn_iters": args.sinkhorn_iters,
            "eps": args.eps,
            "norm_eps": args.norm_eps,
        },
        "timing": {"warmup": args.warmup, "iters": args.iters, "elapsed_s": elapsed_s},
        "current": current,
        "candidate": candidate,
        "candidate_vs_current_error": errors,
        "capabilities": dsv4_kernel.detect_dsv4_kernel_capabilities().__dict__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402
from minisgl.kernel.triton import deepseek_v4 as triton_dsv4  # noqa: E402


def _cuda_time_us(fn: Callable[[], torch.Tensor], *, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    last = None
    for _ in range(repeats):
        last = fn()
    end.record()
    torch.cuda.synchronize()
    if last is not None:
        _ = last.data_ptr()
    return float(start.elapsed_time(end) * 1000.0 / repeats)


def _bench_shape(
    *,
    m: int,
    n: int,
    k: int,
    repeats: int,
    warmup: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = (
        torch.randn(n, k, device=device, dtype=torch.float32)
        .clamp(-4, 4)
        .to(dsv4_kernel.fp8_dtype())
    )
    scale = torch.rand(
        dsv4_kernel.scale_dim(n),
        dsv4_kernel.scale_dim(k),
        device=device,
        dtype=torch.float32,
    ).to(dsv4_kernel.e8m0_dtype())
    cached_scale = scale.float().contiguous()

    raw = triton_dsv4.quantized_linear_fp8(x, weight, scale)
    cached = triton_dsv4.quantized_linear_fp8(x, weight, cached_scale)
    if raw is None or cached is None:
        return {
            "m": m,
            "n": n,
            "k": k,
            "status": "unsupported",
        }
    torch.cuda.synchronize()
    if not torch.allclose(raw, cached, atol=3e-2, rtol=3e-2):
        raise AssertionError(f"cached scale output mismatch for m={m} n={n} k={k}")

    convert_us = _cuda_time_us(lambda: scale.float().contiguous(), repeats=repeats, warmup=warmup)
    raw_us = _cuda_time_us(
        lambda: triton_dsv4.quantized_linear_fp8(x, weight, scale),
        repeats=repeats,
        warmup=warmup,
    )
    cached_us = _cuda_time_us(
        lambda: triton_dsv4.quantized_linear_fp8(x, weight, cached_scale),
        repeats=repeats,
        warmup=warmup,
    )
    delta_us = raw_us - cached_us
    return {
        "m": m,
        "n": n,
        "k": k,
        "scale_shape": list(scale.shape),
        "status": "pass",
        "scale_float_contiguous_us": convert_us,
        "raw_wrapper_us": raw_us,
        "cached_wrapper_us": cached_us,
        "delta_us": delta_us,
        "delta_pct_of_raw": delta_us / raw_us * 100.0 if raw_us > 0 else 0.0,
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    rows = payload.get("rows", [])
    lines = [
        "# TARGET 07.56 Static Scale Cache Microbench",
        "",
        f"- Status: `{payload['status']}`",
        f"- Repeats: `{payload['repeats']}`",
        f"- Warmup: `{payload['warmup']}`",
        "",
        "| M | N | K | scale shape | convert us | raw wrapper us | cached wrapper us | delta us | delta % |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "pass":
            continue
        lines.append(
            "| {m} | {n} | {k} | `{shape}` | `{convert:.4f}` | `{raw:.4f}` | "
            "`{cached:.4f}` | `{delta:.4f}` | `{pct:.2f}` |".format(
                m=row["m"],
                n=row["n"],
                k=row["k"],
                shape=tuple(row["scale_shape"]),
                convert=row["scale_float_contiguous_us"],
                raw=row["raw_wrapper_us"],
                cached=row["cached_wrapper_us"],
                delta=row["delta_us"],
                pct=row["delta_pct_of_raw"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT
        / "performance_milestones"
        / "target07_low_cost_graph_layout_compile_preflight"
        / "raw"
        / "static_scale_cache_microbench.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=ROOT
        / "performance_milestones"
        / "target07_low_cost_graph_layout_compile_preflight"
        / "summaries"
        / "static_scale_cache_microbench.md",
    )
    args = parser.parse_args()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        payload: dict[str, object] = {
            "status": "skipped",
            "reason": "requires an sm80 CUDA device",
            "repeats": args.repeats,
            "warmup": args.warmup,
            "rows": [],
        }
    else:
        os.environ.setdefault("MINISGL_DSV4_SM80_FP8_ACT_QUANT_TRITON", "1")
        payload = {
            "status": "pass",
            "device": torch.cuda.get_device_name(),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "rows": [
                _bench_shape(m=m, n=512, k=128, repeats=args.repeats, warmup=args.warmup)
                for m in (1, 4, 8, 16)
            ],
        }

    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(args.md_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

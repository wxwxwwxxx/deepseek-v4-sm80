#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

os.environ.setdefault("MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE", "1")

from minisgl.kernel import deepseek_v4 as dsv4_kernel  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines) + "\n"


def _target_row(run: dict[str, Any], scenario: str) -> int:
    for item in run.get("scenarios", []):
        if item.get("name") != scenario:
            continue
        labels = [str(row.get("label", "")) for row in item.get("probe_prompts", [])]
        if "target" in labels:
            return labels.index("target")
        for index, label in enumerate(labels):
            if label.startswith("target"):
                return index
    return 0


def _activation_entries(run_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(run_dir / "debug_trace" / "activations.rank0.jsonl")


def _find_activation(
    entries: list[dict[str, Any]],
    *,
    scenario: str,
    phase: str,
    name: str,
) -> dict[str, Any]:
    for entry in entries:
        if (
            entry.get("scenario") == scenario
            and entry.get("stage") == "probe"
            and entry.get("batch", {}).get("phase") == phase
            and entry.get("name") == name
        ):
            return entry
    raise KeyError(f"missing activation scenario={scenario} phase={phase} name={name}")


def _load_tensor(run_dir: Path, entry: dict[str, Any]) -> torch.Tensor:
    payload = torch.load(run_dir / "debug_trace" / entry["tensor_path"], map_location="cpu")
    tensor = payload.get("tensor")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"activation {entry.get('name')} did not contain a tensor")
    return tensor


def _config_params(model_path: Path) -> dict[str, Any]:
    config = _load_json(model_path / "config.json")
    rope_scaling = config.get("rope_scaling") or {}
    compress_ratios = list(config.get("compress_ratios") or [])
    ratio = int(compress_ratios[0]) if compress_ratios else 0
    return {
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "rotary_dim": int(config.get("qk_rope_head_dim") or config.get("head_dim")),
        "base": float(
            config.get("compress_rope_theta")
            if ratio and config.get("compress_rope_theta") is not None
            else config.get("rope_theta", 10000.0)
        ),
        "original_seq_len": int(rope_scaling.get("original_max_position_embeddings", 0) or 0)
        if ratio
        else 0,
        "factor": float(rope_scaling.get("factor", 1.0) or 1.0),
        "beta_fast": int(rope_scaling.get("beta_fast", 32) or 32),
        "beta_slow": int(rope_scaling.get("beta_slow", 1) or 1),
        "layer0_compress_ratio": ratio,
    }


def _reference_q_norm_rope(
    q: torch.Tensor,
    positions: torch.Tensor,
    params: dict[str, Any],
) -> torch.Tensor:
    out = q.clone()
    q_fp32 = out.float()
    scale = torch.rsqrt(q_fp32.square().mean(-1, keepdim=True) + params["rms_norm_eps"])
    out.copy_((q_fp32 * scale).to(out.dtype))
    rotary_dim = int(params["rotary_dim"])
    if rotary_dim <= 0:
        return out
    pos = positions.to(device=out.device, dtype=torch.float32)
    base = float(params["base"])
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=out.device)
            / rotary_dim
        )
    )
    original_seq_len = int(params["original_seq_len"])
    if original_seq_len > 0:
        factor = float(params["factor"])
        beta_fast = int(params["beta_fast"])
        beta_slow = int(params["beta_slow"])

        def correction_dim(num_rotations: float) -> float:
            return (
                rotary_dim
                * math.log(original_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rotary_dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(rotary_dim // 2, dtype=torch.float32, device=out.device) - low)
            / max(high - low, 1),
            0,
            1,
        )
        smooth = 1 - ramp
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    while cos.ndim < out[..., -rotary_dim:].ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rope = out[..., -rotary_dim:].float().unflatten(-1, (-1, 2))
    a, b = rope[..., 0], rope[..., 1]
    rotated = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    out[..., -rotary_dim:] = rotated.to(out.dtype)
    return out


def _active_q_norm_rope(
    q: torch.Tensor,
    positions: torch.Tensor,
    params: dict[str, Any],
) -> tuple[torch.Tensor, str]:
    out = q.clone()
    dsv4_kernel.q_norm_rope_fallback(
        out,
        positions,
        rms_norm_eps=params["rms_norm_eps"],
        rotary_dim=params["rotary_dim"],
        base=params["base"],
        original_seq_len=params["original_seq_len"],
        factor=params["factor"],
        beta_fast=params["beta_fast"],
        beta_slow=params["beta_slow"],
    )
    return out, "q_norm_rope_fallback_or_triton"


def _active_q_kv_norm_rope_store(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    params: dict[str, Any],
) -> tuple[torch.Tensor | None, str]:
    out_q = q.clone()
    out_kv = kv.clone()
    cache = torch.empty_like(out_kv)
    out_loc = torch.arange(out_kv.shape[0], dtype=torch.long, device=out_kv.device)
    norm_weight = torch.ones(out_kv.shape[-1], dtype=out_kv.dtype, device=out_kv.device)
    ok = dsv4_kernel.q_kv_norm_rope_cache_fallback(
        out_q,
        out_kv,
        positions,
        norm_weight=norm_weight,
        rms_norm_eps=params["rms_norm_eps"],
        cache=cache,
        out_loc=out_loc,
        rotary_dim=params["rotary_dim"],
        base=params["base"],
        original_seq_len=params["original_seq_len"],
        factor=params["factor"],
        beta_fast=params["beta_fast"],
        beta_slow=params["beta_slow"],
    )
    if not ok:
        return None, "unavailable_or_returned_false"
    return out_q, "q_kv_norm_rope_cache_bf16"


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"available": True, "shape_a": list(a.shape), "shape_b": list(b.shape), "max_abs": "shape"}
    delta = (a.float() - b.float()).abs()
    return {
        "available": True,
        "allclose_zero": bool(torch.equal(a, b)),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "shape": list(a.shape),
    }


def _fmt(stats: dict[str, Any] | None) -> str:
    if not stats or not stats.get("available"):
        return "n/a"
    if stats.get("max_abs") == "shape":
        return f"shape {stats.get('shape_a')} vs {stats.get('shape_b')}"
    return f"max={stats.get('max_abs', 0.0):.6g}, mean={stats.get('mean_abs', 0.0):.6g}"


def _make_layout(
    *,
    target_q: torch.Tensor,
    target_kv: torch.Tensor,
    filler_q: torch.Tensor,
    filler_kv: torch.Tensor,
    slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = filler_q.clone()
    kv = filler_kv.clone()
    q[slot].copy_(target_q)
    kv[slot].copy_(target_kv)
    return q, kv


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the q_norm_rope same-input microbench.")
    run_dir = Path(args.run_dir)
    run_payload = _load_json(run_dir / "run.json")
    entries = _activation_entries(run_dir)
    params = _config_params(Path(args.model_path))
    device = torch.device(args.device)

    single_q = _load_tensor(
        run_dir,
        _find_activation(
            entries,
            scenario="single_target_alone",
            phase="prefill",
            name="layer0.q_wqb_output",
        ),
    )[0].to(device=device, dtype=torch.bfloat16)
    single_kv = _load_tensor(
        run_dir,
        _find_activation(
            entries,
            scenario="single_target_alone",
            phase="prefill",
            name="layer0.wkv_output",
        ),
    )[0].to(device=device, dtype=torch.bfloat16)
    captured_q_after = _load_tensor(
        run_dir,
        _find_activation(
            entries,
            scenario="single_target_alone",
            phase="prefill",
            name="layer0.q_after_q_norm_rope",
        ),
    )[0].to(device=device, dtype=torch.bfloat16)

    filler_source = args.filler_source
    filler_q = _load_tensor(
        run_dir,
        _find_activation(
            entries,
            scenario=filler_source,
            phase="prefill",
            name="layer0.q_wqb_output",
        ),
    ).to(device=device, dtype=torch.bfloat16)
    filler_kv = _load_tensor(
        run_dir,
        _find_activation(
            entries,
            scenario=filler_source,
            phase="prefill",
            name="layer0.wkv_output",
        ),
    ).to(device=device, dtype=torch.bfloat16)
    if filler_q.shape[0] < 4 or filler_kv.shape[0] < 4:
        raise RuntimeError(f"filler_source {filler_source} did not capture four selected rows")

    pos_single = torch.tensor([args.position], dtype=torch.long, device=device)
    pos_batch = torch.full((4,), args.position, dtype=torch.long, device=device)
    q_single = single_q.unsqueeze(0)
    kv_single = single_kv.unsqueeze(0)

    ref_single = _reference_q_norm_rope(q_single, pos_single, params)
    active_single, active_q_backend = _active_q_norm_rope(q_single, pos_single, params)
    fused_single, fused_backend = _active_q_kv_norm_rope_store(q_single, kv_single, pos_single, params)

    rows: list[dict[str, Any]] = []
    table_rows: list[list[Any]] = []
    for slot in range(4):
        q_batch, kv_batch = _make_layout(
            target_q=single_q,
            target_kv=single_kv,
            filler_q=filler_q,
            filler_kv=filler_kv,
            slot=slot,
        )
        ref_batch = _reference_q_norm_rope(q_batch, pos_batch, params)
        active_batch, _ = _active_q_norm_rope(q_batch, pos_batch, params)
        fused_batch, _ = _active_q_kv_norm_rope_store(q_batch, kv_batch, pos_batch, params)

        row = {
            "target_slot": slot,
            "reference_single_vs_batch": _diff(ref_single[0], ref_batch[slot]),
            "active_q_norm_single_vs_batch": _diff(active_single[0], active_batch[slot]),
            "active_q_norm_vs_reference_single": _diff(active_single[0], ref_single[0]),
            "active_q_norm_vs_reference_batch": _diff(active_batch[slot], ref_batch[slot]),
            "captured_single_vs_reference": _diff(captured_q_after, ref_single[0]),
            "fused_q_kv_backend": fused_backend,
            "fused_q_kv_single_vs_batch": (
                _diff(fused_single[0], fused_batch[slot])
                if fused_single is not None and fused_batch is not None
                else {"available": False}
            ),
            "fused_q_kv_vs_reference_single": (
                _diff(fused_single[0], ref_single[0]) if fused_single is not None else {"available": False}
            ),
            "fused_q_kv_vs_reference_batch": (
                _diff(fused_batch[slot], ref_batch[slot])
                if fused_batch is not None
                else {"available": False}
            ),
        }
        rows.append(row)
        table_rows.append(
            [
                slot,
                _fmt(row["reference_single_vs_batch"]),
                _fmt(row["active_q_norm_single_vs_batch"]),
                _fmt(row["fused_q_kv_single_vs_batch"]),
                _fmt(row["active_q_norm_vs_reference_batch"]),
                _fmt(row["fused_q_kv_vs_reference_batch"]),
            ]
        )

    payload = {
        "run_dir": str(run_dir),
        "model_path": args.model_path,
        "device": args.device,
        "params": params,
        "env": {
            "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE": os.environ.get(
                "MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE", ""
            ),
            "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES": os.environ.get(
                "MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES", ""
            ),
        },
        "active_q_backend": active_q_backend,
        "fused_q_kv_backend": fused_backend,
        "q_norm_rope_triton_enabled": dsv4_kernel.dsv4_sm80_triton_enabled(
            "MINISGL_DSV4_SM80_Q_NORM_ROPE"
        ),
        "fused_q_kv_norm_rope_store_enabled": dsv4_kernel.dsv4_sm80_triton_enabled(
            "MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE"
        ),
        "input_source": {
            "target": "single_target_alone layer0.q_wqb_output/wkv_output selected row 0",
            "filler_source": filler_source,
            "position": args.position,
        },
        "rows": rows,
        "run_scenarios": [item.get("name") for item in run_payload.get("scenarios", [])],
    }

    out_dir = Path(args.output_dir)
    _write_json(out_dir / "same_input_q_norm_rope_microbench.json", payload)
    _write_text(
        out_dir / "same_input_q_norm_rope_microbench.md",
        _markdown_table(
            [
                "Target slot",
                "Reference single-vs-batch",
                "Active q_norm single-vs-batch",
                "Fused q_kv single-vs-batch",
                "Active q_norm vs ref",
                "Fused q_kv vs ref",
            ],
            table_rows,
        ),
    )
    print(json.dumps({"output": str(out_dir), "status": "pass"}))
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TARGET 08.197 same-input q_norm_rope microbench.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="/models/DeepSeek-V4-Flash")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--position", type=int, default=256)
    parser.add_argument("--filler-source", default="target_slot0_fixed_fillers")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

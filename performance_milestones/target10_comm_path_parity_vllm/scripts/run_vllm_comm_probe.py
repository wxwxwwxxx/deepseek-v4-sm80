#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import vllm_comm_probe


def _load_target7_runner(path: Path):
    spec = importlib.util.spec_from_file_location("target7_vllm_matrix_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _aggregate_worker_payloads(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    by_label: dict[str, dict[str, Any]] = {}
    by_op: dict[str, dict[str, Any]] = {}
    for rank, payload in enumerate(payloads):
        for entry in payload.get("entries", []):
            key = (
                entry.get("phase"),
                entry.get("label"),
                entry.get("boundary"),
                entry.get("op"),
                entry.get("dtype"),
                tuple(entry.get("shape") or []),
                tuple(entry.get("output_shape") or []),
                entry.get("capture_state"),
                entry.get("group"),
                str(entry.get("dim")),
                str(entry.get("dst")),
            )
            out = entries.get(key)
            if out is None:
                out = dict(entry)
                out["shape"] = list(entry.get("shape") or [])
                out["output_shape"] = list(entry.get("output_shape") or [])
                out["count"] = 0
                out["bytes"] = 0
                out["elapsed_us"] = 0.0
                out["ranks"] = []
                entries[key] = out
            out["count"] += int(entry.get("count") or 0)
            out["bytes"] += int(entry.get("bytes") or 0)
            out["elapsed_us"] += float(entry.get("elapsed_us") or 0.0)
            out["ranks"].append(rank)
    for entry in entries.values():
        for bucket, key in ((by_label, entry["label"]), (by_op, entry["op"])):
            summary = bucket.setdefault(str(key), {"count": 0, "bytes": 0, "elapsed_us": 0.0})
            summary["count"] += int(entry.get("count") or 0)
            summary["bytes"] += int(entry.get("bytes") or 0)
            summary["elapsed_us"] += float(entry.get("elapsed_us") or 0.0)
    return {
        "entries": sorted(
            entries.values(),
            key=lambda item: (
                str(item.get("phase")),
                str(item.get("label")),
                str(item.get("op")),
                str(item.get("dtype")),
                str(item.get("shape")),
            ),
        ),
        "by_label": dict(sorted(by_label.items())),
        "by_op": dict(sorted(by_op.items())),
        "total_count": int(sum(int(entry.get("count") or 0) for entry in entries.values())),
        "total_bytes": int(sum(int(entry.get("bytes") or 0) for entry in entries.values())),
        "worker_payloads": list(payloads),
    }


def _write_probe_report(output_dir: Path, payload: dict[str, Any]) -> None:
    path = output_dir / "communication_probe.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    script_dir = Path(__file__).resolve().parent
    default_runner = (
        script_dir.parents[1] / "vllm" / "scripts" / "run_vllm_deepseek_v4_matrix.py"
    )
    runner_path = default_runner
    if "--target7-runner" in argv:
        idx = argv.index("--target7-runner")
        runner_path = Path(argv[idx + 1]).resolve()
        del argv[idx : idx + 2]

    runner = _load_target7_runner(runner_path)
    args = runner.parse_args(argv)
    output_dir = Path(args.output_dir)
    original_run_repeat = runner.run_repeat
    probe_runs: list[dict[str, Any]] = []

    vllm_comm_probe.install()

    def run_repeat_with_probe(**kwargs):
        llm = kwargs["llm"]
        scenario = kwargs["scenario"]
        repeat_index = int(kwargs["repeat_index"])
        phase = "warmup" if repeat_index < 0 else "repeat"
        phase_name = f"{scenario.name}:{phase}:{abs(repeat_index)}"
        llm.collective_rpc(vllm_comm_probe.reset_worker, args=(phase_name,))
        result = original_run_repeat(**kwargs)
        worker_payloads = llm.collective_rpc(vllm_comm_probe.snapshot_worker)
        aggregate = _aggregate_worker_payloads(worker_payloads)
        run_payload = {
            "scenario": scenario.name,
            "repeat_index": repeat_index,
            "phase": phase_name,
            "communication": aggregate,
        }
        probe_runs.append(run_payload)
        result["communication_probe"] = {
            "total_count": aggregate["total_count"],
            "total_bytes": aggregate["total_bytes"],
            "by_label": aggregate["by_label"],
            "entries": aggregate["entries"],
        }
        _write_probe_report(
            output_dir,
            {
                "runner_path": str(runner_path),
                "runs": probe_runs,
            },
        )
        return result

    runner.run_repeat = run_repeat_with_probe
    raise SystemExit(runner.run_matrix(args))


if __name__ == "__main__":
    main()

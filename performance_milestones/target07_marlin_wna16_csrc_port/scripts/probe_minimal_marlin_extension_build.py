from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = Path("/workspace/vllm-dsv4-docker")
DEFAULT_SOURCE_ROOT = ROOT / "python/minisgl/kernel/csrc/vendor/vllm_marlin_wna16"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    return {
        "command": command,
        "cwd": cwd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - started,
        "stdout_tail": proc.stdout[-12000:],
        "stderr_tail": proc.stderr[-12000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attempt a minimal mini-owned-style Marlin WNA16 extension build."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_marlin_wna16_csrc_port/raw/"
        / "minimal_marlin_extension_build.json",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT
        / "performance_milestones/target07_marlin_wna16_csrc_port/raw/"
        / "torch_extension_build",
    )
    parser.add_argument("--extension-name", default="minisgl_marlin_wna16_probe")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Mini-owned vendored Marlin source root with core/, quantization/, and moe/.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--max-jobs", default="8")
    parser.add_argument(
        "--moe-kernels",
        choices=["dsv4_bf16_fe2m1f", "all_sm80"],
        default="all_sm80",
        help=(
            "Generated MoE Marlin kernel instantiations to compile. "
            "The ops.cu selector references the full SM80 set unless narrowed."
        ),
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "target": "TARGET_07.391_dsv4_sm80_marlin_wna16_csrc_port",
        "purpose": "minimal source-level build probe; does not import vLLM runtime",
        "source_root": args.source_root,
        "upstream_reference_root": VLLM_ROOT,
        "extension_name": args.extension_name,
        "status": "unknown",
    }

    if args.clean and args.build_dir.exists():
        shutil.rmtree(args.build_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    schema_cpp = args.source_root / "schema.cpp"

    if args.moe_kernels == "dsv4_bf16_fe2m1f":
        moe_kernel_sources = [
            args.source_root / "moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu"
        ]
    else:
        moe_kernel_sources = sorted(
            (args.source_root / "moe/marlin_moe_wna16").glob("sm80_kernel_*.cu")
        )

    sources = [
        schema_cpp,
        args.source_root / "quantization/marlin/gptq_marlin_repack.cu",
        args.source_root / "moe/marlin_moe_wna16/ops.cu",
        *moe_kernel_sources,
    ]
    include_dirs = [
        args.source_root,
        args.source_root / "moe",
        args.source_root / "quantization",
    ]
    report["sources"] = sources
    report["moe_kernels"] = args.moe_kernels
    report["include_dirs"] = include_dirs
    report["source_exists"] = {str(path): path.exists() for path in sources}

    loader = args.build_dir / "load_extension.py"
    loader.write_text(
        f"""from pathlib import Path
from torch.utils.cpp_extension import load

name = {args.extension_name!r}
sources = {[str(path) for path in sources]!r}
include_dirs = {[str(path) for path in include_dirs]!r}

module = load(
    name=name,
    sources=sources,
    extra_include_paths=include_dirs,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-static-global-template-stub=false",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_80,code=compute_80",
    ],
    build_directory=str(Path({str(args.build_dir)!r}).resolve()),
    verbose=True,
    with_cuda=True,
)
print(module)
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["MAX_JOBS"] = str(args.max_jobs)
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.0+PTX")

    try:
        result = _run([sys.executable, str(loader)], cwd=ROOT, env=env)
        report["build"] = result
        report["status"] = "pass" if result["returncode"] == 0 else "error"
    except BaseException as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc(limit=20)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash

setup_vllm_runtime_env() {
  local z3_lib_dir

  z3_lib_dir="$(
    python - <<'PY'
from pathlib import Path

try:
    import z3
except Exception:
    raise SystemExit(0)

lib_dir = Path(z3.__file__).resolve().parent / "lib"
if lib_dir.is_dir():
    print(lib_dir)
PY
  )"

  if [[ -n "${z3_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${z3_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  if [[ "${VLLM_TILELANG_PREFLIGHT:-1}" != "0" ]]; then
    python - <<'PY'
import ctypes
from pathlib import Path

import tilelang

lib_path = Path(tilelang.__file__).resolve().parent / "lib" / "libtvm.so"
if not lib_path.exists():
    raise SystemExit(f"missing TileLang TVM library: {lib_path}")

ctypes.CDLL(str(lib_path))
print(f"TileLang preflight ok: {lib_path}")
PY
  fi
}

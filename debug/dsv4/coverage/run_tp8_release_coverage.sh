#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
OUT="${OUT:-/tmp/minisgl_dsv4_release_coverage}"
COVERAGE_BIN="${COVERAGE_BIN:-$(command -v coverage || true)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(command -v torchrun || true)}"

if [[ -z "${COVERAGE_BIN}" ]]; then
  echo "coverage is unavailable; install the project dev dependencies" >&2
  exit 2
fi
if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "torchrun is unavailable in the active Python environment" >&2
  exit 2
fi
if [[ -d "${OUT}" ]] && [[ -n "$(find "${OUT}" -mindepth 1 -print -quit)" ]]; then
  echo "coverage output directory must be empty: ${OUT}" >&2
  exit 2
fi

mkdir -p "${OUT}/logs"
cd "${ROOT}"

set +e
COVERAGE_FILE="${OUT}/.coverage" "${TORCHRUN_BIN}" \
  --standalone \
  --nproc-per-node="${TP_SIZE:-8}" \
  --log-dir "${OUT}/logs" \
  --redirects 3 \
  --tee 0 \
  --no-python \
  "${COVERAGE_BIN}" run \
  --branch \
  --parallel-mode \
  --source="${ROOT}/python/minisgl" \
  debug/dsv4/benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path "${MODEL_PATH}" \
  --recipe "${RECIPE:-low_m64}" \
  --max-running-req "${MAX_RUNNING_REQ:-4}" \
  --max-seq-len "${MAX_SEQ_LEN:-1024}" \
  --max-extend-tokens "${MAX_EXTEND_TOKENS:-1024}" \
  --num-pages "${NUM_PAGES:-64}" \
  --max-tokens "${MAX_TOKENS:-8}" \
  --output "${OUT}/smoke.json"
smoke_status=$?
set -e

COVERAGE_FILE="${OUT}/.coverage" "${COVERAGE_BIN}" combine "${OUT}"
COVERAGE_FILE="${OUT}/.coverage" "${COVERAGE_BIN}" json \
  --pretty-print \
  -o "${OUT}/coverage.json"
COVERAGE_FILE="${OUT}/.coverage" "${COVERAGE_BIN}" report \
  --show-missing \
  --include='*/python/minisgl/engine/*,*/python/minisgl/models/deepseek_v4.py,*/python/minisgl/attention/deepseek_v4.py,*/python/minisgl/kernel/deepseek_v4.py,*/python/minisgl/scheduler/*,*/python/minisgl/kvcache/*' \
  | tee "${OUT}/coverage.txt"

if [[ "${smoke_status}" -ne 0 ]]; then
  echo "runtime smoke failed with status ${smoke_status}; coverage artifacts were preserved" >&2
fi
exit "${smoke_status}"

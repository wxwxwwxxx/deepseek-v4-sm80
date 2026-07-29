#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${OUT:-/tmp/minisgl_dsv4_lifecycle_coverage}"
COVERAGE_BIN="${COVERAGE_BIN:-$(command -v coverage || true)}"

if [[ -z "${COVERAGE_BIN}" ]]; then
  echo "coverage is unavailable; install the project dev dependencies" >&2
  exit 2
fi
if [[ -d "${OUT}" ]] && [[ -n "$(find "${OUT}" -mindepth 1 -print -quit)" ]]; then
  echo "coverage output directory must be empty: ${OUT}" >&2
  exit 2
fi

mkdir -p "${OUT}"
cd "${ROOT}"

set +e
COVERAGE_FILE="${OUT}/.coverage" "${COVERAGE_BIN}" run \
  --branch \
  --context=dsv4-release-lifecycle \
  --source="${ROOT}/python/minisgl" \
  -m pytest \
  --no-cov \
  -q \
  tests/core/test_cache_allocate.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/core/test_chunked_prefill_lifecycle.py \
  tests/core/test_long_prefill_fairness.py \
  tests/core/test_mixed_phase_fair_policy.py \
  tests/core/test_scheduler.py \
  tests/attention/test_deepseek_v4_backend_metadata.py
test_status=$?
set -e

COVERAGE_FILE="${OUT}/.coverage" "${COVERAGE_BIN}" json \
  --pretty-print \
  -o "${OUT}/coverage.json"
COVERAGE_FILE="${OUT}/.coverage" "${COVERAGE_BIN}" report \
  --show-missing \
  --include='*/python/minisgl/attention/deepseek_v4.py,*/python/minisgl/kvcache/*,*/python/minisgl/scheduler/cache.py,*/python/minisgl/scheduler/prefill.py' \
  | tee "${OUT}/coverage.txt"

if [[ "${test_status}" -ne 0 ]]; then
  echo "lifecycle tests failed with status ${test_status}; coverage artifacts were preserved" >&2
fi
exit "${test_status}"

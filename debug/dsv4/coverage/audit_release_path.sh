#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

status=0

check_absent() {
  local label="$1"
  local pattern="$2"
  shift 2
  local matches
  matches="$(rg -n "${pattern}" "$@" --glob '*.py' || true)"
  if [[ -n "${matches}" ]]; then
    echo "FAIL: ${label}" >&2
    echo "${matches}" >&2
    status=1
  fi
}

check_absent \
  "obsolete DSV4 cache topology selectors remain in production" \
  'enable_dsv4_(radix_prefix_cache|component_loc_ownership|swa_independent_lifecycle)' \
  python/minisgl
check_absent \
  "CUDA Graph capture fail-open remains in production" \
  'cuda_graph_capture_fail_open|capture_fail_open' \
  python/minisgl
check_absent \
  "legacy host-copy graph replay helpers remain in production" \
  '_copy_metadata_for_replay|copy_decode_metadata_for_replay|copy_component_write_locs_for_replay|direct_decode_index_metadata_for_replay' \
  python/minisgl
check_absent \
  "production imports debug helpers" \
  '(^|[[:space:]])(from|import)[[:space:]]+debug([.]|[[:space:]]|$)' \
  python/minisgl

if [[ "${status}" -eq 0 ]]; then
  echo "DSV4 release-path static audit: PASS"
fi
exit "${status}"

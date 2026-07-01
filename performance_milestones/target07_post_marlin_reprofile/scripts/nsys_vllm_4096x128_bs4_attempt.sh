#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
TARGET_MILESTONE_DIR="${TARGET_MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_post_marlin_reprofile}"
VLLM_MILESTONE_DIR="${VLLM_MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target07392_nsys_vllm_4096x128_bs4}"
NSYS_BASE="${NSYS_BASE:-/tmp/nsys_target07392_vllm_4096x128_bs4}"

mkdir -p "${TARGET_MILESTONE_DIR}/raw" "${TARGET_MILESTONE_DIR}/summaries"

MILESTONE_DIR="${VLLM_MILESTONE_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
NSYS_BASE="${NSYS_BASE}" \
"${VLLM_MILESTONE_DIR}/scripts/nsys_vllm_4096x128_bs4.sh" "$@"

for artifact in "${OUTPUT_DIR}" "${NSYS_BASE}.nsys-rep" "${NSYS_BASE}.sqlite"; do
  [[ -e "${artifact}" ]] || continue
  link_name="${TARGET_MILESTONE_DIR}/raw/$(basename "${artifact}")"
  rm -rf "${link_name}"
  ln -s "${artifact}" "${link_name}"
  echo "linked ${artifact}"
done

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  cp "${OUTPUT_DIR}/summary.json" \
    "${TARGET_MILESTONE_DIR}/summaries/vllm_4096x128_bs4_nsys_summary.json"
fi

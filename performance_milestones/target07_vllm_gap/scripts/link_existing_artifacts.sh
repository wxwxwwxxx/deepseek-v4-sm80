#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_vllm_gap}"

mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"

link_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    rm -rf "${MILESTONE_DIR}/raw/$(basename "${path}")"
    ln -s "${path}" "${MILESTONE_DIR}/raw/$(basename "${path}")"
    echo "linked ${path}"
  fi
}

copy_small_summary() {
  local src="$1"
  local dst="$2"
  if [[ -d "${src}" ]]; then
    rm -rf "${MILESTONE_DIR}/summaries/${dst}"
    mkdir -p "${MILESTONE_DIR}/summaries/${dst}"
    for name in run_config.json summary.json matrix.jsonl; do
      if [[ -f "${src}/${name}" ]]; then
        cp "${src}/${name}" "${MILESTONE_DIR}/summaries/${dst}/${name}"
      fi
    done
    echo "copied small summary ${src}"
  fi
}

link_if_exists /tmp/dsv4_v1_moe_4096x1024_bs4
link_if_exists /tmp/dsv4_nsys_mini_v1_4096x128_bs4
link_if_exists /tmp/nsys_mini_v1_moe_4096x128_bs4.nsys-rep
link_if_exists /tmp/nsys_mini_v1_moe_4096x128_bs4.sqlite
link_if_exists /tmp/dsv4_nsys_vllm_4096x128_bs4
link_if_exists /tmp/nsys_vllm_4096x128_bs4.nsys-rep
link_if_exists /tmp/nsys_vllm_4096x128_bs4.sqlite

copy_small_summary "${MINISGL_ROOT}/performance_milestones/v1_moe/summaries/dsv4_v1_moe_4096x1024_bs4" \
  existing_mini_v1_moe_4096x1024_bs4_warmup0
copy_small_summary "${MINISGL_ROOT}/performance_milestones/v1_moe/summaries/dsv4_nsys_mini_v1_4096x128_bs4" \
  existing_mini_v1_moe_4096x128_bs4_warmup0
copy_small_summary /tmp/dsv4_nsys_vllm_4096x128_bs4 \
  existing_vllm_4096x128_bs4_warmup1

if [[ -f /tmp/nsys_vllm_4096x128_bs4.sqlite ]]; then
  python "${MILESTONE_DIR}/scripts/summarize_nsys_sqlite.py" \
    /tmp/nsys_vllm_4096x128_bs4.sqlite \
    --output-json "${MILESTONE_DIR}/summaries/existing_nsys_vllm_4096x128_bs4.json" \
    --output-md "${MILESTONE_DIR}/summaries/existing_nsys_vllm_4096x128_bs4.md" \
    --nvtx-window "repeat:decode_throughput_bs8:0" || true
fi

if [[ -f /tmp/nsys_mini_v1_moe_4096x128_bs4.sqlite ]]; then
  python "${MILESTONE_DIR}/scripts/summarize_nsys_sqlite.py" \
    /tmp/nsys_mini_v1_moe_4096x128_bs4.sqlite \
    --output-json "${MILESTONE_DIR}/summaries/existing_nsys_mini_v1_moe_4096x128_bs4.json" \
    --output-md "${MILESTONE_DIR}/summaries/existing_nsys_mini_v1_moe_4096x128_bs4.md" \
    --nvtx-window "repeat:decode_throughput_bs8:0" \
    --lite || true
fi

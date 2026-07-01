#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_vllm_gap}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target07_mini_v1_4096x1024_bs4_warmup1}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/usr/local/bin/torchrun}"
NPROC="${NPROC:-8}"
MAX_EXTEND_TOKENS="${MAX_EXTEND_TOKENS:-}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cmd=(
  "${TORCHRUN_BIN}" --standalone "--nproc_per_node=${NPROC}"
  benchmark/offline/deepseek_v4_perf_matrix.py
  --model-path "${MODEL_PATH}"
  --variants v1_moe
  --scenarios decode_throughput_bs8
  --prompt-len 4096
  --decode-len 1024
  --batch-size 4
  --repeats 1
  --warmup-repeats 1
  --page-size 256
  --output-dir "${OUTPUT_DIR}"
  --keep-going
)

if [[ -n "${MAX_EXTEND_TOKENS}" ]]; then
  cmd+=(--max-extend-tokens "${MAX_EXTEND_TOKENS}")
fi
cmd+=("$@")

cd "${MINISGL_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"

mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"
if [[ -e "${OUTPUT_DIR}" ]]; then
  rm -rf "${MILESTONE_DIR}/raw/$(basename "${OUTPUT_DIR}")"
  ln -s "${OUTPUT_DIR}" "${MILESTONE_DIR}/raw/$(basename "${OUTPUT_DIR}")"

  summary_dir="${MILESTONE_DIR}/summaries/$(basename "${OUTPUT_DIR}")"
  rm -rf "${summary_dir}"
  mkdir -p "${summary_dir}"
  for name in run_config.json summary.json matrix.jsonl; do
    if [[ -f "${OUTPUT_DIR}/${name}" ]]; then
      cp "${OUTPUT_DIR}/${name}" "${summary_dir}/${name}"
    fi
  done
fi

echo "mini fair macro output: ${OUTPUT_DIR}"

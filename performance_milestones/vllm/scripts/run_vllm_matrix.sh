#!/usr/bin/env bash
set -euo pipefail

MILESTONE_DIR="${MILESTONE_DIR:-/workspace/mini-sglang/performance_milestones/vllm}"
VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm-dsv4-docker}"
VLLM_VENV="${VLLM_VENV:-/workspace/venvs/vllm-dsv4}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_vllm_matrix_tp8}"
TP_SIZE="${TP_SIZE:-8}"
EFFECTIVE_OUTPUT_DIR="${OUTPUT_DIR}"
IS_DRY_RUN=0

ARGS=("$@")
for ((idx = 0; idx < ${#ARGS[@]}; idx++)); do
  case "${ARGS[$idx]}" in
    --dry-run)
      IS_DRY_RUN=1
      ;;
    --output-dir)
      if ((idx + 1 < ${#ARGS[@]})); then
        EFFECTIVE_OUTPUT_DIR="${ARGS[$((idx + 1))]}"
      fi
      ;;
    --output-dir=*)
      EFFECTIVE_OUTPUT_DIR="${ARGS[$idx]#--output-dir=}"
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${VLLM_VENV}/bin/activate"
source "${MILESTONE_DIR}/scripts/vllm_env.sh"
setup_vllm_runtime_env

cd "${VLLM_ROOT}"

python "${MILESTONE_DIR}/scripts/run_vllm_deepseek_v4_matrix.py" \
  --model-path "${MODEL_PATH}" \
  --vllm-root "${VLLM_ROOT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --block-size 256 \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

mkdir -p "${MILESTONE_DIR}/raw"
if [[ "${IS_DRY_RUN}" -eq 0 && -e "${EFFECTIVE_OUTPUT_DIR}" ]]; then
  rm -rf "${MILESTONE_DIR}/raw/$(basename "${EFFECTIVE_OUTPUT_DIR}")"
  ln -s "${EFFECTIVE_OUTPUT_DIR}" "${MILESTONE_DIR}/raw/$(basename "${EFFECTIVE_OUTPUT_DIR}")"
fi

echo "vLLM matrix output: ${EFFECTIVE_OUTPUT_DIR}"
if [[ "${IS_DRY_RUN}" -eq 0 && -e "${MILESTONE_DIR}/raw/$(basename "${EFFECTIVE_OUTPUT_DIR}")" ]]; then
  echo "Milestone raw symlink: ${MILESTONE_DIR}/raw/$(basename "${EFFECTIVE_OUTPUT_DIR}")"
fi

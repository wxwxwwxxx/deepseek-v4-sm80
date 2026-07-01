#!/usr/bin/env bash
set -euo pipefail

MILESTONE_DIR="${MILESTONE_DIR:-/workspace/mini-sglang/performance_milestones/vllm}"
VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm-dsv4-docker}"
VLLM_VENV="${VLLM_VENV:-/workspace/venvs/vllm-dsv4}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_nsys_vllm_4096x128_bs4}"
NSYS_BASE="${NSYS_BASE:-/tmp/nsys_vllm_4096x128_bs4}"
TP_SIZE="${TP_SIZE:-8}"
EFFECTIVE_OUTPUT_DIR="${OUTPUT_DIR}"

ARGS=("$@")
for ((idx = 0; idx < ${#ARGS[@]}; idx++)); do
  case "${ARGS[$idx]}" in
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

nsys profile \
  -t cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cudabacktrace=none \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  -o "${NSYS_BASE}" \
  python "${MILESTONE_DIR}/scripts/run_vllm_deepseek_v4_matrix.py" \
    --model-path "${MODEL_PATH}" \
    --vllm-root "${VLLM_ROOT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --block-size 256 \
    --output-dir "${OUTPUT_DIR}" \
    --scenarios decode_throughput_bs8 \
    --prompt-len 4096 \
    --decode-len 128 \
    --batch-size 4 \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
    --enable-chunked-prefill \
    --cudagraph-capture-sizes 1,2,4 \
    --max-cudagraph-capture-size 4 \
    --repeats 1 \
    --warmup-repeats 1 \
    "$@"

if [[ -f "${NSYS_BASE}.nsys-rep" ]]; then
  nsys export --type sqlite --force-overwrite=true \
    --output "${NSYS_BASE}.sqlite" \
    "${NSYS_BASE}.nsys-rep" || true
fi

mkdir -p "${MILESTONE_DIR}/raw"
for artifact in "${EFFECTIVE_OUTPUT_DIR}" "${NSYS_BASE}.nsys-rep" "${NSYS_BASE}.sqlite"; do
  if [[ -e "${artifact}" ]]; then
    rm -rf "${MILESTONE_DIR}/raw/$(basename "${artifact}")"
    ln -s "${artifact}" "${MILESTONE_DIR}/raw/$(basename "${artifact}")"
    echo "Linked ${artifact}"
  fi
done

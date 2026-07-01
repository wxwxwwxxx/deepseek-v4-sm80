#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/vllm}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_nsys_minisgl_v1_moe_4096x128_bs4}"
NSYS_BASE="${NSYS_BASE:-/tmp/nsys_minisgl_v1_moe_4096x128_bs4}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/usr/local/bin/torchrun}"
NPROC="${NPROC:-8}"
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

cd "${MINISGL_ROOT}"

nsys profile \
  -t cuda,nvtx,osrt,cublas \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cudabacktrace=none \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
  -o "${NSYS_BASE}" \
  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC}" \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants v1_moe \
    --scenarios decode_throughput_bs8 \
    --prompt-len 4096 \
    --decode-len 128 \
    --batch-size 4 \
    --repeats 1 \
    --warmup-repeats 0 \
    --page-size 256 \
    --output-dir "${OUTPUT_DIR}" \
    --keep-going \
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

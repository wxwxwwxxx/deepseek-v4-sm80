#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
TARGET_MILESTONE_DIR="${TARGET_MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_vllm_gap}"
VLLM_MILESTONE_DIR="${VLLM_MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/vllm}"
VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm-dsv4-docker}"
VLLM_VENV="${VLLM_VENV:-/workspace/venvs/vllm-dsv4}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target07_nsys_vllm_4096x128_bs4_warmup1}"
NSYS_BASE="${NSYS_BASE:-/tmp/nsys_target07_vllm_4096x128_bs4_warmup1}"
TP_SIZE="${TP_SIZE:-8}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

source "${VLLM_VENV}/bin/activate"
source "${VLLM_MILESTONE_DIR}/scripts/vllm_env.sh"
setup_vllm_runtime_env

cmd=(
  nsys profile
  -t cuda,nvtx,osrt,cublas
  --sample=none
  --cpuctxsw=none
  --backtrace=none
  --cudabacktrace=none
  --trace-fork-before-exec=true
  --force-overwrite=true
  -o "${NSYS_BASE}"
  python "${VLLM_MILESTONE_DIR}/scripts/run_vllm_deepseek_v4_matrix.py"
  --model-path "${MODEL_PATH}"
  --vllm-root "${VLLM_ROOT}"
  --tensor-parallel-size "${TP_SIZE}"
  --block-size 256
  --output-dir "${OUTPUT_DIR}"
  --scenarios decode_throughput_bs8
  --prompt-len 4096
  --decode-len 128
  --batch-size 4
  --repeats 1
  --warmup-repeats 1
  --max-num-batched-tokens 4096
  --enable-chunked-prefill
  --cudagraph-capture-sizes 1,2,4
  --max-cudagraph-capture-size 4
)
cmd+=("$@")

cd "${VLLM_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"

if [[ -f "${NSYS_BASE}.nsys-rep" ]]; then
  nsys export --type sqlite --force-overwrite=true \
    --output "${NSYS_BASE}.sqlite" \
    "${NSYS_BASE}.nsys-rep" || true
fi

mkdir -p "${TARGET_MILESTONE_DIR}/raw" "${TARGET_MILESTONE_DIR}/summaries"
for artifact in "${OUTPUT_DIR}" "${NSYS_BASE}.nsys-rep" "${NSYS_BASE}.sqlite"; do
  if [[ -e "${artifact}" ]]; then
    rm -rf "${TARGET_MILESTONE_DIR}/raw/$(basename "${artifact}")"
    ln -s "${artifact}" "${TARGET_MILESTONE_DIR}/raw/$(basename "${artifact}")"
  fi
done

if [[ -f "${NSYS_BASE}.sqlite" ]]; then
  python "${TARGET_MILESTONE_DIR}/scripts/summarize_nsys_sqlite.py" \
    "${NSYS_BASE}.sqlite" \
    --output-json "${TARGET_MILESTONE_DIR}/summaries/$(basename "${NSYS_BASE}").json" \
    --output-md "${TARGET_MILESTONE_DIR}/summaries/$(basename "${NSYS_BASE}").md" \
    --nvtx-window "repeat:decode_throughput_bs8:0" || true
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  summary_dir="${TARGET_MILESTONE_DIR}/summaries/$(basename "${OUTPUT_DIR}")"
  rm -rf "${summary_dir}"
  mkdir -p "${summary_dir}"
  for name in run_config.json summary.json matrix.jsonl; do
    if [[ -f "${OUTPUT_DIR}/${name}" ]]; then
      cp "${OUTPUT_DIR}/${name}" "${summary_dir}/${name}"
    fi
  done
fi

echo "vLLM fair nsys output: ${OUTPUT_DIR}"
echo "vLLM fair nsys base: ${NSYS_BASE}"

#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_decode_metadata_deforestation}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC="${NPROC:-8}"
VARIANT="${VARIANT:-dsv4_sm80_a100_victory_metadatadeforest}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
DECODE_LEN="${DECODE_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
REPEATS="${REPEATS:-3}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target0764_${PROMPT_LEN}x${DECODE_LEN}_bs${BATCH_SIZE}_np${NUM_PAGES}}"
RAW_LINK="${MILESTONE_DIR}/raw/macro_${PROMPT_LEN}x${DECODE_LEN}_bs${BATCH_SIZE}_np${NUM_PAGES}"
SUMMARY_OUT="${MILESTONE_DIR}/summaries/macro_${PROMPT_LEN}x${DECODE_LEN}_bs${BATCH_SIZE}_np${NUM_PAGES}_summary.json"
LOG="${LOG:-${MILESTONE_DIR}/raw/macro_${PROMPT_LEN}x${DECODE_LEN}_bs${BATCH_SIZE}_np${NUM_PAGES}.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cd "${MINISGL_ROOT}"
mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC}" \
  benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --scenarios decode_throughput_bs8 \
    --prompt-len "${PROMPT_LEN}" \
    --decode-len "${DECODE_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --output-dir "${OUTPUT_DIR}" \
    --keep-going \
    "$@" 2>&1 | tee "${LOG}"

rm -rf "${RAW_LINK}"
ln -s "${OUTPUT_DIR}" "${RAW_LINK}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  cp "${OUTPUT_DIR}/summary.json" "${SUMMARY_OUT}"
fi

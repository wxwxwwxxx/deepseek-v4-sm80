#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_decode_metadata_deforestation}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC="${NPROC:-8}"
VARIANT="${VARIANT:-dsv4_sm80_a100_victory_metadatadeforest}"
PAGE_SIZE="${PAGE_SIZE:-256}"
OUTPUT="${OUTPUT:-/tmp/dsv4_target0764_text_smoke.json}"
LOG="${LOG:-${MILESTONE_DIR}/raw/text_smoke_${VARIANT}.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cd "${MINISGL_ROOT}"
mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC}" \
  benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --page-size "${PAGE_SIZE}" \
    --output "${OUTPUT}" \
    "$@" 2>&1 | tee "${LOG}"

cp "${OUTPUT}" "${MILESTONE_DIR}/raw/text_smoke_${VARIANT}.json"
variant_output="${OUTPUT%.json}.${VARIANT}.json"
if [[ -f "${variant_output}" ]]; then
  cp "${variant_output}" "${MILESTONE_DIR}/raw/text_smoke_${VARIANT}.variant.json"
fi

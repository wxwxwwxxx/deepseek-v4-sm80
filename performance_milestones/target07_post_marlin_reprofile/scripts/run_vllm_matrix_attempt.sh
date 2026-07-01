#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_post_marlin_reprofile}"
VLLM_MILESTONE_DIR="${VLLM_MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target07392_vllm_4096x128_bs4}"

mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"

MILESTONE_DIR="${VLLM_MILESTONE_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${VLLM_MILESTONE_DIR}/scripts/run_vllm_matrix.sh" \
  --scenarios decode_throughput_bs8 \
  --prompt-len "${PROMPT_LEN:-4096}" \
  --decode-len "${DECODE_LEN:-128}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --repeats "${REPEATS:-1}" \
  --warmup-repeats "${WARMUP_REPEATS:-1}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
  --enable-chunked-prefill \
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES:-1,2,4}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE:-4}" \
  "$@"

link_name="${MILESTONE_DIR}/raw/$(basename "${OUTPUT_DIR}")"
rm -rf "${link_name}"
ln -s "${OUTPUT_DIR}" "${link_name}"
if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  cp "${OUTPUT_DIR}/summary.json" \
    "${MILESTONE_DIR}/summaries/vllm_${PROMPT_LEN:-4096}x${DECODE_LEN:-128}_bs${BATCH_SIZE:-4}_summary.json"
fi
echo "vLLM output: ${OUTPUT_DIR}"
echo "target milestone raw symlink: ${link_name}"

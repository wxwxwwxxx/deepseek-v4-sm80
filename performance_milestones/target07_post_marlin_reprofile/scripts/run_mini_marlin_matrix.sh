#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_post_marlin_reprofile}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/usr/local/bin/torchrun}"
NPROC="${NPROC:-8}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
REPEATS="${REPEATS:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
DECODE_LENS="${DECODE_LENS:-128 1024}"

MARLIN_VARIANT="${MARLIN_VARIANT:-v1_moe_vllm_runner_marlin_wna16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cd "${MINISGL_ROOT}"
mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"

for decode_len in ${DECODE_LENS}; do
  output_dir="${OUTPUT_DIR:-/tmp/dsv4_target07392_marlin_${PROMPT_LEN}x${decode_len}_bs${BATCH_SIZE}_np${NUM_PAGES}}"
  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC}" \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${MARLIN_VARIANT}" \
    --scenarios decode_throughput_bs8 \
    --prompt-len "${PROMPT_LEN}" \
    --decode-len "${decode_len}" \
    --batch-size "${BATCH_SIZE}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --output-dir "${output_dir}" \
    --keep-going \
    "$@"

  link_name="${MILESTONE_DIR}/raw/$(basename "${output_dir}")"
  rm -rf "${link_name}"
  ln -s "${output_dir}" "${link_name}"
  if [[ -f "${output_dir}/summary.json" ]]; then
    cp "${output_dir}/summary.json" \
      "${MILESTONE_DIR}/summaries/mini_marlin_${PROMPT_LEN}x${decode_len}_bs${BATCH_SIZE}_np${NUM_PAGES}_summary.json"
  fi
  echo "mini Marlin output: ${output_dir}"
  echo "milestone raw symlink: ${link_name}"
done

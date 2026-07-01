#!/usr/bin/env bash
set -euo pipefail

MINISGL_ROOT="${MINISGL_ROOT:-/workspace/mini-sglang}"
MILESTONE_DIR="${MILESTONE_DIR:-${MINISGL_ROOT}/performance_milestones/target07_bf16_sparse_decode_splitk}"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/usr/local/bin/torchrun}"
NPROC="${NPROC:-8}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
DECODE_LEN="${DECODE_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dsv4_target07395_nsys_splitk_${PROMPT_LEN}x${DECODE_LEN}_bs${BATCH_SIZE}_np${NUM_PAGES}}"
NSYS_BASE="${NSYS_BASE:-/tmp/nsys_target07395_splitk_${PROMPT_LEN}x${DECODE_LEN}_bs${BATCH_SIZE}_np${NUM_PAGES}}"
NSYS_PROFILE_RANKS="${NSYS_PROFILE_RANKS:-0}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-}"

SPLITK_VARIANT="${SPLITK_VARIANT:-v1_moe_vllm_runner_marlin_wna16_globaltopk_splitkbf16_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NSYS_BASE NSYS_PROFILE_RANKS NSYS_CUDA_GRAPH_TRACE

cd "${MINISGL_ROOT}"
mkdir -p "${MILESTONE_DIR}/raw" "${MILESTONE_DIR}/summaries"

"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC}" --no-python \
  "${MILESTONE_DIR}/scripts/nsys_rank_wrapper.sh" \
  python benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${SPLITK_VARIANT}" \
    --scenarios decode_throughput_bs8 \
    --prompt-len "${PROMPT_LEN}" \
    --decode-len "${DECODE_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --repeats 1 \
    --warmup-repeats 1 \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 \
    --output-dir "${OUTPUT_DIR}" \
    --keep-going \
    "$@"

for report in "${NSYS_BASE}"_rank*.nsys-rep; do
  [[ -f "${report}" ]] || continue
  sqlite="${report%.nsys-rep}.sqlite"
  nsys export --type sqlite --force-overwrite=true --output "${sqlite}" "${report}" || true
done

for artifact in "${OUTPUT_DIR}" "${NSYS_BASE}"_rank*.nsys-rep "${NSYS_BASE}"_rank*.sqlite; do
  [[ -e "${artifact}" ]] || continue
  link_name="${MILESTONE_DIR}/raw/$(basename "${artifact}")"
  rm -rf "${link_name}"
  ln -s "${artifact}" "${link_name}"
  echo "linked ${artifact}"
done

if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
  cp "${OUTPUT_DIR}/summary.json" \
    "${MILESTONE_DIR}/summaries/macro_splitk_${PROMPT_LEN}x${DECODE_LEN}_bs${BATCH_SIZE}_np${NUM_PAGES}_nsys_summary.json"
fi

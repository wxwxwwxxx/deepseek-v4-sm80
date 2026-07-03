#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${MILESTONE_DIR:-${ROOT}/performance_milestones/target08_bf16_cache_graph_memory_attribution}"
RAW_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN="${TORCHRUN:-torchrun}"
TP="${TP:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
VARIANT="${VARIANT:-dsv4_sm80_a100_victory}"
SCENARIO="${SCENARIO:-decode_ladder_bs16}"
REPEATS="${REPEATS:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
RUN_FULL_BASELINE="${RUN_FULL_BASELINE:-1}"
RUN_ALL_FULL_BUCKET_VARIANTS="${RUN_ALL_FULL_BUCKET_VARIANTS:-0}"
PHASE2_CASES="${PHASE2_CASES:-}"

mkdir -p "${RAW_DIR}" "${MILESTONE_DIR}/summaries"

run_case() {
  local label="$1"
  local disable_toggles="$2"
  shift 2
  local buckets=("$@")
  local out_dir="${RAW_DIR}/${label}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "Skipping existing ${label}"
    return
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  (
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES="${disable_toggles}" \
    "${TORCHRUN}" --standalone --nproc_per_node="${TP}" \
      benchmark/offline/deepseek_v4_perf_matrix.py \
      --model-path "${MODEL_PATH}" \
      --variants "${VARIANT}" \
      --scenarios "${SCENARIO}" \
      --page-size "${PAGE_SIZE}" \
      --num-pages "${NUM_PAGES}" \
      --max-seq-len "${MAX_SEQ_LEN}" \
      --repeats "${REPEATS}" \
      --warmup-repeats "${WARMUP_REPEATS}" \
      --allow-dsv4-cuda-graph \
      --cuda-graph-bs "${buckets[@]}" \
      --cuda-graph-capture-greedy-sample \
      --output-dir "${out_dir}" \
      --keep-going
  ) 2>&1 | tee "${out_dir}/torchrun.log"
}

run_named_full_case() {
  local case_name="$1"
  case "${case_name}" in
    no_projection_bf16_caches)
      run_case full_no_projection_bf16_caches projection_bf16_caches 1 2 4 8 16
      ;;
    no_q_wqb_bf16_cache)
      run_case full_no_q_wqb_bf16_cache q_wqb 1 2 4 8 16
      ;;
    no_wo_b_bf16_cache)
      run_case full_no_wo_b_bf16_cache wo_b 1 2 4 8 16
      ;;
    no_wo_a_bf16_bmm_cache)
      run_case full_no_wo_a_bf16_bmm_cache wo_a 1 2 4 8 16
      ;;
    no_indexer_wq_b_bf16_cache)
      run_case full_no_indexer_wq_b_bf16_cache indexer_wq_b 1 2 4 8 16
      ;;
    no_shared_expert_bf16_cache)
      run_case full_no_shared_expert_bf16_cache shared_expert 1 2 4 8 16
      ;;
    no_all_tested_bf16_caches)
      run_case full_no_all_tested_bf16_caches all_tested_bf16_caches 1 2 4 8 16
      ;;
    *)
      echo "Unknown PHASE2 case: ${case_name}" >&2
      return 2
      ;;
  esac
}

# Phase 1: cheap first-graph attribution. TARGET 08.06 showed the first graph
# dominates, so every A/B starts with a single bs16 capture.
run_case single_full_victory "" 16
run_case single_no_projection_bf16_caches projection_bf16_caches 16
run_case single_no_q_wqb_bf16_cache q_wqb 16
run_case single_no_wo_b_bf16_cache wo_b 16
run_case single_no_wo_a_bf16_bmm_cache wo_a 16
run_case single_no_indexer_wq_b_bf16_cache indexer_wq_b 16
run_case single_no_shared_expert_bf16_cache shared_expert 16
run_case single_no_all_tested_bf16_caches all_tested_bf16_caches 16

if [[ "${RUN_FULL_BASELINE}" == "1" ]]; then
  run_case full_full_victory "" 1 2 4 8 16
fi

if [[ "${RUN_ALL_FULL_BUCKET_VARIANTS}" == "1" ]]; then
  PHASE2_CASES="no_projection_bf16_caches no_q_wqb_bf16_cache no_wo_b_bf16_cache no_wo_a_bf16_bmm_cache no_indexer_wq_b_bf16_cache no_shared_expert_bf16_cache no_all_tested_bf16_caches"
fi

for case_name in ${PHASE2_CASES}; do
  run_named_full_case "${case_name}"
done

python "${MILESTONE_DIR}/scripts/summarize_bf16_cache_graph_memory.py" \
  --milestone-dir "${MILESTONE_DIR}"

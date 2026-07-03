#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${MILESTONE_DIR:-${ROOT}/performance_milestones/target08_prefix_cache_serving_stability}"
RAW_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN="${TORCHRUN:-torchrun}"
TP="${TP:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
MAX_EXTEND_TOKENS="${MAX_EXTEND_TOKENS:-20000}"
MAX_RUNNING_REQ="${MAX_RUNNING_REQ:-16}"
REPEATS="${REPEATS:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
VARIANT="${VARIANT:-dsv4_sm80_a100_victory}"

SCENARIOS=(
  prefix_full_hit_257_bs4
  prefix_partial_hit_769_bs8
  prefix_mixed_hit_miss_bs16
  prefix_multi_112req_wave16
  prefix_eviction_pressure_96req_wave16
)

BUCKETS=(1 2 4 8 16)

mkdir -p "${RAW_DIR}" "${MILESTONE_DIR}/summaries"

run_torchrun() {
  local log_path="$1"
  shift
  (
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    "${TORCHRUN}" --standalone --nproc_per_node="${TP}" "$@"
  ) 2>&1 | tee "${log_path}"
}

run_perf_matrix() {
  local out_dir="$1"
  local prefix_mode="$2"
  local prefix_args=()
  if [[ "${prefix_mode}" == "prefix_on" ]]; then
    prefix_args=(--enable-dsv4-radix-prefix-cache)
  fi

  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --scenarios "${SCENARIOS[@]}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --max-extend-tokens "${MAX_EXTEND_TOKENS}" \
    --max-running-req "${MAX_RUNNING_REQ}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs "${BUCKETS[@]}" \
    --output-dir "${out_dir}" \
    --keep-going \
    "${prefix_args[@]}"
}

run_perf_matrix "${RAW_DIR}/prefix_off_control" "prefix_off"
run_perf_matrix "${RAW_DIR}/prefix_on_opt_in" "prefix_on"

python "${MILESTONE_DIR}/scripts/summarize_prefix_cache_serving_stability.py" \
  --milestone-dir "${MILESTONE_DIR}"

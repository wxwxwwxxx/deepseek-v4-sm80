#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${MILESTONE_DIR:-${ROOT}/performance_milestones/target08_serving_graph_bucket_policy}"
RAW_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN="${TORCHRUN:-torchrun}"
TP="${TP:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
REPEATS="${REPEATS:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
VARIANT="${VARIANT:-dsv4_sm80_a100_victory}"

PREFIX_OFF_SCENARIOS=(
  historical_4096_1024_bs4
  historical_4096_128_bs4
  shared_prompt_reuse_bs8
  decode_ladder_bs16
  serving_mixed_112req_wave16
)

PREFIX_ON_SCENARIOS=(
  shared_prompt_reuse_bs8
)

BUCKET_LABELS=(
  1_2_4
  1_2_4_8
  1_2_4_8_16
)

BUCKET_VALUES=(
  "1 2 4"
  "1 2 4 8"
  "1 2 4 8 16"
)

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
  shift 2
  local buckets=("$@")
  local scenarios=("${PREFIX_OFF_SCENARIOS[@]}")
  local prefix_args=()
  if [[ "${prefix_mode}" == "prefix_on" ]]; then
    scenarios=("${PREFIX_ON_SCENARIOS[@]}")
    prefix_args=(--enable-dsv4-radix-prefix-cache)
  fi

  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --scenarios "${scenarios[@]}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs "${buckets[@]}" \
    --output-dir "${out_dir}" \
    --keep-going \
    "${prefix_args[@]}"
}

for idx in "${!BUCKET_LABELS[@]}"; do
  label="${BUCKET_LABELS[$idx]}"
  read -r -a buckets <<< "${BUCKET_VALUES[$idx]}"
  run_perf_matrix "${RAW_DIR}/bucket_${label}_prefix_off" "prefix_off" "${buckets[@]}"
  run_perf_matrix "${RAW_DIR}/bucket_${label}_prefix_on_shared" "prefix_on" "${buckets[@]}"
done

python "${MILESTONE_DIR}/scripts/summarize_bucket_policy.py" \
  --milestone-dir "${MILESTONE_DIR}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${MILESTONE_DIR:-${ROOT}/performance_milestones/target08_cuda_graph_memory_attribution}"
RAW_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN="${TORCHRUN:-torchrun}"
TP="${TP:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PAGE_SIZE="${PAGE_SIZE:-256}"
VARIANT="${VARIANT:-dsv4_sm80_a100_victory}"
SCENARIO="${SCENARIO:-decode_ladder_bs16}"
REPEATS="${REPEATS:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

mkdir -p "${RAW_DIR}" "${MILESTONE_DIR}/summaries"

run_torchrun() {
  local log_path="$1"
  local disable_capture_locs="$2"
  shift 2
  (
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    MINISGL_DSV4_DISABLE_CAPTURE_COMPRESSED_LOCS_IN_GRAPH="${disable_capture_locs}" \
    "${TORCHRUN}" --standalone --nproc_per_node="${TP}" "$@"
  ) 2>&1 | tee "${log_path}"
}

run_case() {
  local label="$1"
  local max_seq_len="$2"
  local num_pages="$3"
  local greedy_mode="$4"
  local metadata_mode="$5"
  shift 5
  local buckets=("$@")
  local out_dir="${RAW_DIR}/${label}"
  local disable_capture_locs=0
  local greedy_arg=(--cuda-graph-capture-greedy-sample)

  if [[ "${greedy_mode}" == "off" ]]; then
    greedy_arg=(--no-cuda-graph-capture-greedy-sample)
  fi
  if [[ "${metadata_mode}" == "off" ]]; then
    disable_capture_locs=1
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "Skipping existing ${label}"
    return
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" "${disable_capture_locs}" \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --scenarios "${SCENARIO}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${num_pages}" \
    --max-seq-len "${max_seq_len}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs "${buckets[@]}" \
    "${greedy_arg[@]}" \
    --output-dir "${out_dir}" \
    --keep-going
}

# Bucket-set sensitivity, baseline: max_seq_len 2048, num_pages 128.
run_case bucketset_1_2_4_np128_sl2048_greedy_on_metadata_on 2048 128 on on 1 2 4
run_case bucketset_1_2_4_8_np128_sl2048_greedy_on_metadata_on 2048 128 on on 1 2 4 8
run_case bucketset_1_2_4_8_16_np128_sl2048_greedy_on_metadata_on 2048 128 on on 1 2 4 8 16

# Single-bucket sensitivity.
run_case single_1_np128_sl2048_greedy_on_metadata_on 2048 128 on on 1
run_case single_4_np128_sl2048_greedy_on_metadata_on 2048 128 on on 4
run_case single_8_np128_sl2048_greedy_on_metadata_on 2048 128 on on 8
run_case single_16_np128_sl2048_greedy_on_metadata_on 2048 128 on on 16

# Greedy-sample and graph-captured compressed-loc metadata A/B.
run_case greedy_off_np128_sl2048_metadata_on 2048 128 off on 1 2 4 8 16
run_case metadata_off_np128_sl2048_greedy_on 2048 128 on off 1 2 4 8 16

# max_seq_len and num_pages sensitivity. The 2048/128 point is covered above.
run_case seq1280_np64_greedy_on_metadata_on 1280 64 on on 1 2 4 8 16
run_case seq1280_np128_greedy_on_metadata_on 1280 128 on on 1 2 4 8 16
run_case seq2048_np64_greedy_on_metadata_on 2048 64 on on 1 2 4 8 16
run_case seq5120_np64_greedy_on_metadata_on 5120 64 on on 1 2 4 8 16
run_case seq5120_np128_greedy_on_metadata_on 5120 128 on on 1 2 4 8 16

python "${MILESTONE_DIR}/scripts/summarize_cuda_graph_memory.py" \
  --milestone-dir "${MILESTONE_DIR}"

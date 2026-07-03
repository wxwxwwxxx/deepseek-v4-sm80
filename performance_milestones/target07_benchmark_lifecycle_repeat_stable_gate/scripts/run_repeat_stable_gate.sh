#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${MILESTONE_DIR:-${ROOT}/performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate}"
RAW_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
TORCHRUN="${TORCHRUN:-torchrun}"
TP="${TP:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
REPEATS="${REPEATS:-3}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
PAGE_SIZE="${PAGE_SIZE:-256}"
NUM_PAGES="${NUM_PAGES:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
LONG_DECODE_LEN="${LONG_DECODE_LEN:-1024}"
SHORT_DECODE_LEN="${SHORT_DECODE_LEN:-128}"

BASELINE_VARIANT="dsv4_sm80_a100_victory"
CANDIDATE_VARIANT="dsv4_sm80_a100_victory_densefp8marlinproj"

mkdir -p "${RAW_DIR}" "${MILESTONE_DIR}/summaries"

run_torchrun() {
  local log_path="$1"
  shift
  (
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${TORCHRUN}" --standalone --nproc_per_node="${TP}" "$@"
  ) 2>&1 | tee "${log_path}"
}

run_smoke() {
  local variant="$1"
  local out_dir="${RAW_DIR}/smoke_${variant}"
  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" \
    benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path "${MODEL_PATH}" \
    --variants "${variant}" \
    --output "${out_dir}/text_smoke.json" \
    --tensor-parallel-size "${TP}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --max-seq-len 1024 \
    --max-extend-tokens 4096 \
    --max-tokens 64 \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 \
    --fail-on-warning
}

run_macro() {
  local label="$1"
  local variant="$2"
  local decode_len="$3"
  local out_dir="${RAW_DIR}/${label}_${variant}"
  mkdir -p "${out_dir}"
  run_torchrun "${out_dir}/torchrun.log" \
    benchmark/offline/deepseek_v4_perf_matrix.py \
    --model-path "${MODEL_PATH}" \
    --variants "${variant}" \
    --scenarios decode_throughput_bs8 \
    --prompt-len "${PROMPT_LEN}" \
    --decode-len "${decode_len}" \
    --batch-size "${BATCH_SIZE}" \
    --repeats "${REPEATS}" \
    --warmup-repeats "${WARMUP_REPEATS}" \
    --page-size "${PAGE_SIZE}" \
    --num-pages "${NUM_PAGES}" \
    --output-dir "${out_dir}" \
    --keep-going
}

run_smoke "${CANDIDATE_VARIANT}"
run_macro "4096x${LONG_DECODE_LEN}" "${BASELINE_VARIANT}" "${LONG_DECODE_LEN}"
run_macro "4096x${LONG_DECODE_LEN}" "${CANDIDATE_VARIANT}" "${LONG_DECODE_LEN}"
run_macro "4096x${SHORT_DECODE_LEN}" "${BASELINE_VARIANT}" "${SHORT_DECODE_LEN}"
run_macro "4096x${SHORT_DECODE_LEN}" "${CANDIDATE_VARIANT}" "${SHORT_DECODE_LEN}"

python "${MILESTONE_DIR}/scripts/summarize_repeat_stable_gate.py" \
  --milestone-dir "${MILESTONE_DIR}"

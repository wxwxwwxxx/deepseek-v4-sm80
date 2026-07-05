#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${ROOT_DIR}/performance_milestones/target09_low_precision_preflight/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
SCENARIOS=(
  historical_4096_128_bs4
  historical_4096_1024_bs4
  serving_mixed_112req_wave16
  prefix_multi_112req_wave16
)

mkdir -p "${OUT_DIR}"

for scenario in "${SCENARIOS[@]}"; do
  case_out="${OUT_DIR}/owner_timing_${scenario}"
  mkdir -p "${case_out}"
  (
    cd "${ROOT_DIR}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1 \
    MINISGL_DSV4_OWNER_TIMING=1 \
    timeout 3600 torchrun --standalone --nproc_per_node=8 \
      benchmark/offline/deepseek_v4_perf_matrix.py \
      --model-path "${MODEL_PATH}" \
      --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
      --page-size 256 --num-pages 128 \
      --enable-dsv4-radix-prefix-cache \
      --enable-dsv4-component-loc-ownership \
      --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
      --scenarios "${scenario}" \
      --repeats 1 --warmup-repeats 0 --seed 20260705 \
      --output-dir "${case_out}" \
      --keep-going
  ) 2>&1 | tee "${OUT_DIR}/owner_timing_${scenario}.log"
done

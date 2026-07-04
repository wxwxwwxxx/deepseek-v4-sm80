#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/performance_milestones/target08_route_b_component_mapping_lifecycle_fix/raw/focused_route_b_graph}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/performance_milestones/target08_route_b_component_mapping_lifecycle_fix/raw/focused_route_b_graph.log}"

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_PATH}")"

cd "${ROOT_DIR}"

MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --scenarios \
    prefix_full_hit_257_bs4 \
    prefix_full_hit_512_bs4 \
    prefix_full_hit_513_bs4 \
    prefix_full_hit_768_bs4 \
    prefix_full_hit_769_bs4 \
    prefix_full_hit_513_longout_bs4 \
    prefix_partial_hit_769_bs8 \
    prefix_mixed_hit_miss_bs16 \
    prefix_multi_112req_wave16 \
  --output-dir "${OUT_DIR}" \
  --keep-going 2>&1 | tee "${LOG_PATH}"

python \
  performance_milestones/target08_route_b_component_mapping_lifecycle_fix/scripts/summarize_focused_route_b.py \
  --output-dir "${OUT_DIR}"

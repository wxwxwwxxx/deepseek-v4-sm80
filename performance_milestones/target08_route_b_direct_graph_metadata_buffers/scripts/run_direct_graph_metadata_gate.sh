#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${ROOT_DIR}/performance_milestones/target08_route_b_direct_graph_metadata_buffers/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

run_torch() {
  local name="$1"
  shift
  echo "== ${name}"
  timeout 1800 torchrun --standalone --nproc_per_node=8 "$@"
}

run_torch large_phase1_prefix_on \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path "${MODEL_PATH}" \
  --scenarios serving_mixed_112req_wave16 \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --keep-going \
  --variants dsv4_sm80_a100_victory \
  --enable-dsv4-radix-prefix-cache \
  --output-dir "${OUT_DIR}/large_phase1_prefix_on"

run_torch large_route_b_graph_baseline \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path "${MODEL_PATH}" \
  --scenarios serving_mixed_112req_wave16 \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --keep-going \
  --variants dsv4_sm80_a100_victory \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output-dir "${OUT_DIR}/large_route_b_graph_baseline"

run_torch large_route_b_direct_graph_metadata \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path "${MODEL_PATH}" \
  --scenarios serving_mixed_112req_wave16 \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --keep-going \
  --variants dsv4_sm80_a100_victory_directgraphmetadata \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output-dir "${OUT_DIR}/large_route_b_direct_graph_metadata_v2"

MINISGL_DSV4_OWNER_TIMING=1 MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=30000 \
run_torch profile_large_route_b_direct_graph_metadata \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path "${MODEL_PATH}" \
  --scenarios serving_mixed_112req_wave16 \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --keep-going \
  --variants dsv4_sm80_a100_victory_directgraphmetadata \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output-dir "${OUT_DIR}/profile_large_route_b_direct_graph_metadata"

run_torch prefix_hit_route_b_direct_graph_metadata_direct_only \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path "${MODEL_PATH}" \
  --scenarios prefix_full_hit_513_longout_bs4 \
  --page-size 256 \
  --num-pages 128 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --keep-going \
  --variants dsv4_sm80_a100_victory_directgraphmetadata \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output-dir "${OUT_DIR}/prefix_hit_route_b_direct_graph_metadata_direct_only"

run_torch eviction_pressure_route_b_direct_graph_metadata \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path "${MODEL_PATH}" \
  --scenarios prefix_eviction_pressure_96req_wave16 \
  --page-size 256 \
  --num-pages 48 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --keep-going \
  --variants dsv4_sm80_a100_victory_directgraphmetadata \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output-dir "${OUT_DIR}/eviction_pressure_route_b_direct_graph_metadata"

run_torch text_smoke_route_b_direct_graph_metadata \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path "${MODEL_PATH}" \
  --page-size 256 \
  --num-pages 64 \
  --max-seq-len 1024 \
  --max-extend-tokens 4096 \
  --max-tokens 16 \
  --fail-on-warning \
  --variants dsv4_sm80_a100_victory_directgraphmetadata \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output "${OUT_DIR}/text_smoke_route_b_direct_graph_metadata.json"

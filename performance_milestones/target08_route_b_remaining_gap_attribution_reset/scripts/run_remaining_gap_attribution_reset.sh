#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${ROOT_DIR}/performance_milestones/target08_route_b_remaining_gap_attribution_reset"
OUT_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"
TIMEOUT_S="${TIMEOUT_S:-2400}"
LARGE_REPEATS="${LARGE_REPEATS:-3}"
RUN_THROUGHPUT="${RUN_THROUGHPUT:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"

mkdir -p "${OUT_DIR}" "${MILESTONE_DIR}/summaries"
cd "${ROOT_DIR}"

run_torch() {
  local name="$1"
  shift
  echo "== ${name}"
  timeout "${TIMEOUT_S}" torchrun --standalone --nproc_per_node="${NPROC}" "$@" \
    2>&1 | tee "${OUT_DIR}/${name}.log"
}

COMMON_ARGS=(
  benchmark/offline/deepseek_v4_perf_matrix.py
  --model-path "${MODEL_PATH}"
  --scenarios serving_mixed_112req_wave16
  --page-size 256
  --num-pages 128
  --allow-dsv4-cuda-graph
  --cuda-graph-bs 1 2 4 8 16
  --keep-going
)

run_phase1() {
  local name="$1"
  run_torch "${name}" \
    "${COMMON_ARGS[@]}" \
    --variants dsv4_sm80_a100_victory \
    --enable-dsv4-radix-prefix-cache \
    --output-dir "${OUT_DIR}/${name}"
}

run_route_b() {
  local name="$1"
  local variant="$2"
  run_torch "${name}" \
    "${COMMON_ARGS[@]}" \
    --variants "${variant}" \
    --enable-dsv4-radix-prefix-cache \
    --enable-dsv4-component-loc-ownership \
    --output-dir "${OUT_DIR}/${name}"
}

if [[ "${RUN_THROUGHPUT}" == "1" ]]; then
  for repeat in $(seq 1 "${LARGE_REPEATS}"); do
    suffix="$(printf "r%02d" "${repeat}")"
    run_phase1 "throughput_${suffix}_phase1_prefix_on"
    run_route_b "throughput_${suffix}_route_b_graph_baseline" dsv4_sm80_a100_victory
    run_route_b "throughput_${suffix}_route_b_direct_c4" dsv4_sm80_a100_victory_directgraphmetadata_c4
    run_route_b "throughput_${suffix}_route_b_direct_full" dsv4_sm80_a100_victory_directgraphmetadata
  done
fi

if [[ "${RUN_PROFILE}" == "1" ]]; then
  MINISGL_DSV4_OWNER_TIMING=1 MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_phase1 profile_phase1_prefix_on
  MINISGL_DSV4_OWNER_TIMING=1 MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_route_b profile_route_b_graph_baseline dsv4_sm80_a100_victory
  MINISGL_DSV4_OWNER_TIMING=1 MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_route_b profile_route_b_direct_c4 dsv4_sm80_a100_victory_directgraphmetadata_c4
  MINISGL_DSV4_OWNER_TIMING=1 MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_route_b profile_route_b_direct_full dsv4_sm80_a100_victory_directgraphmetadata
fi

git status --short > "${OUT_DIR}/git_status_short.txt"
python "${MILESTONE_DIR}/scripts/summarize_remaining_gap_attribution_reset.py"

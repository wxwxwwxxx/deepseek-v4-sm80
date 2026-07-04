#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${ROOT_DIR}/performance_milestones/target08_route_b_lifetime_cache_promotion_gate"
OUT_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"
TIMEOUT_S="${TIMEOUT_S:-3000}"
SERVING_REPEATS="${SERVING_REPEATS:-3}"
PREFIX_MULTI_REPEATS="${PREFIX_MULTI_REPEATS:-3}"
EVICTION_REPEATS="${EVICTION_REPEATS:-2}"
RUN_TEXT_SMOKE="${RUN_TEXT_SMOKE:-1}"
RUN_VERIFY_SERVING="${RUN_VERIFY_SERVING:-1}"
RUN_SERVING="${RUN_SERVING:-1}"
RUN_PREFIX_MULTI="${RUN_PREFIX_MULTI:-1}"
RUN_VERIFY_EVICTION="${RUN_VERIFY_EVICTION:-1}"
RUN_EVICTION="${RUN_EVICTION:-1}"
RUN_DECODE_CONTROL="${RUN_DECODE_CONTROL:-1}"
RUN_COUNTER_PROFILES="${RUN_COUNTER_PROFILES:-1}"

mkdir -p "${OUT_DIR}" "${MILESTONE_DIR}/summaries"
cd "${ROOT_DIR}"

run_torch() {
  local name="$1"
  shift
  echo "== ${name}"
  timeout "${TIMEOUT_S}" torchrun --standalone --nproc_per_node="${NPROC}" "$@" \
    > "${OUT_DIR}/${name}.log" 2>&1
}

VARIANT="dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime"
COMMON_MATRIX_ARGS=(
  benchmark/offline/deepseek_v4_perf_matrix.py
  --model-path "${MODEL_PATH}"
  --variants "${VARIANT}"
  --page-size 256
  --num-pages 128
  --allow-dsv4-cuda-graph
  --cuda-graph-bs 1 2 4 8 16
  --enable-dsv4-radix-prefix-cache
  --enable-dsv4-component-loc-ownership
  --keep-going
)

run_matrix() {
  local name="$1"
  local scenario="$2"
  shift 2
  MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    run_torch "${name}" \
    "${COMMON_MATRIX_ARGS[@]}" \
    --scenarios "${scenario}" \
    --output-dir "${OUT_DIR}/${name}" \
    "$@"
}

run_matrix_verify() {
  local name="$1"
  local scenario="$2"
  shift 2
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1 \
    run_matrix "${name}" "${scenario}" "$@"
}

run_matrix_profile() {
  local name="$1"
  local scenario="$2"
  shift 2
  MINISGL_DSV4_OWNER_TIMING=1 \
  MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_matrix "${name}" "${scenario}" "$@"
}

if [[ "${RUN_TEXT_SMOKE}" == "1" ]]; then
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1 \
  MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    run_torch text_smoke_routeb_lifetime_verify \
    benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path "${MODEL_PATH}" \
    --variants "${VARIANT}" \
    --page-size 256 \
    --num-pages 128 \
    --max-seq-len 512 \
    --max-extend-tokens 512 \
    --max-tokens 8 \
    --enable-dsv4-radix-prefix-cache \
    --enable-dsv4-component-loc-ownership \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 8 16 \
    --output "${OUT_DIR}/text_smoke_routeb_lifetime_verify.json" \
    --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？' \
    --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？' \
    --prompt 'Answer in one short English sentence: what color is the sky on a clear day?'
fi

if [[ "${RUN_VERIFY_SERVING}" == "1" ]]; then
  run_matrix_verify verify_serving_mixed_lifetime serving_mixed_112req_wave16
fi

if [[ "${RUN_SERVING}" == "1" ]]; then
  for repeat in $(seq 1 "${SERVING_REPEATS}"); do
    run_matrix "serving_mixed_r$(printf "%02d" "${repeat}")_lifetime" \
      serving_mixed_112req_wave16
  done
fi

if [[ "${RUN_PREFIX_MULTI}" == "1" ]]; then
  for repeat in $(seq 1 "${PREFIX_MULTI_REPEATS}"); do
    run_matrix "prefix_multi_r$(printf "%02d" "${repeat}")_lifetime" \
      prefix_multi_112req_wave16
  done
fi

if [[ "${RUN_VERIFY_EVICTION}" == "1" ]]; then
  run_matrix_verify verify_prefix_eviction_lifetime prefix_eviction_pressure_96req_wave16
fi

if [[ "${RUN_EVICTION}" == "1" ]]; then
  for repeat in $(seq 1 "${EVICTION_REPEATS}"); do
    run_matrix "prefix_eviction_r$(printf "%02d" "${repeat}")_lifetime" \
      prefix_eviction_pressure_96req_wave16
  done
fi

if [[ "${RUN_DECODE_CONTROL}" == "1" ]]; then
  run_matrix decode_ladder_lifetime decode_ladder_bs16
fi

if [[ "${RUN_COUNTER_PROFILES}" == "1" ]]; then
  run_matrix_profile profile_serving_mixed_lifetime serving_mixed_112req_wave16
  run_matrix_profile profile_prefix_multi_lifetime prefix_multi_112req_wave16
  run_matrix_profile profile_prefix_eviction_lifetime prefix_eviction_pressure_96req_wave16
fi

git status --short > "${OUT_DIR}/git_status_short.txt"
python "${MILESTONE_DIR}/scripts/summarize_lifetime_cache_promotion_gate.py"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${ROOT_DIR}/performance_milestones/target08_post_prefix_reprofile"
OUT_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"
TIMEOUT_S="${TIMEOUT_S:-4200}"
HISTORICAL_REPEATS="${HISTORICAL_REPEATS:-3}"
SERVING_REPEATS="${SERVING_REPEATS:-3}"
VERIFY_REPEATS="${VERIFY_REPEATS:-1}"
RUN_TEXT_SMOKE="${RUN_TEXT_SMOKE:-1}"
RUN_VERIFY="${RUN_VERIFY:-1}"
RUN_MACRO="${RUN_MACRO:-1}"
RUN_OWNER_TIMING="${RUN_OWNER_TIMING:-0}"
RUN_OWNER_PROFILE_HISTORICAL="${RUN_OWNER_PROFILE_HISTORICAL:-1}"
RUN_CONTROL_PROFILE="${RUN_CONTROL_PROFILE:-1}"

PROMOTED_VARIANT="dsv4_sm80_a100_victory_prefix_routeb_lifetime"
CONTROL_VARIANT="dsv4_sm80_a100_victory"

mkdir -p "${OUT_DIR}" "${MILESTONE_DIR}/summaries"
cd "${ROOT_DIR}"

run_torch() {
  local name="$1"
  shift
  echo "== ${name}"
  timeout "${TIMEOUT_S}" torchrun --standalone --nproc_per_node="${NPROC}" "$@" \
    > "${OUT_DIR}/${name}.log" 2>&1
}

COMMON_MATRIX_ARGS=(
  benchmark/offline/deepseek_v4_perf_matrix.py
  --model-path "${MODEL_PATH}"
  --page-size 256
  --num-pages 128
  --allow-dsv4-cuda-graph
  --cuda-graph-bs 1 2 4 8 16
  --keep-going
)

PREFIX_MATRIX_ARGS=(
  "${COMMON_MATRIX_ARGS[@]}"
  --variants "${PROMOTED_VARIANT}"
  --enable-dsv4-radix-prefix-cache
  --enable-dsv4-component-loc-ownership
)

CONTROL_MATRIX_ARGS=(
  "${COMMON_MATRIX_ARGS[@]}"
  --variants "${CONTROL_VARIANT}"
)

run_prefix_matrix() {
  local name="$1"
  shift
  MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
  MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1 \
  MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4 \
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1 \
    run_torch "${name}" "${PREFIX_MATRIX_ARGS[@]}" "$@"
}

run_control_matrix() {
  local name="$1"
  shift
  MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    run_torch "${name}" "${CONTROL_MATRIX_ARGS[@]}" "$@"
}

run_prefix_verify_matrix() {
  local name="$1"
  shift
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1 \
    run_prefix_matrix "${name}" "$@"
}

run_prefix_profile_matrix() {
  local name="$1"
  shift
  MINISGL_DSV4_OWNER_TIMING=1 \
  MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_prefix_matrix "${name}" "$@"
}

run_control_profile_matrix() {
  local name="$1"
  shift
  MINISGL_DSV4_OWNER_TIMING=1 \
  MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_control_matrix "${name}" "$@"
}

run_repeated_prefix_macro() {
  local scenario="$1"
  local repeats="$2"
  local warmup_repeats="$3"
  shift 3
  local repeat
  for repeat in $(seq 1 "${repeats}"); do
    run_prefix_matrix "macro_promoted_${scenario}_r$(printf "%02d" "${repeat}")" \
      --scenarios "${scenario}" \
      --repeats 1 \
      --warmup-repeats "${warmup_repeats}" \
      --output-dir "${OUT_DIR}/macro_promoted_${scenario}_r$(printf "%02d" "${repeat}")" \
      "$@"
  done
}

run_repeated_control_macro() {
  local scenario="$1"
  local repeats="$2"
  local warmup_repeats="$3"
  shift 3
  local repeat
  for repeat in $(seq 1 "${repeats}"); do
    run_control_matrix "macro_control_${scenario}_r$(printf "%02d" "${repeat}")" \
      --scenarios "${scenario}" \
      --repeats 1 \
      --warmup-repeats "${warmup_repeats}" \
      --output-dir "${OUT_DIR}/macro_control_${scenario}_r$(printf "%02d" "${repeat}")" \
      "$@"
  done
}

if [[ "${RUN_TEXT_SMOKE}" == "1" ]]; then
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1 \
  MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
  MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1 \
  MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4 \
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1 \
    run_torch text_smoke_promoted_verify \
    benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path "${MODEL_PATH}" \
    --variants "${PROMOTED_VARIANT}" \
    --page-size 256 \
    --num-pages 128 \
    --max-seq-len 512 \
    --max-extend-tokens 512 \
    --max-tokens 8 \
    --enable-dsv4-radix-prefix-cache \
    --enable-dsv4-component-loc-ownership \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 8 16 \
    --output "${OUT_DIR}/text_smoke_promoted_verify.json" \
    --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？' \
    --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？' \
    --prompt 'Answer in one short English sentence: what color is the sky on a clear day?'
fi

if [[ "${RUN_VERIFY}" == "1" ]]; then
  run_prefix_verify_matrix verify_promoted_serving_mixed \
    --scenarios serving_mixed_112req_wave16 \
    --repeats "${VERIFY_REPEATS}" \
    --warmup-repeats 0 \
    --output-dir "${OUT_DIR}/verify_promoted_serving_mixed"

  run_prefix_verify_matrix verify_promoted_prefix_eviction \
    --scenarios prefix_eviction_pressure_96req_wave16 \
    --repeats "${VERIFY_REPEATS}" \
    --warmup-repeats 0 \
    --output-dir "${OUT_DIR}/verify_promoted_prefix_eviction"
fi

if [[ "${RUN_MACRO}" == "1" ]]; then
  for scenario in historical_4096_1024_bs4 historical_4096_128_bs4; do
    run_repeated_prefix_macro "${scenario}" "${HISTORICAL_REPEATS}" 1
    run_repeated_control_macro "${scenario}" "${HISTORICAL_REPEATS}" 1
  done

  for scenario in serving_mixed_112req_wave16 prefix_multi_112req_wave16 \
    prefix_eviction_pressure_96req_wave16 decode_ladder_bs16; do
    run_repeated_prefix_macro "${scenario}" "${SERVING_REPEATS}" 0
    run_repeated_control_macro "${scenario}" "${SERVING_REPEATS}" 0
  done
fi

if [[ "${RUN_OWNER_TIMING}" == "1" ]]; then
  for scenario in serving_mixed_112req_wave16 prefix_multi_112req_wave16 \
    prefix_eviction_pressure_96req_wave16 decode_ladder_bs16; do
    run_prefix_profile_matrix "profile_promoted_${scenario}" \
      --scenarios "${scenario}" \
      --repeats 1 \
      --warmup-repeats 0 \
      --output-dir "${OUT_DIR}/profile_promoted_${scenario}"
  done

  if [[ "${RUN_OWNER_PROFILE_HISTORICAL}" == "1" ]]; then
    run_prefix_profile_matrix profile_promoted_4096_128 \
      --scenarios historical_4096_128_bs4 \
      --repeats 1 \
      --warmup-repeats 0 \
      --output-dir "${OUT_DIR}/profile_promoted_4096_128"
  fi

  if [[ "${RUN_CONTROL_PROFILE}" == "1" ]]; then
    run_control_profile_matrix profile_control_serving_mixed \
      --scenarios serving_mixed_112req_wave16 \
      --repeats 1 \
      --warmup-repeats 0 \
      --output-dir "${OUT_DIR}/profile_control_serving_mixed"
  fi
fi

git status --short > "${OUT_DIR}/git_status_short.txt"
python "${MILESTONE_DIR}/scripts/summarize_post_prefix_reprofile.py"

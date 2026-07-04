#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${ROOT_DIR}/performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime"
OUT_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"
TIMEOUT_S="${TIMEOUT_S:-2400}"
LARGE_REPEATS="${LARGE_REPEATS:-3}"
RUN_TEXT_SMOKE="${RUN_TEXT_SMOKE:-1}"
RUN_THROUGHPUT="${RUN_THROUGHPUT:-1}"
RUN_VERIFY="${RUN_VERIFY:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"

mkdir -p "${OUT_DIR}" "${MILESTONE_DIR}/summaries"
cd "${ROOT_DIR}"

run_torch() {
  local name="$1"
  shift
  echo "== ${name}"
  timeout "${TIMEOUT_S}" torchrun --standalone --nproc_per_node="${NPROC}" "$@" \
    > "${OUT_DIR}/${name}.log" 2>&1
}

COMMON_ARGS=(
  benchmark/offline/deepseek_v4_perf_matrix.py
  --model-path "${MODEL_PATH}"
  --scenarios serving_mixed_112req_wave16
  --variants dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime
  --page-size 256
  --num-pages 128
  --allow-dsv4-cuda-graph
  --cuda-graph-bs 1 2 4 8 16
  --enable-dsv4-radix-prefix-cache
  --enable-dsv4-component-loc-ownership
  --keep-going
)

run_route_b_lifetime() {
  local name="$1"
  shift
  MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    run_torch "${name}" \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUT_DIR}/${name}" \
    "$@"
}

if [[ "${RUN_TEXT_SMOKE}" == "1" ]]; then
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1 \
  MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
    run_torch text_smoke_routeb_lifetime_verify \
    benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path "${MODEL_PATH}" \
    --variants dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime \
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

if [[ "${RUN_THROUGHPUT}" == "1" ]]; then
  for repeat in $(seq 1 "${LARGE_REPEATS}"); do
    run_route_b_lifetime "throughput_r$(printf "%02d" "${repeat}")_route_b_lifetime"
  done
fi

if [[ "${RUN_VERIFY}" == "1" ]]; then
  MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1 \
    run_route_b_lifetime verify_serving_route_b_lifetime
fi

if [[ "${RUN_PROFILE}" == "1" ]]; then
  MINISGL_DSV4_OWNER_TIMING=1 MINISGL_DSV4_OWNER_TIMING_MAX_SAMPLES=50000 \
    run_route_b_lifetime profile_route_b_lifetime
fi

git status --short > "${OUT_DIR}/git_status_short.txt"
python "${MILESTONE_DIR}/scripts/summarize_route_b_metadata_lifetime.py"

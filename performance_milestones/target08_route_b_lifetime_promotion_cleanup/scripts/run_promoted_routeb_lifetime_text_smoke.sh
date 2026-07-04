#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${ROOT_DIR}/performance_milestones/target08_route_b_lifetime_promotion_cleanup"
OUT_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"
TIMEOUT_S="${TIMEOUT_S:-1200}"
VARIANT="dsv4_sm80_a100_victory_prefix_routeb_lifetime"

mkdir -p "${OUT_DIR}" "${MILESTONE_DIR}/summaries"
cd "${ROOT_DIR}"

MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE_VERIFY=1 \
timeout "${TIMEOUT_S}" torchrun --standalone --nproc_per_node="${NPROC}" \
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
  --output "${OUT_DIR}/text_smoke_promoted_routeb_lifetime_verify.json" \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？' \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？' \
  --prompt 'Answer in one short English sentence: what color is the sky on a clear day?' \
  > "${OUT_DIR}/text_smoke_promoted_routeb_lifetime_verify.log" 2>&1

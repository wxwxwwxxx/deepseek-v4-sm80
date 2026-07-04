#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_PATH="${OUT_PATH:-${ROOT_DIR}/performance_milestones/target08_route_b_component_mapping_lifecycle_fix/raw/text_smoke_route_b_graph.json}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/performance_milestones/target08_route_b_component_mapping_lifecycle_fix/raw/text_smoke_route_b_graph.log}"

mkdir -p "$(dirname "${OUT_PATH}")" "$(dirname "${LOG_PATH}")"

cd "${ROOT_DIR}"

MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
timeout "${TIMEOUT_SECONDS:-900}" \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --page-size 256 \
  --num-pages 64 \
  --max-seq-len 512 \
  --max-extend-tokens 512 \
  --max-tokens 8 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output "${OUT_PATH}" \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？' \
  --prompt '请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？' \
  --prompt 'Answer in one short English sentence: what color is the sky on a clear day?' \
  > "${LOG_PATH}" 2>&1

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MILESTONE_DIR="${ROOT}/performance_milestones/target08_route_b_final_prefix_promotion_gate"
RAW_DIR="${MILESTONE_DIR}/raw"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TORCHRUN_TIMEOUT_SECONDS="${TORCHRUN_TIMEOUT_SECONDS:-7200}"

export CUDA_VISIBLE_DEVICES
export MINISGL_DISABLE_OVERLAP_SCHEDULING=1

cd "${ROOT}"

SCENARIOS=(
  decode_ladder_bs16
  serving_mixed_112req_wave16
  prefix_full_hit_257_bs4
  prefix_full_hit_512_bs4
  prefix_full_hit_513_bs4
  prefix_full_hit_768_bs4
  prefix_full_hit_769_bs4
  prefix_full_hit_513_longout_bs4
  prefix_partial_hit_769_bs8
  prefix_mixed_hit_miss_bs16
  prefix_multi_112req_wave16
  prefix_eviction_pressure_96req_wave16
)

COMMON_PERF_ARGS=(
  --model-path "${MODEL_PATH}"
  --variants dsv4_sm80_a100_victory
  --scenarios "${SCENARIOS[@]}"
  --page-size 256
  --num-pages 128
  --allow-dsv4-cuda-graph
  --cuda-graph-bs 1 2 4 8 16
  --keep-going
)

COMMON_TEXT_ARGS=(
  --model-path "${MODEL_PATH}"
  --variants dsv4_sm80_a100_victory
  --page-size 256
  --num-pages 64
  --max-seq-len 1024
  --max-extend-tokens 4096
  --max-tokens 16
  --fail-on-warning
  --prompt "请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它在哪个城市？"
  --prompt "请阅读前缀：杭州西湖位于浙江省。请用一句中文说出它所在省份？"
  --prompt "Answer in one short English sentence: what color is the sky on a clear day?"
  --prompt "Use one concise English sentence: name one benefit of caching a shared prompt prefix."
)

rm -rf \
  "${RAW_DIR}/perf_prefix_off" \
  "${RAW_DIR}/perf_phase1_prefix_on" \
  "${RAW_DIR}/perf_route_b_graph" \
  "${RAW_DIR}/text_smoke_prefix_off"* \
  "${RAW_DIR}/text_smoke_phase1_prefix_on"* \
  "${RAW_DIR}/text_smoke_route_b_graph"* \
  "${MILESTONE_DIR}/summaries"
mkdir -p "${RAW_DIR}" "${MILESTONE_DIR}/summaries"

pytest -q \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/engine/test_graph_runner.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  > "${RAW_DIR}/pytest_route_b_correctness.log" 2>&1

export MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1

timeout "${TORCHRUN_TIMEOUT_SECONDS}" torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  "${COMMON_PERF_ARGS[@]}" \
  --output-dir "${RAW_DIR}/perf_prefix_off" \
  > "${RAW_DIR}/perf_prefix_off.log" 2>&1

timeout "${TORCHRUN_TIMEOUT_SECONDS}" torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  "${COMMON_PERF_ARGS[@]}" \
  --enable-dsv4-radix-prefix-cache \
  --output-dir "${RAW_DIR}/perf_phase1_prefix_on" \
  > "${RAW_DIR}/perf_phase1_prefix_on.log" 2>&1

timeout "${TORCHRUN_TIMEOUT_SECONDS}" torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  "${COMMON_PERF_ARGS[@]}" \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output-dir "${RAW_DIR}/perf_route_b_graph" \
  > "${RAW_DIR}/perf_route_b_graph.log" 2>&1

timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  "${COMMON_TEXT_ARGS[@]}" \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output "${RAW_DIR}/text_smoke_prefix_off.json" \
  > "${RAW_DIR}/text_smoke_prefix_off.log" 2>&1

timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  "${COMMON_TEXT_ARGS[@]}" \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --output "${RAW_DIR}/text_smoke_phase1_prefix_on.json" \
  > "${RAW_DIR}/text_smoke_phase1_prefix_on.log" 2>&1

timeout 900 torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  "${COMMON_TEXT_ARGS[@]}" \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output "${RAW_DIR}/text_smoke_route_b_graph.json" \
  > "${RAW_DIR}/text_smoke_route_b_graph.log" 2>&1

python "${MILESTONE_DIR}/scripts/quantify_swa_tail_guard.py" \
  --milestone-dir "${MILESTONE_DIR}"

python "${MILESTONE_DIR}/scripts/summarize_final_gate.py" \
  --milestone-dir "${MILESTONE_DIR}"

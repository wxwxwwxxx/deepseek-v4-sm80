#!/usr/bin/env bash
set -euo pipefail

cd /workspace/mini-sglang

MILESTONE_DIR=performance_milestones/target08_prefix_cache_logit_metadata_correctness
MODEL_PATH=${MODEL_PATH:-/models/DeepSeek-V4-Flash}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES
export MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --page-size 256
  --num-pages 128
  --max-seq-len 1024
  --max-extend-tokens 20000
  --max-running-req 16
)

rm -rf \
  "${MILESTONE_DIR}/raw/prefix_off_graph" \
  "${MILESTONE_DIR}/raw/prefix_on_graph" \
  "${MILESTONE_DIR}/raw/prefix_on_eager" \
  "${MILESTONE_DIR}/summaries"

torchrun --standalone --nproc_per_node=8 \
  "${MILESTONE_DIR}/scripts/run_dsv4_prefix_logit_probe.py" \
  "${COMMON_ARGS[@]}" \
  --mode prefix_off \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir "${MILESTONE_DIR}/raw/prefix_off_graph"

torchrun --standalone --nproc_per_node=8 \
  "${MILESTONE_DIR}/scripts/run_dsv4_prefix_logit_probe.py" \
  "${COMMON_ARGS[@]}" \
  --mode prefix_on \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir "${MILESTONE_DIR}/raw/prefix_on_graph"

torchrun --standalone --nproc_per_node=8 \
  "${MILESTONE_DIR}/scripts/run_dsv4_prefix_logit_probe.py" \
  "${COMMON_ARGS[@]}" \
  --mode prefix_on_eager \
  --output-dir "${MILESTONE_DIR}/raw/prefix_on_eager"

python "${MILESTONE_DIR}/scripts/summarize_dsv4_prefix_logit_probe.py" \
  --prefix-off "${MILESTONE_DIR}/raw/prefix_off_graph" \
  --prefix-on "${MILESTONE_DIR}/raw/prefix_on_graph" \
  --prefix-on-eager "${MILESTONE_DIR}/raw/prefix_on_eager" \
  --output-dir "${MILESTONE_DIR}/summaries" \
  --atol 2e-2 \
  --rtol 2e-2

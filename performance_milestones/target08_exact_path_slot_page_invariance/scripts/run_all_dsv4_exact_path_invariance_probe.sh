#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${ROOT}/performance_milestones/target08_exact_path_slot_page_invariance"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --page-size 256
  --num-pages 128
  --max-seq-len 1024
  --max-extend-tokens 20000
  --max-running-req 16
  --probe-max-tokens 2
  --prelude-max-tokens 1
  --capture-activations
  --max-activation-rows 4
)

cd "${ROOT}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  --mode eager \
  --output-dir "${OUT}/raw/eager" \
  "${COMMON_ARGS[@]}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  --mode graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir "${OUT}/raw/graph" \
  "${COMMON_ARGS[@]}"

python performance_milestones/target08_exact_path_slot_page_invariance/scripts/summarize_dsv4_exact_path_invariance_probe.py \
  --eager "${OUT}/raw/eager" \
  --graph "${OUT}/raw/graph" \
  --output-dir "${OUT}/summaries" \
  --atol 2e-2 \
  --rtol 2e-2 \
  --activation-atol 2e-2 \
  --activation-rtol 2e-2

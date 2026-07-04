#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${1:-${ROOT}/performance_milestones/target08_q_path_same_shape_same_input_invariance/raw/same_shape_eager}"

cd "${ROOT}"

export MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
export MINISGL_DISABLE_OVERLAP_SCHEDULING=1

torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_q_path_same_shape_same_input_invariance/scripts/run_target08_197_q_path_probe.py \
  --mode eager \
  --output-dir "${OUT_DIR}" \
  --model-path /models/DeepSeek-V4-Flash \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 1024 \
  --max-extend-tokens 20000 \
  --max-running-req 16 \
  --probe-max-tokens 2 \
  --prelude-max-tokens 1 \
  --capture-activations \
  --debug-attention-components \
  --max-activation-rows 4

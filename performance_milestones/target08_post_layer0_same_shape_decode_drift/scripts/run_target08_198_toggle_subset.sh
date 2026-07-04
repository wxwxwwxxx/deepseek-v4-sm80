#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 LABEL [DISABLE_TOGGLES]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LABEL="$1"
DISABLE_TOGGLES="${2:-}"
OUT_DIR="${ROOT}/performance_milestones/target08_post_layer0_same_shape_decode_drift/raw/${LABEL}"

cd "${ROOT}"

MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES="${DISABLE_TOGGLES}" \
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1 \
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
  --probe-max-tokens 3 \
  --prelude-max-tokens 1 \
  --capture-activations \
  --debug-attention-components \
  --max-activation-rows 4 \
  --scenarios \
    identical_prompts_batch \
    target_slot0_fixed_fillers \
    target_slot1_fixed_fillers \
    target_slot3_fixed_fillers \
    target_slot0_altA_fillers

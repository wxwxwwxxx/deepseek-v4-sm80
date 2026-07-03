#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${ROOT}/performance_milestones/target08_exact_path_slot_page_invariance"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"

SCENARIOS=(
  identical_prompts_batch
  single_target_alone
  target_in_batch_slot0
  target_in_batch_slot1
  target_in_batch_slot2
  target_in_batch_slot3
  swa_boundary_127_128_129_bs3
)

TOGGLES=(
  MINISGL_DSV4_SM80_DECODE_METADATA_DEFOREST
  MINISGL_DSV4_SM80_REPLAY_METADATA_COPY
  MINISGL_DSV4_SM80_INDEXER_FP8_CACHE
  MINISGL_DSV4_SM80_SPARSE_ATTN_BF16
  MINISGL_DSV4_SM80_SPARSE_SPLITK_BF16
  MINISGL_DSV4_SM80_COMPRESS
  MINISGL_DSV4_SM80_FUSED_TOPK_SWA_INDICES
  MINISGL_DSV4_SM80_FUSED_Q_KV_NORM_ROPE_STORE
  MINISGL_DSV4_SM80_COMPRESS_STORE
  MINISGL_DSV4_SM80_FUSED_WQA_WKV_SHARED_ACT
  MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE
  MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE
  MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE
  MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE
  MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE
  MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE
  MINISGL_DSV4_SM80_MOE_VLLM_RUNNER
  MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND
)

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --page-size 256
  --num-pages 128
  --max-seq-len 1024
  --max-extend-tokens 20000
  --max-running-req 16
  --probe-max-tokens 2
  --prelude-max-tokens 1
  --scenarios "${SCENARIOS[@]}"
)

cd "${ROOT}"
mkdir -p "${OUT}/raw/toggles"

for toggle in "${TOGGLES[@]}"; do
  safe_toggle="${toggle//[^A-Za-z0-9_]/_}"
  run_dir="${OUT}/raw/toggles/disable_${safe_toggle}"
  echo "==> Running graph probe with ${toggle} disabled"
  MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES="${toggle}" \
  torchrun --standalone --nproc_per_node="${NPROC}" \
    performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
    --mode graph \
    --cuda-graph-bs 1 2 4 8 16 \
    --output-dir "${run_dir}" \
    "${COMMON_ARGS[@]}"

  python performance_milestones/target08_exact_path_slot_page_invariance/scripts/summarize_dsv4_exact_path_invariance_probe.py \
    --graph "${run_dir}" \
    --output-dir "${OUT}/summaries/toggles/disable_${safe_toggle}" \
    --atol 2e-2 \
    --rtol 2e-2
done

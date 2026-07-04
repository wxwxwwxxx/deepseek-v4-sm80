#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${ROOT}/performance_milestones/target08_batched_attention_indexer_row_coupling"
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V4-Flash}"
NPROC="${NPROC:-8}"

SCENARIOS=(
  single_target_alone
  identical_prompts_batch
  target_in_batch_slot0
  target_in_batch_slot1
  target_in_batch_slot2
  target_in_batch_slot3
  target_same_length_fillers
  target_c4_boundary_fillers
  target_c128_boundary_fillers
  target_swa_boundary_fillers
  target_table_row_after_0_dummy
  target_table_row_after_2_dummy
  target_table_row_after_3_dummy
  target_physical_page_none
  target_physical_page_one_page
  target_physical_page_mixed_pages
  swa_boundary_127_128_129_bs3
  page_boundary_255_256_257_258
  c4_c128_boundary_lengths
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
  --capture-activations
  --debug-attention-components
  --max-activation-rows 4
  --scenarios "${SCENARIOS[@]}"
)

cd "${ROOT}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  --mode eager \
  --output-dir "${OUT}/raw/eager" \
  "${COMMON_ARGS[@]}"

MINISGL_DSV4_CUDA_GRAPH_EXACT_BS_ONLY=1 \
torchrun --standalone --nproc_per_node="${NPROC}" \
  performance_milestones/target08_exact_path_slot_page_invariance/scripts/run_dsv4_exact_path_invariance_probe.py \
  --mode graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output-dir "${OUT}/raw/graph_exact_bs_guard" \
  "${COMMON_ARGS[@]}"

python performance_milestones/target08_batched_attention_indexer_row_coupling/scripts/summarize_target08_196_row_coupling.py \
  --eager "${OUT}/raw/eager" \
  --graph "${OUT}/raw/graph_exact_bs_guard" \
  --output-dir "${OUT}/summaries" \
  --atol 2e-2 \
  --rtol 2e-2 \
  --activation-atol 2e-2 \
  --activation-rtol 2e-2

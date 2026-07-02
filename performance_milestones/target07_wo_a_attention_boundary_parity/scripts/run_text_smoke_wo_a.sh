#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

VARIANT="target0762_woabf16bmmcache"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants "${VARIANT}" \
  --page-size 256 \
  --output performance_milestones/target07_wo_a_attention_boundary_parity/raw/text_smoke_wo_a_bf16_bmm_cache.json

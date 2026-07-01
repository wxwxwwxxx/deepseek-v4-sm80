#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/mini-sglang}
OUT=${OUT:-"$ROOT/performance_milestones/target07_subgraph_parity/summaries"}

cd "$ROOT"
python performance_milestones/target07_subgraph_parity/scripts/mini_subgraph_microbench.py \
  --quick \
  --output "$OUT/mini_subgraph_microbench.json"

PYTHONPATH=/workspace/vllm-dsv4-docker /workspace/venvs/vllm-dsv4/bin/python \
  performance_milestones/target07_subgraph_parity/scripts/vllm_subgraph_microbench.py \
  --quick \
  --output "$OUT/vllm_subgraph_microbench.json"

torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target07_subgraph_parity/scripts/comm_microbench.py \
  --quick \
  --output "$OUT/comm_microbench_torch_nccl.json"

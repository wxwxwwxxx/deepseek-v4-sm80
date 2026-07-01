#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/mini-sglang}
OUT=${OUT:-"$ROOT/performance_milestones/target07_attention_indexer_cache_runtime"}
TOKENS=${TOKENS:-4}
HISTORY=${HISTORY:-4096}
PAGE_SIZE=${PAGE_SIZE:-256}

cd "$ROOT"

python "$OUT/scripts/probe_dispatch_backends.py" \
  --output "$OUT/summaries/dispatch_backend_report.json"

PYTHONPATH="$ROOT/python" python "$OUT/scripts/mini_attention_indexer_cache_microbench.py" \
  --quick \
  --tokens "$TOKENS" \
  --history "$HISTORY" \
  --page-size "$PAGE_SIZE" \
  --output "$OUT/raw/mini_attention_indexer_cache_microbench_t${TOKENS}_h${HISTORY}.json"

PYTHONPATH=/workspace/vllm-dsv4-docker /workspace/venvs/vllm-dsv4/bin/python \
  "$OUT/scripts/vllm_attention_indexer_cache_microbench.py" \
  --quick \
  --tokens "$TOKENS" \
  --history "$HISTORY" \
  --page-size "$PAGE_SIZE" \
  --output "$OUT/raw/vllm_attention_indexer_cache_microbench_t${TOKENS}_h${HISTORY}.json"

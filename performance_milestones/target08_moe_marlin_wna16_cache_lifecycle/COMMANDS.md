# TARGET 08.34 Commands

Run from `/workspace/mini-sglang`.

## Backend Check

```bash
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 PYTHONPATH=python python - <<'PY'
import os
from minisgl.kernel import deepseek_v4 as d
print("backend", d.dsv4_moe_expert_backend())
print("explicit_env", os.environ.get(d.DSV4_SM80_MOE_EXPERT_BACKEND_ENV))
PY
```

Observed:

```text
backend marlin_wna16
explicit_env None
```

## Current Marlin WNA16

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1 \
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/raw \
MINISGL_DSV4_AUDIT_RUN_LABEL=current_marlin_bs16 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --max-seq-len 32768 --max-running-req 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 16 \
  --smoke --batch-size 16 --prompt-len 16 --decode-len 1 \
  --repeats 1 --warmup-repeats 0 \
  --output-dir /tmp/dsv4_target0834_current_marlin_bs16 \
  --keep-going
```

## Forced Grouped FP4

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1 \
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/raw \
MINISGL_DSV4_AUDIT_RUN_LABEL=forced_grouped_fp4_bs16 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_groupedfp4 \
  --page-size 256 --num-pages 128 \
  --max-seq-len 32768 --max-running-req 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 16 \
  --smoke --batch-size 16 --prompt-len 16 --decode-len 1 \
  --repeats 1 --warmup-repeats 0 \
  --output-dir /tmp/dsv4_target0834_forced_grouped_fp4_bs16 \
  --keep-going
```

## Prebuild Marlin WNA16

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1 \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1 \
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/raw \
MINISGL_DSV4_AUDIT_RUN_LABEL=prebuild_marlin_bs16 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --max-seq-len 32768 --max-running-req 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 16 \
  --smoke --batch-size 16 --prompt-len 16 --decode-len 1 \
  --repeats 1 --warmup-repeats 0 \
  --output-dir /tmp/dsv4_target0834_prebuild_marlin_bs16 \
  --keep-going
```

## Prebuild And Release Original FP4

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1 \
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1 \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1 \
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/raw \
MINISGL_DSV4_AUDIT_RUN_LABEL=prebuild_release_marlin_bs16_decode2 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --max-seq-len 32768 --max-running-req 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 16 \
  --smoke --batch-size 16 --prompt-len 16 --decode-len 2 \
  --repeats 1 --warmup-repeats 0 \
  --output-dir /tmp/dsv4_target0834_prebuild_release_marlin_bs16_decode2 \
  --keep-going
```

## Summary Regeneration

```bash
MINISGL_DSV4_SM80_INDEXER_FP8_CACHE=1 \
python performance_milestones/target08_moe_marlin_wna16_cache_lifecycle/scripts/summarize_moe_marlin_lifecycle.py \
  --milestone-dir performance_milestones/target08_moe_marlin_wna16_cache_lifecycle \
  --tp-size 8 \
  --page-size 256 \
  --indexer-fp8-cache
```

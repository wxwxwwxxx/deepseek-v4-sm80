# Commands

All commands were run from `/workspace/mini-sglang`.

## Static Checks

```bash
python -m py_compile \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/models/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

pytest -q \
  tests/models/test_deepseek_v4_forward_fallback.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py -q
```

## Text Smoke Baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1 \
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_marlin_wna16_release_preset_promotion/raw \
MINISGL_DSV4_AUDIT_RUN_LABEL=text_smoke_nonrelease_baseline \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_preset_promotion/raw/text_smoke_nonrelease_baseline.json \
  > performance_milestones/target08_marlin_wna16_release_preset_promotion/raw/text_smoke_nonrelease_baseline.log 2>&1
```

## Text Smoke Prebuild-Only

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1 \
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_marlin_wna16_release_preset_promotion/raw \
MINISGL_DSV4_AUDIT_RUN_LABEL=text_smoke_prebuild_only \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_prebuild \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_preset_promotion/raw/text_smoke_prebuild_only.json \
  > performance_milestones/target08_marlin_wna16_release_preset_promotion/raw/text_smoke_prebuild_only.log 2>&1
```

## Text Smoke Release

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_DEBUG=1 \
MINISGL_DSV4_WARMUP_FORWARD_MEMORY_DEBUG=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_marlin_wna16_release_preset_promotion/raw \
MINISGL_DSV4_AUDIT_RUN_LABEL=text_smoke_release_sync \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_preset_promotion/raw/text_smoke_release_sync.json \
  > performance_milestones/target08_marlin_wna16_release_preset_promotion/raw/text_smoke_release_sync.log 2>&1
```

## Gates Not Run

The prefix smoke and 4096x128 / 4096x1024 macro gates were intentionally not run after `text_smoke_release_sync` failed text sanity. This follows the target's hard blocker: release must not cause text smoke corruption.

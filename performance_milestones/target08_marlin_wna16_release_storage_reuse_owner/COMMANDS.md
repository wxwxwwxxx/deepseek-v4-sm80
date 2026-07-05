# Commands

All commands were run from `/workspace/mini-sglang` on TP8 A100 with page size `256`.

## Immediate Release Owner Ledger

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/release_eager_ledger \
MINISGL_DSV4_AUDIT_RUN_LABEL=release_eager_ledger \
MINISGL_DSV4_MARLIN_WNA16_RELEASE_LEDGER_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_LAYER2_OWNER_PROBE=1 \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_eager_ledger.json
```

## Release After KV Allocation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_AUDIT_LOG_DIR=performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/release_after_kv_ledger \
MINISGL_DSV4_AUDIT_RUN_LABEL=release_after_kv_ledger \
MINISGL_DSV4_MARLIN_WNA16_RELEASE_LEDGER_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_LAYER2_OWNER_PROBE=1 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_kv_alloc \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_after_kv_ledger.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_kv_alloc \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_after_kv_graph.json
```

## Timing Ladder

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_warmup_forward \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_before_warmup_graph.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_warmup_forward \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_after_warmup_graph.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_graph_capture \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_after_graph_capture.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_first_decode \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_after_first_decode_graph.json
```

## Baseline And Prebuild-Only

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory dsv4_sm80_a100_victory_marlin_prebuild \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_baseline_prebuild_graph.json
```

## Poison

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_KEEP_HIDDEN_REF=1 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_POISON_HIDDEN_REF_PATTERN=zero \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_hidden_ref_poison_zero.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_KEEP_HIDDEN_REF=1 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_POISON_HIDDEN_REF_PATTERN=nan \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_hidden_ref_poison_nan.json
```

## Quarantine

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS=1 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_PATTERN=zero \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_quarantine_all_zero.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS=1 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_PATTERN=zero \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_BYTES=6.375GiB \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_quarantine_6p375gib.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS=1 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_PATTERN=zero \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_BYTES=3.1875GiB \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_quarantine_3p1875gib.json
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_RELEASED_BLOCKS=1 \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_PATTERN=deterministic \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_QUARANTINE_BYTES=3.1875GiB \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-tokens 8 \
  --disable-dsv4-cuda-graph \
  --output performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/text_smoke_release_quarantine_3p1875gib_deterministic.json
```

## Static And Unit Verification

```bash
python -m py_compile \
  python/minisgl/utils/dsv4_memory_debug.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/engine/graph.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py
```

```bash
pytest -q \
  tests/models/test_deepseek_v4_forward_fallback.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py
```

# Commands

All commands were run from `/workspace/mini-sglang`.

Common settings:

```bash
OUT=performance_milestones/target08_marlin_wna16_release_correctness_attribution
RAW=$OUT/raw
MODEL=/models/DeepSeek-V4-Flash
DEVICES=0,1,2,3,4,5,6,7
```

## Reused TARGET 08.35 Smoke

The baseline/prebuild/release three-way text smoke artifacts were reused from:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/raw/
```

This target has a symlink:

```text
performance_milestones/target08_marlin_wna16_release_correctness_attribution/raw_08_35_smoke
```

The original commands are recorded in:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/COMMANDS.md
```

## Release Graph/Eager Split

Graph enabled, max token ladder:

```bash
for MT in 1 2 4 16; do
  CUDA_VISIBLE_DEVICES=$DEVICES \
  MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
  MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG=1 \
  MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_LAYERS=0,21,42 \
  MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_MAX_FORWARD_LOGS=2 \
  MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
  MINISGL_DSV4_AUDIT_RUN_LABEL=release_graph_mt${MT} \
  torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path $MODEL \
    --variants dsv4_sm80_a100_victory_marlin_release \
    --page-size 256 \
    --num-pages 128 \
    --max-seq-len 32768 \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 8 16 \
    --max-tokens $MT \
    --output $RAW/text_smoke_release_graph_mt${MT}.json \
    > $RAW/text_smoke_release_graph_mt${MT}.log 2>&1
done
```

Graph disabled:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_LAYERS=0,21,42 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_MAX_FORWARD_LOGS=2 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=release_eager_mt16_rerun \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --disable-dsv4-cuda-graph \
  --max-tokens 16 \
  --output $RAW/text_smoke_release_eager_mt16_rerun.json \
  > $RAW/text_smoke_release_eager_mt16_rerun.log 2>&1
```

Note: `text_smoke_release_eager_mt16.json/.log` is kept as a failed early attempt before the text smoke harness was fixed to apply variant env before LLM construction.

Graph enabled with greedy-sample capture disabled:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_GRAPH_CAPTURE_STAGE_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_LAYERS=0,21,42 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_MAX_FORWARD_LOGS=2 \
MINISGL_DSV4_PREFIX_DEBUG_DIR=$RAW/logit_release_graph_nogreedy \
MINISGL_DSV4_PREFIX_DEBUG_TOPK=5 \
MINISGL_DSV4_PREFIX_DEBUG_SAVE_FULL_LOGITS=1 \
MINISGL_DSV4_PREFIX_DEBUG_MAX_BATCHES=8 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=logit_release_graph_nogreedy \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --disable-cuda-graph-greedy-sample \
  --max-tokens 6 \
  --output $RAW/text_smoke_logit_release_graph_nogreedy.json \
  > $RAW/text_smoke_logit_release_graph_nogreedy.log 2>&1
```

The matching prebuild-only trace used the same debug settings with:

```text
MINISGL_DSV4_PREFIX_DEBUG_DIR=$RAW/logit_prebuild_graph_nogreedy
MINISGL_DSV4_AUDIT_RUN_LABEL=logit_prebuild_graph_nogreedy
--variants dsv4_sm80_a100_victory_marlin_prebuild
--output $RAW/text_smoke_logit_prebuild_graph_nogreedy.json
```

## Logit Parity Ladder

```bash
python benchmark/offline/deepseek_v4_logit_parity_from_prefix_debug.py \
  --prebuild-debug-dir $RAW/logit_prebuild_graph_nogreedy \
  --release-debug-dir $RAW/logit_release_graph_nogreedy \
  --rank 0 \
  --topk 5 \
  --output-json $RAW/logit_parity_graph_nogreedy_rank0.json \
  --output-md $OUT/logit_parity_ladder.md
```

## MoE Packed-Cache Micro Parity

The first attempt failed because the local package path was not set; the kept successful rerun used:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
PYTHONPATH=/workspace/mini-sglang/python \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_marlin_wna16_release_micro_parity.py \
  --model-path $MODEL \
  --variant dsv4_sm80_a100_victory_marlin_prebuild \
  --tensor-parallel-size 8 \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --layers 0 21 42 \
  --tokens 8 \
  --topk 8 \
  --allocator-pressure-mib 768 \
  --output $RAW/moe_micro_parity.json \
  > $RAW/moe_micro_parity_rerun.log 2>&1
```

`moe_micro_parity.md` was generated from `raw/moe_micro_parity.json`.

## Release Lifetime A/B

Force prepacked path while raw attributes/storage remain present:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_FORCE_PREPACKED_WITH_RAW_PRESENT=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=ab_force_prepacked_raw_present \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_prebuild \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --max-tokens 16 \
  --output $RAW/text_smoke_ab_force_prepacked_raw_present.json \
  > $RAW/text_smoke_ab_force_prepacked_raw_present.log 2>&1
```

Keep hidden references after normal attrs are removed:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_KEEP_HIDDEN_REF=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_LAYERS=0,21,42 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=ab_keep_hidden_ref \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --max-tokens 16 \
  --output $RAW/text_smoke_ab_keep_hidden_ref.json \
  > $RAW/text_smoke_ab_keep_hidden_ref.log 2>&1
```

Release after KV allocation and graph capture:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_AFTER_GRAPH_CAPTURE=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_DEBUG=1 \
MINISGL_DSV4_MARLIN_WNA16_CACHE_INTEGRITY_LAYERS=0,21,42 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=ab_release_after_capture \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --max-tokens 16 \
  --output $RAW/text_smoke_ab_release_after_capture.json \
  > $RAW/text_smoke_ab_release_after_capture.log 2>&1
```

Weights-only and scales-only:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_WEIGHTS_ONLY=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=ab_release_weights_only \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --max-tokens 16 \
  --output $RAW/text_smoke_ab_release_weights_only.json \
  > $RAW/text_smoke_ab_release_weights_only.log 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_SCALES_ONLY=1 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=ab_release_scales_only \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_release \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --allow-dsv4-cuda-graph \
  --cuda-graph-bs 1 2 4 8 16 \
  --max-tokens 16 \
  --output $RAW/text_smoke_ab_release_scales_only.json \
  > $RAW/text_smoke_ab_release_scales_only.log 2>&1
```

Partial layer release examples:

```bash
for FILTER in 0 0-7 0-15 0-20 21-42; do
  LABEL=$(echo "$FILTER" | tr '-' '_')
  CUDA_VISIBLE_DEVICES=$DEVICES \
  MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_LAYER_FILTER=$FILTER \
  MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
  MINISGL_DSV4_AUDIT_RUN_LABEL=ab_release_layers_${LABEL} \
  torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
    --model-path $MODEL \
    --variants dsv4_sm80_a100_victory_marlin_release \
    --page-size 256 \
    --num-pages 128 \
    --max-seq-len 32768 \
    --allow-dsv4-cuda-graph \
    --cuda-graph-bs 1 2 4 8 16 \
    --max-tokens 16 \
    --output $RAW/text_smoke_ab_release_layers_${LABEL}.json \
    > $RAW/text_smoke_ab_release_layers_${LABEL}.log 2>&1
done
```

The single-layer case is also saved as `text_smoke_ab_release_layer_0.*`.

## Activation Divergence

Prebuild eager trace:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES \
MINISGL_DSV4_PREFIX_DEBUG_DIR=$RAW/activation_prebuild_eager_mt4 \
MINISGL_DSV4_PREFIX_DEBUG_TOPK=5 \
MINISGL_DSV4_PREFIX_DEBUG_SAVE_FULL_LOGITS=1 \
MINISGL_DSV4_PREFIX_DEBUG_MAX_BATCHES=8 \
MINISGL_DSV4_PREFIX_DEBUG_ACTIVATIONS=1 \
MINISGL_DSV4_PREFIX_DEBUG_MAX_ACTIVATION_ROWS=4 \
MINISGL_DSV4_AUDIT_LOG_DIR=$RAW \
MINISGL_DSV4_AUDIT_RUN_LABEL=activation_prebuild_eager_mt4 \
torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path $MODEL \
  --variants dsv4_sm80_a100_victory_marlin_prebuild \
  --page-size 256 \
  --num-pages 128 \
  --max-seq-len 32768 \
  --disable-dsv4-cuda-graph \
  --max-tokens 4 \
  --output $RAW/text_smoke_activation_prebuild_eager_mt4.json \
  > $RAW/text_smoke_activation_prebuild_eager_mt4.log 2>&1
```

Release eager trace used the same env with:

```text
MINISGL_DSV4_PREFIX_DEBUG_DIR=$RAW/activation_release_eager_mt4
MINISGL_DSV4_AUDIT_RUN_LABEL=activation_release_eager_mt4
--variants dsv4_sm80_a100_victory_marlin_release
--output $RAW/text_smoke_activation_release_eager_mt4.json
```

The comparison artifact is:

```text
raw/activation_divergence_eager_mt4_rank0.json
activation_divergence.md
```

## Static Checks

```bash
python -m py_compile \
  python/minisgl/utils/dsv4_memory_debug.py \
  python/minisgl/models/deepseek_v4.py \
  python/minisgl/engine/engine.py \
  python/minisgl/engine/graph.py \
  benchmark/offline/deepseek_v4_text_smoke.py \
  benchmark/offline/deepseek_v4_logit_parity_from_prefix_debug.py \
  benchmark/offline/deepseek_v4_marlin_wna16_release_micro_parity.py
```

Focused pytest command:

```bash
pytest -q \
  tests/models/test_deepseek_v4_forward_fallback.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py -q
```

## Gates Not Run

4096x128 and 4096x1024 macro runs were intentionally not run. Release text sanity is still corrupt, so macro throughput would not be valid promotion evidence.

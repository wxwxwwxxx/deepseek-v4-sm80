# Minimal Reproducer

## Smallest Reproducer Used

The smallest reliable repro found here is the existing deterministic text smoke with:

- TP8
- page size `256`
- `--num-pages 128`
- `--max-tokens 8`
- CUDA graph disabled
- `dsv4_sm80_a100_victory_marlin_release`
- owner/freed ledger enabled

Command:

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

Observed prompt 0 token ids:

```text
[20, 940, 223, 0, 0, 0, 0, 0]
```

The same command with:

```bash
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_kv_alloc
```

passes and emits:

```text
[20, 940, 223, 20, 223, 15120, 223, 22]
```

## Why Not A Smaller Layer-Only Repro

The owner is not a single layer computation that can be isolated by calling one layer with
fixed inputs.  The failure depends on full-model CUDA allocator geometry:

1. Marlin WNA16 cache prebuild allocates packed persistent expert caches.
2. Immediate release returns raw expert-weight storage to the CUDA allocator.
3. DSV4 KV/component pools allocate into those exact freed address ranges.
4. Decode later exposes corruption around layer2/indexer/logits.

A smaller layer-only reproducer would miss the KV/component allocation owner that makes the
bug visible.

## Minimal Pass/Fail Toggle

Use only this timing toggle to flip the repro:

| Env | Expected status |
| --- | --- |
| unset or `MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=model_prepare` | warn/fail text sanity |
| `MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=after_kv_alloc` | pass |

This is the clearest lifecycle boundary for future debugging.

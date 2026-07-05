# TARGET 08.33 Commands

## Capture-Only Current Width

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4 \
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_indexer_capture_static_width_audit/scripts/capture_width_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --width-mode current \
  --run-label current_bs16_real \
  --page-size 256 --num-pages 128 \
  --max-seq-len 32768 --max-running-req 16 \
  --cuda-graph-bs 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership
```

## Capture-Only Table-Width Counterfactual

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4 \
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1 \
torchrun --standalone --nproc_per_node=8 \
  performance_milestones/target08_indexer_capture_static_width_audit/scripts/capture_width_probe.py \
  --model-path /models/DeepSeek-V4-Flash \
  --width-mode table_width \
  --run-label table_width_bs16_real \
  --page-size 256 --num-pages 128 \
  --max-seq-len 32768 --max-running-req 16 \
  --cuda-graph-bs 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership
```

`table_width` is a diagnostic counterfactual only. It is unsafe to promote for
page-based indexer tables because it can truncate valid C4/indexer positions.

## Text Smoke / Replay Check

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1 \
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4 \
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime \
  --page-size 256 --num-pages 128 \
  --max-seq-len 1024 --max-extend-tokens 4096 \
  --allow-dsv4-cuda-graph --cuda-graph-bs 16 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --output performance_milestones/target08_indexer_capture_static_width_audit/raw/text_smoke_prefix_routeb_current_bs16.json
```

Note: an initial attempt used the historical prompt option
`--verify-dsv4-route-b-cache`; the current text-smoke CLI no longer accepts
that flag, so the command above is the successful run.

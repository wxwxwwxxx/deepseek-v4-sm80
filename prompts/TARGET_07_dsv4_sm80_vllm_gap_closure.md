# TARGET 07: DeepSeek V4 SM80 vLLM Gap Closure

## Status

Closed.

TARGET 07 established the first stable DeepSeek V4 Flash A100/sm80 performance
milestone for mini-sglang and crossed the original old-framework victory line.

Do not start new TARGET 07 implementation threads by default.  Use this file as
the project-history index, and use `prompts/archive/target07/` only when exact
historical prompt scope or command details are needed.

Next active target:

```text
prompts/TARGET_08_radix_prefix_dsv4.md
```

## Original Win Condition

- Model: `/models/DeepSeek-V4-Flash`
- Hardware: TP8, single node, 8x A100 sm80
- Page size: `256`
- Workload: `4096` input tokens, `1024` output tokens, batch size `4`
- Baseline to beat: old vLLM-based serving line, `114.07 output tok/s`
- Default path should remain exact unless a later precision target explicitly
  proves and accepts a quality tradeoff.

## Final TARGET 07 Milestone

Promoted default milestone:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
```

Representative post-07.78 stable retest from
`performance_milestones/target07_post_0778_roofline_bottleneck_reset/`:

| Workload | Output tok/s mean | Decode tok/s mean | TTFT mean | Prefill fwd mean | Decode fwd mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096/1024/batch4 | `131.7561` | `169.7583` | `4.9550s` | `4.2353s` | `24.1049s` |
| 4096/128/batch4 | `62.3925` | `168.8358` | `4.9489s` | `4.2474s` | `3.0089s` |

Graph gate:

| Workload | Graph replay | Eager decode |
| --- | ---: | ---: |
| 4096/1024/batch4 | `4092` | `0` |
| 4096/128/batch4 | `508` | `0` |

This is above the old `114.07 output tok/s` line.  Fresh vLLM offline runs were
still much faster in some reports, but vLLM's strong path uses a different
precision/cache policy, including `deepseek_v4_fp8`, packed `fp8_ds_mla` cache
pieces, and additional runtime machinery.  TARGET 07 deliberately kept the
promoted mini path exact by default.

## Promoted Stack

The final promoted bundle is a composition of the following major wins:

- mini-owned Marlin WNA16 MoE expert backend for model-native MXFP4 experts;
- global topk/lens metadata path;
- BF16 two-scope gather/mask plus split-K sparse decode;
- DSV4 decode CUDA graph replay;
- FP8 indexer cache backend and fused FP8 activation quant helper;
- cached BF16 projection stack for `attn.q_wqb`, `attn.wo_b`,
  `indexer.wq_b`, and `attn.wo_a` grouped BMM;
- cached BF16 shared expert gate/up/down projection weights;
- page size `256`, TP8, fixed benchmark `--num-pages 128`.

Important non-promoted opt-ins:

- `dsv4_sm80_a100_victory_densefp8marlinproj`: speed-neutral, saves about
  `807 MB/rank`; keep as a memory/capacity opt-in, not default speed path.
- Decode metadata deforestation: correct and locally promising, but not enough
  macro/profile gain to promote.
- HC graph cleanup and BF16 small-GEMM pretranspose paths: useful audits, not
  default wins.

## Evolution Path

| Phase | Archived prompts | Main artifacts | Decision |
| --- | --- | --- | --- |
| Foundation and fair parity | `TARGET_07.1`, `07.2`, `07.25`, summarized by archived `TARGET_07.10` | `target07_vllm_gap`, `target07_comm_graph`, `target07_subgraph_parity` | Build fair mini/vLLM comparison, enable graph/comm observability, rank MoE first. |
| MoE backend | `TARGET_07.3` through `07.391`, summarized by archived `TARGET_07.20` | `target07_moe_*`, `target07_marlin_*` | Local MoE wrapper cleanup was not enough; mini-owned Marlin WNA16 solved the main exact MoE gap. |
| Attention/indexer/cache exact path | `TARGET_07.392` through `07.395`, summarized by archived `TARGET_07.30` | `target07_attention_indexer_cache_runtime`, `target07_bf16_sparse_decode_splitk` | BF16 split-K sparse decode matched the comparable vLLM sparse decode probe; do not keep polishing sparse decode without fresh evidence. |
| Runtime, graph, and projection/cache | `TARGET_07.40` through `07.62` | `target07_post_splitk_reprofile`, `target07_projection_gemm_backend_parity`, `target07_cached_bf16_*`, `target07_wo_a_attention_boundary_parity` | Projection/cache boundaries, not more sparse decode, carried mini across the old vLLM serving line. |
| Post-victory cleanup | `TARGET_07.63` through `07.71` | `target07_post_victory_reprofile`, `target07_decode_metadata_deforestation`, `target07_moe_shared_expert_staging_cleanup`, `target07_hc_elementwise_graph_cleanup` | Promote shared expert BF16 cache; keep smaller graph/HC experiments as opt-ins or audits. |
| FP8/Marlin projection research | `TARGET_07.72` through `07.78` | `target07_vllm_quantized_linear_backend_feasibility`, `target07_mini_owned_dense_fp8_marlin_bridge`, `target07_benchmark_lifecycle_repeat_stable_gate` | Dense FP8 Marlin projection is usable and memory-saving, but speed-neutral on the final path. |
| Roofline reset and closeout | `TARGET_07.79` | `target07_post_0778_roofline_bottleneck_reset` | Exact non-prefix speed surfaces are fragmented; start TARGET 08 radix prefix cache. |

## Key Lessons

- Start with vLLM/source parity and fair macro measurements, then implement the
  narrow bottleneck.  This was much more effective than local kernel polishing.
- MoE's winning exact route was not INT8 activation quantization; it was a
  model-native W4A16/MXFP4 Marlin expert backend.
- Sparse decode is no longer the main mini/vLLM gap under the promoted exact
  path.  Reopen it only after a fresh profile puts C4A/C128A attention back in
  the top buckets.
- Dense FP8 Marlin projection is a capacity feature for now.  It should not be
  promoted for speed without a new repeat-stable win.
- Low MFU in the final report is expected for batch4 decode: many small GEMMs,
  sparse MoE, graph replay nodes, communication, metadata, and occupancy effects
  dominate over large dense Tensor Core saturation.
- Automatic KV sizing with `memory_ratio=0.9` currently leaves too little graph
  capture headroom.  Fixed or capped page counts should be used for the next
  feature target until capacity policy is repaired.

## Archive Map

All TARGET 07 execution prompts now live under:

```text
prompts/archive/target07/
```

The most useful historical summaries are:

- `prompts/archive/target07/TARGET_07.10_dsv4_sm80_foundation_history.md`
- `prompts/archive/target07/TARGET_07.20_dsv4_sm80_moe_history.md`
- `prompts/archive/target07/TARGET_07.30_dsv4_sm80_attention_history.md`
- `prompts/archive/target07/TARGET_07.79_dsv4_sm80_post_0778_roofline_bottleneck_reset.md`

Do not use the archived prompts as the main active project map.  Use them only
for exact historical command details, stop rules, and artifact provenance.

## Next Route

Proceed in this order:

1. TARGET 08: DSV4 radix/SWA prefix cache, aligned with vLLM cache-state design
   where useful and guarded by correctness tests.
2. TARGET 09: low-precision research roadmap, including FP8 KV/cache and INT8
   MoE, after prefix-cache baseline exists.
3. TARGET 10: optional attention and communication research, including C4A/C128A
   attention microbenching and PyNCCL/NCCL overlap experiments, only if fresh
   TARGET 08 or TARGET 09 profiles justify them.

# TARGET 09.0 Low Precision Preflight

## Conclusion Summary

Gate: **PASS**.

Recommendation: **TARGET 09.3 FP8 KV/cache parity ledger next**.

Evidence mix:
- Runtime-proven: promoted baseline is active, all four macro scenarios pass, graph replay has zero eager decodes, communication counters are BF16/FP32 as reported below, and memory/page headroom is concrete.
- Source-derived: vLLM/SGLang have DSv4-specific FP8 KV/cache reference paths; INT8 MoE is partial and backend-specific; INT8 communication lacks a CUDA DSv4 path.
- Microbench-proven: none added for this preflight. This target intentionally did not implement or benchmark INT8 MoE, FP8 KV/cache, or INT8 communication.

Run TARGET 09.3 (FP8 KV/cache parity ledger) next. Defer TARGET 09.1 until MoE INT8 has a dedicated microbench and numerical plan, defer TARGET 09.25 because the current CUDA communication path has no DSv4 INT8 protocol, and defer TARGET 09.6 until the FP8 KV/cache ledger clarifies the cache-boundary surface. Do not pause TARGET 09.

## Baseline Command And Environment

Variant:

`dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16`

Macro command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1 timeout 3600 torchrun --standalone --nproc_per_node=8 benchmark/offline/deepseek_v4_perf_matrix.py --model-path /models/DeepSeek-V4-Flash --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 --page-size 256 --num-pages 128 --enable-dsv4-radix-prefix-cache --enable-dsv4-component-loc-ownership --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 --scenarios historical_4096_128_bs4 historical_4096_1024_bs4 serving_mixed_112req_wave16 prefix_multi_112req_wave16 --repeats 1 --warmup-repeats 0 --seed 20260705 --output-dir performance_milestones/target09_low_precision_preflight/raw/promoted_macro_default_four_scenarios --keep-going
```

Owner timing command:

```bash
MINISGL_DSV4_OWNER_TIMING=1 plus the same promoted baseline flags, run once per scenario via scripts/run_owner_timing_single_scenarios.sh.
```

Environment:
- `MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1`
- page size `256`, num pages `128`
- CUDA graph BS `1 2 4 8 16`
- PyNCCL default threshold32m was not passed explicitly; it was activated by the promoted sm80 default. The macro log contains `Defaulting DeepSeek V4 sm80 PyNCCL max buffer size to 32 MiB`.
- Report `communication_backend` records the CPU init group as `gloo`, but `use_pynccl=true` in all aggregate reports.
- GPU: `NVIDIA A100-SXM4-80GB`, CUDA runtime `12.8`, NCCL `2.27.5`, torch `2.9.1+cu128`.
- Git: `d7b5816` on `dsv4-sglang-based`, dirty=True.

## Macro/Profile Results

| scenario | status | elapsed_s | E2E tok/s | decode tok/s | prefill tok/s | graph replay/eager | prefix hit | saved prefill | peak alloc | comm bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | pass | 9.646 | 53.080 | 189.291 | 3187.1 | 127/0 | 0.0% | 0 | 44.31 GiB | 93.43 GB |
| historical_4096_1024_bs4 | pass | 28.778 | 142.329 | 191.808 | 3825.4 | 1023/0 | 0.0% | 0 | 44.31 GiB | 93.43 GB |
| serving_mixed_112req_wave16 | pass | 15.230 | 183.843 | 301.657 | 4477.6 | 441/0 | 0.0% | 0 | 41.56 GiB | 100.08 GB |
| prefix_multi_112req_wave16 | pass | 6.596 | 135.848 | 680.745 | 4210.1 | 49/0 | 41.4% | 49152 | 42.89 GiB | 88.04 GB |

Graph sanity:
- Captured BS from reports: `[16, 8, 4, 2, 1]`
- Graph free memory before/after capture: 55.21 GiB -> 36.42 GiB
- Graph memory delta: 18.78 GiB

## Owner Bottleneck Ranking

Owner timing was collected in separate one-scenario runs. Category values are max-rank label-sum signals, not wall-time percentages, because some owner labels are nested.

| scenario | category | max-rank total signal | captured signal | labels | top label | top label total |
| --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | communication | 3866.3 ms | 3.263 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 1924.1 ms |
| historical_4096_128_bs4 | metadata/runtime | 2277.3 ms | 0.000 ms | 29 | dsv4.prepare.prefill.attention_metadata | 778.6 ms |
| historical_4096_128_bs4 | projection/GEMM | 662.2 ms | 7.586 ms | 112 | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 209.0 ms |
| historical_4096_128_bs4 | MoE | 296.0 ms | 4.774 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 67.6 ms |
| historical_4096_128_bs4 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| historical_4096_128_bs4 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| historical_4096_1024_bs4 | metadata/runtime | 8096.7 ms | 0.000 ms | 29 | dsv4.prepare.decode.attention_metadata | 2847.9 ms |
| historical_4096_1024_bs4 | communication | 4232.7 ms | 3.181 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2346.9 ms |
| historical_4096_1024_bs4 | projection/GEMM | 641.5 ms | 7.583 ms | 112 | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 205.5 ms |
| historical_4096_1024_bs4 | MoE | 283.9 ms | 4.760 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 66.9 ms |
| historical_4096_1024_bs4 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| historical_4096_1024_bs4 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| serving_mixed_112req_wave16 | communication | 4179.8 ms | 15.033 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2421.2 ms |
| serving_mixed_112req_wave16 | metadata/runtime | 4116.0 ms | 0.000 ms | 29 | dsv4.prepare.decode.attention_metadata | 1349.7 ms |
| serving_mixed_112req_wave16 | projection/GEMM | 1877.5 ms | 37.386 ms | 112 | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 667.3 ms |
| serving_mixed_112req_wave16 | MoE | 357.8 ms | 23.457 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 82.5 ms |
| serving_mixed_112req_wave16 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| serving_mixed_112req_wave16 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| prefix_multi_112req_wave16 | communication | 5216.4 ms | 3.402 ms | 4 | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3179.4 ms |
| prefix_multi_112req_wave16 | projection/GEMM | 2864.5 ms | 7.678 ms | 112 | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 1169.0 ms |
| prefix_multi_112req_wave16 | metadata/runtime | 2353.8 ms | 0.000 ms | 29 | dsv4.prepare.prefill.attention_metadata | 847.4 ms |
| prefix_multi_112req_wave16 | MoE | 326.5 ms | 4.826 ms | 132 | dsv4.owner.shared_down.bf16_cache_local_total | 78.7 ms |
| prefix_multi_112req_wave16 | attention | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |
| prefix_multi_112req_wave16 | cache store/gather/dequant | 0.0 ms | 0.000 ms | 0 | not separately exposed by owner timing | 0.0 ms |

Top labels:

| scenario | scope | category | label | max-rank total | captured | count |
| --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 1924.1 ms | 1.471 ms | 3784 |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 802.9 ms | 1.528 ms | 3784 |
| historical_4096_128_bs4 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 778.6 ms | 0.000 ms | 8 |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 687.2 ms | 0.059 ms | 88 |
| historical_4096_128_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 462.2 ms | 0.000 ms | 1024 |
| historical_4096_128_bs4 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 452.1 ms | 0.206 ms | 88 |
| historical_4096_128_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 391.7 ms | 0.000 ms | 8 |
| historical_4096_128_bs4 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 351.3 ms | 0.000 ms | 1016 |
| historical_4096_128_bs4 | cuda | projection/GEMM | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 209.0 ms | 0.022 ms | 88 |
| historical_4096_128_bs4 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 93.0 ms | 1.540 ms | 3784 |
| historical_4096_128_bs4 | cuda | projection/GEMM | dsv4.owner.attn.wo_b.bf16_cache_local_total | 86.8 ms | 1.521 ms | 3784 |
| historical_4096_128_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_write_locs | 72.4 ms | 0.000 ms | 1024 |
| historical_4096_1024_bs4 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 2847.9 ms | 0.000 ms | 8184 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2346.9 ms | 1.465 ms | 3784 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 1114.7 ms | 0.000 ms | 8192 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.direct_graph_metadata.decode.index_buffers | 993.7 ms | 0.000 ms | 8184 |
| historical_4096_1024_bs4 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 782.7 ms | 0.000 ms | 8 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 725.4 ms | 1.430 ms | 3784 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 686.6 ms | 0.060 ms | 88 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_write_locs | 581.5 ms | 0.000 ms | 8192 |
| historical_4096_1024_bs4 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 473.8 ms | 0.226 ms | 88 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 392.2 ms | 0.000 ms | 8 |
| historical_4096_1024_bs4 | cuda | metadata/runtime | dsv4.metadata.decode.make_component_page_tables | 230.7 ms | 0.000 ms | 8192 |
| historical_4096_1024_bs4 | cuda | projection/GEMM | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 205.5 ms | 0.022 ms | 88 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 2421.2 ms | 6.836 ms | 5848 |
| serving_mixed_112req_wave16 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 1349.7 ms | 0.000 ms | 3528 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 854.5 ms | 6.771 ms | 5848 |
| serving_mixed_112req_wave16 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 692.6 ms | 0.000 ms | 56 |
| serving_mixed_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 667.3 ms | 7.651 ms | 5848 |
| serving_mixed_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_activation_quantize | 592.8 ms | 1.594 ms | 5848 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 542.1 ms | 0.338 ms | 136 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 438.0 ms | 0.000 ms | 3584 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 412.8 ms | 0.000 ms | 56 |
| serving_mixed_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 362.0 ms | 1.088 ms | 136 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_component_page_tables | 350.5 ms | 0.000 ms | 3584 |
| serving_mixed_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_write_locs | 253.5 ms | 0.000 ms | 3584 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.attn.wo_b.row_parallel_all_reduce | 3179.4 ms | 1.506 ms | 5848 |
| prefix_multi_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_local_total | 1169.0 ms | 1.558 ms | 5848 |
| prefix_multi_112req_wave16 | cuda | projection/GEMM | dsv4.owner.attn.q_wqb.bf16_cache_activation_quantize | 1115.6 ms | 0.326 ms | 5848 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.moe.reduce_once_all_reduce | 868.3 ms | 1.497 ms | 5848 |
| prefix_multi_112req_wave16 | host | metadata/runtime | dsv4.prepare.prefill.attention_metadata | 847.4 ms | 0.000 ms | 56 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.lm_head_all_gather | 704.6 ms | 0.086 ms | 136 |
| prefix_multi_112req_wave16 | cuda | communication | dsv4.owner.comm.dsv4.embedding_all_reduce | 464.1 ms | 0.313 ms | 136 |
| prefix_multi_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c4_sparse_indices | 366.1 ms | 0.000 ms | 56 |
| prefix_multi_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_c128_indices | 350.3 ms | 0.000 ms | 448 |
| prefix_multi_112req_wave16 | cuda | metadata/runtime | dsv4.metadata.decode.make_component_page_tables | 325.7 ms | 0.000 ms | 448 |
| prefix_multi_112req_wave16 | host | metadata/runtime | dsv4.prepare.decode.attention_metadata | 312.9 ms | 0.000 ms | 392 |
| prefix_multi_112req_wave16 | cuda | projection/GEMM | dsv4.owner.layer0.attn.q_proj.bf16_cache_linear | 214.1 ms | 0.023 ms | 136 |

Interpretation:
- Communication remains the largest owner signal by total timing, mainly `wo_b` and MoE reduce all-reduces.
- Projection/GEMM BF16 cache labels are visible but not the top reason to choose a low-precision target.
- Standalone attention-kernel and cache store/gather/dequant buckets show `0.0 ms` when no separate owner label was exposed; this is instrumentation coverage, not proof of zero physical cost.
- Metadata/runtime labels are still visible in owner timing, but graph replay is stable with zero eager decodes in macro.

## Communication Table

| scenario | label | op | dtype | shape | out_shape | count | bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| historical_4096_128_bs4 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 8 | 1.07 GB |
| historical_4096_128_bs4 | dsv4.lm_head_all_gather | all_gather | float32 | 4x16160 | 32x16160 | 8 | 0.02 GB |
| historical_4096_128_bs4 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| historical_4096_1024_bs4 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| historical_4096_1024_bs4 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 8 | 1.07 GB |
| historical_4096_1024_bs4 | dsv4.lm_head_all_gather | all_gather | float32 | 4x16160 | 32x16160 | 8 | 0.02 GB |
| historical_4096_1024_bs4 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 16384x4096 | 16384x4096 | 344 | 46.17 GB |
| serving_mixed_112req_wave16 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 2496x4096 | 2496x4096 | 2408 | 49.24 GB |
| serving_mixed_112req_wave16 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 2496x4096 | 2496x4096 | 56 | 1.15 GB |
| serving_mixed_112req_wave16 | dsv4.lm_head_all_gather | all_gather | float32 | 16x16160 | 128x16160 | 56 | 0.46 GB |
| serving_mixed_112req_wave16 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 2496x4096 | 2496x4096 | 2408 | 49.24 GB |
| prefix_multi_112req_wave16 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 1024x4096 | 1024x4096 | 2064 | 17.31 GB |
| prefix_multi_112req_wave16 | dsv4.attn.wo_b.row_parallel_projection_all_reduce | all_reduce | bfloat16 | 9216x4096 | 9216x4096 | 344 | 25.97 GB |
| prefix_multi_112req_wave16 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 1024x4096 | 1024x4096 | 48 | 0.40 GB |
| prefix_multi_112req_wave16 | dsv4.embedding_all_reduce | all_reduce | bfloat16 | 9216x4096 | 9216x4096 | 8 | 0.60 GB |
| prefix_multi_112req_wave16 | dsv4.lm_head_all_gather | all_gather | float32 | 16x16160 | 128x16160 | 56 | 0.46 GB |
| prefix_multi_112req_wave16 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 1024x4096 | 1024x4096 | 2064 | 17.31 GB |
| prefix_multi_112req_wave16 | dsv4.v1_moe_reduce_once_all_reduce | all_reduce | bfloat16 | 9216x4096 | 9216x4096 | 344 | 25.97 GB |

Notes:
- Hidden tensor collectives are BF16.
- `lm_head` remains FP32 all-gather.
- The main byte owners are MoE reduce and attention `wo_b` all-reduce.

## Memory Ledger

| item | value | evidence |
| --- | --- | --- |
| KV/cache pages per rank | 2.32 GiB | runtime-reported |
| KV/cache bytes/page | 18.56 MiB | derived from runtime bytes / num_pages |
| BF16 projection weight caches | 1.59 GiB | runtime-reported model_prepare |
| Dense FP8 Marlin projection cache | disabled | runtime-reported model_prepare |
| Allocated before graph capture | 23.24 GiB | runtime-reported |
| Allocated after graph capture | 41.06 GiB | runtime-reported |
| CUDA graph free-memory delta | 18.78 GiB | runtime-reported |
| CUDA graph allocated delta | 17.82 GiB | derived from allocated after-before |
| Loaded model/static residual estimate | 19.33 GiB | capture_memory_allocated_before_graph - reported KV/cache pages - reported BF16 projection weight caches; includes sharded model weights plus base runtime/comm buffers. |
| Context/page capacity | 128 pages x 256 tokens = 32768 tokens | runtime config |

Projection/cache weights:

| cache | enabled | layers | bytes | toggle/backend |
| --- | --- | --- | --- | --- |
| q_wqb_bf16_weight_cache | True | 43 | 0.34 GiB | MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE |
| wo_b_bf16_weight_cache | True | 43 | 0.34 GiB | MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE |
| indexer_wq_b_bf16_weight_cache | True | 21 | 0.33 GiB | MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE |
| wo_a_bf16_bmm_cache | True | 43 | 0.34 GiB | MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE |
| shared_expert_bf16_weight_cache | True | 43 | 0.25 GiB | MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE |
| projection_bf16_weight_cache_total | n/a | n/a | 1.59 GiB |  |
| dense_fp8_marlin_projection_cache | False | 0 | 0.00 GiB | MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION |

Prefix/cache final state by scenario:

| scenario | hit | saved prefill | retained pages | retained memory | available component pages | live full pages | evictions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| historical_4096_128_bs4 | 0.0% | 0 | 64 | 1.15 GiB | 65 | 4 | 0 |
| historical_4096_1024_bs4 | 0.0% | 0 | 124 | 2.23 GiB | 5 | 11 | 1 |
| serving_mixed_112req_wave16 | 0.0% | 0 | 106 | 1.91 GiB | 23 | 23 | 3 |
| prefix_multi_112req_wave16 | 41.4% | 49152 | 87 | 1.56 GiB | 42 | 28 | 4 |

Memory interpretation:
- Allocated memory before graph capture is 23.24 GiB; the explicit KV/cache ledger inside that footprint is 2.32 GiB per rank.
- Prefix/cache component state can retain more than 1 GiB/rank in these workloads; prefix multi ends with meaningful retained state and limited component-page headroom.
- CUDA graph capture consumes about 18.78 GiB of free memory, making cache footprint a real target selector.

## Source Census

### INT8 MoE

- Status: partial reference path; not ready as next integration target
- Evidence: source-derived
- Finding: vLLM has online per-row INT8 MoE loading and a Triton INT8 MoE backend, including W8A16/W8A8 config constructors. The Marlin path is only WNA16-like for this repo and vLLM explicitly asserts that W8A8 INT8 is not supported by Marlin. Mini's current Marlin wrapper requires fp16/bf16 activations.
- Sources:
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/online/int8.py:30`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/fused_moe/oracle/int8.py:32`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/quantization/gptq_marlin.py:535`
  - `python/minisgl/kernel/marlin_wna16.py:250`

### FP8 KV/cache

- Status: strong DSv4-specific reference path
- Evidence: source-derived plus runtime memory pressure
- Finding: vLLM DeepSeek V4 only accepts fp8/fp8_ds_mla KV cache for its sparse FlashMLA path, with paged uint8 cache specs, FP8 quantize/insert, gather/dequant, and an SM80 reference fallback. SGLang has DSv4 MLA FP8 pack/quant/store kernels. This is the most mature low-precision route to study next, but it still needs a parity ledger before implementation.
- Sources:
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:1144`
  - `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py:1189`
  - `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py:7`
  - `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py:160`
  - `/workspace/sglang-main/python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py:1`
  - `/workspace/sglang-main/python/sglang/jit_kernel/triton_store_cache.py:12`

### INT8 communication

- Status: not ready
- Evidence: runtime-proven BF16 traffic plus source-derived lack of CUDA DSv4 path
- Finding: Runtime communication entries are BF16 for hidden all-reduces and FP32 for lm_head all-gather. Mini PyNCCL maps only fp16/bf16/fp32. SGLang exposes quant_all_reduce, but the source marks it as NPU support only and falls back to normal all-reduce on other devices. No DSv4 SM80 CUDA INT8 communication protocol was found.
- Sources:
  - `python/minisgl/kernel/csrc/src/pynccl.cu:50`
  - `/workspace/sglang-main/python/sglang/srt/distributed/parallel_state.py:663`
  - `/workspace/sglang-main/python/sglang/srt/distributed/device_communicators/npu_communicator.py:27`
  - `/workspace/sglang-main/python/sglang/srt/layers/linear.py:1546`

### projection/cache-boundary fusion

- Status: reference path exists, but lower priority for next target
- Evidence: source-derived plus owner timing
- Finding: Mini already has fused q/kv norm+RoPE+BF16 cache store and projection BF16 caches. SGLang/vLLM have DSv4 fused norm/rope/FP8 store and fused pack/store code. The current owner timing does not make this a stronger next move than FP8 KV/cache capacity work.
- Sources:
  - `python/minisgl/kernel/triton/deepseek_v4.py:3557`
  - `python/minisgl/kernel/triton/deepseek_v4.py:3644`
  - `python/minisgl/kernel/triton/deepseek_v4.py:4583`
  - `python/minisgl/kernel/triton/deepseek_v4.py:5197`
  - `/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh:42`
  - `/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:382`

## Recommended Next Target

**Run TARGET 09.3: FP8 KV/cache parity ledger.**

Why:
- Runtime-proven cache/memory pressure exists now: BF16 KV/cache pages, prefix-retained component state, and graph memory delta materially constrain headroom.
- Source-derived FP8 KV/cache references are DSv4-specific and detailed enough to ledger without guessing.
- INT8 MoE has partial source references but needs shape/numerical microbench proof before integration.
- INT8 communication would need a new CUDA quantize/reduce/dequant protocol for these owner boundaries.
- Projection/cache-boundary fusion has references but is not the dominant next decision point from this preflight.

Evidence classification:
- TARGET 09.3: runtime-proven pressure + source-derived implementation references.
- TARGET 09.1: source-derived only; needs microbench-proven feasibility first.
- TARGET 09.25: runtime-proven traffic, but source-derived blocker for CUDA INT8 path.
- TARGET 09.6: source-derived references, weaker runtime priority.

## Stop/Pass Gate

| check | result |
| --- | --- |
| macro_all_pass | True |
| graph_enabled_all_scenarios | True |
| captured_bs_1_2_4_8_16_all_scenarios | True |
| graph_eager_decode_count_zero_all_scenarios | True |
| use_pynccl_true_all_reports | True |
| pynccl_threshold32m_default_log_seen | True |
| owner_timing_enabled_all_single_scenario_runs | True |
| gate | PASS |

Decision: **PASS** for preflight, **do not implement low precision in TARGET 09.0**, and proceed to **TARGET 09.3**.

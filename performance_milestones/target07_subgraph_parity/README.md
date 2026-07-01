# TARGET 07.25: DSV4 sm80 vLLM Subgraph Parity

## Scope

This milestone compares mini-sglang and the old vLLM DeepSeek V4 Flash path for:

- single-node TP8 on 8x A100/sm80
- page/block size 256
- primary workloads: 4096 input / 128 output / batch4 and 4096 input / 1024 output / batch4
- mini default lane: exact bf16-direct activations, no new activation quantization lane

No new MoE V2, attention kernel, or precision lane was implemented here.  The work is evidence collection, subgraph alignment, and target selection.

## Artifacts

| Path | Contents |
| --- | --- |
| `scripts/mini_subgraph_microbench.py` | mini standalone subgraph probes on deterministic synthetic shapes. |
| `scripts/vllm_subgraph_microbench.py` | vLLM kernel probes plus explicit blockers for engine-managed boundaries. |
| `scripts/comm_microbench.py` | TP8 torch/NCCL communication probes. |
| `scripts/summarize_nsys_sqlite.py` | Nsight Systems SQLite summarizer. |
| `summaries/mini_subgraph_microbench.json` | mini quick microbench output. |
| `summaries/vllm_subgraph_microbench.json` | vLLM quick microbench output. |
| `summaries/comm_microbench_torch_nccl.json` | TP8 communication quick microbench output. |
| `summaries/nsys_mini_best_4096x128_rank0_summary.json` | summarized mini best exact graph nsys artifact. |
| `summaries/nsys_vllm_4096x128_bs4_summary.json` | summarized vLLM nsys artifact. |
| `raw/nsys_mini_best_4096x128_rank0.sqlite` | symlink to existing mini nsys raw. |
| `raw/nsys_vllm_4096x128_bs4.sqlite` | symlink to existing vLLM nsys raw. |
| `raw/vllm_4096x128_summary.json` | symlink to vLLM fair 4096/128 summary. |
| `raw/vllm_4096x1024_summary.json` | symlink to vLLM fair 4096/1024 summary. |

## Frozen Baselines

| Framework | Variant / path | 4096/128/bs4 output tok/s | 4096/1024/bs4 output tok/s | Notes |
| --- | --- | ---: | ---: | --- |
| mini | `v1_moe_graph_hc_rmsnorm_fwqakvcache_qkvrope_sample_wqb_wob_idxwqb_gatecache_idxstorecache` | 13.2160 | 25.3076 | Current TARGET 07.2 best exact graph path; captured decode batch sizes `[4,2,1]`; greedy sampler replay captured; no eager decode fallback. |
| vLLM | fair offline comparison in `performance_milestones/vllm/` | 80.9050 | 201.8738 | Existing fair results from `/tmp/dsv4_target07_nsys_vllm_4096x128_bs4_warmup1` and `/tmp/dsv4_target07_vllm_4096x1024_bs4_warmup1`. |

Remaining macro ratio is about 6.12x on 4096/128 and 7.98x on 4096/1024, vLLM over mini.

### Precision Policy Difference

This comparison is not precision-neutral:

- mini target lane remains exact bf16-direct for activations and bf16 KV cache, with checkpoint quantized weights handled without introducing activation quantization as the default.
- vLLM DeepSeek V4 sm80 path uses `deepseek_v4_fp8` policy: FP8 non-MoE linear handling, FP8/UE8M0 `fp8_ds_mla` KV/cache/indexer layout, and MXFP4 MoE experts.
- Therefore, vLLM MoE/KV/cache wins are partly `precision gap`, not just better scheduling or kernels.  They should not be silently ported into mini's exact default.

## Profile Anchors

Mini best exact nsys, rank0, 4096/128 artifact: `summaries/nsys_mini_best_4096x128_rank0_summary.json`.

Top kernel time in that artifact:

| Kernel / category | Count | Time |
| --- | ---: | ---: |
| `_grouped_fp4_w13_kernel` | 11,137 | 28.490s |
| `_grouped_fp4_linear_kernel` | 11,137 | 18.760s |
| sparse attention with compressed source | 10,619 | 8.000s |
| PyTorch copy elementwise | 373,219 | 4.069s |
| `_quantized_linear_fp8_kernel` | 27,499 | 2.375s |
| `_indexer_bf16_logits_kernel` | 5,376 | 2.011s |
| NCCL bf16 all-reduce | 11,396 | 0.610s |
| NCCL f32 all-reduce | 11,137 | 0.597s |

The same target's macro communication counters from TARGET 07.2 for 4096/1024 are:

| Label | Count | Bytes |
| --- | ---: | ---: |
| `dsv4.embedding_all_reduce` | 8 | 1,073,741,824 |
| `dsv4.lm_head_all_gather` | 8 | 16,547,840 |
| `dsv4.row_parallel_projection_all_reduce` | 344 | 46,170,898,432 |
| `dsv4.v1_moe_reduce_once_all_reduce` | 344 | 92,341,796,864 |
| total | 704 | 139,602,984,960 |

The vLLM 4096/128 nsys SQLite is useful for macro NVTX and runtime evidence, but the kernel table is not a reliable subgraph attribution source: it reports only one CUDA stream and 0.982s summed kernel time for a 6.321s repeat window, likely due to CUDA graph / multiprocess collection limitations.  This is recorded as a profile blocker; code-path evidence and paired probes are used instead for stream topology and subgraph decisions.

## Microbench Summary

Quick probes used `warmup=2`, CUDA events, and a CUDA sync after every measured iteration.  Tokens=4 is decode-like.  Tokens=4096 is prefill/chunk-like.  Full JSON contains samples and environment.

| Subgraph probe | mini mean ms, T=4 | vLLM mean ms, T=4 | mini mean ms, T=4096 | vLLM mean ms, T=4096 | Boundary note |
| --- | ---: | ---: | ---: | ---: | --- |
| attention front projection | 0.249 | 0.180 | 0.771 | 0.510 | vLLM probe excludes engine-managed fused qnorm/RoPE/KV quant insert; mini includes bf16 cache store. |
| sparse attention core | 0.588 | 0.131 | 30.178 | 4.947 | vLLM starts after packed-cache gather/dequant; mini reads bf16 caches directly. |
| indexer | 0.330 | 0.086 | 44.504 | 0.407 | mini is bf16 logits + topk; vLLM is Q RoPE + quant/weight fold only, not full `SparseAttnIndexer`. |
| routed MoE experts | 2.270 | blocked | 98.229 | blocked | vLLM exact boundary requires engine `FusedMoE`, transformed MXFP4 weights, static forward context, router, and shared-expert state. |
| shared experts | 0.200 | blocked | 0.226 | blocked | vLLM exact boundary is scheduled by `SharedExperts` on an aux stream and depends on quantized MLP modules. |
| HC/final RMSNorm | 0.338 | not paired | 1.953 | not paired | vLLM uses compiled HC helpers; not a dominant remaining profile item. |
| communication, all-reduce `[4,4096]` bf16 | 0.074 | not paired | n/a | n/a | TP8 torch/NCCL reference, rank-local 32 KiB. |
| communication, all-reduce `[4096,4096]` bf16 | n/a | n/a | 0.457 | not paired | TP8 torch/NCCL reference, rank-local 32 MiB. |
| lm-head all-gather shard | 0.213 | not paired | n/a | n/a | TP8 torch/NCCL reference, rank-local 129,280 bytes. |

The sparse/indexer ratios above are directional, not exact same-boundary ratios.  They are still useful because they identify layout and fusion boundaries that mini does not currently match.

## Subgraph Parity Map

| Required subgraph | mini entrypoint and ops | vLLM entrypoint and ops | Decode/prefill shape | Precision lane | Comm / graph / streams | Latency evidence | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scheduler / graph surface / metadata staging / sampling | `python/minisgl/engine/graph.py::GraphRunner`, `python/minisgl/attention/deepseek_v4.py::prepare_metadata`, `stage_capture_metadata_for_graph`; graph captures `[1,2,4]`, copies replay buffers, optional greedy sample graph. | `vllm/v1/worker/gpu_model_runner.py`, `CUDAGraphDispatcher`, `parallel_state.graph_capture`, pinned async sampled-token copy, static metadata tensors. | 4096 prefill eager, decode T=4 repeated 127/1023. | mini exact bf16-direct; vLLM uses fp8 policy downstream. | mini mostly one graph/current stream; vLLM graph capture stream plus graph-aware custom all-reduce and pinned CPU event for sampling. | mini graph replay count 254/2046; vLLM macro 6.321s for 4096/128. No exact paired microbench. | adapt graph-aware metadata/sampling ideas after MoE/attention; do not treat as top standalone kernel. |
| attention front projection and cache insert | `DSV4Attention.forward`: fused `wq_a+wkv` optional cache, `rms_norm_fallback`, `wq_b`, `q_kv_norm_rope_cache_fallback`, bf16 SWA cache store, `wo_a_grouped_projection_fallback`, `wo_b`. | `DeepseekV4MultiHeadLatentAttentionWrapper.forward`: `MergedColumnParallelLinear` for `wq_a+wkv`, `fused_q_kv_rmsnorm`, `wq_b`, `_fused_qnorm_rope_kv_insert` -> `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`; O path inverse RoPE + `wo_a` BMM + `wo_b`. | T=4 decode; T=4096 projection/chunk-like. Local heads=8, head_dim=512, q_lora=1024, kv=512. | mini bf16 q/KV and bf16 cache; vLLM fp8_ds_mla KV cache after quant insert. | no collectives inside front; row-parallel `wo_b` later all-reduce in mini. mini captured in decode graph; vLLM custom op in graph path. vLLM may overlap KV insert/compressor with indexer. | mini 0.249/0.771 ms; vLLM front-only 0.180/0.510 ms. | adapt fused norm/projection boundaries; reject fp8 KV cache for exact default; consider separate attention/cache target after MoE. |
| sparse attention and indexer | mini backend builds SWA/C4/C128 metadata, `indexer_select_bf16_fallback`, `topk_transform_512_full_fallback`, `dsv4_sparse_attention_two_source_bf16`. | `DeepseekV4Indexer.forward`, `fused_indexer_q_rope_quant`, `SparseAttnIndexer`, `compute_global_topk_indices_and_lens`, `gather_dequant_two_scopes_with_mask`, `_dsv4_sm80_sparse_attn_decode_triton`; sm80 prefill reference path exists but is OOM-prone. | C4 topk 512, SWA 128, local heads=8, index heads=64, index dim=128. | mini bf16 indexer/query/cache; vLLM quantized indexer/cache path. | no TP collectives inside. mini metadata mostly one stream; vLLM attention aux stream overlaps indexer with KV insert/compressor. | mini sparse 0.588/30.178 ms; vLLM decode core 0.131/4.947 ms. mini indexer 0.330/44.504 ms; vLLM Q quant 0.086/0.407 ms but boundary mismatch. | adapt split-K decode core/layout/indexer staging; reject sm80 sparse prefill reference as default; defer full port until exact cache policy is chosen. |
| MoE route and routed experts | `DSV4MoEGate.forward`, `moe_gate_fallback`, `DSV4FusedRoutedExperts.forward`, `moe_route_dispatch_bf16_grouped`, `_grouped_fp4_w13_kernel`, `_grouped_fp4_linear_kernel`, `_moe_route_sum_kernel`; late `dsv4.v1_moe_reduce_once_all_reduce`. | `DeepseekV4MoE`, `FusedMoE`, `Mxfp4MoEMethod`, `FusedMoEModularKernel`, router select, dispatch/combine, optional final TP all-reduce. `DeepseekV4MegaMoEExperts` is disabled on sm80. | T=4 decode; T=4096 prefill-like. Hidden=4096, topk=6, experts=256, local intermediate=256. | mini bf16 activations + packed fp4 checkpoint weights; vLLM MXFP4 optimized MoE path and fp32 router logits. | mini reduce-once after routed+shared local sum; vLLM also supports combined output reduce through FusedMoE runner/custom all-reduce. Graph placement differs. | mini routed 2.270/98.229 ms; macro top kernels W13+W2 = 47.250s of 76.448s rank0 kernel time. vLLM standalone exact boundary blocked. | adapt into TARGET 07.3 MoE exact V2. Reject MegaMoE for sm80; defer vLLM MXFP4 precision semantics for default. |
| shared experts | `DSV4SharedExperts.forward`: local gate/up, `silu_and_mul_clamp_fallback`, down, local merge with routed before reduce-once. | `DeepseekV2MLP` as shared experts inside `FusedMoE`; `SharedExperts` wrapper may run on aux stream and overlap with router/routed work. | T=4/T=4096, local gate/up `[512,4096]`, down `[4096,256]` in mini probe. | mini bf16; vLLM follows quant config / linear method. | mini serial with routed path before single reduce; vLLM `wait_stream` then aux stream execution, current waits when output is needed. | mini local 0.200/0.226 ms. vLLM exact boundary blocked by engine-managed MLP/runner state. | adapt shared-expert overlap and combine boundary as part of MoE V2, not as an isolated target. |
| HC/RMSNorm/final layers | `DeepseekV4DecoderLayer.forward`, `hc_pre_fallback`, `hc_post_fallback`, `hc_head_fallback`, `rms_norm_fallback`, final lm_head all-gather. | vLLM model uses compiled HC helpers, RMSNorm modules, logits/sampling runner path. | HC mult=4, hidden=4096; final logits shard gathered across TP. | mini bf16/fp32 mix; vLLM compiled helpers under fp8 policy. | lm_head all-gather in mini label `dsv4.lm_head_all_gather`; vLLM all-gather observed in NVTX but profile not enough for per-call breakdown. | mini HC/final probe 0.338/1.953 ms; macro HC kernels below MoE/sparse. | defer; keep existing mini HC/RMSNorm helpers. |
| communication | `DistributedCommunicator` labels embedding, row-parallel projection, reduce-once MoE, lm_head all-gather; optional PyNCCL not promoted in 07.2. | `parallel_state.py` custom all-reduce, `custom_all_reduce.py` graph buffer registration, NCCL fallback/all-gather paths. | decode hidden `[4,4096]`, prefill hidden `[4096,4096]`, logits shard `[4,16160]` for TP8. | bf16/f32 collectives depending tensor. | mini best 4096/1024: 704 collectives, 139.6 GB label bytes. vLLM graph-aware custom all-reduce and CUDA graph capture support. | torch/NCCL TP8: decode bf16 AR 0.074 ms, decode f32 AR 0.068 ms, prefill bf16 AR 0.457 ms, prefill f32 AR 0.716 ms, logits all-gather 0.213 ms. | adapt graph-aware custom all-reduce/reduce boundary after MoE/attention; not the top measured kernel-time item in current mini best. |

## vLLM Multi-Stream Topology

vLLM does use multiple CUDA streams in the DeepSeek V4 sm80 path, but the available nsys SQLite did not expose enough per-stream kernel attribution to quantify the exact wall-time saving.

Confirmed code topology:

| Stream / mechanism | Code | Dependency pattern | Overlapped work |
| --- | --- | --- | --- |
| attention aux stream | `DeepseekV4Model.__init__` creates `AuxStreamType.Attention`; `deepseek_v4_attention.py::attention_impl`; `utils/multi_stream_utils.py::maybe_execute_in_parallel` | current records `event0`; aux waits; aux records `event1`; current waits `event1` | C4 layer: indexer on current stream while KV insert + compressor run on aux. Compressor-only layer: compressor on current while KV insert runs on aux. SWA-only has no overlap. |
| MoE shared-expert stream | `fused_moe/runner/shared_experts.py`, `moe_runner.py` | shared input `record_stream`; aux stream waits current before gate/router; current waits aux when shared output is consumed | shared experts can overlap with gate/router/routed expert work when token threshold and env allow it. |
| graph capture stream | `parallel_state.py::graph_capture`, `gpu_model_runner.py::capture_model` | separate capture stream; custom all-reduce registers graph buffers/pointers | graph replay and custom all-reduce placement. |
| sampling / CPU copy event | `gpu_model_runner.py::_to_list` | pinned async copy plus CUDA event instead of global `.tolist()` sync | sampled token transfer/scheduler surface. |

Generic two-GEMM overlap probe in `vllm_subgraph_microbench.json`:

- T=4: serial 0.074 ms, aux-stream 0.105 ms, speedup 0.71x.  Stream overhead dominates tiny work.
- T=4096: serial 1.272 ms, aux-stream 1.268 ms, speedup 1.00x.  Two large GEMMs saturate the GPU, so overlap does not help.

This probe is not the real vLLM attention/MoE overlap.  It prevents a wrong conclusion: some of vLLM's macro advantage can be scheduling/overlap and graph placement, but current artifacts cannot assign that benefit to one kernel.  Treat overlap as a scheduling/graph gap and re-measure with vLLM NVTX ranges around indexer/KV insert/shared experts before changing mini.

## Gap Attribution

### Structural Gap

vLLM's DeepSeek V4 path has different boundaries:

- engine-managed `FusedMoE` with route/select/dispatch/finalize state and shared experts integrated into the runner
- packed `fp8_ds_mla` KV/cache layout with fused insert and gather/dequant kernels
- global topk buffer and sparse metadata helpers integrated into the attention backend
- graph-aware custom all-reduce buffer registration

Mini has narrowed some projection and graph boundaries in TARGET 07.2, but still does MoE, sparse indexer, and cache/layout work in less consolidated shapes.

### Microkernel Gap

The clearest measured microkernel-like gap is sparse attention/indexer:

- mini sparse two-source bf16 kernel: 30.178 ms at T=4096
- vLLM sm80 split-K sparse decode core: 4.947 ms at T=4096, after gather/dequant

This is a real lead for vLLM's decode attention core, but the boundaries are not identical.  The indexer comparison is even more mismatched: mini measured full bf16 logits + topk, vLLM measured Q RoPE + quant/weight fold only.

Attention front projection is smaller:

- mini 0.771 ms at T=4096
- vLLM 0.510 ms at T=4096, front-only

That is worth adapting, but it cannot explain an 8x macro gap.

### Scheduling / Graph Gap

vLLM has attention aux-stream overlap, MoE shared-expert aux-stream overlap, graph capture stream handling, graph-aware all-reduce, and pinned sampling copies.  Mini's current decode graph path is mostly single-stream replay plus buffer copies.  The available vLLM nsys raw does not quantify wall-time overlap, so this remains an evidence-backed but not fully measured gap.

### Communication Gap

Mini still performs many collectives in the best exact macro path: 704 labeled collectives and 139.6 GB label bytes for 4096/1024.  However, rank0 best 4096/128 kernel-time attribution shows NCCL kernels around 1.2s out of 76.4s summed kernel time, below MoE and sparse/indexer.  Communication should be revisited after MoE/attention boundaries reduce compute time, because its relative share will rise.

### Precision Gap

vLLM's macro number benefits from a precision lane mini is not currently choosing as default: FP8 non-MoE linears, FP8/UE8M0 KV/cache/indexer, and MXFP4 MoE.  This may be necessary for absolute vLLM parity, but it should remain a separate precision-lane target, not be mixed into exact bf16-direct work.

## Bottleneck Ranking for 4096/1024/Batch4

1. **MoE routed experts and MoE execution boundary.**  Mini profile is dominated by grouped FP4 W13/W2 kernels, and the standalone routed probe is the heaviest prefill-like probe.  vLLM's comparable boundary is a more integrated FusedMoE/MXFP4 runner and cannot be safely benchmarked standalone without engine context.  This is the next implementation target.
2. **Sparse attention/indexer/cache layout.**  mini's bf16 sparse attention and full bf16 indexer/topk are heavy, while vLLM has a faster split-K decode core plus packed-cache gather/dequant and quantized indexer path.  Needs a separate attention/cache target after MoE V2 or in parallel only if MoE stalls.
3. **Scheduling/graph/stream overlap.**  vLLM overlaps attention sub-work and shared experts; mini does not.  Current profile cannot assign exact benefit, but the topology is real.
4. **Communication/reduce boundary.**  Important structurally, especially after compute shrinks, but not currently the top measured kernel-time item.
5. **Precision lane.**  Large potential source of vLLM advantage, but out of default exact scope.  Defer until exact MoE/attention baselines are stronger.
6. **HC/RMSNorm/final/sampling.**  Current mini helpers and sampler graph are not the dominant remaining gap.

## Port / Adapt / Reject / Defer

| vLLM implementation | Decision | Reason |
| --- | --- | --- |
| FusedMoE runner structure, route metadata, workspace/finalize, single reduce boundary | adapt | Most aligned with observed mini bottleneck; must preserve mini exact lane and local abstractions. |
| vLLM MXFP4 MoE precision semantics | defer | Part of vLLM speed, but precision-lane difference from mini exact default. |
| DeepseekV4MegaMoEExperts | reject for sm80 | vLLM disables it on sm80; not relevant to A100 target. |
| attention aux-stream `maybe_execute_in_parallel` topology | adapt after measurement | Real scheduling gap; add mini only around proven independent branches. |
| fused qnorm/RoPE/KV insert and packed cache layout | adapt/defer | Good boundary, but current vLLM path stores fp8_ds_mla KV cache; exact bf16 default needs a different implementation. |
| vLLM sm80 sparse prefill reference path | reject as default | Known OOM-prone large materialization path. |
| custom all-reduce graph buffer registration | adapt later | Useful once compute bottlenecks shrink and communication dominates more. |
| pinned async sampled-token copy | adapt opportunistically | Low-risk graph/scheduler cleanup, but not a top target. |

## Next Target

Choose **TARGET 07.3 MoE exact V2**.

The best-supported next step is to adapt vLLM's MoE execution shape into mini's exact lane: route metadata/workspace, routed expert scheduling, shared expert overlap/combine, and final reduce boundary.  Do not make vLLM a runtime dependency and do not promote MXFP4/FP8 activation/cache precision as part of 07.3.  If MoE V2 does not explain enough of the gap, the next target should be a dedicated attention/cache-insert/indexer target using this milestone's sparse-attention evidence.

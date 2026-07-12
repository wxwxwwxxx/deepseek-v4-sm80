# TARGET 12: DSV4 SM80 Decode Replay Metadata And Latency Hiding

## Status

Active fallback/native-backend census follow-up after TARGET 12.56.

TARGET 11 MTP is paused and archived.  The release branch should continue from
the non-MTP DSV4 sm80/A100 release bundle candidate:

```text
dsv4_sm80_release_default
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc
MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT=1
MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
PyNCCL threshold32m default for A100/sm80 DSV4
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

The original TARGET 12 trigger was:

```text
performance_milestones/misc_post_mtp_cleanup_replay_attribution/README.md
```

That report shows the remaining regression is not MTP leakage, communication
count/bytes, graph replay coverage, or wrapper count.  The gap is concentrated
around decode CUDA graph replay setup and captured forward time, especially
metadata preparation/staging before `g.replay()`.

TARGET 12.50 resolved the Marlin WNA16 release-default gap and promoted the
Tier A bundle. TARGET 12.51 fixed the SWA independent in-graph metadata blocker
and showed clean oracle/text/graph/macro gates. TARGET 12.52 folded the SWA
independent/direct metadata path into the true no-env release default. TARGET
12.53 removed the HC prenorm prefill temporary and promoted the new HC path.
TARGET 12.54 showed that 32768-token prefill now passes, but 65536-token
single-request prefill runs out of activation/workspace headroom. TARGET 12.55
proved that a small KV reserve is not enough: even `memory_ratio=0.85` still
OOMed, with the owner moving to the full-prefill Marlin WNA16 MoE
`route_out` workspace. TARGET 12.56 hardened the existing `ChunkedReq` /
`max_extend_tokens` path, fixed SWA/component lifecycle issues, and selected
`8192` as the conservative DSV4 A100/sm80 release prefill chunk token budget.
TARGET 12.57 adapted vLLM-style bounded indexer execution and removed the
unbounded full-logits/remap allocations. TARGET 12.58 promoted that path,
passed 512k, and isolated the 1M blocker to eager-prefill C128 metadata after
729088 committed tokens. TARGET 12.59 proved a one-surface C128 contract, and
TARGET 12.595 integrated it and completed all 128 prefill chunks plus decode
graph replay. TARGET 12.597 aligned benchmark and serving max-sequence
semantics and passed the legal 1M total-sequence gate. TARGET 12.60 measured the
practical decode graph envelope, exposed missing pre-KV graph-memory accounting,
and found padded/live-row token drift. TARGET 12.602 localized a blocking
dummy-route dependency to layer-0 MoE route planning/grouped execution. TARGET
12.6025 aligned the live-route contract with SGLang. TARGET 12.603 installed a
safe conservative pre-KV graph reserve for max16/64/128. TARGET 12.604 now
unifies bucket resolution so planner and GraphRunner cannot observe different
lists. TARGET 12.605 now measures a recipe frontier across balanced req256,
high-concurrency req512, and long-context low-request-capacity configurations;
TARGET 12.606 promotes the selected recipes. `M=1024/2048` remain isolated
smoke points, and simultaneous 1M context plus high concurrency is not required.

## Goal

Reduce or explain the per-decode-step overhead in:

```text
GraphRunner._replay_to_buffer
  -> graph input staging
  -> attn_backend.prepare_for_replay
  -> DSV4 metadata copy / direct metadata generation / component table staging
  -> g.replay()
```

Use SGLang and vLLM as design references first.  Do not invent a local runtime
mechanism until the source comparison shows mini genuinely needs one.

TARGET 12 should answer:

```text
Which part of mini's decode replay path remains slower than the TARGET 10.27
baseline?
Does SGLang/vLLM avoid that overhead by in-graph metadata prep, stable graph
buffers, grouped copies, preallocated metadata state, stream overlap, or a
different graph boundary?
Which mechanism is safe and worthwhile to adapt into mini?
```

## Current Evidence

From the post-MTP attribution target:

```text
TARGET10.27 historical_4096_128_bs4:
  output tok/s              53.3065
  decode tok/s              190.4055
  decode_forward_enqueue_s  0.1510
  decode_forward_s          2.6680
  decode_prepare_s          0.2867

post-cleanup baseline:
  output tok/s              50.6196
  decode tok/s              163.9047
  decode_forward_enqueue_s  0.3904
  decode_forward_s          3.0994
  decode_prepare_s          0.2983

owner host after local dtype/fused-copy fix:
  decode_forward_enqueue_s  0.4537
  decode_forward_s          3.1706
  decode_prepare_s          0.3932
```

Important owner/counter facts:

- CUDA graph replay/eager decode stayed `127 / 0`.
- Communication owner count/bytes matched TARGET 10.27.
- Wrapper counters matched TARGET 10.27.
- `dsv4.graph_replay.prepare_for_replay` was about `366.8 ms / 127` decode
  steps on rank0, or about `2.89 ms/step` under host timing.
- `dsv4.prepare.decode.attention_metadata` was about `342.0 ms / 127`, or
  about `2.69 ms/step`.
- Heaviest decode metadata subowners included:

```text
dsv4.metadata.decode.c128_indices_source             87.1 ms / 127
dsv4.metadata.decode.component_write_locs_source     66.6 ms / 127
dsv4.metadata.decode.object_assembly                 53.7 ms / 127
dsv4.metadata.decode.component_page_tables_source    26.0 ms / 127
dsv4.metadata.decode.swa_indices_source              23.7 ms / 127
dsv4.metadata.decode.page_table_source               17.7 ms / 127
dsv4.metadata.decode.scalar_source                   14.6 ms / 127
```

The byte ledger is small:

```text
rank0 decode replay_metadata_copy bytes:  ~0.82 MB total
rank0 decode direct_graph_metadata bytes: ~3.12 MB total
rank0 graph input staging bytes:          ~6 KB total
```

Interpretation: this does not look like a large HBM bandwidth copy bottleneck.
It looks like fixed per-step overhead from small kernels, small D2D copies,
Python/object assembly, launch overhead, and graph replay dependency
boundaries.

## References To Read First

Mini:

```text
python/minisgl/engine/graph.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/utils/dsv4_owner_timing.py
performance_milestones/misc_post_mtp_cleanup_replay_attribution/README.md
prompts/TARGET_misc_dsv4_sm80_post_mtp_cleanup_replay_attribution.md
```

SGLang:

```text
/workspace/sglang-main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner_utils/buffers.py
/workspace/sglang-main/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata_kernel.py
/workspace/sglang-main/python/sglang/srt/state_capturer/base.py
/workspace/sglang-main/python/sglang/srt/batch_overlap/two_batch_overlap.py
```

vLLM:

```text
/workspace/vllm-dsv4-docker/vllm/compilation/cuda_graph.py
/workspace/vllm-dsv4-docker/vllm/distributed/parallel_state.py
/workspace/vllm-dsv4-docker/vllm/v1/worker/gpu_model_runner.py
/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py
/workspace/vllm-dsv4-docker/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py
```

Useful reference observations already made:

- SGLang `DecodeInputBuffers.populate_from_forward_batch()` groups GPU copies
  through `torch._foreach_copy_` by dtype pair.
- SGLang `DeepseekV4AttnBackend` supports raw decode metadata when
  `SGLANG_PREP_IN_CUDA_GRAPH=1`; raw req/seq/out-loc state is upgraded to full
  DSV4 metadata inside the captured graph by `init_forward_metadata_in_graph`.
- SGLang separates fields that must keep graph-captured tensor addresses from
  fields that can be reference-assigned via
  `refresh_for_breakable_cuda_graph_replay_`.
- SGLang has a Triton metadata kernel for DSV4 compressed attention metadata:
  `dsv4/metadata_kernel.py::init_compression_metadata`.
- vLLM `CUDAGraphWrapper` deliberately stays orthogonal to persistent input
  buffers: it assumes stable input addresses are handled outside the wrapper.
- vLLM captures CUDA graphs on a separate graph-capture stream, but warns that
  future multi-stream designs may affect shared graph pool safety.
- vLLM/SGLang both have async copy/overlap mechanisms for output/result
  movement, but these are not direct proof that mini's decode metadata should
  be moved to another stream.

## Priority Reassessment

Do not make multi-stream metadata overlap the first implementation target.

Reasoning:

- Mini's measured metadata bytes are small, so moving copies to a second stream
  is unlikely to fix a bandwidth bottleneck.
- Most metadata is a direct input to the same-step graph replay, so a secondary
  stream would still need an event wait before `g.replay()`.
- Stream/event management adds correctness risk around CUDA graph pools,
  captured-address invariants, and hidden synchronization.
- SGLang's stronger precedent is not "put everything on aux streams"; it is
  "use stable graph buffers, grouped copies, and move suitable metadata prep
  into the graph or into persistent bucket metadata."

Prioritize in this order:

1. **Source parity and exact owner attribution**: prove where mini differs from
   SGLang/vLLM in graph input buffers, metadata lifetime, in-graph prep, and
   stable-address contracts.
2. **Low-risk deforestation**: reduce Python/object assembly, repeated env
   checks, per-field copy calls, and disabled-feature branches in the replay
   hot path.
3. **SGLang-style in-graph metadata prep PoC**: raw decode metadata outside the
   graph, full DSV4 metadata materialized inside the graph, with stable capture
   buffers.
4. **Direct/fused final graph buffer writers**: generate C4/C128/SWA/component
   graph metadata directly into captured buffers when in-graph prep is not a
   good fit.
5. **Do not pursue multi-stream in this phase**: after TARGET 12.1 source
   parity, the current route is stable buffers, safe owner attribution,
   deforestation, SGLang-style in-graph metadata prep, and direct/fused graph
   metadata writers. Multi-stream remains only a historical fallback if a future
   report proves an independent owner that cannot be removed or captured.

## Split Plan

Run these as focused subtargets.  They may be created as separate prompt files
when executed, but this root TARGET 12 is the controlling roadmap.

| Stage | Status | Purpose |
| --- | --- | --- |
| TARGET 12.1 Source Parity And Contract | completed | Compared mini, SGLang, and vLLM at the graph-replay metadata boundary: static buffers, copied fields, in-graph prep, fixed-address fields, reference-assigned fields, and stream usage. Report: `performance_milestones/target12_source_parity_and_contract/README.md`. |
| TARGET 12.2 Safe Timing And Kernel Census | completed | Split `prepare_for_replay` safely without CUDA-event timing; identified a mandatory small-kernel/copy metadata cluster. Report: `performance_milestones/target12_safe_timing_kernel_census/README.md`. |
| TARGET 12.3 Low-Risk Replay Deforestation | inline/deferred | Can harvest small cleanups during 12.4, but TARGET 12.2 showed the dominant owner is not mostly Python/env/debug guard logic. |
| TARGET 12.4 SGLang-Style In-Graph Metadata Prep PoC | completed | Implemented opt-in in-graph decode metadata prep, passed unit/wrapper/text/oracle gates, removed the main `prepare_for_replay` clamp/copy owner in the short probe, and kept current replay metadata as default fallback/oracle. Report: `performance_milestones/target12_sglang_in_graph_metadata_prep/README.md`. |
| TARGET 12.45 In-Graph Metadata Promotion Soak | completed with blocker | Performance was repeat-stable and positive across short, long-decode, serving, and prefix scenarios, but default promotion is blocked by a long-context oracle mismatch in `c4_sparse_raw_indices`. Report: `performance_milestones/target12_ingraph_metadata_promotion_soak/README.md`. |
| TARGET 12.46 C4 Sparse Oracle Contract | completed | Fixed the oracle boundary around indexer-mutated C4 sparse fields; text oracle and long-context short oracle now pass. Report: `performance_milestones/target12_c4_sparse_oracle_contract/README.md`. |
| TARGET 12.47 Final Promotion Rerun | completed | Reran the in-graph metadata promotion subset after the oracle fix with one fresh process per variant. It passed correctness and showed repeat-stable macro wins, so the path is ready for release-default cleanup. Report: `performance_milestones/target12_ingraph_metadata_final_promotion_rerun/README.md`. |
| TARGET 12.48 Release Defaults Promotion Cleanup | completed | Folded the 12.47 recipe into the DSV4 A100/sm80 release defaults: Route-B C4 direct graph metadata, MoE BF16 reduce, in-graph replay metadata prep, page-size 256, radix prefix/component ownership, PyNCCL threshold32m, and CUDA graph buckets `[1,2,4,8,16]`. Keep fallback/oracle paths explicit via `MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS=1` or dedicated benchmark variants. |
| TARGET 12.49 Release Long-Context And Large-Batch Soak | completed with HC prefill blocker | True no-env release default was healthy for text smoke, 8192-token long context, default-bucket large-batch decode through batch 128, and explicit graph buckets through max 128. The main exposed blocker is a 2 GiB HC prenorm temporary in `hc_pre_fallback` at 32768 prefill tokens. Report: `performance_milestones/target12_release_long_context_large_batch_soak/README.md`. |
| TARGET 12.50 Release Bundle Opt-In Promotion Gate | completed with SWA blocker | Promoted the Tier A release bundle: Marlin WNA16 prebuild/release/capacity credit, component-slot clear on page allocation, Route-B metadata, in-graph metadata prep for the non-SWA path, BF16 MoE reduce, and PyNCCL threshold32m. SWA independent remained opt-in because `prep_metadata_in_graph` fail-opened with `swa_independent_lifecycle_not_supported` and macro throughput regressed by about 12-18% despite large capacity wins. Report: `performance_milestones/target12_release_bundle_optin_promotion_gate/README.md`. |
| TARGET 12.51 SWA Independent In-Graph Metadata Promotion | completed | Removed the SWA-independent `prep_metadata_in_graph` compatibility blocker by extending the in-graph metadata prep kernel/API to consume SWA full-to-SWA page mapping. Oracle, text smoke, graph replay, and four-scenario macro gates passed; SWA independent is eligible for default promotion. Report: `performance_milestones/target12_swa_independent_ingraph_metadata_promotion/README.md`. |
| TARGET 12.52 SWA Independent Release Default Cleanup | completed | Promoted the TARGET 12.51 SWA independent/direct metadata path into the true no-env `dsv4_sm80_release_default` bundle. Text smoke, unit tests, four-scenario macro, graph replay, and capacity gates passed. Report: `performance_milestones/target12_swa_independent_release_default_cleanup/README.md`. |
| TARGET 12.53 HC Prenorm Temporary Elimination And Promotion Gate | completed | Promoted `MINISGL_DSV4_SM80_HC_GRAPH_CLEANUP=1` plus `MINISGL_DSV4_SM80_LINEAR_BF16_FP32=1` into the release default, removed the 2 GiB HC prenorm temporary exposed by TARGET 12.49, fixed the `32768/16/1` and `128/64/256` OOM shapes, and improved the four historical macro scenarios by about 5-9%. Report: `performance_milestones/target12_hc_prenorm_temp_elimination/README.md`. |
| TARGET 12.54 Post-HC Release Envelope Rerun | completed with 64k memory-headroom blocker | True no-env default injected the HC release pair, text sanity passed, and the previous `32768/16/1` OOM was fixed. The run stopped at `65536/8/1`: graph capture and KV planning succeeded, but first prefill OOMed on a 128 MiB `wo_a` BF16 BMM allocation with only about 45 MiB free. Report: `performance_milestones/target12_post_hc_release_envelope_rerun/README.md`. |
| TARGET 12.55 Graph And Activation Memory Accounting | completed with chunked-prefill decision | Memory-ratio sweep showed the 64k blocker is not solved by a 1-3 GiB KV reserve. Reducing `memory_ratio` to `0.85` freed about `3.92 GiB` / `134k` tokens but still failed; the owner moved from `wo_a` to gate and then Marlin WNA16 `route_out`. Decision: `CHUNKED_PREFILL_REQUIRED`. Report: `performance_milestones/target12_graph_activation_memory_accounting/README.md`. |
| TARGET 12.56 Chunked Prefill Long-Context Path | completed | Hardened the existing `ChunkedReq` / `max_extend_tokens` path, fixed DSV4 SWA/component capacity and chunk lifecycle issues, and selected `8192` as the conservative release prefill chunk budget after 65k/131k/262k ladder evidence. Report: `performance_milestones/target12_chunked_prefill_long_context/README.md`. |
| TARGET 12.57 Release Fallback Census And Native Backend Gate | completed | Adapted vLLM-style bounded query-row execution for the existing native FP8 paged indexer, added fused Triton Route-B component/full remap, removed the reproduced 2.25 GiB full-logits failures, and kept release `max_extend_tokens=8192`. The next manual 32768-token-chunk owner is a 1.50 GiB Marlin routed output, but it is not yet a release-default blocker. Report: `performance_milestones/target12_release_fallback_census_native_backend_gate/README.md`. |
| TARGET 12.58 Post-Indexer Long-Context Release Envelope | completed with C128 metadata blocker | Promoted TARGET 12.57, passed the true-default 512k smoke, and reached 729088 committed tokens in the 1M probe before a 360 MiB int64 C128 component-mapping allocation failed. The bounded indexer is memory-safe but accounts for about 48% of 512k TTFT. Report: `performance_milestones/target12_post_indexer_long_context_envelope/README.md`. |
| TARGET 12.59 C128 Prefill Metadata Contract And Native Micro | completed | Proved that release eager prefill consumes only final C128 component page indices plus lengths, implemented an exact one-launch Triton helper, and measured zero temporary bytes beyond its final int32 output. Report: `performance_milestones/target12_c128_prefill_metadata_contract_native_micro/README.md`. |
| TARGET 12.595 C128 One-Surface 1M Promotion | completed with benchmark-contract follow-up | Integrated the one-surface helper, removed eager raw/full and int64 matrices, completed 128 chunks plus seven decode graph replays, and retained about 1.13 GiB physical free. The benchmark implicitly raised max sequence to prompt+decode, so serving-default max-sequence parity remains to be closed. Report: `performance_milestones/target12_c128_one_surface_1m_promotion/README.md`. |
| TARGET 12.597 Release Max-Sequence And Benchmark Parity | completed | Separated model-default, explicit-override, and scenario-sized max-sequence modes; passed scheduler/RoPE bounds and the legal `1048568+8=1048576` total-sequence gate; exposed `max_running_req` as the remaining graph-baseline ambiguity. Report: `performance_milestones/target12_release_max_seq_benchmark_parity/README.md`. |
| TARGET 12.60 CUDA Graph Bucket Policy Preflight | completed with two follow-ups | Established the true serving baseline, measured cumulative graph memory through max160 and isolated shape feasibility through 2048, proposed max64 as a conservative candidate, and exposed both missing graph reserve and padded/live-row token drift. Report: `performance_milestones/target12_cuda_graph_bucket_policy_preflight/README.md`. |
| TARGET 12.602 CUDA Graph Padding Live-Row Classification | completed with MoE blocker | Used actual max64 candidate boundaries, valid dummy poison, selected layer-0 boundaries, repeat stability, and natural-language smoke. Dummy routes deterministically changed global route planning and the first live MoE output; text sanity remained clean. Report: `performance_milestones/target12_cuda_graph_padding_live_row_classification/README.md`. |
| TARGET 12.6025 MoE Padding Live-Route Contract Fix | completed | Added graph-visible live-row count, masked padded top-k IDs/weights following SGLang, made one masked route plan authoritative for Marlin, and proved poison-invariant live plans/logits/tokens with neutral E2E performance. Report: `performance_milestones/target12_moe_padding_live_route_contract_fix/README.md`. |
| TARGET 12.603 CUDA Graph Memory Reserve Planner | completed with conservative estimator | Added an automatic DSV4/sm80 graph estimate plus 512 MiB margin before KV planning, validated stable max16/64/128 actual-versus-estimated ledgers, and deferred unsafe temporary full-model profiling until mini has a complete KV/backend detach primitive. Report: `performance_milestones/target12_cuda_graph_memory_reserve_planner/README.md`. |
| TARGET 12.604 CUDA Graph Bucket And Reserve Contract Unification | completed | Added one pure resolver/generator before KV planning, unified estimator/runner/benchmark tuples, preserved no-env max16 and disabled/override behavior, and passed a max-only max64 TP8 smoke. Report: `performance_milestones/target12_cuda_graph_bucket_reserve_contract_unification/README.md`. |
| TARGET 12.605 CUDA Graph Recipe Frontier And Selection | completed | Selected req256/graph256 balanced, req4/graph4 512K, and req1/graph1 1M-smoke recipes; rejected req512 because fixed SWA/request state leaves impractical KV capacity. Report: `performance_milestones/target12_cuda_graph_recipe_frontier_selection/README.md`. |
| TARGET 12.606 CUDA Graph Recipe Promotion And Cleanup | current | Wire the selected balanced, low-M/memory, and long-context recipes; publish a capacity-aware DGX A100 performance card across M=4/16/64/128/256 and 1K/4K/16K contexts; run bounded 512K/1M gates; clean stale diagnostics. Prompt: `prompts/archive/target12/TARGET_12.606_dsv4_sm80_cuda_graph_recipe_promotion_cleanup.md`. |
| TARGET 12.61 Long-Context TTFT Owner Attribution And Backend Parity | current | Use one checkpointed 512K prefill plus production-shape microbenches to re-rank indexer, C4/C128 attention, metadata/cache, MoE, and communication; compare exact SGLang/vLLM dispatch and select one evidence-backed implementation route. Prompt: `prompts/archive/target12/TARGET_12.61_dsv4_sm80_long_context_ttft_owner_attribution.md`. |
| TARGET 12.5 Direct/Fused Graph Metadata Writers | deferred | In-graph metadata prep is now promoted. Reopen only if a fresh profile shows residual `raw_graph_copy` or graph metadata kernels as a top release bottleneck. |
| TARGET 12.6 Multi-Stream Latency-Hiding PoC | deferred | Not part of the current route. Reopen only if future evidence proves a material independent owner that cannot be removed, fused, or moved into graph capture. |
| TARGET 12.7 Promotion Gate | todo | Run the final non-MTP release soak after the post-HC envelope, memory accounting, chunked-prefill decision, and fallback/native-backend census converge; promote only if correctness is clean and macro/capacity tradeoffs are repeat-stable. |

## Required Work

### 1. Source Parity First

Create a compact parity table for mini vs SGLang vs vLLM:

```text
graph input buffers:
  input_ids, out_loc, positions, seq_lens, req_pool_indices, page tables

metadata ownership:
  CPU-built, GPU-built, copied into captured buffer, generated in graph,
  reference-assigned, or persistent bucket cache

address contract:
  fields whose tensor object/address must stay fixed across capture/replay
  fields that may be swapped/referenced per replay

copy strategy:
  per-field copy, grouped foreach copy, fused Triton copy, direct writer,
  in-graph prep

stream strategy:
  same stream, graph capture stream, async output copy stream, plan stream,
  TBO stream group, offloader copy stream
```

Explicitly mark which conclusions are source-derived and which are measured in
mini.  Do not assume vLLM/SGLang behavior is faster until tied to the same
owner boundary or measured with a comparable probe.

### 2. Establish A Stable Probe

Start with the short TP8 probe:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  debug/dsv4/benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --scenarios historical_4096_128_bs4 \
  --num-pages 0 \
  --output-dir /tmp/dsv4_target12_short \
  --keep-going
```

Record at minimum:

```text
output tok/s
decode tok/s
decode_forward_enqueue_s
decode_forward_s
decode_prepare_s
graph replay/eager count
prepare_for_replay owner timing
metadata owner counters
replay metadata copy bytes/calls
direct graph metadata bytes/calls
```

Repeat a short probe if first replay outliers dominate the total.  Prefer
steady-state medians and owner counters when deciding whether a local patch is
worth a full soak.

### 3. Safe Graph-Compatible Attribution

Do not use CUDA-event owner timing inside replay if it destabilizes
`cudaGraphLaunch`.  Acceptable instrumentation:

- host-only owner ranges;
- counters for calls/bytes/field/group/status;
- one-owner-at-a-time differential env gates;
- small no-weight or synthetic graph replay probes;
- Nsight Systems only when it captures clean CUDA activity.

Avoid:

- unconditional `torch.cuda.synchronize()` in the hot path;
- per-step CUDA `.item()`, `.cpu()`, `.tolist()`;
- per-token logging;
- diagnostics that change capture/replay semantics.

### 4. Low-Risk Deforestation Candidates

Evaluate these before multi-stream work:

- cache environment and feature flags outside per-step replay loops;
- skip disabled SWA/direct/Marlin/debug branches before building objects;
- group small graph-input copies like SGLang's `_grouped_foreach_copy_`;
- reduce `DSV4AttentionMetadata` object assembly on decode replay;
- reuse per-bucket/per-shape metadata views instead of rebuilding Python
  containers;
- combine scalar metadata copies into one fused helper when possible;
- avoid creating source metadata for fields that direct graph writers already
  produce in the final captured buffer;
- treat component page-table cache dirty/clean rows explicitly so clean rows do
  no work beyond index selection.

### 5. SGLang-Style In-Graph Metadata Prep PoC

Investigate an opt-in route modeled after SGLang:

```text
outside graph:
  copy only raw decode inputs into stable graph buffers
  req_pool_indices, seq_lens, raw_out_loc/out_loc, positions, page-table handles

inside graph:
  derive compressed metadata, c4/c128 lengths, c4/c128 out locs,
  c128 page indices, SWA write locs, and related graph-consumed fields
```

Reference behavior:

```text
SGLANG_PREP_IN_CUDA_GRAPH=True
DSV4RawDecodeMetadata
init_forward_metadata_in_graph()
make_forward_metadata_from_raw_decode()
DSV4AttnMetadata.init_compression_metadata()
dsv4/metadata_kernel.py::init_compression_metadata
```

Mini PoC constraints:

- start with decode-only, no MTP;
- preserve page size `256`;
- preserve prefix/SWA/component correctness;
- do not require full SWA independent lifecycle changes;
- keep fallback to the current replay metadata path;
- measure graph private-pool/capture memory impact;
- verify zero eager fallback and text sanity.

Expected benefit: reduce or eliminate out-of-graph `prepare_for_replay`
metadata staging, at the cost of extra captured graph nodes.  This is likely
higher ROI than trying to run the same out-of-graph staging on another stream.

### 6. Direct/Fused Final Buffer Writers

If full in-graph metadata prep is too large, build smaller direct writers:

- one fused kernel for scalar decode metadata plus lengths;
- one fused C4/C128 index writer that writes final graph buffers;
- one component write-loc/page-table writer;
- optional SWA writer only if SWA remains a measured owner.

Microbench each candidate against current per-field/fused-copy path.  Count
launches as part of the cost; the goal is fewer operations and cleaner graph
inputs, not just moving bytes faster.

### 7. Deferred Multi-Stream PoC Gate

Do not run a stream-overlap PoC in the current TARGET 12 phase. This section is
kept only as a future fallback gate. Reopen it only if a remaining independent
owner is still large after deforestation/in-graph-prep and cannot be removed,
fused, or moved into the captured graph.

Candidate experiments:

- generate independent C128/SWA/component metadata on auxiliary streams, then
  `main_stream.wait_event()` before `g.replay()`;
- move next-step CPU-only planning earlier, if it does not depend on sampled
  tokens;
- async copy output/logprob-like CPU results if they become a measured owner;
- study SGLang TBO only as a broader batch-overlap mechanism, not as a quick
  metadata-copy fix.

Hard requirements:

- no graph pool corruption;
- no hidden sync;
- deterministic dependency events;
- correctness and text sanity unchanged;
- short TP8 probe must beat same-stream fused baseline, not merely match it.

If stream/event overhead cancels the win, stop and document no-go.

## Validation

Unit/correctness gates for metadata changes:

```bash
python -m pytest -q tests/attention/test_deepseek_v4_backend_metadata.py -k 'swa or replay or ownership'
python -m pytest -q tests/kernel/test_deepseek_v4_wrappers.py
```

Short performance gate:

```text
historical_4096_128_bs4
```

Promotion-style soak after a winning candidate:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
```

Text sanity:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  debug/dsv4/benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_release_default \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_text_smoke.json
```

## Stop Conditions

Stop a child thread when any of these is true:

1. It proves a specific owner accounts for the replay gap and writes the next
   implementation target.
2. A patch recovers most of the short-probe gap and passes the focused
   correctness gates.
3. A candidate mechanism is measured no-go against the current same-stream
   baseline.
4. The remaining unexplained delta is below about `2%` output tok/s on the
   short probe.
5. The work would require weakening prefix/SWA/component ownership contracts.

Do not keep polishing random sub-owners after the main replay gap is explained.

## Deliverables

Use:

```text
performance_milestones/target12_decode_replay_metadata_latency_hiding/
```

Required README sections:

- current git commit and dirty-state summary;
- source parity table for mini/SGLang/vLLM;
- short-probe baseline and after-patch metrics;
- owner attribution table;
- mechanism decision: deforestation, in-graph prep, direct writer, stream PoC,
  or no-go;
- correctness gates;
- promotion recommendation and next child target.

# TARGET 12: DSV4 SM80 Decode Replay Metadata And Latency Hiding

## Status

Active follow-up after post-MTP-cleanup non-MTP baseline attribution.

TARGET 11 MTP is paused and archived.  The release branch should continue from
the non-MTP DSV4 sm80/A100 serving baseline:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
PyNCCL threshold32m default for A100/sm80 DSV4
--page-size 256 --num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

The immediate trigger is:

```text
performance_milestones/misc_post_mtp_cleanup_replay_attribution/README.md
```

That report shows the remaining regression is not MTP leakage, communication
count/bytes, graph replay coverage, or wrapper count.  The gap is concentrated
around decode CUDA graph replay setup and captured forward time, especially
metadata preparation/staging before `g.replay()`.

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
| TARGET 12.45 In-Graph Metadata Promotion Soak | active | Repeat-pair the short probe, run four-scenario soak, verify oracle/text/fallback boundaries, and decide whether to promote, continue opt-in, or move to TARGET 12.5. Prompt: `prompts/TARGET_12.45_dsv4_sm80_ingraph_metadata_promotion_soak.md`. |
| TARGET 12.5 Direct/Fused Graph Metadata Writers | todo | If in-graph prep is incomplete or too risky, fuse direct C4/C128/SWA/component metadata generation into fewer kernels that write final captured buffers. |
| TARGET 12.6 Multi-Stream Latency-Hiding PoC | deferred | Not part of the current route. Reopen only if future evidence proves a material independent owner that cannot be removed, fused, or moved into graph capture. |
| TARGET 12.7 Promotion Gate | todo | Run short and four-scenario non-MTP soak; promote only if correctness is clean and macro wins are repeat-stable. |

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
  benchmark/offline/deepseek_v4_perf_matrix.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --scenarios historical_4096_128_bs4 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
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
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_moereducebf16 \
  --page-size 256 --num-pages 128 \
  --enable-dsv4-radix-prefix-cache \
  --enable-dsv4-component-loc-ownership \
  --allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16 \
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

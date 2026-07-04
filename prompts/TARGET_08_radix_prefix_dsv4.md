# TARGET 08: DSV4 Radix/SWA Prefix Cache

## Status

Phase 1, serving graph bucket policy, CUDA graph memory attribution, prefix
stability, memory ledger, TARGET 08.19 correctness probing, TARGET 08.195
slot/page invariance probing, TARGET 08.196 batched attention/indexer probing,
TARGET 08.197 q-path same-shape/same-input probing, TARGET 08.198 post-layer0
same-shape drift analysis, TARGET 08.20 fail-closed V1 design, TARGET
08.21.1-08.21.4 Route B component ownership/graph integration, TARGET 08.22
final promotion gate rerun, TARGET 08.22.1 component mapping lifecycle fix, and
TARGET 08.24 component-aware metadata deforest/copy-elision experiment, TARGET
08.25 direct graph metadata buffer experiment, TARGET 08.26 remaining-gap
attribution reset, TARGET 08.27 SGLang-aligned Route B metadata lifetime cache,
TARGET 08.28 Route B lifetime-cache promotion gate, and TARGET 08.29 Route B
lifetime promotion cleanup are complete.  Continue with TARGET 08.30: reprofile
the promoted `dsv4_sm80_a100_victory_prefix_routeb_lifetime` prefix preset and
choose the next evidence-based phase.

TARGET 07 is closed.  The promoted non-prefix path is stable enough to start
prefix-cache work:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
```

Start with fixed or capped page counts such as `--num-pages 128`.  Do not treat
automatic `memory_ratio=0.9` graph-mode capacity as a serving default yet:
TARGET 07.79 showed that it can choose a very large KV pool and OOM during graph
capture.

Phase-1 artifact:

```text
performance_milestones/target08_radix_prefix_dsv4/
```

Phase-1 result:

- added `--enable-dsv4-radix-prefix-cache`;
- default DSV4 path still uses the previous naive cache unless explicitly
  enabled;
- requires `page_size % 128 == 0`, with target runs using page size `256`;
- full-token pages remain the canonical owner;
- C4/C128/C4-indexer/compression-state retention is derived from full-token
  pages through the existing DSV4 KV pool refcount path;
- correctness tests and TP8 text smoke passed;
- shared-prefix A/B showed TTFT and prefill-forward reductions;
- do not promote by default yet.

The next work is no longer "implement the first prefix cache".  It is to make
the feature serving-credible by separating DSV4 component ownership from
full-token page ownership, so old full/SWA pages can be released without stale
C4/C128/indexer/state reads.

## Goal

Add correct, measurable radix prefix cache support for DeepSeek V4 Flash in
mini-sglang.

This target is complete when repeated requests with shared prefixes skip
cached-prefix prefill, produce the same decode results as prefix-disabled mode,
and release all DSV4 cache components correctly under eviction.

Primary value:

- reduce TTFT and prefill work for shared-prefix workloads;
- preserve decode graph replay and the promoted TARGET 07 path;
- establish a DSV4-aware cache ownership model before future low-precision or
  capacity work.

## TARGET 08 Subtarget Roadmap

Run in this order:

| Stage | Prompt | Status | Purpose |
| --- | --- | --- | --- |
| TARGET 08 phase 1 | this file + `performance_milestones/target08_radix_prefix_dsv4/` | complete opt-in | Implemented conservative page-aligned/full-page-owner DSV4 radix prefix cache. |
| TARGET 08.05 | `prompts/TARGET_08.05_dsv4_sm80_serving_workload_cuda_graph_bucket_policy.md` | complete | Built serving workload suite and selected `[1,2,4,8,16]` as the smallest measured zero-eager bucket set. |
| TARGET 08.06 | `prompts/TARGET_08.06_dsv4_sm80_cuda_graph_memory_attribution.md` | complete | Confirmed the large capture delta is a real first-graph/private-pool cost, not bucket count, metadata, greedy sample, `max_seq_len`, `num_pages`, or missing pool reuse. |
| TARGET 08.07 | `prompts/TARGET_08.07_dsv4_sm80_bf16_cache_graph_memory_attribution.md` | complete | Ruled out promoted BF16 caches as the material cause of the large CUDA graph private-pool delta. |
| TARGET 08.10 | `prompts/TARGET_08.10_dsv4_sm80_prefix_cache_serving_stability_promotion_gate.md` | complete controlled opt-in | Showed strong shared-prefix wins and stable graph replay, but kept prefix cache opt-in because generated-token correctness was not clean enough for default promotion. |
| TARGET 08.18 | `prompts/TARGET_08.18_dsv4_sm80_prefix_cache_memory_ledger_go_nogo.md` | complete | Quantified full-page-owner memory/capacity cost and recommended guarded component-retention work. |
| TARGET 08.19 | `prompts/TARGET_08.19_dsv4_sm80_prefix_cache_logit_metadata_correctness.md` | complete blocked | Metadata boundary was clean, but logits exposed a DSV4 exact-path slot/page-location blocker. |
| TARGET 08.195 | `prompts/TARGET_08.195_dsv4_sm80_exact_path_slot_page_invariance.md` | complete partial fix | Fixed a real compressor cross-request pooling bug and established guards, but remaining batched attention/indexer row-coupling still blocks broad oracle use. |
| TARGET 08.196 | `prompts/TARGET_08.196_dsv4_sm80_batched_attention_indexer_row_coupling.md` | complete narrowed | Added attention/indexer debug hooks and exact-bs graph guard; found layer0 q-path drift but did not clear broad correctness. |
| TARGET 08.197 | `performance_milestones/target08_q_path_same_shape_same_input_invariance/README.md` | complete classification | Classified the layer0 q-path issue as GEMM shape numeric drift, not q_norm/RoPE row-coupling; still blocked by post-layer0 same-shape decode drift. |
| TARGET 08.198 | `prompts/TARGET_08.198_dsv4_sm80_post_layer0_same_shape_decode_drift.md` | complete guarded | Found tiny later-layer attention/indexer drift amplified by small logits margins; accepted guarded oracle because batch-slot invariance is not guaranteed. |
| TARGET 08.20 | `prompts/TARGET_08.20_dsv4_sm80_sglang_style_swa_component_retention.md` | complete rejected | Added a fail-closed V1 opt-in and proved runtime V1 is unsafe without component-level ownership. |
| TARGET 08.21 | `prompts/TARGET_08.21_dsv4_sm80_component_loc_ownership_route_b.md` | route overview | Splits Route B into small executable targets; do not run it as one monolithic implementation. |
| TARGET 08.21.1 | `prompts/TARGET_08.21.1_dsv4_sm80_component_loc_table_preflight.md` | complete | B0: proved direct C4/C128/indexer/state loc metadata can reproduce phase-1 derived metadata while full pages stay live. |
| TARGET 08.21.2 | `prompts/TARGET_08.21.2_dsv4_sm80_independent_compressed_indexer_ownership.md` | complete | B1: implemented independent C4/C128/indexer ownership behind an opt-in. |
| TARGET 08.21.3 | `prompts/TARGET_08.21.3_dsv4_sm80_compression_state_ownership.md` | complete | B2: implemented independent C4/C128/indexer compression-state ownership; SWA-tail guard remains. |
| TARGET 08.21.4 | `prompts/TARGET_08.21.4_dsv4_sm80_route_b_graph_deforest_serving.md` | complete preferred opt-in candidate | B3: restored Route B graph replay for `[1,2,4,8,16]`; deforest remains guarded; full serving gate still needed. |
| TARGET 08.22 | `prompts/TARGET_08.22_dsv4_sm80_route_b_final_prefix_promotion_gate.md` | complete preferred opt-in | Rerun passed correctness/text/graph, selected Route B as preferred opt-in, and identified guarded metadata deforest as the main remaining gap. |
| TARGET 08.22.1 | `prompts/TARGET_08.22.1_dsv4_sm80_route_b_component_mapping_lifecycle_fix.md` | complete | Fixed `DSV4 component mapping is missing for active C4 full pages`; focused Route B TP8 graph scenarios pass. |
| TARGET 08.23 | `prompts/TARGET_08.23_dsv4_sm80_independent_swa_ownership.md` | deferred conditional | Implement SGLang-aligned independent SWA ownership only if later evidence shows the SWA-tail guard materially blocks serving capacity or hit rate. |
| TARGET 08.24 | `prompts/TARGET_08.24_dsv4_sm80_route_b_metadata_deforest_copy_elision.md` | complete keep experimental | Proved component-aware metadata generation is safe, but the runtime opt-in regressed performance because large source metadata tensors were still materialized and staged. |
| TARGET 08.25 | `prompts/TARGET_08.25_dsv4_sm80_route_b_direct_graph_metadata_buffers.md` | complete keep experimental | Generated SWA/C4/C128 metadata directly into graph buffers, but large-wave gains were too small and full direct generation regressed throughput. |
| TARGET 08.26 | `prompts/TARGET_08.26_dsv4_sm80_route_b_remaining_gap_attribution_reset.md` | complete recommends SGLang-aligned metadata lifetime | Re-ranked Route B direct C4 remaining gap: decode prepare component page-table and stable metadata updates dominate; SWA-tail and forward attention/MoE/communication are not the next owner. |
| TARGET 08.27 | `prompts/TARGET_08.27_dsv4_sm80_sglang_aligned_route_b_metadata_lifetime.md` | complete strong opt-in | Added SGLang-aligned Route B component page-table lifetime cache; `serving_mixed_112req_wave16` improved from `138.13` to `162.47` output tok/s and graph replay stayed `441/0`. |
| TARGET 08.28 | `prompts/TARGET_08.28_dsv4_sm80_route_b_lifetime_cache_promotion_gate.md` | complete promote | Promoted the lifetime cache: verifier/text/eviction/prefix_multi/decode controls passed; `serving_mixed` reached `163.72` output tok/s with `441/0` replay. |
| TARGET 08.29 | `prompts/TARGET_08.29_dsv4_sm80_route_b_lifetime_promotion_cleanup.md` | complete cleanup | Added promoted preset `dsv4_sm80_a100_victory_prefix_routeb_lifetime`, preserved verifier env reset behavior, and kept the old diagnostic lifetime name as an alias. |
| TARGET 08.30 | `prompts/TARGET_08.30_dsv4_sm80_post_prefix_reprofile_next_bottleneck.md` | active next | Reprofile the promoted prefix path and decide whether the next evidence-based phase is TARGET 09, TARGET 10, more TARGET 08 cache work, or serving hardening. |

Rationale:

- TARGET 08.05, 08.06, and 08.07 established the graph-bucket and graph-memory
  context needed for credible serving tests.
- TARGET 08.10 showed the prefix path is useful and operationally stable, but
  synthetic generated-token mismatches mean promotion needs a more deterministic
  correctness boundary.
- TARGET 08.18 proved full-page-owner retention has material logical capacity
  cost, so component retention is worth planning.
- TARGET 08.19 showed phase-1 prefix metadata is clean, but also showed the
  prefix-disabled exact path is not a clean oracle because identical prompts can
  differ across slots/page locations.
- TARGET 08.195 fixed a real compressor cross-request pooling bug and showed
  single-request page/table churn is clean, but remaining drift still appears in
  batched attention/indexer paths.
- TARGET 08.196/08.197 narrowed the blocker: layer0 q_norm/RoPE is not guilty,
  and the original q-path drift is shape-dependent GEMM numeric drift.  However,
  same-shape decode logits can still drift later and change sampled tokens.
- TARGET 08.198 concluded that mini does not currently guarantee batch-slot
  invariance.  Cross-slot/filler/identical-row generated-token equality should
  be diagnostic only.  TARGET 08.20 may continue with a slot-pinned/same-layout
  oracle and text smoke that rejects obvious correctness failures such as
  garbled or invalid-byte text, degenerate output, crashes, leaks, or
  cache-state corruption.
- TARGET 08.20 proved that runtime V1 is unsafe in mini's current full-page-owner
  model and left the V1 opt-in fail-closed.
- TARGET 08.21 is now a Route B family overview.  The executable work is split
  into TARGET 08.21.1/08.21.2/08.21.3/08.21.4 so child threads stop after each
  evidence boundary instead of trying to solve the whole cache ownership model
  at once.  Route A retained-store materialization may be used as an oracle
  only, not as the primary runtime path.
- TARGET 08.21.1-08.21.4 completed the Route B stack: direct component loc
  tables, independent C4/C128/indexer ownership, independent compression-state
  ownership, and graph replay for `[1,2,4,8,16]`.  The remaining guard is SWA
  KV ownership: SWA still lives in the full-token namespace, so Route B keeps a
  live full/SWA tail.
- TARGET 08.22 rerun passed correctness/text/graph and selected Route B as the
  preferred prefix-cache opt-in.  Route B recovered `0.9648x` of phase-1 saved
  prefill tokens, while the exact page-multiple SWA-tail guard accounted for
  only a small saved-token delta in the measured suite.  The larger remaining
  gap is decode metadata overhead: Route B keeps deforest guarded off because
  the old deforest path assumes component metadata can be derived from full
  token locations.
- TARGET 08.24 proved the component-aware formula is safe but did not solve the
  performance problem.  It changed how metadata is generated while still
  materializing large source tensors and staging them into graph buffers, so the
  opt-in remains experimental.
- TARGET 08.25 showed that direct graph metadata generation is safe, and C4-only
  direct generation can give a small local win, but full SWA+C4+C128 direct
  generation did not meet the performance gate.  TARGET 08.26 reset attribution
  around Route B direct C4: the remaining phase1 gap is overwhelmingly decode
  prepare, with component page-table construction as the largest owner and
  per-request/per-prefix-hit metadata rows updated on all 441 decode replay
  steps.  TARGET 08.27 followed SGLang's stable request-row idea and added a
  request/table-slot keyed component page-table lifetime cache instead of a
  from-scratch dirty-row subsystem.  On `serving_mixed_112req_wave16`, Route B
  direct C4 improved from `138.1281` to `162.4726` output tok/s, decode prepare
  dropped from `4.2067 s` to `1.1416 s`, and graph replay stayed `441/0`.
  TARGET 08.28 gated this opt-in across verifier, prefix_multi, eviction
  pressure, decode controls, and table-slot/component-row lifecycle, and reached
  a `promote` decision.  `serving_mixed_112req_wave16` reached `163.7220`
  output tok/s, verifier passed serving and eviction workloads, prefix_multi
  recovered `49152` saved prefill tokens, and graph replay stayed zero-eager.
  TARGET 08.29 cleaned this into the promoted preset
  `dsv4_sm80_a100_victory_prefix_routeb_lifetime`, preserving the verifier env
  opt-in across benchmark/text-smoke variant reset and keeping the old
  `dsv4_sm80_a100_victory_directgraphmetadata_c4_routeb_lifetime` name as a
  historical alias.  TARGET 08.30 should reprofile the whole system with the
  promoted preset.  TARGET 08.23 remains conditional and should be
  revisited only if later workloads show SWA-tail retention or exact
  page-multiple shortening is a real capacity or hit-rate bottleneck.
- TARGET 09 remains reserved for low-precision research.  Do not rename
  SGLang-style SWA retention to TARGET 09.

## Required vLLM/SGLang Alignment

Do not implement this from a blank page.  First inspect and map the relevant
cache-state designs.

Mini references:

- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`

vLLM references:

- `/workspace/vllm-dsv4-docker/vllm/v1/attention/backends/mla/sparse_swa.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_coordinator.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/kv_cache_utils.py`
- `/workspace/vllm-dsv4-docker/vllm/v1/core/block_pool.py`
- `/workspace/vllm-dsv4-docker/vllm/model_executor/layers/deepseek_v4_attention.py`

SGLang references:

- `/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py`
- `/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py`
- `/workspace/sglang-main/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
- `/workspace/sglang-main/python/sglang/srt/environ.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/allocator/swa.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_radix_cache.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
- `/workspace/sglang-main/python/sglang/srt/mem_cache/unified_cache_components/swa_component.py`

Old branch references, use carefully:

- `git show dsv4:python/minisgl/kvcache/deepseek_pool.py`
- `git show dsv4:tests/core/test_deepseek_prefix_cache.py`
- `git show dsv4:prompts/PLAN_tier3_correctness_milestone.md`
- `git show dsv4:prompts/PLAN_tier4_tilelang_sm80_milestone.md`

Useful vLLM ideas to compare against:

- hybrid KV cache coordination across groups with different block sizes;
- block-hash based prefix matching and eviction;
- `DeepseekV4SWACache` block sizing and alignment;
- `SlidingWindowMLASpec`;
- per-layer sparse/SWA metadata builders;
- fixed-point prefix hit length that all cache groups can support.

## DSV4 Cache Model

A prefix hit must represent more than ordinary full-token KV pages.

Track or reconstruct these components consistently:

- full token KV slots;
- SWA-visible slots/window boundary;
- C4 attention compressed slots;
- C128 attention compressed slots;
- C4 indexer slots;
- compression state or enough warmup information to rebuild it;
- logical token length;
- page-aligned prefix length;
- owner/refcount state needed for eviction.

Recommended design rule:

- make full-token pages the canonical prefix ownership unit;
- derive C4/C128/indexer component ownership from the same token range where
  possible;
- make prefix-cache pin/unpin use the same DSV4 component refcount machinery
  that normal allocation/free uses;
- avoid double-free by routing all final token release through one DSV4-aware
  owner path.

Compression state is the hardest part.  Phase 1 may choose correctness-first
reconstruction around the prefix boundary, for example by recomputing the small
ring/warmup state from cached tokens, instead of persisting every internal state
object in the radix tree.

## Phase-1 Plan Already Completed

### Phase 0: Source Parity And Design Note

Produce a short note under:

```text
performance_milestones/target08_radix_prefix_dsv4/
```

It must answer:

- how mini currently matches prefix pages;
- how mini currently frees DSV4 full/C4/C128/indexer components;
- how vLLM coordinates SWA and compressed cache groups;
- what exact prefix state mini will store vs reconstruct;
- what lengths must be page-aligned or component-aligned.

Do not implement before this note exists.

### Phase 1: Minimal Explicit Opt-In

Add an explicit opt-in for DSV4 radix prefix cache.  Default behavior must stay
unchanged until correctness and eviction tests pass.

The minimal path should support:

- full prefix hit;
- partial prefix hit at safe page boundaries;
- prefix miss fallback;
- suffix prefill after a hit;
- decode after suffix prefill;
- eviction after requests finish.

### Phase 2: Correctness Tests

Add tests or smoke scripts for:

- full prefix hit;
- partial prefix hit;
- prefix miss;
- prefix eviction;
- multi-request shared system prompt;
- SWA boundary below, at, and above `window_size=128`;
- C4/C128 component boundary near compression ratios;
- repeated hit/evict cycles;
- logits or generated-token comparison against radix-disabled mode.

The first correctness gate may use small prompt/decode lengths, but it must use
the same DSV4 configuration style as normal usage, especially page size `256`.

### Phase 3: Metrics

Expose at least:

- prefix hit length;
- prefix hit rate;
- saved prefill tokens;
- retained prefix pages;
- retained or pinned DSV4 component slots;
- evicted prefix tokens/pages;
- suffix prefill tokens after hit;
- memory retained by prefix cache.

### Phase 4: Performance Gate

Run a shared-prefix benchmark that includes:

- repeated shared system prompt;
- at least one no-hit control;
- prefix cache on/off comparison;
- TTFT and prefill-forward delta;
- decode throughput sanity check;
- graph replay/eager decode status;
- memory-retention and eviction ledger.

Do not expect single-token decode throughput to improve.  The target is prefill
reuse and TTFT reduction on shared-prefix workloads.

## Current Phase-1 Caveats

- The implementation is intentionally conservative: retaining a prefix retains
  full-token pages as canonical owner.  This is correct and simple but less
  memory-efficient than SGLang-style independent SWA/component retention.
- Runtime opt-in requires `page_size % 128 == 0`.  Keep this requirement unless
  a later target proves a stronger safe hybrid-alignment rule from vLLM/SGLang.
- The first shared-prefix benchmark used a second-stage batch size 7 while
  graph capture covered `[1,2,4]`; this caused eager decode for bs7.  TARGET
  08.05 must establish a serving graph bucket policy before 08.10 promotion
  testing.
- Automatic KV sizing is not yet a safe graph-mode serving default.

## Done Criteria

- DSV4 radix prefix cache can be enabled explicitly.
- Shared-prefix requests skip cached-prefix prefill.
- Outputs match prefix-disabled mode for the tested prompts.
- SWA boundary behavior is correct around `128` tokens.
- C4/C128/indexer component ownership survives hit, suffix prefill, decode, and
  eviction.
- Eviction frees all DSV4 cache components without leaks or double-free.
- Metrics report hit rate, saved prefill tokens, retained memory, and evictions.
- A milestone README records the design, commands, correctness results,
  performance result, and remaining risks.

## Stop Rules

Stop and report blocked if:

- prefix hits corrupt generated text or logits versus radix-disabled mode;
- DSV4 component refcounting cannot be made unambiguous;
- SWA boundary reuse is not understood;
- graph replay is unexpectedly disabled by the feature;
- eviction leaks or double-frees any full/C4/C128/indexer component;
- automatic KV sizing causes graph-capture OOM during the target.

Stop after a correct opt-in and benchmark report.  Do not broaden into FP8 KV
cache, INT8 MoE, PyNCCL, or attention-kernel optimization inside TARGET 08.

## Non-Goals

- Changing the promoted TARGET 07 default path before correctness passes.
- Implementing a new eviction policy beyond correctness and basic memory
  control.
- Adding FP8 KV cache or low-precision cache changes.
- Optimizing C4A/C128A sparse attention kernels.
- Tuning NCCL or PyNCCL.
- Treating prefix cache as a replacement for decode kernel optimization.

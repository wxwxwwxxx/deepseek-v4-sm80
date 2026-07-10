# TARGET 12.50: DSV4 SM80 Release Bundle Opt-In Promotion Gate

## Background

TARGET 12.47 proved that SGLang-style in-graph replay metadata prep is ready for
the A100/sm80 DeepSeek V4 Route-B prefix baseline. TARGET 12.48 folded that
subset into Engine release defaults. TARGET 12.49 then tried to stretch the
release-default path to long context and larger active decode batches, but it
failed at the initial release smoke:

```text
performance_milestones/target12_release_long_context_large_batch_soak/README.md
```

The key 12.49 conclusion was not "CUDA graph bucket policy is bad."  The actual
blocker is that the current release-default recipe is incomplete:

- A100 victory currently implies the `marlin_wna16` MoE expert backend.
- The release default does not yet prebuild Marlin WNA16 caches before KV
  capacity planning.
- It does not release original routed expert weights before KV allocation.
- It does not apply Marlin release-capacity credit to automatic `num_pages`.
- The first graph warmup can therefore lazily build/repack Marlin state after KV
  has consumed most free memory, causing OOM.

12.49 also showed that the Marlin prebuild/release A/B can pass the same smoke:

```text
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc
MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT=1
MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component
```

SWA independent lifecycle is also an important serving/capacity feature from
TARGET 08.  It can provide large memory savings by decoupling SWA pages from
prefix/component/compressed ownership.  It should be treated as a strong
promotion candidate, not as a casual optional experiment.  However, it changes
the KV/cache ownership contract and currently interacts with in-graph replay
metadata prep: when SWA independent lifecycle is enabled, the current
`prep_metadata_in_graph` path can fail open as unsupported.  This target should
try hard to make SWA independent compatible with the release bundle before
leaving it opt-in.

## Goal

Build and validate a complete DeepSeek V4 A100/sm80 release bundle from the
important TARGET 8/10 exact-path features, while keeping TARGET 9
low-precision research out of the default path.

The intended end state is:

```text
LLM("/models/DeepSeek-V4-Flash", ...)
```

should resolve to a high-performance, high-capacity, serving-oriented default
without requiring users to memorize page size, radix/component ownership, graph
buckets, Marlin release flags, communication threshold flags, or metadata flags.

Fallback/oracle routes should remain available through explicit opt-outs or
benchmark variants, but the optimized release recipe should be the normal path.

## Promotion Bias

Use this bias:

1. Marlin WNA16 prebuild/release/capacity credit should be considered part of
   the release recipe unless a correctness gate fails.
2. SWA independent lifecycle should be aggressively pursued for default
   promotion because the memory/capacity benefit is large.
3. If SWA independent lifecycle exposes bugs, use the TARGET 08 development
   playbook: contract-first audit, small reproduction, ownership/lifetime fix,
   then smoke/macro rerun.
4. Keep SWA independent opt-in only if a severe correctness problem remains and
   the fix cost is clearly too large for the release cycle.
5. Do not default-promote low-precision paths from TARGET 09 in this target.

## Candidate Feature Sets

### Tier A: Release-Complete Exact/Capacity Bundle

These are expected to become defaults if gates pass:

```text
page_size=256
attention_backend=dsv4
radix prefix cache enabled
component loc ownership enabled
cuda_graph_bs=[1,2,4,8,16]
PyNCCL threshold32m for DSV4 sm80 TP
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SM80_MOE_REDUCE_BF16=1
MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc
MINISGL_DSV4_MARLIN_WNA16_RELEASE_CAPACITY_CREDIT=1
MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component
```

If release-capacity credit is enabled, report:

- released raw expert bytes/rank;
- guard/reserved bytes;
- net credit bytes/pages/tokens;
- final automatic `num_pages`;
- graph private-pool delta;
- free memory before/after graph capture.

### Tier B: SWA Independent Default Candidate

These are high-priority default candidates, but must be validated as a separate
promotion stage:

```text
MINISGL_DSV4_SWA_INDEPENDENT_LIFECYCLE=1
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_SWA_DIRECT_REPLAY_METADATA_FUSED=1
```

Required checks:

- Does SWA independent lifecycle keep text sanity and request-local correctness?
- Does graph replay remain zero-eager for `[1,2,4,8,16]`?
- Does `MINISGL_DSV4_SM80_PREP_METADATA_IN_GRAPH=1` remain active, or does it
  fail open with `swa_independent_lifecycle_not_supported`?
- If it fails open, is the resulting performance still acceptable, or should
  this target implement a compatibility fix?
- What is the SWA memory saving for no-prefix, prefix-hit, and mixed serving
  cases?
- Does prefix eviction/tombstone behavior remain contract-compliant?

If SWA independent lifecycle fails, do not immediately drop it.  First inspect:

```text
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/archive/target08/TARGET_08.45_dsv4_sm80_swa_independent_lifecycle_contract.md
prompts/archive/target08/TARGET_08.47_dsv4_sm80_swa_contract_unified_fix.md
prompts/archive/target08/TARGET_08.48_dsv4_sm80_marlin_swa_auto_cross_case_lifecycle_fix.md
prompts/archive/target08/TARGET_08.55_dsv4_sm80_compressed_metadata_boundary_replay_cleanup.md
```

Use small targeted tests or no-weight harnesses before running long full-model
bisections.

### Tier C: Keep Out Of Default For Now

Do not promote these in this target:

```text
INT8 MoE / W8A8
FP8 KV cache
Dense FP8 projection as a speed feature
Experimental custom communication beyond PyNCCL threshold32m
MTP/speculative decoding
```

They are research or paused paths unless a fresh profile/capacity ledger makes
them the best next lever.

## Test Matrix

Run each stage in a fresh process.  CUDA graph capture, Marlin release, and env
defaults must not be compared inside a same-process multi-variant run.

### Stage A: 12.47/12.48 Current Release Baseline

Use current release defaults as the baseline.  Record that 12.49 found it can
OOM with auto capacity.

### Stage B: Tier A Bundle

Enable the complete Marlin prebuild/release/capacity recipe on top of the
current release defaults.  This is the first likely new default bundle.

Minimum command shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 \
  benchmark/offline/deepseek_v4_text_smoke.py \
  --model-path /models/DeepSeek-V4-Flash \
  --variants dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release \
  --num-pages 0 \
  --fail-on-warning \
  --output /tmp/dsv4_target12_50_tier_a_text_smoke.json
```

Then run 12.47-equivalent macro scenarios:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
```

### Stage C: Tier A + SWA Independent

Enable SWA independent lifecycle on top of Tier A:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent
```

Measure correctness, capacity, graph replay, and whether in-graph metadata prep
falls back.

### Stage D: Tier A + SWA Independent + Direct SWA Metadata

Enable direct SWA metadata optimizations:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_swa_independent_swadirect
```

If there is no existing variant for direct replay metadata fused, add only the
minimum test harness or env override needed to evaluate it.  Do not mix this
with unrelated cleanup.

### Stage E: Proposed Final Default Bundle

Pick one final candidate:

- Tier A only;
- Tier A + SWA independent;
- Tier A + SWA independent + direct SWA metadata;
- or Tier A default with SWA independent as a documented near-term blocker.

Run a final text smoke and 12.47 macro rerun on that candidate.

## Required Measurements

For every stage, record:

- resolved config: page size, cache type, prefix/component/SWA flags,
  graph buckets, PyNCCL threshold, `num_pages`;
- active env/toggles;
- text sanity result and warnings;
- graph capture success/failure;
- graph replay/eager counts;
- graph private-pool delta;
- Marlin WNA16 prebuild/release report;
- release capacity credit bytes/pages/tokens;
- SWA independent allocation report if enabled;
- output tok/s and decode tok/s for the macro scenarios;
- prefix cache hit/miss behavior for prefix workload;
- any fallback/oracle toggles required to keep sanity.

## Source And Contract Parity

Use mini's existing code first:

```text
python/minisgl/engine/engine.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/scheduler/cache.py
python/minisgl/attention/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
benchmark/offline/deepseek_v4_text_smoke.py
```

Use SGLang/vLLM only where it helps avoid reinventing lifecycle or graph
metadata behavior:

```text
/workspace/sglang-main
/workspace/vllm-dsv4-docker
```

Clearly mark source-derived conclusions when runtime comparison is not run.

## Implementation Guidance

If Stage B passes, implement the Tier A release-default changes directly:

- add Marlin prebuild/release/capacity env defaults to the Engine release
  default bundle;
- make sure explicit fallback/oracle env disables still work;
- update tests to assert the new defaults;
- keep dedicated benchmark variants for old/fallback comparison.

If Stage C or D fails:

- inspect whether the failure is correctness, capacity, graph replay, or
  performance;
- prefer targeted ownership/metadata fixes over disabling SWA independent;
- if `prep_metadata_in_graph` is the only blocker, decide whether to port/fix
  the SWA-independent graph metadata boundary or accept a measured fallback;
- open a follow-up child target only if the fix is broad enough to exceed this
  gate.

## Validation

Minimum local validation after code edits:

```bash
python -m py_compile \
  python/minisgl/engine/engine.py \
  python/minisgl/kernel/deepseek_v4.py \
  python/minisgl/kvcache/deepseek_v4_pool.py \
  python/minisgl/attention/deepseek_v4.py \
  benchmark/offline/deepseek_v4_perf_matrix.py \
  benchmark/offline/deepseek_v4_text_smoke.py

python -m pytest -q \
  tests/engine/test_dsv4_release_defaults.py \
  tests/engine/test_marlin_wna16_release_credit.py \
  tests/core/test_dsv4_cache_option_guards.py \
  tests/core/test_deepseek_v4_kvcache.py \
  tests/attention/test_deepseek_v4_backend_metadata.py \
  tests/benchmark/test_deepseek_v4_perf_matrix.py \
  tests/benchmark/test_deepseek_v4_text_smoke.py \
  tests/kernel/test_deepseek_v4_wrappers.py::test_dsv4_sm80_v0_bf16_bundle_env_policy
```

Full TP8 validation should use fresh `torchrun` processes per stage.

## Output

Write the report to:

```text
performance_milestones/target12_release_bundle_optin_promotion_gate/README.md
```

The report must include:

- final recommended release bundle;
- exact env/config defaults proposed;
- features promoted to default;
- features kept opt-in and why;
- fallback/oracle mechanism;
- text sanity results;
- 12.47 macro comparison table;
- Marlin capacity ledger;
- SWA independent capacity/perf/correctness ledger;
- whether SWA independent should enter default now, after a bounded fix, or
  remain opt-in for a severe reason;
- follow-up target recommendations.

## Stop Conditions

Stop and report when one of these is true:

1. Tier A cannot pass text sanity or graph replay even after targeted Marlin
   release/capacity fixes.
2. Tier A passes and SWA independent also passes the default gates.
3. Tier A passes but SWA independent has a concrete, severe blocker whose fix
   must be split out.
4. The final bundle candidate has clean 12.47 macro results and a clear
   recommendation for rerunning TARGET 12.49.

Do not spend the whole target polishing low-priority metadata kernels or
exploring low precision.  This target is about release bundle completeness and
promotion decisions.

# TARGET 08.19: DSV4 Prefix Cache Logit And Metadata Correctness

## Status

Active next TARGET 08 subtarget.

Run this after TARGET 08.18 and before implementing TARGET 08.20.

TARGET 08.10 showed that the phase-1 DSV4 radix prefix cache is stable as a
controlled opt-in and gives large TTFT/prefill wins, but synthetic generated
tokens diverged between prefix-disabled and prefix-enabled hit workloads.  The
generated-token oracle is not clean enough to prove a semantic bug, because the
prefix-disabled control also showed slot-dependent continuations for identical
prompts.  However, default promotion and component-retention work both need a
cleaner correctness boundary.

## Goal

Determine whether the current phase-1 prefix cache is logits-equivalent to the
prefix-disabled path at the first suffix/decode boundary.

The target should answer:

1. Is the TARGET 08.10 mismatch caused by a real prefix-cache state bug or by a
   noisy generated-token oracle?
2. If it is a real bug, which boundary owns it:
   - `cached_len` and suffix prefill start;
   - positions / sequence lengths;
   - page table entries;
   - SWA window indices and lengths;
   - C4/C128/indexer compressed indices;
   - compression state boundary;
   - CUDA graph replay input copying or greedy-sample tie-breaking.
3. What minimal fix or guard is required before default promotion?

## Starting Point

Read:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.10_dsv4_sm80_prefix_cache_serving_stability_promotion_gate.md`
- `performance_milestones/target08_prefix_cache_serving_stability/README.md`
- `performance_milestones/target08_prefix_cache_memory_ledger/README.md`
- `python/minisgl/scheduler/cache.py`
- `python/minisgl/scheduler/scheduler.py`
- `python/minisgl/kvcache/radix_cache.py`
- `python/minisgl/kvcache/deepseek_v4_pool.py`
- `python/minisgl/attention/deepseek_v4.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`
- `benchmark/offline/deepseek_v4_text_smoke.py`

Use the promoted exact path:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--num-pages 128
cuda_graph_bs=[1,2,4,8,16]
```

## Required Tests

Prefer deterministic logits and metadata comparisons over generated-token
comparisons.

At minimum test:

- single-request full hit;
- single-request partial hit;
- batch full hit with identical prompts in multiple slots;
- mixed hit/miss batch;
- SWA boundary around `128`;
- page boundary around `256`;
- C4 boundary around multiples of `4`;
- C128 boundary around multiples of `128`.

For each case compare prefix-disabled, phase-1 prefix-enabled, and if useful
graph-disabled prefix-enabled mode.

## Required Instrumentation

Add small debug hooks or scripts as needed to capture:

- `cached_len`;
- suffix prefill token range;
- positions and sequence lengths;
- request table row and page table prefix entries;
- SWA page indices and top-k lengths;
- C4 sparse/full/page indices and lengths;
- C128 full/page indices and lengths;
- indexer selected rows and lengths;
- first suffix-prefill logits;
- first decode-step logits;
- sampled token and top-k logits, only as secondary evidence.

Instrumentation should be opt-in and should not change the default serving path.

## Analysis Rules

Use numerical tolerances appropriate for BF16 exact-path inference.  Do not
declare failure from generated token ids alone if logits are equal within a
reasonable tolerance and the difference can be explained by tie-breaking.

A useful report should identify the earliest mismatch:

```text
metadata -> suffix prefill logits -> decode logits -> sampled token
```

If metadata differs but logits still match, report it as a latent risk, not an
immediate correctness failure.

## Deliverables

Create:

```text
performance_milestones/target08_prefix_cache_logit_metadata_correctness/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- tested scenarios;
- metadata comparison table;
- logits comparison table;
- generated-token comparison as secondary context;
- root-cause conclusion or remaining blocker;
- recommendation for TARGET 08.20 and default promotion.

## Decision Rules

Proceed to TARGET 08.20 if:

- phase-1 prefix cache is logits-equivalent for required boundaries; or
- the mismatch is understood and guarded so component-retention work can avoid
  inheriting ambiguity.

Do not promote prefix cache to default if:

- first suffix/decode logits differ without an understood reason;
- SWA/C4/C128/indexer metadata disagree at a hit boundary;
- graph replay changes logits relative to eager for the same prefix path.

## Stop Rules

Stop and report blocked if:

- a deterministic logits comparison cannot be obtained without broad engine
  changes;
- instrumentation changes scheduler/cache behavior;
- the first root cause requires a large component-retention rewrite.

## Non-Goals

- Implementing SGLang-style component retention.
- Low-precision cache experiments.
- Attention-kernel optimization.
- CUDA graph private-pool attribution.
- PyNCCL or communication tuning.

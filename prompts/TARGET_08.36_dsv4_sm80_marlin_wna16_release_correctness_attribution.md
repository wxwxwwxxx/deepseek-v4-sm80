# TARGET 08.36: DSV4 SM80 Marlin WNA16 Release Correctness Attribution

## Status

Active TARGET 08 capacity/correctness follow-up after TARGET 08.35.

Run this target before attempting to promote any preset that releases original
routed FP4 expert weights.  TARGET 08.35 proved that prebuild plus release can
recover about `17.13 GiB/rank`, but it also found stable text sanity failures
after raw routed expert weights/scales were released.  This target exists to
attribute and fix, or explicitly reject, that release behavior.

## Goal

Determine why `dsv4_sm80_a100_victory_marlin_release` produces corrupted text
while both the non-release victory preset and the Marlin prebuild-only preset
produce sane text.

The desired output is a clear attribution:

- CUDA graph replay/capture issue;
- sampling or decode-loop state issue;
- MoE Marlin WNA16 packed-cache issue;
- raw expert tensor lifetime / storage alias issue;
- runtime branch change after `delattr`/release;
- or another owner with evidence.

After attribution, either:

- implement the smallest correctness fix and re-run text smoke gates; or
- keep release fail-closed/experimental and document why it is unsafe to
  promote.

## Background

TARGET 08.35 result:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/README.md
```

Relevant observations:

- `dsv4_sm80_a100_victory` text smoke passed.
- `dsv4_sm80_a100_victory_marlin_prebuild` text smoke passed.
- `dsv4_sm80_a100_victory_marlin_release` text smoke failed.
- release recovered `18,396,217,344` bytes = `17.1328 GiB/rank` according to
  model prepare ledger.
- release graph replay mechanically worked: captured buckets were
  `[1,2,4,8,16]`, `replay_count=63`, `eager_decode_count=0`.
- the failed outputs often start with plausible first words or tokens, then
  degenerate into repeated symbols or line-art-like text.
- `release_clone` and `release_sync` attempts failed similarly, so a simple
  missing-sync or simple shallow-clone explanation is not enough.

This means Marlin WNA16 prebuild itself is not proven wrong.  The correctness
blocker appears only after the original routed expert FP4 tensors/scales are
removed from the live Engine.

## Non-Goals

- Do not run 4096x128 or 4096x1024 macro throughput for a known-corrupt
  release runtime except as a final fixed-path confirmation.
- Do not introduce INT8/FP8 activation quantization.
- Do not change MoE math as a workaround unless a focused parity test proves
  that the current math path is the first divergent owner.
- Do not weaken text sanity gates to make release pass.
- Do not silently fall back to raw grouped/fallback paths after release.
- Do not spend time optimizing prefix-cache performance in this target.

## Required Investigation

### 1. Reproduce The Three-Way Smoke

Start from the exact TARGET 08.35 commands:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/COMMANDS.md
```

Re-run or reuse the existing artifacts for:

1. `dsv4_sm80_a100_victory`;
2. `dsv4_sm80_a100_victory_marlin_prebuild`;
3. `dsv4_sm80_a100_victory_marlin_release`.

Use page size `256`, TP8, fixed `--num-pages 128`, and the same prompts unless
you are deliberately creating a smaller diagnostic case.

Record whether the failure is deterministic across at least two release runs.

### 2. Split Graph From Release

Run the release variant with a small matrix:

- graph enabled, normal 08.35 settings;
- graph disabled;
- graph enabled but greedy-sample graph capture disabled, if supported;
- `--max-tokens 1`;
- `--max-tokens 2`;
- `--max-tokens 4`;
- `--max-tokens 16`.

Interpretation:

- If release is corrupt without graph, the primary issue is not graph replay.
- If first-token logits/text are already wrong, look at forward/MoE/layer
  parity.
- If first tokens are sane but later tokens diverge, inspect decode state,
  cache write/update, replay, and per-step logits.

### 3. Build A Logit Parity Ladder

Create a diagnostic mode or small script if needed.  Compare prebuild-only and
release under identical prompts, seed/sampling, page size, TP, and graph mode.

At minimum compare:

- prefill last logits;
- first decode-step logits;
- second decode-step logits;
- a later decode step near the first visible text degradation.

Record:

- top-k token ids and probabilities;
- max absolute logit difference;
- mean absolute logit difference;
- first generated token id;
- first step where prebuild-only and release diverge.

Generated-token equality is allowed here because this is a slot-matched,
same-shape diagnostic.  Do not use broad batch-slot invariance as the oracle.

### 4. MoE Packed Cache Micro-Parity

Isolate one or more routed MoE layers and compare the same
`hidden_states / weights / indices` through:

1. Marlin WNA16 prebuild-only with raw tensors still present;
2. the same Marlin WNA16 packed cache after release;
3. optionally a grouped/reference path before release as an oracle.

Record output finite ratio, max/mean absolute differences, shape/dtype, and a
small checksum/hash.  If this micro test diverges, focus on packed-cache
lifetime, cache tensor integrity, or custom-op assumptions.

If this micro test matches but full decode diverges, move attribution toward
non-MoE state, graph replay, sampling, attention/cache, or runtime branch
changes.

### 5. Release Lifetime A/B Matrix

Implement temporary diagnostic variants behind explicit debug env flags or
small local hooks.  Do not leave these as user-facing presets unless they
become the chosen fix.

Required A/B cases:

- **keep-hidden-ref**: remove raw expert attributes from the normal lookup path
  but keep the original tensors alive in a private hidden list.  This should
  not save memory; it tests whether physical tensor freeing is the trigger.
- **force-prepacked-with-raw-present**: keep raw attributes present but force
  the runtime to use the prepacked Marlin path.  This tests branch changes
  separately from memory release.
- **release-after-capture**: prebuild and capture while raw tensors are alive,
  then release before replay or smoke if the runtime can safely express this.
- **partial-layer release**: release only a prefix/suffix/subset of layers and
  binary-search the earliest layer or layer group that makes text/logits fail.
- **release-scales-only / release-weights-only**, if easy, to distinguish
  weight storage from scale storage.

The key question is whether corruption follows:

- missing attributes;
- physical storage freeing;
- graph capture timing;
- a specific layer;
- or a specific weight/scale component.

### 6. Cache Integrity And Allocator Checks

For representative ranks/layers, log Marlin WNA16 packed-cache tensor metadata
before and after release:

- `data_ptr`;
- shape;
- dtype;
- stride;
- `is_contiguous`;
- bytes;
- finite ratio;
- small checksum/hash;
- owner label / layer id.

Repeat after:

- `torch.cuda.synchronize()`;
- `torch.cuda.empty_cache()`, if used in the release path;
- graph capture;
- the first decode replay.

If cache tensors change unexpectedly, fix ownership/lifetime before rerunning
text smoke.

### 7. First Divergent Layer / Owner

If logit parity shows divergence but MoE micro tests do not immediately explain
it, add lightweight activation hashing at layer boundaries.  Prefer tiny
rank-local summaries over large tensor dumps:

- finite ratio;
- norm;
- checksum;
- max absolute value;
- optionally top few values for deterministic tiny slices.

Compare prebuild-only versus release and identify the first layer/owner where
the summaries diverge.

## Fix Policy

Acceptable fixes include:

- keeping specific raw tensors alive when proven necessary;
- changing release to a safer sentinel/owner model;
- moving release after a safe lifecycle boundary;
- changing Marlin WNA16 cache construction so packed tensors are fully
  independent of source tensors;
- making release unsupported for the current backend and documenting why.

Do not accept a fix that:

- hides text corruption;
- disables Marlin WNA16 silently;
- falls back to raw grouped/fallback after release;
- changes precision policy without a dedicated target;
- trades the entire recovered `17.13 GiB/rank` away without documenting the
  memory/capacity consequence.

## Validation Gates

Minimum pass criteria for a fixed release path:

1. three-way text smoke passes for baseline, prebuild-only, and release;
2. release eager/no-graph text smoke passes;
3. release graph text smoke passes for buckets `[1,2,4,8,16]`;
4. graph replay remains zero-eager for captured buckets;
5. Marlin WNA16 cache integrity logs are stable across release and graph
   capture/replay;
6. fail-closed backend-switch tests still pass;
7. memory ledger still reports recovered bytes or explicitly reports a smaller
   safe-released amount with equivalent KV pages/tokens.

If the target rejects release:

1. document the smallest evidence proving release is unsafe;
2. keep `dsv4_sm80_a100_victory_marlin_prebuild` as the safe capacity
   lifecycle improvement;
3. keep release preset experimental/no-go;
4. leave clear rollback and future-work notes.

Only after correctness passes should you run a short 4096x128 TP8 macro to
confirm performance neutrality.

## Deliverables

Write results under:

```text
performance_milestones/target08_marlin_wna16_release_correctness_attribution/
```

Include:

- `README.md` with attribution and go/no-go decision;
- command log;
- raw text smoke JSON/logs or symlinks;
- logit parity tables;
- MoE micro-parity results;
- release lifetime A/B table;
- cache integrity summaries;
- first divergent layer/owner if found;
- any code changes and tests;
- final recommendation for TARGET 08.35 release preset promotion status.

## Stop Conditions

Stop early and write the report if:

- release is corrupt in eager/no-graph and MoE micro-parity pinpoints the
  packed-cache/lifetime owner;
- keep-hidden-ref makes text pass while real release fails;
- force-prepacked-with-raw-present fails, proving a branch/path issue
  independent of physical release;
- a single layer or component is isolated and the fix would require a larger
  backend redesign;
- three consecutive reasonable A/B tests fail to narrow attribution, in which
  case document exactly what was ruled out and propose the next probe.

Do not keep running macro benchmarks or prefix workloads while text sanity is
known to fail.

## Suggested First Prompt

Use this target as the child-thread prompt.  Read `prompts/target.md`,
`prompts/TARGET_08_radix_prefix_dsv4.md`, this file, and the TARGET 08.35
report:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/README.md
```

The task is to attribute the release-preset text corruption before any further
promotion attempt.  Start by reproducing or reusing the 08.35 baseline,
prebuild-only, and release text smoke artifacts.  Then run the smallest
graph/eager and `--max-tokens` matrix needed to decide whether the failure is
inside graph replay, decode progression, or the model forward itself.  Next
build logit parity and MoE packed-cache lifetime A/B probes.  If a minimal fix
is obvious, implement it and rerun text smoke plus graph replay gates; otherwise
write a no-go report that clearly states what owner is responsible and why the
release preset must remain blocked.

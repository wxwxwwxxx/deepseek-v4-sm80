# TARGET 08.37: DSV4 SM80 Marlin WNA16 Release Storage-Reuse Owner

## Status

Active TARGET 08 release-correctness follow-up after TARGET 08.36.

Run this target only if we still want to salvage the Marlin WNA16 release
scheme.  TARGET 08.36 showed that releasing original routed FP4 expert
**weight** storage before KV/graph/warmup allocations causes text/logit
corruption, while keeping hidden references or delaying release until after
capture makes the text smoke pass.  This target should identify the concrete
owner that becomes unsafe when the freed expert-weight storage is returned to
the CUDA allocator and reused.

## Goal

Find the root cause of the release-preset correctness failure and turn it into
one of these concrete outcomes:

- a safe release implementation that preserves the intended pre-KV/pre-capture
  memory headroom;
- a narrower safe-release policy, with exact memory tradeoff and KV-token
  impact;
- or a documented no-go that proves full early release is unsafe without a
  larger allocator/lifecycle redesign.

The target is intentionally release-friendly: try hard to make release work.
But do not promote release until text sanity and logit/correctness gates pass.

## Background

Required prior reports:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/README.md
performance_milestones/target08_marlin_wna16_release_correctness_attribution/README.md
```

TARGET 08.36 key evidence:

- baseline and Marlin prebuild-only text smoke pass;
- release text smoke fails in graph and eager/no-graph modes;
- first visible logit divergence is around `decode_step_3`;
- activation trace first observes corruption around
  `layer2.indexer_select.logits`, then attention outputs and downstream layers;
- MoE packed-cache micro parity is exact for sampled layers before/after
  release, even with allocator pressure;
- Marlin packed-cache sampled tensors keep stable `data_ptr` and checksum;
- `force-prepacked-with-raw-present` passes;
- `keep-hidden-ref` passes, so deleting normal attributes is safe if source
  tensor storage remains alive;
- `release-after-capture` passes, so release is not intrinsically impossible;
- `weights-only` fails while `scales-only` passes;
- small partial releases pass, but larger releases fail, with the observed
  threshold between about `3.1875` and `6.3750 GiB/rank`.

The leading hypothesis is not "CUDA graph is broken" by itself.  Eager release
also fails.  A better hypothesis is:

```text
Early release returns large expert-weight storage blocks to the CUDA allocator.
Later KV/cache/warmup/graph/attention/indexer allocations reuse those addresses
and expose a stale pointer, aliasing bug, unsafe workspace lifetime, or
out-of-bounds write/read in the full-model decode path.
```

## Non-Goals

- Do not run large macro throughput while release text sanity is red.
- Do not weaken text sanity or logit checks.
- Do not introduce INT8/FP8 activation quantization.
- Do not silently fall back to raw grouped/fallback paths.
- Do not spend this target on prefix-cache performance.
- Do not conclude "graph bug" unless graph-only evidence remains after eager
  and allocation-order controls.

## Required Investigation

### 1. Build A Freed-Range Ledger

Before releasing routed expert weights, record the exact GPU address ranges for
every raw expert weight tensor that may be freed:

- layer id;
- component name, for example `w13_weight` or `w2_weight`;
- rank;
- `data_ptr`;
- byte size;
- `[start, end)` address range;
- dtype/shape/stride;
- whether the tensor is released in this run.

Write the ledger under:

```text
performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/raw/
```

The report must summarize total released bytes and group ranges by layer and
component.

### 2. Add Owner-Tagged Allocation Ledger After Release

Instrument the high-risk allocation/lifecycle owners after release and record
their tensor ranges:

- KV pools and DSV4 component pools;
- graph input/output/static buffers;
- attention/indexer metadata buffers;
- layer2 indexer and attention scratch/output buffers;
- logits/sampler buffers;
- PyTorch temporary tensors that are explicitly allocated by mini code;
- any Marlin WNA16 workspaces or route-plan buffers allocated after release.

For each owner, record:

- owner label;
- stage, for example `after_model_prepare`, `after_kv_alloc`,
  `after_warmup_forward`, `after_graph_capture`, `decode_step_1`;
- `data_ptr`, bytes, dtype, shape;
- whether it overlaps any freed expert-weight range;
- nearest freed range if exact overlap is not observed.

If direct PyTorch allocator introspection is too noisy, start with the tensors
mini owns and add targeted hooks around layer2/indexer/attention.

### 3. Release Timing Ladder

Run the same short text/logit smoke while moving the release point:

1. release immediately after full Marlin prebuild, before KV planning;
2. release after KV allocation but before warmup forward;
3. release after warmup forward but before graph capture;
4. release after graph capture;
5. release after the first successful decode step, if expressible.

For each run, report:

- text sanity;
- first divergent token/logit step;
- released bytes;
- graph replay/eager counts;
- which allocation owners reused freed expert-weight ranges before divergence.

This should identify the first lifecycle boundary that makes release safe.

### 4. Poison And Quarantine Tests

Use explicit debug env flags or scripts; do not make these normal runtime
features.

Required probes:

- **hidden-ref poison**: keep hidden refs alive, but overwrite the raw weight
  storage with a pattern such as zeros, NaNs, or a deterministic byte-like
  value after Marlin cache prebuild.
  - If text/logits fail, some path still reads original raw weight contents.
  - If text/logits pass, stale reads of raw weight contents are less likely.
- **freed-block quarantine**: after release, immediately allocate dummy tensors
  roughly matching the released blocks so they occupy the freed address ranges.
  Fill them with several patterns.
  - If quarantine changes pass/fail behavior, the failure strongly follows
    allocator reuse of the released ranges.
  - If quarantine makes release pass, consider a safe allocator quarantine or
    workspace-reservation design, but quantify how much memory headroom it
    gives back versus keeping raw weights.
- **pressure sweep**: vary dummy allocation sizes around the known threshold
  between `3.1875` and `6.3750 GiB/rank` to find a smaller reproducible trigger.

### 5. Layer2 Indexer/Attention Owner Probe

Because TARGET 08.36 first observed activation divergence at
`layer2.indexer_select.logits`, add focused diagnostics for layer2 decode:

- input hidden state checksum/finite stats;
- q/kv/indexer projection outputs;
- indexer query/value/weight tensors;
- indexer logits/topk scores;
- sparse attention selected locations and metadata;
- attention output before/after `wo`;
- all relevant tensor address ranges and overlap checks against freed ranges.

The goal is to decide whether layer2 indexer is:

- the root owner allocating/using a corrupted buffer;
- the first consumer of an earlier corrupted state;
- or just the first visible nonfinite-sensitive site.

### 6. Minimal Reproducer

If an owner is found, make the smallest reproducer possible:

- one or a few layers if feasible;
- same release/quarantine behavior;
- no broad serving workload;
- deterministic prompt and token step;
- clear pass/fail condition based on logits or finite checks.

This reproducer is more valuable than a large benchmark while correctness is
red.

## Candidate Fix Directions

Prefer fixes that keep most of the intended memory recovery:

- make Marlin packed weights fully source-independent if a stale source
  dependency is proven;
- delay only the unsafe subset of release until after the necessary lifecycle
  boundary;
- split release into safe and unsafe components, with a memory ledger for each;
- reserve or preallocate the owner buffer that currently reuses dangerous
  ranges, then release raw weights after that owner is initialized;
- add an explicit allocator/workspace owner so the address reuse order is
  deterministic and safe;
- if the issue is an out-of-bounds/alias bug in indexer/attention, fix that
  owner directly and rerun release text smoke.

Avoid fixes that simply keep all raw weights alive while claiming release
success.  A keep-hidden-ref variant is an oracle, not a final release solution.

## Validation Gates

For any proposed fix:

1. baseline text smoke still passes;
2. prebuild-only text smoke still passes;
3. release text smoke passes in eager/no-graph;
4. release text smoke passes with graph buckets `[1,2,4,8,16]`;
5. graph replay stays zero-eager for captured buckets;
6. logit parity no longer collapses at `decode_step_3`;
7. layer2/indexer finite checks stay clean;
8. fail-closed backend-switch tests still pass;
9. memory ledger reports actual released bytes and equivalent KV pages/tokens.

Only after these pass, run one short 4096x128 TP8 macro as a sanity check.  Do
not use throughput to justify a path that still has text/logit corruption.

## Deliverables

Write results under:

```text
performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/
```

Include:

- `README.md` with attribution, fix/no-go decision, and next step;
- command log;
- freed expert-weight range ledger;
- owner-tagged allocation/reuse ledger;
- release timing ladder table;
- poison/quarantine table;
- layer2 indexer/attention owner probe summary;
- minimal reproducer description if found;
- any code changes and tests;
- final recommendation for whether release can continue toward promotion.

## Stop Conditions

Stop and write the report if:

- an exact owner is found and a small fix makes text/logit gates pass;
- an exact owner is found but fixing it requires a larger redesign;
- poison proves raw weight contents are still read after release;
- quarantine proves allocator reuse is the trigger and no small safe release
  policy preserves meaningful headroom;
- three focused owner probes fail to narrow beyond TARGET 08.36, in which case
  document what was ruled out and propose the next concrete instrumentation.

Do not keep repeating broad smoke/macro runs without adding owner-level
evidence.

## Suggested First Prompt

Use this target as the child-thread prompt.  Read `prompts/target.md`,
`prompts/TARGET_08_radix_prefix_dsv4.md`, this file, and:

```text
performance_milestones/target08_marlin_wna16_release_correctness_attribution/README.md
performance_milestones/target08_marlin_wna16_release_preset_promotion/README.md
```

The goal is to continue the release route, not abandon it.  TARGET 08.36 showed
that keep-hidden-ref and release-after-capture pass, while early physical
release of large expert weight storage fails.  Start by building a freed-range
ledger for released raw expert weights and an owner-tagged allocation ledger for
post-release KV/cache/warmup/graph/attention/indexer allocations.  Then run a
release timing ladder, hidden-ref poison tests, freed-block quarantine tests,
and a layer2 indexer/attention owner probe.  If the concrete owner is found,
implement the smallest safe fix and rerun release text/logit/graph gates.  If
not, write a clear no-go or next-instrumentation report without running large
macros for a known-corrupt release path.

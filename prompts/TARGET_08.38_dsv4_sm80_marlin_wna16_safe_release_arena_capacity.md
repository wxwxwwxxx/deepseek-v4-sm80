# TARGET 08.38: DSV4 SM80 Marlin WNA16 Safe Release Arena Capacity

## Status

Active TARGET 08 release-route repair target after TARGET 08.37.

TARGET 08.37 found the concrete unsafe storage-reuse owner for immediate
`model_prepare` release: after early physical release of raw routed expert
weights, DSV4 KV/component pools reuse the freed expert-weight ranges and the
decode path later collapses.  This target should repair the release route
instead of abandoning it.

## Goal

Implement and validate a correctness-clean Marlin WNA16 release policy that
preserves as much of the intended KV capacity gain as possible.

The preferred long-term direction is:

```text
Plan KV capacity with a Marlin WNA16 release credit, find and fix the owner
that makes released expert-weight ranges unsafe, and ultimately allow live
KV/component tensors to use those released ranges safely.
```

The target should turn TARGET 08.37's attribution into an actual safe lifecycle
policy.  If full pre-KV capacity recovery is not safely achievable in this
target, produce a narrower safe policy with an explicit memory/KV-token ledger
and a clear next step.

Important: a guard/quarantine/arena that permanently keeps KV/component tensors
away from all released expert-weight ranges is not the desired final win.  It
is a diagnostic tool or fallback.  The main goal is to make those `~17.13
GiB/rank` usable by KV/component state, because otherwise release does not
deliver the intended capacity gain.

## Source Analysis Starting Point

Current mini source shape:

- `python/minisgl/engine/engine.py`
  - `Engine.__init__` currently does:
    1. create/load model;
    2. `model.prepare_for_cuda_graph_capture()`;
    3. `_determine_num_pages(init_free_memory, config)`;
    4. `create_kvcache_pool(...)`;
    5. optional delayed `after_kv_alloc` release debug hook;
    6. page table, attention backend, graph runner.
  - `_determine_num_pages` uses actual free memory after model prepare.  It
    does not add deferred release bytes as a capacity credit.
- `python/minisgl/models/deepseek_v4.py`
  - `DeepseekV4Model.prepare_for_cuda_graph_capture()` prebuilds Marlin WNA16
    caches when `MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1`.
  - immediate release happens inside model prepare when
    `MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1` and the
    debug release timing is `model_prepare`.
  - delayed release timings from TARGET 08.37 exist as diagnostics:
    `after_kv_alloc`, `before_warmup_forward`, `after_warmup_forward`,
    `after_graph_capture`, `after_first_decode`.
- `python/minisgl/kvcache/deepseek_v4_pool.py`
  - `DeepSeekV4KVCache.__init__` allocates all DSV4 buffers directly with
    `torch.empty`, including `swa_buffer`, `c4_buffer`, `c128_buffer`,
    `c4_indexer_buffer`, optional `c4_indexer_fp8_paged_cache`,
    refcount/mapping tensors, and per-layer `compress_state` /
    `indexer_state` buffers.

Consequences:

- Immediate `model_prepare` release gives the desired page-planning headroom
  but is unsafe because KV/component buffers reuse freed expert-weight ranges.
- `after_kv_alloc` release is correctness-clean in TARGET 08.37, but it is
  capacity-neutral because pages were already planned and allocated before
  release.
- A repair must separate **capacity accounting** from **unsafe address reuse**.

## Background

Required reports:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/README.md
performance_milestones/target08_marlin_wna16_release_correctness_attribution/README.md
performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/README.md
```

TARGET 08.37 decisive evidence:

- immediate release fails in eager/no-graph;
- release after KV allocation passes in eager and graph modes;
- all delayed release timings after KV allocation pass;
- raw weight contents are not read after prebuild: hidden-ref poison with zero
  and NaN passes;
- freed-block quarantine passes, including a `3.1875 GiB/rank` quarantine;
- immediate release owner ledger shows `after_kv_alloc` overlaps in
  `kvcache.dsv4.c4_buffer`, `c4_indexer_buffer`,
  `c4_indexer_fp8_paged_cache`, `c128_buffer`, and per-layer
  `compress_state` / `indexer_state` buffers;
- after-KV release removes those KV/component overlaps.

The likely failure model is:

```text
Early release makes raw expert weight addresses reusable.
Some later stale pointer, alias, or lifecycle assumption can still touch those
old addresses.
If live KV/component tensors occupy those addresses, decode state is corrupted.
If the old addresses are held by hidden refs or quarantine tensors, text passes.
```

Prior diagnostics already tried two relevant poison/quarantine families:

- hidden-ref poison with zero/NaN passed, so the current smoke does not require
  reading original raw expert weight contents after Marlin prebuild;
- freed-block quarantine with zero/deterministic dummy tensors passed, so
  keeping live KV/component state out of those ranges avoids the failure.

What has not yet been proven is which function/kernel touches the old address
ranges, or whether the issue is a stream/lifetime bug, stale pointer, aliasing,
or an out-of-bounds write.  This target should add sentinel integrity checks
designed to catch that owner directly.

## Non-Goals

- Do not use immediate `model_prepare` release as-is.
- Do not promote a path that fails text/logit sanity.
- Do not silently fall back to grouped/raw expert paths.
- Do not introduce INT8/FP8 activation quantization.
- Do not spend this target on generic prefix-cache optimization.
- Do not treat a dummy quarantine tensor as acceptable unless the final policy
  documents its memory cost, owner, lifetime, and net KV-capacity benefit.
- Do not declare victory merely because guard tensors avoid the bad ranges.
  The main success path is making KV/component tensors safely use the released
  ranges, or proving exactly why that requires a larger redesign.

## Required Design

### 1. Define A Safe Release Policy

Introduce a named policy or preset candidate separate from the unsafe immediate
release preset.  Suggested names:

```text
dsv4_sm80_a100_victory_marlin_release_safe_arena
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release_safe_arena
```

The exact names may follow local style, but they must not reuse the old
immediate-release semantics silently.

At minimum the policy must make these states explicit in reports:

- Marlin WNA16 prebuild bytes;
- raw expert source bytes;
- source bytes intended for release;
- source bytes actually released;
- guard/arena/quarantine bytes, if any;
- net capacity credit bytes;
- equivalent KV pages/tokens;
- release timing;
- whether KV/component tensors are allowed to overlap released ranges.

### 2. Capacity Credit

Add an auditable capacity-planning credit for releasable Marlin WNA16 raw expert
weights.

Expected rough value from TARGET 08.37:

```text
raw expert source bytes/rank = 18,396,217,344 = 17.1328125 GiB
theoretical page-size-256 headroom ~= 400.10 pages ~= 102,426 tokens/rank
```

The planner should not blindly add this full number if a guard/arena keeps part
of the released memory unavailable.  Instead report:

```text
net_release_credit = released_source_bytes - guard_or_reserved_bytes - safety_margin
```

Use this credit only for DSV4/Marlin WNA16 release-safe policies.  Non-release
and prebuild-only paths must keep their existing planning behavior.

### 3. Safe Address Strategy

Evaluate and implement the smallest strategy that keeps correctness while
recovering meaningful capacity.

Candidate strategies, in preferred order:

1. **Direct root-cause fix, then KV/component reuse of released ranges**
   - prebuild Marlin WNA16 caches;
   - release raw expert weights early enough to count them in capacity planning;
   - identify and fix the stale pointer / stream lifetime / alias /
     out-of-bounds owner that corrupts KV/component when it reuses those ranges;
   - allocate the DSV4 KV/component pool using the release credit;
   - prove live KV/component tensors may overlap formerly raw expert ranges
     without text/logit corruption.
2. **Release + unsafe-range sentinel guard as a diagnostic**
   - prebuild Marlin WNA16 caches;
   - register raw expert freed ranges;
   - physically release raw expert weights;
   - immediately allocate sentinel guard tensors that capture the unsafe subset
     of released ranges;
   - fill guards with NaN for floating tensors or deterministic byte/integer
     patterns for non-floating storage;
   - verify guard checksums and finite/sample state after KV allocation, warmup
     forward, graph capture, and each early decode step;
   - if a guard mutates unexpectedly, use the stage delta to identify the
     stale writer/reader owner;
   - allocate the DSV4 KV/component pool using the net release credit;
   - keep the guard alive at least until the proven safe lifecycle boundary.
   - TARGET 08.37 showed a `3.1875 GiB/rank` quarantine can protect the short
     smoke, but this target must validate the guard size/lifetime more
     rigorously.
3. **KV/component arena preallocation plus delayed release**
   - allocate the critical KV/component owners before physical release;
   - release raw expert weights after those owners have stable addresses;
   - then optionally allocate additional extension pages from safe ranges.
   - This is correctness-clean but may need an extendable KV pool or a second
     arena to recover capacity.
4. **Independent DSV4 KV/component arena**
   - route DSV4 KV/component allocations through a clearly owned arena or
     allocation sequence that avoids known unsafe ranges;
   - use owner ledger to prove live KV/component buffers do not overlap
     unprotected released ranges.

If none of these can recover meaningful capacity safely, promote the
`after_kv_alloc` release timing only as a runtime-headroom policy and document
that it does not solve KV capacity planning.

The final preferred policy should not require permanently hiding all released
expert-weight ranges from KV/component.  If a guard is retained in the final
policy, report exactly how many bytes it costs and why the remaining net
release credit is still worthwhile.

### 3.5 Old-Address Access Trap

Add a direct trap for accesses to old expert-weight addresses.  This can reuse
the freed-range ledger from TARGET 08.37 but should add mutation checks.

Required probes:

- **sentinel guard mutation check**:
  - after early release, allocate guard tensors intended to occupy the released
    expert-weight ranges;
  - fill them with NaN where dtype permits and deterministic patterns
    otherwise;
  - snapshot checksum/sample/finite state;
  - after each stage, verify whether the guard changed.
- **stage bisection**:
  - check guards after KV allocation, page-table allocation, attention backend
    creation, graph runner init, warmup forward, graph capture, decode step 1,
    decode step 2, and the known failing decode step 3.
- **split-model / split-layer repro**:
  - if guard mutation or logit collapse appears only after certain layers,
    run a partial-layer or layer-window model/debug path when feasible;
  - use it to reduce the failing owner from "full model" to a smaller layer or
    kernel boundary.
- **KV-as-sentinel check**:
  - for KV/component buffers that overlap old expert ranges, initialize
    selected unused or freshly allocated slices with sentinels before the first
    legitimate store;
  - verify whether they mutate before the expected attention/cache write.

Interpretation:

- If hidden-ref poison passes but sentinel guard mutates, the issue is a stale
  writer or out-of-bounds write, not a raw-weight read.
- If sentinel guard does not mutate but KV-as-sentinel does, the bug may be in
  the KV/component owner itself or a write path that only targets live cache
  tensors.
- If neither mutates but logits collapse only when KV overlaps old ranges,
  investigate CUDA stream/lifetime semantics, allocator events, and custom-op
  `record_stream`/synchronization boundaries.

### 4. Owner Ledger Must Prove The Fix

For the chosen policy, run owner/freed ledgers comparable to TARGET 08.37.

Required proof:

- released raw expert ranges are recorded;
- guard/arena ranges are recorded;
- KV/component owners are recorded;
- for the final preferred path, live KV/component owners may overlap formerly
  raw expert ranges only after the root unsafe access has been fixed;
- if overlap is avoided by guard/arena, the report must label this as a
  fallback/narrowed policy rather than the full capacity win;
- transient forward/logits overlaps are understood and appear in passing runs.

### 5. Guard Size And Lifetime Sweep

If using a guard/quarantine/arena strategy, sweep:

- guard size:
  - `0 GiB`;
  - `1 GiB`;
  - `3.1875 GiB`;
  - `6.375 GiB`;
  - full released bytes, as an oracle only;
- guard lifetime:
  - until after KV allocation;
  - until after warmup forward;
  - until after graph capture;
  - until after first decode;
  - full Engine lifetime.

Use text/logit gates and owner ledgers to determine the minimum safe policy.

Do not rely on one short smoke alone for the final guard size.  At least one
longer decode smoke or 4096x128 macro should run after correctness gates pass.

### 6. Capacity And Memory Ledger

Report at least:

- pages planned without release;
- pages planned with unsafe immediate release, if measured only as a no-go
  reference;
- pages planned with `after_kv_alloc` release;
- pages planned with the new safe arena policy;
- whether KV/component actually uses released expert-weight ranges in the final
  policy;
- actual free memory after init;
- graph capture memory deltas;
- released source bytes;
- guard/arena bytes;
- net extra KV pages/tokens;
- any loss versus the theoretical `~400` pages.

The final decision should make the tradeoff obvious, for example:

```text
full theoretical release: +400 pages
safe arena guard: -N pages
net gain: +M pages
```

## Required Validation

Correctness gates:

- baseline text smoke;
- prebuild-only text smoke;
- `after_kv_alloc` release text smoke as a safe lower-bound control;
- safe-arena release text smoke with graph disabled;
- safe-arena release text smoke with graph buckets `[1,2,4,8,16]`;
- logit parity or targeted decode-step check showing no `decode_step_3`
  collapse;
- layer2/indexer finite/boundedness check;
- old-address sentinel guard or KV-as-sentinel integrity checks;
- fail-closed backend-switch tests.

Capacity/performance gates after correctness passes:

- auto-page run without fixed `--num-pages`, or an explicit capacity probe that
  demonstrates additional pages;
- fixed `--num-pages 128` continuity smoke;
- larger `--num-pages` stress run near the newly claimed capacity;
- 4096x128 TP8 macro sanity;
- 4096x1024 TP8 macro if time permits.

Graph gates:

- captured buckets `[1,2,4,8,16]`;
- zero eager fallback for captured decode buckets;
- graph capture memory ledger with release/guard effects.

## Deliverables

Write results under:

```text
performance_milestones/target08_marlin_wna16_safe_release_arena_capacity/
```

Include:

- `README.md` with design, source analysis, and final go/no-go;
- `COMMANDS.md`;
- memory/capacity ledger;
- owner/freed/guard overlap ledger;
- old-address sentinel mutation report;
- split-model/layer-window repro report if used;
- guard size/lifetime sweep;
- text/logit correctness results;
- graph replay results;
- macro/capacity results after correctness passes;
- code changes and tests;
- final recommendation:
  - promote safe arena release preset;
  - promote only `after_kv_alloc` release as a narrowed headroom policy;
  - or keep release blocked pending a larger allocator redesign.

## Stop Conditions

Stop early and report if:

- old-address sentinel checks identify a concrete writer/reader owner and the
  fix is outside this target's reasonable scope;
- no guard/arena policy can pass text/logit gates without keeping most of the
  release bytes unavailable;
- owner ledger shows live KV/component tensors still corrupt when they overlap
  formerly raw expert ranges after the proposed root-cause fix;
- the only passing policy is full hidden-ref or full quarantine, which provides
  no meaningful capacity gain;
- capacity accounting cannot be made auditable;
- three focused attempts fail to beat the `after_kv_alloc` lower-bound policy.

Do not continue to macro benchmarking while text/logit sanity is red.

## Suggested First Prompt

Use this target as the child-thread prompt.  Read `prompts/target.md`,
`prompts/TARGET_08_radix_prefix_dsv4.md`, this file, and the TARGET 08.35-08.37
reports:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/README.md
performance_milestones/target08_marlin_wna16_release_correctness_attribution/README.md
performance_milestones/target08_marlin_wna16_release_storage_reuse_owner/README.md
```

The user wants to keep pursuing the release route.  Start from the current
source order in `python/minisgl/engine/engine.py`,
`python/minisgl/models/deepseek_v4.py`, and
`python/minisgl/kvcache/deepseek_v4_pool.py`.  Design and implement a safe
release lifecycle that accounts for releasable Marlin WNA16 raw expert bytes in
capacity planning.  The preferred goal is not to hide the released ranges from
KV/component forever; it is to find and fix why those ranges are unsafe, then
allow KV/component to use the released `~17.13 GiB/rank` safely.  Use
old-address sentinel guards, NaN/deterministic poisoning, split-layer probes,
and owner/freed ledgers to catch stale access or unsafe mutation.  Validate
text/logit sanity before any macro.  If full capacity recovery is impossible in
this target, produce the best narrowed safe policy with an explicit
memory/KV-page tradeoff.

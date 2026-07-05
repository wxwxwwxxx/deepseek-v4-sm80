# TARGET 08.35: DSV4 SM80 Marlin WNA16 Release Preset Promotion

## Status

Active TARGET 08 capacity follow-up after TARGET 08.34.

Run this after the MoE Marlin WNA16 cache lifecycle attribution is complete.
TARGET 08.34 proved the `~17-18 GiB/rank` warmup-forward memory jump is lazy
Marlin WNA16 expert-weight repack, and that prebuild plus original-weight
release can pass a short smoke.  This target turns that result into a named,
audited high-memory-efficiency preset rather than a loose experimental opt-in.

## Goal

Promote a DSV4 A100/sm80 preset that:

- prebuilds MoE Marlin WNA16 routed expert caches before KV capacity planning;
- releases original routed FP4 expert weights/scales after successful repack;
- reports the saved memory and KV capacity impact clearly;
- remains exact for the selected `marlin_wna16` backend;
- fails closed if any later code path tries to use raw expert weights after
  release;
- passes correctness, graph replay, prefix-cache, and macro performance gates.

The target should not expose release as the main user story of "set this one
experimental env var."  It may keep env flags internally, but the deliverable
should include a named benchmark/runtime preset that owns the full policy.

## Background

TARGET 08.34 established:

```text
current warmup model.forward allocated delta: 17.8126 GiB/rank
prebuild warmup model.forward allocated delta: 0.6798 GiB/rank
difference: 17.1328 GiB/rank
43-layer Marlin WNA16 cache theory: 17.1328 GiB/rank
```

It also showed that prebuild + original-weight release can pass a short
`decode_len=2` smoke with graph replay:

```text
replay_count=1
eager_decode_count=0
released bytes/rank = 17.1328 GiB
```

The important distinction:

- **prebuild** fixes lifecycle and KV capacity accounting;
- **release** is the high-memory-efficiency preset behavior that recovers about
  `400` DSV4 KV pages or `102k` KV tokens per rank at page size `256`.

Because release intentionally removes raw FP4 expert tensors from the live
Engine, it needs a proper preset and promotion gate.

## Non-Goals

- Do not change MoE math or introduce INT8/FP8 activation quantization.
- Do not support switching MoE backend inside the same Engine after release.
- Do not silently fall back to grouped FP4 if raw expert weights were released.
- Do not promote prefix cache to default for no-hit traffic.
- Do not reopen CUDA graph private-pool attribution unless release/prebuild
  regresses graph replay or memory accounting.
- Do not spend time on the remaining `~0.68-0.72 GiB/rank` warmup residual
  unless it appears as a correctness or capacity blocker.

## Required Preset Shape

Define a named preset in the benchmark/runtime preset system.  Suggested names:

```text
dsv4_sm80_a100_victory_marlin_release
dsv4_sm80_a100_victory_prefix_routeb_lifetime_marlin_release
```

The exact names can be adjusted to match local naming style, but they must make
the memory policy obvious.

The preset should imply:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
MINISGL_DSV4_MARLIN_WNA16_PREBUILD=1
MINISGL_DSV4_MARLIN_WNA16_RELEASE_ORIGINAL_EXPERT_WEIGHTS=1
```

For the prefix variant, also include the promoted Route B prefix settings:

```text
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
```

It is acceptable for the preset to use internal env flags.  The external
contract should be the named preset.

## Required Approach

### 1. Audit Existing 08.34 Implementation

Before adding more behavior, review the code changes from TARGET 08.34:

- `python/minisgl/models/deepseek_v4.py`;
- `python/minisgl/kernel/deepseek_v4.py`;
- `python/minisgl/kernel/marlin_wna16.py`;
- `python/minisgl/utils/dsv4_memory_debug.py`;
- `python/minisgl/engine/graph.py`;
- `benchmark/offline/deepseek_v4_perf_matrix.py`.

Verify:

- prebuild runs before `_determine_num_pages()`;
- release happens only after all 43 layer caches are built successfully;
- raw tensors are not deleted after a partial failed prebuild;
- repeated prepare calls are idempotent;
- cache signatures still work after source tensors are released;
- error messages are explicit if a raw-weight path is attempted after release;
- model reports list persistent bytes, source bytes, released bytes, and
  equivalent KV pages/tokens.

### 2. Make Release A Preset, Not A Loose Opt-In

Add benchmark variants and, if applicable, serving preset documentation for the
release policy.

The preset should:

- force `marlin_wna16`;
- enable prebuild;
- enable original-weight release;
- include clear rollback instructions to use the non-release victory preset;
- surface memory accounting in reports.

Do not require users to remember separate release env flags for normal
benchmarking of this route.

### 3. Fail-Closed Semantics

After release, raw routed expert FP4 tensors are no longer valid runtime inputs.
The implementation must make this state explicit.

Required checks:

- `grouped_fp4`/fallback paths raise a clear error if selected after release;
- lazy repack does not run after release;
- `marlin_wna16` forward uses the prebuilt cache directly;
- the state after release is visible in `model_prepare_report`;
- `state_dict()` / debug dumps either skip released tensors or expose clear
  sentinel metadata, without crashing normal serving;
- a fresh Engine can still load from checkpoint normally.

Do not attempt to support backend switching within the same Engine.

### 4. Correctness Gates

Run at least:

- DSV4 text smoke, page size `256`;
- prefix Route B text smoke with the prefix release preset;
- short decode with graph replay for buckets `[1,2,4,8,16]`;
- a longer `decode_len=128` smoke/macro for the non-prefix release preset;
- one prefix shared-workload sanity run if available.

Correctness expectations:

- no乱码 / invalid text;
- no crash;
- graph replay stays zero-eager for captured buckets;
- prefix cache metrics remain sane;
- no raw-weight-path fallback is taken after release.

Use existing project smoke/verifier behavior.  Do not use broad generated-token
equality as an oracle because batch-slot invariance is not guaranteed.

### 5. Performance And Capacity Gates

Compare at least three variants:

1. current non-release victory preset;
2. prebuild without release, if still easy to run;
3. release preset.

Required measurements:

- 4096x128 bs4 TP8 macro;
- 4096x1024 bs4 TP8 macro, if time permits;
- prefix no-hit and shared-prefix sanity if the prefix release preset is added;
- initialization / model prepare time;
- free memory after init;
- planned `num_pages` when `--num-pages` is not fixed;
- fixed `--num-pages 128` behavior for continuity with prior targets;
- graph capture deltas and graph replay counts.

Promotion thresholds:

- release preset must recover about `17.13 GiB/rank` versus prebuild without
  release, or explain any discrepancy;
- throughput must be neutral within noise or better versus non-release
  `marlin_wna16` after warmup;
- graph replay must remain zero-eager;
- no new OOM or capacity regression.

If fixed `--num-pages 128` hides capacity gains, run one auto-page or larger
`--num-pages` capacity probe to show the recovered headroom.

### 6. SGLang/vLLM Lifecycle Alignment

Keep the lifecycle aligned with upstream practice:

- heavy Marlin packing belongs to load/post-load/model-prepare, not first
  serving forward;
- backend-specific packed weights should be the runtime source of truth for
  that backend;
- releasing source weights is acceptable only when the backend contract is
  explicit and fail-closed.

Do a short source review of SGLang/vLLM post-load quantized weight packing if
08.34 did not already cover enough detail.  Avoid porting code unless a small
piece directly improves this preset.

## Required Analysis

The final README must include:

- recap of TARGET 08.34's attribution result;
- exact preset names and env expansion;
- memory ledger:
  - prebuild persistent bytes;
  - released source bytes;
  - net bytes saved;
  - equivalent KV pages/tokens;
- initialization and KV planning impact;
- correctness/smoke results;
- graph capture/replay results;
- macro performance table versus non-release baseline;
- fail-closed behavior tests;
- remaining risks;
- promotion recommendation:
  - promote release preset;
  - keep release preset experimental with named variant;
  - or reject release as preset and keep only prebuild.

## Gates

Pass this target if:

1. the release preset passes correctness, graph replay, and macro gates;
2. memory recovery is within `0.5 GiB/rank` of the expected `17.1328 GiB/rank`;
3. fallback/backend-switch attempts after release fail clearly rather than
   silently using invalid raw tensors;
4. reports expose enough memory accounting to make KV capacity planning
   auditable.

Stop early if:

- release causes any text smoke corruption, graph replay failure, or crash;
- raw-weight release breaks fresh Engine load or ordinary non-release presets;
- macro throughput regresses by more than measurement noise and the regression
  is not understood;
- state_dict/debug/reporting crashes in normal serving paths;
- release can silently fall back to a raw-weight path.

## Deliverables

Write results under:

```text
performance_milestones/target08_marlin_wna16_release_preset_promotion/
```

Include:

- `README.md`;
- command log;
- preset env expansion;
- summary tables;
- raw logs or symlinks;
- code changes for preset promotion and safety checks;
- final go/no-go recommendation.

## Suggested First Prompt

Use this target as the child-thread prompt.  Read `prompts/target.md`,
`prompts/TARGET_08_radix_prefix_dsv4.md`, this file, and the TARGET 08.34
report.  Start by auditing the existing prebuild/release code and defining a
named release preset.  Treat release as a preset policy, not as a loose manual
opt-in.  Then run correctness, graph replay, capacity, and macro gates before
recommending promotion.

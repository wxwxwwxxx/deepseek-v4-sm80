# TARGET misc 01.5: DSV4 Census Manifest Hardening

## Status

Planned correction gate between TARGET misc 01 and TARGET misc 02.

TARGET misc 01 proved that both DSV4 product modes are constructible and
text-sane, but its generated manifests are not yet safe deletion authority.
This target fixes the classification method and produces the final runtime
coverage manifest consumed by the destructive cleanup.

This target does not broadly delete production code.

## Goal

Produce a deletion-grade manifest with this rule:

```text
KEEP a kernel/backend only when the required release or oracle coverage matrix
actually executes it, or when a proven runtime wrapper requires it for one of
those covered semantic surfaces.

DELETE a kernel/backend when it is absent from both product modes, absent from
the required coverage matrix, and has no retained wrapper/native dependency.

REVIEW only when concrete evidence prevents a decision.  Generic statements
such as "dynamic dispatch cannot be disproved" are not sufficient.
```

The final repository may retain only:

1. kernels required by the optimized `0.1.0+dsv4.sm80` release runtime;
2. kernels required by the explicit fallback/oracle runtime;
3. shared headers/runtime infrastructure transitively required to build or
   launch those kernels.

## Required Inputs

Read first:

```text
prompts/TARGET_misc_dsv4_release_cleanup.md
prompts/TARGET_misc_01_dsv4_two_path_census.md
prompts/TARGET_misc_02_dsv4_two_path_runtime_cleanup.md
performance_milestones/misc_release_two_path_census/README.md
performance_milestones/misc_release_two_path_census/env_toggle_manifest.json
performance_milestones/misc_release_two_path_census/two_path_manifest.json
performance_milestones/misc_release_two_path_census/model_dependency_manifest.json
performance_milestones/misc_release_two_path_census/benchmark_manifest.json
debug/release_cleanup/build_manifests.py
debug/release_cleanup/census_runtime.py
```

Record and verify:

```text
cleanup base tag: pre-cleanup-snapshot
cleanup base commit: 106a3abe205b259020f5d73d9b8d138e31764eb9
historical perf tag: v0.0.0, unchanged
package version: 0.1.0+dsv4.sm80
production source changes since census: none, unless explicitly explained
```

Do not create `v0.1.0-dsv4-sm80` in this target.

## Known Manifest Defects To Fix

### 1. String-Valued Release Defaults

The original active-toggle collector missed non-boolean values.  At minimum,
these optimized behaviors were incorrectly classified as `RESEARCH_DEAD`:

```text
MINISGL_DSV4_SM80_MOE_EXPERT_BACKEND=marlin_wna16
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=swa,c4
```

The first selects the promoted expert backend.  The second deliberately limits
direct graph metadata to the validated SWA/C4 groups; removing it without an
equivalent typed constant may unintentionally enable C128 behavior.

Also account for string/integer production values such as:

```text
MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc
MINISGL_DSV4_CLEAR_ALLOCATED_KV_ON_PAGE_ALLOC=component
MINISGL_PYNCCL_MAX_BUFFER_SIZE=32M
```

The historical env spelling may later be deleted, but the validated value and
behavior must be retained in optimized typed config/internal constants.

### 2. Hard-Coded Unknown Counts

Remove hard-coded:

```text
unknown_review_count = 0
unknown_review_hot_path_count = 0
```

Compute these from actual entries.  `REVIEW` is an unresolved state and must be
counted honestly.

### 3. Generic Callable REVIEW

The original AST census assigned 95 callables the same generic REVIEW reason.
This mixed:

- private Triton kernels launched through `kernel[grid](...)`;
- JIT/native symbols reached through loaders;
- properties and dynamically called methods;
- live optimized kernels;
- dead FP4/FP8/INT8 research kernels.

Resolve them at wrapper/private-kernel family granularity.  A private kernel
inherits the runtime reachability of a covered wrapper only when its launch
condition is also satisfied by a captured runtime case.

### 4. Dependency And Test Errors

Correct at least:

- `flashinfer-python` is `KEEP` because retained sampling directly imports
  `flashinfer.sampling`, regardless of deleting generic FlashInfer attention;
- `tests/utils/test_dsv4_long_prefill_timing.py` is deleted or rewritten with
  the timing module, not automatically `KEEP`;
- tests for deleted research branches must be deleted or narrowed even when
  their containing file also covers retained wrappers;
- `accelerate` remains REVIEW only until a real import/build probe decides it.

## Runtime Coverage Is The Primary Kernel Authority

### Coverage Rules

Static source reachability alone is insufficient to keep a kernel.  Test
existence alone is also insufficient: a research-only test does not make a
kernel part of the release or oracle product.

Acceptable KEEP evidence is one of:

```text
RELEASE_RUNTIME
  observed in a required optimized runtime case

ORACLE_RUNTIME
  observed in the explicit fallback full-model path or a retained CUDA oracle
  probe using representative DSV4 tensors/shapes

TRANSITIVE_BUILD_DEPENDENCY
  shared header/schema/loader required to build an observed native kernel,
  demonstrated by a narrowed build/link test
```

Every KEEP entry must name its evidence class and exact case/wrapper.

The following are not KEEP evidence:

- being exported from `__all__`;
- having a historical microbenchmark;
- being covered only by a test scheduled for deletion;
- sharing a file with a live kernel;
- being compiled because a loader globs every source;
- having `fallback`, `oracle`, `v1`, or `v2` in the name;
- inability of the first AST script to understand the call syntax.

### Runtime Coverage Matrix

Use existing evidence when it was collected from the unchanged cleanup-base
production code and includes enough owner/wrapper information.  Run only the
missing cheap cases.  Do not repeat 512K/1M full-model runs solely for this
census.

Required optimized semantic coverage:

```text
OPT-1  default text smoke, greedy, TP8, page256
OPT-2  default text smoke, non-greedy sampling, TP8
OPT-3  4096/128/bs4, captured decode, 127 replay / zero eager
OPT-4  prefix hit + miss with radix/component/SWA independent lifecycle
OPT-5  chunked prefill representative, at least 16K input
OPT-6  graph replay at representative M=1/4/16/64/128/256
OPT-7  above-graph-max eager dispatch if retained as a public contract
OPT-8  release model prepare: cached projections, Marlin WNA16 prepack,
       before-KV raw-weight release, capacity credit, component clear
OPT-9  C4, C128, SWA and indexer semantic surfaces reached by model layers and
       relevant context lengths
OPT-10 PyNCCL all-reduce/all-gather and BF16 MoE reduction
```

Required fallback/oracle semantic coverage:

```text
FB-1   fallback TP8 full-model prefill + decode text smoke, page256
FB-2   raw/grouped routed expert path with original weights retained
FB-3   torch/reference attention for C4, C128 and SWA representative contexts
FB-4   reference indexer/top-k/cache-store path
FB-5   reference projection/HC and shared-expert path
FB-6   torch.distributed reduction/gather path
FB-7   retained operator-level CUDA oracle probes for optimized wrappers
```

If one full-model run cannot cheaply produce every M/context, use no-weight or
partial-model harnesses with production-shaped tensors.  Such a probe counts
only for the exact semantic surface and launch condition it exercises.

Do not keep a low-precision experiment merely because an old microbenchmark can
execute it.  Oracle probes exist to validate retained release kernels, not to
preserve abandoned research alternatives.

## Required Work

### 1. Harden Runtime Value Collection

Update `debug/release_cleanup/` helpers to collect key/value mappings, not only
truthy names:

```text
source release-default map
resolved Engine/Scheduler config
post-adjustment environment, including arbitrary strings and integers
model_prepare_report backend/feature selections
graph/cache/communication resolved reports
```

For each env/config behavior, store:

```text
resolved value
source of value
optimized/fallback mode
reader phase: startup / model prepare / prefill / decode / graph replay
final replacement: typed public config / internal optimized constant /
                   fallback mode / delete
```

### 2. Build Wrapper-To-Kernel Launch Maps

Extend the structured parser to recognize at least:

- ordinary Python calls;
- methods/properties referenced through objects;
- Triton `kernel[grid](...)` launches;
- torch custom ops;
- TVM FFI/JIT registrations;
- dynamically imported backend constructors;
- native extension source globs and schema registrations.

Map:

```text
model/attention owner
 -> public wrapper
 -> launch predicate
 -> private Triton/custom/native kernel
 -> source translation unit/header
```

Attach runtime case ids from the coverage matrix.  For launch predicates based
on dtype, M, context, layer type, ratio, graph/eager mode, or env/config, record
the observed predicate values.

### 3. Capture Actual Kernel/Wrapper Coverage

Prefer debug-side mechanisms:

- existing wrapper counters/reports;
- temporary monkeypatch/wrapper hooks in the harness;
- Triton/JIT compile or launch metadata;
- rank0 Nsight or Torch profiler only for unresolved launch identity;
- model prepare reports for selected Marlin dispatch.

Do not add permanent production instrumentation.  Do not synchronize every
kernel launch.  A clean representative launch census is sufficient; this is
not a timing target.

### 4. Narrow Marlin WNA16 Source Ownership

The current loader globs all `sm80_kernel_*.cu`, which is not evidence that all
variants are required.

Determine the actual release dispatch tuple, including:

```text
activation dtype
weight scalar/quant type
output dtype
MoE versus dense path
```

Map it to the selected WNA16 translation unit(s).  Fallback uses grouped/raw
weights and does not justify extra Marlin variants.

Classify unused FP16, U4/U8, S8, FE4M3, dense, or other variants as DELETE when
they are not selected by release runtime and are not transitive compile
dependencies.  Before granting DELETE, prove a narrowed source list can build,
load, and run a representative Marlin oracle call.

This target may perform the narrowed build as evidence, but must not yet remove
the production source files or change the production loader.

### 5. Resolve Every Callable/Kernel Entry

Final classifications:

```text
KEEP_RELEASE
KEEP_ORACLE
KEEP_SHARED_BUILD
DELETE_RESEARCH
DELETE_DEBUG
REVIEW_BLOCKED
```

`REVIEW_BLOCKED` requires concrete missing evidence and a named resolution
action.  The final destructive target may not delete that entry.

No generic `REVIEW` reason is allowed in the hardened manifest.

### 6. Correct Module, Dependency, Test And Benchmark Manifests

Replace default-KEEP module/test classification with actual import/root
reachability:

- public roots are LLM, CLI, shell, API server, four public benchmarks, DSV4
  model, optimized mode, and fallback mode;
- modules unreachable from those roots are DELETE candidates;
- test ownership follows retained contracts, not file existence;
- debug-only tests leave with their debug implementation;
- benchmark classifications remain four PUBLIC_KEEP plus developer scripts
  moved to `debug/`, unless corrected by explicit evidence.

### 7. Reconcile The Human Report

Update the census README so its counts and conclusions match the regenerated
manifests.  Explicitly correct the string-valued optimized defaults and remove
the unsupported statement that zero unknowns was already proven.

## Deliverables

Write a new milestone instead of silently overwriting the first census:

```text
performance_milestones/misc_release_census_manifest_hardening/README.md
performance_milestones/misc_release_census_manifest_hardening/runtime_values.json
performance_milestones/misc_release_census_manifest_hardening/runtime_coverage.json
performance_milestones/misc_release_census_manifest_hardening/wrapper_kernel_map.json
performance_milestones/misc_release_census_manifest_hardening/kernel_source_manifest.json
performance_milestones/misc_release_census_manifest_hardening/env_toggle_manifest.json
performance_milestones/misc_release_census_manifest_hardening/model_dependency_manifest.json
performance_milestones/misc_release_census_manifest_hardening/benchmark_manifest.json
```

Keep reproducible source helpers in:

```text
debug/release_cleanup/
```

Do not commit generated `raw_census.json`, profiler files, `__pycache__`, or
other machine outputs.  The two helper `.py` files from misc 01 are currently
untracked; harden and commit them with the resulting planning checkpoint.

## Validation

Required cheap gates:

```bash
python -m compileall -q debug/release_cleanup
python debug/release_cleanup/build_manifests.py
python -m pytest -q \
  tests/engine/test_dsv4_release_defaults.py \
  tests/kernel/test_deepseek_v4_wrappers.py \
  tests/attention/test_deepseek_v4_backend_metadata.py
```

Validate mechanically:

- every release-default key/value appears in `runtime_values.json`;
- every KEEP kernel has at least one release/oracle/shared-build evidence id;
- every private Triton/native kernel maps to a retained wrapper or DELETE;
- no generic REVIEW reason remains;
- unknown/review counts are computed from entries;
- `flashinfer-python` is retained for sampling;
- debug timing tests are scheduled with their implementation disposition;
- the narrowed WNA16 candidate source set builds and runs, or remains
  `REVIEW_BLOCKED` with the exact build blocker.

Run missing GPU coverage cases only.  Reuse valid misc 01 and v0.0.0 evidence
when it contains the required owner identity.

## Stop Conditions

Stop without authorizing misc 02 deletion if:

- either string-valued optimized default is still classified dead;
- optimized `swa,c4` direct metadata scope is not preserved explicitly;
- a KEEP kernel lacks release/oracle/shared-build evidence;
- a DELETE kernel still launches in any required coverage case;
- a private Triton/JIT/native symbol remains generic REVIEW;
- the narrowed WNA16 source build cannot identify its required variants;
- production code was modified merely to make census instrumentation easier;
- optimized or fallback text sanity regresses during a missing coverage probe.

Do not chase performance in this target.

## Completion Criteria

- All resolved release values, including strings/integers, are captured.
- Unknown and blocked counts are computed honestly.
- Required optimized and fallback semantic surfaces have runtime coverage ids.
- Every kernel and native source is KEEP_RELEASE, KEEP_ORACLE,
  KEEP_SHARED_BUILD, DELETE_RESEARCH, DELETE_DEBUG, or concretely
  REVIEW_BLOCKED.
- Only release/oracle-required kernels are authorized for retention.
- FlashInfer sampling and retained public dependencies are correctly owned.
- Helper sources are clean and reproducible; generated files remain ignored.
- No broad production deletion has occurred.
- The hardened manifest explicitly authorizes TARGET misc 02 to begin, or
  reports a bounded blocker.

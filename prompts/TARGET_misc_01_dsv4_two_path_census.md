# TARGET misc 01: DSV4 Two-Path Release Census

## Status

Planned.  Run first after committing the pre-cleanup repository state.

## Goal

Produce an evidence-backed deletion manifest for a two-path DSV4-only release:

```text
optimized = current v0.0.0 release default + dsv4_sm80_balanced
fallback  = explicit full-model correctness/oracle mode
```

This target does not perform broad deletion.  Its purpose is to prevent later
cleanup from deleting a required DSV4 dependency or preserving research code
because its ownership is unclear.

## Required Inputs

Read first:

```text
prompts/TARGET_misc_dsv4_release_cleanup.md
prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
```

Record:

```bash
git rev-parse HEAD
git status --short
git describe --tags --always
```

Also verify:

```bash
python -c "from packaging.version import Version; print(Version('0.1.0+dsv4.sm80'))"
```

The cleanup base commit is not the final release tag.  The existing `v0.0.0`
tag remains the performance baseline, while `v0.1.0-dsv4-sm80` is reserved for
the post-cleanup release after TARGET misc 05 passes.

Abort before deletion if the base commit is not recorded or the worktree has
unexplained changes.

## Required Work

### 1. Define The Two Modes Precisely

Document the current optimized behavior resolved by an ordinary DSV4 launch:

- recipe and graph bucket policy;
- page size, prefix/radix cache, component ownership, and SWA lifecycle;
- attention/indexer/C4/C128 backends;
- MoE expert and shared-expert backends;
- projection/HC paths;
- communication backend and dtype;
- Marlin prepack/release lifecycle;
- chunked prefill and long-context behavior.

Document the intended fallback behavior:

- selected before Engine/model construction;
- no Marlin raw-weight release that would make fallback unavailable;
- reference attention/indexer/MoE/projection paths where available;
- CUDA graph disabled unless the fallback implementation is explicitly proven
  graph-safe;
- conservative cache ownership and a page size of 256;
- full-model text smoke plus operator-level oracle use.

Call out any operator that currently lacks a usable fallback.  Classify it as
`REVIEW`, not `DELETE`.

### 2. Inventory Runtime Toggles

Use structured source inspection where possible to enumerate every active
runtime control matching at least:

```text
MINISGL_DSV4_*
MINISGL_PYNCCL_*
DSV4_*_ENV
```

Classify each as:

```text
OPTIMIZED_REQUIRED
FALLBACK_REQUIRED
PUBLIC_RECIPE
PRODUCTION_SAFETY
RESEARCH_DEAD
DEBUG_INSTRUMENTATION
UNKNOWN_REVIEW
```

For each toggle record definition, readers, whether it is consulted during a
hot path, default value, and proposed final replacement.  The census is not
complete while an `UNKNOWN_REVIEW` toggle remains.

Pay special attention to the current release-default map in
`python/minisgl/engine/engine.py`.  Several validated production behaviors are
still encoded through historical env names and must later become direct config
or internal constants.

### 3. Inventory Operator And Backend Reachability

Build a call-site-oriented manifest for:

```text
python/minisgl/models/deepseek_v4.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
python/minisgl/kernel/triton/fused_moe.py
python/minisgl/kernel/marlin_wna16.py
python/minisgl/kernel/moe_impl.py
python/minisgl/kernel/pynccl.py
python/minisgl/kernel/csrc/
```

For each callable/kernel/backend, record:

```text
owner/call site
optimized reachable shapes/phases
fallback reachable shapes/phases
tests that cover it
classification: KEEP / DELETE / REVIEW
```

Do not infer deadness from names such as `_fallback`, `v1`, `v2`, `oracle`, or
`ref`.  Prove reachability from call sites and runtime mode.

Specifically identify and classify:

- old MoE v0/v1/v2 and grouped-route experiments;
- dense FP8 Marlin and FP8 indexer/cache research;
- INT8 research stubs;
- alternative projection/GEMM experiments;
- old metadata deforest/oracle paths;
- unused vendor sources and JIT kernels;
- generic attention/MoE backends used only by removed models.

### 4. Inventory Development Instrumentation

Classify runtime diagnostics separately from safety checks:

```text
python/minisgl/utils/dsv4_owner_timing.py
python/minisgl/utils/dsv4_long_prefill_timing.py
python/minisgl/utils/dsv4_memory_debug.py
python/minisgl/utils/dsv4_prefix_debug.py
NVTX decorators/ranges
graph audit/case-boundary dumps
Marlin poison/quarantine/integrity experiments
padding-boundary recorders
```

The proposed release result should remove the diagnostic machinery.  Retained
checks need a specific production safety justification and should run at
startup or fail paths, not every decode replay.

### 5. Inventory DSV4-Only Dependencies

Start from these public roots:

```text
minisgl.llm.LLM
python -m minisgl
python -m minisgl.shell
OpenAI-compatible API server
benchmark entry files
DeepseekV4ForCausalLM
```

Produce `KEEP/DELETE/REVIEW` classifications for:

- model files and registry entries;
- attention backends;
- KV cache pools;
- generic layers and MoE modules;
- scheduler/server/tokenizer/message infrastructure;
- Python dependencies in `pyproject.toml`;
- packaged C++/CUDA/vendor sources;
- tests and documentation.

DSV4 imports of general utilities make those utilities DSV4 dependencies.  Do
not delete them merely because they were originally introduced for Llama/Qwen.

### 6. Inventory Benchmarks And Public Docs

Classify all files under `benchmark/` as:

```text
PUBLIC_KEEP
MOVE_TO_DEBUG
DELETE_GENERATED
REVIEW
```

Required public files:

```text
benchmark/offline/bench.py
benchmark/offline/bench_wildchat.py
benchmark/online/bench_qwen.py
benchmark/online/bench_simple.py
```

All `deepseek_v4_*` development/microbenchmark scripts and
`dsv4_graph_reserve_lifecycle.py` should normally be `MOVE_TO_DEBUG`, unless a
documented release entry requires otherwise.

Inventory unsupported model claims in `README.md`, `docs/features.md`, and
`docs/structures.md`.  The replacement promise is entry-point support for DSV4,
not compatibility with the old named models.

### 7. Minimal Dynamic Confirmation

Use existing scripts before they are moved:

- default DSV4 text smoke, page size 256;
- short default TP8 `4096/128/bs4` probe;
- smallest practical fallback TP8 text smoke;
- import/`--help` checks for public entry points.

The fallback smoke may use very short prompt/decode lengths.  Do not wait for a
large fallback performance run.

Dynamic coverage is evidence for reachability, not a reason to retain every
branch that did not execute.

## Deliverables

Write:

```text
performance_milestones/misc_release_two_path_census/README.md
performance_milestones/misc_release_two_path_census/two_path_manifest.json
performance_milestones/misc_release_two_path_census/env_toggle_manifest.json
performance_milestones/misc_release_two_path_census/model_dependency_manifest.json
performance_milestones/misc_release_two_path_census/benchmark_manifest.json
```

Temporary census helpers belong under:

```text
debug/release_cleanup/
```

Do not add census instrumentation to production hot paths.

## Stop Conditions

Stop this target after the manifests are reviewable.  Do not begin broad
deletion.

The target is blocked for TARGET misc 02 if:

- optimized or fallback mode cannot be defined at Engine construction time;
- any hot-path env toggle remains `UNKNOWN_REVIEW`;
- a suspected dead kernel still has an unexplained DSV4 call site;
- the default baseline is already unhealthy before cleanup.

## Completion Criteria

- Base commit and baseline commands are recorded.
- Package/release identity is recorded as `minisgl==0.1.0+dsv4.sm80` with the
  historical `v0.0.0` performance tag left intact.
- Both product modes have explicit contracts.
- Every DSV4 env toggle is classified.
- Every DSV4 operator/backend family is `KEEP`, `DELETE`, or justified
  `REVIEW`.
- DSV4-only module/dependency reachability is documented.
- Benchmark and docs migration lists are complete.
- No broad source deletion occurred.

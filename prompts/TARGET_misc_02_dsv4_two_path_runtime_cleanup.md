# TARGET misc 02: DSV4 Two-Path Runtime Cleanup

## Status

Planned after TARGET misc 01.5 produces a hardened, runtime-covered census.

## Goal

Turn the research-era DSV4 runtime into two explicit release modes:

```text
optimized: default, v0.0.0 behavior, dsv4_sm80_balanced recipe
fallback:  opt-in, slow reference/oracle behavior
```

Remove research execution paths, dead kernels, runtime debug machinery, and
historical opt-in plumbing not used by either mode.

## Inputs And Preconditions

Required:

```text
prompts/TARGET_misc_dsv4_release_cleanup.md
prompts/TARGET_misc_01_dsv4_two_path_census.md
prompts/TARGET_misc_01.5_dsv4_census_manifest_hardening.md
performance_milestones/misc_release_two_path_census/README.md
performance_milestones/misc_release_census_manifest_hardening/README.md
performance_milestones/misc_release_census_manifest_hardening/*_manifest.json
```

The worktree must begin at the recorded cleanup base or a documented descendant.
Do not proceed with unresolved `UNKNOWN_REVIEW` hot-path toggles or generic
callable REVIEW entries.  Kernel deletion authority comes from the hardened
release/oracle runtime coverage manifest, not the first AST census.

## Required Design

### 1. One Public Runtime Mode

Introduce one typed runtime choice, for example:

```text
dsv4_runtime_mode = optimized | fallback
```

Expose it consistently through:

- `EngineConfig`/`SchedulerConfig`;
- `LLM(..., dsv4_runtime_mode=...)`;
- CLI `--dsv4-runtime {optimized,fallback}`.

The exact spelling may follow local style, but there must be one canonical
switch.  The mode is immutable after Engine construction.

Do not retain dozens of research env vars as compatibility aliases.  The old
`MINISGL_DSV4_DISABLE_RELEASE_DEFAULTS` may be removed once all scripts/tests
use the canonical fallback mode.

### 2. Direct Optimized Defaults

For DeepSeek V4 on sm80, ordinary construction must directly resolve:

```text
runtime mode: optimized
recipe: dsv4_sm80_balanced
page size: 256
validated radix/component/SWA lifecycle
validated CUDA graph bucket policy
Marlin WNA16 expert preparation and safe raw-weight release
BF16 MoE reduce and promoted PyNCCL policy
in-graph/direct metadata behavior
chunked prefill budget
```

Replace the historical `_DSV4_SM80_RELEASE_DEFAULT_ENV` bundle with typed config
and direct calls/internal constants.  Avoid environment lookups in model,
attention, graph replay, and kernel hot paths.

Retain explicit public recipes such as graph64/128, balanced graph256, 512K, and
1M smoke only if they use the same optimized implementation.  Recipes are
capacity/performance configurations, not separate execution paths.

### 3. Coherent Fallback Mode

Fallback mode must be selected before weights are loaded and must:

- keep raw weights needed by reference execution;
- avoid Marlin-only assumptions and original-weight release;
- use reference/oracle call sites identified by the census;
- disable CUDA graph by default;
- use page size 256 and a bounded smoke-size cache;
- provide a clear error when an operator has no fallback rather than silently
  entering an untested research path.

Do not support switching an already initialized optimized Engine to fallback.

### 4. Delete Non-Release Operator Paths

Delete `DELETE_RESEARCH` and `DELETE_DEBUG` entries from the hardened operator
manifest, including associated Python, Triton, C++/CUDA, vendor, registration,
export, and test code.  Keep only kernels with `KEEP_RELEASE`, `KEEP_ORACLE`, or
`KEEP_SHARED_BUILD` evidence.

Likely candidates requiring manifest confirmation include:

- dense FP8 Marlin projection research and unused vendor bridge;
- FP8 indexer/KV paths not promoted to `v0.0.0`;
- INT8 research stubs;
- superseded MoE v0/v1/v2/grouped experiments;
- superseded metadata oracle/deforest variants;
- unused projection/GEMM experiments.

Do not remove optimized shape dispatch or the actual fallback oracle.
Do not retain a kernel only because it was historically benchmarked, exported,
or compiled by a broad source glob.

### 5. Remove Development Runtime Code

Remove diagnostic-only imports, helpers, state, env parsing, and wrappers from
production modules:

- owner/long-prefill/memory/prefix timing and dumps;
- NVTX annotations/ranges used during TARGET development;
- Marlin poison, quarantine, hidden-reference, layer-filter, and census paths;
- graph audit/run labels/case-boundary dumps;
- padding-boundary capture;
- per-step debug counters and debug-only synchronization.

Delete now-unused utility modules and tests.  Temporary analysis harnesses may
live under `debug/release_cleanup/`, but production modules must not import
them.

Retain cheap production invariants.  Convert validated lifecycle behavior with
a misleading debug name into normal code.  In particular, the safe Marlin
release order must not remain controlled by a `DEBUG_RELEASE_TIMING` env var.

### 6. Simplify Tests Around The Two Modes

Rewrite DSV4 tests to assert:

- default construction resolves optimized/balanced behavior without env vars;
- explicit fallback resolves the oracle mode;
- optimized and fallback are mutually exclusive and immutable;
- unsupported historical opt-ins are absent or rejected clearly;
- raw expert weights are released only in optimized mode at the safe phase;
- graph/cache/SWA production invariants survive cleanup;
- deleted modules/exports cannot be imported accidentally.

Do not keep tests whose only purpose is preserving a removed experiment.

## Validation Ladder

Run cheap gates after each ownership area, then one macro at the end.

Minimum CPU/import gates:

```bash
python -m compileall -q python/minisgl
python -m pytest -q tests/engine tests/kernel tests/attention tests/models
```

Minimum optimized TP8 gates:

- text sanity with page size 256 and no DSV4 env vars;
- `historical_4096_128_bs4`;
- `historical_4096_1024_bs4`;
- one prefix-hit scenario;
- zero unexpected eager fallback for captured buckets.

Minimum fallback gate:

- short TP8 load and text smoke;
- selected attention/indexer/MoE/projection oracle comparisons;
- no raw-weight release or optimized-only graph capture.

Use repeat-stable medians for performance.  The optimized mode should remain
within normal run variance of the `v0.0.0` baseline.  Treat a repeat-stable
regression above 3% as a blocker unless traced to removal of instrumentation
that changed measurement semantics rather than execution.

## Deliverables

```text
performance_milestones/misc_release_two_path_runtime_cleanup/README.md
performance_milestones/misc_release_two_path_runtime_cleanup/default_config.json
performance_milestones/misc_release_two_path_runtime_cleanup/fallback_config.json
performance_milestones/misc_release_two_path_runtime_cleanup/deleted_runtime_files.txt
```

The report must map every census `DELETE` item to its deletion and every
retained `REVIEW` item to a reason.

## Stop Conditions

Stop and do not start model pruning if:

- fallback requires weights already released by optimized initialization;
- the default path still depends on research env vars;
- deleting a candidate changes DSV4 text sanity or cache ownership;
- optimized performance has an unexplained repeat-stable regression over 3%;
- production code still imports deleted debug modules.

Do not optimize unrelated kernels in this target.  Fix cleanup regressions only.

## Completion Criteria

- One canonical optimized/fallback mode exists.
- Optimized balanced behavior is the no-env default.
- Fallback is a documented pre-construction opt-in.
- Research operator paths classified `DELETE` are gone.
- Runtime NVTX/debug instrumentation is gone or explicitly justified as a
  production invariant.
- Focused tests, optimized smoke/macro, and fallback smoke pass.
- A checkpoint commit is recommended before TARGET misc 03.

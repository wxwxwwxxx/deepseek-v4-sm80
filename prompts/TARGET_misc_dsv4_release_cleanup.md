# TARGET misc: DeepSeek V4 Release Cleanup

## Status

Planned final cleanup after the tagged `v0.0.0` performance baseline.

This route prepares mini-sglang for a small, understandable open-source
release.  The supported model scope is intentionally narrow:

```text
Model family:       DeepSeek V4 Flash only
Primary platform:   NVIDIA A100 / sm80
Validated topology: TP8 on DGX A100 80 GB x8
Default runtime:    v0.0.0 optimized path + dsv4_sm80_balanced recipe
Oracle runtime:     explicit fallback mode
MTP:                not included
```

The README must promise that the public mini-sglang entry points work for
DeepSeek V4 Flash.  It must not promise compatibility with the removed Qwen,
Llama, Mistral, or other model families.

## Release Identity

Use the following canonical identity for this downstream release:

```text
distribution/import name: minisgl
Python package version:   0.1.0+dsv4.sm80
human release name:       Mini-SGLang 0.1.0, DSV4 on SM80
recommended final tag:    v0.1.0-dsv4-sm80
historical perf tag:      v0.0.0
```

`0.1.0.dsv4_on_sm80` is not a valid PEP 440 version.  The `+dsv4.sm80`
suffix is the canonical local-version form for this downstream build.  Keep
the existing `v0.0.0` tag unchanged as the pre-cleanup performance baseline;
the final release tag should point to the post-cleanup qualified commit.

The local version is suitable for source/GitHub distribution, direct wheel
installation, or a private package index.  PEP 440 says public PyPI must not
accept local version identifiers.  A future PyPI publication therefore needs
an explicit decision to use a pure public version and must separately verify
ownership/availability of the `minisgl` distribution name.

## Release Contract

The cleaned runtime has exactly two product-level execution modes:

1. `optimized`: the tested `v0.0.0` DSV4 sm80 implementation and its validated
   shape/context dispatch.  This is the default.
2. `fallback`: an explicit, slow reference/oracle mode selected before model
   construction.  It must remain usable for correctness diagnosis.

Shape-aware dispatch inside the optimized mode is not a third product path.
Prefill/decode, C4/C128/SWA, Marlin expert, CUDA graph/eager execution, and
long-context recipes are required sub-surfaces of the same optimized runtime.
Do not delete a required implementation merely because there is more than one
kernel shape or because its function name ends in `_fallback`.

The public default should require no DSV4 environment-variable recipe:

```python
llm = LLM("/models/DeepSeek-V4-Flash", ...)
```

and:

```bash
python -m minisgl --model /models/DeepSeek-V4-Flash --tp-size 8
```

must resolve to the optimized `dsv4_sm80_balanced` behavior unless the user
explicitly selects another public recipe or `fallback` mode.

## Preconditions

Before any deletion target begins:

1. Commit this planning state and all intended pre-cleanup code.
2. Record the immutable base commit in every child report.
3. Start cleanup from a dedicated branch or a clearly named commit series.
4. Keep the existing `v0.0.0` tag as the performance reference.  Do not move or
   rewrite it automatically.  Reserve `v0.1.0-dsv4-sm80` for the qualified
   post-cleanup release commit.
5. Confirm `git status --short` is clean.  If unrelated changes exist, preserve
   them and do not hide them with reset/checkout.

The final soak must verify that package metadata resolves exactly to
`0.1.0+dsv4.sm80` and that any final release tag follows the identity above.

## Child Targets And Order

Run these targets in order:

1. `TARGET_misc_01_dsv4_two_path_census.md`
   - no broad deletion;
   - produce runtime, operator, env-toggle, model-dependency, and benchmark
     `KEEP/DELETE/REVIEW` manifests.
2. `TARGET_misc_01.5_dsv4_census_manifest_hardening.md`
   - correct string-valued default and hard-coded unknown classifications;
   - use actual release/oracle runtime coverage to authorize kernel retention;
   - resolve wrapper/private Triton/JIT/native source ownership.
3. `TARGET_misc_02_dsv4_two_path_runtime_cleanup.md`
   - establish one explicit runtime-mode contract;
   - make optimized direct/default and fallback explicit;
   - remove research paths, dead kernels, NVTX, and runtime debug machinery.
4. `TARGET_misc_03_dsv4_only_repository_prune.md`
   - delete unsupported model implementations and code proven unreachable from
     DSV4 serving;
   - prune tests, registries, dependencies, and docs accordingly.
5. `TARGET_misc_04_dsv4_benchmark_readme_surface.md`
   - preserve the four requested generic benchmark entry files;
   - move DSV4 development/microbench scripts under `debug/`;
   - rewrite README and public examples for DSV4-only support.
6. `TARGET_misc_04.5_dsv4_openai_api_compatibility_hardening.md`
   - harden the text-only OpenAI chat request/response contract;
   - eliminate silently ignored semantic parameters and fabricated usage;
   - validate the OpenAI SDK and vLLM `openai-chat` benchmark without touching
     the model execution path.
7. `TARGET_misc_05_dsv4_release_cleanup_soak.md`
   - validate package, Python/CLI/server/shell/benchmark entries, optimized and
     fallback modes, correctness, capacity, and performance.

Each child target must stop at its own boundary.  Do not combine all deletion
work into one unreviewable commit.

## Deletion Rules

Delete code only when at least one of these is true:

- it is unreachable from both the optimized and fallback DSV4 modes;
- it implements an unsupported model/backend and no retained DSV4 code imports
  it;
- it is a research/failed opt-in excluded by the release manifest;
- it is diagnostic instrumentation with no production correctness role;
- it is a test or documentation file that only describes removed behavior.

Keep or rewrite code when any of these is true:

- optimized DSV4 invokes it for any validated shape/context;
- fallback/oracle invokes it;
- it owns a serving invariant such as cache lifetime, graph reserve, admission,
  tokenization, sampling, tensor parallel communication, or weight loading;
- it is needed by a public entry point or package import;
- it is a startup fail-fast check that prevents memory corruption or silent
  wrong results.

Prefer removing dead branches and env checks over leaving compatibility
aliases.  This release has not promised the historical research env surface.

## Instrumentation Policy

Remove release-runtime development instrumentation, including where proven
unnecessary:

- DSV4 owner timing and long-prefill timing;
- memory/prefix/padding debug recorders and dumps;
- poison, quarantine, sentinel census, and layer-filter experiments;
- graph/case-boundary audit logs;
- per-step NVTX wrappers and decorators;
- debug-only environment parsing in hot paths.

Do not remove a safety invariant merely because its current env name contains
`DEBUG`.  For example, the validated Marlin release phase must become a normal
internal lifecycle decision rather than remain
`MINISGL_DSV4_MARLIN_WNA16_DEBUG_RELEASE_TIMING=before_kv_alloc`.

## Global Non-Goals

- Do not revive MTP.
- Do not add new model families.
- Do not redesign scheduler/cache/kernel algorithms for extra performance.
- Do not change numerical precision or sampling semantics.
- Do not delete archived prompts or historical reports merely to reduce source
  count.
- Do not require old Qwen/Llama README examples to keep working.
- Do not retain dead code solely because another upstream mini-sglang version
  had it.

## Overall Completion Gate

The route is complete only when:

```text
default DSV4 runtime mode: optimized
default recipe: dsv4_sm80_balanced
package version: 0.1.0+dsv4.sm80
fallback/oracle: one documented opt-in selected before Engine construction
active DSV4 research opt-ins: none
active release debug/NVTX machinery: none, except justified production checks
registered model architectures: DeepSeek V4 only
README model claims: DeepSeek V4 only
requested benchmark entry files: runnable with DSV4 defaults
DSV4 development benchmarks: under debug/
optimized text/correctness/performance soak: pass
fallback smoke/oracle gate: pass
Python, CLI, server, shell, offline benchmark, online benchmark entries: pass
```

The final report must include deleted files, retained exceptions, package size
change, source-line change, default/fallback commands, performance comparison
against `prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md`, and any residual release
risk.

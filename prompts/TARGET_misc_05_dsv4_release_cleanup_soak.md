# TARGET misc 05: DSV4 Release Cleanup Final Soak

## Status

Planned final gate after TARGET misc 01-04.

## Goal

Prove that repository cleanup preserved the `v0.0.0` DSV4 behavior while
delivering a smaller, truthful, two-path, DSV4-only release.

Do not use this target for new optimization.  It is a release qualification
and closure target.

## Required Inputs

Read all prior cleanup reports and:

```text
prompts/TARGET_misc_dsv4_release_cleanup.md
prompts/TARGET_misc_04.5_dsv4_openai_api_compatibility_hardening.md
prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md
prompts/TARGET_11_dsv4_sm80_mtp_speculative_decoding.md
prompts/TARGET_12_dsv4_sm80_decode_replay_metadata_latency_hiding.md
```

Record the cleanup base commit, current commit, diffstat, deleted files, and
current package version.  The expected package version is
`0.1.0+dsv4.sm80`.

## Required Gates

### 1. Repository Hygiene

Confirm:

- no tracked/untracked `__pycache__` or `.pyc` debris;
- no active MTP runtime, opt-in, or debug harness;
- no removed model registration/import;
- no release-runtime DSV4 debug/NVTX dependency;
- no research env toggle accepted silently;
- no benchmark data downloaded into tracked source directories;
- no broken README/docs links;
- no generated report is accidentally staged.

Archived prompts may mention old models, MTP, and research toggles.  They are
historical evidence and should not be treated as active runtime references.

### 2. Clean Package Build And Install

In a fresh environment or clean install prefix:

```text
build wheel/sdist
install package
import minisgl
run CLI --help and shell --help
verify native/JIT sources required by DSV4 are packaged
verify removed modules are absent
```

Verify that installed package metadata reports `0.1.0+dsv4.sm80`.  Do not move
the existing `v0.0.0` performance tag.  After all gates pass, the recommended
new release tag is `v0.1.0-dsv4-sm80` on the final cleanup commit.

This local version must not be uploaded to public PyPI.  Public PyPI release is
a separate future decision requiring a pure public version and distribution
name review.

### 3. Unit And Static Gates

Run:

```bash
python -m compileall -q python/minisgl benchmark debug/dsv4
python -m pytest -q
```

Run the repository formatter/linter configured by `pyproject.toml` on changed
files.  Record skips and environmental failures explicitly.

### 4. Default Optimized Correctness

With no DSV4 research env vars:

- load `/models/DeepSeek-V4-Flash` on TP8;
- verify resolved mode `optimized`, recipe `dsv4_sm80_balanced`, and page size
  256;
- run multilingual text sanity prompts;
- check no乱码/NaN/Inf and reasonable stop behavior;
- test greedy and non-greedy sampling;
- test prefix hit/miss and radix reuse;
- test SWA independent lifecycle and Marlin raw-weight release capacity;
- verify captured decode uses expected graph buckets with no unexpected eager
  fallback.

### 5. Explicit Fallback/Oracle Correctness

Select fallback through the one documented public switch before Engine
construction:

- load on TP8 with a bounded cache;
- run short text sanity;
- prove raw expert weights needed by fallback were not released;
- prove optimized-only CUDA graph/Marlin assumptions are disabled;
- run selected operator oracle comparisons;
- verify unsupported post-construction mode switching fails clearly.

Fallback performance is not a release target.

### 6. Public Entry Matrix

Execute every command in the README command matrix:

- Python `LLM` entry under its documented TP launch;
- `python -m minisgl` server;
- `/v1/models` and OpenAI-compatible request;
- OpenAI SDK streaming/non-streaming and a short vLLM `openai-chat` benchmark;
- `python -m minisgl.shell` startup/exit;
- offline `bench.py`;
- offline `bench_wildchat.py` with a small cached shard or documented network
  prerequisite;
- online `bench_simple.py`;
- online `bench_qwen.py` trace workload.

An entry passes only if it performs useful work, not merely if argument parsing
succeeds.

### 7. Performance And Capacity Regression Matrix

Compare repeat-stable optimized results with
`prompts/DSV4_SM80_V0.0.0_RELEASE_BASELINE.md` using the same hardware and
measurement lifecycle.

Minimum scenarios:

```text
historical_4096_128_bs4
historical_4096_1024_bs4
serving_mixed_112req_wave16
prefix_multi_112req_wave16
graph M=4/16/64/128/256 representative decode
16K chunked prefill smoke
512K context smoke with its capacity recipe
1M single-request smoke
```

The 512K/1M cases are capability smoke, not publication-grade performance
requirements.  Do not require 1M and high concurrency simultaneously.

Default short/serving performance should remain within normal variance of the
baseline.  Investigate a repeat-stable regression over 3%.  Do not reopen
kernel optimization for sub-3% noise during release cleanup.

Record:

- output/decode throughput and TTFT where applicable;
- graph replay/eager counts and buckets;
- free memory, graph reserve, KV/component/SWA capacity;
- prefix saved tokens;
- communication backend/bytes;
- text sanity result.

### 8. Unsupported-Surface Behavior

Verify that attempts to load an old Qwen/Llama/Mistral architecture fail early
with a clear DSV4-only support message.  They are not required to run.

Verify that removed research toggles do not silently select partial paths.  An
unknown removed option should be rejected or ignored with an explicit warning,
according to the final public config policy.

## Deliverables

```text
performance_milestones/misc_release_cleanup_final_soak/README.md
performance_milestones/misc_release_cleanup_final_soak/command_matrix.md
performance_milestones/misc_release_cleanup_final_soak/performance.json
performance_milestones/misc_release_cleanup_final_soak/repository_diffstat.txt
performance_milestones/misc_release_cleanup_final_soak/release_checklist.md
```

The README must state `PASS`, `FAIL`, or `BLOCKED` for:

```text
two-path runtime contract
DSV4-only model scope
package build/install
full unit suite
default correctness
fallback correctness
public entries
performance regression
long-context capability
repository hygiene
version/tag identity
```

## Stop Conditions

Mark blocked and do not recommend publication if:

- default or fallback text sanity fails;
- cache/SWA/Marlin release correctness regresses;
- a README public entry cannot complete useful work;
- the package omits a required native source;
- an unexplained repeat-stable default regression exceeds 3%;
- unsupported model code remains registered;
- installed package metadata is not `0.1.0+dsv4.sm80` or the intended tag
  identity remains contradictory.

Do not fix blockers by restoring whole removed model/research trees.  Restore
only the proven DSV4 dependency or correct the public claim.

## Completion Criteria

- Final soak report is complete and all mandatory gates pass.
- `v0.0.0` optimized behavior remains the performance reference.
- Release defaults require no DSV4 env recipe.
- Fallback is one explicit documented opt-in.
- README command matrix is executable.
- Repository scope and package metadata accurately say DSV4-only.
- The user can make the final release commit/tag decision with a concise list
  of known residual risks.

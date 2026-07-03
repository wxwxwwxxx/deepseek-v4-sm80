# TARGET 07.78: DSV4 SM80 Benchmark Lifecycle And Repeat-Stable Gate

Date: 2026-07-03

## Goal

Make the DeepSeek V4 TP8 benchmark harness stable enough to fairly judge
`dsv4_sm80_a100_victory` against
`dsv4_sm80_a100_victory_densefp8marlinproj`.

This target is about measurement lifecycle and promotion policy, not kernel
optimization.

Primary outcome:

```text
a fair repeat-stable benchmark gate for dense FP8 Marlin projection
```

The target should decide whether the dense FP8 Marlin projection opt-in is:

- stable enough to promote;
- neutral/too small and should remain opt-in;
- still noisy because the benchmark harness needs more lifecycle work.

## Starting Evidence

TARGET 07.76 integrated mini-owned dense FP8 Marlin projection runtime:

```text
MINISGL_DSV4_SM80_DENSE_FP8_MARLIN_PROJECTION=1
dsv4_sm80_a100_victory_densefp8marlinproj
```

It passed:

- TP8 text smoke;
- graph replay;
- eager decode `0`;
- vLLM dependency audit;
- memory lifecycle, with about `807 MB/rank` lower peak allocation.

But the first 4096/1024 macro regressed:

```text
127.4409 -> 118.9051 output tok/s  (-6.70%)
```

TARGET 07.77 then diagnosed the regression:

```text
performance_milestones/target07_dense_fp8_marlin_runtime_regression_attribution/README.md
```

Key 07.77 conclusions:

- primary bucket: `measurement fairness/noise`;
- 07.76's `prepare +1.7s` / `TTFT +1.7s` pattern did not reproduce under
  repeated clean runs;
- second repeat was neutral:
  `131.4680 -> 131.5082 output tok/s`;
- dense Marlin pure GEMM/custom-op was faster for `q_wqb`, `wo_b`, and
  `shared_down`;
- layout/copy was not the issue: all reshape boundaries were views and
  `.contiguous()` was skipped;
- communication bytes/counts were unchanged;
- remaining owner-level losses were about `0.19s` over `1023` replays, far
  smaller than the original seconds-scale regression.

Therefore do not optimize kernels in this target.  First make the benchmark
decision rule stable.

## Known Harness Problem

The current perf-matrix harness constructs one `LLM`/Engine per torchrun.
Per-variant env can be applied too late when multiple variants are run in one
process.  TARGET 07.76 saw this directly: a same-run baseline/candidate text
smoke prepared the first variant only and failed the candidate with a missing
packed dense FP8 Marlin cache.

This target should fix or bypass that lifecycle problem.

## Non-Goals

Do not do these in this target:

- change dense FP8 Marlin kernels;
- change owner scope;
- expand Phase B owners;
- add INT8 MoE;
- add TVM FFI;
- change FP8 KV cache;
- tune all-reduce ordering;
- promote based on one noisy run.

Small benchmark harness and reporting changes are expected.

## Artifacts

Create:

```text
performance_milestones/target07_benchmark_lifecycle_repeat_stable_gate/
  README.md
  raw/
  scripts/
  summaries/
```

Large raw profiler files should stay under `raw/` or be symlinked from `/tmp`.

## Implementation Plan

### 1. Fix Or Add A Fair Variant Lifecycle

Implement one of these options:

Option A, preferred:

- add a benchmark mode that constructs a fresh `LLM`/Engine per variant inside
  the same torchrun command;
- apply variant env before model construction, weight loading,
  `prepare_for_cuda_graph_capture()`, and CUDA graph capture;
- tear down or isolate each variant before constructing the next.

Option B, acceptable if Option A is too invasive:

- add scripts that run separate torchrun invocations per variant;
- use the same command template, same `--num-pages`, same repeats/warmup, same
  environment, and collect summaries into one report;
- make the final comparison script explicitly pair the baseline/candidate runs.

Whichever route is chosen, document why.

### 2. Define A Repeat-Stable Macro Gate

Run at least:

```text
TP8
page size 256
--num-pages 128
prompt_len 4096
decode_len 1024
batch_size 4
warmup_repeats >= 1
measured repeats >= 2
```

Variants:

```text
dsv4_sm80_a100_victory
dsv4_sm80_a100_victory_densefp8marlinproj
```

Also run 4096/128 with the same lifecycle for short-shape sanity.

For each variant/report:

- report every repeat individually;
- report warmup separately and exclude it from promotion decision;
- report mean, median, best, worst, standard deviation, and coefficient of
  variation for output tok/s, decode tok/s, TTFT, prefill forward, decode
  forward, and elapsed time.

### 3. Preserve Correctness And Graph Gates

Before macro decision, run TP8 text smoke for the candidate:

- page size 256;
- sane outputs;
- graph replay active;
- eager decode `0`;
- dense FP8 Marlin cache present;
- no duplicate BF16 cache for switched owners;
- no vLLM runtime dependency.

### 4. Promotion Rule For This Target

The dense FP8 Marlin projection opt-in can be promoted only if:

- text smoke passes;
- graph replay/eager gates pass;
- memory lifecycle stays clean;
- 4096/1024 measured-repeat median output tok/s improves over baseline by at
  least `2%`;
- 4096/1024 mean output tok/s improves by at least `1%`;
- 4096/128 median output tok/s does not regress by more than `1%`;
- candidate variation is not materially worse than baseline:
  coefficient of variation no more than `1.5x` baseline or no more than `2%`,
  whichever is looser;
- individual measured repeats do not show a repeatable catastrophic regression
  worse than `-3%` after warmup.

If the candidate is within `[-1%, +2%]` on 4096/1024 median, keep it as an
explicit opt-in because the benefit is too small for a default bundle change.

If results remain noisy enough that the confidence interval overlaps both
promotion and regression thresholds, do not promote; recommend the next
benchmark-lifecycle fix.

### 5. Optional Owner Timing Sanity

If the repeat-stable macro still shows a real regression, run a light owner
timing pass using the 07.77 instrumentation to confirm whether the cause is
still measurement/noise or a repeatable small runtime cost.

Do not optimize it here.  Only classify and recommend the next target.

## Required README Content

The README must include:

- chosen lifecycle route, Option A or B;
- exact commands;
- git status summary;
- text smoke result;
- repeat tables for 4096/1024 and 4096/128;
- warmup handling;
- mean/median/std/CV comparison;
- graph replay/eager counts;
- memory lifecycle check;
- final promote / keep opt-in / inconclusive decision;
- next target recommendation.

## Stop Rules

Stop once a repeat-stable decision is made.

Hard stop if:

- candidate smoke fails;
- fresh-engine lifecycle cannot be made fair and separate invocations also
  cannot be summarized reliably;
- measured repeats still show severe unexplained noise after warmup;
- graph replay fails or eager decode appears.

Do not continue into dense Marlin owner optimization in this target.

## Suggested README Outline

```text
# TARGET 07.78: Benchmark Lifecycle And Repeat-Stable Gate

Status:

## Lifecycle Route
## Correctness / Graph / Memory Gates
## 4096/1024 Repeat-Stable Macro
## 4096/128 Repeat-Stable Macro
## Variance Analysis
## Decision
## Next Target
```


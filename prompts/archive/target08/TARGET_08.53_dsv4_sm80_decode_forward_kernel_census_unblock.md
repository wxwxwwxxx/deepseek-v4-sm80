# TARGET 08.53: DSV4 SM80 Decode Forward Kernel Census Unblock

## Status

Active TARGET 08 follow-up after TARGET 08.52.

TARGET 08.52 confirmed that the remaining SWA-independent direct-token
metadata performance gap is inside captured decode CUDA graph replay, but it
stopped before any kernel or layout patch because Nsight Systems did not
produce usable reports for TP8 `torchrun` serving traces.

This target is deliberately narrow: use low-cost probes to unblock
kernel/operator census and identify the slow kernel class first.  Do not
optimize attention, metadata, or layout until there is a stable replay kernel
delta.

## Goal

Produce a stable kernel-level census and low-cost slow-kernel reproducer for
captured decode graph replay on A100/sm80, comparing:

```text
Route B prefix baseline
SWA independent + direct token metadata
```

The target should answer:

```text
Can we generate usable per-rank kernel summaries for captured graph replay?
Can small no-weight or partial-workload probes identify the slow kernel class
before full model inference?
If yes, which kernels differ between Route B and SWA direct?
If no, what exact profiler mode fails, and what replacement path can produce
an equivalent kernel/operator census?
```

The preferred output is not a full macro benchmark.  Start with no-weight and
partial replay probes, then move to the smallest full-model replay probe needed
to expose the real kernel names.  Run one complete inference/macro only at the
end to validate a finding or to prove that the low-cost probe is not faithful.

## Starting Evidence

Read first:

```text
performance_milestones/target08_swa_decode_forward_graph_replay_parity/README.md
performance_milestones/target08_swa_decode_forward_graph_replay_parity/safe_replay_timing.md
performance_milestones/target08_swa_decode_forward_graph_replay_parity/kernel_timeline_census.md
performance_milestones/target08_swa_decode_forward_graph_replay_parity/operator_microbench.md
performance_milestones/target08_swa_decode_forward_graph_replay_parity/fix_or_next_target.md
prompts/TARGET_08.52_dsv4_sm80_swa_independent_decode_forward_graph_replay_parity.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Key 08.52 numbers:

```text
Clean decode_forward_s delta:
  historical_4096_1024_bs4:      +1.5773 s
  serving_mixed_112req_wave16:   +0.6206 s

External replay timing delta:
  historical_4096_1024_bs4:      +1.8540 s
  serving_mixed_112req_wave16:   +0.9036 s
```

Both variants used captured graph replay only, with eager decode count `0`.
Replay counts and replay input-copy bytes matched.  The remaining gap is
therefore inside captured decode graph replay, not the old Python page-table
construction or graph staging/copy ledger.

Nsight Systems smoke succeeded on a simple CUDA program, but TP8 `torchrun`
attempts completed the benchmark without producing usable `.nsys-rep` files or
hung during profiler shutdown.

## Runtime Shapes

Keep the canonical DSV4 settings unless a probe is explicitly no-weight:

```text
TP8
page size 256
num pages 128 for fixed-capacity comparisons
CUDA graph buckets [1,2,4,8,16]
greedy sample graph capture
eager decode 0 for full-model captured probes
```

Variants:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
dsv4_sm80_a100_victory_prefix_routeb_lifetime_swa_independent_swadirect
```

Keep these opt-ins available for ablation, but do not promote them here:

```text
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
MINISGL_DSV4_GRAPH_REPLAY_TIMING=1
MINISGL_DSV4_GRAPH_REPLAY_TIMING_MAX_SAMPLES=N
```

## Non-Goals

- Do not implement an attention kernel, indexer kernel, metadata-layout, or
  scheduler fix in this target.
- Do not run full long macro matrices unless a short probe cannot expose the
  real kernel owner.
- Do not start with full model loading when a no-weight or synthetic graph
  replay probe can test the profiler mechanism or reproduce the slow kernel
  class.
- Do not weaken the SWA independent lifecycle contract.
- Do not revisit 08.49 page-table cache or 08.50 direct-token metadata unless
  the kernel census points back to them.
- Do not use CUDA events or synchronizes inside captured CUDA graph contents.
- Do not treat non-captured eager operator timing as equivalent to captured
  graph replay unless validated against at least one replay-level probe.

## Required Work

### 0. Cost Discipline And Probe Ladder

Treat this target as a low-cost diagnostic pipeline, not a benchmark sweep.

Use this escalation order:

```text
1. no-weight single-process CUDA graph replay
2. no-weight TP8 torchrun CUDA graph replay
3. synthetic/partial DSV4 operator replay with Route B-like vs SWA-like inputs
4. shortest full-model captured replay probe that exposes real kernel names
5. one complete inference or macro validation run
```

At each step, stop escalating if the current probe already gives a stable
kernel-level difference that can guide the next microbench target.

The no-weight and partial probes should not merely prove that profiling works.
They should actively try to reproduce the suspected difference by preserving
the relevant decode inputs:

- graph bucket size and replay count;
- SWA window size;
- compact SWA locs versus full/Route B locs;
- token-location locality;
- invalid `-1` density;
- `swa_topk_lengths`;
- C4/C128/indexer metadata shapes if the target kernel consumes them;
- dtype, strides, alignment, and page size `256`.

Only the final validation should require a complete model inference path.

### 1. Profiler Harness Preflight Without Full Weights

First prove that the profiling strategy can emit usable kernel summaries in
small controlled programs.  Prefer probes that are already shaped like captured
decode replay so their results can be reused for slow-kernel diagnosis.

Use one or more tiny probes:

```text
single-process torch CUDA graph capture/replay
single-process graph replay with repeated kernels and NVTX ranges
torchrun nproc=8 no-weight graph replay, one CUDA device per rank
torchrun nproc=8 no-weight NCCL/init_process_group smoke if needed
```

Test profiler collection modes systematically:

```text
nsys profile around the Python program
nsys profile around torchrun
per-rank report naming with %p or rank-specific output directories
capture-range cudaProfilerApi, if adding cudaProfilerStart/Stop is simpler
capture-range nvtx, if NVTX ranges are reliable
wait=primary vs wait=all
trace-fork-before-exec true/false
cuda-graph-trace=node vs graph-level/default trace
```

Record which modes produce:

- `.nsys-rep`;
- `.sqlite`;
- `cuda_gpu_kern_sum`;
- graph replay node visibility, if available;
- clean process shutdown.

Stop early if a simple no-weight torchrun graph replay cannot be profiled at
all.  In that case, write a focused profiler blocker report and propose the
next environment/profiler fix instead of loading the model.

### 2. Low-Cost Slow-Kernel Reproducer

Before loading full DSV4 weights, try to build a synthetic or partial replay
that can expose the same kernel class expected in decode forward.

Reasonable probe families:

```text
captured graph replay with repeated attention/indexer helper calls
captured graph replay with synthetic C4/C128/SWA metadata tensors
captured graph replay with compact-SWA locs versus full-layout locs
captured graph replay with representative invalid density and topk lengths
partial model/layer probe if one layer can be instantiated without full weights
```

The goal is to find one of these outcomes:

```text
same kernel set, SWA-like inputs slower
extra copy/memset/indexer helper appears only in SWA-like path
invalid/locality distribution explains the slowdown
no-weight probes cannot reproduce the gap, requiring minimal full-model replay
```

If a no-weight or partial probe finds a stable slow kernel, record it as the
primary evidence and use the full model only to validate that the same kernel
appears on the real path.

### 3. Rank-Local TP8 Profiling Path

If wrapping the outer `torchrun` remains unreliable, build or document a
rank-local profiling path.

Acceptable approaches include:

- a small launcher/wrapper that profiles only selected ranks and lets other
  ranks run normally;
- per-rank output directories keyed by `LOCAL_RANK`, `RANK`, and process id;
- a benchmark flag or helper that starts/stops profiling after engine
  construction and CUDA graph capture;
- `cudaProfilerStart/Stop` or NVTX capture ranges around a short replay loop;
- PyTorch profiler/CUPTI-based kernel summaries only if they show CUDA kernels
  emitted by graph replay and can be exported repeatably.

The key requirement is not a specific tool.  The requirement is a repeatable
per-rank kernel/operator census for captured decode graph replay.

### 4. Small Full-Model Replay Probe

Only after the profiler harness works on no-weight probes, run the smallest
full-model probe that exposes the real DSV4 decode replay kernels or validates
the slow kernel found by the low-cost reproducer.

Prefer one of these over the long macro:

```text
short historical shape, for example 4096 input / 64 or 128 decode / bs4
short serving-style replay probe that exercises buckets [1,2,4,8,16]
default-off benchmark mode that captures the model graph, then replays one
selected bucket N times
```

The probe must still preserve the relevant graph shape:

- page size `256`;
- captured decode graph replay, eager decode `0`;
- same bucket for Route B and SWA direct;
- same request/batch shape;
- same graph capture policy.

If the short probe does not reproduce a replay-level delta, record that and
escalate only one step: a slightly longer decode or the exact 08.52 serving
shape.  Do not jump immediately to a full large matrix.

### 5. Kernel Census Report

For Route B and SWA direct, report at least one stable rank, preferably all
selected ranks:

- kernel names;
- launch counts;
- total GPU time per kernel;
- mean time per replay when replay count is known;
- CUDA memcpy/copy kernels;
- NCCL kernels, if any appear during decode replay;
- graph replay counts and bucket ids;
- whether graph nodes or only kernel summaries are visible;
- profiler overhead and any shutdown issues.

The report should classify differences as:

```text
extra kernels
same kernels but slower
different launch counts
copy/memset/memcpy difference
NCCL/communication difference
no stable kernel-level difference
profiler still blocked
```

### 6. Final Full-Inference Validation

Run exactly one complete inference/macro validation after a candidate slow
kernel or profiler path is identified.  This is a confirmation gate, not the
primary search loop.

Preferred final validation:

```text
historical_4096_1024_bs4, or
serving_mixed_112req_wave16
```

Use the canonical page size, graph buckets, and variants.  Confirm:

- the candidate kernel appears on the real path;
- Route B versus SWA direct direction matches the low-cost probe;
- captured replay remains healthy;
- eager decode stays `0`;
- no correctness, CUDA, or NCCL errors appear.

If a full inference validation contradicts the low-cost probe, report the
mismatch and recommend how to make the small workload faithful before any
implementation target starts.

### 7. Decide The Next Operator Target

If a stable kernel/operator delta is found, do not optimize it in this target.
Instead, write the next target recommendation with enough detail to build a
microbench.

The recommendation must preserve the distributions listed by TARGET 08.52:

- graph bucket rows, especially bs4 and bs16;
- SWA window size;
- `swa_page_indices` and token-location locality;
- invalid `-1` density;
- `swa_topk_lengths`;
- page size `256`;
- compact SWA buffer layout vs full/Route B layout;
- C4/C128/indexer metadata consumed by the target kernel;
- dtype and tensor strides.

If no stable kernel delta is found, decide whether the next step should be:

- better profiler plumbing;
- replay-loop isolation;
- non-captured operator census validated against replay timing;
- or abandoning this branch and returning to prefix/eviction scheduler
  release/free overhead.

## Deliverables

Write results under:

```text
performance_milestones/target08_decode_forward_kernel_census_unblock/
```

Required files:

- `README.md` with final verdict;
- `profiler_harness_preflight.md`;
- `no_weight_graph_replay_probe.md`;
- `low_cost_slow_kernel_reproducer.md`;
- `rank_local_tp8_profile.md`;
- `small_full_model_replay_probe.md`;
- `kernel_census_routeb_vs_swa.md`;
- `final_full_inference_validation.md`;
- `next_operator_target.md`;
- raw scripts/logs/reports under `raw/`.

The README must answer:

1. Which profiler mode can produce usable CUDA kernel summaries?
2. Was that proven first without loading full model weights?
3. Did no-weight or partial probes identify a slow kernel class?
4. Can the same mode profile TP8 captured decode graph replay?
5. Which rank or ranks were profiled?
6. Which kernels differ between Route B and SWA direct, if any?
7. Is the difference extra work, slower same work, copy/memset, NCCL, or
   profiler noise?
8. Did one final full-inference validation confirm the low-cost finding?
9. What exact operator/microbench target should run next?
10. Did this target avoid large workloads until final validation?

## Stop Conditions

Stop and report instead of patching performance if:

- no-weight CUDA graph replay cannot be profiled repeatably;
- TP8 no-weight torchrun profiling cannot produce reports and no rank-local
  workaround is found;
- no-weight and partial probes cannot name a plausible kernel class and the
  full-model probe also cannot produce a census;
- short full-model replay probes do not keep eager decode at `0`;
- profiler collection changes graph replay behavior or introduces CUDA/NCCL
  errors;
- kernel census is noisy or inconsistent across reruns;
- a suspected operator cannot be named precisely enough for a microbench;
- a fix would require broad kernel/layout changes before a kernel census
  exists.

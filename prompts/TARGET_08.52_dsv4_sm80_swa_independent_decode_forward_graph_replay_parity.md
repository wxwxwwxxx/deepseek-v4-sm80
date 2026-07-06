# TARGET 08.52: DSV4 SM80 SWA Independent Decode Forward Graph Replay Parity

## Status

Active TARGET 08 follow-up after TARGET 08.51.

TARGET 08.51 ruled out the old SWA metadata hypotheses for the remaining
SWA-independent performance gap:

- TARGET 08.49 dirty-row page-table cache is not the next implementation
  surface.
- TARGET 08.50 direct token metadata bypasses decode full SWA page-table
  materialization, but gives only a small incremental gain over 08.49.
- Graph staging/replay copy byte ledgers are identical across Route B,
  08.49 cache, and 08.50 direct.
- C4/C128/component replay-copy ledgers are identical across comparable owner
  runs.

The remaining gap is workload-dependent:

- decode-heavy serving/historical cases are dominated by captured
  `decode_forward_s`;
- prefix/eviction pressure cases are dominated by scheduler/free/release
  bookkeeping.

This target focuses on the first class: captured decode forward / graph replay
parity.  Prefix scheduler release/free batching should be a separate follow-up
unless this target disproves decode-forward as the primary owner.

## Goal

Find why SWA independent direct metadata has slower captured decode forward
than non-SWA Route B on A100/sm80, and either fix the issue or produce a
focused implementation target with precise kernel/operator evidence.

Primary evidence from TARGET 08.51:

```text
serving_mixed_112req_wave16:
  direct vs Route B elapsed gap:       +0.874s
  direct vs Route B decode_forward_s:  +0.566s

historical_4096_1024_bs4:
  direct vs Route B elapsed gap:       +1.840s
  direct vs Route B decode_forward_s:  +1.542s
```

The target should answer:

```text
Is the decode-forward gap caused by captured graph replay runtime?
If yes, which captured kernels/operators are slower?
Is the root cause SWA attention memory layout/locality, SWA loc distribution,
extra invalid entries, different graph bucket shape, or another forward owner?
Can a focused operator/microbench reproduce and guide a fix?
```

## Starting Evidence

Read first:

```text
performance_milestones/target08_prefix_decode_metadata_graph_copy_attribution/README.md
performance_milestones/target08_prefix_decode_metadata_graph_copy_attribution/baseline_matrix.md
performance_milestones/target08_prefix_decode_metadata_graph_copy_attribution/decode_metadata_split.md
performance_milestones/target08_prefix_decode_metadata_graph_copy_attribution/graph_copy_replay_ledger.md
performance_milestones/target08_prefix_decode_metadata_graph_copy_attribution/microbench_or_partial_probes.md
performance_milestones/target08_prefix_decode_metadata_graph_copy_attribution/next_target_recommendation.md
performance_milestones/target08_swa_direct_token_metadata_parity/README.md
performance_milestones/target08_swa_direct_token_metadata_parity/owner_timing_after.md
performance_milestones/target08_swa_direct_token_metadata_parity/source_shape_reconfirmation.md
performance_milestones/target08_swa_direct_token_metadata_parity/direct_token_design.md
performance_milestones/target08_swa_metadata_page_table_perf_parity/README.md
prompts/DSV4_SWA_INDEPENDENT_LIFECYCLE_CONTRACT.md
prompts/TARGET_08.51_dsv4_sm80_prefix_decode_metadata_graph_copy_attribution.md
prompts/TARGET_08_radix_prefix_dsv4.md
prompts/target.md
```

Keep these opt-ins available for ablation:

```text
MINISGL_DSV4_SWA_METADATA_PAGE_TABLE_CACHE=1
MINISGL_DSV4_SWA_DIRECT_TOKEN_METADATA=1
```

Do not assume either should be promoted by default.

## Reference Source Paths

Mini:

```text
python/minisgl/engine/graph.py
python/minisgl/scheduler/scheduler.py
python/minisgl/attention/deepseek_v4.py
python/minisgl/kvcache/deepseek_v4_pool.py
python/minisgl/models/deepseek_v4.py
python/minisgl/kernel/deepseek_v4.py
python/minisgl/kernel/triton/deepseek_v4.py
benchmark/offline/deepseek_v4_perf_matrix.py
tests/attention/test_deepseek_v4_backend_metadata.py
tests/benchmark/test_deepseek_v4_perf_matrix.py
```

SGLang / external reference, use selectively:

```text
/workspace/sglang-main/python/sglang/srt/layers/attention/dsv4/metadata.py
/workspace/sglang-main/python/sglang/srt/layers/attention/deepseek_v4_backend.py
/workspace/sglang-main/python/sglang/jit_kernel/dsv4/attn.py
/workspace/sglang-main/python/sglang/jit_kernel/csrc/deepseek_v4/
```

## Non-Goals

- Do not keep tuning 08.49 page-table cache.
- Do not keep tuning 08.50 direct token metadata unless a fresh owner points
  back to it.
- Do not implement scheduler release/free batching in this target; write a
  separate target if prefix/eviction pressure remains the owner.
- Do not redesign SWA lifecycle correctness.
- Do not implement FP8 KV/cache, INT8 MoE, or quantized communication.
- Do not rewrite attention kernels before a replay/kernel/microbench result
  proves the kernel-side owner.
- Do not use CUDA event owner timing inside captured graph replay.  Prior
  TARGET 08.50 evidence showed this can destabilize graph replay.

## Required Work

### 1. Reproduce Clean Forward Gap

Re-run or reuse clean fixed128 matrix data to confirm the gap still exists.

Required shape:

```text
TP8
page size 256
num pages 128
CUDA graph buckets [1,2,4,8,16]
greedy sample graph capture
eager decode 0
```

Compare:

```text
non-SWA Route B baseline
SWA independent + 08.50 direct token metadata
```

Primary scenarios:

```text
historical_4096_1024_bs4
serving_mixed_112req_wave16
```

Record clean phase totals:

- prefill prepare;
- prefill forward;
- decode prepare;
- decode forward;
- scheduler overhead;
- elapsed;
- graph replay count by bucket;
- eager count.

Stop if the gap no longer reproduces.

### 2. Safe Replay-Level Timing

Build a safe replay-level timing probe that measures captured decode forward
from outside the graph internals.

Allowed approaches:

- host wall-clock around graph replay with explicit synchronization outside the
  graph boundary;
- CUDA events around the whole replay call only if first proven safe in a tiny
  no-model graph replay probe and not inserted into captured graph work;
- NVTX ranges around graph replay and nsys collection for short runs;
- a replay microbench that captures once, then repeatedly replays the same
  graph/bucket and measures outside replay.

Do not insert CUDA events or synchronization nodes inside captured graph
contents.

Probe at least:

- bs4 historical decode graph;
- bs16 serving decode graph;
- Route B vs SWA independent direct under identical bucket shape;
- enough replay iterations to reduce noise, without changing graph semantics.

The output should say whether clean `decode_forward_s` deltas are reproduced at
the replay boundary.

### 3. Kernel Timeline / Operator Census

If replay-level timing confirms a forward gap, identify the captured kernel or
operator owner.

Recommended path:

1. Capture short nsys traces for Route B and SWA direct with identical
   scenarios and graph buckets.
2. Export kernel summaries if possible.
3. Compare:
   - kernel names;
   - launch counts;
   - total time per kernel;
   - per-replay kernel time;
   - graph replay count;
   - memcpy/copy kernels;
   - NCCL kernels, if any appear in decode forward.

Use small representative traces first:

```text
historical_4096_1024_bs4, or a shorter decode length with the same bs4 decode shape
serving_mixed_112req_wave16, or a shorter serving replay probe with bs16 decode shape
```

Do not require a full long nsys trace if a short replay probe reveals the
kernel owner.

### 4. Operator / Workload Microbench

For any kernel/operator that appears slower under SWA independent, construct a
minimal workload and microbench it before attempting a broad fix.

The microbench should preserve the relevant distribution, not merely the shape:

- graph bucket rows, especially bs4 and bs16;
- SWA window size;
- `swa_page_indices` / token-loc locality;
- invalid `-1` density;
- `swa_topk_lengths`;
- page size `256`;
- compact SWA buffer layout vs full/Route B layout;
- C4/C128/indexer metadata if the kernel consumes it;
- dtype and tensor strides.

Microbench sequence:

1. Extract or synthesize representative tensors from macro reports / debug
   dumps.
2. Run the isolated operator repeatedly.
3. Compare Route B-style and SWA-independent-style metadata/layout.
4. If the difference is real, test one small kernel/layout/workload variant.
5. Only then run the full macro gate.

Possible findings to check:

- compact SWA buffer causes poorer memory locality than Route B/full layout;
- translated SWA locs are less contiguous or less cache-friendly;
- invalid entries or topk length distribution changed the attention kernel
  branch behavior;
- graph bucket padding interacts differently with SWA independent metadata;
- the issue is not attention but another model forward kernel in the captured
  graph.

### 5. Fix Or Next Implementation Target

If a low-risk fix is obvious and local, it may be implemented in this target,
but only after microbench evidence.

Acceptable local fixes might include:

- metadata/layout normalization that improves locality without changing
  lifecycle semantics;
- a small tensor layout adjustment that preserves graph shape;
- a targeted helper that makes Route B and SWA direct feed the same fast path;
- an opt-in kernel/backend switch if a pre-existing backend is clearly better.

If the fix is larger, write the next target instead of implementing it here.

Do not weaken the SWA lifecycle contract to regain speed.

### 6. Correctness And Macro Gates

For instrumentation-only changes, run focused tests and at least one clean
macro confirmation.

For any runtime logic or kernel/layout change, run:

```text
focused attention/kvcache/perf-matrix tests
fixed128 text smoke for SWA direct
fixed128 macro for historical_4096_1024_bs4 and serving_mixed_112req_wave16
auto same-Engine Marlin release + SWA direct gate
explicit cap4096 Marlin release + SWA direct smoke or macro
```

All full-model rows must show:

- pass status;
- sane text where text smoke is used;
- graph replay healthy;
- eager decode `0` for captured buckets;
- no CUDA illegal memory access;
- no NCCL watchdog;
- no SWA stale metadata / negative refcount / double free;
- capacity plan unchanged.

## Deliverables

Write results under:

```text
performance_milestones/target08_swa_decode_forward_graph_replay_parity/
```

Required files:

- `README.md` with final verdict;
- `clean_forward_gap_repro.md`;
- `safe_replay_timing.md`;
- `kernel_timeline_census.md`;
- `operator_microbench.md`;
- `fix_or_next_target.md`;
- `correctness_macro_gate.md`;
- raw logs/JSON/traces under `raw/`.

The README must answer:

1. Did safe replay-level timing reproduce the decode-forward gap?
2. Which captured kernel/operator explains the gap, if any?
3. Does the operator microbench reproduce the gap outside full macro?
4. Is the root cause SWA loc layout/locality, invalid density, graph bucket
   padding, attention backend behavior, or something else?
5. Was a local fix implemented?  If yes, what did it improve?
6. If no fix was implemented, what exact next implementation target should run?
7. Do 08.49 cache and 08.50 direct remain opt-in only?
8. Should prefix/eviction scheduler release/free batching be opened as a
   separate target?

## Stop Conditions

Stop and report instead of continuing to patch if:

- clean decode-forward gap no longer reproduces;
- safe replay timing cannot be implemented without perturbing graph replay;
- nsys/kernel census shows no stable kernel-level difference;
- the suspected operator cannot be reproduced in microbench;
- the fix would require broad attention-kernel rewrite without a microbench
  showing clear upside;
- correctness or graph replay regresses;
- the dominant owner turns out to be prefix scheduler/free rather than decode
  forward for the target workloads.


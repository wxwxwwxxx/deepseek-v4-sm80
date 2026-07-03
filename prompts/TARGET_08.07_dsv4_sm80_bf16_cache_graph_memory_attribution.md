# TARGET 08.07: DSV4 SM80 BF16 Cache Graph Memory Attribution

## Status

Active next TARGET 08 subtarget.

Run this after TARGET 08.06 and before TARGET 08.10.

TARGET 08.06 showed that the promoted graph bucket set has a real and stable
CUDA graph capture memory delta of about `19 GiB/rank`.  It also showed that the
cost is dominated by the first captured graph and is not explained by bucket
count, greedy sample capture, captured attention metadata, `max_seq_len`,
`num_pages`, or missing graph pool reuse.

This target tests the remaining concrete hypothesis raised after TARGET 08.06:
the promoted BF16 weight-cache paths may indirectly inflate the CUDA graph
private pool through captured GEMM/BMM temporary allocations or cuBLAS/cuBLASLt
workspace behavior.

## Goal

Attribute whether BF16 projection/shared-expert caches are a material cause of
the CUDA graph private-pool delta.

The target should answer:

1. Are the BF16 cache tensors themselves already accounted for before graph
   capture?
2. Do the cached BF16 runtime paths cause a large extra captured allocation or
   workspace inside the first graph?
3. Which cache owner, if any, is responsible:
   - attention `q_wqb`;
   - attention `wo_b`;
   - attention `wo_a` BF16 BMM;
   - indexer `wq_b`;
   - shared experts gate/up or down;
   - the broader projection-cache bundle.
4. If one owner is material, is there a small pre-08.10 fix, or should the cost
   be carried into TARGET 08.18 memory/capacity accounting?

## Starting Point

Read:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.06_dsv4_sm80_cuda_graph_memory_attribution.md`
- `performance_milestones/target08_cuda_graph_memory_attribution/README.md`
- `python/minisgl/kernel/deepseek_v4.py`
- `python/minisgl/models/deepseek_v4.py`
- `python/minisgl/engine/engine.py`
- `python/minisgl/engine/graph.py`
- `benchmark/offline/deepseek_v4_perf_matrix.py`

Use the promoted exact path as the baseline:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--num-pages 128
cuda_graph_bs=[1,2,4,8,16]
```

Do not use automatic `memory_ratio=0.9` as a baseline.

## Important Background

The current A100 victory bundle enables these cache-related toggles by default:

```text
MINISGL_DSV4_SM80_Q_WQB_BF16_WEIGHT_CACHE
MINISGL_DSV4_SM80_WO_B_BF16_WEIGHT_CACHE
MINISGL_DSV4_SM80_WO_A_BF16_BMM_CACHE
MINISGL_DSV4_SM80_INDEXER_WQB_BF16_WEIGHT_CACHE
MINISGL_DSV4_SM80_SHARED_EXPERT_BF16_WEIGHT_CACHE
MINISGL_DSV4_SM80_FUSED_WQA_WKV_WEIGHT_CACHE
MINISGL_DSV4_SM80_GATE_FP32_WEIGHT_CACHE
MINISGL_DSV4_SM80_INDEXER_STORE_NORM_FP32_WEIGHT_CACHE
```

`Engine` calls `prepare_for_cuda_graph_capture()` before KV cache allocation and
graph capture.  Therefore the persistent cache tensors should appear in the
pre-capture allocated baseline, not as new graph-capture allocations.

The hypothesis is subtler:

```text
cached BF16 runtime path -> captured temporary tensor / GEMM workspace /
layout staging -> large CUDA graph private-pool delta
```

Do not conclude that BF16 caches are innocent just because their persistent
tensor bytes are visible before capture.

## Required Instrumentation

First audit whether a true per-cache disable path exists while keeping the rest
of `dsv4_sm80_a100_victory` enabled.

If no such path exists, add a small attribution-only hook.  Acceptable designs:

- a comma-separated env denylist such as
  `MINISGL_DSV4_SM80_A100_VICTORY_DISABLE_TOGGLES`;
- benchmark-only variants that build the victory env minus selected toggles;
- an explicit `dsv4_env_flag` false-override for named toggles.

The hook must be:

- documented as an attribution/debug hook, not a promoted serving option;
- applied before model construction, model prepare, KV allocation, and graph
  capture;
- reflected in the run summary so every raw result records which cache owners
  were actually enabled;
- removed or left disabled by default after the target unless it is generally
  useful and low risk.

Add or reuse summary fields for:

- `model_prepare_report_rank0`;
- per-cache `enabled`, `layers_cached`, `total_bytes`, and pretransposed bytes
  if available;
- `capture_memory_allocated_before_bytes`;
- `capture_memory_allocated_after_bytes`;
- `capture_memory_allocated_delta_bytes`;
- `capture_memory_reserved_before_bytes`;
- `capture_memory_reserved_after_bytes`;
- `capture_memory_reserved_delta_bytes`;
- free-memory before/after/delta;
- graph replay/eager counts;
- captured batch sizes and graph pool reuse.

## Measurement Plan

Use separate `torchrun` invocations for each variant.  Do not create and destroy
multiple engines inside one Python process.

### Phase 1: Cheap Single-Graph Attribution

Because TARGET 08.06 showed that the first captured graph dominates, start with
one bucket:

```text
cuda_graph_bs=[16]
```

Run at least:

| Variant | Purpose |
| --- | --- |
| full victory baseline | Reproduce `~18.8-19.0 GiB/rank` first-graph delta. |
| no projection BF16 caches | Disable `q_wqb`, `wo_b`, `wo_a`, and indexer `wq_b` caches together. |
| no `q_wqb` BF16 cache | Isolate attention q_wqb. |
| no `wo_b` BF16 cache | Isolate attention output local projection. |
| no `wo_a` BF16 BMM cache | Isolate wo_a BMM cache. |
| no indexer `wq_b` BF16 cache | Isolate indexer projection. |
| no shared-expert BF16 cache | Isolate shared expert projection cache. |
| no all tested BF16 caches | Check whether the total explains the graph delta. |

If a variant fails correctness or shape assumptions, keep the failure as a
result and do not silently compare a different path.

### Phase 2: Full Bucket Confirmation

For the baseline and any Phase 1 variant that changes graph delta by more than
`1 GiB/rank`, repeat with:

```text
cuda_graph_bs=[1,2,4,8,16]
```

This verifies that the single-bucket attribution carries over to the selected
TARGET 08.05 serving bucket policy.

### Phase 3: Optional Workspace Probe

Only if Phase 1 identifies a likely owner, add one or two narrow probes:

- compare memory after model prepare but before graph capture;
- compare eager warmup peak allocation for the same runtime path;
- capture a tiny profiler/memory snapshot around the first graph if it can be
  done without making the run too noisy.

Do not turn this into a broad graph allocator redesign.

## Required Analysis

The README must distinguish three memory classes:

1. **Persistent cache baseline**: allocated before graph capture by
   `prepare_for_cuda_graph_capture()`.
2. **Graph private-pool delta**: allocations captured during the first graph.
3. **KV/page capacity**: memory controlled by `--num-pages` and page size.

For every A/B row, report:

- enabled cache owners;
- persistent cache bytes from `model_prepare_report`;
- pre-capture allocated/reserved memory;
- post-capture allocated/reserved memory;
- graph delta;
- graph replay/eager sanity;
- any correctness or smoke result.

Then explain whether BF16 caches are:

- direct baseline memory only;
- indirect graph-private-pool contributors;
- not material to the `~19 GiB/rank` delta.

## Deliverables

Create:

```text
performance_milestones/target08_bf16_cache_graph_memory_attribution/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- hardware/software baseline;
- instrumentation changes, if any;
- cache-owner matrix;
- single-bucket attribution table;
- full-bucket confirmation table for material variants;
- direct vs indirect BF16-cache conclusion;
- recommendation for TARGET 08.10 and TARGET 08.18.

## Decision Rules

Recommend a small pre-08.10 fix if:

- disabling one cache owner or owner group reduces graph delta by more than
  `2 GiB/rank`;
- the owner can be fixed by a narrow workspace/cache-path change;
- the fix preserves text smoke and graph replay.

Carry the graph delta into TARGET 08.18 if:

- all BF16-cache A/B rows stay within about `1 GiB/rank` of baseline;
- or the only improvements require broad graph/workspace redesign.

If BF16 caches reduce graph delta but hurt E2E performance substantially, report
the tradeoff and do not promote the slower path in this target.

## Stop Rules

Stop and report blocked if:

- a true disable hook cannot be made without rewriting the kernel toggle system;
- the baseline cannot reproduce the TARGET 08.06 graph delta under
  `--num-pages 128`;
- graph capture becomes unstable for the promoted path;
- a variant OOMs before producing enough memory data to compare;
- attribution requires changing prefix-cache semantics.

Do not spend more than one focused target iteration on sub-`1 GiB/rank` memory
differences.  The purpose is to find or rule out a major owner before 08.10.

## Non-Goals

- Prefix-cache promotion or eviction work.
- Changing radix prefix-cache ownership.
- CUDA graph pool allocator replacement.
- Unified cache/workspace manager design.
- Low-precision experiments.
- Attention-kernel optimization.
- PyNCCL or communication overlap tuning.

# TARGET 08.06: DSV4 SM80 CUDA Graph Memory Attribution

## Status

Active next TARGET 08 subtarget.

Run this after TARGET 08.05 and before TARGET 08.10.

TARGET 08.05 selected:

```text
cuda_graph_bs = [1, 2, 4, 8, 16]
```

That bucket set removed common eager decode fallbacks and gave the best measured
serving-style throughput.  However, graph capture reported a large free-memory
delta of about `19 GiB/rank` under `--num-pages 128`.  This target explains that
memory cost before prefix-cache promotion testing.

## Goal

Attribute the CUDA graph capture memory delta for the promoted DSV4 SM80 path.

The target should answer:

1. Is the reported `~19 GiB/rank` delta a real CUDA graph private-pool cost, or
   does the measurement include unrelated initialization/cache effects?
2. Why do bucket sets `[1,2,4]`, `[1,2,4,8]`, and `[1,2,4,8,16]` have nearly the
   same capture delta?
3. How much do these factors contribute:
   - maximum captured batch size;
   - `capture_greedy_sample`;
   - captured compressed-loc/attention metadata;
   - `max_seq_len`;
   - `--num-pages`;
   - graph pool reuse;
   - temporary tensors/workspaces allocated during capture.
4. Does the result change the recommended TARGET 08.10 bucket policy?
5. Is there a low-risk fix that should happen before TARGET 08.10, or should
   the cost simply be carried into the memory/capacity ledger?

## Starting Point

Read:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.05_dsv4_sm80_serving_workload_cuda_graph_bucket_policy.md`
- `performance_milestones/target08_serving_graph_bucket_policy/README.md`
- `python/minisgl/engine/graph.py`
- `python/minisgl/attention/deepseek_v4.py`

Use the promoted exact path:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
```

Use fixed/capped pages first:

```text
--num-pages 128
```

Do not use automatic `memory_ratio=0.9` as the baseline because it is known to
OOM during graph capture in earlier TARGET 07.79 capacity probing.

## Required Measurements

Use separate `torchrun` invocations for lifecycle-sensitive comparisons.

At minimum measure:

### 1. Bucket Set Sensitivity

Compare:

```text
[1,2,4]
[1,2,4,8]
[1,2,4,8,16]
```

Record:

- free memory before graph capture;
- free memory after graph capture;
- capture memory delta;
- per-bucket delta;
- capture elapsed time;
- peak allocated/reserved during capture;
- captured batch sizes;
- graph replay/eager sanity on a tiny decode run.

### 2. Maximum Bucket Sensitivity

If the current harness allows it, test single-bucket captures:

```text
[1]
[4]
[8]
[16]
```

This determines whether the first largest graph dominates the pool and whether
smaller graphs add only tens of MB because graph pool reuse is working.

### 3. Greedy-Sample Capture Sensitivity

Compare with and without graph-captured greedy sampling, if there is already a
flag or a small safe instrumentation path.

Question:

```text
How much memory is tied to capturing logits/argmax/sample output?
```

Do not remove greedy-sample capture from default behavior in this target unless
the fix is obviously safe and correctness-smoked.

### 4. Compressed-Loc / Attention Metadata Sensitivity

Compare the current promoted graph path against a run that disables captured
compressed-loc/attention metadata only if such a flag already exists or can be
added safely as instrumentation.

Question:

```text
How much graph memory is caused by DSV4 attention metadata captured inside the
graph?
```

### 5. `max_seq_len` And `num_pages` Sensitivity

Run a small matrix if runtime permits:

```text
max_seq_len: 1280, 2048, 5120
num_pages: 64, 128
```

This determines whether graph memory is mostly independent of KV capacity or is
scaling with page table / metadata / capture buffers.

Do not run unsafe automatic KV sizing as the default.  If probing it is useful,
use fail-open instrumentation and label it non-serving.

## Implementation Guidance

Prefer measurement and small instrumentation over optimization.

Allowed:

- add graph capture memory fields if missing;
- add small scripts under the milestone directory;
- add benchmark flags that only expose existing graph options;
- add assertions/logging for graph pool reuse;
- add a tiny A/B hook for `capture_greedy_sample` or captured metadata if it is
  low risk.

Not allowed unless clearly justified:

- large graph runner rewrite;
- cache/workspace manager redesign;
- prefix-cache ownership changes;
- CUDA graph pool allocator replacement;
- attention-kernel changes;
- low-precision experiments.

## Deliverables

Create:

```text
performance_milestones/target08_cuda_graph_memory_attribution/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- hardware/software baseline;
- bucket sensitivity table;
- single-bucket sensitivity table if available;
- greedy-sample capture A/B if available;
- compressed-loc/metadata A/B if available;
- `max_seq_len` and `num_pages` sensitivity;
- graph pool reuse conclusion;
- attribution summary;
- recommendation for TARGET 08.10.

## Decision Rules

Keep `[1,2,4,8,16]` for TARGET 08.10 if:

- graph capture is stable under `--num-pages 128`;
- the large delta is mostly fixed first-graph/private-pool cost;
- expanding from `[1,2,4]` to `[1,2,4,8,16]` adds little extra memory;
- no low-risk pre-08.10 fix exists.

Recommend a pre-08.10 fix only if:

- one small, isolated cause explains a large part of the delta;
- it can be fixed without changing prefix-cache semantics;
- text smoke and graph replay can verify the fix quickly.

Recommend carrying the cost into TARGET 08.18 if:

- the cost is real and stable;
- no small fix exists;
- it mostly affects capacity accounting rather than correctness.

## Stop Rules

Stop and report blocked if:

- graph capture becomes unstable on the promoted path;
- a measurement variant OOMs under `--num-pages 128`;
- instrumentation changes graph behavior in a way that prevents fair
  comparison;
- attribution requires a broad graph/workspace manager redesign.

## Non-Goals

- Promoting prefix cache.
- Changing prefix-cache ownership or eviction.
- Implementing independent SWA/component retention.
- Low-precision KV/cache or INT8 MoE.
- Attention-kernel optimization.
- PyNCCL or communication overlap tuning.

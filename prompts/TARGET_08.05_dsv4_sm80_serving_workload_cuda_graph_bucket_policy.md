# TARGET 08.05: DSV4 SM80 Serving Workload And CUDA Graph Bucket Policy

## Status

Active next TARGET 08 subtarget.

Run this before TARGET 08.10.  TARGET 08 phase 1 has already produced a correct
explicit opt-in radix prefix cache at:

```text
performance_milestones/target08_radix_prefix_dsv4/
```

The phase-1 shared-prefix benchmark exposed an important measurement issue: the
second stage used batch size 7, while the promoted graph capture set only
contained `[1, 2, 4]`, so part of the decode path ran eager.  Before deciding
whether prefix cache is promotable, establish a serving-oriented workload suite
and a CUDA graph batch-bucket policy.

## Goal

Build a reproducible serving workload and graph-bucket evaluation for DeepSeek
V4 Flash on A100/sm80.

This target should answer:

1. Which active decode batch sizes appear in representative serving-style
   mini-sglang workloads?
2. Which CUDA graph bucket set gives the best tradeoff between replay coverage,
   capture memory, startup/capture cost, and throughput?
3. How much performance is lost when decode falls back to eager for common
   batch sizes?
4. What graph bucket policy should TARGET 08.10 use for prefix-cache serving
   stability and promotion tests?

## Starting Point

Use the promoted exact TARGET 07 path:

```text
dsv4_sm80_a100_victory
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
```

Prefix cache may be measured as a workload dimension, but this target should
not modify prefix-cache ownership or eviction logic.

Use fixed or capped pages such as:

```text
--num-pages 128
```

Do not treat automatic `memory_ratio=0.9` KV sizing as default serving policy;
TARGET 07.79 showed that it can OOM during graph capture.

## Candidate Bucket Sets

Do not blindly capture `1..256`.

Evaluate a small ladder first:

```text
[1, 2, 4]
[1, 2, 4, 8]
[1, 2, 4, 8, 16]
```

Only expand if evidence says common workloads need it:

```text
[1, 2, 4, 8, 16, 24, 32]
[1, 2, 4, 8, 16, 24, 32, 48, 64]
```

Treat `128` and `256` as special high-concurrency experiments, not default
choices, unless workload traces show they are frequent and graph capture memory
is acceptable.

## Workload Suite

At minimum include:

- fixed historical baseline:
  - prompt `4096`, decode `1024`, batch `4`;
  - prompt `4096`, decode `128`, batch `4`.
- prefix-cache serving scenario:
  - shared prefix warm request plus reuse requests;
  - prefix cache off/on;
  - hit/miss mixture if supported by the harness.
- decode-concurrency scenarios:
  - short prompt / longer decode;
  - mixed output lengths;
  - active decode batch sizes expected around `1,2,4,8,16`.
- early serving-style load shape:
  - at least one workload with `requests >= 100` or a documented smaller
    substitute if runtime is too high;
  - record queueing/TTFT/ITL/TPOT if the harness supports them.

If the existing offline harness cannot represent one workload cleanly, add the
smallest benchmark extension needed and document the limitation.

## Required Metrics

For each graph bucket set and workload, record:

- graph capture success/failure;
- captured batch sizes;
- graph capture memory delta;
- graph capture/setup time if available;
- replay count by actual batch size;
- eager decode count by actual batch size;
- output tok/s, decode tok/s, TTFT, TPOT/ITL where available;
- peak allocated/reserved memory;
- KV cache memory;
- prefix metrics if prefix cache is enabled;
- communication counters if already available.

Produce a bucket coverage table:

```text
actual decode bs -> replay count, eager count, tokens, wall share
```

## Deliverables

Create:

```text
performance_milestones/target08_serving_graph_bucket_policy/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact commands;
- git status summary;
- hardware/software assumptions;
- workload definitions;
- bucket-set comparison table;
- memory/capture cost table;
- recommended bucket policy for TARGET 08.10;
- remaining risks and unsupported workload shapes.

## Decision Rules

Recommend a default controlled bucket set if:

- it covers the common serving batch sizes in the measured suite;
- it materially reduces eager decode versus `[1,2,4]`;
- capture memory and setup cost leave safe headroom under `--num-pages 128`;
- it does not break the promoted 4096/1024/batch4 path.

Recommend keeping a smaller benchmark-only bucket set if:

- large buckets cost too much memory;
- common serving workloads do not actually hit those sizes;
- capture failures or graph memory pressure make broad capture unsafe.

## Stop Rules

Stop and report blocked if:

- graph capture becomes unstable on the promoted path;
- a bucket set causes OOM under fixed `--num-pages 128`;
- benchmark variance is too high to rank bucket policies;
- implementing a full serving harness would be required to answer the basic
  graph-bucket question.

## Non-Goals

- Promoting prefix cache.
- Rewriting radix prefix ownership.
- Implementing SGLang-style SWA component retention.
- Adding FP8 KV cache, INT8 MoE, attention-kernel changes, or PyNCCL changes.
- Capturing every batch size up to 256 without evidence.

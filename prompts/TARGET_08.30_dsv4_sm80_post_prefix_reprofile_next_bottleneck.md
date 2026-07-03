# TARGET 08.30: DSV4 Post-Prefix Reprofile And Next Bottleneck Reset

## Status

Planned after TARGET 08.10 and TARGET 08.18, and after TARGET 08.20 only if
TARGET 08.20 is actually run.

## Goal

Re-establish the DeepSeek V4 SM80 bottleneck map after prefix-cache and graph
bucket policy work.

This target decides whether the project should proceed to:

- TARGET 09 low-precision research;
- TARGET 10 attention/communication research;
- more prefix/cache/capacity work;
- serving-system polish.

## Required Inputs

Read:

- `prompts/TARGET_08_radix_prefix_dsv4.md`
- TARGET 08.05 result README;
- TARGET 08.06 result README;
- TARGET 08.10 result README;
- TARGET 08.18 result README;
- TARGET 08.20 result README if it exists;
- `prompts/TARGET_09_dsv4_sm80_low_precision_research.md`
- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`

## Measurement Scope

Run the final recommended TARGET 08 configuration:

- promoted TARGET 07 exact bundle;
- selected CUDA graph bucket policy from TARGET 08.05;
- graph memory/capacity conclusion from TARGET 08.06;
- prefix cache status from TARGET 08.10;
- component-retention status from TARGET 08.18/08.20.

Measure at least:

- historical fixed baseline:
  - 4096/1024/batch4;
  - 4096/128/batch4.
- serving workload suite from TARGET 08.05;
- shared-prefix serving workload from TARGET 08.10;
- one stress/capacity case with retained prefixes.

If runtime allows, include a release-style serving pass:

- `requests >= 100`;
- multiple request-rate or arrival patterns;
- fixed max concurrency;
- short-output and long-output mixes;
- GPU utilization;
- KV cache usage;
- batch-size distribution;
- queueing latency, TTFT, ITL/TPOT.

## Required Analysis

Produce:

- stable macro table;
- graph replay/eager table by batch size;
- prefix hit/miss/eviction/memory table;
- TTFT and prefill-forward savings table;
- decode throughput table;
- memory capacity ledger;
- owner/profile bottleneck ranking if available;
- comparison to TARGET 07.79 baseline;
- comparison to old vLLM baseline where applicable.

Answer:

1. Did prefix cache materially improve serving-style TTFT/prefill?
2. Did graph bucket policy remove common eager decode fallbacks?
3. Is memory retention now a practical limiter?
4. Is the remaining bottleneck more likely precision/cache, attention,
   communication, metadata/runtime, or serving scheduler behavior?
5. Should the next target be TARGET 09, TARGET 10, or another TARGET 08 cache
   target?

## Deliverables

Create:

```text
performance_milestones/target08_post_prefix_reprofile/
  README.md
  raw/
  scripts/
  summaries/
```

The README must include:

- exact final configuration;
- commands;
- git status summary;
- workload tables;
- performance tables;
- graph bucket coverage;
- memory/capacity ledger;
- ranked bottleneck table;
- explicit next-target recommendation.

## Decision Rules

Recommend TARGET 09 if:

- exact/cache/prefix work has plateaued;
- remaining gaps point to vLLM-style low-precision cache, FP8 KV, or INT8 MoE;
- quality gates can be isolated.

Recommend TARGET 10 if:

- attention, communication, or graph/runtime buckets are top contributors after
  prefix cache;
- there is a coherent `>=2%` E2E opportunity with vLLM or hardware evidence.

Recommend more TARGET 08 work if:

- prefix cache still has correctness, memory, eviction, or graph-bucket issues;
- serving workload coverage is not yet credible.

## Stop Rules

Stop and report blocked if:

- the selected prefix-cache configuration is not correctness-clean;
- graph replay is unexpectedly disabled for recommended buckets;
- serving benchmark variance prevents ranking;
- capacity pressure causes OOM before producing useful data.

## Non-Goals

- Implementing low-precision kernels.
- Tuning attention or communication inside this target.
- Changing prefix-cache ownership unless a previous TARGET 08 result already
  requires it.

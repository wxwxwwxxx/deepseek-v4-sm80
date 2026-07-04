# TARGET 08.30: DSV4 Post-Prefix Reprofile And Next Bottleneck Reset

## Status

Run this after TARGET 08.29.

TARGET 08.28 promoted the Route B component page-table lifetime cache as the
preferred Route B prefix-cache opt-in, and TARGET 08.29 made that path the
clear benchmark/text-smoke preset
`dsv4_sm80_a100_victory_prefix_routeb_lifetime`.

This target is a measurement and decision target.  Do not implement a new major
optimization here.  Reprofile the promoted prefix path and decide the next
engineering direction from evidence.

## Goal

Re-establish the DeepSeek V4 SM80 bottleneck map after the prefix-cache Route B
work is promoted as an opt-in.

Decide whether the next project phase should be:

- more TARGET 08 prefix/cache/runtime work;
- TARGET 09 low-precision research;
- TARGET 10 attention/communication/graph-runtime research;
- serving-system polish and release hardening.

## Required Inputs

Read:

- `prompts/target.md`
- `prompts/TARGET_08_radix_prefix_dsv4.md`
- `prompts/TARGET_08.29_dsv4_sm80_route_b_lifetime_promotion_cleanup.md`
- `performance_milestones/target08_route_b_lifetime_promotion_cleanup/README.md`
- `performance_milestones/target08_route_b_lifetime_cache_promotion_gate/README.md`
- `performance_milestones/target08_sglang_aligned_route_b_metadata_lifetime/README.md`
- `performance_milestones/target08_serving_graph_bucket_policy/README.md`
- `performance_milestones/target08_prefix_cache_memory_ledger/README.md`
- `prompts/TARGET_09_dsv4_sm80_low_precision_research.md`
- `prompts/TARGET_10_dsv4_sm80_optional_attention_comm_research.md`

If the TARGET 09 or TARGET 10 prompt is missing, note that in the report and
still rank whether the evidence points toward low precision or attention/comm.

## Final Recommended Configuration

Use the promoted TARGET 08.29 prefix preset:

```text
dsv4_sm80_a100_victory_prefix_routeb_lifetime
```

It should include:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_BUFFERS=1
MINISGL_DSV4_SM80_DIRECT_GRAPH_METADATA_GROUPS=c4
MINISGL_DSV4_SM80_ROUTE_B_COMPONENT_PAGE_TABLE_CACHE=1
page_size=256
--num-pages 128
--enable-dsv4-radix-prefix-cache
--enable-dsv4-component-loc-ownership
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Also keep a non-prefix promoted TARGET 07 control:

```text
MINISGL_DSV4_SM80_A100_VICTORY_BUNDLE=1
page_size=256
--allow-dsv4-cuda-graph --cuda-graph-bs 1 2 4 8 16
```

Do not use automatic `memory_ratio=0.9` graph-mode capacity as the primary
serving configuration in this target.  Prefer fixed page counts such as
`--num-pages 128` unless a capacity subtest explicitly varies them.

## Measurement Scope

Use separate `torchrun` invocations per variant.

Measure at least:

1. Historical fixed baselines:
   - `4096/1024/batch4`;
   - `4096/128/batch4`.
2. Serving suite:
   - `serving_mixed_112req_wave16`;
   - `prefix_multi_112req_wave16`;
   - `prefix_eviction_pressure_96req_wave16`;
   - `decode_ladder_bs16`.
3. Prefix/cache controls:
   - prefix off / TARGET 07 promoted control;
   - phase1 prefix-on if still easy to run;
   - promoted Route B lifetime prefix path.

If runtime allows, add a release-style serving pass:

- at least 100 requests;
- mixed prompt/decode lengths;
- fixed max concurrency or wave size;
- graph bucket coverage;
- TTFT, ITL/TPOT, and output tok/s;
- prefix hit/miss/eviction metrics;
- memory/capacity ledger.

## Required Analysis

Produce tables for:

- stable macro throughput and variance;
- graph replay/eager count by scenario and padded batch size;
- prefix hit/miss/saved-prefill/eviction metrics;
- memory/capacity ledger, including retained prefix pages and available KV or
  component pages where available;
- TTFT/prefill-forward savings for prefix workloads;
- decode prepare versus decode forward;
- owner/profile bottleneck ranking if profiling is available;
- comparison to TARGET 07.79 non-prefix baseline;
- comparison to old vLLM baseline where applicable.

Answer:

1. Did the promoted Route B lifetime prefix path materially improve
   serving-style TTFT or prefill work?
2. Does the `[1,2,4,8,16]` bucket policy still give zero-eager decode coverage
   for the serving suite?
3. Is memory retention or SWA-tail/full-tail guarding now a practical limiter?
4. Is the remaining bottleneck more likely:
   - prefix metadata/runtime;
   - SWA/cache capacity or hit rate;
   - attention/indexer/compressor kernels;
   - MoE;
   - communication;
   - low precision/cache format;
   - scheduler/serving runtime?
5. Should the next target be TARGET 09, TARGET 10, another TARGET 08 cache
   follow-up, or serving-system hardening?

## Profiling Guidance

Start with unprofiled repeat runs.  Use owner timing or nsys only after the
macro picture is stable.

If profiling is needed, prefer targeted profiles for:

- decode prepare/runtime if it again exceeds phase1 by a meaningful amount;
- attention/indexer/compressor if decode forward dominates;
- NCCL/communication if collectives become visible;
- graph replay/runtime if eager fallback or graph overhead returns.

Keep profiling overhead out of final throughput tables.

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

- exact final configuration and promoted variant name;
- commands and environment variables;
- git status summary;
- workload throughput tables;
- graph bucket coverage;
- prefix/cache/eviction metrics;
- memory/capacity ledger;
- ranked bottleneck table;
- comparison to TARGET 07.79, TARGET 08.28, and old vLLM baseline where
  relevant;
- explicit next-target recommendation.

## Decision Rules

Recommend TARGET 09 if:

- exact BF16/prefix work has plateaued;
- remaining opportunity points to vLLM-style low precision, FP8 KV/cache,
  FP8/quantized projection/cache fusion, or INT8 MoE;
- quality gates can be isolated from the promoted exact path.

Recommend TARGET 10 if:

- attention, indexer, communication, or graph/runtime buckets are top
  contributors after prefix cache;
- there is a coherent `>=2%` E2E opportunity with vLLM/SGLang or hardware
  evidence.

Recommend more TARGET 08 work if:

- prefix cache still has correctness, memory, eviction, graph-bucket, or
  capacity/hit-rate issues;
- SWA-tail/full-tail guarding materially limits serving capacity or hit rate;
- promoted Route B lifetime cache has a local lifecycle issue.

Recommend serving hardening if:

- performance is good enough and remaining bottlenecks are not clearly larger
  than measurement noise;
- the main gaps are ergonomics, presets, smoke coverage, docs, or release
  confidence.

## Stop Rules

Stop and report blocked if:

- TARGET 08.29 did not produce a clean promoted prefix preset;
- the selected prefix-cache configuration is not correctness-clean;
- graph replay is unexpectedly disabled for recommended buckets;
- serving benchmark variance prevents ranking;
- capacity pressure causes OOM before producing useful data.

## Non-Goals

- Implementing low-precision kernels.
- Tuning attention, MoE, or communication inside this target.
- Changing prefix-cache ownership.
- Implementing independent SWA ownership unless this reprofile only recommends
  it as a future target.
